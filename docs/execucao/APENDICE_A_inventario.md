# Apêndice A — Inventário do repositório e estado congelado

> **Ler em toda sessão nova, junto com o briefing do marco.** Este arquivo substitui
> a leitura exploratória do repositório: ele diz o que existe, o que cada coisa é,
> e o que não se toca. Se um número aqui divergir do arquivo real, **o arquivo real
> manda** — e corrija este apêndice no mesmo commit.

---

## 1. Números do experimento (decorar estes)

| grandeza | valor | onde se confere |
|---|---:|---|
| universo (`fase == 'eval'`) | **148.176** | `results/metricas/eval_composicao_resumo.json` |
| bonafide / spoof no universo | 14.816 / 133.360 (**9,0:1**) | idem |
| treino (70%) | **103.723** | `data/processed/split.csv` |
| validação (15%) | **22.226** | idem |
| teste (15%) | **22.227** | idem |
| subamostra do braço principal | **30.000** (3.000 bonafide / 27.000 spoof) | `results/metricas/subamostra_30k.json` |
| estratos da subamostra | 98 (classe × codec × ataque) | idem |
| espectrogramas necessários | **74.453** = 30.000 + 22.226 + 22.227 | `DECISOES_PENDENTES_CNN.md` §Nota de engenharia |
| frames por áudio | **251** = `1 + 64000 // 256`, valor único nas 148.176 linhas | coluna `n_frames_total` |
| amostras por áudio | 64.000 = 4,0 s × 16 kHz | `config.yaml → audio` |
| frames válidos (`n_frames_validos`) | mín 8 · **mediana 120** · média 132,43 · máx 250 | `results/metricas/auditoria_decisoes_cnn.json` |
| áudios com >50% de padding | **53,07%** | idem |
| fração de padding, bonafide × spoof | 47,11% × 47,25% (**0,14 p.p.**) | idem |
| `prop_fala`, bonafide × spoof | 62,94% × 84,65% (**21,71 p.p.**) | idem |
| correlação `prop_fala` × `n_frames_validos` | **r = 0,145** | idem |

> **A armadilha número um do Bloco 4** está nas três últimas linhas: `prop_fala` e
> fração de padding **não são a mesma coisa** e quase não se correlacionam. A v1 do
> `DECISOES_PENDENTES_CNN.md` inferiu assimetria de padding entre classes a partir de
> `prop_fala` e errou. Não repita.

---

## 2. Artefatos congelados — não regenerar

| arquivo | MD5 | o que é |
|---|---|---|
| `data/features/features.csv` | `51b2f439bf6f1e10237acbc620bb92d9` | 148.176 × 50 colunas. **CONGELADO desde 30/08/2026.** |
| `data/processed/split.csv` | `9143f0c7b83ec2db4aa144ed5deb3402` | partição 70/15/15 por `arquivo` |
| `data/processed/subamostra_30k.csv` | `654cb796b738512388b28e15ffb14a9d` | lista de IDs do braço principal |
| `data/features/features.meta.json` | — | assinatura da definição de feature vigente |

**Esquema do `features.csv`** (50 colunas, nesta ordem):

```
arquivo, label, classe_binaria,               # identificação (3)
prop_fala, n_frames_validos, n_frames_total,  # DIAGNÓSTICO — fora do X (3)
mfcc1_media .. mfcc20_media,                  # 20
mfcc1_std   .. mfcc20_std,                    # 20
zcr_media, zcr_std, centroide_media, centroide_std   # 4  -> 44 features
```

`colunas_features` (`src/data/split.py`) é o **ponto único** que devolve as 44. As três
colunas de diagnóstico ficam fora do X porque medem quantidade de fala/silêncio, e o
bonafide perde mais sinal no VAD — usá-las daria ao modelo um atalho por duração, não
por artefato de síntese.

**Convenção de classe, em todo o projeto:** `bonafide = 0`, `spoof = 1`.
**`spoof` é a classe MAJORITÁRIA** (≈ 9:1). Maior score = maior evidência de spoof.

---

## 3. Resultados fechados do Bloco 3 (a régua que a CNN vai enfrentar)

Todos na **validação** (22.226), com limiar selecionado na validação, regra
`score >= limiar`:

| modelo | n treino | limiar | f1_macro | EER | ROC-AUC | recall bonafide |
|---|---:|---:|---:|---:|---:|---:|
| RF ajustado — **principal** | 30.000 | 0,6516 | **0,7225** | 0,1930 | 0,8873 | 0,5441 |
| RF ajustado — **referência** | 103.723 | 0,6196 | **0,7723** | 0,1579 | 0,9191 | 0,6008 |
| SVM RBF ajustado — principal | 30.000 | −0,0329 | **0,7987** | 0,1462 | 0,9289 | 0,6625 |

Notas que importam para a comparação:

- O limiar do SVM é negativo porque vive na escala do `decision_function`, não em
  [0, 1]. **Não é inconsistência:** `selecionar_limiar` opera sobre `np.unique(scores)`
  e é agnóstico de escala.
- **Bootstrap pareado (1.000 reamostragens, mesmas 22.226 amostras):**
  Δf1_macro = **+0,0762** IC95 [0,0663; 0,0856]; ΔEER = **−0,0466** IC95
  [−0,0560; −0,0377]. Nenhum IC contém zero; SVM melhor em 100% das reamostragens.
  **A vantagem do SVM é real.**
- O braço de referência **não** é concorrente direto de SVM/CNN — ele só quantifica o
  custo da subamostragem. Isso tem de estar dito explicitamente no texto.
- A curva de aprendizado do RF **não satura**: extrapolando `f1 ~ a·ln(n)+b`
  (R² = 0,9934), o RF precisaria de ~276.116 áudios de treino para alcançar o f1 do
  SVM — 1,86× o universo eval inteiro.

**Tempos** (`results/metricas/tempo_pipeline_completo.json`, por áudio, batch = 1):
carregar 0,639 ms + VAD/padding 0,371 ms + features 3,705 ms; predição **RF 6,405 ms**
(total 11,12) contra **SVM 0,613 ms** (total 5,33). Em lote o RF é ~19× melhor.
A hipótese de que o pré-processamento dominaria **não se confirmou**.

---

## 4. Mapa do código

```
src/
├── data/
│   ├── carregar_dados.py      # trial_metadata.txt -> labels.csv
│   ├── preprocessamento.py    # carregar -> normalizar pico -> VAD -> 4,0 s
│   └── split.py               # criar_split, carregar_dados_split,
│                              # filtrar_treino_braco, colunas_features, resumo_split
├── features/
│   └── extrair_features.py    # nomes_features, COLUNAS_DIAGNOSTICO,
│                              # frames_validos, extrair_vetor, executar
├── models/
│   ├── avaliacao.py           # aplicar_limiar, predizer_rf, selecionar_limiar,
│   │                          # calcular_eer, avaliar, plotar_matriz_confusao
│   ├── treinar_rf.py          # RF baseline, dois braços
│   ├── ajustar_rf.py          # Random Search + RF final, dois braços
│   ├── treinar_svm.py         # SVM RBF, só braço principal
│   ├── modelos_ajustados.py   # MODELOS_PRINCIPAIS, carregar_modelo_ajustado,
│   │                          # scores_de, hashes_congelados
│   └── tempo.py               # medir_tempos, ambiente
└── utils/
    ├── config.py              # carregar_config
    ├── seeds.py               # fixar_seeds, fixar_seeds_torch
    └── serializacao.py        # json_seguro
```

### Assinaturas que você vai chamar no Bloco 4

```python
# src/data/preprocessamento.py
preprocessar_audio(caminho: str, cfg: dict) -> (y[64000] float32, prop_fala, n_amostras_validas)

# src/features/extrair_features.py
frames_validos(n_amostras_validas: int, hop: int, n_frames_total: int) -> int
# = max(min(ceil(n_amostras_validas / hop), n_frames_total), 1)

# src/models/avaliacao.py
aplicar_limiar(scores, limiar) -> y_pred            # regra: score >= limiar
selecionar_limiar(y_true, scores, criterio="f1_macro", conjunto="validacao") -> dict
calcular_eer(y_true, scores) -> (eer, limiar_no_eer)
avaliar(y_true, scores, nome: str, limiar: float) -> dict

# src/models/tempo.py
medir_tempos(predizer, X, cfg_tempo) -> dict        # predizer: callable(X) -> scores
ambiente(n_jobs_inferencia: int) -> dict

# src/utils/seeds.py
fixar_seeds(semente=42) -> int
fixar_seeds_torch(semente=42, estrito=True) -> int
```

### Scripts existentes que servem de molde

| quando você precisar de | copie o padrão de |
|---|---|
| alocação estratificada por maior resto com piso 1 | `scripts/gerar_subamostra.py` |
| guarda de integridade de split/subamostra por hash | `scripts/validar_split_pos_lote.py` |
| checagem obrigatória A/B antes de um lote caro | `scripts/verificar_mascaramento.py` |
| auditoria que **falha com código 1** ao violar invariante | `scripts/auditar_decisoes_cnn.py` |
| medição ponta a ponta do pipeline | `scripts/tempo_pipeline_completo.py` |
| verificação de reprodução campo a campo de JSONs | `scripts/guarda_reproducao.py` |
| sanidade de ambiente / GPU | `scripts/verificar_ambiente.py`, `scripts/compatibilidade_gpu.py` |

---

## 5. Ambiente

- Windows 10, Python **3.10.11**, numpy 2.2.6, sklearn 1.7.2, pandas 2.3.3.
- **GPU: RTX 5060 Ti 16 GB, Blackwell, compute capability (12, 0) = sm_120.**
  Por isso o projeto usa **PyTorch ≥ 2.7 com wheels cu128**, não TensorFlow
  (não distribui kernels para sm_120; e não tem GPU nativa no Windows desde a 2.10).
  Instalação **obrigatoriamente** com índice explícito:
  `pip install torch --index-url https://download.pytorch.org/whl/cu128`
  (o PyPI padrão entrega build CPU-only e `torch.cuda.is_available()` volta `False`
  em silêncio).
- `soundfile` está **pinado em 0.12.1**: os wheels 0.13/0.14 para Windows embutem um
  libsndfile cujo decodificador FLAC falha em ~44% dos FLACs válidos do dataset.
  **Não atualizar.**
- `librosa ≥ 0.11`.

---

## 6. `.gitignore` — o que já está protegido

Já existe regra para `data/espectrogramas/*` com exceção de
`espectrogramas.meta.json`, acrescentada **antes** de o primeiro tensor existir. Ou
seja: um `git add .` distraído **não** vai tentar versionar ~9,6 GB. Também ignorados:
`data/raw/`, `data/features/*.csv`, `models/*` (exceto `.gitkeep`).

**O que é versionado e precisa ser:** `config.yaml`, todo `src/` e `scripts/`, tudo em
`results/`, `split.csv`, `subamostra_30k.csv`, `features.meta.json` e —
quando existirem — `espectrogramas.meta.json`, `normalizacao_cnn.json` e
`split_interno_cnn.csv`.

---

## 7. Sete decisões fechadas em 17/09 — resumo operacional

| # | decisão |
|---|---|
| P4 | `n_fft=1024`, `win_length=400`, `hop_length=256`, `n_mels=128`, `fmin=0`, `fmax=8000`, `power=2.0`, `center=true` |
| P5 | Log-Mel em dB, `ref=1.0`, `top_db=80`. **Nunca `ref=np.max`** |
| P3 | Normalização por estatísticas **globais do treino**, média e desvio **por faixa Mel**. Validação e teste nunca entram no cálculo |
| P1 | `largura = 251`, `altura = 128`. Não interpolar para 256 |
| P2 | Espectrograma cru de formato fixo + **agregação mascarada dentro da rede** |
| P6 | Split interno 27k/3k para early stopping + **refit final nos 30k** |
| P7 | *Loss* ponderada: `CrossEntropyLoss` com 2 saídas e peso maior para **bonafide (classe 0)**, razão 9:1. **Sem** over/undersampling |

**As três armadilhas que acompanham essas decisões:**

1. **`BCEWithLogitsLoss(pos_weight=9)` inverteria a intenção** — spoof é a
   majoritária. Daí a escolha por `CrossEntropyLoss` com duas saídas.
2. **A máscara temporal tem de acompanhar a redução do eixo do tempo** dentro da
   rede. O resultado final é *masked global pooling* (soma só das posições válidas,
   dividida pelo nº de posições válidas), nunca um `GlobalAveragePooling` comum sobre
   regiões inválidas. A máscara atua **só no tempo**; a frequência continua integral.
   **`Flatten` está vetado.**
3. **`top_db=80` é relativo ao máximo de cada exemplo.** O piso do padding é
   `max − 80`, e portanto **muda de exemplo para exemplo**. Não invalida a decisão,
   mas tem de estar documentado.

---

## 8. Duas formulações proibidas no texto

A banca notaria as duas. Estão aqui porque aparecem naturalmente ao escrever.

**(a) Não escrever que subir o `n_fft` de 512 para 1024 «aumenta a resolução
espectral real».** A janela continua em 400 amostras (~25 ms), então não se cria
informação nova. Formulação correta:

> mantemos os mesmos 25 ms de análise e usamos FFT de 1024 para **aumentar a
> densidade de amostragem da DFT por zero-padding**, permitindo representar o banco de
> 128 filtros Mel de maneira numericamente mais adequada.

**(b) Não escrever que `ref=1.0` «preserva o ganho original das gravações».** O
pipeline já faz normalização de pico por áudio **antes** do VAD, então esse ganho já
foi removido a montante. Justificativa correta:

> `ref=1.0` evita introduzir uma **segunda** normalização, relativa ao máximo do
> próprio espectrograma, e mantém paridade com a transformação que o MFCC já aplica.

**E a paridade entre os ramos descreve-se assim:** entregar Log-Mel à CNN reduz a
diferença entre os dois pipelines **principalmente à forma de representação e
aprendizado** — no ramo clássico resumimos as características acústicas e
classificamos com RF/SVM; na CNN preservamos a estrutura tempo-frequência e deixamos a
rede aprender a representação.

---

## 9. O que está lacrado

O conjunto de **teste (22.227)** é usado **uma única vez**, em **B5.1**, com os
limiares já escolhidos na validação. Até lá:

- nenhum script de diagnóstico, curva ou ablação toca em `conjunto == 'teste'`;
- `carregar_modelo_ajustado` (`src/models/modelos_ajustados.py`) **recusa** um limiar
  cujo JSON não registre `selecao_limiar.conjunto == 'validacao'`. A regra que protege
  o teste mora no código, não só no texto — mantenha assim ao estender o módulo para
  a CNN.
