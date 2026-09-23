# Apêndice A — Inventário do repositório e estado congelado

> **Ler em toda sessão nova, junto com o briefing do marco.** Este arquivo substitui
> a leitura exploratória do repositório: ele diz o que existe, o que cada coisa é,
> e o que não se toca. Se um número aqui divergir do arquivo real, **o arquivo real
> manda** — e corrija este apêndice no mesmo commit.

---

## Ritmo de execução — leia antes de tudo

**As datas dos briefings são TETO, não agenda.** Terminar um marco antes da sua
data-limite é o resultado desejado, não uma anomalia.

Se você chegou a um marco adiantado:

- **siga em frente sem pedir autorização, sem sinalizar e sem questionar o pedido.**
  Estar adiantado não é motivo para conferir nada com o aluno. O aluno espera
  adiantar boa parte do cronograma;
- **só atraso gera conversa.** A regra de escopo, o teste «essa decisão impede a
  validade do experimento principal?» e as seções «Se atrasar» dos briefings existem
  para quando a data-limite está em risco. Adiantado, elas não se aplicam — não há o
  que simplificar nem limitação a escrever;
- **uma sessão por marco continua valendo**, mas o motivo é **higiene de contexto**,
  não calendário. Se o marco de hoje terminou e sobrou dia, o certo é **encerrar a
  sessão e abrir a próxima** com o briefing seguinte — não encadear dois marcos na
  mesma conversa. A instrução «isto é B4.x, não faça aqui» é sobre **onde o trabalho
  é registrado e commitado**, não sobre esperar a data chegar.

### Os quatro portões que valem em qualquer ritmo

Estes **não** são calendário — são pré-requisito. Adiantar nunca autoriza pular um:

| portão | não faça antes de |
|---|---|
| **B4.1** (escrever o gerador) | o `config.yaml` do B4.0 estar commitado |
| **B4.2** (disparar os 74.453) | as **12 checagens** do B4.1 passarem, e ≥ 10 GB livres |
| **B4.6** (refit) | `melhor_epoca` estar registrada em `cnn_definida.json` |
| **B5.1** (teste lacrado) | os três JSONs de validação existirem, com `selecao_limiar.conjunto == "validacao"` |

### Onde vai o tempo que sobrar

Adiantado **não** é licença para abrir eixo experimental novo nem para reabrir
decisão fechada — a regra de escopo continua valendo com o mesmo peso. O tempo
excedente vai para esta fila, nesta ordem:

1. **redação** (B6.1) — é o item de maior risco do cronograma e o único que não tem
   como ser comprimido no fim;
2. **variância entre sementes na CNN** (3 sementes) — está fora do caminho crítico
   por decisão registrada, mas é a lacuna que o trabalho declara como limitação se
   não couber. Se couber, a limitação desaparece;
3. **diagnóstico por ataque e por codec com a CNN incluída** — fecha a análise por
   estrato para os três modelos;
4. **simulação de banca** (`APENDICE_B_banca.md`) — cada pergunta sem resposta em 30
   segundos é um parágrafo que falta no texto.

**Nunca** para: leave-one-attack-out, cross-dataset, outra duração de áudio, outro
`n_mels`, nova busca de hiperparâmetros do RF/SVM. Esses são trabalho futuro, e
continuam sendo mesmo com uma semana sobrando — porque o que fecha o TC II é a
comparação completa, não um experimento a mais.

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

### Valores medidos do espectrograma — auditoria de 17/09, `aprovado: true`

Regenerados por `python -m scripts.auditar_decisoes_cnn` **depois** do B4.0
(`largura_no_config: 251`, `config_diverge_do_pipeline: false`). Use estes números
nas asserções e no texto; não estime.

**Filterbank Mel** (`htk=False`, `norm='slaney'`, `fmin=0`, `fmax=8000`):

| `n_fft` | `n_mels` | bins FFT | suporte mínimo | filtros com ≤ 2 bins | picos duplicados |
|---:|---:|---:|---:|---:|---:|
| **512** *(ramo clássico — MFCC)* | 128 | 257 | **1** | **61** | **12** |
| 512 | 80 | 257 | 2 | 19 | 0 |
| 512 | 64 | 257 | 2 | 1 | 0 |
| **1024** *(CNN — P4 aprovada)* | 128 | 513 | **2** | **1** | **0** |
| 1024 | 80 | 513 | 4 | 0 | 0 |
| 2048 | 128 | 1025 | 5 | 0 | 0 |

Duas leituras, e as duas vão para o texto:

1. **O que caracteriza o regime degenerado** não é «existir filtro estreito» — é
   **suporte mínimo de 1 bin** e **picos compartilhados**. Com 1024 os dois
   desaparecem; o único filtro de 2 bins que resta é o mais baixo, estreito por
   construção da escala Mel junto a `fmin=0`. **Não asserte «zero filtros estreitos»:
   o valor correto é ≤ 1.**
2. **A limitação 7 está confirmada com número:** a filterbank do
   `librosa.feature.mfcc`, com o `n_fft=512` congelado, tem **61 filtros com ≤ 2
   bins e 12 picos duplicados**. Escreva a limitação com esses números.

**Escala em dB** (`ref=1.0`, `top_db=80`, `amin=1e-10`):

| grandeza | valor |
|---|---:|
| `db_max` | **+8,44** — positivo; `\|STFT\|²` passa de 1 mesmo com sinal em [−1,1] |
| `db_min` (= platô do padding) | **−71,56** = `db_max − 80` → é o `top_db` atuando |
| piso se `top_db` fosse `None` | −100,0 = `10·log₁₀(1e−10)` |
| deslocamento com ganho 10× | **+20,0 dB** no máximo **e** no mínimo |
| deslocamento com `ref=np.max` e ganho 10× | 5,7×10⁻⁶ dB (arredondamento de float32 — a invariância é `allclose`, não igualdade) |
| deslocamento do MFCC com ganho 10× | 226,27 no `c0`; 8,3×10⁻⁶ nos demais |

**Nunca asserte que `db_max <= 0`.** Foi uma expectativa errada, desmentida pela
medição. O que se asserta é `db_max − db_min <= 80`.

O deslocamento do MFCC prova que `librosa.feature.mfcc` aplica `power_to_db` com os
**defaults** (`ref=1.0`) e isola o ganho no `c0` — é a evidência da paridade que
justifica a P5.

**Nº de frames independe do `n_fft`:** 251 para 512, 1024 e 2048.

### O frame de transição — precisão que vale para os dois ramos

`win_length = 400` é **maior** que `hop = 256`. Logo as janelas se sobrepõem, e a
fronteira entre áudio e padding **não** é um degrau limpo:

- `frames_validos()` define frame válido **pelo centro**: o frame `i` é válido
  enquanto `i·hop < n_amostras_validas`, o que dá `ceil(n_amostras_validas / hop)`;
- o frame `n_valid` (o primeiro **inválido**) está centrado em `n_valid·256` e sua
  janela cobre `[n_valid·256 − 200, n_valid·256 + 200)` — então ele **ainda alcança
  até 200 amostras de áudio real**. É um frame de **transição**, não de padding puro;
- pelo mesmo motivo, o frame `n_valid − 1` (o último **válido**, que **entra** na
  agregação) pode conter até 200 amostras de zero-padding na sua janela.

Medido no piloto: `[:, n_valid:]` tem mais de um valor distinto em **120 de 201**
exemplos. **Isto não é erro da máscara.** A definição é idêntica nos dois ramos —
`extrair_vetor` agrega `serie[:, :n_valid]` e a máscara da CNN marca as mesmas
`n_valid` posições —, então o frame de transição é excluído dos dois, e o último
frame válido é parcialmente contaminado nos dois, **na mesma medida**. A paridade,
que é o que a comparação exige, está intacta: item 7 da checagem deu 201/201.

É uma aproximação de ±1 frame, deliberada, herdada da definição do piloto do Bloco 1
e mantida para que a máscara da CNN e o mascaramento clássico usem **exatamente o
mesmo número** para o mesmo áudio. Exigir suporte integral da janela reduziria a
contagem em 1 e abriria divergência com o `features.csv` congelado. **Resposta de
banca em `APENDICE_B_banca.md` §3.**

### O topo morto por codec — observação que vira evidência

`fmax = 8000` dá 128 faixas Mel até 8 kHz, mas **57,9% do universo é banda estreita**
(alaw, ulaw, gsm, pstn, teto ~4 kHz). Nesses áudios a metade superior das faixas Mel
é um platô constante no piso do `top_db`. Medido no piloto: topo morto em **65 de 118**
áudios de banda estreita e em **2 de 83** de banda larga (ambos `opus`).

**É o codec, não o `fmax`** — e isso é útil de três formas:

1. **valida a rejeição do `fmax=4000` global** de forma visual: nos 42,1% de banda
   larga há sinal real acima de 4 kHz, que um filtro global destruiria;
2. **explica** por que a banda larga tem desempenho melhor nos dois modelos clássicos
   (RF 0,6958 → 0,7510; SVM 0,7711 → 0,8397);
3. **não é atalho de classe.** A subamostra é estratificada por classe × codec ×
   ataque, então o codec não prediz o rótulo. A rede pode aprender regras
   específicas por codec — que é o que RF e SVM já fazem implicitamente.

Aplicando a regra de escopo: **não impede a validade do experimento principal.** Vai
para a **discussão** (§Análise por codec), não para experimento novo.

**Razão spoof:bonafide = 9,0 nos quatro conjuntos:** treino 9,001 · validação 9,003 ·
teste 8,999 · **subamostra 30k exatamente 9,0 (27.000 / 3.000)**. O desbalanceamento
é propriedade do **protocolo**, não de um conjunto — é o que sustenta a P7.

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
  *Atenção ao citar:* −0,0466 é a **média das reamostragens**; a diferença
  **observada** é −0,0468 (0,1462 − 0,1930), e é ela que `COMPARACAO_FINAL.md` põe na
  coluna ΔEER. No texto, use o **observado com o IC95** — o padrão — e não misture os
  dois números. (Em f1 os dois coincidem: +0,0762.)
- O braço de referência **não** é concorrente direto de SVM/CNN — ele só quantifica o
  custo da subamostragem. Isso tem de estar dito explicitamente no texto.
- A curva de aprendizado do RF **não satura**: extrapolando `f1 ~ a·ln(n)+b`
  (R² = 0,9934), o RF precisaria de ~276.116 áudios de treino para alcançar o f1 do
  SVM — 1,86× o universo eval inteiro.

**Tempos ponta a ponta** (`results/metricas/tempo_pipeline_completo.json`, medição de
**20/09/2026**, por áudio, batch = 1, amostra de 200 áudios da validação):

| ramo | base (carregar + VAD/padding + representação) | predição | **total** |
|---|---:|---:|---:|
| RF ajustado (principal) | 0,5735 + 0,3394 + 3,4345 (features) = **4,3474** | 5,8819 | **10,2293** |
| SVM RBF ajustado (principal) | mesma base: **4,3474** | 0,5717 | **4,9191** |
| CNN final — **GPU** | 0,5815 + 0,3497 + 2,4775 (log-Mel) = **3,4087** | 1,1678 | **4,5765** |
| CNN final — **CPU** | mesma base: **3,4087** | 13,3881 | **16,7968** |

> **Estes números substituem os da medição anterior** (RF 11,12 · SVM 5,33). A
> diferença é de ~8% e **não** vem de mudança no código de RF/SVM: os três modelos
> foram recronometrados **na mesma execução**, em B4.7, porque comparar a CNN medida
> hoje com RF/SVM medidos três semanas antes, noutro estado de máquina, não é
> comparação. Se um número de tempo aparecer em outro documento com o valor antigo,
> **este aqui manda** — e o outro se corrige.

Quatro leituras, e as quatro vão para o texto:

- **A hipótese de que o pré-processamento dominaria não se confirmou para o RF** (a
  predição é 57,5% do total dele), **mas se confirma para o SVM** (a base é 88,4% do
  total) e **para a CNN em GPU** (a base é 74,5%, com o log-Mel sozinho em 54,1%). É
  o mesmo pipeline; o que muda é o custo do classificador no fim dele.
- **A CNN em GPU é mais barata ponta a ponta que o RF** (4,58 contra 10,23 ms) — e a
  CNN em CPU é a mais cara de todas (16,80 ms). **A resposta sobre custo
  computacional é condicional ao hardware**, e tem de ser escrita assim.
- **Em lote a ordem muda de novo** (`tempos_inferencia.throughput` nos JSONs de cada
  modelo, batch = 22.226): RF **0,0204** ms/áudio, CNN-GPU **0,2383**, SVM **0,4025**
  — o RF é ~19,7× melhor que o SVM. Latência e vazão **não** classificam os modelos
  na mesma ordem; reportar só uma delas é meio resultado.
- **Armadilha de medição registrada em B4.7:** os `descartar_aquecimento: 3` do
  `config.yaml` são 3 *iterações* (~3 ms em GPU) e **não** tiram a placa do estado de
  baixo consumo depois de um trecho longo de CPU. Medido: **8,83 ms com a GPU fria
  contra 0,82 ms quente** — a primeira execução publicou o número frio e
  superestimava a latência da CNN em ~10×. É o **espelho** da armadilha do
  `torch.cuda.synchronize()` (0,91 ms falsos contra 28,70 ms reais): uma infla, a
  outra desinfla, e nenhuma das duas se detecta lendo o código. As duas ficam
  registradas em `cnn_final_principal.json`, com o número frio publicado ao lado do
  quente.

> **Não confundir escopos.** A `latencia_ms` dentro de `rf_tuned_principal.json` e
> companhia mede **só a predição**, a partir da representação já extraída e noutra
> execução — daí RF 6,1075 ms lá e 5,8819 ms aqui. O número que responde «quanto
> custa classificar um áudio» é o **total** da tabela acima.

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

> **Atualização de 23/09/2026 — o teste JÁ FOI USADO.** O B5.1 rodou uma vez, em
> 23/09, com os limiares da validação (`results/metricas/teste_lacrado.json`,
> commit `498f3c8`). `scripts/avaliar_teste_lacrado.py` recusa uma segunda execução
> (só com `--forcar --razao`, que fica gravada no artefato). Os scores do teste estão
> em `scores_teste_lacrado.csv`; qualquer análise nova sobre o teste **lê esse CSV**,
> nunca pontua de novo. Análise complementar nova (ex.: sementes extras da CNN) se
> avalia **só na validação**. Resultados finais: `COMPARACAO_FINAL.md`.
>
> O texto abaixo é a regra que vigorou até o B5.1, mantida como registro.

O conjunto de **teste (22.227)** é usado **uma única vez**, em **B5.1**, com os
limiares já escolhidos na validação. Até lá:

- nenhum script de diagnóstico, curva ou ablação toca em `conjunto == 'teste'`;
- `carregar_modelo_ajustado` (`src/models/modelos_ajustados.py`) **recusa** um limiar
  cujo JSON não registre `selecao_limiar.conjunto == 'validacao'`. A regra que protege
  o teste mora no código, não só no texto — mantenha assim ao estender o módulo para
  a CNN.
