# 00 — Mapa de execução do TC II (Bloco 4 → congelamento em 11/10)

> **O que este arquivo é.** O índice e o contrato de uso dos briefings de execução.
> Cada marco do cronograma vigente tem **um** arquivo próprio nesta pasta, dimensionado
> para ser executado numa **sessão nova e limpa** do Claude Code, sem precisar carregar
> o repositório inteiro no contexto.
>
> **O que este arquivo não é.** Não é documento de consulta ao orientador. As sete
> decisões do espectrograma foram **fechadas em 17/09** e não se reabrem
> (regra 10 das instruções do projeto).

---

## 1. Como usar esta pasta

**Uma sessão nova do Claude Code por marco.** Não encadeie dois marcos na mesma
sessão: o Bloco 4 tem etapas que produzem artefatos grandes (74.453 tensores, curvas
de treino, JSONs de métrica) e uma sessão que arrasta o contexto de três etapas
anteriores começa a errar caminhos de arquivo e a reabrir decisões fechadas.

O procedimento, por marco:

```
1. sessão nova
2. ler docs/execucao/<arquivo do marco>.md                  (o briefing)
3. ler docs/execucao/APENDICE_A_inventario.md               (o estado congelado)
4. ler SOMENTE os arquivos listados em "Contexto mínimo" do briefing
5. executar
6. fechar pelo "Critério de pronto"
7. commitar com a mensagem-modelo do briefing
8. encerrar a sessão
```

Os passos 3 e 4 existem porque o repositório tem ~80 arquivos de resultado e
~30 scripts. Ler tudo é o caminho mais rápido para estourar o contexto e passar a
alucinar nomes de coluna. **O briefing já fez essa triagem — confie nela.**

### Orçamento de contexto por sessão

| o que carregar | custo aproximado |
|---|---|
| o briefing do marco | 3–6 mil tokens |
| `APENDICE_A_inventario.md` | ~4 mil tokens |
| os 2 a 5 arquivos de "Contexto mínimo" | 6–20 mil tokens |
| **subtotal de partida** | **~15–30 mil tokens** |

Sobra folga confortável para o trabalho em si. Se durante a execução você sentir
necessidade de abrir mais de **três** arquivos que não estavam na lista, isso é
sinal de que o briefing está errado ou de que você saiu do escopo do marco —
pare e releia a seção 3 deste arquivo antes de continuar.

---

## 2. Índice

| arquivo | marco | data-limite | bloqueia o quê |
|---|---|---|---|
| `B4.0_fechamento_metodologico.md` | B4.0 | **18/09** | tudo. Sem o `config.yaml` atualizado, o gerador não tem de onde ler os parâmetros |
| `B4.1_pipeline_logmel.md` | B4.1 | **20/09** | o lote definitivo |
| `B4.2_lote_definitivo.md` | B4.2 | **21/09** | todo o treino da CNN |
| `B4.3_split_interno_normalizacao.md` | B4.3 | **21/09** | o treino (é onde nasce `normalizacao_cnn.json`) |
| `B4.4_cnn_baseline.md` | B4.4 | **23/09** | a definição da arquitetura |
| `B4.5_cnn_definida.md` | B4.5 | **25/09** | o refit |
| `B4.6_refit_final.md` | B4.6 | **27/09** | a validação externa |
| `B4.7_validacao_externa.md` | B4.7 | **28/09** | o teste lacrado |
| `B5_teste_lacrado_e_comparacao.md` | B5.1 + B5.2 | **29/09** | a redação dos resultados |
| `B6_redacao_e_fechamento.md` | B6.1–B6.5 | **04/10** e 11/10 | a entrega |

**Apêndices** (consulta, não execução):

| arquivo | para quê |
|---|---|
| `APENDICE_A_inventario.md` | estado congelado do repositório: hashes, contagens, o que cada artefato é, o que não se toca. **Ler em toda sessão.** |
| `APENDICE_B_banca.md` | as frentes críticas de banca, com a resposta já ancorada no código. Usar em B6.1 e na simulação de banca. |
| `APENDICE_C_riscos.md` | tabela de correção de rota: se o marco X atrasar, o que se simplifica e qual limitação se escreve. **Abrir no primeiro sinal de atraso, não depois.** |

---

## 3. A regra de escopo (vigora até 11/10)

Colada aqui porque é o que decide, na hora, se algo entra ou sai. Diante de
qualquer questão metodológica nova:

> **"Essa decisão impede a validade do experimento principal?"**
>
> - **Sim** → resolve agora.
> - **Não** → registra como **limitação**, **análise complementar** ou **trabalho
>   futuro**, e **continua o desenvolvimento**.

Três consequências práticas, para quem está executando:

1. **Achado interessante não é tarefa.** Se você descobrir algo relevante durante
   um marco, escreva uma linha em `docs/execucao/ACHADOS.md` (crie o arquivo na
   primeira vez) com a frase «não impede a validade — vai como limitação/trabalho
   futuro» e **volte ao marco**. Não abra script, não abra experimento.
2. **Sugestão sem orçamento de dias não é sugestão.** Toda proposta de melhoria
   precisa vir com uma estimativa em dias confrontada com a data-limite do marco
   em curso. Se não cabe, é trabalho futuro.
3. **Atraso corrige-se simplificando**, com a limitação escrita junto — nunca
   omitida. A tabela está em `APENDICE_C_riscos.md`.

**A data real de entrega é 04/10.** O que existe entre 04/10 e 11/10 é o tempo do
orientador, não o seu.

### A regra de escopo é sobre atraso, não sobre ritmo

As datas dos briefings são **teto, não agenda**. Adiantar é o resultado desejado, e
a regra de escopo **não** se aplica a quem está na frente: não há o que simplificar
nem limitação a escrever. Ela entra em cena quando uma data-limite está em risco.

O que continua valendo em qualquer ritmo são **os quatro portões** e a **fila do
tempo excedente**, ambos em `APENDICE_A_inventario.md` → «Ritmo de execução» — que é
onde a instrução mora, para que toda sessão do Claude Code a leia junto com o
briefing. Em resumo: adiantar sim; pular pré-requisito e abrir eixo experimental
novo, não.

E «uma sessão por marco» é sobre **higiene de contexto**, não sobre esperar a data
chegar: terminou o marco e sobrou dia, encerre a sessão e abra a próxima com o
briefing seguinte.

---

## 4. Estado do repositório em 23/09/2026

> **Esta seção é datada de propósito e precisa ser atualizada ao fechar cada
> marco.** Ela existe para que uma sessão nova não precise inspecionar o
> repositório inteiro para saber onde o trabalho está. Uma seção de estado
> desatualizada é pior que nenhuma: manda a sessão executar o que já foi feito.

| item | estado |
|---|---|
| **Blocos 1, 2, 3, 4 e 5** | **concluídos** — o Bloco 4 fechou em 20/09 (8 dias antes do teto) e o Bloco 5 em 23/09 (6 dias antes) |
| `data/features/features.csv` | **CONGELADO** — MD5 `51b2f439bf6f1e10237acbc620bb92d9`, 148.176 linhas |
| `data/processed/split.csv` | **CONGELADO** — MD5 `9143f0c7b83ec2db4aa144ed5deb3402` |
| `data/processed/subamostra_30k.csv` | **CONGELADO** — MD5 `654cb796b738512388b28e15ffb14a9d`, n = 30.000 |
| `data/espectrogramas/` | **CONGELADO** — 74.453 tensores `(128, 251)` float32 gerados em B4.2; `espectrogramas.meta.json` versionado |
| `normalizacao_cnn.json` (27k) e `normalizacao_cnn_30k.json` (refit) | gravados e versionados |
| RF e SVM ajustados | fechados; artefatos e métricas em `models/` e `results/metricas/` |
| **CNN final** | **fechada** — refit nos 30k (37 épocas fixas, 240.866 parâmetros), validada externamente em B4.7 |
| `MODELOS_PRINCIPAIS` | contém os **três** modelos; `scores_de` cobre RF, SVM e CNN |
| Conjunto de **teste** | **USADO — uma única vez, em 23/09 (B5.1)**. `avaliar_teste_lacrado.py` recusa nova execução; os scores estão em `scores_teste_lacrado.csv` e nada mais pontua o teste |
| **Modelos** | **congelados desde o B5.2** — nenhuma mudança, salvo erro grave |
| Comparação final | `results/metricas/COMPARACAO_FINAL.md` (gerado — não editar à mão), `comparacao_final.csv`, `comparacao_estatistica.json`, 5 figuras `comparacao_*.png` |
| `config/config.yaml` → bloco `espectrograma:` | **fechado em B4.0**, com os defaults do librosa registrados explicitamente |
| `results/metricas/DECISOES_PENDENTES_CNN.md` | **respondido** — as sete decisões transcritas em B4.0; não se reabre |
| Figuras de matriz de confusão | regeradas em 23/09 (correção de título cortado); regeram-se com `scripts/replotar_matrizes_confusao.py`, sem retreinar |
| ramo git | `master` (não `main`) |
| último commit | `c07390b` — *B5.2: comparacao experimental fechada — RF x SVM x CNN* (+ o commit de revisão das leituras e dos documentos de estado) |
| **próximo marco** | **B6.1 — redação** (`B6_redacao_e_fechamento.md`) |

**Resultado final no teste lacrado (22.227), execução única, limiar escolhido na
validação:**

| modelo | limiar | f1_macro | EER | ROC-AUC | recall bonafide |
|---|---:|---:|---:|---:|---:|
| RF ajustado — principal | 0,6516 | 0,7210 | 0,1946 | 0,8878 | 0,5326 |
| RF ajustado — referência | 0,6196 | 0,7602 | 0,1629 | 0,9167 | 0,5794 |
| SVM RBF ajustado — principal | −0,0329 | 0,7981 | 0,1411 | 0,9315 | 0,6635 |
| **CNN final — principal** | 0,3252 | **0,8883** | **0,0737** | **0,9783** | **0,7877** |

Bootstrap pareado: os três pares (SVM−RF, CNN−RF, CNN−SVM) com IC95 sem zero em
f1_macro e em EER, na validação e no teste. Leituras completas, geradas a partir dos
números, em `COMPARACAO_FINAL.md`.

**Resultado na validação externa (22.226), com limiar escolhido na validação:**

| modelo | limiar | f1_macro | EER | ROC-AUC | recall bonafide |
|---|---:|---:|---:|---:|---:|
| RF ajustado — principal | 0,6516 | 0,7225 | 0,1930 | 0,8873 | 0,5441 |
| RF ajustado — referência | 0,6196 | 0,7723 | 0,1579 | 0,9191 | 0,6008 |
| SVM RBF ajustado — principal | −0,0329 | 0,7987 | 0,1462 | 0,9289 | 0,6625 |
| **CNN final — principal** | 0,3252 | **0,8975** | **0,0738** | **0,9799** | **0,8101** |

Se um número aqui divergir do JSON correspondente em `results/metricas/`, **o JSON
manda** — e esta tabela se corrige no mesmo commit.

> **Nota sobre o GitHub.** O espelho em
> `github.com/EduardoZanellaBento/Tcc_Deepfake` não é consultável por ferramenta
> automática (o GitHub bloqueia a listagem de commits por `robots.txt`). A pasta
> `C:\dev\Tcc_Deepfake` é a fonte de verdade destes briefings, e pode estar adiante
> do que qualquer leitura do GitHub mostraria.

---

## 5. Convenções que valem em todos os marcos

**Execução.** Sempre da raiz do projeto, com `python -m` (os imports são
relativos): `python -m src.features.gerar_espectrogramas`, nunca
`python src/features/gerar_espectrogramas.py`.

**Configuração.** Todo parâmetro metodológico vem do `config.yaml`, lido por
`carregar_config` (`src/utils/config.py`). Nenhum número metodológico
hard-codado em script — «config que ninguém lê é comentário com sintaxe YAML».

**Semente.** `fixar_seeds(cfg["semente"])` para o ramo clássico;
`fixar_seeds_torch(cfg["semente"])` para a CNN, chamada **antes de qualquer
chamada CUDA**. Resultado principal em **seed 42**.

**Fonte única.** Nunca reimplemente o que já existe:

| você precisa de | use | nunca |
|---|---|---|
| carregar + normalizar + VAD + padding | `preprocessar_audio` (`src/data/preprocessamento.py`) | reescrever o VAD |
| nº de frames válidos | `frames_validos` (`src/features/extrair_features.py`) | recontar «na mão» |
| decisão a partir de score | `aplicar_limiar` (`src/models/avaliacao.py`) | `modelo.predict()` / argmax |
| escolher limiar | `selecionar_limiar` (idem) | grade fixa `np.arange` |
| métricas | `avaliar` (idem) | `classification_report` solto |
| EER | `calcular_eer` (idem) | implementação nova |
| cronometrar | `medir_tempos` (`src/models/tempo.py`) | `time.time()` avulso |
| gravar JSON com tipos numpy | `json_seguro` (`src/utils/serializacao.py`) | `str(obj)` |

A razão é sempre a mesma: **cópia é a origem clássica da divergência silenciosa
entre quem grava e quem lê**, e o trabalho inteiro depende de RF, SVM e CNN serem
medidos pela mesma régua.

**Git diário.** Cada marco fecha com commit. A mensagem-modelo está no briefing.
Todo marco deve deixar evidência de **o que rodou**, **com quais parâmetros** e
**qual foi o resultado** — se não está num artefato versionado, não aconteceu.

**Artefato gerado não recebe edição à mão.** O Bloco 3 já foi mordido por isso: um
bloco de limitação escrito manualmente dentro de um JSON gerado foi apagado em
silêncio pela re-execução seguinte. Se uma informação precisa sobreviver, ela é
**derivada pelo código** que gera o artefato.

---

## 6. O caminho crítico, em uma frase

```
config fechado → gerador validado → 74.453 tensores → split 27k/3k + normalização
→ CNN que funciona → CNN definida → refit nos 30k → limiar na validação externa
→ teste lacrado (uma vez) → tabela RF × SVM × CNN → texto → 04/10
```

Nada fora dessa linha entra antes de 29/09. A análise de variância entre sementes
(3 sementes na CNN), o experimento cross-attack e qualquer ablação adicional são,
por decisão registrada, **fora do caminho crítico**: fazem-se se sobrar tempo, e
sua ausência entra como limitação computacional declarada.