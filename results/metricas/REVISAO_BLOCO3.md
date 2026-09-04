# Revisão do Bloco 3 — registro completo (03/09/2026)

> **O que é este documento.** Em 03/09/2026 o repositório passou por uma revisão
> profunda: código do Bloco 3, JSONs de resultado, notas técnicas, `config.yaml`,
> `README.md` e os artefatos de `results/`. Cada observação virou uma tarefa, e
> cada tarefa está registrada aqui com **a evidência que a sustenta**, **a
> decisão tomada** e **o estado**. É este o documento que responde "por que
> estes commits existem" daqui a um mês, e é dele que sai o relato ao
> orientador.
>
> **Guardrails que valeram para toda a revisão:** teste LACRADO (nenhum script
> novo ou alterado lê `conjunto == 'teste'`); features CONGELADAS desde
> 30/08/2026 (`features.csv` MD5 `51b2f439bf6f1e10237acbc620bb92d9`);
> `split.csv` e `subamostra_30k.csv` imutáveis; artefatos históricos preservados
> sob nome novo.

## Índice de estado

| id | assunto | estado |
|---|---|---|
| R1 | diagnósticos por ataque e por codec sobre os modelos ajustados | **feito** |
| R2.1 | tempo de treino comparável entre RF e SVM | **feito** |
| R2.2 | `roc_auc_cv_std` no JSON do SVM | **feito** |
| R2.3 | re-execução + guarda de reprodução | **feito** |
| R3 | flag de saturação honesta na curva de aprendizado | **feito** |
| R4a | escopo declarado do cronômetro de inferência | **feito** |
| R4b | tempo do pipeline completo (featurização + predição) | **feito** |
| R5.1 | docstring do EER × limitação declarada do split | **feito** |
| R5.2 | sombreamento de `limiar` em `avaliar()` | **feito** |
| R5.3 | README lidera com o bootstrap pareado | **feito** |
| R6.1 | extrapolação da curva de aprendizado do RF | **feito** |
| R6.2 | variância entre subamostras | **feito** |
| R6.3 | `permutation_importance` no RF ajustado | **feito** |
| R7 | registro em texto (este documento e os demais) | **feito** |
| **Tópico 2** | definição do espectrograma da CNN | **BLOQUEADO** — ver `DECISOES_PENDENTES_CNN.md` |

---

## Correções à própria revisão

Duas afirmações da revisão original foram refinadas na reconferência sobre os
arquivos. **Valem os valores conferidos, não os primeiros.**

| item | citado antes | **conferido** | onde |
|---|---|---|---|
| fator de dados para o RF alcançar o SVM | "~2,3× / ~240 mil" (estimativa de olho) | **ajuste por mínimos quadrados** sobre `curva_aprendizado_rf_tuned_eval.csv` — número exato em `extrapolacao_curva_rf.json` | R6.1 |
| frames do espectrograma | "251" | **251, confirmado em três fontes independentes** | `1 + 64000//256 = 251`; `frames_validos()` em `src/features/extrair_features.py:102-118`; `n_frames_total` único = 251 no CSV |

Tudo o mais da revisão se confirmou na releitura.

---

## R1 — Diagnósticos por ataque e por codec estavam obsoletos *(a pendência mais séria)*

**Observação.** `diagnostico_por_ataque.csv` e `diagnostico_por_codec.csv` eram
ambos de **13/08/2026** — anteriores ao lote único, medidos no **RF baseline**
com limiar **0,50**. E o README os citava como *a* mitigação da limitação mais
frágil declarada no trabalho (split aleatório por utterance: o modelo pode
decorar a assinatura do vocoder).

**A evidência de que a tabela estava vazia.** Naquele arquivo, `recall = 1,0` em
**todos os 13 ataques** e `recall_bonafide` entre 0,00 e 0,23. Isso não é
diagnóstico por ataque — é o modelo dizendo "spoof" para quase tudo no limiar
0,50. Não havia informação nenhuma ali.

**Diagnóstico técnico do que estava errado.** Os dois scripts
(`diagnostico_por_ataque.py:73-79`, `diagnostico_por_codec.py:71-78`)
**re-treinavam um RF baseline** (`n_estimators=100, max_depth=None,
class_weight='balanced'`) no **treino completo (103.723)** — não era o modelo do
braço principal, não tinha os hiperparâmetros do Random Search, e não era o
modelo cujos números estão no README. Além disso, os dois decidiam por
`modelo.predict()` (argmax @ 0,50), violando a regra única do protocolo
(`score >= limiar`, `src/models/avaliacao.py`) — a docstring de `aplicar_limiar`
existe exatamente para impedir isso. E nenhum dos dois tocava no **SVM**, que é o
modelo vencedor do braço principal.

**Decisão e implementação.** Os dois scripts foram reescritos para **carregar os
modelos persistidos** em vez de treinar, e para medir **RF e SVM**:

- limiar **lido** de `selecao_limiar.limiar` no JSON companheiro — nunca
  recalculado, nunca 0,50; o carregador recusa um limiar que não tenha sido
  selecionado na validação;
- decisão por `aplicar_limiar` (`src/models/avaliacao.py`), nada de
  `modelo.predict()`;
- score na escala nativa de cada modelo — `predict_proba[:,1]` no RF,
  `decision_function` no SVM (o `Pipeline` já embute o `StandardScaler`);
- `colunas_features(df)` calculado **antes** do merge com `labels.csv` — um X com
  colunas a mais ou em outra ordem faz o modelo carregado devolver lixo **sem
  levantar exceção**;
- **só validação**; o teste segue lacrado.

O par (modelo, limiar) passou a ser montado num ponto único,
`src/models/modelos_ajustados.py`, para que os dois diagnósticos (e o que vier
depois) não possam divergir silenciosamente.

**Artefatos.** `diagnostico_por_{ataque,codec}_{rf,svm}_tuned_principal.csv`
(+ figuras) e os dois JSONs de resumo com limiar, hashes MD5 dos três artefatos
congelados e ambiente. Os arquivos de 13/08 foram **renomeados** para
`*_baseline_2026-08-13.*` e continuam citáveis como referência "antes".

**Leitura crítica — os números medidos.**

*Por ataque:* o EER vai de **0,0628 (A13) a 0,3548 (A16)** no RF — amplitude
**0,2920**, razão 5,7× — e de **0,0617 (A09) a 0,2598 (A16)** no SVM, amplitude
0,1981. O recall deixou de saturar: no RF vai de 0,7266 a 0,9987, contra 1,0 em
todos os 13 ataques na versão baseline. **A amplitude entre sistemas de síntese
(até 0,29) é ~6× a distância entre os dois modelos (ΔEER 0,0466)** — evidência
**a favor** do risco declarado: o número agregado depende de quais ataques
compõem o conjunto, e o split por utterance deixa o modelo ver a assinatura de
cada vocoder já no treino. O experimento cross-attack segue necessário.

*Por codec:* a hipótese da banda alta **se sustenta nos dois modelos**. RF —
banda estreita f1 0,6958 / EER 0,2056 contra banda larga f1 0,7510 / EER 0,1676.
SVM — 0,7711 / 0,1652 contra 0,8397 / 0,1174. Como o **EER é independente de
limiar**, o contraste **não** era artefato do limiar 0,50: é propriedade das
features, e reforça a rejeição de um `fmax=4000` global.

---

## R2 — Assimetrias e omissões nos artefatos de resultado

### R2.1 — Tempo de treino não era comparável

**Observação.** O RF treinava com `n_jobs=-1` (4,27 s) e o SVM é single-thread
(11,24 s). O JSON do RF registrava `n_jobs_treino: -1`; o do SVM **não registrava
nada**. Se essa linha entrasse na tabela final como estava, compararia 6 núcleos
contra 1.

**Decisão.** Duas medições, cada uma rotulada no artefato:
`tempo_treino_s` (n_jobs=-1) é o **custo real de uso**;
`tempo_treino_s_n_jobs_1` é o refit da mesma configuração e da mesma semente com
`n_jobs=1` — o **único número comparável ao SVM**. O `treinar_svm.py` passou a
gravar `n_jobs_treino = 1`, com o comentário de que o `SVC` é single-thread **por
natureza** (libsvm não paraleliza o fit): não é escolha de configuração, é o
algoritmo. O campo `nota_tempo_treino` explica no próprio JSON qual é qual.

### R2.2 — `svm_random_search.json` sem `roc_auc_cv_std`

**Observação.** O JSON do RF registrava `roc_auc_cv_std`; o do SVM não.
Assimetria gratuita entre dois artefatos que vão para a mesma tabela.
**Decisão.** Acrescentado, espelhando `ajustar_rf.py`.

### R2.3 — Guarda de reprodução

R2.1, R2.2 e R4a mudaram o **esquema** dos JSONs, então `ajustar_rf` e
`treinar_svm` tiveram de rodar de novo. Como os dois são **determinísticos**
(semente 42 fixada, `RandomizedSearchCV` com `random_state=42`, `SVC` com
`probability=False`, e `n_jobs` do RF não altera resultado), a re-execução
**tinha de reproduzir** os números anteriores — o que antecipou de graça a
verificação de reprodutibilidade prevista para o B6.1.

**Procedimento:** JSONs antigos copiados para `results/metricas/_pre_revisao/`
(versionado); re-execução; comparação campo a campo por
`scripts/guarda_reproducao.py`. Ignorados na comparação: `tempos_inferencia`,
`tempo_treino_s`, `tempo_treino_s_n_jobs_1`, `tempo_busca_s`,
`tempo_um_fit_fold_s` e `ambiente` — medidas de relógio **variam por
definição**, e compará-las produziria alarme falso.

**A guarda pegou duas coisas na primeira passada — e é por isso que ela existe.**

**(1) Achado real: uma limitação metodológica que vivia só num arquivo gerado.**
O bloco `limitacao_otimo_na_borda` de `rf_random_search.json` — o registro de
que o `min_samples_leaf` ótimo (5) caiu no **limite inferior** da faixa exigida
pelo orientador, e o `min_samples_split` (10) no topo da lista — **nunca esteve
no código**: foi escrito à mão dentro de um artefato **gerado**. A re-execução o
apagou, e a guarda acusou "regressão de rastreabilidade".

O diagnóstico é o que importa: *uma limitação declarada que mora num arquivo
gerado é destruída, em silêncio, por qualquer re-execução* — inclusive pelas do
Bloco 5 e do B6.1. A correção não foi reescrevê-la à mão. Foi criar
`analisar_bordas()` em `src/models/ajustar_rf.py`, que **deriva** o bloco do
espaço de busca e da configuração vencedora, e portanto: (a) volta sozinho a
cada execução; (b) não pode ficar desatualizado se a busca um dia eleger outra
configuração. A função reproduz as três notas do texto original, inclusive a
distinção fina que o texto fazia — `max_depth=30` é o maior valor **finito** da
lista, mas `None` estava disponível e **não** foi escolhido, logo o ótimo **não
está censurado** nessa dimensão. O campo `censurados` do JSON passa a listar
exatamente as dimensões em que a faixa limita a conclusão.

**(2) Alarme falso corrigido: `projecao_horas_antes_da_busca` (0,03 → 0,04).**
Esse campo é `tempo_um_fit_fold_s × n_iter × 5 ÷ paralelismo` — puro derivado de
relógio (o fit de referência foi 4,6 s antes e 5,3 s agora). Entrou na lista de
ignorados, **com o corte explícito entre medida e consequência**: a projeção é
ignorada, mas o `n_iter_efetivo` que ela decide continua sendo comparado — se um
dia a projeção estourar o orçamento e cortar o `n_iter`, o resultado muda de
verdade e a guarda tem de gritar.

**(3) Segundo achado real: `predict_proba` do RF não é reprodutível bit a bit
com `n_jobs=-1`.** Na segunda passada a guarda pegou
`selecao_limiar.n_candidatos` do braço de referência mudando de **22.103 para
22.104** — um campo de resultado, num pipeline declarado determinístico.

A hipótese foi testada, não suposta. Carregando o **mesmo** `.joblib` e
predizendo quatro vezes sobre a **mesma** validação:

| `n_jobs` | `n_candidatos` nas 4 execuções | vetores idênticos bit a bit? |
|---|---|---|
| −1 | 22.104 / 22.102 / 22.104 / 22.103 | **não** |
| 1 | 22.104 / 22.104 / 22.104 / 22.104 | **sim** |

Maior diferença absoluta entre os dois vetores de score: **4,4 × 10⁻¹⁶**.
`f1_macro` e `EER` batem **até a décima casa decimal**.

**Causa.** `RandomForestClassifier.predict_proba` com `n_jobs != 1` acumula a
contribuição das 300 árvores num array compartilhado, em paralelo. A **ordem da
soma** varia entre execuções, e soma de ponto flutuante não é associativa —
então os últimos bits do score mudam. Nenhuma métrica sente isso; `n_candidatos`
sente, porque é uma **contagem de valores distintos** e portanto enxerga o
último bit.

**Por que corrigir, se nenhuma métrica muda.** Porque `n_candidatos` é um número
**publicado** — é ele que prova, na `NOTA_LIMIAR.md`, que o score do RF deixou de
ser degrau (22.207 valores distintos em 22.226 amostras) — e porque um artefato
que não reproduz byte a byte obriga quem confere a decidir, no calor da hora, se
a diferença importa. Uma guarda que às vezes grita à toa deixa de ser guarda.

**Correção.** `predizer_rf()` em `src/models/avaliacao.py` força `n_jobs=1` na
**predição** (o treino segue paralelo — lá o paralelismo não afeta o resultado,
só o tempo). Todos os pontos que predizem com RF passaram a usá-la:
`ajustar_rf`, `treinar_rf`, `modelos_ajustados.scores_de` (logo os dois
diagnósticos do R1), `curva_aprendizado_rf_tuned`, `ablacao_mfcc1_std`,
`estabilidade_modelos` e `estabilidade_subamostra`. Custo: cerca de 1 s a mais
por predição de lote.

**Consequência para o Bloco 5:** o limiar — que é o que atravessa do protocolo
para o teste lacrado — sempre reproduziu (0,6516 no principal, 0,6196 na
referência). O risco era de rastreabilidade, não de contaminação do teste. Mas a
avaliação final agora roda sobre um caminho de predição determinístico, que é
como tinha de ser.

**Resultado após as três correções: REPRODUZIU** em todos os artefatos
comparados. A evidência está em `results/metricas/reproducao_bloco3.json`, com a
lista de campos comparados, os campos acrescentados e a data. **É este o arquivo
a citar na resposta de banca sobre reprodutibilidade** — e o achado (1) é, por
si, uma boa resposta à pergunta "por que vocês fizeram uma guarda de
reprodução?".

`estabilidade_modelos` e `ablacao_mfcc1_std` dependem dos `.joblib` re-gerados e
foram re-rodados em seguida, sob a mesma guarda.

---

## R3 — `satura_em_n` dizia o contrário do que o dado mostra

**Observação.** `satura_em_n: 103723` lia-se como "saturou em 103k", quando o
próprio dado mostra que **não saturou**: o último salto (80k → 103.723) ainda
rende **+0,0121** em f1_macro, mais que o dobro da tolerância de 0,005.

**Diagnóstico técnico.** Em `curva_aprendizado_rf_tuned.py:192-193`,

```python
f1_max = res["f1_macro"].max()
saturado = res[res["f1_macro"] >= f1_max - TOL_SATURACAO].iloc[0]
```

Numa curva **monótona crescente** o máximo é sempre o último ponto, e a
expressão devolve o último ponto sempre que nenhum anterior esteja a menos da
tolerância dele. **O campo era estruturalmente incapaz de distinguir "saturou no
fim" de "nunca saturou".** O script já suspeitava disso — a conclusão hedgeava a
frase —, mas o JSON e a legenda da figura afirmavam saturação sem ressalva.

**Decisão.** A saturação virou um booleano de definição explícita:
`saturou = (f1 do maior n) − (f1 do n anterior) < TOL`. O JSON passou a trazer
`saturou`, `maior_n_medido`, `ganho_ultimo_passo_f1` e `definicao_saturacao`;
`satura_em_n` **só é emitido quando `saturou` é verdadeiro**. A figura só desenha
a linha vertical de saturação quando ela existe — caso contrário anota **"não
satura no maior n disponível"**. A conclusão foi reescrita nas duas ramificações,
**sem hedge**.

---

## R4 — `incluir_extracao_features: true` estava declarado e nunca foi implementado

**Observação.** `medir_tempos` lia apenas `repeticoes`,
`descartar_aquecimento`, `medir_latencia` e `medir_throughput` — e mesmo assim o
JSON gravava `"fonte": "config.yaml -> tempo"`, **sugerindo conformidade total**
com um bloco que inclui `incluir_extracao_features: true`. O próprio comentário
do `config.yaml:225-230` já dizia que a hipótese era que o pré-processamento
**dominasse** o custo.

### R4a — honestidade de escopo *(obrigatório agora)*

O protocolo gravado nos JSONs passou a declarar:

- `escopo`: "somente a predição do modelo (`predict_proba` / `decision_function`)
  a partir do vetor de features **já extraído** — não inclui carregamento do
  áudio, VAD, nem extração de features";
- `incluir_extracao_features: false`;
- ponteiro para onde o pipeline completo é medido.

Isso era obrigatório **antes** de escrever o cronômetro da CNN: sem declarar o
escopo, os três modelos acabariam medidos por protocolos diferentes.

### R4b — o custo de featurização, medido

`scripts/tempo_pipeline_completo.py` cronometra, com o mesmo protocolo do
config, as etapas separadas — `carregar áudio` → `VAD + padding` →
`extrair_vetor` → `predict` — sobre uma amostra fixa de 200 áudios da
**validação** (semente 42). As etapas **não são reimplementadas**: são as funções
reais de `src/data/preprocessamento.py` e `src/features/extrair_features.py`,
encadeadas na mesma ordem do pipeline. Há ainda uma **guarda de fidelidade**: o
vetor recalculado é comparado feature a feature contra o `features.csv`
congelado, e o script aborta se divergir — porque então o que se estaria
cronometrando não é o caminho que gerou os dados do trabalho.

**Resultado — e a hipótese do `config.yaml` NÃO se confirmou.** Por áudio
(batch = 1): carregar 0,639 ms + VAD/padding 0,371 ms + features 3,705 ms, mais
a predição — **RF 6,405 ms** (total 11,12 ms) contra **SVM 0,613 ms** (total
5,33 ms). A predição é **57,6%** do custo real no RF: longe de irrelevante. E a
leitura inverte a intuição da tabela de throughput: **por áudio, o pipeline do
SVM é ~2,1× mais barato que o do RF**, pelo mesmo mecanismo da linha de
*latência* (300 árvores custam caro por chamada unitária; 7.605 vetores de
suporte, não). Em lote o RF continua ~19× melhor. Conclusão para o texto: **não
existe "o modelo mais barato" sem dizer o regime** — e a featurização (4,7
ms/áudio) é custo comum aos dois, que não desempata nada.

*Nota de método que a própria medição exigiu:* a base compartilhada é medida
**uma vez** e somada à predição de cada modelo. Medi-la dentro do laço de cada
modelo dava 5,51 ms/áudio no laço do RF contra 3,79 ms no do SVM — para código
idêntico, e de forma reprodutível entre execuções: a predição do RF despeja o
cache da CPU e contamina a etapa vizinha.

**Amarra para o Bloco 4:** `medir_pipeline` recebe uma **lista de etapas
nomeadas**. A CNN entra acrescentando `gerar mel-espectrograma` e
`forward da CNN` — mesmo código, mesmo protocolo, como manda a docstring de
`src/models/tempo.py`. **Não reescreva o cronômetro para a CNN: acrescente
etapas.**

---

## R5 — Correções pontuais

### R5.1 — Docstring do EER contradizia a limitação declarada

**Observação.** O README já declara que o split interno torna os números **não
comparáveis** com Yamagishi et al. — mas a docstring de `calcular_eer` dizia o
oposto: *"é o que permite comparar esse número com o da literatura (… 1,32% em
LA)"*. Está no código, que a banca pode abrir.

**Decisão.** Reescrita: o EER é a métrica **padrão** da literatura ASVspoof, o
que torna a **grandeza** comparável; os **valores** deste trabalho **não** são
diretamente comparáveis, porque aqui o protocolo é split interno aleatório por
utterance (mesmos ataques, codecs e locutores em treino e validação) e o oficial
é deliberadamente cross-attack. A docstring aponta para a seção "Limitação
declarada do split" do README e para o bloco `split:` do config.

### R5.2 — `avaliar()` sombreava `limiar`

`eer, limiar = calcular_eer(...)` reatribuía o parâmetro depois de `m['limiar']`
já ter sido gravado. Não gerava bug; era armadilha para quem editasse depois.
Renomeado para `limiar_no_eer`. **O JSON não muda** (as chaves continuam `eer` e
`limiar_eer`), e `scripts/teste_reproducao_limiar.py` continua passando.

### R5.3 — README liderava com o estatístico errado

**Observação.** O parágrafo de estabilidade liderava com *"a diferença é maior
que qualquer dispersão medida"* — exatamente a heurística que o próprio
`estabilidade_rf_svm.json` marca como estatisticamente errada em
`leitura_critica.nota_metodo`: comparar contra a maior dispersão **individual**
ignora a correlação entre os erros dos dois modelos nas mesmas 22.226 amostras;
o teste correto é o IC da **diferença pareada**.

**Decisão.** O parágrafo passou a liderar com o **bootstrap pareado**:
Δf1_macro = **+0,0762**, IC95 [0,0663; 0,0856]; ΔEER = **−0,0466**,
IC95 [−0,0560; −0,0377]; nenhum IC contém zero; SVM melhor em **100%** das 1.000
reamostragens nas duas métricas. A comparação contra a dispersão individual ficou
**depois**, rotulada como heurística complementar — exatamente como o JSON a
rotula.

---

## R6 — Análises complementares

### R6.1 — Extrapolação da curva de aprendizado

Ajustando `f1_macro ~ a·ln(n) + b` por mínimos quadrados sobre os 7 pontos
medidos, e resolvendo para o f1_macro do SVM (lido de
`svm_tuned_principal.json`, **nunca** hardcodado), obtém-se o n que o RF
precisaria para alcançar o SVM. Coeficientes, R², `n_necessario` e as razões
sobre o treino completo (103.723) e sobre o universo eval inteiro (148.176) estão
em `results/metricas/extrapolacao_curva_rf.json`.

**Resultado:** a = 0,0312, b = 0,4079, **R² = 0,9934**; n* ≈ **276.116** áudios
— **2,66× o treino completo** (103.723) e **1,86× o universo eval inteiro**
(148.176). Isto é: **dentro dos dados disponíveis, o RF não alcança o SVM nem
usando tudo.** Converte "o RF não saturou" de fraqueza da análise em **achado
quantificado**.

**Ressalva obrigatória, gravada no campo `limitacao` do próprio JSON:** é uma
**extrapolação log-linear fora da faixa medida**, não uma medição. Curvas de
aprendizado costumam achatar, então a estimativa é um **limite otimista para o
RF** — na prática ele precisaria provavelmente de *mais* que isso, não menos. No
texto do TC II o número entra como ordem de grandeza estimada, jamais como
medida.

### R6.2 — Variância entre subamostras *(a frente que faltava)*

**Observação.** A variância entre sementes do RF é ±0,0004 em f1 — praticamente
zero. Isso porque a fonte de incerteza que domina o braço principal **não é a
semente das árvores**: é **qual subamostra de 30k caiu**, e isso nunca tinha sido
variado para modelo nenhum. A resposta ao pedido de "3 sementes no SVM" ("o SVC é
determinístico") está tecnicamente certa, mas deixava a pergunta de fundo do
orientador sem resposta.

**Decisão.** `scripts/estabilidade_subamostra.py` gera **3 subamostras
alternativas** (sementes 43, 44, 45) pela **mesma função** de
`scripts/gerar_subamostra.py` — a regra foi extraída para `montar_subamostra`,
para que "variar só a semente" seja literalmente verdade —, treina RF e SVM em
cada uma com os **mesmos hiperparâmetros vencedores** (sem nova busca: um fator
por vez) e seleciona o limiar na validação pelo protocolo normal. A subamostra
oficial de semente 42 entra como quarto ponto. Nada sobrescreve
`data/processed/subamostra_30k.csv`: as alternativas ficam em
`data/processed/subamostras_estabilidade/`.

**Guarda que legitima a comparação:** antes de gerar qualquer alternativa, o
script re-gera a subamostra com semente 42 pela função compartilhada e confere o
**MD5** contra o artefato congelado. Se não bater, aborta.

**Leitura crítica gravada no JSON:** compara, na mesma unidade (f1_macro), o
desvio **entre subamostras** (medido aqui), o desvio **entre sementes de treino**
(±0,0004, `estabilidade_rf_svm.json`) e a **distância RF × SVM** (Δf1 ≈ 0,0762,
bootstrap pareado). Se a distância RF × SVM ficasse **menor** que a dispersão entre
subamostras, isso **teria de ser dito no texto** — é a regra do item 6 do
protocolo do orientador. **Não é o caso.**

| fonte de variação | desvio em f1_macro |
|---|---:|
| semente das árvores (RF, 5 sementes) | 0,0004 |
| **qual subamostra de 30k caiu** (4 subamostras) | **RF 0,0023 · SVM 0,0017** |
| distância RF × SVM (bootstrap pareado) | **0,0762** |

Duas leituras: (a) a dispersão entre subamostras é **5,8×** a dispersão entre
sementes — *a semente das árvores era a fonte de variação errada de se olhar*;
(b) a distância RF × SVM é **33×** a maior dispersão entre subamostras — *a
conclusão do Bloco 3 sai mais forte, não mais fraca*.

O ponto de semente 42 é a subamostra oficial e reproduz **exatamente**
`rf_tuned_principal.json` (0,7225 / 0,1930 / limiar 0,6516) e
`svm_tuned_principal.json` (0,7987 / 0,1462 / limiar −0,0329) — é uma âncora
verificável, não "mais um ponto". Chegar a isso exigiu duas correções que a
guarda do próprio script pegou: a tabela que alimenta `montar_subamostra` tem de
vir de `split.csv` (o `sample` sorteia **por posição**, então a ordem das linhas
muda o sorteio — pelo features.csv, a "semente 42" batia em só 8.611 dos 30.000
IDs oficiais), e a seleção do X tem de preservar a ordem da tabela de features,
como `filtrar_treino_braco` faz.

### R6.3 — `permutation_importance` no RF ajustado

**Observação.** O próprio código já alertava (`treinar_rf.py:181-184`) que a
importância por **redução de impureza** é enviesada a favor de features contínuas
de alta cardinalidade — é "uma pista sobre o que o modelo usou", não prova de
causalidade acústica. E o `top10_features` que está nos JSONs e vai para o texto
é justamente a métrica enviesada. `ablacao_mfcc1_std.json →
leituras_pre_registradas.c` já previa o uso de `permutation_importance` como
confirmação.

**Decisão.** `scripts/importancia_permutacao_rf.py` roda `permutation_importance`
sobre `rf_tuned_principal.joblib`, **na validação**, `n_repeats=10`, semente 42,
scoring por **AUC** (independente de limiar).

**Resultado: os dois rankings CONCORDAM** — Spearman ρ = **0,8516**
(p = 2,4×10⁻¹³), **9 das 10** features do topo em comum, com `mfcc1_media` e
`mfcc1_std` em 1º e 2º lugar nos dois. Todas as 44 features têm importância
significativa (média − desvio > 0). O `top10_features` publicado nos JSONs, que é
a métrica enviesada, fica **confirmado** por uma que não tem esse viés — a
leitura acústica do texto pode ser mantida, ainda como "o que o modelo usou" e
não causalidade.

---

## As três leituras positivas *(resultado, não pendência — entram no Capítulo 4)*

### 1. A exigência do orientador funcionou, e há prova numérica

A `NOTA_LIMIAR.md` diagnosticou que as folhas puras neutralizavam o
`class_weight` e que o `predict_proba` tinha **~79 valores distintos**. Agora,
com `min_samples_leaf=5`, o RF tem **22.207 scores distintos em 22.226
amostras** (`rf_tuned_principal.json → selecao_limiar.n_candidatos`). O score
deixou de ser degrau e virou contínuo. **Esse é o número que prova, em uma linha,
que a exigência do orientador não foi cosmética** — ele tem de estar no texto.

### 2. A inversão de custo latência × throughput

Já está bem escrita em `NOTA_RF_VS_SVM.md` §1 (o SVM é ~9× melhor em latência e
~19× pior em lote; o mecanismo são os 7.605 vetores de suporte = 25,4% do
treino). Depois do R4, ela ganha uma **ressalva**: enquanto a extração de
features não entrava na conta, a comparação de custo cobria **só a etapa de
predição**. Agora o pipeline completo está medido, e a §1 remete a ele.

### 3. Contexto de literatura, com a ressalva certa

Um EER de 14,6% com features manuais fica **na faixa dos baselines oficiais GMM
do ASVspoof 2021 LA**. Isso permite dizer que o resultado é **plausível e não é
ruído** — sem afirmar equivalência de protocolo, porque o README já declara que o
split interno torna os números não comparáveis com Yamagishi et al. Escrever com
exatamente essa cautela: "na faixa dos baselines", nunca "equivalente a".

---

## O que continua pendente

**Tópico 2 — definição do espectrograma da CNN.** Levado ao orientador,
**sem decisão**. As três perguntas em aberto (largura 251 × 256; a assimetria de
mascaramento; a normalização) estão registradas em
**`results/metricas/DECISOES_PENDENTES_CNN.md`**, com evidência, opções, custo e
recomendação — para que o orientador responda sobre um documento, e não sobre
memória.

Até haver resposta: **não alterar** o bloco `espectrograma:` do
`config/config.yaml`, **não criar** script de geração de espectrograma, **não**
começar o B4.1.
