# Nota de leitura — RF × SVM no braço principal (Bloco 3)

> Todos os números desta nota vêm do **conjunto de validação** (22.226 áudios) e
> dos artefatos congelados do lote único (`features.csv` MD5 `51b2f439…`). O
> **teste segue lacrado** até o Bloco 5. Fontes: `rf_tuned_principal.json`,
> `rf_tuned_referencia.json`, `svm_tuned_principal.json`,
> `estabilidade_rf_svm.json`, `ablacao_mfcc1_std.json`,
> `curva_aprendizado_rf_tuned_eval.json`, `extrapolacao_curva_rf.json`,
> `estabilidade_subamostra.json`, `tempo_pipeline_completo.json`.

Os três JSONs de resultado registram os números; o que não estava registrado em
lugar nenhum era a **leitura** deles. As três seções abaixo são exatamente os
três pontos que a banca vai perguntar: por que os custos se invertem entre
latência e lote, por que o SVM vence, e quanto vale um f1 escolhido entre 22 mil
limiares no próprio conjunto em que é reportado.

## 1. A inversão de custo: dois regimes, nenhum vencedor absoluto

Os tempos, medidos pelo mesmo protocolo (`config.yaml → tempo`: 10 repetições,
3 de aquecimento descartadas, `n_jobs=1` na inferência):

| modelo | latência (batch = 1) | throughput | áudios/s |
|---|---:|---:|---:|
| RF ajustado (30k, 300 árvores) | 5,215 ms | 0,0189 ms/áudio | 53.035 |
| SVM RBF (30k) | **0,557 ms** | 0,358 ms/áudio | 2.793 |

O SVM é **~9× melhor em latência** e **~19× pior em lote**. Parece
contraditório e não é — os dois custos têm naturezas diferentes, e o mecanismo
está no próprio JSON do SVM: `n_vetores_suporte` = 2.177 + 5.428 = **7.605
vetores de suporte — 25,4% do treino de 30.000**. Cada predição do SVM avalia o
kernel RBF contra esses 7.605 vetores; é um custo **por amostra**, proporcional
ao número de vetores de suporte, que não amortiza em lote nenhum. O RF, ao
contrário, paga um custo fixo alto por *chamada* (overhead de Python/NumPy para
percorrer 300 árvores), e esse custo se dilui entre as 22.226 amostras do lote.

Traduzindo para a pergunta de pesquisa (custo computacional): o SVM é melhor
para **detectar um áudio por vez** — o cenário de uso real de um detector em
produção: chega um áudio, decide; o RF é melhor para **varrer um acervo**. Não
há vencedor absoluto — há dois regimes, e o trabalho mede os dois justamente
porque `config.yaml → tempo` exige latência e throughput separados.

**Previsão (marcada como previsão, não como medida):** o número de vetores de
suporte cresce aproximadamente com n, então um SVM treinado nos 103.723 do
braço de referência teria custo de inferência proporcionalmente maior —
enquanto o custo de inferência do RF **não depende de n**, só do número e da
profundidade das árvores. Isso não foi medido: o treino do SVM-RBF é
O(n²)–O(n³) e não roda em 103k, que é a razão de existir a subamostra. É
exatamente por isso que entra aqui como previsão fundamentada, não como
resultado.

> **Ressalva de escopo (03/09/2026, tarefa R4):** os tempos desta tabela cobrem
> **somente a etapa de predição**, a partir do vetor de 44 features **já
> extraído**. Carregar o `.flac`, VAD, padding e extração de features ficam de
> fora — e isso agora está declarado no próprio artefato
> (`tempos_inferencia.protocolo.escopo` nos JSONs de RF e SVM).

### 1.1 O pipeline completo, medido — e a hipótese do `config.yaml` que **não** se confirmou

O comentário do `config.yaml` (bloco `tempo`, chave `incluir_extracao_features`)
previa que o pré-processamento **dominaria** o custo, tornando a escolha do
classificador quase irrelevante. Medido ponta a ponta, por áudio (batch = 1),
sobre 200 áudios fixos da validação, com as **mesmas funções** do pipeline real
(`tempo_pipeline_completo.json`):

| etapa | ms/áudio | % no RF | % no SVM |
|---|---:|---:|---:|
| carregar áudio | 0,639 | 5,7% | 12,0% |
| VAD + padding | 0,371 | 3,3% | 7,0% |
| extrair 44 features | 3,705 | 33,3% | 69,5% |
| **predizer** | **RF 6,405 / SVM 0,613** | **57,6%** | **11,5%** |
| **TOTAL** | **RF 11,12 / SVM 5,33** | | |

A hipótese **não se confirma**: a predição responde por **57,6%** do custo real
no RF — longe de desprezível. E a leitura inverte a intuição da tabela anterior:
**por áudio, o pipeline do SVM é ~2,1× mais barato que o do RF**, porque as 300
árvores custam 6,4 ms por chamada unitária enquanto os 7.605 vetores de suporte
custam 0,6 ms. Isso é o mesmo fenômeno da linha de **latência** da tabela acima,
agora somado ao custo que faltava — e não contradiz o throughput: em lote o RF
continua ~19× melhor, porque lá o overhead por chamada se dilui.

Conclusão para o texto: **não existe "o modelo mais barato" sem dizer o regime.**
Um áudio por vez (o cenário de uso real de um detector) → SVM. Varredura de
acervo → RF. A featurização (4,7 ms/áudio) é um custo **comum aos dois** e não
desempata nada.

> **Nota de método:** a base compartilhada é medida **uma vez** e somada à
> predição de cada modelo. Medi-la dentro do laço de cada modelo dava 5,51
> ms/áudio no laço do RF contra 3,79 ms no do SVM — para código idêntico: a
> predição do RF despeja o cache da CPU e contamina a etapa vizinha. Como os
> dois consomem o mesmo vetor, a base é a mesma por construção.

O Bloco 4 medirá a CNN **com o mesmo protocolo** (`src/models/tempo.py`, mesmas
repetições, mesmo descarte de aquecimento, `n_jobs` declarado; e o pipeline
completo pelo mesmo `medir_pipeline`, acrescentando as etapas "gerar
mel-espectrograma" e "forward da CNN") — é isso que torna a comparação final
defensável.

## 2. Por que o SVM vence — e por que isso é a discussão do trabalho

Os números (validação):

| | f1_macro | EER | AUC |
|---|---:|---:|---:|
| SVM RBF (30k) | **0,7987** | **0,1462** | **0,9289** |
| RF ajustado (30k) | 0,7225 | 0,1930 | 0,8873 |
| RF ajustado (103.723, braço de referência) | 0,7723 | 0,1579 | 0,9191 |

O SVM bate o RF no braço principal **e** bate o RF treinado no treino completo
do braço de referência — com 3,5× menos dados. E a vantagem não é ruído: o
**bootstrap pareado** da diferença (`estabilidade_rf_svm.json →
bootstrap_pareado_svm_menos_rf`; mesmas 22.226 linhas reamostradas para os dois
modelos, 1.000 vezes) dá Δf1_macro = +0,0762 com IC95 [0,0663; 0,0856] e
ΔEER = −0,0466 com IC95 [−0,0560; −0,0377] — nenhum IC contém zero, e o SVM é
melhor em **100% das reamostragens**, nas duas métricas.

**Hipótese explicativa (registrada como hipótese):** as 44 features são
contínuas, densas e padronizadas, e a fronteira entre bonafide e spoof nesse
espaço é provavelmente **suave** — terreno natural do kernel RBF, que constrói
fronteiras curvas a partir de poucos exemplos. O Random Forest aproxima a mesma
fronteira com partições paralelas aos eixos e precisa de muito mais dado para
chegar perto. Isso casa com evidência que já está no repositório: a curva de
aprendizado do **RF ajustado** (`curva_aprendizado_rf_tuned_eval.json`, features
congeladas, limiar selecionado na validação em cada ponto) vai de **0,6764**
(5k) a **0,7723** (103.723) e **não satura** — o último passo (80.000 →
103.723) ainda rende **+0,0121** em f1_macro, 2,4× a tolerância de 0,005, e por
isso o campo `satura_em_n` não é emitido. O RF ainda está com fome de dados no
ponto em que o SVM já extraiu o que precisava.

E dá para dizer **quanto** de fome. Ajustando `f1_macro ~ a·ln(n) + b` sobre os
sete pontos medidos (`extrapolacao_curva_rf.json`): a = 0,0312, b = 0,4079,
**R² = 0,9934**. Resolvendo para o f1_macro do SVM (0,7987), o RF precisaria de
cerca de **276.116 áudios de treino** — **2,66× o treino completo** (103.723) e
**1,86× o universo eval inteiro** (148.176), que é tudo o que existe neste
conjunto. Ou seja: **dentro dos dados disponíveis, o RF não alcança o SVM nem
usando tudo.** A vantagem do SVM não é artefato do tamanho do treino do braço
principal.

> **Ressalva que tem de acompanhar o número.** É uma **extrapolação log-linear
> fora da faixa medida**, não uma medição. Curvas de aprendizado costumam
> achatar, então a forma log-linear superestima o ganho por dado adicional na
> cauda: 276 mil é um **limite otimista para o RF** — na prática ele precisaria
> de *mais*, não menos. O R² alto atesta o ajuste **dentro** da faixa medida;
> não valida a extrapolação fora dela. No texto: ordem de grandeza estimada,
> jamais medida.

A curva **anterior** (`curva_aprendizado_rf_eval.json`, RF *baseline* sobre as
features pré-mascaramento, decisão por argmax: f1_macro 0,476 → 0,557) continua
citável como referência "antes" — mas o argumento passa a ser feito com a curva
certa. (A ablação da `mfcc1_std`,
`ablacao_mfcc1_std.json`, descarta a explicação alternativa de que a vantagem
viria de uma feature contaminada: removê-la custa ΔEER +0,0050 — bootstrap
pareado, IC95 [+0,0006; +0,0111], detectável mas equivalente a apenas 10,7% da
distância RF × SVM.)

E aqui está a tensão que **é** o coração do TC II, não um detalhe: o SVM ganha
em 30k, mas o treino O(n²)–O(n³) o impede de ser levado aos 103k; o RF escala,
mas nesse regime ainda perde (0,7723 do braço de referência < 0,7987 do SVM em
30k). **Um modelo é melhor onde cabe; o outro cabe onde o primeiro não entra.**
A comparação justa entre os dois só existe dentro do envelope computacional em
que ambos rodam — que é o que o braço principal delimita — e essa restrição
mútua é análise crítica, não limitação envergonhada: é o resultado que a
pergunta sobre custo computacional foi feita para produzir.

## 3. O viés de seleção do limiar — e por que ele é desprezível aqui

O limiar do RF foi escolhido entre **22.207 candidatos** (`np.unique(scores)`)
no **mesmo conjunto de validação** em que o f1_macro é reportado (o do SVM,
entre 22.226). Por construção, isso enviesa a métrica da validação para cima:
escolher o máximo sobre 22 mil candidatos captura um pouco de sorte da amostra.

A defesa já está medida, em `estabilidade_rf_svm.json`: entre as 5 sementes do
RF o limiar variou de **0,6463 a 0,6681** (±1,7%) enquanto o f1_macro variou
**0,0011** (0,7214–0,7225). Ou seja, **a superfície do f1_macro é plana em
volta do ótimo** — errar o limiar em alguns centésimos quase não muda o
resultado, logo o sobreajuste do limiar à validação é pequeno.

Frase de protocolo que fecha a seção: o número honesto de generalização
continua sendo o do **teste lacrado** (Bloco 5), onde o limiar entra
**congelado**, sem reajuste. É exatamente para absorver esse desgaste que a
validação existe.

### 3.1 A fonte de incerteza que faltava: **qual subamostra caiu**

A estabilidade acima varia a *semente das árvores* (±0,0004) e a *amostra de
validação* (bootstrap). Faltava variar o que o braço principal realmente tem de
arbitrário: **qual subamostra de 30k caiu** — nunca variada, para modelo nenhum.
`scripts/estabilidade_subamostra.py` gera três alternativas (sementes 43, 44,
45) pela **mesma** função de estratificação, treina RF e SVM em cada uma com os
**mesmos** hiperparâmetros, e inclui a subamostra oficial como quarto ponto (uma
âncora verificável: ela reproduz exatamente `rf_tuned_principal.json` e
`svm_tuned_principal.json`).

| fonte de variação | desvio em f1_macro |
|---|---:|
| semente das árvores (RF, 5 sementes) | 0,0004 |
| **qual subamostra de 30k caiu** (4 subamostras) | **RF 0,0023 · SVM 0,0017** |
| distância RF × SVM (bootstrap pareado) | **0,0762** |

Duas leituras, as duas citáveis:

1. **a semente das árvores era a fonte de variação errada de se olhar.** A
   dispersão entre subamostras é **5,8×** a dispersão entre sementes. A resposta
   dada ao pedido de "3 sementes no SVM" ("o SVC é determinístico") continua
   correta — mas respondia a outra pergunta;
2. **a conclusão do Bloco 3 sai mais forte.** Mesmo a maior dispersão entre
   subamostras (0,0023) é **33× menor** que a distância RF × SVM (0,0762): a
   vantagem do SVM não é acidente de qual subamostra caiu. Pela regra do item 6
   do protocolo do orientador, o caso que exigiria ressalva no texto seria o
   inverso — e não é o que se observa.
