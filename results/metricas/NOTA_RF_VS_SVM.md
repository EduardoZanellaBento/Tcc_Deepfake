# Nota de leitura — RF × SVM no braço principal (Bloco 3)

> Todos os números desta nota vêm do **conjunto de validação** (22.226 áudios) e
> dos artefatos congelados do lote único (`features.csv` MD5 `51b2f439…`). O
> **teste segue lacrado** até o Bloco 5. Fontes: `rf_tuned_principal.json`,
> `rf_tuned_referencia.json`, `svm_tuned_principal.json`,
> `estabilidade_rf_svm.json`, `ablacao_mfcc1_std.json`,
> `curva_aprendizado_rf_eval.json`.

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

O Bloco 4 medirá a CNN **com o mesmo protocolo** (`src/models/tempo.py`, mesmas
repetições, mesmo descarte de aquecimento, `n_jobs` declarado) — é isso que
torna a comparação final defensável.

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
aprendizado (`curva_aprendizado_rf_eval.json`, medida sobre o RF baseline) só
para de subir no treino completo (f1_macro 0,476 → 0,557 de 5k a 103.723, sem
platô antes do fim) — o RF ainda está com fome de dados no ponto em que o SVM
já extraiu o que precisava. (A ablação da `mfcc1_std`,
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
