# Nota técnica — limiar de decisão do RF baseline

Resposta à observação do orientador: *"F1 macro 0,5675 com AUC 0,9055 sugere que o
ranking do modelo pode estar razoável, mas o limiar de decisão, o desbalanceamento ou
a distribuição das classes está prejudicando a classificação final."*

**A observação está correta**, e a evidência já existia no repositório. Esta nota a
consolida.

## 1. A varredura de limiar (validação, universo 181k — `diagnostico_limiar_varredura.csv`)

| limiar | f1_macro | recall_bonafide | recall_spoof |
|---:|---:|---:|---:|
| 0,05–0,20 | 0,4732 | 0,000 | 1,000 |
| 0,30 | 0,4788 | 0,005 | 1,000 |
| 0,40 | 0,5028 | 0,030 | 1,000 |
| **0,50** | **0,5598** | **0,092** | 1,000 |
| 0,60 | 0,6320 | 0,184 | 0,999 |
| 0,70 | 0,7291 | 0,348 | 0,994 |
| **0,80** | **0,7730** | **0,577** | 0,958 |
| 0,85 | 0,7373 | 0,733 | 0,892 |
| 0,90 | 0,6406 | 0,883 | 0,749 |

O que a tabela mostra: **mesmo modelo, mesmos pesos, zero re-treino** — só mover o
limiar de 0,50 para 0,80 leva o f1_macro de 0,5598 para **0,7730** e o recall
bonafide de 0,092 para 0,577. O limiar do EER é 0,88. O ranking do modelo é
razoável (AUC-ROC ≈ 0,9055); a decisão binária em 0,50 é que está no lugar errado.
(Sobre a pequena diferença 0,5675 × 0,5598 no "mesmo" limiar 0,50, ver
`nota_divergencia_f1.md` — é o tratamento de empates no score exato 0,50.)

O padrão se repete no universo `eval` aprovado: o rebaseline
(`rf_baseline_eval.json`) tem f1_macro 0,5573 com recall bonafide 0,089 no
limiar padrão e limiar de EER 0,88.

## 2. A causa raiz: `class_weight` configurado, mas neutralizado pelas folhas puras

O RF usa `class_weight="balanced"` **e** `max_depth=None`. Com 44 features
contínuas e ~127k amostras (103k no eval), as árvores crescem até folhas **puras**.
O `predict_proba` de cada árvore é a fração ponderada de classes na folha — mas numa
folha pura essa fração é 1,0 para a classe presente, **independentemente do peso
aplicado**. Resultado: o `class_weight` influencia a escolha dos splits durante o
crescimento, mas quase nada a probabilidade final. Na prática, o desbalanceamento
foi *configurado* mas **não tratado** — e o limiar 0,50 corta uma distribuição de
scores que está inteira deslocada para o lado spoof (9:1).

Evidência adicional da mesma causa: os scores têm granularidade grossa (~79 valores
distintos na validação; cada árvore vota 0 ou 1 e a média é múltiplo de ~1/100),
consistente com folhas puras dominando o ensemble.

## 3. As três correções (para a fase de ajuste — semana 7 do cronograma, não agora)

1. **`min_samples_leaf` entre 5 e 20** — folhas deixam de ser puras, o
   `predict_proba` volta a ser uma fração informativa e o `class_weight` volta a
   valer na probabilidade final.
2. **`class_weight="balanced_subsample"`** — recalcula o peso dentro de cada
   bootstrap, tratando o desbalanceamento onde ele de fato age.
3. **Seleção de limiar na validação** — maximizando f1_macro ou usando o limiar do
   EER, reportada explicitamente como parte do protocolo (nunca escolhida no teste).

## 4. Consequência de protocolo (a mais importante)

A seleção de limiar **deixa de ser detalhe do RF e passa a ser regra do
experimento**: o mesmo critério (seleção na validação, teste lacrado) deve valer
para RF, SVM (`class_weight`) e CNN (peso na loss). Caso contrário, a comparação
central do trabalho — clássicos × CNN — fica enviesada por uma escolha arbitrária
de limiar que favorece quem tiver a distribuição de scores mais bem centrada.

Convenção de desempate registrada em `nota_divergencia_f1.md`: decisão por
`score >= limiar`, para os três modelos.

Os itens 1 e 3 estão registrados como TODO no `config.yaml` (seção `tuning`),
sem alteração de valores nesta rodada.
