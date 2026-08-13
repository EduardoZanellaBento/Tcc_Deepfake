# Nota — divergência de f1_macro entre `rf_baseline.json` e `diagnostico_limiar_varredura.csv`

**Números que não fechavam:**

| Fonte | f1_macro no "limiar 0,50" | recall_bonafide |
|---|---|---|
| `rf_baseline.json` (via `modelo.predict()`) | 0,5675 | 0,1012 |
| `diagnostico_limiar_varredura.csv` (via `scores >= 0,50`) | 0,5598 | 0,0921 |

## Causa raiz: tratamento de empates no score 0,50 exato

Não é bug de dados nem de split. Verificação empírica (modelo `models/rf_baseline.joblib`
carregado do disco, avaliado no mesmo conjunto de validação de 27.235 amostras,
`split.csv` com hash MD5 `58cea82b2513c0f5c1e5797895a92571` — o **mesmo** hash registrado
em `curva_aprendizado_rf.json`, ou seja, **não houve** regeneração de split entre as
duas execuções):

- As duas regras de decisão divergem em **exatamente 25 amostras**, todas com
  `predict_proba` **exatamente igual a 0,50** — empate de 50 das 100 árvores.
  (Com `max_depth=None` as folhas são praticamente puras: cada árvore vota 0 ou 1
  e a probabilidade agregada é um múltiplo de ~1/100, o que torna o empate exato
  em 0,50 um evento comum, não uma coincidência de ponto flutuante. Na validação
  há só 79 valores distintos de score.)
- `modelo.predict()` decide por `argmax` das probabilidades; em empate, o `argmax`
  do NumPy devolve o **primeiro** índice — classe 0, **bonafide**.
- A varredura de limiar usa `scores >= 0,50`; o empate satisfaz o `>=` — classe 1,
  **spoof**.
- Detalhe que explica o tamanho do efeito no f1_macro: as 25 amostras empatadas são
  **todas bonafide reais**. Como o recall bonafide do baseline é baixíssimo
  (~280 acertos), mover 25 acertos tem efeito visível no f1 da classe minoritária:
  recall bonafide 0,1012 → 0,0921, f1_macro 0,5675 → 0,5598.

Reprodução: `predict()` sobre o joblib devolve f1_macro 0,5676 (≈ 0,5675 do JSON,
que foi gerado num modelo retreinado com a mesma seed) e `scores >= 0,50` devolve
0,5598 — idêntico ao CSV da varredura. As duas fontes estão **corretas**; medem
regras de decisão diferentes no mesmo modelo.

## Classificação: diferença legítima de protocolo (não é bug), mas exige regra

Nenhum dos dois números está "maquiado" ou errado. O que a divergência expõe é que
**"limiar 0,50" é ambíguo** quando o score tem granularidade grossa: `> 0,50`,
`>= 0,50` e `argmax` são três regras diferentes no empate.

**Regra adotada a partir desta nota (protocolo):** toda comparação que envolva limiar
explícito usa `scores >= limiar` (convenção da varredura, determinística e a mesma
para qualquer limiar); o `predict()` do sklearn fica restrito ao baseline histórico já
publicado. Quando a seleção de limiar na validação entrar no protocolo (ver
`NOTA_LIMIAR.md`), essa convenção vale para RF, SVM e CNN igualmente — e o ponto fica
irrelevante na prática, porque o limiar operacional estará longe de 0,50 e, com
`min_samples_leaf` no espaço de busca, as folhas deixam de ser puras e os empates
exatos praticamente desaparecem.
