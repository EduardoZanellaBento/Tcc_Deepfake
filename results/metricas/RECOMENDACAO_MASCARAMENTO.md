# Recomendação — mascaramento de padding na agregação temporal

**Contexto.** O pipeline é `VAD → padronizar para 4,0 s com zero-padding`. As 44
features são média/desvio-padrão de séries (MFCC, ZCR, centróide) calculadas sobre os
4,0 s inteiros — frames de silêncio puro incluídos. Como bonafide perde mais sinal no
VAD (prop_fala média 0,63 vs 0,85 do spoof), a fração de padding é maior no bonafide,
e a suspeita era que o "atalho de silêncio" estivesse codificado dentro das features.
Três diagnósticos foram rodados no universo `eval` (seed 42 em todos).

## Evidência

**1. Correlação features × prop_fala (148.176 áudios — 4.1).** Nenhuma das 44
features passa de |r| = 0,3 (máximo: mfcc19_media, r = −0,26; a mais importante do
RF, mfcc4_std, fica em +0,22). A contaminação existe, mas não domina nenhuma feature
isoladamente.

**2. RF pareado por faixa de prop_fala (4.2).** Com bonafide e spoof pareados em 10
faixas de prop_fala (atalho neutralizado por construção): o RF treinado no pareado
atinge **EER 0,1631 — melhor que o baseline (0,1818)**; o baseline avaliado no
pareado degrada pouco (EER 0,2102, Δ = +0,028). Leitura: **o sinal acústico é
genuíno**; o atalho contribui marginalmente, não sustenta o desempenho.

**3. Piloto de mascaramento (2.000 áudios, estratificados por classe × codec — 4.3).**
Cada áudio teve as 44 features agregadas de dois modos: (A) padding incluído, como
hoje; (B) somente frames válidos. Resultados (`piloto_padding_delta_features.csv`,
`piloto_padding_rf_ab.json`):

- A distorção é **grande em magnitude**: ~46% de delta relativo médio nas features
  `_media` e 12–32% nas `_std`. As features atuais medem, em parte, a formatação do
  tensor, não a fala.
- A distorção é **assimétrica entre classes em parte das features**: destaque para
  `centroide_std` (assimetria padronizada +0,52) e `mfcc1_std` (−0,29) — é por essas
  vias que o confundidor entra no modelo.
- RF interno do piloto (70/30): **(B) mascarado levemente melhor** — f1_macro 0,5068
  vs 0,4905; EER 0,2657 vs 0,2815. Com n = 2.000 os números são instáveis;
  **indicativo**, não conclusivo.

## Recomendação: **MASCARAR** (incluir no lote único de re-extração)

Justificativa em três linhas:

1. **Validade de medida** — com mascaramento, cada feature volta a descrever
   exclusivamente o áudio real; o argumento "o modelo detecta síntese, não silêncio"
   passa a valer por construção, e a pergunta de banca correspondente morre.
2. **Não custa desempenho** — no piloto o mascaramento foi neutro-para-melhor
   (Δf1 +0,016, ΔEER −0,016); e o diagnóstico 4.2 já mostrou que o modelo não
   depende do atalho.
3. **Custo zero adicional** — é uma mudança de agregação dentro da extração que já
   vai rodar (junto com o `win_length` do centróide e o filtro `eval`).

Condições registradas para o lote (já na lista de pendências do plano):
`n_frames_validos` entra como coluna de diagnóstico ao lado de `prop_fala` e,
como ela, **fora do X** (lista `excluir` de `colunas_features`). Áudios em que o
VAD zera tudo (prop_fala = 0) mantêm a convenção atual (usa-se o áudio original) e
têm `n_frames_validos` = total.

**Decisão final é do orientador.** Nada foi re-extraído: o `features.csv` está
intocado (hash MD5 conferido no fim da rodada) e o piloto vive em arquivo separado
(`data/features/piloto_padding.csv`).
