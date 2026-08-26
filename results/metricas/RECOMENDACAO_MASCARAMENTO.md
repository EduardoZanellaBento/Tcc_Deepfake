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

---

## Adendo (26/08) — correção do MECANISMO alegado

A decisão de mascarar **se mantém**; o que precisa ser corrigido é a explicação de
*por que* a distorção é assimétrica. O texto acima (seção "Contexto") afirma:
"como bonafide perde mais sinal no VAD (prop_fala média 0,63 vs 0,85 do spoof), a
fração de padding é maior no bonafide". Duas medições independentes **não sustentam
essa cadeia causal** — a checagem do bloco 1 (`checagem_mascaramento.json`, 200
áudios balanceados) e o próprio `piloto_padding.csv` (2.000 áudios):

Medição sobre os 2.000 áudios do piloto (200 bonafide, 1.800 spoof):

| medida | bonafide | spoof | leitura |
|---|---:|---:|---|
| `prop_fala` (média ± dp) | 0,621 ± 0,125 | 0,842 ± 0,150 | diferença grande (~1,6 dp) |
| `n_frames_validos` (média ± dp) | 134,5 ± 42,3 | 136,0 ± 54,2 | médias praticamente iguais |
| fração de padding (média ± dp) | 0,464 ± 0,168 | 0,458 ± 0,216 | diferença de 0,6 p.p. (~0,03 dp) |

A fração de padding é, **em média, igual** entre as classes, apesar de `prop_fala`
diferir em mais de um desvio-padrão. A inferência "prop_fala menor ⇒ mais padding"
era um salto: `prop_fala` é a fração do áudio **original** que o VAD manteve,
enquanto o padding depende do comprimento **absoluto** do que sobrou comparado ao
alvo de 4,0 s.

**Por que os dois efeitos se cancelam neste dataset.** Estimando a duração da fala
por `n_frames_validos × hop / sr`: bonafide ≈ 2,15 s e spoof ≈ 2,18 s de fala
efetiva. Dividindo por `prop_fala`, as durações originais ficam em ≈ 3,47 s
(bonafide) e ≈ 2,59 s (spoof) — os áudios bonafide são cerca de **34% mais longos**,
e é exatamente isso que compensa o VAD mais agressivo neles. (Estimativa
aproximada: `n_frames_validos` é truncado em 250 frames, o que subestima os áudios
que passam de 4,0 s.) O cancelamento é uma característica **deste** conjunto, não
uma lei — mais uma razão para medir em vez de inferir.

**O que continua verdadeiro, e agora com o mecanismo certo:**

1. **Validade de medida** (argumento principal, independente de classe) — a mediana
   é de 121 frames válidos em 251: **metade do tensor é padding**. Incluí-lo desloca
   as features em 30% na mediana e até 47% no máximo, em **44 de 44** features. Uma
   feature assim descreve, em parte, a formatação do vetor, não a fala.
2. **A distorção É assimétrica entre classes**, apesar da fração de padding igual em
   média — assimetria padronizada de −0,47 (`mfcc1_std`), +0,44 (`centroide_std`),
   −0,43 (`mfcc5_media`). Duas vias plausíveis, ambas coerentes com os dados e não
   mutuamente exclusivas:
   - **conteúdo acústico** — misturar zeros a espectros diferentes não desloca as
     médias na mesma proporção;
   - **dispersão do padding** — as médias coincidem, mas as distribuições não: o
     desvio-padrão da fração de padding é 0,216 no spoof contra 0,168 no bonafide
     (~29% maior). Distribuições com a mesma média e formatos diferentes produzem
     distorções diferentes, sobretudo nas features `_std` — que são justamente as
     que lideram a lista de assimetria (`mfcc1_std`, `centroide_std`, `mfcc8_std`).
   Teste que separaria as duas vias (opcional, sobre dados já existentes):
   correlacionar, DENTRO de cada classe, o delta de cada feature com a fração de
   padding do áudio. Correlação alta e inclinações iguais nas duas classes apontam
   para a segunda via; inclinações diferentes, para a primeira.
3. O diagnóstico 4.2 (RF pareado por faixa de `prop_fala`) permanece válido e
   continua indicando que o atalho não sustenta o desempenho do modelo.

**Consequência para o texto do TC II:** ao justificar o mascaramento, usar (1) como
argumento central — ele é de construção, não de resultado — e (2) como argumento
secundário, com a explicação correta. **Não** afirmar que o bonafide recebe mais
padding: o dado próprio do trabalho diz o contrário, e essa é uma pergunta de banca
fácil de fazer e constrangedora de errar.