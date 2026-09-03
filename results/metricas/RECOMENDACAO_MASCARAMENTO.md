# Recomendação — mascaramento de padding na agregação temporal

> ⚠️ Números medidos sobre o `features.csv` **pré-mascaramento**
> (MD5 `c01c3c5c…`). Servem como referência "antes"; para as features congeladas
> (lote único de 30/08), ver o **adendo de 30/08** ao fim deste documento e
> `padding_corr_features_propfala_pos_lote.csv`.

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

---

## Adendo 2 (pós-lote único de 30/08) — o mascaramento medido no CSV congelado, e a correção de uma afirmação

O lote único de re-extração foi executado em 30/08/2026 (`DOSSIE_LOTE_UNICO.md`);
este adendo recalcula a correlação das 44 features com `prop_fala` sobre o
`features.csv` **congelado** (MD5 `51b2f439…`, universo eval, 148.176) e a compara
com a rodada pré-mascaramento (arquivada em
`padding_corr_features_propfala_pre_lote.csv`; a rodada nova está em
`padding_corr_features_propfala_pos_lote.csv`). Como no adendo de 26/08, o texto
original acima **não foi reescrito** — ele é anotado.

| | features antigas (pré-masc.) | features congeladas |
|---|---:|---:|
| média de \|r\| | 0,150 | **0,101** |
| máximo de \|r\| | 0,260 (`mfcc19_media`) | **0,320 (`mfcc1_std`)** |
| features com \|r\| > 0,30 | nenhuma | **`mfcc1_std`** |

**1. A contaminação média caiu 33% — e a queda está toda nas features `_std`.**
`mfcc19_std` 0,230 → 0,044; `mfcc8_std` 0,156 → 0,003; `mfcc11_std` 0,160 → 0,022;
`mfcc15_std` 0,164 → 0,051; `mfcc16_std` 0,183 → 0,073; `mfcc20_std` 0,179 → 0,075.
É a confirmação quantitativa de que o mascaramento funcionou: os zeros do padding
inflavam o desvio-padrão **proporcionalmente à quantidade de padding**, então o
desvio carregava informação sobre silêncio. Removidos os zeros, essa via fecha.
Esta evidência é nova (não existia em nenhum diagnóstico anterior) e é favorável à
decisão de mascarar — usar no texto.

**2. CORREÇÃO de afirmação: o item 1 da seção "Evidência" ficou falso para as
features congeladas.** Ele diz: *"Nenhuma das 44 features passa de |r| = 0,3"*.
Isso era verdade para as features antigas e **não é mais**: `mfcc1_std` foi na
direção oposta de todas as outras, sozinha — **+0,032 → −0,320** — e é a única
acima de 0,30. Interpretação: `mfcc1` é o coeficiente ligado à **energia**; antes,
seu desvio era dominado pelos frames de padding (todos com energia nula e
idênticos), que **achatavam** a variação real e escondiam a relação com
`prop_fala`. Depois do mascaramento, `mfcc1_std` mede a dinâmica de energia da fala
que sobreviveu ao VAD — e um áudio do qual o VAD cortou muito é, por construção, um
áudio de energia irregular. A correlação residual é **acústica e genuína**, não o
atalho de formatação do tensor.

Evidência de apoio: contra `n_frames_validos` (a contagem direta de frames que
entraram na agregação — quantidade de silêncio em forma pura), o máximo é
\|r\| = **0,14** (também `mfcc1_std`), **menor** que contra `prop_fala`. Se o
resíduo fosse "quantidade de padding" disfarçada, seria o contrário.

Cruzamento com o modelo (diagnóstico re-rodado sobre o RF **ajustado** do Bloco 3,
`rf_tuned_principal.json`): `mfcc1_std` está no top10 de importância (2ª posição,
0,057). Ou seja, o modelo usa a feature com o maior resíduo — o que torna a
interpretação acústica acima obrigatória no texto, junto com a ressalva de que a
resposta causal definitiva (pareamento por `prop_fala` sobre as features
congeladas) fica como análise complementar.

**Consequência para o texto do TC II:** onde o trabalho citar o item 1 da
Evidência, citar junto esta revisão: para as features congeladas, a frase correta é
*"uma única feature (`mfcc1_std`, r = −0,32) passa de |r| = 0,3, com interpretação
acústica e não de formatação"*. A afirmação foi revista à luz de dado novo — o
rastro fica, como no adendo de 26/08.
### Fechamento com dado: a ablação da `mfcc1_std` (03/09)

A interpretação acústica acima era, até aqui, argumento. A ablação
(`results/metricas/ablacao_mfcc1_std.json`, `scripts/ablacao_mfcc1_std.py`) põe um
número no lugar: o mesmo RF ajustado, mesma semente, mesmo treino de 30k, mesma
validação, **43 features em vez de 44** — só a `mfcc1_std` fora.

| | com `mfcc1_std` (44) | sem `mfcc1_std` (43) | Δ |
|---|---:|---:|---:|
| f1_macro | 0,7225 | 0,7148 | **−0,0077** |
| EER | 0,1930 | 0,1980 | **+0,0050** |

Medido por **bootstrap pareado** (1.000 reamostragens, mesmos índices aplicados aos
dois modelos — os dois são avaliados nas mesmas 22.226 linhas e compartilham 43 das
44 features, logo os erros são correlacionados e o bootstrap não pareado
superestimaria a incerteza): IC95 do ΔEER = **[+0,0006; +0,0111]** e do Δf1_macro =
**[+0,0016; +0,0140]**. Nenhum contém zero — **a feature contribui de verdade**.

E, ao mesmo tempo, **a magnitude é desprezível**: +0,0050 de EER é **+2,6%
relativo** e apenas **10,7%** da distância entre RF e SVM (0,0468). As duas
afirmações convivem e as duas devem ser ditas: *estatisticamente detectável,
praticamente irrelevante*.

**O que isso encerra.** A ablação mede quanto o desempenho **depende** da feature —
ela não decide sozinha se a `mfcc1_std` é atalho ou acústica. Mas ela dá o **limite
superior do estrago**: ainda que a feature fosse atalho puro, removê-la custaria
0,005 de EER. O modelo **não se apoia** nela. Somado à evidência de que o resíduo
contra `n_frames_validos` (0,14) é menor que contra `prop_fala` (0,32), a leitura
acústica se sustenta e a pergunta de banca *"o seu modelo não está detectando
silêncio?"* passa a ter resposta numérica, não retórica.

**Ressalva de método, declarada:** a régua de leitura foi pré-registrada com duas
faixas ("piora menor que a dispersão" / "piora muito maior que a dispersão") e o
resultado caiu no vão entre elas — 1,2× o desvio não pareado, 1,8× o pareado. Foi
preciso acrescentar uma faixa intermediária **depois** de ver o número, o que é
exatamente aquilo que o pré-registro serve para evitar. A decisão fica registrada,
com os atenuantes, em `nota_sobre_o_pre_registro` dentro do JSON — e não apagada.
