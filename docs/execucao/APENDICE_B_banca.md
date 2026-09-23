# Apêndice B — Simulação de banca, com as respostas ancoradas no repositório

> **Como usar.** Não leia as respostas de primeira. Responda **em voz alta**, com o
> repositório fechado, e só depois confira. Toda pergunta cuja resposta você não
> consegue dar em 30 segundos é **um parágrafo que falta no texto** — anote e escreva.
>
> As perguntas seguem as frentes críticas determinadas nas instruções do projeto.
> As marcadas com ⚑ são as que o orientador nomeou explicitamente.

---

## 1. Pré-processamento

**Por que essa ordem — normalizar, depois VAD, depois fixar a duração?**
Normalizar antes do VAD faz a conversão para `int16` usar toda a faixa dinâmica, o
que dá decisões de fala/silêncio mais confiáveis em gravações baixas. E o VAD vem
antes de fixar a duração porque primeiro se remove o silêncio natural e só depois se
corta/preenche: na ordem inversa, o clipe voltaria a ter tamanho variável e o padding
de zeros ficaria picotado no meio do áudio.

**Por que normalização de pico e não RMS/LUFS?**
Simples, determinística e não altera a forma da onda, só a escala. RMS seria mais
perceptual, mas adiciona parâmetros sem ganho claro para a tarefa. E o objetivo é
removido o volume como atalho: sem normalizar, o modelo pode aprender o **volume** em
vez do artefato de síntese.

**Por que `webrtcvad` com agressividade 2?**
Meio-termo entre conservador (0, mantém mais) e agressivo (3, corta mais). O VAD
exige PCM `int16`, mono, sr em {8,16,32,48} kHz e frames de 10/20/30 ms — a conversão
para `int16` serve **só para o VAD decidir**; o áudio final é remontado a partir do
array `float` original, preservando precisão para o MFCC.

**E se o VAD zerar o áudio inteiro?**
O código devolve o original com `prop_fala = 0.0`, sinalizando o caso em vez de
quebrar a etapa seguinte.

**Vocês removem silêncio no VAD e depois adicionam zeros de volta. Não é
contraditório?**
Não. Os zeros do padding são determinísticos, idênticos para todas as amostras, e
servem só para igualar o formato do tensor — não são «silêncio natural» e não
informam nada ao modelo. E, justamente por isso, eles são **excluídos da agregação**:
é o mascaramento de padding.

---

## 2. Features manuais e sua ligação com a teoria

**Por que MFCC, ZCR e centróide espectral, e com estes parâmetros?**
`n_mfcc=20`: os primeiros coeficientes capturam o envelope grosso do espectro
(ressonância do trato vocal), os mais altos captam detalhes finos onde artefatos de
síntese tendem a aparecer. `win_length=400` ≈ 25 ms a 16 kHz, a janela em que a fala
é aproximadamente estacionária — é o janelamento discutido na fundamentação.
`hop_length=256` ≈ 16 ms de passo. ZCR capta ruído e fricativas de alta frequência;
o centróide é o «centro de massa» do espectro.

**Como uma série temporal vira um vetor de tamanho fixo?**
Média e desvio-padrão ao longo do tempo. 20 MFCC × 2 + ZCR × 2 + centróide × 2 =
**44 features**.

**Por que as features não estão padronizadas no CSV?**
Porque padronizar o CSV inteiro de uma vez faria o *scaler* enxergar média e desvio
das amostras de teste — **vazamento**. O `StandardScaler` entra **dentro** do
`Pipeline` de cada modelo, ajustado por fold. O RF não precisa dele (é invariante a
escala monotônica); o SVM precisa, porque depende de distância euclidiana.

**Por que `win_length=400` no centróide, se o librosa não exige?**
Porque o CSV antigo foi extraído **sem** esse parâmetro e o librosa caiu no default
(janela = `n_fft` = 512), deixando o centróide com resolução temporal **diferente** da
do MFCC e do ZCR. Corrigido no lote único de re-extração de 30/08.

**Por que não limitar as features a 4 kHz?**
Porque 57,9% do universo eval é banda estreita (alaw/ulaw/gsm/pstn, teto ~4 kHz) mas
**42,1% é banda larga** (g722/opus/none). Filtrar tudo em 4 kHz destruiria sinal real
em 42% dos dados, justamente na banda alta onde vivem os artefatos de síntese.
Decisão ratificada pelo orientador, com métricas reportadas **por codec** — e os
números sustentam a hipótese: RF 0,6958 (estreita) × 0,7510 (larga); SVM 0,7711 ×
0,8397, com **EER independente de limiar**, logo não é artefato do limiar.

---

## 3. Mascaramento de padding

**Por que mascarar a agregação?**
Dois argumentos, e não confundi-los é importante.

*(1) Validade de medida* — o argumento principal, que não depende de viés nenhum.
Metade do tensor é padding na prática (mediana de **120 frames válidos em 251**;
**53,07%** dos áudios têm mais da metade em padding), e incluí-lo desloca cada feature
em ~30% na mediana e até 47% no pior caso. Uma feature assim descreve, em boa parte,
a **formatação do vetor**, não o áudio.

*(2) Assimetria da distorção* — a distorção é assimétrica entre classes (até |0,47|
padronizado em `mfcc1_std`, `centroide_std`, `mfcc5_media`) porque o deslocamento que
o silêncio causa depende do **conteúdo acústico** de cada classe, não da quantidade de
zeros.

**A classe bonafide recebe mais padding que a spoof?**
**Não.** A fração de padding é 47,11% (bonafide) × 47,25% (spoof) — **0,14 p.p.**
A confusão vem de `prop_fala`, que é 62,94% × 84,65% (**21,71 p.p.**) e mede outra
coisa: a fração do áudio **original** que o VAD manteve. A correlação entre as duas é
de apenas **r = 0,145**. *(Esta é uma pergunta-armadilha e a v1 do
`DECISOES_PENDENTES_CNN.md` caiu nela; a v2 corrigiu.)*

**A janela é de 400 amostras e o passo é de 256. A janela do último frame válido
invade o zero-padding. Isso não contamina a agregação mascarada?**
Contamina parcialmente, sim, e por construção — a definição de frame válido é **pelo
centro** (convenção `center=True` do librosa), não pelo suporte integral da janela.
Como `400 > 256`, o último frame válido pode conter até 200 amostras de padding, e o
primeiro frame inválido — o frame de **transição** — ainda alcança até 200 amostras de
áudio real. Medido no piloto: a região `[:, n_valid:]` tem mais de um valor distinto
em **120 de 201** exemplos, justamente por isso.

O que torna isso aceitável é que a aproximação de ±1 frame é **idêntica nos dois
ramos**: `extrair_vetor` agrega `serie[:, :n_valid]` e a máscara da CNN marca as
mesmas `n_valid` posições, com `n_valid` vindo da **mesma função**
(`frames_validos`). Então o frame de transição é excluído dos dois e o último frame
válido é contaminado nos dois, na mesma medida — e a paridade, que é o que a
comparação exige, está preservada (checagem: 201/201).

A alternativa, exigir suporte integral da janela, reduziria a contagem em um frame e
abriria divergência com o `features.csv`, congelado desde 30/08 — pelo ganho de
excluir um frame que é ~78% áudio real. A escolha está registrada como aproximação
deliberada, não como descuido.

**Por que `prop_fala`, `n_frames_validos` e `n_frames_total` estão no CSV mas fora
do X?**
As três medem **quantidade de fala/silêncio**, não timbre. Como o bonafide perde mais
sinal no VAD que o spoof, qualquer uma delas dentro do X daria um atalho estatístico:
o modelo acertaria classificando pela **duração da fala**, não por artefato de
síntese. Ficam no CSV porque são auditáveis; ficam fora do X porque não são o objeto
de estudo. A exclusão é feita em `colunas_features`, o **ponto único** que define o X.

---

## 4. Espectrograma: tipo, resolução e normalização

**Que espectrograma é, exatamente?**
Log-Mel: mel-espectrograma de potência (`power=2.0`) com 128 filtros Mel de 0 a
8000 Hz, `n_fft=1024`, `win_length=400`, `hop_length=256`, `center=True`, convertido
para dB com `ref=1.0` e `top_db=80`. Formato **128 × 251**.

**⚑ Por que zero-padding da DFT não é aumento de resolução espectral?**
Porque a **resolução física** é determinada pela **janela de análise**, que continua
em 400 amostras (~25 ms). Subir o `n_fft` de 512 para 1024 não cria informação nova:
aumenta a **densidade de amostragem da DFT** (513 bins em vez de 257) por
zero-padding do sinal janelado. O ganho é **numérico**: com 257 bins, 61 dos 128
filtros Mel leriam ≤ 2 bins e 12 pares compartilhariam o bin de pico — regime
degenerado sobre o qual o librosa **não emite aviso**. Com 1024 bins, o banco de 128
filtros passa a ser representado adequadamente.

**Por que 251 e não 256?**
Porque 251 é o que o pipeline produz: `1 + 64000 // 256`, valor único nas 148.176
linhas do `features.csv`. Interpolar para 256 inventaria frames; e 251 é o número que
mantém a paridade exata com a definição de frame válido do ramo clássico.

**Por que `ref=1.0` e não `ref=np.max`?**
Duas razões, e a primeira é a que importa: `ref=np.max` **é** normalização por
exemplo (invariante a ganho, verificado), o que **contradiria** a normalização global
por estatísticas do treino da P3. A segunda: `librosa.feature.mfcc` aplica
`power_to_db` com os **defaults**, então a paridade de não-linearidade com o ramo
clássico exige `ref=1.0`.
**⚠ O que não se deve dizer:** que `ref=1.0` «preserva o ganho original das
gravações». Esse ganho já foi removido a montante pela normalização de pico por áudio,
antes do VAD.

**Como o espectrograma é normalizado, e por quê assim?**
Por **estatísticas globais do treino**, média e desvio **por faixa Mel** (dois vetores
de 128), calculados **somente sobre o treino daquele estágio** — 27k na fase de
*early stopping*, 30k no refit — e **somente sobre frames válidos**. Validação e teste
nunca entram no cálculo e nunca recalculam. O artefato `normalizacao_cnn.json` grava
média, desvio, conjunto de origem, nº de exemplos, hash da lista de IDs e semente — é
o que torna a inferência reproduzível.

**Por que por faixa Mel e não um escalar global?**
Porque as faixas Mel têm energia média muito diferente entre si (as graves concentram
a energia da fala). Normalizar por faixa iguala a escala entre elas sem apagar o
contraste temporal, que é o que a convolução lê.

**O que é a ressalva do `top_db=80`?**
O piso fica limitado a 80 dB abaixo do máximo **daquele exemplo**. A região de padding
vira, por isso, um platô **constante dentro de cada exemplo**, mas com valor
(`max − 80`) que **varia entre exemplos**. Não invalida a decisão; está documentado.

**Vocês entregam Log-Mel à CNN e MFCC ao RF/SVM. A comparação é justa?**
A paridade é descrita assim: entregar Log-Mel à CNN reduz a diferença entre os dois
pipelines **principalmente à forma de representação e aprendizado** — no ramo clássico
resumimos as características acústicas e classificamos com RF/SVM; na CNN preservamos
a estrutura tempo-frequência e deixamos a rede aprender a representação. O áudio de
entrada é **o mesmo**, produzido pela **mesma função** de pré-processamento.

**Em mais da metade do dataset, metade das faixas Mel do espectrograma é constante.
Por que gerar 128 faixas até 8 kHz então?**
Porque o dataset é heterogêneo, e é exatamente essa heterogeneidade que a decisão
respeita. 57,9% do universo é banda estreita (alaw, ulaw, gsm, pstn, teto ~4 kHz) e
nesses áudios a metade superior das faixas é platô no piso do `top_db` — medido no
piloto: **65 de 118** áudios de banda estreita, contra **2 de 83** de banda larga.
Mas os outros **42,1%** têm sinal real acima de 4 kHz, e é justamente ali que vivem
os artefatos de síntese. Um `fmax = 4000` global destruiria esse sinal em 42% dos
dados para economizar faixas mortas em 58% — e os números por codec mostram que a
banda larga é onde os dois modelos clássicos vão melhor (RF 0,6958 → 0,7510;
SVM 0,7711 → 0,8397, com EER independente de limiar).

O platô não é um atalho de classe: a subamostra é estratificada por classe × codec ×
ataque, então o codec não prediz o rótulo. O que a rede pode aprender é uma regra
específica por codec — que é o que RF e SVM já fazem implicitamente, via ZCR e
centróide, que se deslocam com o teto de banda.

**A filterbank é a mesma nos dois ramos?**
A **convenção da escala Mel e as frequências centrais dos 128 filtros** são as mesmas
(`htk=False`, `norm='slaney'` em ambos). O que difere é a **densidade de amostragem da
DFT** (`n_fft` 512 no clássico, 1024 na CNN). E há uma limitação registrada aqui: a
filterbank interna do `librosa.feature.mfcc`, com o `n_fft=512` congelado, **herda o
mesmo regime degenerado** que motivou a P4 — não corrigido porque as features estão
congeladas desde 30/08 e re-extrair abriria o Bloco 2; atenuado pela DCT do MFCC, que
retém 20 de 128 coeficientes.

---

## 5. Split, validação cruzada e braço duplo

**⚑ Onde exatamente o 70/15/15 e o 5-fold acontecem?**

```
148.176 (fase == 'eval')
 ├── TREINO 70% (103.723)  → é AQUI que o Stratified 5-Fold roda, durante a
 │                            busca de hiperparâmetros. O 5-fold particiona
 │                            SÓ o treino.
 ├── VALIDAÇÃO 15% (22.226) → comparar modelos, escolher o limiar
 └── TESTE 15% (22.227)    → INTOCADO até o final. Usado UMA vez.
```

**Por que três conjuntos e não dois?**
Se hiperparâmetros são escolhidos olhando o teste, o teste deixa de ser estimativa
honesta de generalização. A validação absorve esse desgaste.

**Por que estratificado?**
Com 9:1, um split aleatório simples deixa a proporção oscilar entre subconjuntos. A
estratificação por `classe_binaria` força cada subconjunto a manter a proporção
original — conferido automaticamente em `resumo_split`.

**Por que o universo é `eval` e não os 181.566 do `labels.csv`?**
Os 181.566 são `eval` (148.176) + `progress` (16.464) + `hidden` (16.926). O `eval` é
o conjunto oficialmente pontuado do ASVspoof 2021 LA. O `hidden` tem silêncio
**pré-cortado na origem** (`trim == 'only_speech'`), um pré-processamento distinto do
restante que contamina qualquer análise de proporção de fala. Verificado por script,
não presumido: `trim == 'notrim'` em 100% do `eval`.

**Qual a limitação do split, e como foi mitigada?**
É aleatório **por utterance**: cada ataque (A07–A19), codec e locutor aparece em treino
**e** teste, então o modelo pode memorizar a assinatura de um vocoder ou locutor.
Métricas potencialmente **otimistas** e **não comparáveis** ao EER de 1,32% de
Yamagishi et al. (2022), cujo protocolo é deliberadamente **cross-attack**. Mitigação:
métricas por ataque e por codec. Leave-one-attack-out previsto como análise
complementar.

**Por que dois braços?**
A curva de aprendizado mostra que o RF **não satura** antes do treino completo: o
último passo (80.000 → 103.723) ainda rende +0,0121 de f1_macro, 2,4× a tolerância de
0,005. Logo a subamostra de 30k **custa desempenho real**. O braço principal (RF, SVM
e CNN nas mesmas 30k) é o que responde à pergunta de pesquisa; o de referência (RF no
treino completo) **quantifica o custo da subamostragem** e **não é concorrente direto**
de SVM/CNN. Validação e teste permanecem **completos nos dois braços**.

**Por que subamostrar, então?**
Porque o SVM-RBF é O(n²)–O(n³) e não roda nos 103.723. Uma subamostra única,
estratificada por classe × codec × ataque (98 estratos, seed 42) e **compartilhada**
pelos três modelos preserva o «mesmo ambiente experimental» da pergunta de pesquisa.

**Quanto faltaria para o RF alcançar o SVM?**
Ajustando `f1_macro ~ a·ln(n) + b` sobre sete pontos (R² = 0,9934): ~**276.116**
áudios de treino — 2,66× o treino completo e 1,86× o universo eval. Ou seja:
**dentro dos dados disponíveis, o RF não alcança o SVM nem usando tudo.**
*Ressalva obrigatória:* é extrapolação log-linear **fora da faixa medida**; curvas de
aprendizado costumam achatar, então o número é um **limite otimista para o RF**.

---

## 6. Desbalanceamento e `class_weight`

**Como cada modelo trata o desbalanceamento de 9:1?**
RF e SVM: `class_weight='balanced'` (`w_c = n / (k·n_c)`), com
`balanced_subsample` testado como alternativa no Random Search do RF. CNN:
*loss* ponderada — `CrossEntropyLoss` com duas saídas e peso maior para **bonafide**.
**Sem** undersampling ou oversampling em nenhum dos três.

**⚑ Por que a CNN não usa `BCEWithLogitsLoss(pos_weight=9)`?**
Porque **inverteria a intenção**. A convenção do projeto é `bonafide = 0`,
`spoof = 1`, e **spoof é a classe MAJORITÁRIA** (≈9:1). `pos_weight` pesa a classe
positiva — daria peso 9 à majoritária. A implementação aprovada é
`CrossEntropyLoss` com duas saídas e peso maior para a classe 0.

**Como validar se o `class_weight` bastou?**
Não bastou, e isso está medido. Com `class_weight='balanced'` e limiar 0,50 o RF tinha
AUC ~0,90 e f1_macro baixo: **as folhas puras do RF neutralizam o peso de classe** no
`predict_proba`, e o limiar ótimo fica longe de 0,50. As três correções: pôr
`min_samples_leaf ∈ [5,20]` no espaço de busca, testar `balanced_subsample`, e
transformar a **seleção de limiar na validação** em regra de protocolo. Causa raiz em
`NOTA_LIMIAR.md`.

---

## 7. Protocolo de limiar e métricas

**Como o limiar é escolhido?**
Na **validação**, maximizando f1_macro, com regra `score >= limiar`, candidatos =
`np.unique(scores)`, e apenas **aplicado** em qualquer outro conjunto. Mesma regra
para RF, SVM e CNN, na mesma função (`selecionar_limiar`).

**Por que `np.unique` e não uma grade?**
Sob a regra `>=`, qualquer limiar entre dois scores observados consecutivos produz
**exatamente a mesma partição** — logo os valores únicos são o conjunto candidato
**completo e mínimo**. Uma grade de 0,05 cai *entre* degraus e pode passar ao largo do
ótimo.

**O limiar do SVM é negativo. Isso não é inconsistência?**
Não. Ele vive na escala do `decision_function` (real, centrada em zero), não em
[0,1] como o `predict_proba` do RF. A regra do protocolo é **agnóstica de escala** e
`selecionar_limiar` opera sobre `np.unique(scores)` sem supor intervalo.

**Por que `>=` e não `>` ou argmax?**
Porque as três regras **divergem nos empates** — três regras diferentes sob o mesmo
nome «limiar 0,50». O projeto fixou uma, escrita (`REGRA_DECISAO`), compartilhada
pelos três modelos, e `avaliar` **deriva** o `y_pred` internamente para que nenhum
chamador possa usar outra regra por acidente. A divergência concreta está medida em
`nota_divergencia_f1.md`: 25 amostras.

**⚑ Como cada métrica é calculada?**
`f1_macro` é a média **não ponderada** das duas classes — com 9:1, muito mais
informativo que a acurácia, que deixaria a majoritária esconder o fracasso na
minoritária. O **EER** vem da curva ROC: o ponto onde FNR = FPR, tomado como
`|FNR − FPR|` mínimo, com classe positiva = spoof. `zero_division=0` nas precisões,
para o caso de o modelo nunca prever uma classe.

**Por que a busca de hiperparâmetros pontua por EER e não por f1_macro?**
Porque o EER é **independente de limiar**: usar f1_macro na busca misturaria a escolha
de hiperparâmetros com a escolha de limiar, e o limiar é decidido depois, na
validação. *(Na CNN, a seleção de arquitetura e época usa f1_macro nos 3k, porque ali
se escolhe **quando parar**, que reflete uma decisão binária. Os dois números estão
reportados para cada configuração.)*

**Por que o EER não é comparável ao 1,32% da literatura?**
A **grandeza** é a mesma; os **valores**, não. O protocolo aqui é split interno
aleatório por utterance; o do ASVspoof é cross-attack. Ser independente de limiar
remove a arbitrariedade do 0,50, **não** a diferença de protocolo.

**Quando o teste foi usado?**
**Uma vez**, em 29/09, com os limiares já escolhidos na validação e todos os modelos
carregados dos artefatos persistidos. A regra mora no código:
`carregar_modelo_ajustado` **recusa** um limiar cujo JSON não registre
`selecao_limiar.conjunto == 'validacao'`. E o script tem guarda de execução única.

---

## 8. A CNN

**Descreva a arquitetura.**
CNN pequena: blocos `Conv2d(3×3) → BatchNorm → ReLU → MaxPool(2×2)`, seguidos de média
integral no eixo de frequência, **masked global pooling** no eixo do tempo, dropout e
uma camada densa com **duas saídas**.

**⚑ Como a máscara temporal acompanhou a redução do eixo do tempo dentro da CNN?**
A máscara é reduzida pela **mesma operação** que reduz o mapa de ativação:
`max_pool1d` com o mesmo kernel e stride do eixo do tempo do `MaxPool2d`, a cada
bloco. Assim os comprimentos coincidem **por construção**, não por uma fórmula
aritmética que poderia ficar desatualizada ao mudar a arquitetura — e há um `assert`
de comprimento dentro do `forward`. A convenção é de teto: uma posição reduzida é
válida se **qualquer** posição original dela era válida, coerente com o `ceil` de
`frames_validos`. Ao final, o *pooling* soma **apenas** as posições válidas e divide
pelo **número** de posições válidas daquele exemplo.

**Por que a máscara atua só no tempo?**
Porque só existe padding no tempo. As 128 faixas Mel são válidas em todo exemplo;
mascarar em frequência seria inventar um problema.

**Por que `Flatten` está vetado?**
Duas razões. Infla os parâmetros (128×251 achatado dá ~32 mil entradas na densa) e,
mais importante, **amarra a rede às posições de padding** — a rede aprenderia «onde»
o áudio termina, que é exatamente a variável espúria que o Bloco 1 removeu do X do
ramo clássico.

**Por que a máscara na agregação e não na entrada?**
Porque é a **tradução literal** do que o Bloco 1 aprovou para o ramo clássico:
mascarar a **agregação**, não a entrada. As alternativas foram avaliadas e recusadas:
cortar e redimensionar **destrói a duração** (um áudio de 8 frames viraria 251, fator
31×, com 243 frames de conteúdo inventado); máscara como 2º canal duplica o tensor e
é em boa parte redundante com o platô de dB; espectrograma cru sem mascarar deixaria a
diluição do *pooling* por conta da sorte, num fator que **varia por exemplo** (8 a
250 frames válidos).

**Como vocês provaram que o *pooling* mascarado funciona?**
Teste de invariância ao padding: dois exemplos idênticos na parte válida, com padding
de valores diferentes, produzem praticamente a mesma saída — e **a mesma comparação
sem a máscara** produz diferença muito maior. Os dois números estão em
`checagem_mascara_cnn.json`. *(A diferença não é exatamente zero porque a convolução
com `padding=1` vaza ~1 frame por camada na fronteira; o que o teste demonstra é a
ordem de grandeza entre com e sem máscara.)*

**⚑ Por que a CNN parou de treinar onde parou?**
*Early stopping* com critério **f1_macro num conjunto interno de 3.000 exemplos**,
separado dos 30k por estratificação classe × codec × ataque, com paciência de 8
épocas. Paciência longa de propósito: com apenas 300 bonafide nos 3k, a oscilação
entre épocas é **variância de estimativa**, e paciência curta subestimaria a melhor
época. A curva por época mostra o ponto em que a loss de treino continua caindo
enquanto a dos 3k para ou sobe.

**Por que não usar a validação externa para *early stopping*?**
Porque então o limiar selecionado depois nessa mesma validação estaria escolhido sobre
um conjunto que já influenciou o treino — e o protocolo de RF, SVM e CNN deixaria de
ser o mesmo. A validação externa de 22.226 **não** é usada para *early stopping* em
fase nenhuma.

**⚑ Por que o refit nos 30k remove o privilégio que RF e SVM não tiveram?**
Porque o *early stopping* consumiu 3.000 dos 30.000, deixando a CNN treinada em 27k
enquanto RF e SVM treinaram em 30k. Sem o refit, a comparação carregaria um fator
escondido de 3.000 amostras além do modelo. O refit retreina a arquitetura escolhida
nos **30.000 completos**, pelo número de épocas já definido, com as estatísticas de
normalização **recomputadas** nesse conjunto — e com isso a CNN final vê **as mesmas
30 mil amostras** que RF e SVM. RF e SVM nunca precisaram separar um pedaço do treino
para decidir quando parar; a CNN precisou, e o refit paga essa dívida.

**O refit usa o mesmo nº de épocas. Mas 30k é maior que 27k.**
Correto, e está registrado como limitação: o mesmo nº de épocas nos 30k corresponde a
**11,1% mais atualizações de gradiente**. A alternativa (fixar o nº de atualizações)
foi considerada e descartada por simplicidade de descrição.

---

## 9. Tempo de inferência

**⚑ Como o tempo foi medido, e a comparação é justa?**
RF e SVM rodam em CPU; a CNN em GPU. Comparar 6,4 ms de RF em CPU com uma CNN em GPU
não é justo em nenhuma direção: medir tudo em CPU penaliza a CNN, e medir cada um no
seu hardware natural compara **máquinas**, não **modelos**. Por isso o protocolo
reporta **quatro** medições: RF-CPU, SVM-CPU, **CNN-GPU** e **CNN-CPU**.

**Qual é o argumento por trás dessas quatro?**
A vantagem dos clássicos, na pergunta de pesquisa, **não** é serem mais rápidos em
igualdade de hardware — é **não exigirem GPU**. O par CNN-GPU / CNN-CPU é o único
arranjo que evidencia isso.

**Quais cuidados de medição?**
Descartar as primeiras execuções (o contexto CUDA, a carga de kernels e o autotune do
cuDNN podem tornar a 1ª inferência 100× mais lenta); `torch.cuda.synchronize()`
**antes e depois** do trecho cronometrado (CUDA é assíncrono: sem isso mede-se o
tempo de **enfileirar**, não de executar, e a CNN pareceria milhares de vezes mais
rápida); medir latência (`batch=1`) e throughput (lote) **separadamente**; `n_jobs`
fixo e declarado (1) para todos; mediana + mínimo + máximo de N execuções, nunca
medição única; hardware e versões no JSON. E — achado de B4.7 — **aquecer o
dispositivo antes do cronômetro**: os 3 descartes do protocolo são 3 *iterações*
(~3 ms em GPU) e não tiram a placa do estado de baixo consumo depois de um trecho
longo de CPU. Medimos **8,83 ms com a GPU fria contra 0,82 ms quente**. É o
**espelho** do erro do `synchronize()`: um infla a medida, o outro a desinfla, e
nenhum dos dois aparece na leitura do código. Os dois números ficam publicados lado a
lado em `cnn_final_principal.json`.

**Qual foi o achado?**
A hipótese registrada de que o pré-processamento dominaria **não se confirmou para o
RF**, mas **se confirmou para o SVM e para a CNN em GPU**. Por áudio, batch = 1, os
três medidos **na mesma execução** (20/09): base do ramo clássico 4,3474 ms
(carregar 0,5735 + VAD 0,3394 + features 3,4345) e base do ramo CNN 3,4087 ms
(carregar 0,5815 + VAD 0,3497 + log-Mel 2,4775); predição **RF 5,8819** (total
**10,2293** — 57,5% no classificador), **SVM 0,5717** (total **4,9191** — 88,4% na
base), **CNN-GPU 1,1678** (total **4,5765** — 74,5% na base, com o log-Mel sozinho em
54,1%) e **CNN-CPU 13,3881** (total **16,7968** — 79,7% no classificador).

Três conclusões: **a CNN em GPU é mais barata ponta a ponta que o RF** (4,58 contra
10,23 ms) e a **CNN em CPU é a mais cara de todas** (16,80 ms), logo a resposta sobre
custo é **condicional ao hardware**; **em lote a ordem muda outra vez** (RF 0,0204
ms/áudio, CNN-GPU 0,2383, SVM 0,4025 — o RF é ~19,7× melhor que o SVM); e portanto
**não existe «o modelo mais barato» sem dizer o regime** — latência unitária,
throughput em lote e presença de GPU dão respostas diferentes.

---

## 10. Reprodutibilidade e estabilidade

**Como vocês garantem reprodutibilidade?**
Semente única no `config.yaml`, `random_state` explícito em todo componente do
scikit-learn (nunca dependendo do estado global), artefatos assinados
(`features.meta.json`, `espectrogramas.meta.json`, `normalizacao_cnn.json`) com
parâmetros, hashes e commit, guardas de esquema que **abortam** retomadas inválidas, e
`scripts/guarda_reproducao.py`, que compara os JSONs de resultado campo a campo contra
cópias pré-revisão (deixando de fora medidas de relógio, que variam por definição).

**Fixar semente resolve tudo?**
Não. Determinismo ≠ robustez: você pode ter fixado a semente numa partição «sortuda».
Semente fixa dá reprodutibilidade; k-fold e bootstrap dão confiança estatística.

**No PyTorch, `manual_seed` basta?**
Não. São necessárias quatro travas: `manual_seed`/`manual_seed_all`,
`cudnn.deterministic=True`, `cudnn.benchmark=False` (o benchmark escolhe o algoritmo
medindo em tempo de execução, e essa escolha varia com a carga da máquina) e
`CUBLAS_WORKSPACE_CONFIG`, que precisa ser definido **antes** da primeira chamada
CUDA. Operações do cuDNN somam em ponto flutuante em ordem variável, e soma de float
não é associativa — a diferença nos últimos bits se amplifica ao longo do treino.

**Vocês encontraram algum caso concreto disso?**
Sim, no RF. `predict_proba` com `n_jobs=-1` **não é reprodutível bit a bit**: quatro
predições do mesmo `.joblib` sobre a mesma validação deram 22.104 / 22.102 / 22.104 /
22.103 scores distintos; com `n_jobs=1`, 22.104 nas quatro, idênticas. A maior
diferença foi 4,4×10⁻¹⁶ e **nenhuma métrica muda** — mas
`selecao_limiar.n_candidatos` é uma contagem de valores distintos e enxerga o último
bit. Correção: `predizer_rf()` força `n_jobs=1` na predição; o treino segue paralelo.

**A vantagem do SVM sobre o RF é real ou ruído?**
Real. O estatístico que decide é o **bootstrap pareado**: os dois são avaliados
exatamente nas mesmas 22.226 amostras, então os erros são **correlacionados**, e é a
diferença reamostrada em conjunto (1.000 vezes, um único vetor de índices por
reamostragem) que responde. Δf1_macro = **+0,0762** IC95 [0,0663; 0,0856];
ΔEER = **−0,0466** IC95 [−0,0560; −0,0377]. **Nenhum IC contém zero**, e o SVM é
melhor em **100%** das 1.000 reamostragens.

**E as sementes?**
O RF entre 5 sementes varia ±0,0004 de f1_macro; o SVM com `probability=False` é
**determinístico** (o `random_state` só afeta o Platt scaling), então não há variância
de treino a medir. Essas são **heurísticas de contexto, não o teste** — comparar
contra a maior dispersão individual ignora a correlação entre os erros dos dois
modelos, e o próprio `estabilidade_rf_svm.json` as rotula assim. A fonte de variação
que o braço principal de fato tem é **qual subamostra de 30k caiu**, medida em
`estabilidade_subamostra.json`.

**Como sabemos que o `top10_features` não é artefato do viés de impureza?**
Porque foi confirmado por `permutation_importance` na validação (scoring por AUC,
`n_repeats=10`): os dois rankings concordam com Spearman ρ = 0,8516, 9 das 10 do topo
em comum, `mfcc1_media` e `mfcc1_std` em 1º e 2º nos dois. Continua sendo «o que o
modelo usou», não causalidade acústica.

---

## 11. Perguntas difíceis de processo

**Por que registrar as decisões do espectrograma num documento antes de gerar?**
Porque gerar os 74.453 tensores é o «lote único do Bloco 4»: caro, demorado e não
repetível dentro do cronograma. É a mesma situação do Bloco 2, em que a decisão de
mascarar padding foi registrada **antes** da extração — e foi isso que permitiu,
depois, defender a escolha com um documento em vez de com memória.

**Vocês geraram 74.453 e não 148.176. Por quê?**
Porque o braço de referência é **RF-only** (RF no treino completo) e nunca precisa de
espectrograma. Gerar o universo inteiro produziria 73.723 tensores que nada consome, e
~9,5 GB a mais.

**Alguma vez um artefato gerado destruiu informação importante?**
Sim, e é uma lição registrada. Um bloco de limitação
(`limitacao_otimo_na_borda`, o registro de que o `min_samples_leaf` ótimo caiu no
limite inferior da faixa exigida) tinha sido escrito **à mão dentro de um JSON
gerado** — e a re-execução o apagou em silêncio. Correção: `analisar_bordas()`
**deriva** o bloco do espaço de busca e da configuração vencedora, de modo que ele
volta sozinho a cada execução e não pode ficar desatualizado. **Uma limitação
metodológica que mora num artefato gerado é destruída por qualquer re-execução.**

**Se você tivesse mais tempo, o que faria primeiro?**
Leave-one-attack-out / split por ataque — porque é a limitação mais forte do trabalho
e porque o próprio diagnóstico por ataque já mostra que a dificuldade varia **muito
mais** entre sistemas de síntese (até 0,29 de EER) do que entre modelos (ΔEER 0,0466).
Depois: cross-dataset, e a CNN com o treino completo (que exigiria abrir mão da
comparação em igualdade de n).
