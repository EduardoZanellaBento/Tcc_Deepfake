# Decisões pendentes — definição do espectrograma da CNN (Bloco 4)

> **Status:** **RESPONDIDO E FECHADO em 17/09/2026.** As sete perguntas foram
> respondidas e aprovadas pelo orientador. As decisões estão transcritas na seção
> «Respostas do orientador — 17/09/2026», abaixo do sumário, e aplicadas no bloco
> `espectrograma:` do `config/config.yaml`. **Nenhuma delas se reabre.**
> **Aberto em:** 03/09/2026 · **Revisado em:** 17/09/2026 (v4) ·
> **Fechado em:** 17/09/2026 (v5 — transcrição das respostas).
> **Aluno:** Eduardo Zanella Bento · **Orientador:** Prof. Anderson Fola.

## Respostas do orientador — 17/09/2026

As sete perguntas foram respondidas e **aprovadas**. Esta seção é a transcrição das
decisões; **as perguntas, a evidência medida, as opções e as recomendações continuam
abaixo, intactas** — é a evidência que sustenta a decisão numa arguição, e apagá-la
para deixar só a resposta destruiria exatamente o que fez este documento valer a
pena. Cada `## Pergunta N` recebeu uma linha de marcação apontando para cá.

**Todas as sete decisões já estão aplicadas no bloco `espectrograma:` do
`config/config.yaml`** (B4.0, 17/09/2026), junto com a justificativa de cada uma e o
sub-bloco `defaults_librosa_registrados`. **Nenhuma se reabre.**

| # | decisão |
|---|---|
| **P4** | `n_fft=1024`, `win_length=400`, `hop_length=256`, `n_mels=128`, `fmin=0`, `fmax=8000`, `power=2.0`, `center=true` |
| **P5** | Log-Mel em dB, `ref=1.0`, `top_db=80`. **Nunca `ref=np.max`** |
| **P3** | Normalização por estatísticas **globais do treino**, média e desvio **por faixa Mel**. Validação e teste nunca entram no cálculo |
| **P1** | `largura = 251`, `altura = 128`. Não interpolar para 256 |
| **P2** | Espectrograma cru de formato fixo + **agregação mascarada dentro da rede** (opção D) |
| **P6** | Split interno 27k/3k para *early stopping* + **refit final nos 30k** |
| **P7** | *Loss* ponderada: `CrossEntropyLoss` com 2 saídas e peso maior para **bonafide (classe 0)**, razão 9:1. **Sem** over/undersampling |

A ordem da tabela é a **ordem de urgência** da seção «O que se pede ao orientador»,
não a ordem numérica: P4 primeiro porque é a que falha em silêncio.

### As duas formulações proibidas no texto

Vieram junto com a aprovação, e as duas aparecem naturalmente ao escrever.

**(a) Não escrever que subir o `n_fft` de 512 para 1024 «aumenta a resolução
espectral real».** A janela continua em 400 amostras (~25 ms), então nenhuma
informação nova é criada. Formulação correta:

> mantemos os mesmos 25 ms de análise e usamos FFT de 1024 para **aumentar a
> densidade de amostragem da DFT por zero-padding**, permitindo representar o banco
> de 128 filtros Mel de maneira numericamente mais adequada.

**(b) Não escrever que `ref=1.0` «preserva o ganho original das gravações».** O
pipeline já faz normalização de pico por áudio **antes** do VAD
(`preprocessamento.py:62-77`), então esse ganho já foi removido a montante.
Justificativa correta:

> `ref=1.0` evita introduzir uma **segunda** normalização, relativa ao máximo do
> próprio espectrograma — que contradiria a normalização global da P3 —, e mantém
> paridade com a transformação que o MFCC já aplica.

**E a paridade entre os ramos descreve-se assim:** entregar Log-Mel à CNN reduz a
diferença entre os dois pipelines **principalmente à forma de representação e
aprendizado** — no ramo clássico resumimos as características acústicas e
classificamos com RF/SVM; na CNN preservamos a estrutura tempo-frequência e deixamos
a rede aprender a representação.

---

### Refinamentos obrigatórios que vieram com a aprovação

As quatro exigências abaixo **não são decisões novas**: são a forma aprovada de
executar as decisões acima. Duas delas (a máscara e a *loss*) corrigem uma
implementação que pareceria certa e estaria errada.

#### 1. A máscara temporal tem de acompanhar a redução do eixo do tempo (P2)

A opção D não é «mascarar a entrada». Dentro da rede, a cada `stride`, *pooling* ou
qualquer camada que encurte o eixo do tempo, **a máscara é reduzida de forma
coerente**. O resultado final é um **masked global pooling** — soma apenas das
posições válidas, dividida pelo número de posições válidas —, nunca um
`GlobalAveragePooling` comum sobre regiões inválidas.

- A máscara atua **só no tempo**; o eixo de frequência continua integral.
- **`Flatten` está vetado** como fuga do problema: infla os parâmetros e amarra a
  rede às posições de padding.
- A definição de frame válido é **a mesma** do ramo clássico: `frames_validos()` em
  `src/features/extrair_features.py:119-135`, e `n_frames_validos` já está no
  `features.csv` por `arquivo`.
- Forma da arquitetura: **CNN pequena + agregação global mascarada.**

É o que torna a P2 a tradução literal do que o Bloco 1 aprovou para RF e SVM —
mascarar a **agregação**, não a entrada.

#### 2. Artefato de normalização: `normalizacao_cnn.json` (P3)

As estatísticas não podem viver só dentro do script que as calculou. **Salvar
`normalizacao_cnn.json` no Git**, com:

| campo | por quê |
|---|---|
| média por faixa Mel (128 valores) | é o que se aplica em inferência |
| desvio por faixa Mel (128 valores) | idem |
| conjunto de origem | prova que validação e teste ficaram fora |
| nº de exemplos | prova de qual fase gerou o artefato (27k ou 30k — ver refinamento 3) |
| hash / lista de IDs utilizada | torna o cálculo reconferível |
| *seed* | reprodutibilidade |

É **o que torna a inferência reproduzível**, e é por isso que o orientador o exigiu
versionado. Nasce no **B4.3**. Os tensores em disco **não** ficam normalizados: a
média/desvio são aplicados em **tempo de carga**, justamente porque o refit da P6
recomputa as estatísticas.

#### 3. Protocolo da CNN em duas fases (P6)

Esta é a forma aprovada, e ela **substitui** a versão da recomendação original, que
deixaria a CNN final treinada em apenas 27k:

```
30k → split interno fixo (27k treino / 3k early stopping, seed 42,
      estratificado por classe × codec × ataque, salvo em disco)
    → estatísticas de normalização calculadas nos 27k
    → treina, seleciona arquitetura/regularização, registra a MELHOR ÉPOCA
    → REFIT FINAL nos 30k completos: recomputa média/desvio nos 30k,
      treina a arquitetura escolhida por nº FIXO de épocas
    → validação externa APENAS para escolher o limiar
    → teste lacrado, execução única
```

**A validação externa de 22.226 não é usada para *early stopping* em fase nenhuma.**
Com o refit, a CNN final vê **as mesmas 30 mil amostras** que RF e SVM — que era a
objeção de simetria que a opção A, sozinha, deixaria em aberto. O corte 30k → 27k
passa a valer só na fase de seleção.

#### 4. A armadilha da *loss* ponderada (P7)

A convenção do projeto é `bonafide = 0`, `spoof = 1`, e **spoof é a classe
MAJORITÁRIA** (≈ 9:1, medido nos quatro conjuntos — trava 11 do script de auditoria).
Portanto:

> **`BCEWithLogitsLoss(pos_weight=9)` INVERTERIA a intenção**, dando peso maior à
> majoritária. É o erro que a implementação «óbvia» comete.

Implementação aprovada: **`CrossEntropyLoss` com duas saídas e peso maior para a
classe 0 (bonafide)**, na razão 9:1 — ou pelos pesos `balanced` da mesma lógica do
scikit-learn. **Sem** over/undersampling: a composição da época fica intacta, e a
seleção de limiar continua sendo o **segundo** mecanismo, compartilhado pelos três
modelos. A saída da avaliação continua compatível com o protocolo do repositório:
**maior score = maior evidência de spoof**.

---

### O que esta seção NÃO fecha

Duas coisas que dependem destas decisões mas pertencem a marcos posteriores, e que
**não** foram decididas aqui:

- a **arquitetura** da CNN (nº de blocos, canais, regularização) — B4.4/B4.5;
- os **hiperparâmetros de treino** (otimizador, *learning rate*, *batch*, épocas
  máximas) — B4.4.

O que está fechado é a **entrada** e o **protocolo**.

---

## Por que isto está por escrito, e não num chat

Gerar os espectrogramas é o **"lote único" do Bloco 4**: caro, demorado e não
repetível dentro do cronograma. É a mesma situação do Bloco 2, em que a decisão
de mascarar o padding foi registrada **antes** da extração — e foi isso que
permitiu, depois, defender a escolha com um documento em vez de com memória.
Errar a definição do espectrograma agora custa a semana inteira de CPU.

São **sete perguntas** e **uma regra de implementação**. Cada pergunta traz a
evidência conferida no repositório, as opções com custo e risco, e uma
recomendação.

### Procedência dos números desta versão

Os números deste documento são recomputados por
**`scripts/auditar_decisoes_cnn.py`**, cuja saída fica em
`results/metricas/auditoria_decisoes_cnn.json`. É o mesmo papel de
`validar_split_pos_lote.py` no Bloco 2: a prosa vira trava.

**Até onde a trava alcança, exatamente** (a v2 prometia mais do que o script
entrega; a v3 corrigiu a promessa e a v4 acrescentou a décima primeira trava). O
script **falha com código de saída 1** se, e somente se, uma destas **onze**
condições deixar de valer:

| # | o que trava | protege |
|---|---|---|
| 1 | MD5 do `features.csv` | procedência |
| 2 | 148.176 linhas | procedência |
| 3 | `n_frames_total` único e igual a `1 + 64000//256` | P1 |
| 4 | fração de padding entre classes diverge < 1 p.p. | P2 (a simetria) |
| 5 | `prop_fala` entre classes diverge > 5 p.p. | P2 (a assimetria) |
| 6 | 512/128 continua degenerado | P4 |
| 7 | nº de frames independe do `n_fft` | P4 (opção A) |
| 8 | o padding continua virando platô constante | P2/P5 |
| 9 | `ref=np.max` invariante a ganho e `ref=1.0` não | P5 |
| 10 | o MFCC continua sendo dB absoluto, com o ganho isolado no `c0` | P5 |
| 11 | **9,00:1 nos quatro conjuntos, subamostra inclusive** *(nova na v4)* | **P7** |

O script **recomputa e publica** no JSON, mas **não trava**, os valores
numéricos em si (estatísticas de `n_frames_validos`, os 53,07%, a tabela de
filtros mel, os dB da nuance do `top_db`, disco e escopo): eles são conferidos
por comparação com o JSON, não por asserção.

**O que continua fora do script, e por quê.** Sobraram exatamente duas coisas, e
as duas estão identificadas como conferência de mão onde aparecem: os deltas de
**30,3% / 47,03%** da P2, que vêm do piloto (`checagem_mascaramento.json →
parte_a_ab_controlado`) e não do universo, e os valores em **GiB e float16 do
universo inteiro**, que são aritmética sobre um número que o script já publica.
Fora dessas duas, todo número deste documento tem contrapartida no
`auditoria_decisoes_cnn.json` — e essa é a regra que passa a valer: se uma
afirmação nova não tiver onde se apoiar lá, ou ela ganha um bloco no script, ou
não entra no documento.

As estatísticas vêm do **`features.csv` completo** (148.176 linhas, MD5
`51b2f439bf6f1e10237acbc620bb92d9`, reconferido pelo script), e **não** das
amostras do piloto em `checagem_mascaramento.json`. Onde um número vier de uma
das amostras do piloto, está dito explicitamente — e **qual** delas, porque são
duas (ver a nota abaixo). Os cálculos de filtros mel da P4 e
os testes de escala da P5 são executados com `librosa` nos mesmos parâmetros do
`config.yaml`.

> **Nota de divergência (esperada, não é erro):** o repositório cita em vários
> lugares "mediana de **121** frames válidos" — esse número vem da amostra do
> runner do piloto, **n=500** (`checagem_mascaramento.json → parte_b_runner`:
> 50 bonafide e 450 spoof). No universo de 148.176 a mediana é **120**. As duas
> medidas estão certas; são universos diferentes. O mesmo vale para a fração de
> padding (44,8%/45,2% na amostra **n=200 balanceada** do A/B controlado —
> `parte_a_ab_controlado`, que é outra amostra — contra 47,1%/47,3% no
> universo) — e a conclusão é a mesma nos dois.
>
> **Cuidado ao citar:** o `checagem_mascaramento.json` tem **duas** amostras
> distintas, e a v2 deste documento as trocou. `parte_a_ab_controlado` é o A/B
> de **n=200 balanceado** (de onde saem os deltas de 30,3%/47,03% e as frações
> de padding por classe); `parte_b_runner` é a checagem de execução de **n=500**
> (de onde sai a mediana de 121). Não são a mesma coisa.

---

## Pergunta 1 — Largura do espectrograma: **251** ou **256**?

> **RESPONDIDA em 17/09/2026 — `largura = 251`, `altura = 128`; sem interpolar para 256. Ver «Respostas do orientador».**

**O que o `config.yaml:180` diz hoje:** `largura: 256`.

**O que o pipeline produz de fato: 251 frames.** Conferido em três fontes
independentes:

| fonte | evidência |
|---|---|
| aritmética | `1 + 64000 // 256 = 251` (4,0 s × 16 kHz = 64.000 amostras, `hop_length = 256`, `center=True`) |
| código | `frames_validos()` em `src/features/extrair_features.py:119-135` |
| dados | `n_frames_total` no `features.csv` congelado tem **um único valor: 251**, nas 148.176 linhas |

**Se ficar 256, só há dois caminhos, e os dois têm custo metodológico:**

1. **redimensionar** 251 → 256 por interpolação: o eixo temporal da CNN deixa de
   ser o mesmo eixo temporal das features do ramo clássico, e "mesmo
   pré-processamento para os três modelos" deixa de ser verdade;
2. **acrescentar 5 frames de zero**: reintroduz padding logo depois de o Bloco 1
   inteiro ter sido gasto tirando padding da agregação.

**Recomendação: `largura: 251`.** É o único valor que mantém o eixo temporal da
CNN idêntico ao do ramo clássico, e é a única leitura de "mesmo
pré-processamento" que se sustenta numa arguição. O ajuste é de uma linha no
config e não custa nada agora; custaria a re-geração inteira depois.

> **Nota técnica que a P4 vai usar:** o número de frames depende **apenas** de
> `hop_length` e da duração — é `1 + 64000//256 = 251` independentemente do
> `n_fft`. Verificado com `n_fft` de 512, 1024 e 2048: os três produzem
> `(128, 251)`.

---

## Pergunta 2 — O padding na entrada da CNN

> **RESPONDIDA em 17/09/2026 — opção D — espectrograma cru + agregação mascarada DENTRO da rede, com a máscara reduzida junto com o eixo do tempo. Ver «Respostas do orientador».**

> ⚠️ **Esta pergunta foi reescrita na v2.** A v1 afirmava que o padding é
> *"assimétrico entre as classes"*. **Isso está errado** e a afirmação está
> retirada. O detalhamento do erro está no *Registro de revisão*; o que segue já
> é a versão corrigida.

### O fato que continua valendo: metade do tensor é padding

RF e SVM agregam **somente os frames válidos**. No `features.csv` congelado,
`n_frames_validos` (de um total fixo de **251**) tem:

| estatística | valor |
|---|---:|
| mínimo | 8 |
| mediana | 120 |
| média | 132,43 |
| máximo | 250 |

**Em 53,07% dos áudios, mais da metade do tensor é padding.** Incluir esse
padding na agregação desloca as features em **30,3% (mediana) e até 47,0%
(máximo)** em delta relativo — este par de números vem do A/B controlado de 200
áudios em `checagem_mascaramento.json → parte_a_ab_controlado`, não do universo.

Este é o argumento de **validade de medida**, e ele não depende de viés nenhum:
uma estatística tirada de um vetor que é metade zeros descreve, em boa parte, a
formatação do vetor. Foi ele que justificou o mascaramento no Bloco 1, e é ele
que continua de pé aqui.

### O fato que a v1 errou: a QUANTIDADE de padding é simétrica

Medido no universo de 148.176 (não na amostra de 200):

| | bonafide | spoof | diferença |
|---|---:|---:|---:|
| fração de padding (média) | 47,11% | 47,25% | **0,14 p.p.** |
| `n_frames_validos` (mediana) | 124 | 120 | 4 frames |
| `prop_fala` (média) | 62,94% | 84,65% | 21,71 p.p. |

`prop_fala` **é** fortemente assimétrica — mas ela mede outra coisa, e
`preprocessamento.py:180-184` já dizia isso antes desta pergunta existir:

> *"`prop_fala` e este contador medem coisas DIFERENTES. prop_fala é a fração do
> áudio ORIGINAL que o VAD manteve; n_amostras_validas é o tamanho ABSOLUTO do
> que sobrou, comparado ao alvo de 4,0 s. Um áudio longo com prop_fala baixa pode
> acabar com MENOS padding que um curto com prop_fala alta. **Não inferir um do
> outro.**"*

A correlação medida entre as duas é **r = 0,145**. E `extrair_features.py:32-40`
já registrava a mesma advertência com os números do piloto.

**Consequência:** **não existe atalho de classe pela quantidade de zeros.** A
CNN não pode "acertar contando padding", porque as duas classes recebem
praticamente o mesmo. Qualquer justificativa de tratamento de padding tem de vir
da validade de medida, não de viés entre classes.

### A pergunta que sobra — e ela depende da cabeça da CNN

O paralelo real com RF/SVM não é *"a rede enxerga os zeros"*; é **onde a rede
agrega**:

- **Se a CNN terminar em *global average pooling*:** a representação agrupada é
  diluída pelo padding num fator que **varia por exemplo** (de 8 a 250 frames
  válidos). Isso é *exatamente* o problema que o Bloco 1 corrigiu no ramo
  clássico, reaparecendo na CNN.
- **Se terminar em *flatten* + densa:** o padding ocupa posições fixas na
  entrada, e a rede tem capacidade para aprender a descontá-lo.

Há ainda um fato que reduz o custo do problema: com a escala em dB e `top_db`
(P5), a região de padding vira um **platô exatamente constante** — medido: um
único valor distinto. A fronteira entre áudio e padding é, portanto,
trivialmente detectável a partir da própria imagem, sem canal extra.
*(Constante **dentro de cada exemplo**; o valor do platô é `max − top_db` e
portanto muda de exemplo para exemplo — ver a nuance do `top_db` na P5. Para o
que a P2 precisa — detectar a fronteira — isso é indiferente.)*

### As opções

| # | opção | como fica | custo | risco |
|---|---|---|---|---|
| **A** | cortar em `n_frames_validos` e **redimensionar** | cada exemplo esticado para a largura padrão | médio | **destrói a duração**: um áudio de 8 frames viraria 251 — fator de 31×, com 243 frames de conteúdo inventado |
| **B** | **máscara como 2º canal** | entrada com 2 canais: mel + máscara binária | médio-alto: muda a 1ª camada; **afeta a geração** | nenhum risco metodológico, mas informação em boa parte redundante com o platô de dB |
| **C** | espectrograma cru, **limitação declarada** | padding visível | zero | deixa a agregação da CNN por conta da sorte |
| **D** | **espectrograma cru + agregação mascarada DENTRO da CNN** — o *pooling* final soma apenas sobre os `n_frames_validos` do exemplo | entrada idêntica à C; a máscara age onde o Bloco 1 agiu: na **agregação** | baixo: `n_frames_validos` já está no `features.csv`, por `arquivo` | nenhum — é a tradução literal do que RF e SVM fazem |

**Recomendação: opção D.** Ela é a única que reproduz na CNN *a mesma operação*
que o Bloco 1 aprovou para o ramo clássico — mascarar a **agregação**, não a
entrada — usando **a mesma definição de frame válido** (`frames_validos()`).
Não fabrica escala temporal (A), não duplica o tensor (B) e não deixa a
diluição por conta da sorte (C). E tem uma consequência prática importante:

> **Sob a opção D, a P2 deixa de bloquear o B4.1.** O espectrograma gerado é o
> cru nos dois casos; o que muda é a camada de *pooling*, que é B4.2. Só a opção
> **B** obriga a decidir antes de gerar.

Se o orientador preferir a **B**, ela continua defensável — e aí a decisão
precisa vir antes da geração. A **C** só é aceitável com limitação declarada por
escrito, jamais por omissão.

**Contexto de apoio:** `RECOMENDACAO_MASCARAMENTO.md`,
`checagem_mascaramento.json`, `auditoria_decisoes_cnn.json → p2_*`, e a resposta
3 do apêndice do `DOSSIE_LOTE_UNICO.md`.

---

## Pergunta 3 — Normalização do espectrograma

> **RESPONDIDA em 17/09/2026 — estatísticas globais do treino, média e desvio POR FAIXA MEL, gravadas em `normalizacao_cnn.json`. Ver «Respostas do orientador».**

Duas famílias, com implicações diferentes:

- **Estatísticas globais (média/desvio por banda mel):** têm de ser calculadas
  **somente sobre o treino** — a subamostra de 30k —, **nunca** sobre o CSV
  inteiro nem sobre validação/teste. É exatamente o raciocínio do
  `StandardScaler` estar **dentro** do `Pipeline` do SVM
  (`src/models/treinar_svm.py:85-91`): o scaler é ajustado só no fold de treino.
  Estatísticas tiradas do conjunto inteiro são **vazamento**, e vazamento num
  artefato gerado uma vez só é irreversível.
- **Normalização por exemplo** (cada espectrograma normalizado por si): **não há
  vazamento**, mas isso muda o que a CNN vê — remove diferenças globais de
  energia entre gravações, que podem ser parte do sinal discriminativo. Se for
  essa a escolha, precisa estar **declarada**, e não apenas implementada.

**Recomendação:** estatísticas globais **calculadas na subamostra de 30k de
treino** e aplicadas a validação e teste.

> **Dependência crítica com a P5 — e a armadilha que a v1 caiu.** Estas duas
> perguntas não são independentes: **`ref=np.max` (recomendado pela v1 da P5)
> JÁ É normalização por exemplo.** Medido: multiplicar o áudio por 10× produz um
> espectrograma **idêntico dentro da precisão do float32** — a diferença máxima
> é da **ordem de 10⁻⁶ dB**, isto é, arredondamento, não sinal. O valor exato
> medido está em `auditoria_decisoes_cnn.json`, na chave
> `p5_escala_db.ref_max_diferenca_maxima_db_com_ganho_10x`, e o próprio script
> testa com `np.allclose`, não com igualdade (*ver as correções 1 e 6 da v4*).
> Ou seja, a v1 recomendava a família
> "global" na P3 e implementava a família "por exemplo" na P5, ao mesmo tempo.
> A recomendação da P5 foi trocada por causa disso — ver adiante. **Responder a
> P3 sem responder a P5 deixa a decisão pela metade.**
>
> Vale notar que `normalizar_amplitude` (normalização de **pico**, em
> `preprocessamento.py:62-77`) já roda antes, **para os dois ramos**: boa parte
> da variação de ganho já foi removida a montante, simetricamente. O que as
> estatísticas globais fazem é normalizar a escala **residual por banda**.

---

## Pergunta 4 — `n_mels = 128` com `n_fft = 512` produz filtros degenerados *(a mais grave)*

> **RESPONDIDA em 17/09/2026 — `n_fft = 1024` PRÓPRIO do espectrograma, `win_length = 400`, `hop_length = 256`, `n_mels = 128`, `fmin = 0`, `fmax = 8000`, `power = 2.0`, `center = true`. Ver «Respostas do orientador».**

**É a pergunta com maior chance de estragar o lote único sem dar nenhum aviso em
tempo de execução.**

O `config.yaml:178-179` pede `n_mels: 128` e `altura: 128`, e o bloco `features`
fixa `n_fft: 512` (`config.yaml:119`). A 16 kHz, `n_fft = 512` dá **257 bins de FFT com resolução de
31,25 Hz por bin**. Mas os primeiros filtros mel, na faixa de 0 a 8000 Hz com
128 bandas, têm **23,4 Hz de largura — mais estreitos que um único bin de FFT**.

Medição executada com `librosa.filters.mel(sr=16000, ...)` e reproduzida pelo
script de auditoria:

| `n_fft` | `n_mels` | bins FFT | suporte mínimo (bins por filtro) | filtros com ≤ 2 bins | **grupos** de filtros com o mesmo bin de pico |
|---:|---:|---:|---:|---:|---:|
| **512** | **128** | 257 | **1** | **61** | **12** |
| 512 | 80 | 257 | 2 | 19 | 0 |
| 512 | 64 | 257 | 2 | 1 | 0 |
| **1024** | **128** | 513 | 2 | **1** | **0** |
| 1024 | 80 | 513 | 4 | 0 | 0 |
| 2048 | 128 | 1025 | 5 | 0 | 0 |

**Leitura:** na configuração atual, **61 dos 128 filtros (48%) leem 2 bins de
FFT ou menos**, e há **12 grupos de filtros que compartilham o bin de pico** —
aqui os 12 grupos são exatamente **12 pares, ou seja, 24 filtros** (a última
coluna da tabela conta grupos, não filtros; nesta configuração nenhum grupo tem
mais de dois) —, isto é, são linhas redundantes. Quase metade da "imagem" que a CNN receberia é
interpolação dos mesmos poucos bins graves, não informação nova. O `librosa`
**não emite aviso** nesse regime (não há filtro totalmente vazio), então o erro
passaria silenciosamente e só apareceria como "a CNN não aprendeu nada nas
bandas baixas" — depois de gastar a semana gerando 74 mil tensores.

### As duas saídas

| # | opção | resultado | o que muda em relação ao ramo clássico |
|---|---|---|---|
| **A** | **`n_fft = 1024`**, mantendo `n_mels = 128`, `hop = 256`, `win_length = 400` | `(128, 251)` — 1 filtro degenerado, 0 picos duplicados | só o **zero-padding da FFT**. A janela de análise continua sendo os mesmos **400 amostras (25 ms)** do MFCC/ZCR/centróide, e o eixo temporal continua com **251** frames |
| **B** | **`n_mels = 64`**, mantendo `n_fft = 512` | `(64, 251)` — 1 filtro degenerado, 0 duplicados | paridade perfeita de FFT com o ramo clássico, mas `altura` passa de 128 para **64** e a CNN perde metade da resolução em frequência |

**Recomendação: opção A (`n_fft = 1024` só para o espectrograma).** O `n_fft` é
o tamanho da FFT, não a janela de análise: com `win_length = 400` fixo, subir o
`n_fft` apenas preenche a janela com zeros e interpola mais densamente o
espectro — **não altera o conteúdo do sinal analisado**, só a densidade com que
ele é amostrado em frequência. Portanto a promessa de "mesmo pré-processamento"
continua honesta nos eixos que importam (mesmo VAD, mesma duração de 4,0 s,
mesma janela de 25 ms, mesmo passo de 16 ms, mesmos 251 frames), e a diferença
— que é real — fica **declarada em uma linha** no texto. A opção B é defensável
e mais conservadora; o preço é `altura: 64`.

**Seja qual for a escolha, o `config.yaml` precisa passar a registrar
explicitamente o `n_fft` do espectrograma — e também `fmin`/`fmax`.** Hoje o
bloco `espectrograma:` não tem nenhuma dessas chaves e herdaria o `n_fft` do
bloco `features:` por acidente — precisamente o tipo de acoplamento silencioso
que a guarda de esquema do Bloco 2 existe para evitar. O `fmin`/`fmax` entram
pelo mesmo motivo: a decisão de **não** limitar a banda em 4 kHz foi ratificada
pelo orientador (`config.yaml:83-93`, porque 57,9% do dataset é banda estreita e
42,1% não é), hoje vale por *default* do librosa (0–8000 Hz), e uma decisão
ratificada não deve depender de default de biblioteca.

> **Acoplamento silencioso que já está no config, e que esta pergunta faz
> aparecer (acrescentado na v3).** `n_mels: 128` e `altura: 128` são **a mesma
> grandeza escrita duas vezes** (`config.yaml:178-179`): a altura da imagem *é*
> o número de bandas mel. Hoje isso é inofensivo porque os dois valores
> coincidem — mas se a resposta a esta pergunta for a **opção B** (`n_mels: 64`)
> e alguém esquecer de mexer em `altura`, o config passa a mentir sem que nada
> quebre nem avise. É exatamente o mesmo tipo de acoplamento que o parágrafo
> acima denuncia no `n_fft`. **Pedido junto com a resposta da P4:** que `altura`
> seja **derivada** de `n_mels` (e `largura` de `hop_length` e da duração), ou
> eliminada do config e calculada no gerador — nunca digitada em paralelo.

---

## Pergunta 5 — Escala de amplitude: potência linear ou **dB (log-mel)**?

> **RESPONDIDA em 17/09/2026 — Log-Mel em dB com `ref = 1.0` e `top_db = 80`; nunca `ref = np.max`. Ver «Respostas do orientador».**

**Ausente do `config.yaml`:** o bloco `espectrograma:` diz apenas `tipo: "mel"`,
o que não determina a escala de amplitude — e essa escolha muda radicalmente o
que a CNN enxerga.

`librosa.feature.melspectrogram` devolve **potência linear**, cuja faixa
dinâmica cobre várias ordens de grandeza. Alimentar a CNN com isso faz os poucos
frames de alta energia dominarem completamente o gradiente. A prática consagrada
é converter para dB com `librosa.power_to_db`.

**O argumento que liga isso ao TC I — e que vale ouro na banca:** o MFCC do ramo
clássico **já embute exatamente esse logaritmo** (o pipeline canônico é
mel → **log** → DCT). Se a CNN receber log-mel, os dois ramos compartilham a
mesma não-linearidade perceptual, e o único ponto em que eles divergem passa a
ser o que o trabalho quer de fato comparar: **DCT + agregação estatística
manual** (clássico) contra **convolução aprendendo o padrão tempo-frequência**
(CNN). Se a CNN receber potência linear, os ramos divergem em **dois** pontos ao
mesmo tempo, e a comparação deixa de isolar a variável de interesse.

### Qual `ref`? — a correção da v2

Para esse argumento de paridade valer, não basta "usar dB": é preciso usar **a
mesma conversão que o MFCC usa**. E `librosa.feature.mfcc` faz internamente
`S = power_to_db(melspectrogram(...))` — isto é, com os **defaults**
(`ref=1.0`, `top_db=80`), **não** com `ref=np.max`.

Prova executada pelo script de auditoria (ganho de 10× no áudio):

| medição | resultado | leitura |
|---|---:|---|
| deslocamento do MFCC em `c0` | **226,3** | o ganho sobrevive → dB **absoluto** |
| deslocamento nos demais coeficientes | **~10⁻⁵ ou menos** | o ganho fica isolado no `c0`; o resto é arredondamento de float32 |
| `power_to_db(ref=np.max)` com ganho 10× | **idêntico** | é normalização **por exemplo** |
| `power_to_db(ref=1.0)` com ganho 10× | **diferente** | preserva a energia global |
| padding sob `ref=1.0, top_db=80` | **1 valor distinto** | continua virando platô constante |

> **Por que a segunda linha não traz o número exato (v4).** Ela traz — no
> `auditoria_decisoes_cnn.json`
> (`p5_escala_db.mfcc_deslocamento_demais_coeficientes`) —, mas o valor **não é
> uma propriedade do pipeline**: é o resíduo de arredondamento do `float32`, e
> muda com a versão de `numpy`/BLAS da máquina. Reproduzido em dois ambientes
> com o **mesmo `librosa` 0.11.0**: 8,34 × 10⁻⁶ e 7,63 × 10⁻⁶. A leitura — sete
> ordens de grandeza abaixo do deslocamento do `c0`, logo ruído numérico — é que é
> estável, e é ela que o argumento usa. **Citar a ordem de grandeza, nunca o
> dígito**; o dígito vive no JSON, ao lado da versão que o produziu.

**Recomendação (trocada na v2): `librosa.power_to_db(S)` com os defaults —
`ref=1.0`, `top_db=80` — registrado como chave explícita no `config.yaml`.**
Três razões:

1. **É literalmente a mesma conversão que o ramo clássico aplica** dentro do
   MFCC. A paridade deixa de ser analogia e passa a ser identidade.
2. **Não conflita com a P3.** `ref=np.max` normalizaria cada exemplo pelo
   próprio máximo, tornando as estatísticas globais da P3 quase vazias; com
   `ref=1.0` sobra energia global para elas normalizarem.
3. **`top_db=80` continua resolvendo o problema que motivou a pergunta** — a
   faixa dinâmica fica limitada a 80 dB — e o padding continua sendo um platô
   constante, o que a P2 usa.

> **Nuance do `top_db` que precisa ficar declarada (acrescentada na v3;
> procedência fechada na v4).** O `top_db` do `librosa` corta em
> `max(S_dB) − 80` **do próprio exemplo**, não num piso absoluto. Consequências
> medidas — todas em `auditoria_decisoes_cnn.json → p5_escala_db.nuance_top_db`,
> sobre o mesmo sinal sintético que o script usa no resto da P5 (1,5 s de ruído
> + zeros até 4,0 s, semente 42):
>
> | medição | sem ganho | com ganho 10× | deslocamento |
> |---|---:|---:|---:|
> | máximo (dB) | 8,44 | 28,44 | **+20,000** |
> | piso / platô do padding (dB) | −71,56 | −51,56 | **+20,000** |
>
> - a **energia global sobrevive**, que é o ponto 2 acima: o ganho desloca o
>   espectrograma inteiro — máximo *e* piso — pelo mesmo número de dB. Então
>   `ref=1.0` + `top_db=80` **não** é normalização por exemplo, e a recomendação
>   continua de pé;
> - mas o **valor absoluto do platô de padding varia por exemplo** (é sempre
>   `max − 80`). Duas consequências a declarar: (i) a região de padding carrega,
>   indiretamente, a energia máxima do exemplo; (ii) as estatísticas globais por
>   banda da **P3** serão calculadas sobre um piso que difere de exemplo para
>   exemplo. Nenhuma das duas invalida a recomendação — mas a v2 descrevia o
>   platô como se fosse um valor fixo do dataset, e ele não é.
>
> Se o orientador quiser eliminar a nuance, a alternativa é `top_db=None` com
> um piso absoluto declarado (medido, no mesmo sinal: o mínimo vai a **−100 dB**,
> o *floor* do `power_to_db` — `nuance_top_db.db_min_sem_top_db`), ao custo de
> perder o corte de faixa dinâmica que motivou a pergunta. **Recomendação
> mantida: `top_db=80`, com esta nuance declarada por escrito.**

---

## Pergunta 6 — Em qual conjunto a CNN pára de treinar (*early stopping*)?

> **RESPONDIDA em 17/09/2026 — opção A refinada — split interno 27k/3k para o *early stopping* MAIS refit final nos 30k completos. Ver «Respostas do orientador».**

**Não bloqueia a geração dos espectrogramas, mas é decisão de protocolo.**

Assimetria que existe hoje e ninguém declarou:

- **RF e SVM** escolheram hiperparâmetros por `StratifiedKFold(5)` **dentro do
  treino** — RF em `ajustar_rf.py:5-9` (docstring) e `ajustar_rf.py:215-222`
  (código), SVM em `treinar_svm.py:132,158`. A validação foi usada **uma única
  vez**, e só para selecionar o limiar — conferido no código:
  `ajustar_rf.py:327-328` e `treinar_svm.py:227-228`, os dois chamando
  `selecionar_limiar(..., conjunto="validacao")`.

  > **Correção de citação (v3).** A v2 apontava aqui `treinar_rf.py:94-103`.
  > Está errado nos dois sentidos: essas linhas são o fatiamento do braço e do
  > split, e o `treinar_rf.py` **não seleciona limiar nenhum** — ele é o
  > baseline, com limiar **fixo em 0,50** (`treinar_rf.py:143-146`), justamente
  > o que a `NOTA_LIMIAR.md` motivou a substituir. Quem seleciona limiar para o
  > RF é o `ajustar_rf.py`. A afirmação continua verdadeira; a prova apontada
  > é que estava errada.
- **A CNN**, se usar a validação para *early stopping*, passa a usar a validação
  **duas vezes**: para decidir quando parar **e** para escolher o limiar. Isso é
  um privilégio que os modelos clássicos não tiveram, e enviesa a comparação
  central do trabalho a favor da CNN — exatamente o tipo de vantagem escondida
  que a `NOTA_LIMIAR.md §4` já se preocupou em impedir no caso do limiar.

| # | opção | simetria com RF/SVM | custo |
|---|---|---|---|
| **A** | fatiar um *dev* de dentro dos 30k de treino (ex.: 10%, estratificado, semente 42) e parar por ele | **simétrica**: a validação continua tocada uma vez só | a CNN treina em ~27k em vez de 30k — diferença a declarar |
| **B** | parar pela validação e **declarar a assimetria** | assimétrica, mas honesta | zero |
| **C** | nº fixo de épocas, sem *early stopping* | simétrica | risco de parar cedo demais ou tarde demais sem critério |

**Recomendação: A.** Mantém "mesmo ambiente experimental" literal, e o corte de
30k → 27k é pequeno e declarável. Como é só uma lista de IDs, **não afeta a
geração dos espectrogramas** — pode ser decidido junto, mas executado no B4.2.

> Se a **A** for escolhida, uma decorrência a registrar: as estatísticas de
> normalização da P3 devem sair dos **~27k** de treino efetivo, não dos 30k —
> senão o *dev* entra no cálculo do scaler e a simetria com o SVM se perde.

---

## Pergunta 7 — Peso das classes na *loss* da CNN *(NOVA na v2)*

> **RESPONDIDA em 17/09/2026 — opção A — `CrossEntropyLoss` com 2 saídas e peso maior para bonafide (classe 0), razão 9:1, sem over/undersampling. Ver «Respostas do orientador».**

**Estava faltando, e a `NOTA_LIMIAR.md §4` já a nomeia explicitamente:** a regra
de protocolo vale para *"RF, SVM (`class_weight`) e **CNN (peso na loss)**"*. As
perguntas 3 a 6 tratam de entrada, parada e limiar; ninguém tinha perguntado
como a CNN trata o desbalanceamento.

O desbalanceamento medido é **9,00 : 1 em todos os conjuntos**, inclusive na
subamostra que a CNN vai usar:

| conjunto | n | spoof | bonafide | razão |
|---|---:|---:|---:|---:|
| treino completo | 103.723 | 93.352 | 10.371 | 9,00:1 |
| **subamostra 30k (braço principal)** | 30.000 | 27.000 | 3.000 | **9,00:1** |
| validação | 22.226 | 20.004 | 2.222 | 9,00:1 |
| teste | 22.227 | 20.004 | 2.223 | 9,00:1 |

> **Procedência (mudou na v4).** Até a v3 esta tabela era conferência de mão —
> era o **único** número do documento a sustentar uma recomendação de protocolo
> sem nenhuma trava por trás. Agora ela é a **trava 11**: o script recomputa as
> contagens (`features.csv` dá o rótulo, `split.csv` o conjunto,
> `subamostra_30k.csv` a `classe_binaria`), publica em
> `auditoria_decisoes_cnn.json → p7_desbalanceamento` e **falha** se a razão
> deixar de ser 9,00:1 em qualquer um dos quatro. As razões exatas são 9,001 /
> 9,003 / 8,999 / 9,000 — divisão de inteiros não dá 9 redondo nestes tamanhos,
> por isso a trava compara com 2 casas.

RF e SVM tratam isso com `class_weight` (e o RF aprendeu, na `NOTA_LIMIAR.md
§2`, que configurá-lo não basta — com folhas puras ele era neutralizado no
`predict_proba`). A CNN precisa do equivalente declarado.

| # | opção | análogo clássico | risco |
|---|---|---|---|
| **A** | ***loss* ponderada** (peso ∝ inverso da frequência) | é o análogo literal de `class_weight='balanced'` | nenhum; a composição da época fica intacta |
| **B** | ***sampler* balanceado** (sobre-amostrar bonafide) | mais próximo de `balanced_subsample` | muda a composição efetiva da época; o bonafide é visto ~9× por época, com risco de memorização |
| **C** | sem ponderação, confiando só na seleção de limiar | nenhum | a CNN seria o único modelo sem tratamento de desbalanceamento — assimetria direta na comparação central |

**Recomendação: A (*loss* ponderada), com o peso registrado no
`config.yaml`.** É o análogo literal do que RF e SVM fazem, não altera quantas
vezes cada exemplo é visto, e mantém a seleção de limiar como o **segundo**
mecanismo, compartilhado pelos três modelos. A **C** deve ser recusada: deixaria
a CNN como o único dos três sem tratamento de desbalanceamento, e a diferença
apareceria no resultado final sem que se soubesse se é do modelo ou do protocolo.

**Não bloqueia o B4.1** — é B4.2 —, mas entra aqui porque é decisão de protocolo
e a consulta ao orientador está sendo feita agora.

---

## Regra de implementação (não é pergunta — é condição para o B4.1)

**O gerador de espectrograma tem de CHAMAR `preprocessar_audio` de
`src/data/preprocessamento.py`, não reimplementar VAD e padding.**

Se o carregamento, o VAD e o *padding* forem reescritos no script novo, os dois
ramos divergem **silenciosamente** — mesma intenção, código diferente — e a
frase "mesmo pré-processamento para os três modelos", que é a base da pergunta
de pesquisa, vira falsa sem que nada quebre nem avise. É a mesma lógica que
levou `colunas_features` a importar `COLUNAS_DIAGNOSTICO` do módulo que escreve
as colunas em vez de manter uma cópia (`src/data/split.py:251-255`): *a fonte
única existe para impedir divergência silenciosa entre quem grava e quem lê.*

O mesmo vale para a definição de frame válido: se a opção **B** ou **D** da P2
for escolhida, a máscara tem de vir de `frames_validos()`
(`src/features/extrair_features.py:119-135`), **a mesma função** que o ramo
clássico usa — nunca de uma reimplementação.

O script de geração deve, além disso, gravar um `espectrogramas.meta.json`
assinando a definição vigente (n_mels, n_fft, hop, win_length, fmin, fmax,
escala de amplitude, normalização, tratamento do padding, semente, commit git) —
o mesmo mecanismo de `features.meta.json` que salvou o Bloco 2.

---

## Nota de engenharia (não é decisão do orientador — é planejamento)

### Escopo: gerar 74.453, não 148.176

O braço de referência é **RF-only** (RF treinado nos 103.723 do treino completo)
e nunca precisa de espectrograma.

| conjunto | n |
|---|---:|
| subamostra de treino (braço principal) | 30.000 |
| validação (completa) | 22.226 |
| teste (completo, lacrado) | 22.227 |
| **total necessário** | **74.453** |
| universo eval inteiro | 148.176 |

Gerar os 148.176 seria produzir **73.723** tensores que nada consome.

### Armazenamento (128 × 251 × 4 bytes = 128.512 B = **125,5 KiB** por áudio)

| escopo | float32 | float16 |
|---|---:|---:|
| 74.453 áudios (necessário) | **9,57 GB** (8,91 GiB) | 4,78 GB (4,46 GiB) |
| 148.176 áudios (universo) | 19,04 GB (17,73 GiB) | 9,52 GB (8,87 GiB) |

*(Se a opção B da P2 for escolhida, o segundo canal — a máscara — pode ser
gravado como `uint8` ou reconstruído em tempo de carga a partir de
`n_frames_validos`, que já está no `features.csv`; nesse caso o custo extra em
disco é praticamente nulo. **Sob a opção D, recomendada, o custo extra é zero** —
nada muda no tensor. Se a opção B da P4 for escolhida (`n_mels=64`), todos os
valores acima caem pela metade.)*

### Armadilha do `.gitignore` — **já corrigida em 04/09/2026**

O `.gitignore` cobria `data/raw/`, `data/processed/*` (com exceções explícitas)
e `data/features/*.csv`, mas **não** uma pasta nova `data/espectrogramas/` — um
`git add .` distraído tentaria versionar ~9,6 GB. A regra foi acrescentada
preventivamente (não depende de decisão do orientador), versionando apenas o
`espectrogramas.meta.json`.

---

## O que se pede ao orientador

Resposta às **sete perguntas** — de preferência anotada neste próprio arquivo,
para que a decisão fique versionada no repositório junto com a evidência que a
motivou, exatamente como foi feito no Bloco 2.

**Ordem de urgência real:**

| ordem | pergunta | bloqueia o B4.1? |
|---|---|---|
| 1 | **P4** — `n_mels`/`n_fft` (+ `fmin`/`fmax`) | **sim** — e é a que falha em silêncio |
| 2 | **P5** — escala dB e qual `ref` | **sim** |
| 3 | **P3** — normalização (depende da P5) | **sim** |
| 4 | **P1** — largura 251 | **sim** (uma linha) |
| 5 | **P2** — padding | **só se a resposta for a opção B**; sob a D recomendada, é B4.2 |
| 6 | **P6** — *early stopping* | não — é B4.2 |
| 7 | **P7** — peso na *loss* | não — é B4.2 |

---

## Registro de revisão

### v5 — 17/09/2026 (transcrição das respostas — o documento fecha)

Motivo: **as sete perguntas foram respondidas e aprovadas pelo orientador**, que
pediu textualmente que a decisão ficasse anotada **neste próprio arquivo**, «para que
a decisão fique versionada no repositório junto com a evidência que a motivou,
exatamente como foi feito no Bloco 2». Então não se criou documento novo — editou-se
este. **Nenhuma pergunta, evidência, opção ou recomendação foi apagada ou alterada:**
a v5 só acrescenta as respostas e marca as perguntas como respondidas. Este é o marco
**B4.0** (`docs/execucao/B4.0_fechamento_metodologico.md`).

| # | o que mudou | por quê |
|---|---|---|
| **1** | **Status: `BLOQUEADO` → `RESPONDIDO E FECHADO em 17/09/2026`.** | O aviso «enquanto este documento não for respondido, o bloco `espectrograma:` não é alterado e o B4.1 não começa» cumpriu sua função e passou a ser **falso**: as decisões saíram e o config foi aplicado no mesmo dia. Manter o aviso deixaria o repositório com duas afirmações contraditórias sobre o mesmo bloco. O cabeçalho passa a registrar as três datas — aberto (03/09), revisado (17/09, v4) e fechado (17/09, v5). |
| **2** | **Seção «Respostas do orientador — 17/09/2026» inserida**, logo após o cabeçalho, com a tabela das sete decisões, as duas formulações proibidas e os **quatro refinamentos obrigatórios**. | A tabela é a decisão em si. As formulações proibidas e os refinamentos vieram **junto** com a aprovação e não são decisões novas: são a forma aprovada de executar as sete. Dois deles corrigem implementações que pareceriam certas e estariam erradas — a máscara que não acompanha a redução do eixo do tempo (P2) e o `BCEWithLogitsLoss(pos_weight=9)`, que **inverteria** a intenção porque spoof é a majoritária (P7). Os outros dois criam obrigações de artefato e de protocolo: `normalizacao_cnn.json` versionado (P3) e o **refit final nos 30k** (P6), que substitui a recomendação original da P6 — sozinha, ela deixaria a CNN final treinada em 27k contra os 30k de RF e SVM. |
| **3** | **Cada `## Pergunta N` recebeu uma linha `> **RESPONDIDA em 17/09/2026 — <decisão>**`** — e **só** isso: o corpo das sete seções está intacto. | A alternativa era apagar a pergunta e deixar só a resposta. Isso destruiria exatamente o que fez este documento valer a pena: a evidência medida é o que sustenta a decisão numa arguição, e sem ela a decisão volta a ser opinião. A linha de marcação resolve a leitura (ninguém lê uma pergunta já respondida achando que está aberta) sem custo nenhum de conteúdo. |
| **4** | **Registro de revisão: esta entrada.** | Mesma disciplina das v2–v4. |

**Aplicado no mesmo commit, fora deste arquivo:**

| arquivo | o que mudou |
|---|---|
| `config/config.yaml` | bloco `espectrograma:` completo, com os 17 parâmetros metodológicos e o sub-bloco `defaults_librosa_registrados`; o bloco de comentário «⚠️ BLOCO CONGELADO ATÉ A RESPOSTA DO ORIENTADOR» foi **removido** e substituído pela justificativa de cada decisão — inclusive as duas formulações proibidas e a ressalva do `top_db`. `largura: 256` → **251**, e o espectrograma passa a ter `n_fft` **próprio** (1024), matando o acoplamento silencioso com o bloco `features:` que a P4 existe para impedir. |
| `config/config.yaml`, bloco `features:` | **nenhum valor alterado** — as features estão congeladas desde 30/08 e o `features.meta.json` assina `n_fft: 512`; mexer ali invalidaria a guarda de retomada e reabriria o Bloco 2. Só um comentário foi acrescentado: a **limitação registrada** de que a filterbank Mel interna do `librosa.feature.mfcc` herda o mesmo regime degenerado da P4 com o `n_fft=512` congelado. |

> **A limitação da filterbank foi medida antes de ser escrita**, não deduzida. Com
> `librosa` 0.11.0, `librosa.filters.mel(sr=16000, n_fft=512, n_mels=128, fmin=0,
> fmax=8000, htk=False, norm="slaney")` dá **61 filtros com ≤ 2 bins ativos** e
> **12 pares** com o mesmo bin de pico — os mesmos números da P4 —, e o `n_mels`
> default de `librosa.feature.mfcc` é **128**, confirmado por execução
> (`melspectrogram` sem `n_mels` devolve `(128, 251)`). Ou seja, o ramo clássico cai
> nessa filterbank **de fato**, não só em tese. Ela vai ao texto como **LIMITAÇÃO, não
> como experimento**, pela regra de escopo de 17/09: o impacto é atenuado pela DCT do
> MFCC, que retém 20 dos 128 coeficientes, e a comparação central exige que os dois
> ramos partam do **mesmo áudio pré-processado** — e partem, por construção
> (`preprocessar_audio` é o mesmo). A diferença de representação é o **objeto de
> estudo**, não um confundidor. Já consta como **limitação 7** da lista do
> `docs/execucao/B6_redacao_e_fechamento.md`.

**O que a v5 NÃO fez, de propósito:** não criou `DECISOES_CNN_FECHADAS.md` nem
qualquer outro documento de consulta (duas fontes = divergência garantida, e o
orientador pediu a anotação no arquivo existente); não tocou em `src/models/*`; não
gerou tensor nenhum; e não decidiu arquitetura nem hiperparâmetros de treino, que são
B4.4/B4.5. A partir de **21/09 (B4.2)** a definição do espectrograma fica
**congelada** no mesmo sentido que as features, com a assinatura vigente em
`data/espectrogramas/espectrogramas.meta.json`.

### v4 — 17/09/2026 (as três pontas soltas da v3)

Motivo: a v3 foi revisada por um leitor externo ao lote, que conferiu as seis
correções contra o repositório — todas verdadeiras — e apontou **três coisas que
a própria v3 deixou em aberto**. Ao fechá-las, o script de auditoria foi
**executado de ponta a ponta num segundo ambiente**, e a execução trouxe um
sexto achado que nenhuma leitura pegaria (item 6). Nenhum dos seis muda
pergunta, opção, recomendação ou ordem de urgência.

| # | o que mudou | por quê |
|---|---|---|
| **1** | **P3: "byte a byte idêntico" → "idêntico dentro da precisão do float32", com o número.** | Resíduo de exagero da v1 que passou pela auditoria da v2 e pela conferência da v3. `ref=np.max` com ganho 10× deixa **6,7 × 10⁻⁶ dB** de diferença máxima — arredondamento, não sinal. O próprio script sempre testou com `np.allclose`, e não com igualdade (`auditar_decisoes_cnn.py`, bloco da P5), então o documento afirmava mais forte do que a sua própria prova. A diferença passa a ser **publicada** no JSON, para que a frase possa citar a ordem de grandeza em vez de arredondá-la para zero. |
| **2** | **P5: os quatro números da nuance do `top_db` ganharam procedência.** | A v3 acrescentou `−51,56`, `8,44`, `28,44` e `−100` medindo à mão e **não declarou** — e a própria v3 enumerava, na seção de procedência, o que era conferência de mão (P7, deltas 30,3/47,03, GiB/float16). Eram números novos fora das duas listas: nem travados, nem publicados, nem declarados. Era o mesmo defeito que a v3 estava corrigindo, um nível acima. Agora saem do script (`p5_escala_db.nuance_top_db`), sobre o mesmo sinal sintético do resto da P5, e o documento cita a chave. |
| **3** | **P7: a tabela de desbalanceamento virou a trava 11.** | Era o único número do documento que sustentava **uma recomendação de protocolo** (a *loss* ponderada) e não tinha trava nenhuma — a v3 chegou a declarar isso por escrito, e declarar é melhor que esconder, mas não é o suficiente aqui. Conferência de mão não sobrevive a uma re-geração do `split.csv` ou da subamostra, que é justamente o que o Bloco 5 pode disparar. Custo: um `groupby` sobre dados que o script já tinha carregado. |
| **4** | **`TOL = 0.05` removida de `auditar_decisoes_cnn.py`.** | Constante morta: definida, nunca usada, e era ela que dava verossimilhança à promessa falsa da v2 ("todo número é recomputado"). A alternativa — fazer o script ler o `.md` e comparar valores com tolerância — foi **descartada de propósito**: acoplaria a trava à formatação da prosa, que muda a cada revisão. O caminho adotado é o inverso, e é o que o item 3 faz: cada afirmação que sustenta decisão vira asserção sobre o **artefato**, e o documento cita o JSON. A remoção ficou registrada em nota no lugar onde a constante estava, para que a pergunta "por que não parseia o `.md`?" tenha resposta no código. |
| **5** | **Seção de procedência: 10 → 11 travas, e a lista do que fica fora ficou fechada.** | Com a P7 travada, sobram **exatamente duas** exceções (deltas do piloto e GiB/float16 do universo), as duas identificadas onde aparecem. Ficou escrita a regra que passa a valer: afirmação nova sem contrapartida no JSON ou ganha bloco no script, ou não entra no documento. |
| **6** | **P5: o `8,34 × 10⁻⁶` deixou de ser citado como dígito.** *(achado ao executar o script, não ao lê-lo)* | O script foi rodado de ponta a ponta num **segundo ambiente**, com o mesmo `librosa` 0.11.0 mas `numpy`/BLAS diferentes: `mfcc_deslocamento_demais_coeficientes` veio **7,63 × 10⁻⁶** em vez de 8,34 × 10⁻⁶. Ou seja, esse número — e o `ref_max_diferenca_maxima_db_com_ganho_10x` da correção 1, que é da mesma natureza — **não é propriedade do pipeline**: é resíduo de arredondamento de `float32` e depende da máquina. A v3 gastou uma correção inteira para fazer o documento citar o dígito do JSON, o que era certo quanto à disciplina e errado quanto à natureza do número. O documento passa a citar a **ordem de grandeza** e a apontar a chave do JSON para o valor exato. Nada disso toca a leitura: o ganho continua isolado no `c0` por sete ordens de grandeza. |

**Reconferido na v4 e mantido sem alteração:** tudo o que a v3 corrigiu —
as citações `ajustar_rf.py:327-328`, `treinar_svm.py:227-228`,
`ajustar_rf.py:215-222`, `treinar_svm.py:132,158`, `treinar_rf.py:143-146`,
`config.yaml:178-180` e `:119`, `extrair_features.py:119-135` e `:97`,
`split.py:251-255`, `preprocessamento.py:62-77` e `:180-184` — todas conferidas
de novo contra a cópia de trabalho; o `8,34 × 10⁻⁶`; a tabela de filtros mel
(suporte mínimo 1, 61 filtros com ≤ 2 bins, 12 grupos — e a distribuição de
tamanhos dos grupos é `{2: 12}`, confirmando os 12 pares = 24 filtros); o
espaçamento de **23,38 Hz** contra **31,25 Hz** por bin; os 251 frames para
`n_fft` 512/1024/2048; a leitura do fonte do `librosa` 0.11 confirmando
`power_to_db(melspectrogram(...))` com defaults em `librosa.feature.mfcc`; e a
tabela da P7, recontada de `split.csv` + `labels.csv` + `subamostra_30k.csv`
antes de virar trava.

> **Nota de execução.** As travas 11 e os campos novos do JSON só existem depois
> de rodar `python -m scripts.auditar_decisoes_cnn` uma vez. Rodar reescreve o
> `auditoria_decisoes_cnn.json` — o que é o comportamento desejado agora que a
> conferência terminou —, e o JSON regenerado deve entrar no mesmo *commit* que
> estas mudanças, senão o documento cita chaves que o artefato não tem.

### v3 — 17/09/2026 (conferência de procedência)

Motivo: antes de enviar, a v2 foi reconferida contra o repositório — código,
`config.yaml`, `auditoria_decisoes_cnn.json` e `checagem_mascaramento.json` —, e
os cálculos de `librosa` foram refeitos de forma independente. **Nenhuma
pergunta, opção, recomendação ou ordem de urgência mudou.** Seis defeitos de
citação e procedência apareceram, e duas lacunas técnicas foram acrescentadas.

| # | o que mudou | por quê |
|---|---|---|
| **1** | **P6: citação `treinar_rf.py:94-103` substituída por `ajustar_rf.py:327-328`.** | Era a única citação errada por **conteúdo**, não por deslocamento. As linhas apontadas são o fatiamento do braço/split, e o `treinar_rf.py` nem seleciona limiar — é o baseline com limiar fixo em 0,50 (`treinar_rf.py:143-146`). A afirmação ("a validação foi tocada uma vez só") continua verdadeira e agora aponta para o código que de fato a prova. |
| **2** | **P5: `7,6 × 10⁻⁶` corrigido para `8,34 × 10⁻⁶`** (duas ocorrências: tabela da P5 e item 3 do registro da v2). | O valor publicado não batia com o `auditoria_decisoes_cnn.json` (`8.34e-06`), que o próprio documento cita como procedência. Reproduzido de forma independente: **8,344 × 10⁻⁶**. A leitura não muda (é arredondamento de float32 nos dois casos), mas um número que o documento afirma ter recomputado tem de bater com o artefato. *(Revisto na v4: a disciplina estava certa, o alvo não — esse dígito varia com `numpy`/BLAS. Ver a correção 6 da v4.)* |
| **3** | **Três citações de linha atualizadas para a cópia de trabalho:** `config.yaml:145` → **:180**; `config.yaml:141-145` → **:178-179**; `extrair_features.py:102-118` → **:119-135** (duas ocorrências); `extrair_features.py:80` → **:97**. | Estavam certas no `HEAD`, mas as edições **não commitadas do mesmo lote** que gerou a v2 as deslocaram — inclusive o bloco de aviso acrescentado ao próprio `config.yaml`. A v2 declarava tê-las conferido; a conferência tinha sido feita contra o `HEAD`, não contra o arquivo real. |
| **4** | **Nota de procedência: `parte_b_runner` é n=500, não n=200.** | A v2 atribuía a mediana de 121 à "amostra de n=200 do piloto (`parte_b_runner`)". O `checagem_mascaramento.json` tem **duas** amostras: `parte_a_ab_controlado` (n=200 balanceado — de onde saem os deltas 30,3%/47,03% e as frações de padding por classe) e `parte_b_runner` (n=500: 50 bonafide, 450 spoof — de onde sai a mediana 121). A v2 trocou as duas. Conclusões inalteradas. |
| **5** | **Alcance da trava de auditoria declarado com precisão.** | A v2 dizia que "todo número é recomputado" e que o script "falha se **qualquer** afirmação deixar de valer". Não é o caso: `TOL` é definido em `auditar_decisoes_cnn.py:65` e **nunca usado** — o script não compara valor nenhum com o `.md`. Ele trava **dez** condições (tabela na seção de procedência) e publica o resto sem travar; a tabela da **P7**, os deltas 30,3%/47,03% e os GiB/float16 do universo não são tocados por ele. A P7 foi conferida à mão (`split.csv` + `features.csv` + `subamostra_30k.csv`) e bate exatamente. *(Atualizado na v4: a P7 deixou de ser conferência de mão e virou a trava 11; e a constante `TOL` foi removida. São **onze** travas a partir da v4.)* |
| **6** | **P4: o rótulo "filtros com pico duplicado" virou "grupos".** | O script conta **grupos** de filtros que compartilham o bin de pico, não filtros. Nesta configuração os 12 grupos são 12 pares (24 filtros), então a prosa "12 pares" estava certa e só o cabeçalho da tabela estava mal nomeado. |
| **7** | **P4: acrescentada a duplicação `n_mels` × `altura`** no `config.yaml`. | São a mesma grandeza digitada duas vezes. Se a resposta for a opção B (`n_mels: 64`) e `altura` ficar em 128, o config mente sem quebrar — o mesmo acoplamento silencioso que a própria P4 denuncia no `n_fft`, e que a v2 não viu no arquivo que estava analisando. |
| **8** | **P5: acrescentada a nuance do `top_db`** (e referência cruzada na P2). | `top_db` corta em `max − 80` **do próprio exemplo**. Medido: um ganho de 10× desloca máximo e piso por igual (+20 dB), logo a energia global sobrevive e a recomendação continua de pé; mas o valor absoluto do platô varia por exemplo, o que a v2 descrevia como se fosse fixo. Interação P5 × P3 que faltava declarar. |

**Reconferido na v3 e mantido sem alteração:** as sete perguntas, todas as
opções e todas as recomendações; a tabela de filtros mel da P4, recalculada de
forma independente (23,38 Hz por filtro mel contra 31,25 Hz por bin de FFT,
suporte mínimo 1, 61 filtros com ≤ 2 bins, 12 pares com o mesmo pico); o
deslocamento de **226,27** no `c0`; a confirmação, lida no fonte do `librosa`
0.11 instalado, de que `librosa.feature.mfcc` aplica
`power_to_db(melspectrogram(...))` com os **defaults**; a tabela inteira de
desbalanceamento da **P7** (103.723 / 93.352 / 10.371; 30.000 / 27.000 / 3.000;
22.226 / 20.004 / 2.222; 22.227 / 20.004 / 2.223); a simetria de padding da P2;
e a regra do `.gitignore`, conferida no diff.

### v2 — 04/09/2026 (auditoria numérica)

Motivo: antes de enviar, cada número do documento foi recomputado sobre os
artefatos reais. Um erro metodológico e três imprecisões apareceram.

| # | o que mudou | por quê |
|---|---|---|
| **1** | **P2: retirada a afirmação de que o padding é "assimétrico entre as classes".** | A v1 inferia assimetria de **padding** a partir da média de **`prop_fala`** (62,9% bonafide × 84,6% spoof). São grandezas diferentes. Medida a fração de padding de verdade no universo: **47,11% × 47,25% — 0,14 p.p. de diferença**, com r = 0,145 entre as duas. `preprocessamento.py:180-184` (*"Não inferir um do outro"*) e `extrair_features.py:32-40` (*"CUIDADO com a intuição fácil"*) já advertiam contra exatamente essa inferência; e o próprio `checagem_mascaramento.json` citado como fonte traz `fracao_padding_media_bonafide: 0.448` / `..._spoof: 0.4523` duas linhas abaixo do `prop_fala` que a v1 usou. |
| **2** | **P2: recomendação passou de B para D** (agregação mascarada dentro da CNN). | Com o argumento de assimetria retirado, o que sobra é validade de medida — e ela se resolve mascarando a **agregação**, não a entrada, que é literalmente o que o Bloco 1 fez para RF/SVM. Custo zero em disco, e deixa de bloquear o B4.1. Acrescentada a distinção *global average pooling* × *flatten*, que é o que de fato determina a exposição da CNN ao padding, e o achado de que `top_db` transforma o padding em platô constante (máscara recuperável da própria imagem). |
| **3** | **P5: recomendação passou de `ref=np.max` para `ref=1.0` (defaults do `power_to_db`).** | Duas razões. (a) `librosa.feature.mfcc` aplica `power_to_db` com os **defaults**, então a paridade de não-linearidade com o ramo clássico — o argumento central da P5 — exige `ref=1.0`; medido: ganho de 10× desloca o MFCC em 226,3 no `c0` e 8,34×10⁻⁶ nos demais. (b) `ref=np.max` **é** normalização por exemplo (invariante a ganho, verificado), o que **contradizia a recomendação da P3** na mesma versão do documento. |
| **4** | **P7 acrescentada** (peso das classes na *loss*). | A `NOTA_LIMIAR.md §4`, citada pelo próprio documento, nomeia *"CNN (peso na loss)"* como parte da mesma regra de protocolo, e o desbalanceamento de 9,00:1 vale também na subamostra de 30k. Nenhuma pergunta cobria isso. |
| **5** | **Regra de implementação: caminho corrigido.** | Dizia `preprocessar_audio` de `src/features/extrair_features.py`; a função mora em **`src/data/preprocessamento.py`** (`extrair_features.py:97` apenas a importa). Erro de citação numa seção sobre fonte única. |
| **6** | **P4: `fmin`/`fmax` acrescentados** à exigência de registro no config. | Pelo mesmo motivo do `n_fft`: a decisão de não limitar a banda em 4 kHz foi ratificada pelo orientador e hoje depende de um default do librosa. |
| **7** | **Nota de divergência 120 × 121** na procedência. | O repositório cita 121 (amostra n=200) e este documento cita 120 (universo). Ambos certos; a divergência estava sem explicação. |
| **8** | **Criado `scripts/auditar_decisoes_cnn.py`.** | A maior parte dos números passa a ser reproduzível por execução. *(Ressalva acrescentada na v3: o script trava dez condições, não "qualquer afirmação" — ver a tabela em «Até onde a trava alcança».)* |

**Conferido e mantido sem alteração** (recomputado, bate exatamente): MD5 e
148.176 linhas do `features.csv`; `n_frames_total` = 251 único; estatísticas de
`n_frames_validos` (8 / 120 / 132,43 / 250) e os 53,07%; deltas 30,3% e 47,03%;
**toda a tabela de filtros mel da P4**; independência do nº de frames em relação
ao `n_fft`; contagens 103.723 / 22.226 / 22.227, 74.453 e 73.723; armazenamento
(125,5 KiB, 9,57 GB / 8,91 GiB); e as citações de linha `treinar_svm.py:85-91`,
`ajustar_rf.py:5-9`, `split.py:251-255`, `preprocessamento.py:62-77`,
`preprocessamento.py:180-184`, `extrair_features.py:32-40` e `NOTA_LIMIAR.md §4`
— todas reconferidas na v3 e mantidas.

> **Ressalva da v3 sobre esta linha.** A v2 afirmava ter conferido **todas** as
> citações de linha. Três não estavam valendo: `extrair_features.py:102-118` e
> `config.yaml:145`/`141-145` estavam certas no último *commit*, mas já haviam
> sido deslocadas pelas edições **não commitadas do mesmo lote** que produziu a
> v2 — o próprio bloco de aviso acrescentado ao `config.yaml` empurrou as linhas
> que o documento citava. E `treinar_rf.py:94-103` estava errada por conteúdo,
> não por deslocamento (ver P6). Todas corrigidas na v3, contra a **cópia de
> trabalho**. Lição registrada: citação de linha conferida contra o `HEAD` não
> vale quando o documento e o código citado mudam no mesmo lote.

### v1 — 03/09/2026

Versão original, com seis perguntas. Substituída pela v2 (ver acima).
