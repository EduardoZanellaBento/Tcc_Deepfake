# Decisões pendentes — definição do espectrograma da CNN (Bloco 4)

> **Status:** BLOQUEADO, aguardando decisão do orientador.
> **Aberto em:** 03/09/2026 · **Aluno:** Eduardo Zanella Bento · **Orientador:** Prof. Anderson Fola.
> **Enquanto este documento não for respondido:** o bloco `espectrograma:` do
> `config/config.yaml` NÃO é alterado, nenhum script de geração de espectrograma
> é criado, e o B4.1 não começa.

## Por que isto está por escrito, e não num chat

Gerar os espectrogramas é o **"lote único" do Bloco 4**: caro, demorado e não
repetível dentro do cronograma. É a mesma situação do Bloco 2, em que a decisão
de mascarar o padding foi registrada **antes** da extração — e foi isso que
permitiu, depois, defender a escolha com um documento em vez de com memória.
Errar a definição do espectrograma agora custa a semana inteira de CPU.

São três perguntas. Cada uma traz a evidência conferida no repositório, as
opções com custo e risco, e uma recomendação.

---

## Pergunta 1 — Largura do espectrograma: **251** ou **256**?

**O que o `config.yaml:145` diz hoje:** `largura: 256`.

**O que o pipeline produz de fato: 251 frames.** Conferido em três fontes
independentes:

| fonte | evidência |
|---|---|
| aritmética | `1 + 64000 // 256 = 251` (4,0 s × 16 kHz = 64.000 amostras, `hop_length = 256`, `center=True`) |
| código | `frames_validos()` em `src/features/extrair_features.py:102-118` |
| dados | `n_frames_total` no `features.csv` congelado tem **um único valor: 251**, nas 148.176 linhas (`DOSSIE_LOTE_UNICO.md` §6, e confirmado de novo em 03/09/2026) |

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

---

## Pergunta 2 — A assimetria de mascaramento *(a mais importante das três)*

**O fato, medido e congelado:** RF e SVM agregam **somente os frames válidos**.
No `features.csv` congelado, `n_frames_validos` tem **mediana 120 de 251** —
ou seja, em metade dos áudios **mais da metade do tensor é padding** (média
132,4; mínimo 8; máximo 250). Incluir esse padding na agregação desloca as
features em **30,3% (mediana) e até 47,0% (máximo)** em delta relativo
(`checagem_mascaramento.json → parte_a_ab_controlado`), e a distorção é
**assimétrica entre as classes**, porque o VAD mantém 64,0% do áudio bonafide
contra 85,4% do spoof.

**A CNN, se receber o espectrograma cru, vai enxergar esses zeros.** A pergunta
de banca — *"você mascarou o padding para os modelos clássicos e não para a CNN;
a comparação é justa?"* — **não tem resposta hoje**.

### As três opções, em ordem de custo

| # | opção | como fica | custo | risco |
|---|---|---|---|---|
| **A** | **cortar em `n_frames_validos` e redimensionar** para largura fixa | cada exemplo entra na CNN só com áudio real, esticado para a largura padrão | médio: exige `n_frames_validos` por arquivo (já está no `features.csv`) + uma interpolação por exemplo | a escala temporal passa a variar por exemplo (um áudio de 8 frames vira 251): distorce a duração, que é informação real |
| **B** | **máscara como segundo canal** | entrada com 2 canais: mel + máscara binária de validade | médio-alto: dobra o tensor em memória e muda a primeira camada da CNN | nenhum risco metodológico; a rede recebe explicitamente onde o áudio acaba, e pode aprender a ignorar |
| **C** | **declarar a assimetria como limitação** | espectrograma cru, com o padding visível | zero | o argumento passa a ser "a convolução aprende a ignorar região constante" — plausível, mas **não medido**; e a assimetria bonafide × spoof do padding é exatamente do tipo que vira atalho estatístico |

**Recomendação: opção B (máscara como segundo canal).** Ela é a única que
preserva a duração real *e* informa a rede sobre a validade, sem inventar
escala temporal (A) nem deixar a assimetria por conta da sorte (C). O custo é
de engenharia, não de método, e cai inteiro dentro do B4.1 — que ainda não
começou. Se o custo em disco/memória for considerado proibitivo, a segunda
escolha é **C com limitação declarada por escrito**, jamais C por omissão.

**Contexto de apoio:** `RECOMENDACAO_MASCARAMENTO.md`,
`checagem_mascaramento.json`, e a resposta 3 do apêndice do
`DOSSIE_LOTE_UNICO.md`.

---

## Pergunta 3 — Normalização do espectrograma

Duas famílias, com implicações diferentes:

- **Estatísticas globais (média/desvio por banda mel):** têm de ser calculadas
  **somente sobre o treino** — a subamostra de 30k —, **nunca** sobre o CSV
  inteiro nem sobre validação/teste. É exatamente o raciocínio do
  `StandardScaler` estar **dentro** do `Pipeline` do SVM
  (`src/models/treinar_svm.py:85-91`): o scaler é ajustado só no fold de
  treino. Estatísticas tiradas do conjunto inteiro são **vazamento**, e vazamento
  num artefato gerado uma vez só é irreversível.
- **Normalização por exemplo** (cada espectrograma normalizado por si): **não
  há vazamento**, mas isso muda o que a CNN vê — remove diferenças globais de
  energia entre gravações, que podem ser parte do sinal discriminativo. Se for
  essa a escolha, precisa estar **declarada**, e não apenas implementada.

**Recomendação:** estatísticas globais **calculadas na subamostra de 30k de
treino** e aplicadas a validação e teste. É a opção coerente com o resto do
protocolo do trabalho.

---

## Nota de engenharia (não é decisão do orientador — é planejamento)

**Gerar espectrograma só para `subamostra 30k + validação + teste`, não para os
148.176 áudios do eval.**

| conjunto | n |
|---|---:|
| subamostra de treino (braço principal) | 30.000 |
| validação (completa) | 22.226 |
| teste (completo, lacrado) | 22.227 |
| **total necessário** | **74.453** |
| universo eval inteiro | 148.176 |

O braço de referência é **RF-only** (RF treinado nos 103.723 do treino completo)
e nunca precisa de espectrograma. Gerar os 148.176 seria produzir ~73.700
tensores que nada consome.

**Conta de armazenamento:** 128 mels × 251 frames × 4 bytes ≈ **125 KiB por
áudio** em `float32`.

| escopo | float32 | float16 |
|---|---:|---:|
| 74.453 áudios (necessário) | ~9,3 GB | ~4,7 GB |
| 148.176 áudios (universo) | ~18,5 GB | ~9,3 GB |

Metade do tempo e metade do disco. *(Se a opção B da Pergunta 2 for escolhida, o
segundo canal — a máscara — pode ser gravado como `uint8` ou reconstruído em
tempo de carga a partir de `n_frames_validos`, que já está no `features.csv`;
neste caso o custo extra em disco é praticamente nulo.)*

---

## O que se pede ao orientador

Uma resposta às **três perguntas** — de preferência anotada neste próprio
arquivo, para que a decisão fique versionada no repositório junto com a
evidência que a motivou, exatamente como foi feito no Bloco 2.
