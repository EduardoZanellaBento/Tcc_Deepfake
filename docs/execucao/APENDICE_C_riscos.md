# Apêndice C — Correção de rota: o que simplificar quando um marco atrasar

> **Abra este arquivo no PRIMEIRO sinal de atraso, não depois.** A regra do orientador
> é explícita: se uma etapa atrasar, a correção de rota é sempre em direção à
> **SIMPLIFICAÇÃO**, e com a limitação registrada **por escrito**. E:
> **uma dificuldade pontual não pode empurrar o restante do cronograma.**

---

## 1. A tabela do orientador (é ela que manda)

| situação | correção |
|---|---|
| CNN muito complexa não estabilizou | simplifica a arquitetura |
| busca de hiperparâmetros ficou cara | reduz o espaço de busca |
| análise complementar ficou demorada | sai do escopo principal |
| 3 ou 5 sementes ficaram inviáveis | executa o protocolo mínimo e registra a limitação |

---

## 2. Regra de decisão diária

Todo dia, antes de começar, uma pergunta: **o marco de hoje ainda cabe hoje?**

```
Se SIM  → execute e não leia o resto deste arquivo.
Se NÃO  → não estenda o dia. Aplique a simplificação prevista para o marco,
          escreva a limitação NO MESMO DIA, e entre no marco seguinte no
          horário previsto.
```

**O erro que este apêndice existe para prevenir** é o de setembro: uma etapa
escorregou, o resto do cronograma foi empurrado, e o cronograma inteiro se perdeu.
Um marco entregue simplificado, com a limitação escrita, é infinitamente melhor que
um marco perfeito entregue três dias tarde.

---

## 3. Semáforo por data

| se em… | …o estado não for | então |
|---|---|---|
| **19/09** | B4.0 commitado | pare tudo e faça B4.0. É horas, não dias |
| **21/09** | gerador validado (B4.1) | corte as figuras e as checagens 9–11; **nunca** as checagens 1, 3, 6 e 7 |
| **22/09** | lote de **treino** gerado | gere só treino + validação; o teste pode ser gerado em 28/09 |
| **22/09** | `normalizacao_cnn.json` existindo | B4.3 vira prioridade absoluta — sem ele não há treino |
| **24/09** | CNN treinando, loss caindo | arquitetura mínima (§4). Se nem isso, §7 |
| **26/09** | melhor época registrada | feche o B4.5 por **julgamento documentado**, sem grade |
| **28/09** | refit feito | use a CNN dos 27k com a limitação do §6 |
| **29/09** | validação externa da CNN feita | corte throughput, mantenha latência + EER + matriz |
| **30/09** | teste lacrado rodado | **isto não se adia.** É o resultado final |
| **02/10** | metodologia + resultados escritos | escreva discussão e conclusão **curtas** e entregue completo em 04/10 |
| **04/10** | versão completa pronta | entregue com as lacunas **marcadas em destaque** e um parágrafo dizendo o que falta e quando chega |

---

## 4. Simplificações da CNN, em ordem de aplicação

Aplique **uma por vez**, na ordem. Cada degrau vem com a limitação já redigida —
copie, adapte o número, e siga.

### Degrau 1 — reduzir a grade de busca (custo: zero de qualidade)

3 configurações em vez de 6.

> A exploração de hiperparâmetros da CNN foi deliberadamente restrita a três
> configurações, por restrição de cronograma. O espaço explorado está integralmente
> registrado em `results/metricas/busca_cnn.csv`. Uma busca mais ampla não foi
> executada e fica registrada como trabalho futuro.

### Degrau 2 — arquitetura menor

`(16, 32, 64)`, 3 blocos, sem `BatchNorm`, dropout 0,0.

> A arquitetura da CNN foi mantida deliberadamente pequena (três blocos
> convolucionais, 16–64 canais). A escolha reflete o tamanho do conjunto de treino do
> braço principal — 30.000 exemplos, dos quais 3.000 bonafide — e a restrição de
> cronograma. Arquiteturas mais profundas, e arquiteturas específicas do domínio
> (LCNN, RawNet2, SincNet), ficam como trabalho futuro.

### Degrau 3 — menos épocas

Teto de 20 épocas em vez de 60.

> O teto de épocas foi fixado em 20 por restrição de cronograma. A curva de treino
> registrada em `curva_treino_cnn_definida.csv` mostra o comportamento até esse
> limite; não se pode afirmar que a rede não continuaria melhorando além dele.

### Degrau 4 — uma configuração só

A do B4.4, com *early stopping* implementado e a melhor época registrada.

> A arquitetura da CNN foi escolhida por julgamento, a partir do comportamento
> observado na configuração inicial, sem busca de hiperparâmetros. A justificativa
> está registrada em `cnn_definida.json`; uma busca sistemática fica como trabalho
> futuro.

**Isto ainda cumpre o B4.5.** A entrega obrigatória é «arquitetura final, tratamento
de desbalanceamento aplicado, early stopping interno concluído, número de épocas
definido» — e uma arquitetura escolhida por julgamento documentado satisfaz os quatro.

---

## 5. O que **nunca** se simplifica

| item | por que é intocável |
|---|---|
| **a máscara temporal** da CNN | é a P2, aprovada, e é a pergunta de banca nomeada pelo orientador. Sem ela a CNN reintroduz o problema que o Bloco 1 corrigiu no ramo clássico |
| **a loss ponderada** com razão 9:1 a favor de bonafide | é a P7. Sem ela a rede colapsa na majoritária e o f1_macro fica em ~0,47 — não há resultado |
| **a normalização só com o treino** | vazamento irreversível num artefato gerado uma vez só |
| **a separação 27k/3k** e a proibição de usar a validação externa para *early stopping* | destrói o protocolo dos três modelos de uma vez |
| **seleção de limiar na validação**, regra `score >= limiar` | é a régua comum. Sem ela a comparação não existe |
| **o teste usado uma única vez** | não há segunda chance |
| **`torch.cuda.synchronize()`** na medição de tempo | um tempo medido sem sincronizar não é incompleto, é **errado** — publicá-lo seria pior que omiti-lo |
| **as checagens 1, 3, 6 e 7 do B4.1** | se falharem, os 74.453 tensores são inúteis, e regerá-los não cabe no cronograma |
| **a lista de limitações** no texto | é onde está a credibilidade do trabalho |

---

## 6. Limitações pré-redigidas, prontas para colar

### Sementes na CNN (a mais provável)

> O resultado principal da CNN é reportado para a semente 42, sem análise de variância
> entre sementes. Essa análise estava prevista como complementar e **fora do caminho
> crítico** (decisão 6, de 17/09), e não foi executada porque a fase experimental foi
> encerrada pelo orientador em 24/09, para priorizar a redação dentro do cronograma de
> recuperação. *(Nota de 24/09: não escrever «por restrição de tempo computacional» —
> o custo medido era de ~15 min de GPU por semente; o motivo real é de escopo e prazo.)* Para contexto: o RF entre cinco sementes varia ±0,0004 de f1_macro, e o
> SVM com `probability=False` é determinístico; a fonte de variação que domina o braço
> principal é **qual subamostra de 30k foi sorteada**, medida em
> `estabilidade_subamostra.json`. A variância de inicialização da CNN é, portanto, uma
> fonte **não quantificada** neste trabalho, e fica registrada como limitação.

### CNN final treinada em 27k (refit não executado)

> Por restrição de cronograma, a CNN final é a treinada nos 27.000 exemplos da fase de
> *early stopping*, e não o refit nos 30.000 previsto no protocolo. A comparação com
> RF e SVM carrega, portanto, uma diferença de 3.000 exemplos de treino (10%) além da
> diferença de modelo. Pela curva de aprendizado do RF, 10% menos treino nessa faixa
> custa da ordem de 0,01 de f1_macro, o que torna a estimativa da CNN um **limite
> inferior**. O protocolo de refit está descrito na metodologia e fica registrado como
> a execução pendente.

### Determinismo estrito não obtido

> `torch.use_deterministic_algorithms(True)` não pôde ser aplicado em modo estrito
> porque a operação `<nome>` não possui implementação determinística em CUDA na versão
> utilizada. O treino foi executado com `warn_only=True`, o que significa que a
> re-execução com a mesma semente pode produzir pesos finais marginalmente diferentes.
> Todos os demais controles de aleatoriedade (`manual_seed`, `cudnn.deterministic`,
> `cudnn.benchmark=False`, `CUBLAS_WORKSPACE_CONFIG`) foram aplicados.

### Espectrograma em `float16`

> Os espectrogramas foram armazenados em `float16` (resolução ~0,06 dB na faixa
> utilizada, de 80 dB) por restrição de espaço em disco. A conversão para `float32`
> ocorre em tempo de carga, antes da normalização. Limitação computacional declarada;
> não altera a definição da representação.

### Throughput da CNN não medido

> O throughput em lote da CNN não foi medido, por restrição de cronograma. A latência
> em `batch = 1`, que é o cenário de uso real, está reportada nos dois cenários de
> hardware (GPU e CPU). Como a comparação RF × SVM já mostra que latência unitária e
> throughput em lote podem dar respostas **opostas**, a ausência do throughput da CNN
> é uma lacuna reconhecida na análise de custo.

### Bootstrap só na validação

> O intervalo de confiança da diferença entre modelos foi estimado por bootstrap
> pareado no conjunto de validação (22.226 amostras, 1.000 reamostragens). No conjunto
> de teste reporta-se a medida pontual, por restrição de cronograma. Como o teste é
> usado uma única vez e tem tamanho comparável à validação, espera-se ordem de
> incerteza semelhante.

### Diagnóstico por ataque/codec sem a CNN

> As análises por sistema de síntese (A07–A19) e por codec foram executadas para RF e
> SVM, com o limiar do protocolo. A CNN não foi incluída nessas análises por restrição
> de cronograma; a comparação por estrato fica registrada como análise complementar.

---

## 7. O plano de contingência absoluto

**Se em 26/09 a CNN não estiver produzindo métricas plausíveis** (loss não cai,
colapso na majoritária que não cede, erro técnico não resolvido), pare de depurar e
execute isto:

1. **Diagnóstico de 2 horas, nesta ordem** — são as quatro causas prováveis, da mais
   comum para a menos:
   - **alinhamento**: `classe_binaria` e `n_frames_validos` do índice × `features.csv`
     nas mesmas linhas (asserções 5 e 6 do B4.2). Acurácia ~50% com loss caindo é
     **sempre** isto;
   - **normalização**: `assert torch.isfinite(x).all()` no `Dataset`;
     `faixas_com_desvio_clampado` em `normalizacao_cnn.json`;
   - **pesos da loss**: imprima os dois valores. `w_bonafide` tem de ser ~9× `w_spoof`;
   - **disjunção**: os 3k não podem estar dentro dos 27k.
2. **Se resolver, siga o cronograma pelo degrau 4 do §4.**
3. **Se não resolver até o fim do dia 26/09:** aplique o degrau 2 (arquitetura mínima,
   sem BatchNorm, sem dropout, 3 blocos, 15 épocas) **do zero**, em arquivo novo. Uma
   rede de 3 blocos sobre 128×251 normalizado, com masked pooling e loss ponderada,
   **treina**. Se a versão mínima treina e a anterior não, o problema estava na
   arquitetura, e a mínima é a sua CNN — com o degrau 2 registrado como limitação.
4. **Em 28/09, com o que existir**, rode a validação externa e siga para o B5.
   Uma CNN simples que funciona, comparada honestamente, **responde à pergunta de
   pesquisa**. Uma CNN sofisticada que não chega ao dia 04/10 não responde nada.

> **O que fecha o TC II é a comparação, não a CNN.** A pergunta de pesquisa é se
> modelos clássicos com features manuais mantêm desempenho competitivo frente a CNNs
> no mesmo ambiente experimental. Uma CNN baseline honesta, com protocolo idêntico e
> limitações declaradas, responde isso. Uma CNN otimizada fora do prazo não responde.

---

## 8. Registro de desvios

Crie `docs/execucao/DESVIOS.md` na primeira vez que aplicar qualquer simplificação
deste apêndice, e registre **no dia**:

```markdown
| data | marco | desvio aplicado | limitação escrita em | commit |
|---|---|---|---|---|
| 24/09 | B4.5 | grade reduzida de 6 para 3 configurações | busca_cnn.csv + §Limitações | abc1234 |
```

Esse arquivo é o que transforma «não deu tempo» em **decisão metodológica
registrada** — e é a diferença entre uma limitação que a banca aceita e uma lacuna que
ela cobra.
