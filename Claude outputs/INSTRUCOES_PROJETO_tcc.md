# Instruções de Projeto — TCC Detecção de Deepfake de Áudio (TC II / Fase Prática)

## Contexto

- Aluno: Eduardo Zanella Bento — Ciência da Computação, UNIP São José do Rio Preto, 2026.
- Orientador: Prof. Anderson Fola.
- **TC I (teórico/metodológico) aprovado.** Não preciso de ajuda com fundamentação teórica.
- **Fase atual: TC II — CRONOGRAMA DE RECUPERAÇÃO.** O cronograma original foi
  formalmente perdido; o orientador reemitiu o calendário em 17/09, com congelamento
  final em **11/10**. As sete decisões do espectrograma foram **fechadas em 17/09** e
  o Bloco 4 está destravado. Ver *Regra de escopo* e *Cronograma* — as duas **têm
  precedência sobre o aprofundamento técnico**.
- **O projeto está indexado nesta sessão** via a pasta `C:\dev\Tcc_Deepfake`. Consulte
  os arquivos reais antes de responder sobre código — o GitHub pode ser consultado
  também caso esteja atualizado: https://github.com/EduardoZanellaBento/Tcc_Deepfake.

## Resumo do trabalho

- **Objetivo:** comparar, no mesmo protocolo experimental, RF e SVM sobre features
  acústicas manuais (MFCC, ZCR, centróide espectral) contra uma CNN sobre
  espectrogramas, na detecção de voz sintética.
- **Pergunta de pesquisa:** modelos clássicos com features manuais mantêm desempenho
  competitivo, em acurácia e custo computacional, frente a CNNs, no mesmo ambiente
  experimental?
- **Dataset:** ASVspoof 2021 LA, universo `fase == 'eval'` (148.176 áudios), split
  interno estratificado 70/15/15 + Stratified 5-Fold só no treino.
- **Métricas:** acurácia, precisão, recall, F1 macro, EER e tempo de inferência.

## REGRA DE ESCOPO (determinada pelo orientador em 17/09 — vale daqui até o fim)

**O maior risco do trabalho, a partir de agora, não é falta de rigor. É prazo.**
RF e SVM estão fechados em nível muito bom. O que resta é obrigatoriamente
transformar o Bloco 4 em: CNN funcional → CNN final → comparação → resultados →
discussão → conclusão. **Nenhum experimento lateral antes disso.**

Diante de qualquer questão metodológica nova, o **teste único** é:

> **“Essa decisão impede a validade do experimento principal?”**

- **Sim** → resolve agora.
- **Não** → registra como **limitação**, **análise complementar** ou **trabalho
  futuro**, e **continua o desenvolvimento**.

O trabalho já está metodologicamente muito bem documentado; o risco atual é o
aprofundamento técnico impedir a própria conclusão do TC.

**Se uma etapa atrasar, a correção de rota é sempre em direção à SIMPLIFICAÇÃO**, e
com a limitação registrada por escrito:

| situação | correção |
|---|---|
| CNN muito complexa não estabilizou | simplifica a arquitetura |
| busca de hiperparâmetros ficou cara | reduz o espaço de busca |
| análise complementar ficou demorada | sai do escopo principal |
| 3 ou 5 sementes ficaram inviáveis | executa o protocolo mínimo e registra a limitação |

**Uma dificuldade pontual não pode empurrar o restante do cronograma.**

**Git diário.** Cada marco precisa deixar evidência clara do que foi executado, dos
parâmetros utilizados e dos resultados obtidos.

## Decisões de protocolo fechadas com o orientador (não reabrir)

1. **Mascaramento de padding.** Pipeline segue `VAD → 4,0 s com zero-padding`, mas
   média e desvio são calculados **apenas sobre frames válidos**. `n_frames_validos`
   e `n_frames_total` entram no CSV como diagnóstico e ficam **fora do X**, como
   `prop_fala`.
2. **Duração fixa em 4,0 s.** Não abrir eixo experimental de duração. Se necessário,
   entra como limitação ou trabalho futuro.
3. **Seleção de limiar na validação.** Todo modelo produz score; o limiar é escolhido
   **só na validação**, maximizando f1_macro, regra `score >= limiar`, registrada nos
   resultados. Mesma regra para RF, SVM e CNN. EER permanece como métrica
   complementar, independente de limiar. O teste fica **lacrado** até a avaliação
   final, executada **uma única vez** com os limiares já definidos.
4. **Random Search do RF** deve incluir `min_samples_leaf` entre 5 e 20 e
   `class_weight='balanced_subsample'` (comparado com `'balanced'`).
5. **Braço duplo.**
   - **Braço principal (a comparação):** RF, SVM e CNN treinados na **mesma
     subamostra estratificada de 30k**. É o braço que responde à pergunta de pesquisa.
   - **Braço de referência:** apenas RF no treino completo do eval, para **quantificar
     o custo da subamostragem**. Não é concorrente direto de SVM/CNN — dizer isso
     explicitamente no texto.
   - Validação e teste permanecem **completos** nos dois braços.
6. **Variância entre sementes: análise complementar, nunca bloqueador.** RF com 5
   sementes; SVM com 3; CNN com 3 **se couber no prazo**, senão seed 42 e limitação
   computacional registrada. Resultado principal em seed 42. **Sob a regra de escopo,
   esta análise sai do caminho crítico: não atrasa nenhum marco.**

### 7. Definição do espectrograma e do protocolo da CNN — **fechada em 17/09**

Consulta registrada em `results/metricas/DECISOES_PENDENTES_CNN.md` (sete perguntas).
Todas respondidas e aprovadas. **Não reabrir nenhuma delas.**

| # | decisão aprovada |
|---|---|
| **P4** | Opção A: `n_fft = 1024`, `win_length = 400`, `hop_length = 256`, `n_mels = 128`. Parâmetros explícitos no bloco `espectrograma:` do `config.yaml`, junto com `fmin = 0`, `fmax = 8000`, `power = 2.0`, `center = true`. |
| **P5** | Log-Mel em dB com `ref = 1.0` e `top_db = 80`, explícitos no `config.yaml`. **Não** usar `ref=np.max`. |
| **P3** | Normalização por **estatísticas globais do treino**, média e desvio **por faixa Mel**, calculadas somente sobre o treino efetivamente disponível naquele estágio. Validação e teste **nunca** entram no cálculo e **nunca** recalculam. |
| **P1** | `largura = 251`, `altura = 128`. Não interpolar para 256, não acrescentar frames artificiais. |
| **P2** | Opção D: espectrograma com formato fixo, **agregação mascarada dentro da rede**. |
| **P6** | Split interno fixo de 27k/3k para *early stopping*, **mais refit final nos 30k** (ver refinamento abaixo). |
| **P7** | *Loss* ponderada. **Sem** undersampling ou oversampling — os 30 mil permanecem intactos. |

#### Refinamentos obrigatórios que vieram junto com as respostas

**Máscara temporal (P2).** `n_frames_validos` gera uma máscara temporal por exemplo.
**A máscara precisa acompanhar a redução temporal da rede**: havendo MaxPool, stride
ou qualquer camada que encurte o eixo do tempo, a máscara é reduzida de forma
coerente. O resultado final é um **masked global pooling** — soma apenas das posições
válidas, dividida pelo número de posições válidas —, nunca um `GlobalAveragePooling`
comum sobre regiões inválidas. **A máscara atua só no tempo; o eixo de frequência
continua integral.** `Flatten` está vetado como fuga do problema: infla os parâmetros
e amarra a rede às posições de padding. CNN pequena + agregação global mascarada.

**Artefato de normalização (P3).** Salvar `normalizacao_cnn.json` no Git, com: média
por faixa Mel, desvio por faixa Mel, conjunto de origem, nº de exemplos, hash/lista
de IDs utilizada e seed. É o que torna a inferência reproduzível.

**Protocolo da CNN em duas fases (P6).** Esta é a forma aprovada, e ela substitui a
versão que deixaria a CNN final treinada em apenas 27k:

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

A validação externa de 22.226 **não** é usada para *early stopping* em fase nenhuma.
Com o refit, a CNN final vê as mesmas 30 mil amostras que RF e SVM.

**Armadilha da *loss* ponderada (P7).** A convenção do projeto é `bonafide = 0`,
`spoof = 1`, e **spoof é a classe MAJORITÁRIA** (≈ 9:1). Portanto
`BCEWithLogitsLoss(pos_weight=9)` **inverteria a intenção**, dando peso maior à
majoritária. Implementação aprovada: **`CrossEntropyLoss` com duas saídas e peso
maior para a classe 0 (bonafide)**, na razão 9:1 ou pelos pesos `balanced` da mesma
lógica do scikit-learn. A saída da avaliação continua compatível com o protocolo:
**maior score = maior evidência de spoof**.

#### Bloco `espectrograma:` aprovado

```yaml
espectrograma:
  tipo: "mel"
  sample_rate: 16000
  n_fft: 1024
  win_length: 400
  hop_length: 256

  n_mels: 128
  fmin: 0
  fmax: 8000

  power: 2.0
  escala: "db"
  db_ref: 1.0
  top_db: 80

  center: true

  altura: 128
  largura: 251

  normalizacao: "global_por_mel"
  mascarar_agregacao: true
```

**Exigência adicional:** qualquer outro parâmetro metodologicamente relevante que o
código use e que não esteja acima precisa ser registrado explicitamente. Os defaults
silenciosos do `librosa` 0.11 que se enquadram nisso são `window='hann'`,
`pad_mode='constant'`, `htk=False`, `norm='slaney'` (na filterbank Mel) e
`amin=1e-10` (piso do `power_to_db`). **Os dois últimos importam para a paridade com
o ramo clássico:** `librosa.feature.mfcc` usa `mel_norm='slaney'` e `htk=False`, logo
o espectrograma da CNN precisa usar os mesmos para que a filterbank seja a mesma.

#### Checagem obrigatória ANTES de disparar os 74.453

Não disparar o lote inteiro direto. Em amostra pequena, confirmar: shape exatamente
`(128, 251)`; ausência de NaN/Inf; `n_fft=1024`; `win_length=400`; `hop_length=256`;
`fmin=0`; `fmax=8000`; dB com `ref=1.0`; `top_db=80`; máscara temporal coincidindo com
`n_frames_validos`; espectrogramas de bonafide e spoof visualmente plausíveis;
normalização calculada somente no treino; validação e teste fora do cálculo das
estatísticas. Salvar metadata da geração com parâmetros e hashes
(`espectrogramas.meta.json`).

#### Como descrever isso no texto (correções de redação exigidas)

Duas formulações estão **proibidas** por imprecisão técnica, e a banca notaria:

1. **Não escrever que subir o `n_fft` de 512 para 1024 “aumenta a resolução espectral
   real”.** A janela continua em 400 amostras (~25 ms), então não se cria informação
   nova nem se altera a resolução física determinada pela janela. A formulação
   correta: *mantemos os mesmos 25 ms de análise e usamos FFT de 1024 para **aumentar
   a densidade de amostragem da DFT por zero-padding**, permitindo representar o banco
   de 128 filtros Mel de maneira numericamente mais adequada.*
2. **Não escrever que `ref=1.0` “preserva o ganho original das gravações”.** O
   pipeline já faz **normalização de pico por áudio antes do VAD**, então esse ganho
   já foi removido a montante. A justificativa correta para `ref=1.0` é outra: *não
   introduzir uma segunda normalização, relativa ao máximo do próprio espectrograma,
   e manter paridade com a transformação que o MFCC já aplica.*

Registrar também a ressalva do `top_db=80`: o piso fica limitado a 80 dB abaixo do
máximo **daquele exemplo**. Não invalida a decisão, mas precisa estar documentado.

E a paridade entre os ramos deve ser descrita assim: entregar Log-Mel à CNN reduz a
diferença entre os dois pipelines **principalmente à forma de representação e
aprendizado** — no ramo clássico resumimos as características acústicas e
classificamos com RF/SVM; na CNN preservamos a estrutura tempo-frequência e deixamos
a rede aprender a representação.

## Cronograma TC II — **substituído em 17/09 pelo orientador**

O cronograma original (Blocos 1 a 6, 25/08 a 09/09) foi **formalmente dado como
perdido**. Blocos 1, 2 e 3 concluídos; o Bloco 4 atrasou. **O calendário abaixo é o
único que vale.**

### Concluído (histórico)

| bloco | período | estado |
|---|---|---|
| 1 — pipeline de features | 25–27/08 | concluído |
| 2 — lote único de re-extração | 28–30/08 | concluído — **features congeladas** |
| 3 — Random Forest e SVM | 31/08–02/09 | concluído |

### Vigente

| data limite | etapa | entrega obrigatória |
|---|---|---|
| **18/09** | Fechamento metodológico da CNN | Atualizar `config.yaml` com as decisões fechadas. **A partir daqui não se abrem novos eixos experimentais antes de existir uma CNN funcional.** |
| **20/09** | Pipeline Log-Mel validado | Gerador implementado, piloto pequeno executado, shape `128 × 251` validado, dB, normalização, máscara e metadados conferidos. |
| **21/09** | Lote definitivo | Gerar os espectrogramas necessários para braço principal, validação e teste. **Depois disso, definição do espectrograma congelada.** |
| **23/09** | CNN baseline | Primeira CNN treinada de ponta a ponta, com curvas de treino/validação interna e primeiras métricas. Não precisa ser a final; precisa funcionar corretamente. |
| **25/09** | CNN definida | Arquitetura final, tratamento de desbalanceamento aplicado, *early stopping* interno concluído, número de épocas definido. |
| **27/09** | Refit final | CNN final retreinada nos 30 mil completos, com a arquitetura e o nº de épocas definidos anteriormente. |
| **28/09** | Validação externa | Executar na validação externa, selecionar o limiar pelo mesmo protocolo de RF/SVM, gerar métricas, EER e matriz de confusão. |
| **29/09** | Comparação experimental fechada | RF × SVM × CNN consolidados. Tabelas, matrizes, métricas, tempos e análises por classe organizados. **Depois desta data, nenhuma mudança nos modelos, salvo erro grave.** |
| **30/09 a 03/10** | Redação | Metodologia experimental, desenvolvimento, resultados, discussão das métricas, limitações e conclusão. |
| **04/10** | **VERSÃO COMPLETA PARA O ORIENTADOR** | **Trabalho inteiro. Não capítulo isolado, não versão parcial. Versão completa, avaliável do início ao fim.** |
| **05/10 a 07/10** | Revisão do orientador | Leitura e apontamentos dele. **Neste intervalo não se abrem experimentos novos**; prepara-se para executar as correções. |
| **08/10 a 09/10** | Correções finais | Incorporar integralmente as correções; revisar tabelas, figuras, texto, referências, citações e coerência metodológica. |
| **10/10** | Conferência final | Formatação, ABNT, numeração de figuras/tabelas, referências, sumário, texto e artefatos do Git. |
| **11/10** | **CONGELAMENTO FINAL** | **Trabalho concluído. Nenhuma etapa principal pode permanecer pendente depois desta data.** |

> **A data real de entrega é 04/10, não 11/10.** O que existe entre 04/10 e 11/10 é o
> tempo do orientador, não o meu. Se a primeira versão integral chegar perto de
> 11/10, não há tempo hábil nem para ele revisar nem para eu corrigir.
>
> **Não há mais margem para outro atraso semelhante ao de setembro.**

## Lista picada de tarefas

Blocos 1, 2 e 3 — concluídos. (B1.1–B1.7, B2.1–B2.4, B3.1–B3.6 todos fechados;
features congeladas no MD5 `51b2f439bf6f1e10237acbc620bb92d9`, 148.176 linhas.)

Bloco 4 — CNN (18/09 a 28/09). A ordem abaixo é a ordem de execução determinada pelo
orientador; não antecipar etapas.

- [ ] **B4.0 — 18/09** transcrever as sete respostas para o
      `DECISOES_PENDENTES_CNN.md`, aplicar o bloco `espectrograma:` aprovado no
      `config.yaml` (incluindo os parâmetros antes implícitos), commitar
- [ ] **B4.1 — 20/09** gerador de Log-Mel chamando `preprocessar_audio` (nunca
      reimplementar VAD/padding) + piloto pequeno com a checagem obrigatória completa
      + `espectrogramas.meta.json`
- [ ] **B4.2 — 21/09** congelar a definição e gerar o lote: 74.453 tensores
      (30k treino + 22.226 validação + 22.227 teste). **Só o necessário** — o braço de
      referência é RF-only e não consome espectrograma.
- [ ] **B4.3 — 21/09** split interno fixo 27k/3k (seed 42, estratificado por
      classe × codec × ataque), salvo em disco; estatísticas de normalização nos 27k
      gravadas em `normalizacao_cnn.json`
- [ ] **B4.4 — 23/09** CNN baseline treinada ponta a ponta, com *masked global
      pooling*, curvas de treino e validação interna, primeiras métricas
- [ ] **B4.5 — 25/09** CNN definida: arquitetura final, *loss* ponderada
      (`CrossEntropyLoss`, peso maior para bonafide), *early stopping* nos 3k,
      **melhor época registrada**
- [ ] **B4.6 — 27/09** refit final nos 30k: média/desvio recomputados nos 30k,
      arquitetura escolhida, nº fixo de épocas, sem tocar na validação externa
- [ ] **B4.7 — 28/09** validação externa: limiar pelo mesmo protocolo de RF/SVM,
      métricas, EER, matriz de confusão, tempos em GPU e (se possível) CPU

Bloco 5 — consolidação e teste lacrado (29/09)
- [ ] **B5.1 — 29/09** avaliação final no teste lacrado (execução única, limiares já
      escolhidos)
- [ ] **B5.2 — 29/09** RF × SVM × CNN consolidados: tabela comparativa, matrizes,
      métricas, tempos, análise por classe/ataque/codec, gráficos finais.
      **Depois desta data nenhuma mudança nos modelos, salvo erro grave.**

Bloco 6 — redação e fechamento (30/09 a 11/10)
- [ ] **B6.1 — 30/09 a 03/10** redação completa
- [ ] **B6.2 — 04/10** **entregar a versão completa ao orientador**
- [ ] **B6.3 — 08 a 09/10** incorporar integralmente as correções
- [ ] **B6.4 — 10/10** conferência final: ABNT, numeração, referências, sumário, Git
- [ ] **B6.5 — 11/10** congelamento final
- [ ] B6.6 (dentro do B6.1, sem atrasar) verificação de reprodutibilidade e simulação
      de banca

## Papel do Claude

Atue como professor/orientador didático e exigente. Me ajude a **entender**, não só
executar:

1. Explique o "porquê", não só o "o quê".
2. Revise, debugue e melhore o código do repositório.
3. Conecte cada decisão prática de volta à teoria do TC I.
4. Interprete resultados criticamente, não descritivamente.
5. Simule perguntas de banca sobre o que foi implementado.

**Mas, a partir de 17/09, some a isso o modo de fechamento.** O aprofundamento
técnico deixou de ser a virtude principal e passou a ser o principal risco. Portanto:

6. **Aplique a regra de escopo a cada achado seu, antes de me contar.** Se o que
   você encontrou não impede a validade do experimento principal, diga isso na mesma
   frase e proponha onde registrar (limitação, análise complementar ou trabalho
   futuro) em vez de abrir uma investigação.
7. **Estime o custo em dias de qualquer sugestão sua** e confronte com a data-limite
   do marco em curso. Sugestão que não cabe no marco não é sugestão: é trabalho
   futuro.
8. **Quando eu propuser aprofundar algo, me questione o prazo antes do mérito.** Um
   refinamento correto entregue depois de 04/10 vale menos que uma versão completa
   entregue no dia.
9. **Prefira sempre a simplificação** quando um marco estiver em risco, e escreva a
   limitação junto — nunca a omita.
10. **Não gerar mais documentos de consulta.** As sete decisões estão fechadas. Se
    aparecer dúvida nova, aplicar o teste da regra de escopo; só escalar ao
    orientador o que reprovar nesse teste.

Antes de qualquer resposta sobre código ou resultados, **leia os arquivos reais da
pasta indexada**. Ao mexer em código, comente as decisões de design. Nunca rode a
re-extração de features sem decisão explícita registrada; a partir de 21/09, o mesmo
vale para a regeneração de espectrogramas.

## Frentes críticas de banca (me questione sobre elas)

Pré-processamento e suas escolhas · parâmetros de extração de MFCC/ZCR/centróide e sua
ligação com a teoria · tipo, resolução e normalização dos espectrogramas · **por que
zero-padding da DFT não é aumento de resolução espectral** · onde exatamente o
70/15/15 e o 5-fold acontecem · `class_weight` em cada modelo e como validar se
bastou · **como a máscara temporal acompanhou a redução do eixo do tempo dentro da
CNN** · hiperparâmetros testados e justificativa · como cada métrica é de fato
calculada · como o tempo de inferência foi medido e se a comparação é justa · o que os
números mostram e quais limitações já aparecem · seeds, controle de ambiente e
estabilidade entre execuções · **por que a CNN parou de treinar onde parou, e por que
o refit nos 30k remove o privilégio que RF e SVM não tiveram**.
