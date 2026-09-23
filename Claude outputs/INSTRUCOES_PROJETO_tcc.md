# Instruções de Projeto — TCC Detecção de Deepfake de Áudio (TC II / Fase Prática)

> **Versão de 23/09/2026, revisão pós-Bloco 5.** Substitui integralmente as versões
> de 17/09 e de 23/09 (pós-Bloco 4). O histórico do que mudou está no fim do arquivo.
> Este arquivo é a cópia versionada em Git das instruções que vivem nas configurações
> do projeto no claude.ai — as duas precisam ser idênticas. O cronograma original do
> TC II já se perdeu uma vez por existir em um lugar só; é por isso que esta cópia
> existe.

## Contexto

- Aluno: Eduardo Zanella Bento — Ciência da Computação, UNIP São José do Rio Preto, 2026.
- Orientador: Prof. Anderson Fola.
- **TC I (teórico/metodológico) aprovado.** Não preciso de ajuda com fundamentação teórica.
- **Fase atual: TC II — CRONOGRAMA DE RECUPERAÇÃO.** O cronograma original foi
  formalmente perdido; o orientador reemitiu o calendário em 17/09, com congelamento
  final em **11/10**. As sete decisões do espectrograma foram **fechadas em 17/09** e
  não se reabrem.
- **Blocos 1, 2, 3, 4 e 5 estão concluídos.** O teste lacrado foi usado — uma única
  vez — em 23/09. A frente atual é o **Bloco 6 — redação (B6.1)**. Ver *Estado atual*.
- **As datas do cronograma são TETO, não agenda.** Hoje o trabalho está **adiantado**;
  ver *Regra de escopo → ritmo*.
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

---

## ESTADO ATUAL — 23/09/2026 (pós-Bloco 5)

**Bloco 4 fechado em 20/09**, oito dias antes do teto (28/09). **Bloco 5 fechado em
23/09**, seis dias antes do teto (29/09): B5.1 no commit `498f3c8`, B5.2 no `c07390b`,
seguidos da revisão das leituras do B5.2 (três frases que afirmavam mais do que o dado
sustenta — nenhum número mudou). **A partir do B5.2, nenhuma mudança nos modelos,
salvo erro grave.**

**Resultado final — teste lacrado (22.227), execução única, limiares escolhidos só na
validação, regra `score >= limiar`:**

| modelo | n treino | limiar | f1_macro | EER | ROC-AUC | recall bonafide |
|---|---:|---:|---:|---:|---:|---:|
| RF ajustado — principal | 30.000 | 0,6516 | 0,7210 | 0,1946 | 0,8878 | 0,5326 |
| RF ajustado — referência | 103.723 | 0,6196 | 0,7602 | 0,1629 | 0,9167 | 0,5794 |
| SVM RBF ajustado — principal | 30.000 | −0,0329 | 0,7981 | 0,1411 | 0,9315 | 0,6635 |
| **CNN final — principal** | 30.000 | 0,3252 | **0,8883** | **0,0737** | **0,9783** | **0,7877** |

Na validação (22.226), onde os limiares foram escolhidos: RF 0,7225 / 0,1930 · RF
referência 0,7723 / 0,1579 · SVM 0,7987 / 0,1462 · CNN 0,8975 / 0,0738 (f1_macro /
EER). Os quatro deltas validação → teste cabem no ruído de amostragem dos dois
conjuntos, e os quatro Δf1 são negativos — o otimismo da seleção de limiar, esperado
por construção. A CNN perde 0,0092 de f1 com o EER parado (−0,0001).

**Bootstrap pareado (1.000 reamostragens, semente 42, mesmo vetor de índices para
todos os modelos) — no teste:** SVM − RF Δf1 +0,0771 [0,0671; 0,0868]; CNN − RF
+0,1673 [0,1568; 0,1775]; CNN − SVM +0,0902 [0,0801; 0,1005]. Em EER, os três pares
também têm IC95 sem zero. **A ordem CNN > SVM > RF é real neste protocolo**, na
validação e no teste. O custo da subamostragem para o RF (referência − principal) é
Δf1 +0,0392 [0,0329; 0,0451] no teste — e mesmo com 3,5× mais dados o RF de referência
fica abaixo de SVM e CNN.

**Custo ponta a ponta por áudio** (batch = 1, os quatro arranjos medidos na **mesma
execução**, 20/09): RF **10,2293 ms** · SVM **4,9191 ms** · CNN-GPU **4,5765 ms** ·
CNN-CPU **16,7968 ms**. Em lote (batch = 22.226): RF 0,0204 ms/áudio · CNN-GPU 0,2383
· SVM 0,4025.

**A leitura que vai para o texto:** em acurácia a CNN abre distância clara (EER
praticamente metade do SVM) e a diferença é estatisticamente real. Em custo a resposta
é **condicional ao hardware**: em CPU os dois clássicos são mais baratos que a CNN, e a
CNN em CPU é a mais cara de todas; com GPU, **CNN-GPU e SVM empatam na prática** (a
folga de 7,5% fica dentro da variação de 4,1–13,7% que os tempos mostraram entre
sessões) e os dois custam menos da metade do RF. A vantagem de custo dos clássicos é
**não exigirem GPU**. Latência unitária e vazão em lote **não ordenam os modelos da
mesma forma**. Por ataque, a variação entre sistemas de síntese domina a diferença
entre modelos no RF e no SVM; na CNN ela fica do mesmo tamanho da distância para o RF.
Por codec, a hipótese de a CNN ganhar mais com banda larga fica **não decidida**.

> Se algum número acima divergir de `results/metricas/*.json` ou de
> `results/metricas/COMPARACAO_FINAL.md`, **o artefato manda**, e esta seção se
> corrige no mesmo commit.

---

## REGRA DE ESCOPO (determinada pelo orientador em 17/09 — vale até o fim)

**O maior risco do trabalho não é falta de rigor. É prazo.** O que resta é
obrigatoriamente: teste lacrado → comparação → resultados → discussão → conclusão →
texto completo. **Nenhum experimento lateral antes disso.**

Diante de qualquer questão metodológica nova, o **teste único** é:

> **"Essa decisão impede a validade do experimento principal?"**

- **Sim** → resolve agora.
- **Não** → registra como **limitação**, **análise complementar** ou **trabalho
  futuro**, e **continua o desenvolvimento**.

**Se uma etapa atrasar, a correção de rota é sempre em direção à SIMPLIFICAÇÃO**, com
a limitação registrada por escrito:

| situação | correção |
|---|---|
| CNN muito complexa não estabilizou | simplifica a arquitetura |
| busca de hiperparâmetros ficou cara | reduz o espaço de busca |
| análise complementar ficou demorada | sai do escopo principal |
| 3 ou 5 sementes ficaram inviáveis | executa o protocolo mínimo e registra a limitação |

**Uma dificuldade pontual não pode empurrar o restante do cronograma.**

### A regra de escopo é sobre ATRASO, não sobre ritmo

**As datas do cronograma são teto, não agenda.** Terminar um marco antes da data é o
resultado desejado, não anomalia. Estando adiantado:

- **siga para o marco seguinte sem pedir autorização e sem sinalizar.** A regra de
  escopo, o teste da validade e as seções «Se atrasar» dos briefings existem para
  quando uma data-limite está **em risco**. Adiantado, não há o que simplificar nem
  limitação a escrever;
- **«uma sessão por marco» é higiene de contexto, não calendário.** Terminou o marco
  e sobrou dia? Encerre a sessão e abra a próxima com o briefing seguinte — não é
  para esperar a data chegar, e também não é para encadear dois marcos na mesma
  conversa;
- **adiantar nunca autoriza pular um pré-requisito** nem abrir eixo experimental
  novo. Os quatro portões estão em `APENDICE_A_inventario.md`.

**Fila do tempo excedente**, nesta ordem, sem exceção:

1. **redação (B6.1)** — maior risco do cronograma e o único item que não comprime no
   fim;
2. **variância entre sementes na CNN** (3 sementes) — custo medido: ~15 min de GPU
   por semente, ~1 h no total. Se couber, a limitação computacional declarada
   desaparece. **Avaliada só na validação**: o teste já foi usado;
3. **diagnóstico por ataque e por codec com a CNN incluída** — executado em B4.7 e
   consolidado em B5.2; revisar e integrar ao texto;
4. **simulação de banca** (`APENDICE_B_banca.md`) — cada pergunta sem resposta em 30
   segundos é um parágrafo que falta.

**Nunca** entra: leave-one-attack-out, cross-dataset, outra duração de áudio, outro
`n_mels`, nova busca de hiperparâmetros de RF/SVM. São trabalho futuro mesmo com uma
semana sobrando — o que fecha o TC II é a comparação completa, não um experimento a
mais.

**Git diário.** Cada marco precisa deixar evidência clara do que foi executado, dos
parâmetros utilizados e dos resultados obtidos.

---

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
   final, executada **uma única vez** com os limiares já definidos — **executada em
   23/09 (B5.1)**; `scripts/avaliar_teste_lacrado.py` recusa uma segunda execução.
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
   sementes; SVM com 3; CNN com 3. Resultado principal em seed 42. **Custo medido em
   B4.6: 887 s (~15 min) de GPU por refit**, logo as duas sementes extras da CNN são
   ~1 h de máquina — cabe. Ainda assim **continua fora do caminho crítico**: entra
   depois de a redação estar andando, e sua ausência é limitação declarada, nunca
   motivo de atraso. **As sementes extras se avaliam só na validação** — o teste foi
   usado uma vez e não se reabre.
7. **Definição do espectrograma e protocolo da CNN** — fechada em 17/09, sete
   perguntas respondidas. Detalhe integral na seção abaixo. **Não reabrir.**
8. **Regime numérico e protocolo de medição de tempo da CNN** — *registrado em B4.7,
   já implementado e commitado; não é eixo experimental novo, é invariante de
   execução*:
   - **TF32 desligado** (`torch.backends.cudnn.allow_tf32 = False` e
     `torch.backends.cuda.matmul.allow_tf32 = False`) antes de qualquer score da CNN.
     Motivo: com TF32 o score passa a depender do tamanho do lote (até 8,6e-05) e
     difere do float32 em até 2,3e-03 — mesma ordem do espaçamento entre scores
     vizinhos (~3,5e-03). Como `selecionar_limiar` opera sobre `np.unique(scores)`,
     isso mexe em `n_candidatos` e pode **deslocar o limiar**. A guarda mora dentro de
     `scores_de` (`src/models/modelos_ajustados.py`), e o B5.1 a herdou — o
     `teste_lacrado.json` registra `tf32_desligado: true`.
   - **Aquecimento declarado da GPU antes do cronômetro.** Os
     `descartar_aquecimento: 3` do `config.yaml` são 3 *iterações* (~3 ms em GPU) e
     não tiram a placa do estado de baixo consumo depois de um trecho longo de CPU.
     Medido: **8,83 ms com a GPU fria contra 0,82 ms quente**. `src/models/tempo.py`
     **não** foi alterado — o aquecimento acontece antes de o callable ser entregue ao
     cronômetro, e o número frio fica publicado ao lado do quente.
   - **Comparação de tempo só vale medida na mesma execução.** RF, SVM e CNN foram
     recronometrados juntos em B4.7 por isso. Número de tempo medido em outra sessão
     não entra em tabela comparativa. **E mesmo dentro dela, diferença menor que a
     variação entre sessões (4,1–13,7%, `reproducao_cnn.json`) é empate prático**, não
     ordem.

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
| **P6** | Split interno fixo de 27k/3k para *early stopping*, **mais refit final nos 30k**. |
| **P7** | *Loss* ponderada. **Sem** undersampling ou oversampling — os 30 mil permanecem intactos. |

#### Refinamentos obrigatórios que vieram junto com as respostas

**Máscara temporal (P2).** `n_frames_validos` gera uma máscara temporal por exemplo.
**A máscara precisa acompanhar a redução temporal da rede**: havendo MaxPool, stride
ou qualquer camada que encurte o eixo do tempo, a máscara é reduzida de forma
coerente. O resultado final é um **masked global pooling** — soma apenas das posições
válidas, dividida pelo número de posições válidas —, nunca um `GlobalAveragePooling`
comum sobre regiões inválidas. **A máscara atua só no tempo; o eixo de frequência
continua integral.** `Flatten` está vetado: infla os parâmetros e amarra a rede às
posições de padding. CNN pequena + agregação global mascarada.

**Artefato de normalização (P3).** `normalizacao_cnn.json` versionado no Git, com:
média por faixa Mel, desvio por faixa Mel, conjunto de origem, nº de exemplos,
hash/lista de IDs utilizada e seed. É o que torna a inferência reproduzível.

**Protocolo da CNN em duas fases (P6):**

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
`BCEWithLogitsLoss(pos_weight=9)` **inverteria a intenção**. Implementação aprovada e
executada: **`CrossEntropyLoss` com duas saídas e peso maior para a classe 0
(bonafide)**, razão 9:1. A saída da avaliação continua compatível com o protocolo:
**maior score = maior evidência de spoof**.

#### Bloco `espectrograma:` aprovado (já aplicado no `config.yaml` em B4.0)

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

Os defaults silenciosos do `librosa` 0.11 registrados explicitamente são
`window='hann'`, `pad_mode='constant'`, `htk=False`, `norm='slaney'` (filterbank Mel)
e `amin=1e-10` (piso do `power_to_db`). **Os dois últimos importam para a paridade com
o ramo clássico:** `librosa.feature.mfcc` usa `mel_norm='slaney'` e `htk=False`, logo
o espectrograma da CNN precisa usar os mesmos para que a filterbank seja a mesma.

#### Como descrever isso no texto (correções de redação exigidas)

Duas formulações estão **proibidas** por imprecisão técnica, e a banca notaria:

1. **Não escrever que subir o `n_fft` de 512 para 1024 "aumenta a resolução espectral
   real".** A janela continua em 400 amostras (~25 ms), então não se cria informação
   nova nem se altera a resolução física determinada pela janela. A formulação
   correta: *mantemos os mesmos 25 ms de análise e usamos FFT de 1024 para **aumentar
   a densidade de amostragem da DFT por zero-padding**, permitindo representar o banco
   de 128 filtros Mel de maneira numericamente mais adequada.*
2. **Não escrever que `ref=1.0` "preserva o ganho original das gravações".** O
   pipeline já faz **normalização de pico por áudio antes do VAD**, então esse ganho
   já foi removido a montante. A justificativa correta: *não introduzir uma segunda
   normalização, relativa ao máximo do próprio espectrograma, e manter paridade com a
   transformação que o MFCC já aplica.*

Registrar também a ressalva do `top_db=80`: o piso fica limitado a 80 dB abaixo do
máximo **daquele exemplo**. Não invalida a decisão, mas precisa estar documentado.

**A paridade entre os ramos descreve-se assim:** entregar Log-Mel à CNN reduz a
diferença entre os dois pipelines **principalmente à forma de representação e
aprendizado** — no ramo clássico resumimos as características acústicas e
classificamos com RF/SVM; na CNN preservamos a estrutura tempo-frequência e deixamos
a rede aprender a representação.

**Três formulações do Bloco 5 que também não se escrevem** (a versão correta está em
`COMPARACAO_FINAL.md`, gerada a partir dos números):

1. *«A CNN em GPU é o modelo mais barato.»* — 4,58 contra 4,92 ms do SVM é empate
   prático; o correto é «CNN-GPU e SVM empatam; em CPU os clássicos são mais baratos».
2. *«O teste mostrou sobreajuste do limiar.»* — os quatro deltas cabem no ruído; o
   recuo do f1 com EER parado é o **otimismo da seleção de limiar**.
3. *«X é ligeiramente melhor»* diante de um IC que contém zero — o correto é «não
   distinguíveis neste protocolo».

---

## Cronograma TC II — reemitido pelo orientador em 17/09

O cronograma original (25/08 a 09/09) foi **formalmente dado como perdido**. **O
calendário abaixo é o único que vale** — e suas datas são **teto**.

### Concluído

| bloco | teto | fechado em | estado |
|---|---|---|---|
| 1 — pipeline de features | 27/08 | no prazo | concluído |
| 2 — lote único de re-extração | 30/08 | no prazo | concluído — **features congeladas** |
| 3 — Random Forest e SVM | 02/09 | no prazo | concluído |
| B4.0 — fechamento metodológico | 18/09 | 17/09 | concluído |
| B4.1 — pipeline Log-Mel validado | 20/09 | 17/09 | concluído |
| B4.2 — lote definitivo (74.453) | 21/09 | 17/09 | concluído — **espectrogramas congelados** |
| B4.3 — split interno + normalização | 21/09 | 18/09 | concluído |
| B4.4 — CNN baseline | 23/09 | 18/09 | concluído |
| B4.5 — CNN definida | 25/09 | 18/09 | concluído |
| B4.6 — refit final nos 30k | 27/09 | 19/09 | concluído |
| B4.7 — validação externa | 28/09 | 20/09 | concluído |
| B5.1 — teste lacrado (execução única) | 29/09 | 23/09 | concluído — **teste usado** |
| B5.2 — comparação RF × SVM × CNN | 29/09 | 23/09 | concluído — **modelos congelados** |

### Vigente

| data limite | etapa | entrega obrigatória |
|---|---|---|
| **30/09 a 03/10** | Redação | Metodologia experimental, desenvolvimento, resultados, discussão das métricas, limitações e conclusão. **Pode começar já** — o Bloco 5 fechou em 23/09. |
| **04/10** | **VERSÃO COMPLETA PARA O ORIENTADOR** | **Trabalho inteiro. Não capítulo isolado, não versão parcial. Versão completa, avaliável do início ao fim.** |
| **05/10 a 07/10** | Revisão do orientador | Leitura e apontamentos dele. **Neste intervalo não se abrem experimentos novos.** |
| **08/10 a 09/10** | Correções finais | Incorporar integralmente as correções; revisar tabelas, figuras, texto, referências, citações e coerência metodológica. |
| **10/10** | Conferência final | Formatação, ABNT, numeração de figuras/tabelas, referências, sumário, texto e artefatos do Git. |
| **11/10** | **CONGELAMENTO FINAL** | **Trabalho concluído. Nenhuma etapa principal pode permanecer pendente depois desta data.** |

> **A data real de entrega é 04/10, não 11/10.** O que existe entre 04/10 e 11/10 é o
> tempo do orientador, não o meu. Se a primeira versão integral chegar perto de
> 11/10, não há tempo hábil nem para ele revisar nem para eu corrigir.
>
> **Não há mais margem para outro atraso semelhante ao de setembro.** Em compensação,
> o Bloco 4 fechou 8 dias antes do teto e o Bloco 5, 6 dias antes — essa folga é para
> a redação, não para experimento novo.

## Lista picada de tarefas

**Blocos 1 a 5 — concluídos.** (B1.1–B1.7, B2.1–B2.4, B3.1–B3.6, B4.0–B4.7, B5.1–B5.2
fechados; features congeladas no MD5 `51b2f439bf6f1e10237acbc620bb92d9`, 148.176
linhas; 74.453 espectrogramas congelados; CNN final com refit nos 30k e validação
externa; teste lacrado usado uma vez em 23/09; comparação consolidada em
`results/metricas/COMPARACAO_FINAL.md`.)

Bloco 5 — consolidação e teste lacrado (teto 29/09) — **fechado em 23/09**
- [x] **B5.1** avaliação final no teste lacrado — execução única, limiares já
      escolhidos, via `carregar_modelo_ajustado` / `scores_de` (commit `498f3c8`)
- [x] **B5.2** RF × SVM × CNN consolidados: tabela comparativa, matrizes, métricas,
      tempos, bootstrap pareado, análise por classe/ataque/codec, gráficos finais
      (commit `c07390b` + revisão das leituras). **Nenhuma mudança nos modelos daqui
      em diante, salvo erro grave.**

Bloco 6 — redação e fechamento (30/09 a 11/10)
- [ ] **B6.1 — 30/09 a 03/10** redação completa (pode começar antes)
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

**Some a isso o modo de fechamento.** O aprofundamento técnico deixou de ser a virtude
principal e passou a ser o principal risco. Portanto:

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
11. **Estando adiantado, não me segure.** Não peça autorização para seguir, não
    sugira esperar a data do marco chegar e não trate adiantamento como anomalia a
    confirmar. A regra de escopo serve para atraso; folga se gasta na fila declarada
    (redação primeiro), não em eixo experimental novo.

Antes de qualquer resposta sobre código ou resultados, **leia os arquivos reais da
pasta indexada**. Ao mexer em código, comente as decisões de design. **Nunca** rode a
re-extração de features nem a regeneração de espectrogramas sem decisão explícita
registrada — os dois lotes estão congelados. **Nunca** rode de novo o
`avaliar_teste_lacrado.py` — o teste já foi usado.

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

**Acrescentadas depois do B4.7:**

- **Por que o TF32 foi desligado, e o que teria acontecido se não fosse** — a resposta
  precisa citar que o erro do TF32 é da mesma ordem do espaçamento entre scores
  vizinhos e que isso desloca o limiar, não só "arredonda".
- **As duas armadilhas de medição de tempo em GPU, e por que nenhuma delas aparece
  lendo o código** — `synchronize()` ausente mede o enfileiramento (subestima ~31×);
  GPU fria mede o estado de baixo consumo (superestima ~10×). Uma infla, a outra
  desinfla.
- **Por que a ordem dos modelos muda entre latência unitária e vazão em lote**, e por
  que reportar só uma das duas é meio resultado.
- **Por que a vitória da CNN em acurácia não encerra a pergunta de pesquisa** — a
  pergunta é sobre acurácia *e* custo, e em CPU o resultado se inverte.
- **Por que as figuras da comparação saem de uma função compartilhada** (mesma régua
  de figura, pelo mesmo motivo que `avaliar` é a mesma régua de métrica).

**Acrescentadas depois do B5:**

- **Por que bootstrap PAREADO, e não IC individual de cada modelo** — os modelos são
  avaliados nas mesmas amostras, os erros são correlacionados; um vetor de índices por
  reamostragem, aplicado a todos.
- **Por que o f1 cai da validação para o teste com o EER parado** — otimismo da
  seleção de limiar: o f1 da validação é um máximo, e máximo transportado recua.
- **Vocês mudaram o critério do delta depois de ver o teste?** — sim, porque o
  primeiro ignorava a variância do próprio teste (errado a priori); e o delta é
  heurística — o que decide é o bootstrap pareado, que não mudou.
- **O bootstrap cobre a aleatoriedade do treino?** — não: reamostra a avaliação. A
  variância de treino é a das sementes (RF: ±0,0004; SVM: determinístico; CNN: semente
  única, limitação declarada ou fila item 2).
- **Como o EER é calculado** — no vértice da ROC mais próximo de FPR = FNR, sem
  interpolação; a diferença para o EER interpolado é ≤ 0,0003 nos quatro modelos.

---

## Histórico de revisões

**23/09/2026 (pós-Bloco 5)** — mudou:

1. *Contexto*: frente atual passa a ser o **Bloco 6 (B6.1)**; teste lacrado usado.
2. *Estado atual*: números do **teste lacrado**, do **bootstrap pareado** e a leitura
   corrigida de custo (CNN-GPU × SVM = empate prático).
3. *Decisões 3, 6 e 8*: teste executado e trancado; sementes extras só na validação;
   diferença de tempo menor que a variação entre sessões é empate.
4. *Formulações proibidas*: três do Bloco 5.
5. *Cronograma* e *lista picada*: B5.1 e B5.2 concluídos em 23/09.
6. *Frentes de banca*: cinco frentes do B5.
7. **Esta cópia Git estava na versão de 17/09** — a de 23/09 (pós-Bloco 4) existia só
   nas configurações do claude.ai. Exatamente o risco que a cópia existe para evitar;
   corrigido neste commit.

**23/09/2026 (pós-Bloco 4)** — revisão pós-Bloco 4. Mudou:

1. *Contexto*: "o Bloco 4 está destravado" → **Blocos 1 a 4 concluídos, frente atual é
   o Bloco 5**. A versão anterior mandava uma sessão nova se orientar por um mapa
   vencido.
2. Nova seção **Estado atual**, com os resultados da validação externa e os tempos
   ponta a ponta, para que uma sessão nova tenha os números sem varrer o repositório.
3. *Regra de escopo*: acrescentada a subseção **"é sobre atraso, não sobre ritmo"** e
   a **fila do tempo excedente**. A regra já existia em `00_MAPA_EXECUCAO.md` §3 e em
   `APENDICE_A_inventario.md`, mas não nas instruções — que, lidas sozinhas, soavam
   como calendário a cumprir e poderiam segurar o trabalho até a data do marco.
4. **Decisão 8** (nova): regime numérico (TF32 desligado) e protocolo de medição de
   tempo (aquecimento de GPU, medição conjunta). Não é eixo experimental novo — é
   invariante já implementado em `scores_de` e commitado em B4.7, registrado aqui
   porque **B5.1 depende dele**.
5. **Decisão 6** atualizada: as 3 sementes da CNN deixaram de ser hipotéticas — custo
   medido de ~15 min/semente. Continuam fora do caminho crítico.
6. *Cronograma*: acrescentada a coluna **"fechado em"** para o Bloco 4, que documenta
   os 8 dias de adiantamento; as linhas já vencidas saíram da tabela "vigente".
7. *Papel do Claude*: **item 11** — estando adiantado, não segurar o trabalho.
8. *Frentes de banca*: cinco frentes novas, vindas do B4.7.

**17/09/2026** — versão do orientador: cronograma de recuperação reemitido, sete
decisões do espectrograma fechadas, regra de escopo instituída.
