# Comparação final — RF × SVM × CNN (B5.2)

> **GERADO** por `scripts/comparacao_final.py` em 2026-09-23 a partir dos JSONs de métricas — **não editar à mão**: a reexecução sobrescreve. Toda frase de leitura abaixo é montada a partir dos números.

## 1. Teste lacrado (22.227) — execução única, B5.1

| modelo | braço | n treino | limiar | acurácia | f1_bonafide | f1_spoof | **f1_macro** | **EER** | ROC-AUC | recall bonafide | latência ms | ms/áudio (lote) | pipeline completo ms | hardware |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RF | principal | 30.000 | 0,6516 | 0,8940 | 0,5013 | 0,9407 | **0,7210** | **0,1946** | 0,8878 | 0,5326 | 6,1075 | 0,0204 | 10,23 | CPU, 1 thread |
| RF (referência) | referência | 103.723 | 0,6196 | 0,9123 | 0,5692 | 0,9512 | **0,7602** | **0,1629** | 0,9167 | 0,5794 | 7,8572 | 0,0290 | não medido | CPU, 1 thread |
| SVM | principal | 30.000 | -0,0329 | 0,9247 | 0,6381 | 0,9580 | **0,7981** | **0,1411** | 0,9315 | 0,6635 | 0,6285 | 0,4025 | 4,92 | CPU, 1 thread |
| CNN | principal | 30.000 | 0,3252 | 0,9603 | 0,7986 | 0,9780 | **0,8883** | **0,0737** | 0,9783 | 0,7877 | 1,1093 (GPU) / 14,2110 (CPU) | 0,2383 (GPU) / 15,4073 (CPU) | 4,58 (GPU) / 16,80 (CPU) | GPU NVIDIA GeForce RTX 5060 Ti / CPU, 1 thread |

## 2. Validação (22.226) — onde os limiares foram escolhidos

| modelo | braço | n treino | limiar | acurácia | f1_bonafide | f1_spoof | **f1_macro** | **EER** | ROC-AUC | recall bonafide | latência ms | ms/áudio (lote) | pipeline completo ms | hardware |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RF | principal | 30.000 | 0,6516 | 0,8933 | 0,5048 | 0,9402 | **0,7225** | **0,1930** | 0,8873 | 0,5441 | 6,1075 | 0,0204 | 10,23 | CPU, 1 thread |
| RF (referência) | referência | 103.723 | 0,6196 | 0,9168 | 0,5908 | 0,9537 | **0,7723** | **0,1579** | 0,9191 | 0,6008 | 7,8572 | 0,0290 | não medido | CPU, 1 thread |
| SVM | principal | 30.000 | -0,0329 | 0,9252 | 0,6392 | 0,9583 | **0,7987** | **0,1462** | 0,9289 | 0,6625 | 0,6285 | 0,4025 | 4,92 | CPU, 1 thread |
| CNN | principal | 30.000 | 0,3252 | 0,9633 | 0,8154 | 0,9796 | **0,8975** | **0,0738** | 0,9799 | 0,8101 | 1,1093 (GPU) / 14,2110 (CPU) | 0,2383 (GPU) / 15,4073 (CPU) | 4,58 (GPU) / 16,80 (CPU) | GPU NVIDIA GeForce RTX 5060 Ti / CPU, 1 thread |

Regra de decisão `score >= limiar`, com o limiar selecionado na validação e só aplicado no teste. Latência e ms/áudio (lote) medem só a predição, a partir da representação já extraída (`tempos_inferencia` de cada JSON); pipeline completo é `tempo_pipeline_completo.json` — medições distintas. RF (referência) não é concorrente direto — ver frase 1.

## 3. Validação → teste

| modelo | f1_macro val | f1_macro teste | Δf1 | ruído Δf1 | ΔEER | ruído ΔEER | dentro do ruído? |
|---|---|---|---|---|---|---|---|
| RF | 0,7225 | 0,7210 | -0,0015 | ±0,0133 | 0,0016 | ±0,0115 | sim |
| RF (referência) | 0,7723 | 0,7602 | -0,0121 | ±0,0132 | 0,0050 | ±0,0105 | sim |
| SVM | 0,7987 | 0,7981 | -0,0007 | ±0,0126 | -0,0051 | ±0,0108 | sim |
| CNN | 0,8975 | 0,8883 | -0,0092 | ±0,0098 | -0,0001 | ±0,0079 | sim |

*HEURÍSTICA — ruído da diferença entre duas amostras independentes, 1,96·√(dp_val² + dp_teste²), aproximação normal.* Os quatro deltas validação→teste (f1_macro e EER) cabem no ruído de amostragem dos dois conjuntos: a seleção de limiar sobre 22 mil amostras não sobreajustou a validação — evidência de que o protocolo funcionou. Mais perto do limite do ruído: RF (referência) e CNN (Δf1 -0,0121 para um ruído de ±0,0132 e -0,0092 para um ruído de ±0,0098). Os quatro Δf1 são negativos — o sinal esperado quando o limiar é escolhido maximizando o f1 no próprio conjunto de validação: o f1 da validação é otimista por construção (otimismo da seleção de limiar), e recua um pouco quando o limiar é transportado. Para a CNN, o EER praticamente não muda (-0,0001) enquanto o f1_macro cai: como o EER não depende de limiar, o teste não ficou mais difícil para esse modelo — a queda do f1 é inteiramente o otimismo da seleção de limiar, não uma perda de capacidade de separar as classes. A ordem dos três modelos principais é a mesma na validação e no teste.

## 4. Bootstrap pareado — o teste que decide

1.000 reamostragens, semente 42; um vetor de índices por reamostragem aplicado aos scores de todos os modelos; limiar fixo no do protocolo. Conferido: a extensão para três modelos não mudou o teste: mesmo sorteio de índices, mesmas linhas, mesmos scores — o par SVM−RF sai idêntico ao do Bloco 3.

| conjunto | diferença | Δf1_macro | IC95 | ΔEER | IC95 | 1º vence (f1 / EER) | f1_macro | EER |
|---|---|---|---|---|---|---|---|---|
| validacao | SVM − RF | 0,0762 | [0,0663; 0,0856] | -0,0468 | [-0,0560; -0,0377] | 100,0% / 100,0% | SVM melhor (real) | SVM melhor (real) |
| validacao | CNN − RF | 0,1750 | [0,1650; 0,1853] | -0,1192 | [-0,1275; -0,1102] | 100,0% / 100,0% | CNN melhor (real) | CNN melhor (real) |
| validacao | CNN − SVM | 0,0988 | [0,0887; 0,1093] | -0,0724 | [-0,0812; -0,0635] | 100,0% / 100,0% | CNN melhor (real) | CNN melhor (real) |
| validacao | RF (referência) − RF *(custo da subamostragem — não é par da comparação)* | 0,0498 | [0,0438; 0,0561] | -0,0351 | [-0,0406; -0,0292] | 100,0% / 100,0% | RF (referência) melhor (real) | RF (referência) melhor (real) |
| teste | SVM − RF | 0,0771 | [0,0671; 0,0868] | -0,0535 | [-0,0630; -0,0444] | 100,0% / 100,0% | SVM melhor (real) | SVM melhor (real) |
| teste | CNN − RF | 0,1673 | [0,1568; 0,1775] | -0,1209 | [-0,1304; -0,1118] | 100,0% / 100,0% | CNN melhor (real) | CNN melhor (real) |
| teste | CNN − SVM | 0,0902 | [0,0801; 0,1005] | -0,0674 | [-0,0763; -0,0582] | 100,0% / 100,0% | CNN melhor (real) | CNN melhor (real) |
| teste | RF (referência) − RF *(custo da subamostragem — não é par da comparação)* | 0,0392 | [0,0329; 0,0451] | -0,0317 | [-0,0368; -0,0252] | 100,0% / 100,0% | RF (referência) melhor (real) | RF (referência) melhor (real) |

IC que não contém zero ⇒ a diferença é real neste protocolo. IC que contém zero ⇒ «não distinguíveis neste protocolo».

## 5. As três frases obrigatórias

**1. Sobre o braço de referência.** O braço de referência (RF no treino completo de 103.723) não é concorrente direto de SVM e CNN: ele existe para quantificar o custo da subamostragem de 30k, que foi imposta pela complexidade O(n²)–O(n³) do SVM-RBF. A comparação entre modelos é a do braço principal, em que os três viram as mesmas 30.000 amostras. Medido: no teste o RF de referência chega a f1_macro 0,7602 contra 0,7210 do RF principal (Δ 0,0392, IC95 [0,0329; 0,0451]); na validação, Δ 0,0498. É esse o custo da subamostragem para o RF — e, com 3,5× mais dados, ele ainda fica abaixo do SVM (0,7981) e da CNN (0,8883) treinados em 30.000.

**2. Sobre o EER e a literatura.** EER no teste: RF 19,46%, SVM 14,11%, CNN 7,37%. Esses valores NÃO são comparáveis ao EER de 1,32% de Yamagishi et al. (2022): o protocolo aqui é um split interno aleatório POR UTTERANCE — os mesmos ataques (A07–A19), codecs e locutores aparecem em treino e avaliação —, enquanto o protocolo oficial do ASVspoof é deliberadamente CROSS-ATTACK. Ser independente de limiar remove a arbitrariedade do 0,50; NÃO remove a diferença de protocolo.

**3. Sobre o custo.** Não existe «o modelo mais barato» sem dizer o regime: por áudio, ponta a ponta, o SVM é ~2,1× mais barato que o RF (4,92 contra 10,23 ms), mas em lote o RF é ~19,7× melhor (0,0204 contra 0,4025 ms por áudio). A CNN acrescenta a dimensão que de fato importa para a pergunta de pesquisa — exige GPU ou não? — e é o par CNN-GPU / CNN-CPU que responde. Ponta a ponta, a CNN em GPU (4,58 ms) e o SVM (4,92 ms) empatam na prática como os cenários mais baratos: nesta execução a diferença é de 7,5% e as faixas mín–máx (soma por etapa) não se sobrepõem, mas a diferença fica dentro da faixa de variação (4,1–13,7%) que os tempos de predição da CNN mostraram entre duas sessões da mesma máquina — a ordem entre os dois vale para esta execução, não para outra sessão ou máquina. A CNN em CPU (16,80 ms) é a mais cara (3,7× a própria versão em GPU); em lote a diferença CPU/GPU chega a 64,7×. Em igualdade de hardware (CPU) os dois clássicos são mais baratos ponta a ponta que a CNN (4,92 e 10,23 contra 16,80 ms); o quadro só se inverte quando a CNN ganha uma GPU. A vantagem de custo dos clássicos, portanto, é NÃO EXIGIREM GPU — e é contra essa vantagem que o ganho de desempenho da CNN tem de ser pesado.

## 6. Por ataque e por codec (validação)

**Por ataque.** RF: amplitude 0,2920 supera a distância agregada para os dois outros modelos; SVM: amplitude 0,1981 supera a distância agregada para os dois outros modelos; CNN: amplitude 0,1171 é da mesma ordem da distância para o RF (0,1192; diferença 0,0021, dentro do piso de ruído de ±0,0086) e supera a distância para o SVM (0,0724). A leitura do Bloco 3 vale integralmente para o RF e o SVM; para a CNN, a variação entre ataques deixa de dominar e fica do mesmo tamanho da distância entre modelos: a amplitude entre ataques cai de 0,2920 (RF) para 0,1981 (SVM) e 0,1171 (CNN), enquanto a maior distância entre modelos é 0,1192. O ataque A16 está entre os três mais difíceis nos três modelos — a dificuldade é, em parte, do sistema de síntese e não do classificador. Em razão (EER do mais difícil / do mais fácil) a CNN é a mais desigual (11,09x), porque o seu ataque mais fácil fica perto de zero. A amplitude grande em qualquer dos modelos segue sendo evidência a favor do risco declarado na limitação do split (por utterance, os mesmos vocoders em treino e avaliação).

**Por codec.** *(leitura DESCRITIVA — não há IC para a diferença entre ganhos)* A banda larga é melhor nos três modelos (f1_macro maior E EER menor), e o EER é independente de limiar — o contraste não é artefato do limiar. Ganho larga − estreita: RF Δf1 0,0552, ΔEER -0,0380 (18,5% do EER); SVM Δf1 0,0686, ΔEER -0,0478 (28,9% do EER); CNN Δf1 0,0524, ΔEER -0,0314 (37,0% do EER). Em termos ABSOLUTOS o ganho da CNN não é o maior e em termos RELATIVOS (fração do EER da banda estreita removida) é o maior dos três. As duas escalas apontam em sentidos opostos, e sem IC para a diferença entre ganhos a hipótese do marco fica NÃO DECIDIDA: não se sustenta em escala absoluta e é compatível com os dados em escala relativa. O que a CNN acrescenta com segurança é que o contraste aparece também num modelo que não usa ZCR nem centróide — a evidência é sobre o sinal, não sobre aquelas duas features.

## 7. Heurísticas — contexto, NÃO o teste

IC individual e dispersão por semente ignoram a correlação entre os erros dos modelos nas MESMAS amostras; comparar a diferença contra a maior dispersão individual superestima o ruído da comparação. O teste é o IC da diferença pareada (bloco `bootstrap_pareado`).

- RF, variância de treino (5 sementes): f1_macro 0,7220 ± 0,0004.
- SVM: não existe: SVC com probability=False é determinístico.
- CNN: não medida: a CNN final é um treino único (refit do B4.6, semente 42); re-treinar com outras sementes está fora do escopo do B5.

| conjunto | modelo | f1_macro IC95 individual | EER IC95 individual |
|---|---|---|---|
| validacao | RF | [0,7124; 0,7315] | [0,1845; 0,2005] |
| validacao | RF (referência) | [0,7629; 0,7814] | [0,1503; 0,1641] |
| validacao | SVM | [0,7898; 0,8074] | [0,1376; 0,1533] |
| validacao | CNN | [0,8905; 0,9043] | [0,0678; 0,0791] |
| teste | RF | [0,7114; 0,7300] | [0,1862; 0,2019] |
| teste | RF (referência) | [0,7512; 0,7689] | [0,1557; 0,1714] |
| teste | SVM | [0,7890; 0,8062] | [0,1330; 0,1482] |
| teste | CNN | [0,8812; 0,8949] | [0,0671; 0,0787] |

## Fontes

- `results/metricas/rf_tuned_principal.json`
- `results/metricas/rf_tuned_referencia.json`
- `results/metricas/svm_tuned_principal.json`
- `results/metricas/cnn_final_principal.json`
- `results/metricas/teste_lacrado.json`
- `results/metricas/tempo_pipeline_completo.json`
- `results/metricas/estabilidade_rf_svm.json`
- `results/metricas/diagnostico_por_ataque_resumo.json`
- `results/metricas/diagnostico_por_codec_resumo.json`
- `results/metricas/scores_validacao.csv`
- `results/metricas/scores_teste_lacrado.csv`
- `results/metricas/_pre_revisao/cnn_final_principal.json + cnn_final_principal_reexecucao.json (variação dos tempos entre sessões)`
