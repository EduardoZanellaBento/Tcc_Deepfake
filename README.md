# Detecção Automática de Voz Sintética com Machine Learning

TCC — Eduardo Zanella Bento · Ciência da Computação · UNIP São José do Rio Preto · 2026
Orientador: Prof. Anderson Fola

## Objetivo

Comparar, **dentro de um mesmo protocolo experimental**, classificadores clássicos
(Random Forest e SVM) alimentados por features acústicas manuais (MFCC, ZCR e
Centróide Espectral) contra uma Rede Neural Convolucional (CNN) operando sobre
espectrogramas, avaliando tanto desempenho preditivo quanto viabilidade computacional.

**Pergunta de pesquisa:** modelos clássicos baseados em características acústicas manuais
mantêm desempenho competitivo, em acurácia e custo computacional, frente a CNNs na
detecção de deepfakes de áudio dentro do mesmo ambiente experimental?

## Dataset

ASVspoof 2021 — subconjunto **Logical Access (LA)**.

> ATENÇÃO METODOLÓGICA: o ASVspoof 2021 LA NÃO fornece dados de treino/dev próprios.
> Estratégia adotada neste trabalho: usar o conjunto de **avaliação** do 2021 LA + arquivo
> de chaves (labels), e aplicar split interno estratificado (70/15/15) + Stratified 5-Fold
> Cross-Validation. Limitação assumida: não há teste de generalização cross-dataset.

## Decisões metodológicas fechadas (aprovadas pelo orientador)

1. **Subamostra estratificada única de ~30k para o treino compartilhado.**
   O SVM-RBF é O(n²)–O(n³) e não roda nos ~103k de treino; uma única subamostra
   (classe × codec × ataque, seed 42), compartilhada por RF, SVM e CNN, preserva o
   "mesmo ambiente experimental" da pergunta de pesquisa.
   → `scripts/gerar_subamostra.py`, IDs em `data/processed/subamostra_30k.csv`.
2. **Universo do experimento = `fase == 'eval'` (148.176 áudios).**
   `progress` (16.464) e `hidden` (16.926) foram excluídos por controle metodológico:
   o `eval` é o conjunto oficialmente pontuado do ASVspoof 2021 LA, e o `hidden` tem
   silêncio pré-cortado na origem (`trim == 'only_speech'`), um pré-processamento
   distinto que contamina qualquer análise de proporção de fala.
   → `scripts/composicao_eval.py`, filtro em `src/data/split.py`.
3. **Sem `fmax=4000` global; métricas reportadas por codec.**
   58% do universo eval é banda estreita (alaw/ulaw/gsm/pstn, teto ~4 kHz) e 42%
   banda larga (g722/opus/none): filtrar tudo em 4 kHz destruiria justamente a banda
   alta onde vivem artefatos de síntese. → `scripts/diagnostico_por_codec.py`.
4. **Split aleatório por utterance mantido como baseline, com limitação declarada**
   (ver seção "Limitação do split" abaixo). → `src/data/split.py`.

## Universo do experimento

`fase == 'eval'`: **148.176 áudios** (14.816 bonafide, 133.360 spoof, razão 9,0:1),
`trim == 'notrim'` em 100% das linhas (verificado por script — a exclusão do `hidden`
é comprovada, não presumida). Composição completa por classe, codec e ataque em
`results/metricas/eval_composicao_*.csv` e `eval_composicao_resumo.json`.

## Descrição do split

```
148.176 áudios (fase == 'eval')
    |
    +-- TREINO 70% (103.723) --> é AQUI que o Stratified 5-Fold roda,
    |                            durante a busca de hiperparâmetros.
    |                            O 5-fold particiona SÓ o treino.
    |
    +-- VALIDAÇÃO 15% (22.226) -> comparar modelos / decisões de projeto
    |
    +-- TESTE 15% (22.227) ----> INTOCADO até o resultado final. Usado UMA vez.
```

**Por que três conjuntos e não dois:** se hiperparâmetros são escolhidos olhando o
teste, o teste deixa de ser estimativa honesta de generalização. A validação absorve
esse desgaste; o teste é a prova final, usada uma única vez.

**Por que estratificado:** com 9:1 de desbalanceamento, um split aleatório simples
deixa a proporção oscilar entre subconjuntos; a estratificação por `classe_binaria`
força cada subconjunto a manter a proporção original (conferência automática no
`resumo_split`).

O split é um **artefato versionado** (`data/processed/split.csv`, hash MD5
`9143f0c7b83ec2db4aa144ed5deb3402`), idêntico para RF, SVM e CNN. O split anterior,
do universo 181.566, está preservado em `data/processed/split_181k.csv` (hash
`58cea82b2513c0f5c1e5797895a92571`) — é ele que gerou o baseline histórico.

### Limitação declarada do split

O split é aleatório **por utterance**: cada ataque (A07–A19), codec e locutor aparece
em treino E teste. O modelo pode memorizar a assinatura de um vocoder/locutor
específico, então as métricas são **potencialmente otimistas** e **não comparáveis**
ao EER de 1,32% de Yamagishi et al. (2022), cujo protocolo é deliberadamente
cross-attack. Mitigação adotada: métricas por ataque (`diagnostico_por_ataque.py`) e
por codec (`diagnostico_por_codec.py`).

> **Atualização de 03/09/2026:** essa mitigação passou a ser medida **sobre os
> modelos ajustados** (RF **e** SVM do braço principal), com o **limiar do
> protocolo** — e não mais sobre o RF baseline em 0,50. A versão anterior era
> vazia: em 0,50 o modelo previa "spoof" para quase tudo e o recall dava 1,0 em
> todos os 13 ataques. Artefatos:
> `diagnostico_por_ataque_resumo.json` e `diagnostico_por_codec_resumo.json`
> (campo `leitura_critica`); os arquivos de 13/08 ficam preservados como
> `*_baseline_2026-08-13.*`.

Experimento de robustez (split por ataque / leave-one-attack-out) previsto como
**análise complementar**, após fechar RF, SVM e CNN.

## Braço principal × braço de referência

A curva de aprendizado mostra que o RF **não satura** antes do treino completo —
e isso se mantém depois do lote único e do ajuste de hiperparâmetros
(`curva_aprendizado_rf_tuned_eval.json`, features congeladas, config ajustada,
limiar selecionado na validação em cada ponto):

| n treino | f1_macro | EER | limiar |
|---:|---:|---:|---:|
| 5.000 | 0,6764 | 0,2274 | 0,6937 |
| 10.000 | 0,6953 | 0,2128 | 0,6787 |
| 20.000 | 0,7156 | 0,2002 | 0,6795 |
| 30.000 | 0,7255 | 0,1891 | 0,6803 |
| 40.000 | 0,7365 | 0,1873 | 0,6485 |
| 80.000 | 0,7602 | 0,1668 | 0,6393 |
| 103.723 | **0,7723** | **0,1579** | 0,6196 |

O último ponto **reproduz exatamente** `rf_tuned_referencia.json` (0,7723 / 0,1579):
por construção ele é o braço de referência, e essa coincidência é a checagem embutida
da curva. O ponto de 30.000 (0,7255 / 0,1891) fica bem próximo do braço principal
medido sobre a `subamostra_30k.csv` (0,7225 / 0,1930) — evidência de que a subamostra
não é atípica. **Atenção:** o ponto de 30k da curva é estratificado por
classe × codec e **não é** a `subamostra_30k.csv` (classe × codec × ataque).

A curva **anterior** (`curva_aprendizado_rf_eval.json`, RF baseline sobre as features
pré-mascaramento, decisão por argmax) fica preservada como referência "antes":
f1_macro ia de 0,4760 (5k) a 0,5573 (103.723), EER de 0,3030 a 0,1818.

Logo, a subamostra de 30k **custa desempenho real**. Por isso o experimento tem dois
braços:

- **Braço principal (comparação):** RF, SVM e CNN treinados na subamostra de 30k —
  único arranjo em que a comparação entre modelos é limpa (mesmo ambiente
  experimental, exigência da pergunta de pesquisa). Validação e teste **completos**.
- **Braço de referência:** RF treinado no treino completo do eval (custa ~8 s) —
  quantifica exatamente quanto a subamostra custou. Entra como linha extra na tabela
  final.

Implementação: a chave `experimento.braco` do `config.yaml` é lida por
`filtrar_treino_braco` (`src/data/split.py`); `treinar_rf.py` roda os dois braços e
salva artefatos com sufixo `_principal`/`_referencia` (o `rf_baseline_eval.*`
pré-braço-duplo fica preservado). O SVM, quando existir, roda só no principal.

## Como rodar cada script

Sempre a partir da **raiz**, com `python -m` (imports relativos):

```bash
# pipeline principal
python -m src.data.carregar_dados            # regenera labels.csv do trial_metadata.txt
python -m src.data.split                     # gera/recarrega o split 70/15/15 (universo eval)
python -m src.models.treinar_rf              # RF baseline nos DOIS braços -> rf_baseline_eval_{principal,referencia}.{json,joblib} + matrizes de confusão
python -m src.models.ajustar_rf              # Bloco 3: Random Search (EER, 5-fold no treino do braço principal) + treino final nos DOIS braços -> rf_tuned_{principal,referencia}.*
python -m src.models.treinar_svm             # Bloco 3: SVM RBF (só braço principal) — cronometra 1 fit antes da busca; decision_function -> svm_tuned_principal.*
python -m src.features.extrair_features     # ATENÇÃO: features CONGELADAS (lote único de 30/08); a guarda de esquema aborta retomadas inválidas — ver aviso abaixo

# composição e subamostra
python -m scripts.composicao_eval            # composição por classe/codec/ataque no eval (verifica 148.176)
python -m scripts.gerar_subamostra           # subamostra 30k estratificada do treino (só a lista de IDs)

# diagnósticos
python -m scripts.diagnostico_composicao_dataset   # composição fase/trim/codec do dataset bruto (181k)
python -m scripts.diagnostico_vazamento_duracao    # prop_fala como atalho de duração (árvore só com prop_fala)
python -m scripts.diagnostico_limiar               # ROC, PR e varredura de limiar do RF na validação
python -m scripts.diagnostico_por_ataque           # Bloco 3: RF **e** SVM AJUSTADOS por ataque A07–A19, com o limiar do protocolo (carrega os .joblib, não re-treina)
python -m scripts.diagnostico_por_codec            # Bloco 3: RF **e** SVM AJUSTADOS por codec (banda estreita × larga), com o limiar do protocolo
python -m scripts.curva_aprendizado_rf             # curva de aprendizado -> evidência do braço duplo
python -m scripts.curva_aprendizado_rf_tuned       # Bloco 3: curva com config AJUSTADA + features congeladas (artefatos _tuned_eval)
python -m scripts.ablacao_mfcc1_std                # Bloco 3: ablação da mfcc1_std com bootstrap pareado (rodar após ajustar_rf)
python -m scripts.diagnostico_padding_features     # correlação das 44 features com prop_fala e n_frames_validos (pós-lote; o "antes" está em *_pre_lote.*)
python -m scripts.diagnostico_rf_pareado_propfala  # RF com classes pareadas por faixa de prop_fala
python -m scripts.piloto_mascaramento_padding      # piloto A/B de mascaramento (2.000 áudios, arquivo separado)
python -m scripts.verificar_mascaramento           # CHECAGEM do bloco 1: A/B controlado + esquema do CSV (rodou ANTES do lote único)
python -m scripts.validar_split_pos_lote           # split/subamostra preservados e íntegros após o lote único (hashes + IDs)
python -m scripts.teste_reproducao_limiar          # TESTE: a avaliar/selecionar_limiar novas reproduzem os 0,5598 e as 25 amostras de nota_divergencia_f1.md
python -m scripts.estabilidade_modelos             # Bloco 3: 5 sementes no RF + bootstrap da validação em RF e SVM (rodar após ajustar_rf e treinar_svm)
python -m scripts.estabilidade_subamostra          # Bloco 3: 3 subamostras alternativas (sementes 43-45) — a fonte de variação que domina o braço principal
python -m scripts.extrapolacao_curva_rf            # Bloco 3: ajuste f1 ~ a·ln(n) + b e n necessário para o RF alcançar o SVM (extrapolação, ver campo `limitacao`)
python -m scripts.importancia_permutacao_rf        # Bloco 3: permutation_importance na validação × importância por impureza (rodar após ajustar_rf)
python -m scripts.tempo_pipeline_completo          # Bloco 3: tempo ponta a ponta (carregar → VAD → features → predizer); src/models/tempo.py mede só a última etapa
python -m scripts.guarda_reproducao                # Bloco 3: re-execução reproduz os JSONs de _pre_revisao/? -> reproducao_bloco3.json

python scripts/verificar_ambiente.py         # sanidade do ambiente (versões, GPU, pastas)
```

## Resultados atuais (conjunto de validação)

### Bloco 3 — features congeladas, hiperparâmetros ajustados, limiar selecionado na validação

| Configuração | n treino | limiar | f1_macro | EER | Arquivo |
|---|---:|---:|---:|---:|---|
| RF ajustado, **braço principal** (30k) | 30.000 | 0,6516 | **0,7225** | 0,1930 | `results/metricas/rf_tuned_principal.json` |
| RF ajustado, **braço de referência** (treino completo, mesma config) | 103.723 | 0,6196 | **0,7723** | 0,1579 | `results/metricas/rf_tuned_referencia.json` |
| SVM RBF ajustado, braço principal | 30.000 | −0,0329* | **0,7987** | 0,1462 | `results/metricas/svm_tuned_principal.json` |

\* O limiar do SVM vive na escala do `decision_function` (real, centrada em zero),
não em [0, 1] como o `predict_proba` do RF — não é inconsistência: a regra do
protocolo ("selecionar na validação, aplicar no teste") é agnóstica de escala, e
`selecionar_limiar` opera sobre `np.unique(scores)` sem supor intervalo.

Protocolo comum aos modelos (`src/models/avaliacao.py`): decisão por
`score >= limiar`, limiar **selecionado na validação** maximizando f1_macro sobre
os `np.unique(scores)` e apenas **aplicado** em qualquer outro conjunto; busca de
hiperparâmetros por **EER** (independente de limiar — justificativa em
`rf_random_search.json`), `StratifiedKFold(5)` só no treino do braço principal.
Estabilidade (`estabilidade_rf_svm.json`) — **o estatístico que decide é o
bootstrap pareado**: RF e SVM são avaliados exatamente nas mesmas 22.226
amostras, então os erros dos dois são correlacionados, e é a diferença
reamostrada em conjunto (1.000 vezes, um único vetor de índices por
reamostragem) que responde se a vantagem é real. Resultado:
Δf1_macro = **+0,0762**, IC95 [0,0663; 0,0856]; ΔEER = **−0,0466**,
IC95 [−0,0560; −0,0377]. **Nenhum IC contém zero**, e o SVM é melhor em
**100% das 1.000 reamostragens** nas duas métricas. A vantagem do SVM no braço
principal é real, não ruído.

*Heurísticas complementares* (úteis para contexto, não são o teste): o RF entre
5 sementes varia ±0,0004 de f1_macro; o bootstrap individual de cada modelo dá
IC95 [0,712; 0,732] para o RF e [0,790; 0,808] para o SVM — a diferença RF×SVM
é maior que qualquer uma dessas dispersões, mas comparar contra a maior
dispersão *individual* ignora a correlação entre os erros dos dois modelos, e é
por isso que o próprio `estabilidade_rf_svm.json` a rotula como heurística em
`leitura_critica.nota_metodo`. O SVM com `probability=False` é determinístico
(não há variância de treino a medir; a estabilidade dele é só a de estimativa,
por bootstrap). A fonte de variação que o braço principal de fato tem —
**qual subamostra de 30k caiu** — é medida em `estabilidade_subamostra.json`
(`scripts/estabilidade_subamostra.py`).

### Referência "antes" (baseline argmax, features pré-mascaramento)

| Configuração | n treino | f1_macro | EER | Arquivo |
|---|---:|---:|---:|---|
| RF baseline histórico (universo 181k) | 127.096 | 0,5675 | 0,1761 | `results/metricas/rf_baseline.json` |
| RF baseline **eval** (universo aprovado) | 103.723 | 0,5573 | 0,1818 | `results/metricas/rf_baseline_eval.json` |
| RF **braço principal** (subamostra 30k) | 30.000 | 0,4980 | 0,2367 | `results/metricas/rf_baseline_eval_principal.json` |
| RF **braço de referência** (treino completo) | 103.723 | 0,5573 | 0,1818 | `results/metricas/rf_baseline_eval_referencia.json` |

Valores medidos sobre o `features.csv` **antigo** (pré-mascaramento), com decisão
por `predict()`/argmax no limiar implícito 0,50 — a referência "antes" da
comparação (ressalva registrada no cabeçalho das notas técnicas correspondentes).

Métricas por codec e por ataque **deste baseline** (13/08/2026, limiar 0,50):
`diagnostico_por_codec_baseline_2026-08-13.csv` e
`diagnostico_por_ataque_baseline_2026-08-13.csv` — as versões sobre os modelos
**ajustados** estão na seção acima. Diagnóstico do atalho de silêncio:
`rf_pareado_propfala.json`, `padding_corr_features_propfala_{pre,pos}_lote.csv` e
`RECOMENDACAO_MASCARAMENTO.md` (adendos de 26/08 e pós-lote).

### Nota sobre o limiar de decisão

O f1_macro baixo do baseline com AUC alta (~0,90) **não** significava ausência de
sinal: o par (`class_weight='balanced'`, limiar 0,50) estava desajustado — as
folhas puras do RF neutralizam o peso de classe e o limiar ótimo fica longe de
0,50. O Bloco 3 implementou as três correções da nota (min_samples_leaf no espaço
de busca, `balanced_subsample` como alternativa e seleção de limiar na validação
como **regra de protocolo** para RF, SVM e CNN, em `src/models/avaliacao.py` —
com teste de reprodução em `scripts/teste_reproducao_limiar.py`). Causa raiz e
histórico: **`results/metricas/NOTA_LIMIAR.md`** e `nota_divergencia_f1.md`.

## ⚠️ Re-extração de features — LOTE ÚNICO EXECUTADO, features CONGELADAS

O lote único de re-extração rodou em **30/08/2026** (148.176 áudios, 6 min 15 s,
zero erros — dossiê completo em `results/metricas/DOSSIE_LOTE_UNICO.md`). O
`data/features/features.csv` atual (hash MD5 `51b2f439bf6f1e10237acbc620bb92d9`)
reflete as quatro pendências abaixo e está **CONGELADO**: nova re-extração só por
erro grave, com decisão registrada. O CSV anterior está arquivado como
`features_pre_mascaramento_c01c3c5c.csv` (hash `c01c3c5c6afdcad0dd95236ffd6910ad`).

`python -m src.features.extrair_features` agora é protegido por uma **guarda de
esquema/retomada** (`_validar_retomada`): retomar sobre um CSV de outra definição
de feature (cabeçalho ou `features.meta.json` divergentes) aborta com erro
explícito, em vez de produzir um CSV meio antigo, meio novo.

O `split.csv` e a `subamostra_30k.csv` foram **preservados** através do lote (o
split é partição de IDs; a re-extração muda valores, não o conjunto de arquivos) e
revalidados por `python -m scripts.validar_split_pos_lote` — hashes inalterados,
conjuntos de IDs idênticos (justificativa completa na seção 7 do dossiê).

### Pendências do lote único — APROVADAS e APLICADAS no features.csv em 30/08/2026

1. **`win_length` do centróide espectral** — o CSV antigo foi extraído sem
   `win_length` no `spectral_centroid` (default `n_fft=512` em vez dos 400 de
   MFCC/ZCR), deixando o centróide com resolução temporal diferente das demais
   features. ✔ corrigido em `extrair_vetor`.
2. **Filtro `fase == 'eval'`** — extrair somente o universo aprovado (148.176), e não
   os 181.566 do `labels.csv`. ✔ aplicado dentro de `executar()`.
3. **Mascaramento de padding na agregação temporal** — média e desvio calculados
   apenas sobre os frames válidos (centro do frame dentro do áudio real pós-VAD).
   ✔ implementado; governado por `features.mascarar_padding` no config.
   Justificativa: `RECOMENDACAO_MASCARAMENTO.md` (+ adendo de 26/08) e
   `checagem_mascaramento.json`.
4. **`n_frames_validos` e `n_frames_total` como colunas de diagnóstico** — no CSV,
   mas **fora do X**, junto com `prop_fala`. ✔ `COLUNAS_DIAGNOSTICO` em
   `src/features/extrair_features.py`, consumida por `colunas_features`
   (`src/data/split.py`), que é o ponto único que define o X.

**Checagem obrigatória antes do lote:** `python -m scripts.verificar_mascaramento` —
prova, em teste A/B controlado, que o mascaramento altera as features; mede a
assimetria da distorção entre as classes; e valida o esquema do CSV gerado pelo
runner de produção (50 colunas = 3 identificação + 3 diagnóstico + 44 features;
`colunas_features` devolvendo exatamente 44; sem NaN; universo só `eval`). Evidência
em `results/metricas/checagem_mascaramento.json`.

**As features estão CONGELADAS desde 30/08/2026** — nova re-extração apenas por
erro grave, com decisão registrada. A definição vigente está assinada em
`data/features/features.meta.json` e conferida pela guarda de retomada.

## Estrutura de pastas

```
Tcc_Deepfake/
├── config/             # parâmetros do experimento (config.yaml)
├── data/
│   ├── raw/            # áudios + chaves originais (NÃO versionado no git)
│   ├── processed/      # labels.csv, split.csv, split_181k.csv, subamostra_30k.csv
│   └── features/       # features.csv (CONGELADO) + históricos e pilotos — inventário em data/features/LEIA-ME.md
├── notebooks/          # exploração e prototipagem
├── src/
│   ├── data/           # carregamento de labels, pré-processamento, split
│   ├── features/       # extração de features e geração de espectrogramas
│   ├── models/         # treino/avaliação de RF, SVM e CNN
│   └── utils/          # funções auxiliares (seeds)
├── models/             # modelos treinados salvos (.joblib)
├── results/
│   ├── figuras/        # matrizes de confusão, curvas, diagnósticos (PNG)
│   └── metricas/       # tabelas (CSV), métricas (JSON) e notas técnicas (MD)
└── scripts/            # scripts executáveis (diagnósticos, subamostra, ambiente)
```

## Mapa pasta × cronograma do TC II

| Foco                         | Onde mexe                          |
|------------------------------|------------------------------------|
| Ambiente + dataset           | `config/`, `data/raw/`, `scripts/` |
| Leitura de labels            | `src/data/`, `notebooks/`          |
| Pré-processamento            | `src/data/`, `data/processed/`     |
| Extração de features         | `src/features/`, `data/features/`  |
| Random Forest + SVM          | `src/models/`, `results/`          |
| CNN                          | `src/features/`, `src/models/`     |
| Comparação final             | `results/`                         |

## Como começar

```bash
python -m venv .venv
source .venv/Scripts/activate      # Linux/mac: source .venv/bin/activate
pip install -r requirements.txt
python scripts/verificar_ambiente.py
```

## Preparar ambiente

```bash
source .venv/Scripts/activate
jupyter notebook
```
