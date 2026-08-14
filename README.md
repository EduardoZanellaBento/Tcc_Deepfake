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
por codec (`diagnostico_por_codec.py`). Experimento de robustez (split por ataque /
leave-one-attack-out) previsto como **análise complementar**, após fechar RF, SVM e CNN.

## Braço principal × braço de referência

A curva de aprendizado no universo eval (`curva_aprendizado_rf_eval.json`) mostra que
o RF **não satura** antes do treino completo:

| n treino | f1_macro | EER |
|---:|---:|---:|
| 5.000 | 0,4760 | 0,3030 |
| 10.000 | 0,4801 | 0,2843 |
| 20.000 | 0,4882 | 0,2512 |
| 40.000 | 0,5059 | 0,2226 |
| 80.000 | 0,5356 | 0,1945 |
| 103.723 | 0,5573 | 0,1818 |

Logo, a subamostra de 30k **custa desempenho real**. Por isso o experimento tem dois
braços:

- **Braço principal (comparação):** RF, SVM e CNN treinados na subamostra de 30k —
  único arranjo em que a comparação entre modelos é limpa (mesmo ambiente
  experimental, exigência da pergunta de pesquisa). Validação e teste **completos**.
- **Braço de referência:** RF treinado no treino completo do eval (custa ~8 s) —
  quantifica exatamente quanto a subamostra custou. Entra como linha extra na tabela
  final.

## Como rodar cada script

Sempre a partir da **raiz**, com `python -m` (imports relativos):

```bash
# pipeline principal
python -m src.data.carregar_dados            # regenera labels.csv do trial_metadata.txt
python -m src.data.split                     # gera/recarrega o split 70/15/15 (universo eval)
python -m src.models.treinar_rf              # RF baseline -> rf_baseline_eval.{json,joblib} + matriz de confusão
python -m src.features.extrair_features     # ATENÇÃO: re-extração leva HORAS; ver aviso abaixo

# composição e subamostra
python -m scripts.composicao_eval            # composição por classe/codec/ataque no eval (verifica 148.176)
python -m scripts.gerar_subamostra           # subamostra 30k estratificada do treino (só a lista de IDs)

# diagnósticos
python -m scripts.diagnostico_composicao_dataset   # composição fase/trim/codec do dataset bruto (181k)
python -m scripts.diagnostico_vazamento_duracao    # prop_fala como atalho de duração (árvore só com prop_fala)
python -m scripts.diagnostico_limiar               # ROC, PR e varredura de limiar do RF na validação
python -m scripts.diagnostico_por_ataque           # desempenho do RF por ataque A07–A19
python -m scripts.diagnostico_por_codec            # desempenho do RF por codec (banda estreita × larga)
python -m scripts.curva_aprendizado_rf             # curva de aprendizado -> evidência do braço duplo
python -m scripts.diagnostico_padding_features     # correlação das 44 features com prop_fala
python -m scripts.diagnostico_rf_pareado_propfala  # RF com classes pareadas por faixa de prop_fala
python -m scripts.piloto_mascaramento_padding      # piloto A/B de mascaramento (2.000 áudios, arquivo separado)

python scripts/verificar_ambiente.py         # sanidade do ambiente (versões, GPU, pastas)
```

## Resultados atuais (RF baseline, conjunto de validação)

| Configuração | n treino | f1_macro | EER | Arquivo |
|---|---:|---:|---:|---|
| RF baseline histórico (universo 181k) | 127.096 | 0,5675 | 0,1761 | `results/metricas/rf_baseline.json` |
| RF baseline **eval** (universo aprovado) | 103.723 | 0,5573 | 0,1818 | `results/metricas/rf_baseline_eval.json` |

Métricas por codec e por ataque: `results/metricas/diagnostico_por_codec.csv` e
`diagnostico_por_ataque.csv`. Diagnóstico do atalho de silêncio:
`rf_pareado_propfala.json`, `padding_corr_features_propfala.csv` e
`RECOMENDACAO_MASCARAMENTO.md`.

### Nota sobre o limiar de decisão

O f1_macro baixo com AUC alta (~0,90) **não** significa ausência de sinal: o par
(`class_weight='balanced'`, limiar 0,50) está desajustado — as folhas puras do RF
neutralizam o peso de classe e o limiar ótimo fica em ~0,80–0,88. Movendo só o limiar,
o f1_macro vai de 0,56 para 0,77 **sem re-treino**. Causa raiz, correções planejadas e
consequência de protocolo (mesma regra de limiar para RF, SVM e CNN):
**`results/metricas/NOTA_LIMIAR.md`** e `nota_divergencia_f1.md`.

## ⚠️ Re-extração de features

`python -m src.features.extrair_features` é uma operação de **horas** de CPU e **só
roda com decisão explícita registrada**. O `data/features/features.csv` atual é o
artefato de referência desta fase; não sobrescrever, não regerar parcialmente.

### Pendências acumuladas para o lote único de re-extração (NÃO executar agora)

1. **`win_length` do centróide espectral** — o `features.csv` atual foi extraído sem
   `win_length` no `spectral_centroid` (usou o default `n_fft=512` em vez dos 400 de
   MFCC/ZCR). A correção já está no código e vale a partir do próximo lote.
2. **Filtro `fase == 'eval'`** — extrair somente o universo aprovado (148.176).
3. **Mascaramento de padding na agregação temporal** — recomendado pelo piloto
   (`RECOMENDACAO_MASCARAMENTO.md`); aguarda aprovação do orientador.
4. **`n_frames_validos` como coluna de diagnóstico** — junto com `prop_fala`, e
   igualmente **fora** do X (adicionar à lista `excluir` em `colunas_features`).

Cada item esquecido aqui é uma re-extração inteira desperdiçada — manter a lista
atualizada.

## Estrutura de pastas

```
Tcc_Deepfake/
├── config/             # parâmetros do experimento (config.yaml)
├── data/
│   ├── raw/            # áudios + chaves originais (NÃO versionado no git)
│   ├── processed/      # labels.csv, split.csv, split_181k.csv, subamostra_30k.csv
│   └── features/       # features.csv (44 features/áudio) + piloto_padding.csv
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
