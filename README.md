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

## Estrutura de pastas

```
deteccao-voz-sintetica/
├── config/             # parâmetros do experimento (config.yaml)
├── data/
│   ├── raw/            # áudios + chaves originais (NÃO versionado no git)
│   ├── processed/      # áudios após pré-processamento
│   └── features/       # CSVs de features extraídas (MFCC, ZCR, centróide)
├── notebooks/          # exploração e prototipagem
├── src/
│   ├── data/           # carregamento de áudios e labels
│   ├── features/       # extração de features e geração de espectrogramas
│   ├── models/         # treino/avaliação de RF, SVM e CNN
│   └── utils/          # funções auxiliares (métricas, plots, seed)
├── models/             # modelos treinados salvos
├── results/
│   ├── figuras/        # matrizes de confusão, gráficos
│   └── metricas/       # tabelas de resultados (CSV)
└── scripts/            # scripts executáveis (verificar ambiente, organizar dataset)
```

## Mapa pasta × cronograma do TC II

| Semana | Foco                         | Onde mexe                          |
|--------|------------------------------|------------------------------------|
| 1      | Ambiente + dataset           | `config/`, `data/raw/`, `scripts/` |
| 2      | Leitura de labels            | `src/data/`, `notebooks/`          |
| 3      | Pré-processamento            | `src/data/`, `data/processed/`     |
| 4      | Extração de features         | `src/features/`, `data/features/`  |
| 5–7    | Random Forest + SVM          | `src/models/`, `results/`          |
| 8–10   | CNN                          | `src/features/`, `src/models/`     |
| 11     | Comparação final             | `results/`                         |
| 12–13  | Escrita                      | —                                  |

## Como começar

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install numpy pandas scipy librosa soundfile scikit-learn matplotlib seaborn tqdm pyyaml jupyter nbformat
python scripts/verificar_ambiente.py
```
## Ambiente
jupyter notebook