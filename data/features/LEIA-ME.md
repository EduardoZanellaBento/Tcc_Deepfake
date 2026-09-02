# data/features/ — inventário dos artefatos

Este LEIA-ME é versionado (o `.gitignore` exclui só os `*.csv`) e documenta **todos**
os CSVs desta pasta: qual é o vigente, quais são evidência histórica e por que cada
um existe (decisão D2 do fechamento do Bloco 2 — os históricos ficam **preservados,
não apagados**). MD5 e contagens conferidos em 01/09/2026.

## O arquivo vigente

| arquivo | MD5 | linhas (sem header) | definição |
|---|---|---:|---|
| `features.csv` | `51b2f439bf6f1e10237acbc620bb92d9` | 148.176 | **CONGELADO desde 30/08/2026** (lote único, commit `3e09e91`). Universo `fase == 'eval'`; 50 colunas (44 features no X + `arquivo`, `label`, `classe_binaria`, `prop_fala`, `n_frames_validos`, `n_frames_total`); **com** mascaramento de padding; centróide com `win_length=400`. É o **único** CSV que os modelos usam. Assinatura legível por máquina em `features.meta.json`, conferida pela guarda de retomada (`_validar_retomada` em `src/features/extrair_features.py`). Dossiê completo: `results/metricas/DOSSIE_LOTE_UNICO.md`. |

**Não regerar.** Nova re-extração só por erro grave, com decisão registrada.

## Evidência histórica (preservar — decisão D2)

| arquivo | MD5 | linhas | universo | por que existe |
|---|---|---:|---|---|
| `features_pre_mascaramento_c01c3c5c.csv` | `c01c3c5c6afdcad0dd95236ffd6910ad` | 181.566 | 181k (eval + progress + hidden) | O `features.csv` **anterior ao lote único** (16/07/2026): **sem** mascaramento, centróide com `win_length` default (n_fft=512), 48 colunas. É a base de **todos os diagnósticos anteriores a 30/08** (`rf_baseline*.json`, `NOTA_LIMIAR.md`, varredura de limiar, correlações com `prop_fala`) e o lado "antes" de toda comparação antes/depois do mascaramento. O sufixo é o próprio MD5. |
| `features_proveniencia_mista.csv` | `1c2175e6ecaea1001e1be4a08f0854d0` | 181.566 | 181k | Mesmo tamanho do anterior, **MD5 diferente**: proveniência **mista** — contém linhas extraídas antes E depois da correção do carregamento de áudio (`librosa.load` → `soundfile`, 15/07/2026). Preservado como evidência de que a troca de backend foi detectada e tratada; **não usar para análise**. |
| `erros_run1.csv` | `31b1d2e666ea3ec3d7c0dac74f0b1ed8` | 79.645 | — | Registro das 79.645 falhas `NoBackendError()` da rodada antiga (15/07/2026), causadas pelo bug de FLAC dos wheels soundfile ≥ 0.13 no Windows. Motivou o pin `soundfile==0.12.1` e a leitura direta via `sf.read` (ver README). |

## Pilotos de validação (arquivos separados por construção)

| arquivo | MD5 | linhas | por que existe |
|---|---|---:|---|
| `piloto_padding.csv` | `fdd473553833b2b70f1d86ccb33b98a7` | 2.000 | Piloto A/B do mascaramento (12/08/2026): cada áudio com as 44 features agregadas com e sem padding. Base de `piloto_padding_delta_features.csv`, `piloto_padding_rf_ab.json` e da recomendação de mascarar (`RECOMENDACAO_MASCARAMENTO.md`). |
| `features_piloto_mascarado.csv` | `7e69f4438d2b6dde15d698a4889ec849` | 500 | Piloto do Bloco 1 (30/08/2026) com a definição final (mascaramento + `win_length=400` + universo eval), rodado ANTES do lote único para validar o pipeline; tem `features_piloto_mascarado.meta.json` ao lado. Evidência: `results/metricas/checagem_mascaramento.json`. |

## Metadados

`features.meta.json` e `features_piloto_mascarado.meta.json` — assinatura da definição
de extração (semente, fase, mascaramento, n_mfcc/n_fft/hop/win, commit). Nota: o
`commit_git` gravado no lote único (`9a23152`) **não contém** a guarda
`_validar_retomada`, que rodou ainda não commitada — inofensivo (a guarda não toca em
valor de feature), e é a razão de o campo `git_dirty` ter sido adicionado à assinatura
no Bloco 3.
