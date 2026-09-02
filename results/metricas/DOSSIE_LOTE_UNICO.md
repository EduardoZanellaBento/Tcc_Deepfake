# Dossiê do lote único de re-extração de features (Bloco 2 — B2.3)

**Data de execução:** 30/08/2026 · **Executor:** pipeline `python -m src.features.extrair_features`
**Commit em que o lote rodou:** `9a23152e690208d3a0ecf4aa235ee8fc9cecaba8`
(*"Bloco 1: fecha o pipeline de features para o lote unico"* + as alterações do
Bloco 2 descritas na seção 8, commitadas junto com este dossiê)

Este documento registra a execução do lote único que aplicou ao `features.csv`
as quatro pendências aprovadas pelo orientador (win_length do centróide, filtro
`fase == 'eval'`, mascaramento de padding, colunas de diagnóstico). É o artefato
que o orientador pediu para aprovar o **congelamento das features**.

---

## 1. Hashes MD5

| artefato | MD5 |
|---|---|
| `features.csv` **antigo** (pré-lote, arquivado como `features_pre_mascaramento_c01c3c5c.csv`) | `c01c3c5c6afdcad0dd95236ffd6910ad` |
| `features.csv` **novo** (pós-lote, CONGELADO) | `51b2f439bf6f1e10237acbc620bb92d9` |
| `split.csv` (INALTERADO — ver seção 7) | `9143f0c7b83ec2db4aa144ed5deb3402` |
| `subamostra_30k.csv` (INALTERADA) | `654cb796b738512388b28e15ffb14a9d` |

Nota de proveniência: `features_proveniencia_mista.csv` tem o **mesmo tamanho**
(85.455.802 bytes) do `features.csv` antigo, mas **MD5 diferente**
(`1c2175e6ecaea1001e1be4a08f0854d0`) — não é uma cópia byte a byte (provável
diferença de ordem de linhas entre rodadas). Ambos ficam arquivados em
`data/features/` como histórico; nenhum foi apagado. O inventário completo da
pasta (papel, MD5 e contagem de cada CSV, incluindo os pilotos) está em
**`data/features/LEIA-ME.md`** (decisão D2 — artefatos históricos documentados).

## 2. Contagens — as contas fecham

| grandeza | valor |
|---:|---|
| áudios no universo (`fase == 'eval'` no labels.csv) | **148.176** |
| processados com sucesso | **148.176** |
| com erro (`erros_features.csv`) | **0** (arquivo não gerado) |
| linhas no CSV final (`arquivo` únicos) | **148.176** (148.177 com cabeçalho) |
| bonafide / spoof | 14.816 / 133.360 (9,0 : 1) |

`processados + erros == universo` ✔ · `linhas == processados` ✔ ·
`arquivo` sem duplicatas ✔

## 3. Esquema

- 50 colunas = 3 identificação (`arquivo`, `label`, `classe_binaria`)
  + 3 diagnóstico (`prop_fala`, `n_frames_validos`, `n_frames_total`)
  + 44 features — **idêntico, na mesma ordem, a `esquema_esperado()`**
  (`src/features/extrair_features.py`). ✔
- `colunas_features()` (`src/data/split.py`) devolve exatamente **44** colunas,
  **nenhuma diagnóstica dentro do X**. ✔

## 4. Universo

- `fase == 'eval'` em **100%** das 148.176 linhas (merge com `labels.csv`). ✔
- `trim == 'notrim'` em **100%** das linhas. ✔
- Varredura prévia de legibilidade (P3): os 148.176 `.flac` abriram no
  `sf.info` **sem nenhum problema** (existência, header e sample rate 16 kHz
  conferidos antes do lote).

## 5. `head()` do CSV (colunas selecionadas)

```
     arquivo  classe_binaria  prop_fala  n_frames_validos  n_frames_total  mfcc1_media  centroide_std
LA_E_9332881               1     0.9878               203             251   -405.56050      649.73930
LA_E_6866159               1     0.9792               177             251   -402.81090      452.71994
LA_E_5464494               1     0.9146                47             251   -395.35153      296.89830
LA_E_4759417               1     0.9600                90             251   -440.77618      613.37910
LA_E_2667748               1     0.9779               222             251   -416.18690      457.44333
```

## 6. Sanidade numérica

- NaN nas 44 features: **0** ✔ · inf: **0** ✔
- `1 <= n_frames_validos <= n_frames_total` em **todas** as linhas ✔
- `n_frames_total` constante = **251** (único valor no CSV) ✔
- mediana de `n_frames_validos`: **120** (consistente com os 121 do piloto de
  500 — metade do tensor de 4,0 s é padding, exatamente o que o mascaramento
  passa a excluir da agregação)

## 7. Integridade do split e da subamostra — PRESERVADOS, não regerados

**Decisão metodológica** (diverge da redação original do B2.4 do cronograma —
comunicar ao orientador): o `split.csv` é uma partição de **IDs**
(`[arquivo, conjunto]`); a re-extração mudou os **valores** das features, não o
conjunto de arquivos. Regerar trocaria a partição de graça (a ordem das linhas
do CSV novo difere da do antigo e `train_test_split` é sensível a ela), mudaria
o hash citado no README, forçaria regerar a subamostra em cascata e destruiria a
comparabilidade com todos os artefatos anteriores (`rf_baseline_eval*.json`,
curva de aprendizado, diagnósticos) — a comparação "antes × depois do
mascaramento" perderia a única variável controlada que tinha.

Validação executada (`python -m scripts.validar_split_pos_lote`, registro em
`validacao_split_pos_lote.json`):

1. conjunto de `arquivo` do `features.csv` novo **idêntico** ao do `split.csv`
   (148.176 = 148.176, nem sobra nem falta) ✔
2. `carregar_dados_split()` roda sem erro e devolve 148.176 linhas ✔
3. `filtrar_treino_braco(treino, 'principal', ...)` devolve **30.000** linhas
   = tamanho da subamostra ✔
4. hashes MD5 de `split.csv` e `subamostra_30k.csv` **inalterados** em relação
   aos registrados no README, `subamostra_30k.json` e
   `rf_baseline_eval_principal.json` ✔

## 8. Guarda de esquema/retomada (P1) — a trava contra corrupção silenciosa

O checkpoint de `executar()` pula áudios por **nome**, não por versão da
feature. Retomar sobre um CSV de outra definição produziria um arquivo meio
antigo, meio novo — ou, pior, um append de 50 colunas sob cabeçalho de 48,
**desalinhado sem nenhuma exceção**. Implementado neste bloco
(`_validar_retomada` em `src/features/extrair_features.py`):

- o cabeçalho do CSV existente é comparado a `esquema_esperado()` **antes** de
  retomar; divergência → `ValueError` com instrução de arquivar;
- um `features.meta.json` assina a definição vigente (semente, fase,
  `mascarar_padding`, n_mfcc, n_fft, hop, win, commit git) e é conferido na
  retomada — "confio no nome do arquivo" virou "confiro a definição da feature".

**A trava foi testada abortando de propósito** antes do lote: apontada para o
`features.csv` antigo (48 colunas), `executar()` levantou
`ValueError: ... foi gerado por OUTRA definição de feature (colunas
divergentes: faltando=['n_frames_validos', 'n_frames_total'] ...)` sem gravar
nada; CSV com esquema vigente passou; meta.json com `win_length` divergente
abortou.

## 9. Ambiente e tempo

| item | valor |
|---|---|
| Python | 3.10.11 |
| SO | Windows-10-10.0.26200-SP0 (Windows 11 Home) |
| librosa / numpy / pandas | 0.11.0 / 2.2.6 / 2.3.3 |
| soundfile | 0.12.1 (pinado — wheels 0.13/0.14 quebram FLACs no Windows) |
| webrtcvad | 2.0.14 |
| N_JOBS | 12 (todos os núcleos, default) |
| início / fim | 2026-08-30 15:45:27 / 15:51:42 (−03:00) |
| duração | **6 min 15 s** (375,1 s) |
| taxa | **~395 áudios/s** |
| checkpoint/retomada | não foi necessário — rodada única, sem quedas |

## 10. Evidências pré-lote (ordem de execução do Bloco 2)

1. `python -m scripts.verificar_mascaramento` re-rodado com o código final:
   **APROVADO** (`aprovado_para_lote_unico: true`) — 44/44 features alteradas
   pelo mascaramento, delta relativo mediano 30,30%, máximo 47,03%; fração de
   padding 44,8% (bonafide) vs 45,2% (spoof); runner de produção com esquema
   50/44 limpo. Registro: `checagem_mascaramento.json` +
   `checagem_mascaramento_delta.csv`.
2. Guarda de esquema testada abortando (seção 8).
3. Varredura de legibilidade dos 148.176 FLACs: zero problemas (seção 4).
4. MD5 do CSV antigo registrado e arquivo renomeado ANTES do disparo (seção 1).

## 11. Declaração de congelamento

**A partir deste dossiê, as 44 features do `features.csv`
(MD5 `51b2f439bf6f1e10237acbc620bb92d9`) estão CONGELADAS.** Nova re-extração
só ocorre por erro grave, com decisão registrada por escrito do orientador.
Ideias de melhoria de feature (fmax, outra duração, delta-MFCC) entram no texto
como **trabalho futuro**, nunca como segundo lote. O Bloco 3 (RF/SVM) treina
exclusivamente sobre este artefato.

---

## Apêndice — respostas prontas para a banca

1. **"Como você garante que o `features.csv` final não mistura duas definições
   de feature?"** — Trava de código (`_validar_retomada`: esquema +
   `features.meta.json`), CSV antigo renomeado antes do disparo, rodada única
   sem retomada, hash MD5 único registrado neste dossiê.
2. **"Por que o split não mudou, se as features mudaram?"** — O split é uma
   partição de IDs; os valores das features não entram na sua construção. A
   identidade dos conjuntos foi validada (seção 7).
3. **"Metade do tensor é padding. Isso não invalida a duração fixa de 4,0 s?"**
   — O argumento do mascaramento é de **validade de medida**: a agregação passa
   a descrever só o áudio real. A duração fixa continua sendo restrição de
   formato (necessária para a CNN) e está declarada como limitação.
4. **"O bonafide recebe mais padding que o spoof?"** — Não: 44,8% vs 45,2%. A
   assimetria da distorção (até |0,47| padronizado) vem do conteúdo acústico e
   da dispersão da fração de padding, não da quantidade de zeros — usar o
   adendo de 26/08 do `RECOMENDACAO_MASCARAMENTO.md`, não a versão original.
5. **"Quantos áudios falharam na extração e o que você fez com eles?"** —
   **Zero** nesta rodada (seção 2), com varredura prévia de legibilidade
   também zerada. O precedente histórico de 79.645 falhas (`erros_run1.csv`)
   era o bug do fallback `audioread` do `librosa.load` sem ffmpeg, corrigido ao
   ler via `soundfile` direto; o arquivo fica arquivado como histórico.
