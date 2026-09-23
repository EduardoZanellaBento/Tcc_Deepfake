# Achados — o que não impede a validade e para onde vai

> Arquivo previsto em `00_MAPA_EXECUCAO.md` §3: **achado interessante não é tarefa.**
> Cada linha diz o achado, a resposta ao teste da regra de escopo («impede a validade
> do experimento principal?») e o destino. Criado em 23/09/2026, na varredura
> pós-Bloco 5.

| data | achado | impede a validade? | destino |
|---|---|---|---|
| 23/09 | **EER no vértice da ROC**, sem interpolação e com `roc_curve(drop_intermediate=True)`. Conferido sobre os scores publicados: diferença para o EER com todos os vértices ou interpolado linearmente **≤ 0,0003** nos quatro modelos, validação e teste | não — duas ordens abaixo das diferenças entre modelos (≥ 0,046) e abaixo do dp do bootstrap (~0,004); mesma régua para os três | nota de método em §Protocolo de avaliação; resposta pronta em `APENDICE_B_banca.md` §7 |
| 23/09 | **CNN-GPU × SVM em latência ponta a ponta** (4,58 × 4,92 ms): folga de 7,5% dentro da variação de 4,1–13,7% entre sessões (`reproducao_cnn.json`) | não — muda a *redação*, não o resultado | leitura corrigida em `COMPARACAO_FINAL.md` (frase 3, empate prático); limitação 12 do B6 |
| 23/09 | **Por ataque, CNN**: amplitude 0,1171 × distância CNN−RF 0,1192 — diferença de 0,0021, dentro do piso de ruído (±0,0086) | não | leitura corrigida em `COMPARACAO_FINAL.md` §6 («da mesma ordem»); limitação 14 do B6 |
| 23/09 | **ΔEER SVM − RF citado de dois jeitos**: −0,0466 (média do bootstrap, Bloco 3) e −0,0468 (observado, `COMPARACAO_FINAL.md`) | não — é o mesmo resultado | no texto, usar o **observado + IC95**; nota em `APENDICE_A_inventario.md` §3 |
| 23/09 | **Cópia Git das instruções do projeto estava na versão de 17/09**; a de 23/09 existia só no claude.ai | não — mas é o risco de perda que a cópia existe para evitar | corrigido: `Claude outputs/INSTRUCOES_PROJETO_tcc.md` na versão pós-Bloco 5 |
| 23/09 | **`requirements.txt` só com pisos (`>=`)** — as versões exatas estão nos campos `ambiente` dos JSONs, mas não num arquivo instalável | não | **resolvido em 23/09:** `requirements-lock.txt` gravado (`pip freeze`) no ambiente que gerou os resultados |
| 23/09 | **`models/` fora do Git** (regra do `.gitignore`): o `.pt` da CNN final (~1 MB) e o SVM (~2,8 MB) que produziram o teste lacrado só existem no disco local; os RF têm 60–190 MB | não — os scores do teste estão versionados (`scores_teste_lacrado.csv`), então os números são auditáveis sem os modelos | **resolvido em 23/09 (parte Git):** CNN final e SVM versionados (exceção no `.gitignore`), MD5 de todos em `HASHES_MODELOS.txt`. **Em aberto até 11/10:** restaurar o backup externo dos RF e conferir o MD5 — checklist 10/10 do B6 |
| 23/09 | **Worktree antigo em `.claude/worktrees/amazing-yonath-d9913c`** (24/08), com cópia velha de `src/` e `config.yaml` | não — mas uma busca de texto na pasta encontra código desatualizado | **resolvido em 23/09:** `git worktree remove`, depois de conferir que o ramo não tinha nada que faltasse no `master` |
| 23/09 | **`requirements-lock.txt` fixa `torch==2.11.0+cu128`**, versão que só existe no índice do PyTorch: `pip install -r requirements-lock.txt` puro falha | não — é o comando de instalação, não o experimento | `--extra-index-url https://download.pytorch.org/whl/cu128` no «Como começar» do `README` e no B6.6 |
