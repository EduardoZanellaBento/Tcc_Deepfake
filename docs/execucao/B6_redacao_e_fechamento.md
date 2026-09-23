# B6 — Redação, entrega ao orientador e congelamento final

| | |
|---|---|
| **B6.1 redação** | 30/09 a 03/10 |
| **B6.2 ENTREGA** | **04/10 — versão completa, avaliável do início ao fim** |
| **B6.3 correções** | 08 a 09/10 (a revisão do orientador é 05 a 07/10) |
| **B6.4 conferência** | 10/10 |
| **B6.5 congelamento** | **11/10** |

> **A data real de entrega é 04/10, não 11/10.** O que existe entre 04/10 e 11/10 é o
> tempo do orientador, não o seu. Se a primeira versão integral chegar perto de
> 11/10, não há tempo hábil nem para ele revisar nem para você corrigir.
>
> **Entre 05/10 e 07/10 não se abre experimento novo.** Esse intervalo é para
> preparar a execução das correções.

---

## Contexto mínimo (ler só isto)

1. `docs/execucao/APENDICE_A_inventario.md` — **todo**
2. `docs/execucao/APENDICE_B_banca.md` — **todo**
3. `README.md` do repositório — já é metade da metodologia escrita
4. `results/metricas/COMPARACAO_FINAL.md` — **as leituras do B5 já redigidas**, geradas
   a partir dos números (as três frases obrigatórias, o bootstrap pareado, o delta
   validação → teste, ataque e codec) — mais `comparacao_final.csv` e
   `teste_lacrado.json`
5. as notas técnicas: `NOTA_LIMIAR.md`, `NOTA_RF_VS_SVM.md`,
   `RECOMENDACAO_MASCARAMENTO.md`, `REVISAO_BLOCO3.md`, `DOSSIE_LOTE_UNICO.md`,
   `DECISOES_PENDENTES_CNN.md` (v5)

---

## O ativo que você já tem, e que quase ninguém tem

**O trabalho já está metodologicamente muito bem documentado.** As notas técnicas do
repositório não são rascunho: são texto de metodologia com a evidência anexada. A
redação do TC II é, em grande parte, **transposição e costura**, não escrita nova.

| seção do TC II | fonte pronta no repositório |
|---|---|
| Universo do experimento e exclusões | `README` §Universo · `eval_composicao_resumo.json` |
| Pré-processamento (normalização, VAD, duração fixa) | docstrings de `preprocessamento.py` |
| Mascaramento de padding | `RECOMENDACAO_MASCARAMENTO.md` · `checagem_mascaramento.json` · docstring de `extrair_features.py` |
| Features manuais e seus parâmetros | docstring de `extrair_vetor` · bloco `features:` do config |
| Split 70/15/15 e onde o 5-fold roda | docstring de `split.py` (tem até o diagrama) |
| Limitação do split (por utterance) | `README` §Limitação declarada · bloco `split:` do config |
| Braço duplo | `README` §Braço principal × referência · bloco `experimento:` |
| RF e Random Search | `ajustar_rf.py` · `rf_random_search.json` |
| SVM | `treinar_svm.py` · `NOTA_RF_VS_SVM.md` |
| Protocolo de limiar | `NOTA_LIMIAR.md` · `nota_divergencia_f1.md` · docstring de `avaliacao.py` |
| Definição do espectrograma (P1–P7) | `DECISOES_PENDENTES_CNN.md` v5 — **com a evidência que motivou cada decisão** |
| Arquitetura da CNN e máscara temporal | docstring de `cnn.py` · `checagem_mascara_cnn.json` |
| Protocolo de medição de tempo | comentário do bloco `tempo:` do config · `tempo.py` |
| Resultados, comparação estatística, ataque/codec, custo | `COMPARACAO_FINAL.md` · `comparacao_estatistica.json` · 5 figuras `comparacao_*.png` |
| Reprodutibilidade | `reproducao_bloco3.json` · `reproducao_cnn.json` · `REVISAO_BLOCO3.md` · `guarda_reproducao.py` |
| Limitações | ver §Lista de limitações, abaixo |

**Regra de ouro da redação:** cada afirmação numérica do texto tem de existir num
artefato versionado. Se você se pegar escrevendo um número de memória, pare e
localize o JSON. A recíproca é a força do trabalho: você **pode** citar o arquivo.

---

## A lista de limitações — monte-a primeiro, não por último

É a seção que a banca lê com mais atenção, e você já tem todas. Faça-a **antes** de
escrever resultados: ela organiza o que você pode e o que não pode afirmar.

| # | limitação | origem |
|---|---|---|
| 1 | **Split aleatório por utterance**: cada ataque, codec e locutor aparece em treino e teste. Métricas potencialmente otimistas e **não comparáveis** ao EER de 1,32% de Yamagishi et al. (2022), cujo protocolo é cross-attack. Mitigação: métricas por ataque e por codec | `README` |
| 2 | **Sem generalização cross-dataset**: o ASVspoof 2021 LA não fornece treino/dev próprios; usou-se o `eval` com split interno | `README` |
| 3 | **`progress` e `hidden` excluídos** por controle metodológico (o `hidden` tem silêncio pré-cortado na origem) | config `dataset:` |
| 4 | **Subamostra de 30k custa desempenho real**: a curva de aprendizado do RF não satura; extrapolando, o RF precisaria de ~276.116 áudios para alcançar o SVM — 1,86× o universo eval. A subamostra foi imposta pela complexidade O(n²)–O(n³) do SVM-RBF | `extrapolacao_curva_rf.json` |
| 5 | **Duração fixa em 4,0 s** não foi tratada como eixo experimental | config `audio:` |
| 6 | **`top_db=80` é relativo ao máximo de cada exemplo**: o piso do padding é `max−80` e varia entre exemplos | P5 |
| 7 | **A filterbank Mel do `librosa.feature.mfcc`, com o `n_fft=512` congelado, herda o mesmo regime degenerado** que motivou a P4 a usar 1024 na CNN. Não corrigido: features congeladas desde 30/08, e re-extrair abriria o Bloco 2. Atenuado pela DCT, que retém 20 de 128 coeficientes | B4.0 |
| 8 | **Refit com o mesmo nº de épocas**, não de atualizações: os 30k são 11,1% maiores que os 27k | B4.5 |
| 9 | **Variância entre sementes na CNN**: a CNN final é **semente única** (42) — limitação computacional declarada, salvo se o item 2 da fila do tempo excedente for executado (3 sementes, ~1 h, **avaliadas só na validação**). O bootstrap pareado reamostra a avaliação, não o treino. A fonte de variação que domina o braço principal é *qual subamostra de 30k caiu*, medida em `estabilidade_subamostra.json` | P6 · decisão 6 |
| 10 | ~~Determinismo estrito no PyTorch~~ — **não se aplica**: `estrito=True` no refit e na inferência (`refit_cnn.json → determinismo`, `teste_lacrado.json → cnn.determinismo`). Escreva como *controle*, não como limitação | B4.4 · B4.6 |
| 11 | **Experimento cross-attack / leave-one-attack-out** previsto como análise complementar, não executado | `README` |
| 12 | **Hardware único**: todas as medições numa máquina (Windows 10, RTX 5060 Ti). Tempos são indicativos de ordem de grandeza, não benchmark. **Medido:** os tempos variaram 4,1–13,7% entre duas sessões da mesma máquina (`reproducao_cnn.json`), por isso CNN-GPU (4,58 ms) e SVM (4,92 ms) são empate prático, não ordem | config `tempo:` · B5.2 |
| 13 | **Hipótese da banda larga na CNN não decidida**: o ganho da CNN com banda larga é o maior em termos relativos e não o maior em termos absolutos, e não há IC para a diferença entre ganhos | `COMPARACAO_FINAL.md` §6 |
| 14 | **Por ataque, amplitude sem IC**: a comparação «amplitude entre ataques × distância entre modelos» usa um piso de ruído (1,96·dp do ΔEER pareado), não um IC da amplitude, que vem de subconjuntos menores | `COMPARACAO_FINAL.md` §6 |

Acrescente as que aparecerem, e nunca omita uma para o texto ficar mais bonito. A
credibilidade deste trabalho está construída em cima de limitações declaradas — é o
padrão do repositório desde o Bloco 1.

---

## Estrutura sugerida dos capítulos da fase prática

**Metodologia experimental**

1. Universo do experimento e justificativa das exclusões
2. Pré-processamento (a cadeia, e por que nesta ordem)
3. Ramo clássico: features, parâmetros, agregação mascarada
4. Ramo CNN: espectrograma (P1, P4, P5), normalização (P3), agregação mascarada
   na rede (P2)
5. Partição, braço duplo e onde o 5-fold roda
6. Protocolo de avaliação: regra de decisão, seleção de limiar, EER, teste lacrado
7. Protocolo de medição de tempo (e por que quatro medições)
8. Controle de reprodutibilidade: sementes, artefatos assinados, guardas de código

**Desenvolvimento**

9. Implementação do pipeline (com o argumento da fonte única)
10. Bloco 2: o lote único e a checagem obrigatória
11. Bloco 3: RF e SVM
12. Bloco 4: CNN — e a decisão registrada **antes** da geração

**Resultados**

13. Tabelas de validação e teste
14. Comparação estatística (bootstrap pareado)
15. Análise por ataque e por codec
16. Custo computacional

**Discussão** — a resposta à pergunta de pesquisa, em uma frase, no primeiro
parágrafo. Depois: o que os números mostram, o que não mostram, e por quê.

**Limitações** · **Conclusão** · **Trabalhos futuros**

> **A discussão é o lugar onde este trabalho pode se destacar** — e onde a maioria
> dos TCCs vira descrição. A diferença: não escreva «a CNN obteve f1_macro de X,
> superior/inferior a Y». Escreva **o que isso significa para a pergunta de pesquisa**,
> com o intervalo de confiança ao lado, e diga onde a evidência não alcança.

---

## Como escrever a resposta à pergunta de pesquisa

A pergunta é: *modelos clássicos com features manuais mantêm desempenho competitivo,
em acurácia e custo computacional, frente a CNNs, no mesmo ambiente experimental?*

A resposta tem **dois eixos**, e é erro comum responder só um.

**Eixo acurácia.** Compare no braço principal (mesmas 30k), com o bootstrap pareado.
**Desfecho medido em 23/09: o primeiro da tabela** — CNN à frente dos dois clássicos,
e SVM à frente do RF, com IC95 sem zero em f1_macro e em EER, na validação e no teste
(`COMPARACAO_FINAL.md` §4). Três desfechos possíveis, e o vocabulário de cada um:

| desfecho | como escrever |
|---|---|
| IC da diferença não contém zero, CNN à frente | «a CNN supera os clássicos de forma estatisticamente distinguível neste protocolo, por Δf1_macro = … IC95 […]» |
| IC contém zero | «**não distinguíveis neste protocolo**» — nunca «empataram», nunca «X é ligeiramente melhor» |
| IC não contém zero, clássico à frente | «sim, mantêm desempenho competitivo — e mais: superam, neste protocolo e com este n de treino» |

**Eixo custo.** Aqui o resultado é mais interessante que um número único. Tudo o que
você precisa já está medido:

- por áudio, ponta a ponta (`tempo_pipeline_completo.json`, os quatro na mesma
  execução, 20/09): RF **10,23 ms** (predição 57,5% do custo) · SVM **4,92 ms** ·
  CNN-GPU **4,58 ms** · CNN-CPU **16,80 ms**. *Os números antigos (RF 11,12 · SVM 5,33)
  são de outra sessão e não entram no texto;*
- CNN-GPU e SVM **empatam na prática** (folga de 7,5%, dentro da variação entre
  sessões) — não escreva que a CNN-GPU é «a mais barata»;
- em lote: o RF é ~19,7× melhor que o SVM (0,0204 × 0,4025 ms/áudio);
- «não existe "o modelo mais barato" sem dizer o regime»;
- a hipótese de que o pré-processamento dominaria **não se confirmou para o RF**, mas
  **se confirmou para o SVM** (base = 88,4%) e **para a CNN em GPU** (base = 74,5%, com
  o log-Mel sozinho em 54,1%) — medido, não mais «provavelmente»;
- e o ponto que o `config.yaml` registrou como o diferencial: **a vantagem dos
  clássicos não é serem mais rápidos em igualdade de hardware — é não exigirem GPU.**
  O par CNN-GPU / CNN-CPU é o que evidencia isso.

**A frase de conclusão tem de ser condicional ao protocolo.** «Neste protocolo, com
30.000 exemplos de treino e split aleatório por utterance…» — porque é isso que a
evidência sustenta, e porque a limitação 1 existe.

---

## B6.2 — A entrega de 04/10

**Versão completa, avaliável do início ao fim.** Não capítulo isolado, não versão
parcial.

Checklist de 04/10:

- [ ] todos os capítulos escritos, inclusive conclusão e trabalhos futuros
- [ ] todas as tabelas com números lidos dos artefatos
- [ ] todas as figuras inseridas, numeradas e referenciadas no texto
- [ ] limitações completas (as 14 acima + as que apareceram)
- [ ] referências e citações no lugar
- [ ] ABNT: a base é o `TC1_EDUARDO_numeracao_ABNT.pdf` já aprovado — **reuse a
      formatação**, não recomece
- [ ] repositório commitado e coerente com o texto
- [ ] enviado ao orientador **em 04/10**

**Se algum capítulo não estiver pronto em 03/10:** envie a versão completa com a
lacuna **marcada em destaque** e um parágrafo dizendo o que falta e quando chega. Uma
versão completa com duas lacunas sinalizadas é revisável; uma versão de 80% sem
sinalização não é, e desperdiça os três dias do orientador.

---

## B6.6 — Reprodutibilidade e simulação de banca (dentro do B6.1, sem atrasar)

**Verificação de reprodutibilidade** — do zero, num diretório limpo, seguindo só o
`README`:

```bash
python -m src.data.split                      # recarrega (idempotente)
python -m scripts.validar_split_pos_lote      # hashes e IDs intactos
python -m scripts.guarda_reproducao           # JSONs reproduzem campo a campo
python -m scripts.verificar_espectrogramas --lote
```

Se algum passo do `README` não funcionar como escrito, **corrija o README** — ele é
parte da entrega e a banca pode pedir para ver.

**`README`** — *feito em 23/09 (pós-B5):* comandos do ramo CNN e do Bloco 5, a tabela
final do teste e a nota de que o teste foi usado uma única vez. Falta, se couber: a
seção narrativa do Bloco 4 (hoje o README cobre o Bloco 4 só pelos tempos).

**Simulação de banca:** use `APENDICE_B_banca.md`. Responda **em voz alta**, com o
repositório fechado, e anote toda pergunta cuja resposta você não soube dizer em 30
segundos — cada uma dessas é um parágrafo que falta no texto.

---

## B6.3 e B6.4 — Correções e conferência

**08 e 09/10:** incorporar **integralmente** as correções do orientador. Se discordar
de alguma, implemente e registre a divergência num parágrafo — não ignore em silêncio.

**10/10, conferência final:**

- [ ] ABNT: margens, fontes, espaçamento, citações, referências
- [ ] numeração de **todas** as figuras e tabelas, e todas referenciadas no texto
- [ ] sumário atualizado
- [ ] revisão de português (leia em voz alta; erros de concordância aparecem no ouvido)
- [ ] coerência metodológica: nenhuma afirmação sem artefato
- [ ] o repositório reflete o texto; nada não commitado
- [ ] `git log` legível, com um commit por marco

**11/10 — congelamento.** Nenhuma etapa principal pendente depois desta data.

---

## Armadilhas desta fase

| armadilha | consequência |
|---|---|
| escrever resultados antes das limitações | você afirma mais do que a evidência sustenta e depois tem de reescrever |
| digitar números de memória | a tabela divergindo do artefato — e a banca lê a tabela |
| «a CNN foi ligeiramente melhor» com IC contendo zero | interpretação descritiva onde se pediu crítica |
| citar o EER ao lado do 1,32% da literatura sem a ressalva de protocolo | erro que a banca pega na hora |
| abrir experimento entre 05 e 07/10 | proibido; esse tempo é do orientador |
| deixar a versão completa para 08/10 | o orientador perde a janela de revisão e você a de correção |
| omitir uma limitação para o texto ficar melhor | a credibilidade do trabalho está construída sobre limitações declaradas |
| recomeçar a formatação ABNT do zero | o TC I já está aprovado e formatado — reuse |
