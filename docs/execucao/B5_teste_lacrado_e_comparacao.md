# B5 — Teste lacrado (execução única) e comparação experimental fechada

| | |
|---|---|
| **data-limite** | **29/09/2026** (B5.1 e B5.2 no mesmo dia) |
| **custo estimado** | 1 dia |
| **pré-requisito** | os três JSONs de validação existem, todos com `selecao_limiar.conjunto == "validacao"` |
| **entrega** | métricas no teste + tabela comparativa RF × SVM × CNN, matrizes, tempos, análise por classe/ataque/codec, gráficos finais |
| **efeito permanente** | **depois de 29/09, nenhuma mudança nos modelos, salvo erro grave** |

> **Leia isto antes de escrever a primeira linha de código.** O teste de 22.227 é
> usado **uma única vez**, com os limiares **já escolhidos**. Não há segunda chance:
> se você rodar, olhar o resultado e ajustar qualquer coisa, o teste deixa de ser uma
> estimativa honesta de generalização e passa a ser um segundo conjunto de validação.
> Por isso a ordem dos passos abaixo não é sugestão — **escreva o script inteiro,
> revise-o, e só então rode.**

---

## Contexto mínimo (ler só isto)

1. `src/models/modelos_ajustados.py` — **inteiro**
2. `src/models/avaliacao.py` — **inteiro**
3. os três JSONs de validação: `rf_tuned_principal.json`,
   `svm_tuned_principal.json`, `cnn_final_principal.json` (+ `rf_tuned_referencia.json`)
4. `scripts/estabilidade_modelos.py` — **o molde do bootstrap pareado**
5. `docs/execucao/APENDICE_A_inventario.md` — seções 1, 3 e 9

---

# B5.1 — Avaliação final no teste lacrado

## O contrato, em cinco regras

1. **Os limiares vêm dos JSONs de validação, lidos do disco.** Nenhum é recalculado,
   nenhum é 0,50. `carregar_modelo_ajustado` já **recusa** um JSON cujo
   `selecao_limiar.conjunto` não seja `"validacao"` — essa guarda é o que vale aqui,
   e é o motivo de a CNN ter entrado em `MODELOS_PRINCIPAIS` no B4.7.
2. **Nenhum modelo é treinado.** Todos são **carregados** dos artefatos persistidos.
3. **Execução única.** O script grava e encerra.
4. **Quatro linhas de resultado**, não três: RF principal, RF referência, SVM
   principal, CNN principal. O braço de referência entra porque quantifica o custo da
   subamostragem — e **tem de vir com a ressalva** de que não é concorrente direto de
   SVM/CNN.
5. **A régua é `avaliar`**, a mesma função dos três. Nenhuma métrica nova.

## O script

`scripts/avaliar_teste_lacrado.py`:

```python
"""Avaliacao final no teste lacrado — EXECUCAO UNICA.

POR QUE ESTE SCRIPT NAO TREINA NADA:
    o teste e a prova final. Treinar aqui, ainda que "so para conferir",
    significaria que os hiperparametros e os limiares foram escolhidos com o
    teste em vista. Todos os modelos sao CARREGADOS, e o limiar de cada um vem
    do JSON da VALIDACAO — a guarda de conjunto em carregar_modelo_ajustado
    recusa qualquer outra procedencia.

GUARDA DE EXECUCAO UNICA:
    se results/metricas/teste_lacrado.json JA EXISTIR, o script aborta com erro
    explicito, exigindo --forcar e a razao por escrito. Nao e burocracia: uma
    reexecucao distraida sobrescreveria a unica medida honesta do trabalho, e o
    Bloco 3 ja perdeu um bloco de limitacao por edicao silenciosa de artefato
    gerado.
"""
```

Implemente essa guarda. Ao usar `--forcar`, exija também `--razao "..."` e grave a
razão dentro do JSON, num campo `reexecucoes: [{data, razao}]`.

## Saídas

**`results/metricas/teste_lacrado.json`:**

```json
{
  "conjunto": "teste", "n": 22227,
  "protocolo": {
    "regra": "score >= limiar",
    "origem_dos_limiares": "JSON de validacao de cada modelo, campo selecao_limiar.limiar",
    "nenhum_modelo_treinado_aqui": true,
    "execucao": "unica"
  },
  "modelos": {
    "rf_tuned_principal":  {"limiar": 0.6516,  "origem_limiar": "...", "...": "saida de avaliar()"},
    "rf_tuned_referencia": {"limiar": 0.6196,  "...": "..."},
    "svm_tuned_principal": {"limiar": -0.0329, "...": "..."},
    "cnn_final_principal": {"limiar": 0.0,     "...": "..."}
  },
  "delta_validacao_teste": {
    "nota": "diferenca f1_macro(teste) - f1_macro(validacao) por modelo; |delta| pequeno indica que a validacao nao foi sobreajustada pela selecao de limiar",
    "rf_tuned_principal": 0.0, "svm_tuned_principal": 0.0, "cnn_final_principal": 0.0
  },
  "hashes": {"features": "51b2f439...", "split": "9143f0c7...", "subamostra": "654cb796..."},
  "ambiente": {"...": "..."}
}
```

O campo `delta_validacao_teste` é o mais interessante do arquivo e você o ganha de
graça: se o f1_macro no teste cair muito frente à validação, o limiar estava
sobreajustado à validação — e isso é uma observação honesta a fazer no texto, não um
problema a esconder. Espere deltas pequenos (a seleção de limiar sobre 22 mil amostras
é bem determinada), e **diga isso quando for o caso**: é evidência de que o protocolo
funcionou.

**Matrizes de confusão** no teste, uma por modelo, via `plotar_matriz_confusao`.

---

# B5.2 — Comparação experimental fechada

## A tabela principal

`results/metricas/comparacao_final.csv` + a versão para o texto. Colunas:

| modelo | braço | n treino | limiar | acurácia | f1_bonafide | f1_spoof | **f1_macro** | **EER** | ROC-AUC | recall bonafide | latência ms | ms/áudio (lote) | pipeline completo ms | hardware |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

Quatro linhas (RF principal, RF referência, SVM principal, CNN principal) × duas
tabelas (validação e teste). Monte-as **lendo os JSONs**, nunca digitando números à
mão — é o que garante que a tabela do texto e o artefato não divirjam.

### Três frases que a tabela obriga a escrever

1. **Sobre o braço de referência.** «O braço de referência (RF no treino completo de
   103.723) **não é concorrente direto** de SVM e CNN: ele existe para quantificar o
   custo da subamostragem de 30k, que foi imposta pela complexidade
   O(n²)–O(n³) do SVM-RBF. A comparação entre modelos é a do braço principal, em que
   os três viram as mesmas 30.000 amostras.» — exigência textual do orientador.
2. **Sobre o EER e a literatura.** Os valores **não** são comparáveis ao EER de 1,32%
   de Yamagishi et al. (2022): o protocolo aqui é split interno aleatório **por
   utterance**, e o oficial do ASVspoof é deliberadamente **cross-attack**. Ser
   independente de limiar remove a arbitrariedade do 0,50; **não** remove a diferença
   de protocolo.
3. **Sobre o custo.** «Não existe "o modelo mais barato" sem dizer o regime»: por
   áudio o SVM é ~2,1× mais barato que o RF, mas em lote o RF é ~19× melhor. A CNN
   acrescenta a dimensão que de fato importa para a pergunta de pesquisa — **exige
   GPU ou não?** — e é o par CNN-GPU / CNN-CPU que responde.

## O teste estatístico: bootstrap pareado

**Isto é o que decide se a diferença é real**, e o molde está em
`scripts/estabilidade_modelos.py`. Estenda-o para três modelos.

Por que pareado, e não IC individual: os três modelos são avaliados **exatamente nas
mesmas** 22.226 (validação) / 22.227 (teste) amostras, então os erros são
**correlacionados**. É a diferença reamostrada **em conjunto** — um único vetor de
índices por reamostragem — que responde à pergunta.

```
1.000 reamostragens, um vetor de indices por reamostragem, aplicado aos TRES
vetores de score simultaneamente.
Pares: SVM - RF, CNN - RF, CNN - SVM.
Metricas: Δf1_macro e ΔEER, com IC95 e a fracao de reamostragens em que
o primeiro vence.
```

Referência já medida (validação, SVM − RF): Δf1_macro **+0,0762** IC95
[0,0663; 0,0856]; ΔEER **−0,0466** IC95 [−0,0560; −0,0377]; SVM melhor em **100%**
das 1.000 reamostragens.

**Como ler:** IC que não contém zero ⇒ a diferença é real. IC que contém zero ⇒
**diga «não distinguíveis neste protocolo»**, e não «empataram» nem «X é ligeiramente
melhor». Essa é a diferença entre interpretação crítica e descritiva, e a banca nota.

**Rotule a heurística como heurística.** O `estabilidade_rf_svm.json` já faz isso:
a variação do RF entre 5 sementes (±0,0004) e os IC individuais de cada modelo são
úteis para contexto, mas **não são o teste** — comparar contra a maior dispersão
individual ignora a correlação entre os erros. Mantenha esse rigor ao incluir a CNN.

## Análise por classe, ataque e codec

Rode, com os três modelos em `MODELOS_PRINCIPAIS`:

```bash
python -m scripts.diagnostico_por_ataque    # A07-A19, limiar do protocolo
python -m scripts.diagnostico_por_codec     # banda estreita x larga
```

Eles **carregam** os artefatos e leem o limiar do JSON — nada é re-treinado.

**O que você já sabe e deve confrontar com a CNN:**

- **Por ataque:** o EER varia de 0,0628 (A13) a 0,3548 (A16) no RF — amplitude
  **0,2920** — e de 0,0617 (A09) a 0,2598 (A16) no SVM. A leitura registrada é que
  *a dificuldade varia muito mais entre sistemas de síntese (até 0,29 de EER) do que
  entre modelos (ΔEER 0,0466)*. **Se isso valer também para a CNN, é um resultado
  forte** — e é evidência direta a favor do risco declarado na limitação do split.
- **Por codec:** a hipótese da banda alta se sustenta nos dois modelos clássicos.
  RF: 0,6958 (estreita) × 0,7510 (larga); SVM: 0,7711 × 0,8397. Como o **EER é
  independente de limiar**, o contraste não é artefato do limiar. Confronte a CNN:
  ela **vê a banda inteira** (`fmax=8000`) e preserva a estrutura tempo-frequência,
  então há uma hipótese interessante a testar na leitura — se o ganho da banda larga
  for **maior** na CNN, isso sustenta que a representação preservada capta melhor os
  artefatos acima de 4 kHz.

## Gráficos finais

| figura | conteúdo |
|---|---|
| `comparacao_f1_eer.png` | barras agrupadas: f1_macro e EER, 4 modelos, validação × teste |
| `comparacao_roc.png` | curvas ROC dos três modelos na validação, num eixo só, com os EER anotados |
| `comparacao_tempos.png` | tempos: latência e lote, RF-CPU, SVM-CPU, CNN-GPU, CNN-CPU. **Escala logarítmica** (as ordens de grandeza são muito distintas) e o pipeline completo ao lado da predição pura |
| `comparacao_por_ataque.png` | EER por ataque A07–A19, três séries |
| `comparacao_por_codec.png` | f1_macro e EER por codec, três séries |

Nas figuras de tempo: **rotule o hardware em cada barra**. Uma barra «CNN» sem dizer
«GPU» é a única forma de essa figura mentir.

## Verificação (o passo que o projeto sempre fez e vale manter)

Estenda `scripts/guarda_reproducao.py` para cobrir os artefatos novos da CNN. Ele
compara campo a campo contra cópias em `_pre_revisao/`, deixando de fora as medidas
de relógio. Duas pegadinhas que ele já pegou no Bloco 3 e que podem reaparecer:

- **limitação escrita à mão dentro de artefato gerado é destruída pela
  re-execução.** Se algo precisa sobreviver, o código tem de **derivá-lo**.
- **soma paralela em ponto flutuante não é reprodutível bit a bit.** No RF isso levou
  `predizer_rf` a forçar `n_jobs=1`. Na CNN o análogo é o cuDNN — já tratado por
  `fixar_seeds_torch`. Se `n_candidatos` da CNN variar entre execuções, é isto, e a
  resposta é a mesma: registrar e tratar, não ignorar.

---

## Armadilhas deste dia

| armadilha | consequência |
|---|---|
| rodar o teste antes de o script estar revisado | queima o único uso permitido |
| recalcular limiar no teste | destrói o protocolo dos três modelos de uma vez |
| digitar números na tabela em vez de ler dos JSONs | tabela do texto divergindo do artefato — e é sempre a tabela que a banca lê |
| comparar o braço de referência com SVM/CNN sem a ressalva | contraria exigência textual do orientador |
| citar o EER ao lado do 1,32% de Yamagishi sem a ressalva de protocolo | erro metodológico que a banca pega na hora |
| IC que contém zero descrito como «X é melhor» | interpretação descritiva onde se pediu crítica |
| barra «CNN» na figura de tempo sem o hardware | a figura passa a afirmar algo falso |
| mudar um modelo depois de 29/09 | proibido, salvo erro grave |

---

## Critério de pronto

**B5.1**

- [ ] `scripts/avaliar_teste_lacrado.py` com guarda de execução única (`--forcar` +
      `--razao`)
- [ ] os quatro limiares lidos dos JSONs de **validação**, nenhum recalculado
- [ ] nenhum modelo treinado no script
- [ ] `results/metricas/teste_lacrado.json` com as quatro linhas e o
      `delta_validacao_teste`
- [ ] quatro matrizes de confusão no teste
- [ ] rodado **uma vez**

**B5.2**

- [ ] `comparacao_final.csv`, montado por leitura dos JSONs
- [ ] bootstrap pareado com os três pares (SVM−RF, CNN−RF, CNN−SVM), 1.000
      reamostragens, IC95, fração de vitórias — em validação **e** teste
- [ ] heurísticas rotuladas como heurísticas
- [ ] diagnóstico por ataque e por codec com os três modelos
- [ ] as cinco figuras finais, com hardware rotulado nas de tempo
- [ ] as três frases obrigatórias escritas (braço de referência, EER × literatura,
      custo por regime)
- [ ] `guarda_reproducao.py` estendido
- [ ] commit feito

## Commits (dois, separados)

```
B5.1: avaliacao final no teste lacrado — execucao unica

- scripts/avaliar_teste_lacrado.py: carrega RF principal, RF referencia, SVM e
  CNN dos artefatos persistidos e aplica, em cada um, o limiar lido do SEU JSON
  de validacao. Nenhum modelo treinado, nenhum limiar recalculado. A guarda de
  conjunto de carregar_modelo_ajustado recusa limiar de outra procedencia.
- guarda de execucao unica: aborta se teste_lacrado.json existir, exigindo
  --forcar e --razao, gravada no proprio artefato.
- teste_lacrado.json com as quatro linhas, matrizes de confusao e o campo
  delta_validacao_teste (f1_macro teste - validacao por modelo), que mede se a
  selecao de limiar sobreajustou a validacao.
```

```
B5.2: comparacao experimental fechada — RF x SVM x CNN

- comparacao_final.csv montado por LEITURA dos JSONs (tabela do texto e
  artefato nao podem divergir), validacao e teste, quatro linhas.
- bootstrap pareado estendido a tres modelos: um vetor de indices por
  reamostragem aplicado aos tres vetores de score, 1.000 reamostragens, IC95 de
  Δf1_macro e ΔEER para SVM-RF, CNN-RF e CNN-SVM, mais a fracao de
  reamostragens em que cada um vence. Heuristicas (sementes, IC individual)
  rotuladas como heuristicas, nao como o teste.
- diagnostico por ataque (A07-A19) e por codec (banda estreita x larga) com os
  tres modelos, carregados e com o limiar do protocolo.
- cinco figuras finais; nas de tempo, o hardware esta rotulado em cada barra.
- guarda_reproducao.py estendida aos artefatos da CNN.

A PARTIR DESTE COMMIT NENHUMA MUDANCA NOS MODELOS, SALVO ERRO GRAVE.
```

## Se atrasar

**O B5.1 não se simplifica nem se adia**: sem o teste não há resultado final.

No B5.2, corte nesta ordem e registre:

1. as figuras `comparacao_por_ataque.png` e `comparacao_por_codec.png` viram
   **tabelas** (os CSVs já saem dos scripts de diagnóstico);
2. o bootstrap pareado roda **só na validação** (onde já há a referência do Bloco 3),
   com o teste reportado como medida pontual. Limitação: «o intervalo de confiança da
   diferença foi estimado na validação; no teste reporta-se a medida pontual, por
   restrição de cronograma»;
3. `comparacao_roc.png` sai (o EER já resume a informação).

**Nunca corte:** a tabela comparativa, o bootstrap pareado em pelo menos um conjunto,
e as três frases obrigatórias. São elas que transformam quatro linhas de números na
**resposta à pergunta de pesquisa**.
