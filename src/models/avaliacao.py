"""
Avaliação compartilhada — métricas e matriz de confusão (RF, SVM e CNN)
=======================================================================

POR QUE ISTO EXISTE:
    As três funções abaixo nasceram dentro de treinar_rf.py, mas nada nelas é
    específico do Random Forest: o EER, o dicionário de métricas e a matriz de
    confusão valem para qualquer classificador binário bonafide × spoof. Num
    trabalho cuja pergunta central é COMPARAR modelos, a régua de avaliação
    precisa ser uma só — se cada pipeline (RF, SVM, CNN) tivesse a sua cópia,
    uma divergência silenciosa entre cópias invalidaria a comparação.

    Tudo que é específico de um modelo (nome nos artefatos, título do plot)
    entra por parâmetro (`nome`, `titulo`) — nada de RF hard-coded aqui.

A REGRA DE DECISÃO É UMA SÓ (protocolo do Bloco 3):
    y_pred = (score >= limiar), com o limiar SELECIONADO NA VALIDAÇÃO e apenas
    APLICADO em qualquer outro conjunto. `avaliar` recebe o limiar e deriva o
    y_pred internamente — nenhum pipeline recebe y_pred de fora, logo nenhum
    pipeline consegue usar outra regra (p.ex. o argmax do `modelo.predict()`,
    que diverge do `>=` nos empates — ver nota_divergencia_f1.md) por acidente.
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")          # backend sem janela: só salva arquivo
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
)

# Convenção única de decisão (registrada em nota_divergencia_f1.md): o empate
# exato no limiar decide SPOOF. `> limiar`, `>= limiar` e `argmax` divergem
# entre si justamente nos empates — três regras diferentes com o mesmo nome
# "limiar 0,50". Aqui a regra é uma, escrita, e compartilhada por RF, SVM e CNN.
REGRA_DECISAO = "score >= limiar"


def aplicar_limiar(scores: np.ndarray, limiar: float) -> np.ndarray:
    """Converte scores em decisão binária pela regra única do protocolo.

    Funciona em QUALQUER escala de score: predict_proba do RF ([0, 1]) e
    decision_function do SVM (real, centrado em zero) passam pela mesma regra.
    """
    return (np.asarray(scores) >= limiar).astype(int)


def selecionar_limiar(y_true, scores, criterio: str = "f1_macro",
                      conjunto: str = "validacao") -> dict:
    """Seleciona o limiar de decisão que maximiza `criterio` sobre (y_true, scores).

    DECISÕES DE PROJETO (Bloco 3 — valem para RF, SVM e CNN):

    1. Candidatos = np.unique(scores), NÃO uma grade fixa. Sob a regra `>=`,
       qualquer limiar entre dois scores observados consecutivos produz
       exatamente a mesma partição — logo os valores únicos são o conjunto
       candidato COMPLETO e MÍNIMO. Uma grade de 0,05 cai *entre* degraus
       (o predict_proba do RF tem ~79 valores distintos na validação) e pode
       passar ao largo do ótimo. Custo: ~10³ candidatos × 22k amostras, trivial.
    2. Nenhuma suposição de escala: nada de np.arange(0, 1, ...). O
       decision_function do SVM é real e centrado em zero; funciona igual.
    3. Empates no máximo: np.argmax devolve o PRIMEIRO máximo, isto é, o MENOR
       limiar entre os empatados. `n_empates_no_maximo` sai no retorno — muitos
       empates indicam score de granularidade grossa e merecem nota no texto.
    4. Guarda de conjunto: o retorno registra em qual conjunto o limiar foi
       escolhido (`conjunto`, default 'validacao'). A avaliação final do teste
       lacrado só aceita limiar lido de um JSON de validação — a regra que
       protege o teste mora no código, não só no texto.

    Returns:
        dict com criterio, regra, conjunto, limiar, o valor do critério no
        ótimo, n_candidatos e n_empates_no_maximo.
    """
    if criterio != "f1_macro":
        raise ValueError(f"Critério de seleção não implementado: '{criterio}'. "
                         "Implementado: 'f1_macro'.")
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    candidatos = np.unique(scores)
    valores = np.array([
        f1_score(y_true, aplicar_limiar(scores, c), average="macro",
                 zero_division=0)
        for c in candidatos
    ])
    i = int(np.argmax(valores))                     # primeiro máximo = MENOR limiar
    n_empates = int(np.sum(valores == valores[i]))
    return {
        "criterio": criterio,
        "regra": REGRA_DECISAO,
        "conjunto": conjunto,
        "limiar": float(candidatos[i]),
        criterio: float(valores[i]),
        "n_candidatos": int(len(candidatos)),
        "n_empates_no_maximo": n_empates,
    }


def calcular_eer(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Equal Error Rate — a métrica padrão do ASVspoof.

    O QUE É: acurácia/F1 dependem de um limiar fixo (0,5). O EER remove essa
    arbitrariedade — ele varre TODOS os limiares possíveis e encontra o ponto em
    que a taxa de falsos positivos (FPR) iguala a de falsos negativos (FNR). É esse
    valor comum que se reporta. Menor = melhor. Por ser independente de limiar, é
    o que permite comparar esse número com o da literatura (Yamagishi et al. 2022
    reportam EER de 1,32% em LA).

    Convenção adotada aqui: classe positiva = spoof (classe_binaria = 1), e `scores`
    é a probabilidade predita de ser spoof. Trocar a classe positiva troca o significado de FPR e FNR.

    Returns:
        (eer, limiar_no_eer)
    """
    fpr, tpr, limiares = roc_curve(y_true, scores, pos_label=1)
    fnr = 1 - tpr
    # O ponto onde as duas curvas se cruzam: |FNR - FPR| mínimo.
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = (fpr[i] + fnr[i]) / 2
    return float(eer), float(limiares[i])


def avaliar(y_true, scores, nome: str, limiar: float) -> dict:
    """Calcula o conjunto de métricas do config.yaml a partir de (scores, limiar).

    ASSINATURA NOVA (Bloco 3) — quebra deliberada de compatibilidade: a versão
    anterior recebia `y_pred` pronto de fora, vindo de `modelo.predict()`, que
    decide por argmax e diverge da regra `>=` nos empates (as 25 amostras de
    nota_divergencia_f1.md). Agora o y_pred é derivado AQUI, por
    `aplicar_limiar` — nenhum chamador consegue usar outra regra por acidente.

    `calcular_eer` continua independente de limiar; `limiar_eer` é reportado ao
    lado do limiar usado, para comparação.
    """
    y_pred = aplicar_limiar(scores, limiar)
    m = {
        "modelo": nome,
        "limiar": float(limiar),
        "regra": REGRA_DECISAO,
        "acuracia": float(accuracy_score(y_true, y_pred)),
        # zero_division=0: se o modelo NUNCA prevê uma classe, a precisão dela é 0/0.
        # Sem isso o sklearn emite warning e devolve 0 silenciosamente.
        "precisao_spoof": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_spoof": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_spoof": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        # A classe minoritária é a que revela se o modelo realmente aprendeu:
        "precisao_bonafide": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "recall_bonafide": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "f1_bonafide": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        # f1 macro: média não ponderada das duas classes. Com dados desbalanceados, é MUITO mais informativo que a acurácia — não deixa a classe majoritária esconder o fracasso na minoritária.
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    eer, limiar = calcular_eer(y_true, scores)
    m["eer"] = eer
    m["limiar_eer"] = limiar
    return m


def plotar_matriz_confusao(cm: np.ndarray, caminho: Path, titulo: str) -> None:
    """Matriz de confusão com contagens absolutas E percentual por linha.

    Percentual POR LINHA (normalizado pelo total real de cada classe) é o que
    importa aqui: com 8,8:1, os números absolutos da linha 'spoof' esmagam
    visualmente os da 'bonafide' e escondem o erro que interessa.
    """
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    nomes = ["bonafide (0)", "spoof (1)"]
    ax.set_xticks([0, 1], labels=nomes)
    ax.set_yticks([0, 1], labels=nomes)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title(titulo)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n({cm_pct[i, j]:.1f}%)",
                    ha="center", va="center",
                    color="white" if cm_pct[i, j] > 50 else "black",
                    fontsize=11)

    fig.colorbar(im, ax=ax, label="% da classe real")
    fig.tight_layout()
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    print(f"Matriz de confusão salva em {caminho}")
