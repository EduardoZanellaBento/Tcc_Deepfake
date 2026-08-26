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


def avaliar(y_true, y_pred, scores, nome: str) -> dict:
    """Calcula o conjunto de métricas do config.yaml e imprime a leitura crítica."""
    m = {
        "modelo": nome,
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
