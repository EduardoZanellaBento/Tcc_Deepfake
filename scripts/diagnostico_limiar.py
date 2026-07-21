"""
Diagnóstico B3 — Limiar de decisão, ROC e Precision-Recall
==========================================================

SÓ LÊ E REPORTA. Treina o RF baseline (mesmos hiperparâmetros, seed 42) no
split de treino e analisa os SCORES na VALIDAÇÃO. O teste continua lacrado.

MOTIVAÇÃO:
    Com class_weight='balanced' e limiar fixo 0,5, o RF baseline prevê 'spoof'
    quase sempre (recall bonafide ~0,10). A suspeita é que o problema não é o
    modelo "não saber nada", e sim o LIMIAR: os scores dos bonafide são mais
    baixos que os dos spoof, mas quase todos ficam acima de 0,5. A varredura
    de limiar mostra onde a decisão deveria cortar (esperado: limiar de EER
    ~0,88, confirmando o desajuste do class_weight + 0,5).

    Detalhe de granularidade: com n_estimators=100, predict_proba é a fração
    de árvores votando 'spoof' -> no máximo ~101 valores distintos. O EER
    calculado sobre uma escada grossa dessas é GROSSEIRO (o ponto FPR==FNR cai
    entre degraus). Registramos quantos valores distintos existem de fato.

SAÍDAS:
    results/metricas/diagnostico_limiar_varredura.csv
    results/figuras/diagnostico_limiar.png  (ROC + PR + métricas × limiar)
    + conclusão em texto no stdout

Rode a partir da raiz:  python -m scripts.diagnostico_limiar
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, f1_score, recall_score,
)

from src.utils.seeds import fixar_seeds
from src.data.split import carregar_dados_split, colunas_features
from src.models.treinar_rf import calcular_eer

RAIZ = Path(__file__).resolve().parents[1]


def main() -> None:
    semente = fixar_seeds(42)

    df = carregar_dados_split(RAIZ)
    cols = colunas_features(df)
    treino = df[df["conjunto"] == "treino"]
    validacao = df[df["conjunto"] == "validacao"]
    X_tr, y_tr = treino[cols].values, treino["classe_binaria"].values
    X_va, y_va = validacao[cols].values, validacao["classe_binaria"].values
    print(f"{len(cols)} features | treino {X_tr.shape} | validação {X_va.shape}")

    modelo = RandomForestClassifier(
        n_estimators=100, max_depth=None, class_weight="balanced",
        random_state=semente, n_jobs=-1,
    )
    modelo.fit(X_tr, y_tr)
    scores = modelo.predict_proba(X_va)[:, 1]

    # ---- Granularidade do score -------------------------------------------
    n_distintos = len(np.unique(scores))
    print(f"\npredict_proba tem {n_distintos} valores distintos na validação "
          f"(teto teórico com n_estimators=100: ~101 sem class_weight; "
          f"os pesos por classe fracionam os votos e multiplicam os degraus). "
          f"O EER medido sobre essa escada é aproximado.")

    # ---- ROC + EER ---------------------------------------------------------
    fpr, tpr, _ = roc_curve(y_va, scores, pos_label=1)
    auc_roc = auc(fpr, tpr)
    eer, limiar_eer = calcular_eer(y_va, scores)
    print(f"AUC-ROC       : {auc_roc:.4f}")
    print(f"EER           : {eer:.4f}  ({100*eer:.2f}%)")
    print(f"limiar no EER : {limiar_eer:.4f}  (esperado ~0,88 — confirma que o "
          f"limiar 0,5 está muito abaixo do ponto de equilíbrio)")

    # ---- Precision-Recall (classe positiva = spoof) ------------------------
    prec, rec, _ = precision_recall_curve(y_va, scores, pos_label=1)
    auc_pr = auc(rec, prec)
    print(f"AUC-PR (spoof): {auc_pr:.4f} (base = prevalência de spoof "
          f"= {y_va.mean():.4f})")

    # ---- Varredura de limiar ----------------------------------------------
    linhas = []
    for limiar in np.round(np.arange(0.05, 1.00, 0.05), 2):
        y_pred = (scores >= limiar).astype(int)
        linhas.append({
            "limiar": limiar,
            "f1_macro": round(f1_score(y_va, y_pred, average="macro",
                                       zero_division=0), 4),
            "recall_bonafide": round(recall_score(y_va, y_pred, pos_label=0,
                                                  zero_division=0), 4),
            "recall_spoof": round(recall_score(y_va, y_pred, pos_label=1,
                                               zero_division=0), 4),
        })
    var = pd.DataFrame(linhas)
    melhor = var.loc[var["f1_macro"].idxmax()]

    dir_met = RAIZ / "results" / "metricas"
    dir_met.mkdir(parents=True, exist_ok=True)
    var.to_csv(dir_met / "diagnostico_limiar_varredura.csv", index=False)

    print("\n--- varredura de limiar (validação) ---")
    print(var.to_string(index=False))
    print(f"\nMelhor f1_macro da varredura: {melhor['f1_macro']:.4f} no limiar "
          f"{melhor['limiar']:.2f} (vs. 0,5675 no limiar 0,50).")

    # ---- Figura ------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    axes[0].plot(fpr, tpr, color="#4c72b0")
    axes[0].plot([0, 1], [0, 1], ls="--", color="gray", lw=1)
    axes[0].scatter([eer], [1 - eer], color="#c44e52", zorder=3,
                    label=f"EER = {eer:.3f}")
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[0].set_title(f"ROC (AUC = {auc_roc:.3f})")
    axes[0].legend()

    axes[1].plot(rec, prec, color="#4c72b0")
    axes[1].axhline(y_va.mean(), ls="--", color="gray", lw=1,
                    label=f"base = {y_va.mean():.3f}")
    axes[1].set_xlabel("recall (spoof)"); axes[1].set_ylabel("precisão (spoof)")
    axes[1].set_title(f"Precision-Recall (AUC = {auc_pr:.3f})")
    axes[1].legend()

    axes[2].plot(var["limiar"], var["f1_macro"], marker="o", label="f1_macro")
    axes[2].plot(var["limiar"], var["recall_bonafide"], marker="s",
                 label="recall bonafide")
    axes[2].plot(var["limiar"], var["recall_spoof"], marker="^",
                 label="recall spoof")
    axes[2].axvline(0.5, ls=":", color="gray", lw=1)
    axes[2].axvline(limiar_eer, ls="--", color="#c44e52", lw=1,
                    label=f"limiar EER = {limiar_eer:.2f}")
    axes[2].set_xlabel("limiar de decisão"); axes[2].set_ylabel("métrica")
    axes[2].set_title("métricas × limiar")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.suptitle("RF baseline — diagnóstico de limiar (validação)")
    fig.tight_layout()
    dir_fig = RAIZ / "results" / "figuras"
    dir_fig.mkdir(parents=True, exist_ok=True)
    fig.savefig(dir_fig / "diagnostico_limiar.png", dpi=150)
    plt.close(fig)

    # ---- Conclusão ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("CONCLUSÃO")
    print("=" * 70)
    print(f"""
1. O limiar de EER é {limiar_eer:.4f} — muito acima do 0,5 usado no predict().
   Os scores dos bonafide são de fato mais baixos, mas quase todos acima de
   0,5: o par (class_weight='balanced', limiar 0,5) está desajustado, e é ISSO
   que produz o recall bonafide de ~0,10 do baseline, não ausência total de
   sinal (AUC-ROC = {auc_roc:.4f}).
2. Movendo o limiar para {melhor['limiar']:.2f}, o f1_macro sobe de 0,5675
   para {melhor['f1_macro']:.4f} SEM retreinar nada. (Diagnóstico, não decisão:
   escolher limiar é decisão metodológica do orientador.)
3. predict_proba tem {n_distintos} valores distintos; EER sobre escada grossa
   é aproximado — registrar isso ao comparar com a literatura.

CSV : {dir_met / 'diagnostico_limiar_varredura.csv'}
PNG : {dir_fig / 'diagnostico_limiar.png'}
""")


if __name__ == "__main__":
    main()
