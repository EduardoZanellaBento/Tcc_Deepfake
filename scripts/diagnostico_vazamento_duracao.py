"""
Diagnóstico B1 — Vazamento de duração via prop_fala
===================================================

SÓ LÊ E REPORTA. Não altera o pipeline; avalia em treino/validação apenas.

HIPÓTESE:
    O pré-processamento faz VAD (remove silêncio) e depois padroniza a duração
    em 4 s preenchendo com zeros. Se bonafide e spoof tiverem durações de fala
    sistematicamente diferentes, a FRAÇÃO DE PADDING contamina MFCC/ZCR/
    centróide (todos agregados por média/std sobre janelas que incluem os
    zeros) — e `prop_fala` (diagnóstico do VAD) viraria um atalho de duração
    que prevê a classe sem olhar para artefato de síntese nenhum.

REFINAMENTO (achado da composição, ver B4):
    O subconjunto fase=='hidden' tem trim=='only_speech' (silêncio já cortado
    na origem, prop_fala ≈ 1 por construção). Ele é FILTRADO (trim=='notrim')
    antes da análise para não poluir as distribuições; as estatísticas dele
    são reportadas em separado, só para contraste.

TESTES:
    1. Estatísticas de prop_fala por classe + histograma sobreposto.
    2. DecisionTreeClassifier(max_depth=3, class_weight='balanced', seed 42)
       treinado SÓ em prop_fala (treino), avaliado na validação:
       f1_macro > ~0,60 => o atalho existe.
    3. Correlação Pearson E Spearman entre prop_fala, centroide_media,
       centroide_std, zcr_media, zcr_std e classe_binaria (|r|>0,3 destacado).

SAÍDAS:
    results/metricas/vazamento_prop_fala_stats.csv
    results/metricas/vazamento_prop_fala_arvore.csv
    results/metricas/vazamento_correlacoes.csv
    results/figuras/vazamento_hist_prop_fala.png
    + conclusão em texto no stdout

Rode a partir da raiz:  python -m scripts.diagnostico_vazamento_duracao
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, confusion_matrix

from src.utils.seeds import fixar_seeds
from src.data.split import carregar_dados_split
from src.models.treinar_rf import calcular_eer

RAIZ = Path(__file__).resolve().parents[1]
COLS_CORR = ["prop_fala", "centroide_media", "centroide_std",
             "zcr_media", "zcr_std", "classe_binaria"]


def main() -> None:
    semente = fixar_seeds(42)

    df = carregar_dados_split(RAIZ)
    labels = pd.read_csv(RAIZ / "data" / "processed" / "labels.csv",
                         usecols=["arquivo", "trim", "fase"])
    n_antes = len(df)
    df = df.merge(labels, on="arquivo", how="inner")
    assert len(df) == n_antes, "merge perdeu linhas — investigar"

    dir_met = RAIZ / "results" / "metricas"
    dir_fig = RAIZ / "results" / "figuras"
    dir_met.mkdir(parents=True, exist_ok=True)
    dir_fig.mkdir(parents=True, exist_ok=True)

    # ---- Contraste only_speech (reportado à parte, depois filtrado) --------
    os_ = df[df["trim"] == "only_speech"]["prop_fala"]
    print(f"Contraste — trim=='only_speech' (hidden, n={len(os_)}): "
          f"prop_fala média {os_.mean():.3f}, mediana {os_.median():.3f} "
          f"(≈1 por construção; excluído da análise abaixo).")

    df = df[df["trim"] == "notrim"].copy()
    print(f"Análise sobre trim=='notrim': {len(df)} utterances.\n")

    # ---- (1) Estatísticas de prop_fala por classe --------------------------
    stats = df.groupby("label")["prop_fala"].describe(
        percentiles=[0.25, 0.5, 0.75]).round(4)
    stats.to_csv(dir_met / "vazamento_prop_fala_stats.csv")
    print("--- prop_fala por classe (notrim) ---")
    print(stats.to_string())

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, cor in [("bonafide", "#4c72b0"), ("spoof", "#c44e52")]:
        ax.hist(df.loc[df["label"] == label, "prop_fala"], bins=50,
                density=True, alpha=0.55, color=cor, label=label)
    ax.set_xlabel("prop_fala (fração mantida pelo VAD)")
    ax.set_ylabel("densidade")
    ax.set_title("Distribuição de prop_fala por classe (trim=='notrim')")
    ax.legend()
    fig.tight_layout()
    fig.savefig(dir_fig / "vazamento_hist_prop_fala.png", dpi=150)
    plt.close(fig)

    # ---- (2) Árvore rasa treinada SÓ em prop_fala --------------------------
    treino = df[df["conjunto"] == "treino"]
    validacao = df[df["conjunto"] == "validacao"]
    X_tr = treino[["prop_fala"]].values
    y_tr = treino["classe_binaria"].values
    X_va = validacao[["prop_fala"]].values
    y_va = validacao["classe_binaria"].values

    arvore = DecisionTreeClassifier(max_depth=3, class_weight="balanced",
                                    random_state=semente)
    arvore.fit(X_tr, y_tr)
    y_pred = arvore.predict(X_va)
    scores = arvore.predict_proba(X_va)[:, 1]

    f1m = f1_score(y_va, y_pred, average="macro", zero_division=0)
    eer, _ = calcular_eer(y_va, scores)
    cm = confusion_matrix(y_va, y_pred, labels=[0, 1])

    res_arvore = pd.DataFrame([{
        "modelo": "arvore_so_prop_fala",
        "n_treino": len(X_tr), "n_validacao": len(X_va),
        "f1_macro": round(f1m, 4), "eer": round(eer, 4),
        "cm_bonafide_como_bonafide": int(cm[0, 0]),
        "cm_bonafide_como_spoof": int(cm[0, 1]),
        "cm_spoof_como_bonafide": int(cm[1, 0]),
        "cm_spoof_como_spoof": int(cm[1, 1]),
    }])
    res_arvore.to_csv(dir_met / "vazamento_prop_fala_arvore.csv", index=False)

    print("\n--- árvore (max_depth=3) treinada SÓ em prop_fala — validação ---")
    print(f"f1_macro : {f1m:.4f}   (limiar de alerta: > ~0,60)")
    print(f"EER      : {eer:.4f}")
    print("matriz de confusão (linhas=real, colunas=predito):")
    print(f"  bonafide -> [{cm[0,0]:>7}, {cm[0,1]:>7}]")
    print(f"  spoof    -> [{cm[1,0]:>7}, {cm[1,1]:>7}]")

    # ---- (3) Correlações Pearson e Spearman --------------------------------
    pear = df[COLS_CORR].corr(method="pearson").round(3)
    spear = df[COLS_CORR].corr(method="spearman").round(3)
    corr = pd.concat({"pearson": pear, "spearman": spear})
    corr.to_csv(dir_met / "vazamento_correlacoes.csv")

    print("\n--- correlação de Pearson ---")
    print(pear.to_string())
    print("\n--- correlação de Spearman ---")
    print(spear.to_string())

    print("\n--- pares com |r| > 0,3 (fora da diagonal) ---")
    destacados = []
    for metodo, mat in [("pearson", pear), ("spearman", spear)]:
        for i, a in enumerate(COLS_CORR):
            for b in COLS_CORR[i + 1:]:
                r = mat.loc[a, b]
                if abs(r) > 0.3:
                    destacados.append(f"  {metodo:<9} {a} × {b}: r = {r:+.3f}")
    print("\n".join(destacados) if destacados else "  (nenhum)")

    # ---- Conclusão ---------------------------------------------------------
    atalho = "SIM — o atalho existe" if f1m > 0.60 else "NÃO no limiar de alerta"
    print("\n" + "=" * 70)
    print("CONCLUSÃO")
    print("=" * 70)
    print(f"""
1. prop_fala sozinho prevê a classe? f1_macro = {f1m:.4f} na validação
   ({atalho}; referência: > ~0,60 indicaria atalho de duração).
2. As correlações acima mostram o quanto prop_fala se mistura com as
   features declaradas (centróide/ZCR): pares |r|>0,3 estão destacados.
3. prop_fala NÃO entra no X dos modelos (colunas_features a exclui), mas o
   PADDING que a origina contamina as 44 features. A correção do padding
   está no lote único de re-extração futura — decisão do orientador.
""")


if __name__ == "__main__":
    main()
