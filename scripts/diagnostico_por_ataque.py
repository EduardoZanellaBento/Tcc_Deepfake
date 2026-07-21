"""
Diagnóstico B2 — Desempenho do RF por sistema de ataque (A07–A19)
=================================================================

SÓ LÊ E REPORTA. Treina o RF baseline (mesmos hiperparâmetros, seed 42) no
split de treino e avalia na VALIDAÇÃO, quebrando recall e f1 por ataque.
O split de teste continua lacrado.

MOTIVAÇÃO:
    O eval do ASVspoof 2021 LA contém 13 sistemas de síntese (A07–A19; A01–A06
    não existem no eval). Um número agregado (f1_macro, EER) esconde que o
    modelo pode ir bem em TTS "fáceis" e despencar em conversão de voz difícil
    (ex.: A17). O recall por ataque é o mapa desse comportamento — e vira
    parágrafo do Capítulo 4.

MÉTRICAS POR ATAQUE:
    Para cada A07–A19: n na validação, recall (fração de spoofs daquele
    sistema corretamente detectados) e f1 calculado no subconjunto
    {bonafide ∪ ataque} (o f1 da classe spoof precisa dos bonafide como
    negativos). Para 'bonafide': recall da classe 0 (specificidade).

    Como o limiar 0,5 com class_weight='balanced' satura o modelo em 'spoof'
    (ver B3), o recall binarizado pode ficar ~1,0 em todos os ataques e não
    mostrar contraste nenhum. Por isso reportamos também o SCORE contínuo:
    média de P(spoof) por ataque e EER do subconjunto {bonafide ∪ ataque} —
    é aí que os sistemas difíceis aparecem.

SAÍDAS:
    results/metricas/diagnostico_por_ataque.csv
    results/figuras/diagnostico_por_ataque.png
    + conclusão em texto no stdout

Rode a partir da raiz:  python -m scripts.diagnostico_por_ataque
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score

from src.utils.seeds import fixar_seeds
from src.data.split import carregar_dados_split, colunas_features
from src.models.treinar_rf import calcular_eer

RAIZ = Path(__file__).resolve().parents[1]


def main() -> None:
    semente = fixar_seeds(42)

    df = carregar_dados_split(RAIZ)
    cols = colunas_features(df)

    labels = pd.read_csv(RAIZ / "data" / "processed" / "labels.csv",
                         usecols=["arquivo", "ataque"])
    n_antes = len(df)
    df = df.merge(labels, on="arquivo", how="inner")
    assert len(df) == n_antes, "merge perdeu linhas — investigar"

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
    y_pred = modelo.predict(X_va)
    scores = modelo.predict_proba(X_va)[:, 1]

    validacao = validacao.assign(y_pred=y_pred, score=scores)
    bona = validacao[validacao["classe_binaria"] == 0]

    # ---- Por ataque --------------------------------------------------------
    ataques = sorted(a for a in validacao["ataque"].unique() if a.startswith("A"))
    linhas = []
    for atq in ataques:
        sub = validacao[validacao["ataque"] == atq]
        recall = recall_score(sub["classe_binaria"], sub["y_pred"],
                              pos_label=1, zero_division=0)
        # f1 da classe spoof no subconjunto {bonafide ∪ este ataque}
        par = pd.concat([bona, sub])
        f1 = f1_score(par["classe_binaria"], par["y_pred"],
                      pos_label=1, zero_division=0)
        eer, _ = calcular_eer(par["classe_binaria"].values, par["score"].values)
        linhas.append({"ataque": atq, "n_validacao": len(sub),
                       "recall": round(recall, 4), "f1_vs_bonafide": round(f1, 4),
                       "media_prob_spoof": round(sub["score"].mean(), 4),
                       "eer_vs_bonafide": round(eer, 4)})

    recall_bona = recall_score(bona["classe_binaria"], bona["y_pred"],
                               pos_label=0, zero_division=0)
    linhas.append({"ataque": "bonafide", "n_validacao": len(bona),
                   "recall": round(recall_bona, 4), "f1_vs_bonafide": np.nan,
                   "media_prob_spoof": round(bona["score"].mean(), 4),
                   "eer_vs_bonafide": np.nan})

    res = pd.DataFrame(linhas)
    dir_met = RAIZ / "results" / "metricas"
    dir_met.mkdir(parents=True, exist_ok=True)
    res.to_csv(dir_met / "diagnostico_por_ataque.csv", index=False)

    print("\n--- recall e f1 por ataque (validação) ---")
    print(res.to_string(index=False))

    # ---- Figura ------------------------------------------------------------
    so_ataques = res[res["ataque"] != "bonafide"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].bar(so_ataques["ataque"], so_ataques["recall"], color="#4c72b0")
    axes[0].axhline(recall_bona, color="gray", ls="--", lw=1,
                    label=f"recall bonafide = {recall_bona:.3f}")
    axes[0].set_ylim(0, 1.02)
    axes[0].set_ylabel("recall (limiar 0,5)")
    axes[0].set_title("recall por ataque — saturado pelo limiar")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(so_ataques["ataque"], so_ataques["eer_vs_bonafide"], color="#c44e52")
    axes[1].set_ylabel("EER vs. bonafide (menor = melhor)")
    axes[1].set_title("EER por ataque — onde o contraste aparece")
    axes[1].grid(axis="y", alpha=0.3)
    fig.suptitle("RF baseline por sistema de ataque (validação)")
    fig.tight_layout()
    dir_fig = RAIZ / "results" / "figuras"
    dir_fig.mkdir(parents=True, exist_ok=True)
    fig.savefig(dir_fig / "diagnostico_por_ataque.png", dpi=150)
    plt.close(fig)

    # ---- Conclusão ---------------------------------------------------------
    pior = so_ataques.loc[so_ataques["eer_vs_bonafide"].idxmax()]
    melhor = so_ataques.loc[so_ataques["eer_vs_bonafide"].idxmin()]
    print("\n" + "=" * 70)
    print("CONCLUSÃO")
    print("=" * 70)
    print(f"""
No limiar 0,5 o recall satura (~1,0 em todos os ataques) porque o modelo
quase sempre prevê 'spoof' (recall bonafide = {recall_bona:.4f}; ver B3) —
o recall binarizado NÃO discrimina os sistemas.
No SCORE contínuo o contraste aparece: EER vs. bonafide varia de
{melhor['eer_vs_bonafide']:.4f} ({melhor['ataque']}, mais fácil) a
{pior['eer_vs_bonafide']:.4f} ({pior['ataque']}, mais difícil).
Sistemas com EER alto são os que produzem áudio mais próximo do bonafide no
espaço MFCC/ZCR/centróide. Este mapa vira parágrafo do Capítulo 4.

CSV : {dir_met / 'diagnostico_por_ataque.csv'}
PNG : {dir_fig / 'diagnostico_por_ataque.png'}
""")


if __name__ == "__main__":
    main()
