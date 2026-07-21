"""
Diagnóstico B5 — Desempenho do RF por codec (o achado central)
==============================================================

SÓ LÊ E REPORTA. Treina o RF baseline (mesmos hiperparâmetros, seed 42) no
split de treino e avalia na VALIDAÇÃO, quebrando as métricas por codec.
O split de teste continua lacrado.

HIPÓTESE:
    Os 7 codecs do ASVspoof 2021 LA dividem-se em banda estreita (alaw, ulaw,
    gsm, pstn — teto ~4 kHz) e banda larga (g722, opus, none — preservam
    >4 kHz). Os artefatos de síntese que ZCR e centróide espectral capturam
    vivem principalmente na banda alta; codecs telefônicos a destroem. Logo,
    espera-se desempenho MELHOR nos codecs de banda larga e PIOR nos de banda
    estreita. Como codec é fator perfeitamente balanceado (25.938 cada, mesma
    proporção spoof/bonafide), a comparação é limpa.

SAÍDAS:
    results/metricas/diagnostico_por_codec.csv
    results/figuras/diagnostico_por_codec.png
    + conclusão em texto no stdout

Rode a partir da raiz:  python -m scripts.diagnostico_por_codec
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

BANDA = {
    "alaw": "estreita", "ulaw": "estreita", "gsm": "estreita", "pstn": "estreita",
    "g722": "larga", "opus": "larga", "none": "larga",
}
ORDEM_CODECS = ["alaw", "ulaw", "gsm", "pstn", "g722", "opus", "none"]


def main() -> None:
    semente = fixar_seeds(42)

    # ---- Dados: features + split, depois metadados por 'arquivo' -----------
    df = carregar_dados_split(RAIZ)
    cols = colunas_features(df)   # calculado ANTES do merge: só as 44 acústicas

    labels = pd.read_csv(RAIZ / "data" / "processed" / "labels.csv",
                         usecols=["arquivo", "codec", "trim", "fase"])
    n_antes = len(df)
    df = df.merge(labels, on="arquivo", how="inner")
    assert len(df) == n_antes, "merge perdeu linhas — investigar"

    treino = df[df["conjunto"] == "treino"]
    validacao = df[df["conjunto"] == "validacao"]
    X_tr, y_tr = treino[cols].values, treino["classe_binaria"].values
    X_va, y_va = validacao[cols].values, validacao["classe_binaria"].values
    print(f"{len(cols)} features | treino {X_tr.shape} | validação {X_va.shape}")

    # ---- RF baseline (idêntico ao treinar_rf.py) ---------------------------
    modelo = RandomForestClassifier(
        n_estimators=100, max_depth=None, class_weight="balanced",
        random_state=semente, n_jobs=-1,
    )
    modelo.fit(X_tr, y_tr)

    y_pred = modelo.predict(X_va)
    scores = modelo.predict_proba(X_va)[:, 1]

    # ---- Métricas por codec ------------------------------------------------
    linhas = []
    for codec in ORDEM_CODECS:
        m = (validacao["codec"] == codec).values
        yt, yp, sc = y_va[m], y_pred[m], scores[m]
        eer, _ = calcular_eer(yt, sc)
        linhas.append({
            "codec": codec,
            "banda": BANDA[codec],
            "n": int(m.sum()),
            "n_bonafide": int((yt == 0).sum()),
            "f1_macro": round(f1_score(yt, yp, average="macro", zero_division=0), 4),
            "recall_bonafide": round(recall_score(yt, yp, pos_label=0, zero_division=0), 4),
            "recall_spoof": round(recall_score(yt, yp, pos_label=1, zero_division=0), 4),
            "eer": round(eer, 4),
        })

    res = pd.DataFrame(linhas)

    # Agregado por banda (média simples entre codecs; grupos são balanceados)
    agg = res.groupby("banda")[["f1_macro", "recall_bonafide", "eer"]].mean().round(4)

    dir_met = RAIZ / "results" / "metricas"
    dir_met.mkdir(parents=True, exist_ok=True)
    res.to_csv(dir_met / "diagnostico_por_codec.csv", index=False)

    print("\n--- métricas por codec (validação) ---")
    print(res.to_string(index=False))
    print("\n--- média por banda ---")
    print(agg.to_string())

    # ---- Figura: barras de f1_macro, recall_bonafide e EER por codec -------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True)
    cores = ["#c44e52" if BANDA[c] == "estreita" else "#4c72b0" for c in ORDEM_CODECS]
    for ax, met, titulo in zip(
        axes,
        ["f1_macro", "recall_bonafide", "eer"],
        ["f1_macro", "recall bonafide", "EER (menor = melhor)"],
    ):
        ax.bar(res["codec"], res[met], color=cores)
        ax.set_title(titulo)
        ax.set_xlabel("codec")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("RF baseline por codec — vermelho: banda estreita | azul: banda larga")
    fig.tight_layout()
    dir_fig = RAIZ / "results" / "figuras"
    dir_fig.mkdir(parents=True, exist_ok=True)
    fig.savefig(dir_fig / "diagnostico_por_codec.png", dpi=150)
    plt.close(fig)

    # ---- Conclusão ---------------------------------------------------------
    estreita = agg.loc["estreita"]
    larga = agg.loc["larga"]
    print("\n" + "=" * 70)
    print("CONCLUSÃO")
    print("=" * 70)
    print(f"""
Banda estreita (alaw, ulaw, gsm, pstn): f1_macro médio {estreita['f1_macro']:.4f},
recall bonafide médio {estreita['recall_bonafide']:.4f}, EER médio {estreita['eer']:.4f}.
Banda larga (g722, opus, none):         f1_macro médio {larga['f1_macro']:.4f},
recall bonafide médio {larga['recall_bonafide']:.4f}, EER médio {larga['eer']:.4f}.

Diferença (larga - estreita): f1_macro {larga['f1_macro']-estreita['f1_macro']:+.4f} |
recall bonafide {larga['recall_bonafide']-estreita['recall_bonafide']:+.4f} | EER {larga['eer']-estreita['eer']:+.4f}.

Leitura: se o desempenho é sistematicamente melhor na banda larga, a evidência
suporta a hipótese de que os artefatos discriminativos capturados por
ZCR/centróide vivem na banda alta (>4 kHz), que os codecs telefônicos removem.
Isso também REJEITA um fmax=4000 global (destruiria os 43% de banda larga) —
ver TODO metodológico 9.1 no config.yaml.

CSV : {dir_met / 'diagnostico_por_codec.csv'}
PNG : {dir_fig / 'diagnostico_por_codec.png'}
""")


if __name__ == "__main__":
    main()
