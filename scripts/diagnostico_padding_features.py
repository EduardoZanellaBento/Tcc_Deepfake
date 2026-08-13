"""
Diagnóstico de padding (4.1) — correlação das 44 features com prop_fala
=======================================================================

SÓ LÊ E REPORTA. Nenhuma extração de áudio; usa o features.csv existente.

HIPÓTESE EM TESTE:
    O pipeline é VAD -> padronizar para 4,0 s. Quem perde mais sinal no VAD
    recebe mais zero-padding, e os frames de silêncio puro entram na média e
    no desvio-padrão de MFCC, ZCR e centróide. Ou seja: prop_fala foi excluída
    do X (colunas_features), mas pode estar voltando PELA PORTA DOS FUNDOS,
    codificada dentro das 44 features.

LEITURA:
    Features com |r| alto contra prop_fala são candidatas a estar medindo
    padding, não acústica. O cruzamento decisivo é com o top10 de importância
    do RF: se as features mais importantes do modelo são também as mais
    correlacionadas com prop_fala, o atalho de silêncio está DENTRO do modelo.

RECORTE: universo eval (148.176), o aprovado pelo orientador.

SAÍDAS:
    results/metricas/padding_corr_features_propfala.csv
    results/figuras/padding_corr_features_propfala.png
    + conclusão em texto no stdout

Rode a partir da raiz:  python -m scripts.diagnostico_padding_features
"""

import json
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.split import carregar_dados_split, colunas_features

RAIZ = Path(__file__).resolve().parents[1]


def main() -> None:
    # carregar_dados_split já entrega o universo eval (split novo) validado
    df = carregar_dados_split(RAIZ)
    cols = colunas_features(df)
    print(f"universo: {len(df)} áudios (eval) | {len(cols)} features")

    # ---- Correlação de cada feature com prop_fala ---------------------------
    pear = df[cols].corrwith(df["prop_fala"], method="pearson")
    spear = df[cols].corrwith(df["prop_fala"], method="spearman")

    res = pd.DataFrame({
        "feature": cols,
        "pearson": pear.round(4).values,
        "spearman": spear.round(4).values,
    })
    res["abs_pearson"] = res["pearson"].abs()
    res = res.sort_values("abs_pearson", ascending=False).reset_index(drop=True)

    # ---- Cruzamento com o top10 de importância do RF ------------------------
    top10 = {}
    for nome_json in ["rf_baseline_eval.json", "rf_baseline.json"]:
        caminho = RAIZ / "results" / "metricas" / nome_json
        if caminho.exists():
            with open(caminho, encoding="utf-8") as f:
                top10[nome_json] = list(json.load(f)["top10_features"].keys())
    em_top10 = set().union(*top10.values()) if top10 else set()
    res["no_top10_rf"] = res["feature"].isin(em_top10)

    dir_met = RAIZ / "results" / "metricas"
    dir_met.mkdir(parents=True, exist_ok=True)
    res.drop(columns="abs_pearson").to_csv(
        dir_met / "padding_corr_features_propfala.csv", index=False)

    print("\n--- 44 features × prop_fala, ordenado por |pearson| ---")
    print(res.drop(columns="abs_pearson").to_string(index=False))

    # ---- Figura: barras ordenadas -------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 11))
    ordenado = res.iloc[::-1]  # maior |r| no topo
    cores = ["#c44e52" if t else "#4c72b0" for t in ordenado["no_top10_rf"]]
    ax.barh(ordenado["feature"], ordenado["pearson"], color=cores)
    ax.axvline(0, color="black", lw=0.8)
    for lim in (-0.3, 0.3):
        ax.axvline(lim, ls="--", color="gray", lw=0.8)
    ax.set_xlabel("correlação de Pearson com prop_fala")
    ax.set_title("Features × prop_fala — universo eval\n"
                 "(vermelho = feature no top10 de importância do RF)")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    dir_fig = RAIZ / "results" / "figuras"
    dir_fig.mkdir(parents=True, exist_ok=True)
    fig.savefig(dir_fig / "padding_corr_features_propfala.png", dpi=150)
    plt.close(fig)

    # ---- Conclusão -----------------------------------------------------------
    fortes = res[res["abs_pearson"] > 0.3]
    top_rf_fortes = fortes[fortes["no_top10_rf"]]
    print("\n" + "=" * 70)
    print("CONCLUSÃO")
    print("=" * 70)
    print(f"""
1. {len(fortes)} de {len(res)} features têm |r_pearson| > 0,3 contra prop_fala.
2. Dessas, {len(top_rf_fortes)} estão no top10 de importância do RF:
   {', '.join(top_rf_fortes['feature']) if len(top_rf_fortes) else '(nenhuma)'}
3. mfcc4_std (a feature MAIS importante do RF) tem r_pearson =
   {res.loc[res['feature'] == 'mfcc4_std', 'pearson'].iloc[0]:+.4f} e
   r_spearman = {res.loc[res['feature'] == 'mfcc4_std', 'spearman'].iloc[0]:+.4f}.
   |r| alto aqui é o sinal forte de que o padding contamina o que o modelo usa.
   A resposta causal (quanto da distorção é padding de fato) vem do piloto de
   mascaramento (4.3).

CSV : {dir_met / 'padding_corr_features_propfala.csv'}
PNG : {dir_fig / 'padding_corr_features_propfala.png'}
""")


if __name__ == "__main__":
    main()
