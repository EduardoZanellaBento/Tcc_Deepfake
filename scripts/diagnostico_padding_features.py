"""
Diagnóstico de padding — correlação das 44 features com prop_fala (e n_frames_validos)
======================================================================================

SÓ LÊ E REPORTA. Nenhuma extração de áudio; usa o features.csv existente.

O QUE ESTE SCRIPT MEDE HOJE (pós-lote único, features CONGELADAS de 30/08/2026):
    o RESÍDUO de correlação com prop_fala que sobrevive ao mascaramento de
    padding. A hipótese ORIGINAL (medida na 1ª rodada, artefatos *_pre_lote.*)
    era outra: sem mascaramento, os frames de silêncio puro entravam na média e
    no desvio, e prop_fala voltava "pela porta dos fundos" codificada nas 44
    features. O mascaramento removeu essa via por construção; o que se pergunta
    agora é quanto de correlação RESTA — e se o que resta é conteúdo acústico
    genuíno ou atalho remanescente.

    A comparação antes/depois É a evidência: por isso os artefatos da 1ª rodada
    foram ARQUIVADOS com sufixo _pre_lote (não sobrescrever), e esta rodada
    escreve com sufixo _pos_lote.

CORRELAÇÃO ADICIONAL — n_frames_validos:
    coluna de diagnóstico nova do lote. Se o resíduo contra prop_fala fosse
    "quantidade de silêncio" disfarçada, a correlação contra n_frames_validos
    (a contagem direta de frames que entraram na agregação) deveria ser tão ou
    mais alta. Se ela vier MENOR, o resíduo é conteúdo acústico, não formatação.

CRUZAMENTO COM O MODELO:
    o top10 de importância lido é o do RF AJUSTADO (rf_tuned_principal.json,
    B3.3) — treinado nas MESMAS features congeladas que este script analisa.
    Cruzar estas correlações com o top10 do RF antigo (features pré-lote)
    misturaria dois mundos.

RECORTE: universo eval (148.176), o aprovado pelo orientador.

SAÍDAS:
    results/metricas/padding_corr_features_propfala_pos_lote.csv
    results/figuras/padding_corr_features_propfala_pos_lote.png
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

# Sufixo desta rodada. A 1ª rodada (features pré-mascaramento, hipótese
# original) está arquivada como *_pre_lote.* — é o lado "antes" da evidência.
SUFIXO = "_pos_lote"

# Fonte do top10 de importância: o RF ajustado do Bloco 3, treinado nas
# features congeladas (mesmo mundo desta análise).
JSON_MODELO = "rf_tuned_principal.json"


def main() -> None:
    # carregar_dados_split já entrega o universo eval (split novo) validado
    df = carregar_dados_split(RAIZ)
    cols = colunas_features(df)
    print(f"universo: {len(df)} áudios (eval) | {len(cols)} features | "
          f"saídas com sufixo '{SUFIXO}'")

    # ---- Correlação de cada feature com prop_fala e n_frames_validos --------
    pear = df[cols].corrwith(df["prop_fala"], method="pearson")
    spear = df[cols].corrwith(df["prop_fala"], method="spearman")
    pear_nfv = df[cols].corrwith(df["n_frames_validos"], method="pearson")

    res = pd.DataFrame({
        "feature": cols,
        "pearson": pear.round(4).values,
        "spearman": spear.round(4).values,
        "pearson_n_frames_validos": pear_nfv.round(4).values,
    })
    res["abs_pearson"] = res["pearson"].abs()
    res = res.sort_values("abs_pearson", ascending=False).reset_index(drop=True)

    # ---- Cruzamento com o top10 de importância do RF ajustado ---------------
    caminho_modelo = RAIZ / "results" / "metricas" / JSON_MODELO
    if caminho_modelo.exists():
        with open(caminho_modelo, encoding="utf-8") as f:
            em_top10 = set(json.load(f)["top10_features"].keys())
        origem_top10 = JSON_MODELO
    else:
        em_top10 = set()
        origem_top10 = f"{JSON_MODELO} NÃO ENCONTRADO — rode src.models.ajustar_rf antes"
        print(f"AVISO: {origem_top10}")
    res["no_top10_rf"] = res["feature"].isin(em_top10)

    dir_met = RAIZ / "results" / "metricas"
    dir_met.mkdir(parents=True, exist_ok=True)
    res.drop(columns="abs_pearson").to_csv(
        dir_met / f"padding_corr_features_propfala{SUFIXO}.csv", index=False)

    print(f"\n--- 44 features × prop_fala (e n_frames_validos), "
          f"ordenado por |pearson| --- [top10: {origem_top10}]")
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
    ax.set_title("Features × prop_fala — eval, features CONGELADAS (pós-lote)\n"
                 "(vermelho = feature no top10 de importância do RF ajustado)")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    dir_fig = RAIZ / "results" / "figuras"
    dir_fig.mkdir(parents=True, exist_ok=True)
    fig.savefig(dir_fig / f"padding_corr_features_propfala{SUFIXO}.png", dpi=150)
    plt.close(fig)

    # ---- Conclusão -----------------------------------------------------------
    fortes = res[res["abs_pearson"] > 0.3]
    top_rf_fortes = fortes[fortes["no_top10_rf"]]
    media_abs = res["abs_pearson"].mean()
    max_nfv = res["pearson_n_frames_validos"].abs().max()
    feat_max_nfv = res.loc[res["pearson_n_frames_validos"].abs().idxmax(),
                           "feature"]
    mais_forte = res.iloc[0]
    print("\n" + "=" * 70)
    print("CONCLUSÃO (resíduo pós-mascaramento)")
    print("=" * 70)
    print(f"""
1. Média de |r_pearson| contra prop_fala: {media_abs:.3f}
   (antes do mascaramento era 0,150 — ver *_pre_lote.csv; a comparação
   antes/depois é a evidência de que o mascaramento fechou a via do padding).
2. {len(fortes)} de {len(res)} features têm |r_pearson| > 0,3; a mais forte é
   {mais_forte['feature']} (r = {mais_forte['pearson']:+.4f}).
3. Dessas, {len(top_rf_fortes)} estão no top10 de importância do RF ajustado
   ({origem_top10}):
   {', '.join(top_rf_fortes['feature']) if len(top_rf_fortes) else '(nenhuma)'}
4. Contra n_frames_validos, o máximo é |r| = {max_nfv:.3f} ({feat_max_nfv}) —
   se MENOR que contra prop_fala, o resíduo correlaciona com a FRAÇÃO de fala
   do áudio original (conteúdo acústico), não com a quantidade de frames que
   entrou na agregação (formatação do tensor).

CSV : {dir_met / f'padding_corr_features_propfala{SUFIXO}.csv'}
PNG : {dir_fig / f'padding_corr_features_propfala{SUFIXO}.png'}
""")


if __name__ == "__main__":
    main()
