"""
Importância por PERMUTAÇÃO no RF ajustado (R6.3)
=================================================

POR QUE ISTO EXISTE — o próprio código já avisava:
    `src/models/treinar_rf.py:181-184` registra que a importância do Random
    Forest é por REDUÇÃO DE IMPUREZA, e que ela é enviesada a favor de features
    contínuas de alta cardinalidade: "é uma pista sobre 'o que o modelo usou',
    não prova de causalidade acústica. Para uma afirmação mais forte, usar
    `permutation_importance`."

    O problema é que o `top10_features` gravado nos JSONs — e que vai para o
    texto do Capítulo 4 — é exatamente a métrica enviesada. Este script mede a
    alternativa, e `ablacao_mfcc1_std.json -> leituras_pre_registradas.c` já
    previa esse uso como confirmação.

A DIFERENÇA ENTRE AS DUAS MÉTRICAS (é o que torna a comparação informativa):
    - IMPUREZA: calculada DURANTE o treino, olhando quantas divisões usaram cada
      feature e quanto cada divisão reduziu a impureza. Enviesada: uma feature
      contínua com muitos valores distintos oferece mais pontos de corte, logo
      tende a ser escolhida mais vezes mesmo sem ser mais informativa.
    - PERMUTAÇÃO: calculada DEPOIS do treino, no conjunto de VALIDAÇÃO,
      embaralhando uma coluna por vez e medindo quanto a métrica piora. Mede o
      que o modelo REALMENTE perde sem aquela informação, sem privilegiar
      cardinalidade.

    Se os dois rankings concordarem, o `top10_features` já publicado ganha
    respaldo. Se discordarem muito, isso é ACHADO — e significa que a leitura
    acústica feita a partir do ranking por impureza precisa ser corrigida.

DECISÕES DE MEDIÇÃO:
    - na VALIDAÇÃO, nunca no treino (no treino a permutação mede memorização) e
      nunca no teste, que segue lacrado;
    - `scoring='roc_auc'`: INDEPENDENTE DE LIMIAR, pela mesma razão que a busca
      de hiperparâmetros usa EER — medir a queda de f1 num limiar fixo mistura
      "a feature era importante" com "o limiar deixou de ser o ótimo";
    - `n_repeats=10`, semente 42: a permutação é estocástica, e o desvio entre as
      10 repetições sai no JSON — uma importância cujo desvio cobre o zero não
      sustenta afirmação nenhuma.

SAÍDAS:
    results/metricas/importancia_permutacao_rf.json
    results/figuras/importancia_permutacao_rf.png

Rode a partir da raiz (após ajustar_rf):
    python -m scripts.importancia_permutacao_rf
"""

import json
import platform
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance

from src.utils.config import carregar_config
from src.utils.serializacao import json_seguro
from src.utils.seeds import fixar_seeds
from src.data.split import carregar_dados_split, colunas_features
from src.models.modelos_ajustados import (
    carregar_modelo_ajustado, hashes_congelados,
)

RAIZ = Path(__file__).resolve().parents[1]
N_REPEATS = 10
SCORING = "roc_auc"          # independente de limiar — ver docstring
TOP_N = 10                   # o mesmo recorte do top10_features publicado


def main() -> None:
    cfg = carregar_config(RAIZ)
    semente = fixar_seeds(cfg["semente"])

    df = carregar_dados_split(RAIZ)
    cols = colunas_features(df)
    validacao = df[df["conjunto"] == "validacao"]      # NUNCA 'teste'
    X_va = validacao[cols].values
    y_va = validacao["classe_binaria"].values

    carregado = carregar_modelo_ajustado(RAIZ, "rf")
    modelo = carregado["modelo"]
    print(f"{carregado['rotulo']} | validação {X_va.shape} | "
          f"{N_REPEATS} repetições | scoring={SCORING}")

    t0 = time.perf_counter()
    r = permutation_importance(modelo, X_va, y_va, scoring=SCORING,
                               n_repeats=N_REPEATS, random_state=semente,
                               n_jobs=-1)
    t = time.perf_counter() - t0
    print(f"permutation_importance em {t:.1f}s")

    # ---- Os dois rankings, lado a lado --------------------------------------
    perm = pd.Series(r.importances_mean, index=cols)
    perm_std = pd.Series(r.importances_std, index=cols)
    impureza = pd.Series(modelo.feature_importances_, index=cols)

    tabela = pd.DataFrame({
        "importancia_permutacao": perm.round(6),
        "desvio_permutacao": perm_std.round(6),
        "importancia_impureza": impureza.round(6),
    })
    tabela["posicao_permutacao"] = tabela["importancia_permutacao"] \
        .rank(ascending=False, method="min").astype(int)
    tabela["posicao_impureza"] = tabela["importancia_impureza"] \
        .rank(ascending=False, method="min").astype(int)
    tabela["delta_posicao"] = (tabela["posicao_impureza"]
                               - tabela["posicao_permutacao"])
    # Uma importância cujo intervalo media ± desvio cobre o zero não sustenta
    # afirmação: a feature pode não estar contribuindo nada.
    tabela["significativa"] = (tabela["importancia_permutacao"]
                               - tabela["desvio_permutacao"]) > 0
    tabela = tabela.sort_values("importancia_permutacao", ascending=False)

    print("\n--- top 15 por permutação ---")
    print(tabela.head(15).to_string())

    rho, p_valor = spearmanr(tabela["importancia_permutacao"],
                             tabela["importancia_impureza"])

    top_perm = list(tabela.index[:TOP_N])
    top_imp = list(impureza.sort_values(ascending=False).index[:TOP_N])
    intersecao = [f for f in top_perm if f in top_imp]
    so_permutacao = [f for f in top_perm if f not in top_imp]
    so_impureza = [f for f in top_imp if f not in top_perm]
    maior_queda = tabela.nsmallest(3, "delta_posicao")
    maior_subida = tabela.nlargest(3, "delta_posicao")

    print(f"\nSpearman entre os dois rankings: rho={rho:.4f} (p={p_valor:.2e})")
    print(f"top{TOP_N}: {len(intersecao)}/{TOP_N} em comum")
    print(f"  só no top{TOP_N} por permutação: {so_permutacao}")
    print(f"  só no top{TOP_N} por impureza  : {so_impureza}")

    # ---- Figura: os dois rankings lado a lado -------------------------------
    _plotar(tabela, carregado["rotulo"])

    # ---- Leitura crítica ----------------------------------------------------
    n_nao_signif = int((~tabela["significativa"]).sum())
    if rho >= 0.8 and len(intersecao) >= TOP_N - 2:
        veredito = (
            f"OS DOIS RANKINGS CONCORDAM (Spearman rho = {rho:.4f}; "
            f"{len(intersecao)} de {TOP_N} features em comum no topo). O "
            "`top10_features` já publicado nos JSONs — medido por redução de "
            "impureza, e portanto enviesado a favor de features contínuas de "
            "alta cardinalidade — é CONFIRMADO por uma métrica que não tem esse "
            "viés e que é medida na validação. A leitura acústica feita a partir "
            "dele pode ser mantida no texto, agora com respaldo; continua sendo "
            "'o que o modelo usou', não causalidade acústica.")
    elif rho >= 0.5:
        veredito = (
            f"CONCORDÂNCIA PARCIAL (Spearman rho = {rho:.4f}; "
            f"{len(intersecao)} de {TOP_N} em comum). O ranking por impureza "
            "acerta o grosso, mas há trocas relevantes de posição — as features "
            f"{so_impureza} aparecem no topo por impureza e NÃO no topo por "
            f"permutação, e {so_permutacao} fazem o inverso. No texto, citar o "
            "ranking por permutação como o principal e mencionar a divergência: "
            "é ela que mostra o viés de cardinalidade agindo.")
    else:
        veredito = (
            f"OS RANKINGS DISCORDAM (Spearman rho = {rho:.4f}; apenas "
            f"{len(intersecao)} de {TOP_N} em comum). ISTO É ACHADO: o "
            "`top10_features` por impureza publicado nos JSONs NÃO descreve o "
            "que o modelo de fato perde quando cada feature é embaralhada. A "
            "leitura acústica baseada nele precisa ser CORRIGIDA no texto, e o "
            "ranking por permutação passa a ser o citado — exatamente o cenário "
            "que o alerta de treinar_rf.py:181-184 antecipava.")
    if n_nao_signif:
        veredito += (f" Ressalva adicional: {n_nao_signif} das {len(cols)} "
                     "features têm importância por permutação cuja média menos "
                     "um desvio não passa de zero — sobre essas não se deve "
                     "afirmar contribuição alguma.")
    print("\n" + "=" * 74)
    print("LEITURA CRÍTICA")
    print("=" * 74)
    print(veredito + "\n")

    registro = {
        "analise": "importancia_permutacao_rf",
        "data": date.today().isoformat(),
        "modelo": carregado["nome_arquivo"],
        "conjunto": "validacao",
        "teste_lacrado": True,
        "metodo": {
            "funcao": "sklearn.inspection.permutation_importance",
            "n_repeats": N_REPEATS,
            "scoring": SCORING,
            "por_que_este_scoring": (
                "AUC é independente de limiar: medir a queda de f1 num limiar "
                "fixo misturaria 'a feature era importante' com 'o limiar "
                "deixou de ser o ótimo'"),
            "semente": semente,
            "tempo_s": round(t, 1),
        },
        "por_que_esta_analise_existe": (
            "a importância do RF publicada nos JSONs (top10_features) é por "
            "REDUÇÃO DE IMPUREZA, enviesada a favor de features contínuas de "
            "alta cardinalidade — alerta registrado em treinar_rf.py:181-184"),
        "spearman_entre_rankings": {"rho": round(float(rho), 4),
                                    "p_valor": float(p_valor)},
        "top_n": TOP_N,
        f"top{TOP_N}_permutacao": top_perm,
        f"top{TOP_N}_impureza": top_imp,
        "intersecao_topo": intersecao,
        "so_no_topo_por_permutacao": so_permutacao,
        "so_no_topo_por_impureza": so_impureza,
        "maiores_quedas_de_posicao": maior_queda["delta_posicao"].to_dict(),
        "maiores_subidas_de_posicao": maior_subida["delta_posicao"].to_dict(),
        "n_features_nao_significativas": n_nao_signif,
        "tabela_completa": tabela.reset_index(names="feature").to_dict("records"),
        "leitura_critica": veredito,
        "hashes_md5": hashes_congelados(
            RAIZ, cfg["experimento"]["caminho_subamostra"]),
        "ambiente": {"python": platform.python_version(),
                     "sistema": f"{platform.system()} {platform.release()}"},
    }
    dir_met = RAIZ / "results" / "metricas"
    caminho = dir_met / "importancia_permutacao_rf.json"
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(registro, f, indent=2, ensure_ascii=False,
                  default=json_seguro)
    print(f"Salvo em {caminho.relative_to(RAIZ)}")


def _plotar(tabela: pd.DataFrame, rotulo: str) -> None:
    topo = tabela.head(15)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    y = np.arange(len(topo))
    axes[0].barh(y, topo["importancia_permutacao"],
                 xerr=topo["desvio_permutacao"], color="#4c72b0")
    axes[0].set_yticks(y, labels=topo.index)
    axes[0].invert_yaxis()
    axes[0].set_xlabel(f"queda de {SCORING} ao embaralhar (média ± desvio)")
    axes[0].set_title("importância por PERMUTAÇÃO (validação)")
    axes[0].axvline(0, color="gray", lw=1)
    axes[0].grid(axis="x", alpha=0.3)

    ordem_imp = tabela.sort_values("importancia_impureza", ascending=False).head(15)
    y2 = np.arange(len(ordem_imp))
    axes[1].barh(y2, ordem_imp["importancia_impureza"], color="#c44e52")
    axes[1].set_yticks(y2, labels=ordem_imp.index)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("redução de impureza (enviesada por cardinalidade)")
    axes[1].set_title("importância por IMPUREZA (a publicada nos JSONs)")
    axes[1].grid(axis="x", alpha=0.3)

    fig.suptitle(f"{rotulo} — dois rankings de importância, mesmas 44 features")
    fig.tight_layout()
    dir_fig = RAIZ / "results" / "figuras"
    dir_fig.mkdir(parents=True, exist_ok=True)
    fig.savefig(dir_fig / "importancia_permutacao_rf.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
