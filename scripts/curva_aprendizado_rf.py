"""
Curva de aprendizado do RF — evidência para a decisão da subamostra do SVM
==========================================================================

SÓ LÊ E REPORTA. Retreina o RF baseline ~6 vezes com tamanhos crescentes de
treino e avalia SEMPRE no mesmo split de validação. O teste continua lacrado.
Não altera features, não re-extrai nada, não implementa a subamostra do SVM.

OBJETIVO:
    O SVM-RBF é O(n²)–O(n³) e não roda nos ~127k de treino; a recomendação
    pendente (TODO 9.2 no config.yaml) é uma subamostra estratificada (~30k)
    compartilhada por RF, SVM e CNN. Esta curva responde à objeção da banca de
    que "o SVM viu menos dado": se o desempenho do RF SATURA antes de 127k,
    subamostrar não prejudica a comparação. Isto é EVIDÊNCIA para a decisão,
    não a decisão em si.

PROTOCOLO:
    Para cada tamanho em [5.000, 10.000, 20.000, 40.000, 80.000, TODOS]:
      - subamostra do split de TREINO estratificada por classe_binaria × codec
        (random_state=42); estratos com menos de 10 membros são colapsados em
        'outros' antes da amostragem (senão o split estratificado quebra);
      - treina o RF baseline (mesmos hiperparâmetros do treinar_rf.py);
      - avalia no split de VALIDAÇÃO fixo: f1_macro, EER, tempo de treino.

SAÍDAS:
    results/metricas/curva_aprendizado_rf.csv
    results/metricas/curva_aprendizado_rf.json  (seed, versões, hash do split)
    results/figuras/curva_aprendizado_rf.png
    + ponto de saturação no stdout

Rode a partir da raiz:  python -m scripts.curva_aprendizado_rf
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from src.utils.seeds import fixar_seeds
from src.data.split import carregar_dados_split, colunas_features
from src.models.treinar_rf import calcular_eer

RAIZ = Path(__file__).resolve().parents[1]
TAMANHOS = [5000, 10000, 20000, 40000, 80000, None]   # None = treino inteiro
MIN_ESTRATO = 10          # estratos menores que isto são colapsados em 'outros'
TOL_SATURACAO = 0.005     # satura no 1º tamanho a menos de 0,005 do f1 máximo


def subamostrar(treino: pd.DataFrame, n: int, semente: int) -> pd.DataFrame:
    """Subamostra estratificada por classe_binaria × codec do split de treino."""
    estrato = treino["classe_binaria"].astype(str) + "_" + treino["codec"]
    contagem = estrato.value_counts()
    pequenos = contagem[contagem < MIN_ESTRATO].index
    if len(pequenos):
        estrato = estrato.where(~estrato.isin(pequenos), "outros")
    sub, _ = train_test_split(
        treino, train_size=n, stratify=estrato,
        random_state=semente, shuffle=True,
    )
    return sub


def main() -> None:
    semente = fixar_seeds(42)

    df = carregar_dados_split(RAIZ)
    cols = colunas_features(df)
    labels = pd.read_csv(RAIZ / "data" / "processed" / "labels.csv",
                         usecols=["arquivo", "codec"])
    n_antes = len(df)
    df = df.merge(labels, on="arquivo", how="inner")
    assert len(df) == n_antes, "merge perdeu linhas — investigar"

    treino = df[df["conjunto"] == "treino"]
    validacao = df[df["conjunto"] == "validacao"]
    X_va, y_va = validacao[cols].values, validacao["classe_binaria"].values
    print(f"{len(cols)} features | treino {len(treino)} | validação {len(validacao)}")

    linhas = []
    for n in TAMANHOS:
        sub = treino if n is None else subamostrar(treino, n, semente)
        X_tr, y_tr = sub[cols].values, sub["classe_binaria"].values

        modelo = RandomForestClassifier(
            n_estimators=100, max_depth=None, class_weight="balanced",
            random_state=semente, n_jobs=-1,
        )
        t0 = time.perf_counter()
        modelo.fit(X_tr, y_tr)
        t_treino = time.perf_counter() - t0

        y_pred = modelo.predict(X_va)
        scores = modelo.predict_proba(X_va)[:, 1]
        f1m = f1_score(y_va, y_pred, average="macro", zero_division=0)
        eer, _ = calcular_eer(y_va, scores)

        linhas.append({
            "n_treino": len(sub),
            "f1_macro": round(f1m, 4),
            "eer": round(eer, 4),
            "tempo_treino_s": round(t_treino, 2),
        })
        print(f"n={len(sub):>6}: f1_macro={f1m:.4f}  EER={eer:.4f}  "
              f"treino={t_treino:.1f}s")

    res = pd.DataFrame(linhas)
    dir_met = RAIZ / "results" / "metricas"
    dir_met.mkdir(parents=True, exist_ok=True)
    res.to_csv(dir_met / "curva_aprendizado_rf.csv", index=False)

    # ---- Ponto de saturação ------------------------------------------------
    f1_max = res["f1_macro"].max()
    saturado = res[res["f1_macro"] >= f1_max - TOL_SATURACAO].iloc[0]

    # ---- Registro de reprodutibilidade (seção 10) --------------------------
    split_csv = RAIZ / "data" / "processed" / "split.csv"
    hash_split = hashlib.md5(split_csv.read_bytes()).hexdigest()
    registro = {
        "semente": semente,
        "versoes": {"numpy": np.__version__, "sklearn": sklearn.__version__,
                    "pandas": pd.__version__},
        "hash_md5_split_csv": hash_split,
        "estratificacao": "classe_binaria x codec",
        "tamanhos": [int(x) for x in res["n_treino"]],
        "resultados": res.to_dict(orient="records"),
        "f1_macro_maximo": float(f1_max),
        "satura_em_n": int(saturado["n_treino"]),
        "tolerancia_saturacao": TOL_SATURACAO,
    }
    with open(dir_met / "curva_aprendizado_rf.json", "w", encoding="utf-8") as f:
        json.dump(registro, f, indent=2, ensure_ascii=False)

    # ---- Figura ------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(res["n_treino"], res["f1_macro"], marker="o", color="#4c72b0",
             label="f1_macro")
    ax1.axvline(saturado["n_treino"], ls="--", color="gray", lw=1,
                label=f"saturação ≈ {int(saturado['n_treino'])}")
    ax1.set_xscale("log")
    ax1.set_xlabel("nº de amostras de treino (escala log)")
    ax1.set_ylabel("f1_macro (validação)", color="#4c72b0")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(res["n_treino"], res["eer"], marker="s", color="#c44e52",
             label="EER")
    ax2.set_ylabel("EER (validação)", color="#c44e52")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center right")
    ax1.set_title("Curva de aprendizado — RF baseline (validação fixa)")
    fig.tight_layout()
    dir_fig = RAIZ / "results" / "figuras"
    dir_fig.mkdir(parents=True, exist_ok=True)
    fig.savefig(dir_fig / "curva_aprendizado_rf.png", dpi=150)
    plt.close(fig)

    # ---- Conclusão ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("CONCLUSÃO")
    print("=" * 70)
    print(f"""
f1_macro máximo = {f1_max:.4f}. Dentro da tolerância de {TOL_SATURACAO}, a curva
SATURA em n = {int(saturado['n_treino'])} amostras de treino
(f1_macro = {saturado['f1_macro']:.4f}, EER = {saturado['eer']:.4f}).
Leitura para a decisão 9.2 (subamostra do SVM): se a saturação ocorre em
tamanho <= ~30k, treinar com a subamostra estratificada proposta não
prejudica o desempenho do RF — evidência de que a comparação RF × SVM × CNN
no mesmo subconjunto é justa. A decisão em si segue pendente (orientador).

CSV : {dir_met / 'curva_aprendizado_rf.csv'}
JSON: {dir_met / 'curva_aprendizado_rf.json'}
PNG : {dir_fig / 'curva_aprendizado_rf.png'}
""")


if __name__ == "__main__":
    main()
