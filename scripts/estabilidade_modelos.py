"""
Estabilidade (B3.6) — sementes no RF, bootstrap da validação em RF e SVM
========================================================================

AS DUAS FONTES DE VARIAÇÃO SÃO COISAS DIFERENTES — e este script mede as duas,
dizendo qual é qual:

  1. VARIÂNCIA DE TREINO — do algoritmo (bootstrap das árvores, subsample dos
     pesos). O RF tem; o SVM com probability=False NÃO TEM: o SVC é
     DETERMINÍSTICO (o random_state do SVC só controla o embaralhamento interno
     das estimativas de probabilidade — com probability=False é ignorado).
     Rodar "3 sementes" do SVM produziria três resultados idênticos, e
     apresentar isso como análise de estabilidade seria um erro fácil de
     detectar na banca. Medida aqui: 5 sementes (42–46) SÓ no RF, re-treinando
     o modelo com a MESMA configuração vencedora; o resultado principal
     continua sendo o de seed 42.

  2. VARIÂNCIA DE ESTIMATIVA — de a validação ser uma amostra finita. Os dois
     modelos têm. Medida aqui: bootstrap do conjunto de validação (1.000
     reamostragens com reposição das 22.226 linhas), aplicado IGUALMENTE a RF e
     SVM para que as duas barras sejam comparáveis. O limiar fica FIXO no valor
     selecionado pelo modelo (reamostra-se a avaliação, não a seleção).

REGRA DO ORIENTADOR (a leitura que importa): se a diferença entre RF e SVM for
menor que a dispersão medida, isso TEM de ser discutido no texto — não
escondido atrás de uma tabela de médias. O JSON final traz essa comparação
calculada.

Insumos: rf_random_search.json (config vencedora), rf_tuned_principal.joblib /
svm_tuned_principal.joblib (+ JSONs com os limiares selecionados).
Saída: results/metricas/estabilidade_rf_svm.json

Rode a partir da raiz (APÓS ajustar_rf e treinar_svm):
    python -m scripts.estabilidade_modelos
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from src.utils.config import carregar_config
from src.data.split import (carregar_dados_split, colunas_features,
                            filtrar_treino_braco)
from src.models.avaliacao import aplicar_limiar, calcular_eer, selecionar_limiar

RAIZ = Path(__file__).resolve().parents[1]

SEMENTES_RF = [42, 43, 44, 45, 46]
N_BOOTSTRAP = 1000
SEMENTE_BOOTSTRAP = 42


def _resumo(valores: list[float]) -> dict:
    v = np.asarray(valores, dtype=float)
    return {"media": round(float(v.mean()), 4),
            "desvio": round(float(v.std(ddof=1)), 4),
            "min": round(float(v.min()), 4),
            "max": round(float(v.max()), 4)}


def sementes_rf(cfg, params, X_tr, y_tr, X_va, y_va) -> dict:
    """Variância de TREINO do RF: 5 re-treinos, mesma config, sementes 42–46.
    O limiar é re-selecionado na validação em cada semente (o protocolo inteiro
    é repetido — semente nova, decisão nova)."""
    execucoes = []
    for s in SEMENTES_RF:
        t0 = time.perf_counter()
        modelo = RandomForestClassifier(**params, random_state=s, n_jobs=-1)
        modelo.fit(X_tr, y_tr)
        scores = modelo.predict_proba(X_va)[:, 1]
        sel = selecionar_limiar(y_va, scores)
        eer, _ = calcular_eer(y_va, scores)
        execucoes.append({"semente": s,
                          "limiar": round(sel["limiar"], 4),
                          "f1_macro": round(sel["f1_macro"], 4),
                          "eer": round(eer, 4)})
        print(f"  seed {s}: f1_macro {sel['f1_macro']:.4f} | eer {eer:.4f} | "
              f"limiar {sel['limiar']:.4f} ({time.perf_counter()-t0:.0f}s)")
    return {
        "o_que_mede": "variância de TREINO (bootstrap das árvores) — o SVM não tem esta fonte",
        "config": "mesma configuração vencedora de rf_random_search.json; só a semente muda",
        "execucoes": execucoes,
        "f1_macro": _resumo([e["f1_macro"] for e in execucoes]),
        "eer": _resumo([e["eer"] for e in execucoes]),
        "limiares": [e["limiar"] for e in execucoes],
    }


def bootstrap_validacao(y_va, scores, limiar, rng) -> dict:
    """Variância de ESTIMATIVA: IC por bootstrap da validação, limiar fixo."""
    n = len(y_va)
    f1s, eers = [], []
    for _ in range(N_BOOTSTRAP):
        i = rng.integers(0, n, size=n)
        y_b, s_b = y_va[i], scores[i]
        f1s.append(f1_score(y_b, aplicar_limiar(s_b, limiar), average="macro",
                            zero_division=0))
        eers.append(calcular_eer(y_b, s_b)[0])
    def ic(v):
        v = np.asarray(v)
        return {"media": round(float(v.mean()), 4),
                "desvio": round(float(v.std(ddof=1)), 4),
                "ic95": [round(float(np.percentile(v, 2.5)), 4),
                         round(float(np.percentile(v, 97.5)), 4)]}
    return {"n_reamostragens": N_BOOTSTRAP, "limiar_fixo": round(float(limiar), 4),
            "f1_macro": ic(f1s), "eer": ic(eers)}


def main() -> None:
    cfg = carregar_config(RAIZ)
    dir_met = RAIZ / "results" / "metricas"

    with open(dir_met / "rf_random_search.json", encoding="utf-8") as f:
        params = json.load(f)["melhor"]["params"]
    with open(dir_met / "rf_tuned_principal.json", encoding="utf-8") as f:
        rf_json = json.load(f)
    with open(dir_met / "svm_tuned_principal.json", encoding="utf-8") as f:
        svm_json = json.load(f)

    df = carregar_dados_split(RAIZ)
    cols = colunas_features(df)
    treino = filtrar_treino_braco(df[df["conjunto"] == "treino"], "principal",
                                  cfg, RAIZ)
    validacao = df[df["conjunto"] == "validacao"]
    X_tr, y_tr = treino[cols].values, treino["classe_binaria"].values
    X_va, y_va = validacao[cols].values, validacao["classe_binaria"].values

    print("=== RF: 5 sementes (variância de treino) ===")
    res_sementes = sementes_rf(cfg, params, X_tr, y_tr, X_va, y_va)

    print("\n=== Bootstrap da validação (variância de estimativa, RF e SVM) ===")
    rng = np.random.default_rng(SEMENTE_BOOTSTRAP)
    rf = joblib.load(RAIZ / "models" / "rf_tuned_principal.joblib")
    scores_rf = rf.predict_proba(X_va)[:, 1]
    boot_rf = bootstrap_validacao(y_va, scores_rf,
                                  rf_json["selecao_limiar"]["limiar"], rng)
    print(f"  RF : f1_macro {boot_rf['f1_macro']['media']:.4f} "
          f"IC95 {boot_rf['f1_macro']['ic95']} | eer {boot_rf['eer']['media']:.4f}")

    svm = joblib.load(RAIZ / "models" / "svm_tuned_principal.joblib")
    scores_svm = svm.decision_function(X_va)
    boot_svm = bootstrap_validacao(y_va, scores_svm,
                                   svm_json["selecao_limiar"]["limiar"], rng)
    print(f"  SVM: f1_macro {boot_svm['f1_macro']['media']:.4f} "
          f"IC95 {boot_svm['f1_macro']['ic95']} | eer {boot_svm['eer']['media']:.4f}")

    # ---- A leitura crítica exigida pelo orientador ---------------------------
    dif_f1 = abs(rf_json["f1_macro"] - svm_json["f1_macro"])
    dif_eer = abs(rf_json["eer"] - svm_json["eer"])
    disp_f1 = max(boot_rf["f1_macro"]["desvio"], boot_svm["f1_macro"]["desvio"],
                  res_sementes["f1_macro"]["desvio"])
    disp_eer = max(boot_rf["eer"]["desvio"], boot_svm["eer"]["desvio"],
                   res_sementes["eer"]["desvio"])
    leitura = {
        "diferenca_f1_macro_rf_svm": round(dif_f1, 4),
        "maior_dispersao_f1_macro": round(disp_f1, 4),
        "diferenca_maior_que_dispersao_f1": bool(dif_f1 > disp_f1),
        "diferenca_eer_rf_svm": round(dif_eer, 4),
        "maior_dispersao_eer": round(disp_eer, 4),
        "diferenca_maior_que_dispersao_eer": bool(dif_eer > disp_eer),
        "regra": ("se a diferença entre RF e SVM for menor que a dispersão "
                  "medida, isso é discutido no texto — não escondido atrás de "
                  "uma tabela de médias"),
    }
    print(f"\n  |Δf1_macro| RF−SVM = {dif_f1:.4f} vs dispersão {disp_f1:.4f} "
          f"-> {'MAIOR' if dif_f1 > disp_f1 else 'MENOR — discutir no texto'}")
    print(f"  |ΔEER|     RF−SVM = {dif_eer:.4f} vs dispersão {disp_eer:.4f} "
          f"-> {'MAIOR' if dif_eer > disp_eer else 'MENOR — discutir no texto'}")

    saida = {
        "modelo": "estabilidade_rf_svm",
        "conjunto": "validacao (o teste continua lacrado)",
        "svm_deterministico": (
            "SVC com probability=False é determinístico: não há variância de "
            "treino a medir; random_state do SVC é ignorado sem Platt scaling"),
        "rf_sementes": res_sementes,
        "bootstrap_validacao_rf": boot_rf,
        "bootstrap_validacao_svm": boot_svm,
        "semente_bootstrap": SEMENTE_BOOTSTRAP,
        "leitura_critica": leitura,
    }
    with open(dir_met / "estabilidade_rf_svm.json", "w", encoding="utf-8") as f:
        json.dump(saida, f, indent=2, ensure_ascii=False)
    print(f"\nSalvo em results/metricas/estabilidade_rf_svm.json")


if __name__ == "__main__":
    main()
