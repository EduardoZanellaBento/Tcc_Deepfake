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

  3. BOOTSTRAP PAREADO DA DIFERENÇA (a comparação certa): os dois modelos são
     avaliados EXATAMENTE nas mesmas 22.226 amostras, então os erros deles são
     correlacionados — e essa correlação é informação que os dois bootstraps
     individuais jogam fora. Comparar o Δ observado contra "a maior dispersão
     individual" é heurística, não teste. Aqui, a cada reamostragem, UM ÚNICO
     vetor de índices é sorteado e aplicado aos scores dos DOIS modelos; o que
     se guarda é a DIFERENÇA (SVM − RF) de f1_macro e de EER. O IC95 da
     diferença é a grandeza sobre a qual a conclusão é feita. Os bootstraps
     individuais são MANTIDOS (o IC de cada modelo é útil no texto).

REGRA DO ORIENTADOR (a leitura que importa): se a diferença entre RF e SVM for
menor que a dispersão medida, isso TEM de ser discutido no texto — não
escondido atrás de uma tabela de médias. O JSON final traz essa comparação
calculada — agora respondida pelo estatístico certo (o IC da diferença
pareada), com a heurística anterior mantida ao lado.

Insumos: rf_random_search.json (config vencedora), rf_tuned_principal.joblib /
svm_tuned_principal.joblib (+ JSONs com os limiares selecionados).
Saída: results/metricas/estabilidade_rf_svm.json

Rode a partir da raiz (APÓS ajustar_rf e treinar_svm):
    python -m scripts.estabilidade_modelos
"""

import hashlib
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


def bootstrap_pareado(y_va, scores_rf, scores_svm,
                      limiar_rf, limiar_svm) -> dict:
    """Bootstrap PAREADO da diferença SVM − RF: um único vetor de índices por
    reamostragem, aplicado aos scores dos dois modelos. Os erros dos dois
    modelos nas mesmas 22.226 linhas são correlacionados, e essa correlação
    ENCOLHE a variância da diferença — jogá-la fora (bootstraps independentes)
    superestima o ruído da comparação. RNG próprio, semeado de novo, para não
    perturbar a sequência dos bootstraps individuais (reprodutibilidade dos
    números já publicados)."""
    rng = np.random.default_rng(SEMENTE_BOOTSTRAP)
    n = len(y_va)
    d_f1, d_eer = [], []
    for _ in range(N_BOOTSTRAP):
        i = rng.integers(0, n, size=n)          # UM sorteio, DOIS modelos
        y_b = y_va[i]
        f1_rf = f1_score(y_b, aplicar_limiar(scores_rf[i], limiar_rf),
                         average="macro", zero_division=0)
        f1_svm = f1_score(y_b, aplicar_limiar(scores_svm[i], limiar_svm),
                          average="macro", zero_division=0)
        d_f1.append(f1_svm - f1_rf)
        d_eer.append(calcular_eer(y_b, scores_svm[i])[0]
                     - calcular_eer(y_b, scores_rf[i])[0])

    def ic(v):
        v = np.asarray(v)
        return {"media": round(float(v.mean()), 4),
                "desvio": round(float(v.std(ddof=1)), 4),
                "ic95": [round(float(np.percentile(v, 2.5)), 4),
                         round(float(np.percentile(v, 97.5)), 4)]}

    d_f1, d_eer = np.asarray(d_f1), np.asarray(d_eer)
    ic_f1, ic_eer = ic(d_f1), ic(d_eer)
    # SVM melhor: f1 MAIOR (delta > 0), EER MENOR (delta < 0)
    zero_fora = (0.0 < ic_f1["ic95"][0] or 0.0 > ic_f1["ic95"][1]) and \
                (0.0 < ic_eer["ic95"][0] or 0.0 > ic_eer["ic95"][1])
    return {
        "n_reamostragens": N_BOOTSTRAP,
        "semente": SEMENTE_BOOTSTRAP,
        "limiares_fixos": {"rf": round(float(limiar_rf), 4),
                           "svm": round(float(limiar_svm), 4)},
        "delta_f1_macro": ic_f1,
        "delta_eer": ic_eer,
        "fracao_reamostragens_svm_melhor_f1":
            round(float((d_f1 > 0).mean()), 4),
        "fracao_reamostragens_svm_melhor_eer":
            round(float((d_eer < 0).mean()), 4),
        "leitura": ("IC95 da diferença não contém zero => a vantagem do SVM "
                    "não é ruído de amostragem" if zero_fora else
                    "IC95 da diferença contém zero => a vantagem do SVM pode "
                    "ser ruído de amostragem — discutir no texto"),
    }


def _conferir_hashes_congelados(raiz: Path, *jsons: dict) -> None:
    """Os .joblib são comparados sobre o features.csv/split.csv EM DISCO; os
    JSONs registram os hashes dos artefatos com que os modelos foram treinados.
    Se divergirem, os scores comparados não seriam os dos modelos salvos."""
    h_feat = hashlib.md5(
        (raiz / "data" / "features" / "features.csv").read_bytes()).hexdigest()
    h_split = hashlib.md5(
        (raiz / "data" / "processed" / "split.csv").read_bytes()).hexdigest()
    for j in jsons:
        if (j["hash_md5_features_csv"] != h_feat
                or j["hash_md5_split_csv"] != h_split):
            raise RuntimeError(
                f"Hashes de features.csv/split.csv em disco divergem dos "
                f"registrados em {j['modelo']}.json — os artefatos congelados "
                "mudaram desde o treino; a comparação seria inválida.")


def main() -> None:
    cfg = carregar_config(RAIZ)
    dir_met = RAIZ / "results" / "metricas"

    with open(dir_met / "rf_random_search.json", encoding="utf-8") as f:
        params = json.load(f)["melhor"]["params"]
    with open(dir_met / "rf_tuned_principal.json", encoding="utf-8") as f:
        rf_json = json.load(f)
    with open(dir_met / "svm_tuned_principal.json", encoding="utf-8") as f:
        svm_json = json.load(f)
    _conferir_hashes_congelados(RAIZ, rf_json, svm_json)

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

    print("\n=== Bootstrap PAREADO da diferença SVM - RF (mesmos índices) ===")
    pareado = bootstrap_pareado(y_va, scores_rf, scores_svm,
                                rf_json["selecao_limiar"]["limiar"],
                                svm_json["selecao_limiar"]["limiar"])
    print(f"  Δf1_macro (SVM-RF): {pareado['delta_f1_macro']['media']:+.4f} "
          f"IC95 {pareado['delta_f1_macro']['ic95']}")
    print(f"  ΔEER      (SVM-RF): {pareado['delta_eer']['media']:+.4f} "
          f"IC95 {pareado['delta_eer']['ic95']}")
    print(f"  {pareado['leitura']}")

    # ---- A leitura crítica exigida pelo orientador ---------------------------
    # A regra continua sendo respondida ("diferença × dispersão"), mas a
    # evidência principal passa a ser o IC95 da DIFERENÇA pareada — o
    # estatístico certo; a comparação contra a maior dispersão individual fica
    # ao lado como heurística de leitura rápida.
    dif_f1 = abs(rf_json["f1_macro"] - svm_json["f1_macro"])
    dif_eer = abs(rf_json["eer"] - svm_json["eer"])
    disp_f1 = max(boot_rf["f1_macro"]["desvio"], boot_svm["f1_macro"]["desvio"],
                  res_sementes["f1_macro"]["desvio"])
    disp_eer = max(boot_rf["eer"]["desvio"], boot_svm["eer"]["desvio"],
                   res_sementes["eer"]["desvio"])
    leitura = {
        "evidencia_principal": (
            "bootstrap PAREADO (bootstrap_pareado_svm_menos_rf): "
            f"IC95 do Δf1_macro {pareado['delta_f1_macro']['ic95']} e do "
            f"ΔEER {pareado['delta_eer']['ic95']} — "
            + ("nenhum contém zero: a vantagem do SVM não é ruído de amostragem"
               if "não é ruído" in pareado["leitura"] else
               "contém zero: discutir no texto")),
        "diferenca_f1_macro_rf_svm": round(dif_f1, 4),
        "maior_dispersao_f1_macro": round(disp_f1, 4),
        "diferenca_maior_que_dispersao_f1": bool(dif_f1 > disp_f1),
        "diferenca_eer_rf_svm": round(dif_eer, 4),
        "maior_dispersao_eer": round(disp_eer, 4),
        "diferenca_maior_que_dispersao_eer": bool(dif_eer > disp_eer),
        "regra": ("se a diferença entre RF e SVM for menor que a dispersão "
                  "medida, isso é discutido no texto — não escondido atrás de "
                  "uma tabela de médias"),
        "nota_metodo": ("a comparação contra a maior dispersão individual é "
                        "heurística: ignora a correlação entre os erros dos "
                        "dois modelos nas mesmas 22.226 amostras; o teste "
                        "correto é o IC da diferença pareada, acima"),
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
        "bootstrap_pareado_svm_menos_rf": pareado,
        "semente_bootstrap": SEMENTE_BOOTSTRAP,
        "leitura_critica": leitura,
    }
    with open(dir_met / "estabilidade_rf_svm.json", "w", encoding="utf-8") as f:
        json.dump(saida, f, indent=2, ensure_ascii=False)
    print(f"\nSalvo em results/metricas/estabilidade_rf_svm.json")


if __name__ == "__main__":
    main()
