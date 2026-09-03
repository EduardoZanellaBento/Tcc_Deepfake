"""
SVM (RBF) — braço principal: Random Search + treino final + limiar + tempos (B3.5)
==================================================================================

DECISÕES DE PROJETO:

1. StandardScaler DENTRO do Pipeline — obrigatório. Ajustado por fold, ele nunca
   enxerga média/desvio da validação. Padronizar o CSV inteiro antes seria
   vazamento — é a razão pela qual as features são salvas cruas (docstring de
   src/features/extrair_features.py). O RF não precisa de scaler (invariante a
   escala monotônica); o SVM depende de distância euclidiana e precisa.

2. decision_function, NÃO probability=True. O Platt scaling (probability=True)
   roda uma validação cruzada interna de 5 folds: encarece muito em 30k,
   introduz mais uma fonte de variância e não compra nada aqui — EER e seleção
   de limiar precisam apenas de um score MONOTÔNICO, não de probabilidade
   calibrada. CONSEQUÊNCIA DOCUMENTADA: o limiar do SVM vive numa escala
   diferente da do RF (real, centrado em zero, não em [0, 1]). Isso NÃO quebra
   o protocolo — a regra "selecionar na validação, aplicar no teste" é
   agnóstica de escala, e selecionar_limiar usa np.unique(scores), sem supor
   [0, 1] (decisão de projeto 2 de src/models/avaliacao.py).

3. SÓ braço principal (regra de negócio desde o config.yaml): SVM-RBF é
   O(n²)–O(n³) e não roda nos 103.723 do braço de referência — é exatamente a
   motivação da subamostra.

4. CUIDADO PRÁTICO (cronograma): antes de disparar o Random Search, UM fit em
   tamanho de fold é cronometrado e o custo total projetado
   (n_iter × 5 folds ÷ paralelismo). Se a projeção estourar o orçamento,
   reduz-se n_iter (registrado no JSON) — NUNCA a subamostra, que é o "mesmo
   ambiente experimental" sustentando a comparação RF × SVM × CNN.

5. Busca por EER (independente de limiar), mesma justificativa do RF — ver
   docstring de src/models/ajustar_rf.py. Scorer via decision_function.

Artefatos:
    results/metricas/svm_random_search.json + svm_random_search_cv.csv
    models/svm_tuned_principal.joblib
    results/metricas/svm_tuned_principal.json
    results/figuras/matriz_confusao_svm_tuned_principal.png

Rode a partir da raiz:  python -m src.models.treinar_svm
"""

import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.metrics import confusion_matrix, make_scorer, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ..utils.seeds import fixar_seeds
from ..data.split import (carregar_dados_split, colunas_features,
                          filtrar_treino_braco)
from .avaliacao import avaliar, plotar_matriz_confusao, selecionar_limiar
from .tempo import ambiente, medir_tempos

# Orçamento de parede para a BUSCA (regra 4 da docstring). O Bloco 3 tem dias,
# não semanas; 6h de busca é o teto antes de reduzir n_iter.
ORCAMENTO_BUSCA_HORAS = 6.0
# Paralelismo efetivo estimado para a projeção: nº de núcleos FÍSICOS (6 no
# Ryzen 5 7600) — estimativa conservadora; os 12 threads lógicos ajudam menos
# em carga numérica densa.
PARALELISMO_EFETIVO = 6


def _eer_df(y_true, y_score) -> float:
    """EER a partir do decision_function (1D, real, centrado em zero)."""
    from .avaliacao import calcular_eer
    return calcular_eer(y_true, y_score)[0]


SCORER_EER_DF = make_scorer(_eer_df, response_method="decision_function",
                            greater_is_better=False)


def montar_pipeline(semente: int) -> Pipeline:
    """Pipeline scaler+SVC. random_state é irrelevante com probability=False
    (só controla o embaralhamento do Platt scaling), mas fica explícito."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", cache_size=1000, random_state=semente)),
    ])


def espaco_busca() -> list[dict]:
    """Dois sub-espaços com 50/50 de probabilidade: gamma='scale' (heurística
    do sklearn) e gamma contínuo em loguniform(1e-4, 1e-1)."""
    base = {
        "svc__C": loguniform(0.1, 100),
        "svc__class_weight": ["balanced", None],
    }
    return [
        {**base, "svc__gamma": ["scale"]},
        {**base, "svc__gamma": loguniform(1e-4, 1e-1)},
    ]


def cronometrar_um_fit(X_tr: np.ndarray, y_tr: np.ndarray, semente: int,
                       kfold: StratifiedKFold) -> float:
    """Um fit em tamanho de FOLD (4/5 do treino), como a busca fará."""
    i_tr, _ = next(kfold.split(X_tr, y_tr))
    pipe = montar_pipeline(semente)
    t0 = time.perf_counter()
    pipe.fit(X_tr[i_tr], y_tr[i_tr])
    t_fit = time.perf_counter() - t0
    print(f"1 fit em {len(i_tr)} amostras (tamanho de fold): {t_fit:.1f}s")
    return t_fit


def buscar_e_treinar(cfg: dict, raiz: Path) -> dict:
    semente = fixar_seeds(cfg["semente"])
    nome = "svm_tuned_principal"

    df = carregar_dados_split(raiz)
    cols = colunas_features(df)
    treino = filtrar_treino_braco(df[df["conjunto"] == "treino"], "principal",
                                  cfg, raiz)
    validacao = df[df["conjunto"] == "validacao"]
    X_tr, y_tr = treino[cols].values, treino["classe_binaria"].values
    X_va, y_va = validacao[cols].values, validacao["classe_binaria"].values
    print(f"SVM braço principal: treino {X_tr.shape} | validação {X_va.shape}")

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=semente)

    # ---- Regra 4: projetar o custo ANTES de disparar a busca ----------------
    t_fit = cronometrar_um_fit(X_tr, y_tr, semente, kfold)
    n_iter_cfg = int(cfg["tuning"]["n_iter"])
    horas_projetadas = (t_fit * n_iter_cfg * 5) / PARALELISMO_EFETIVO / 3600
    n_iter = n_iter_cfg
    if horas_projetadas > ORCAMENTO_BUSCA_HORAS:
        n_iter = max(5, int(ORCAMENTO_BUSCA_HORAS * 3600
                            * PARALELISMO_EFETIVO / (t_fit * 5)))
        print(f"PROJEÇÃO {horas_projetadas:.1f}h > orçamento "
              f"{ORCAMENTO_BUSCA_HORAS:.0f}h -> n_iter reduzido "
              f"{n_iter_cfg} -> {n_iter} (registrado no JSON). A subamostra "
              f"NÃO é reduzida.")
    else:
        print(f"Projeção da busca: {horas_projetadas:.1f}h "
              f"(n_iter={n_iter} × 5 folds ÷ {PARALELISMO_EFETIVO}) — dentro "
              f"do orçamento de {ORCAMENTO_BUSCA_HORAS:.0f}h.")

    # ---- Random Search (5-fold no treino do braço principal) ----------------
    busca = RandomizedSearchCV(
        montar_pipeline(semente),
        espaco_busca(),
        n_iter=n_iter,
        scoring={"eer": SCORER_EER_DF, "roc_auc": "roc_auc"},
        refit=False,
        cv=kfold,
        random_state=semente,
        n_jobs=-1,
        verbose=1,
    )
    t0 = time.perf_counter()
    busca.fit(X_tr, y_tr)
    t_busca = time.perf_counter() - t0

    cv = pd.DataFrame(busca.cv_results_)
    i_melhor = int(cv["mean_test_eer"].idxmax())   # scores negados: máx = menor EER
    params = {k: (v.item() if isinstance(v, np.generic) else v)
              for k, v in cv.loc[i_melhor, "params"].items()}
    print(f"\nBusca concluída em {t_busca/60:.1f} min. Melhor: {params}")
    print(f"  EER CV: {-cv.loc[i_melhor, 'mean_test_eer']:.4f} "
          f"± {cv.loc[i_melhor, 'std_test_eer']:.4f} | "
          f"AUC CV: {cv.loc[i_melhor, 'mean_test_roc_auc']:.4f}")

    dir_met = raiz / "results" / "metricas"
    dir_met.mkdir(parents=True, exist_ok=True)
    cv_salvar = cv.drop(columns=[c for c in cv.columns if c.startswith("split")])
    cv_salvar.to_csv(dir_met / "svm_random_search_cv.csv", index=False)
    with open(dir_met / "svm_random_search.json", "w", encoding="utf-8") as f:
        json.dump({
            "modelo": "svm_random_search",
            "braco": "principal",
            "n_treino": int(len(X_tr)),
            "cv": "StratifiedKFold(5, shuffle=True, random_state=42) sobre o treino do braço principal",
            "n_iter_config": n_iter_cfg,
            "n_iter_efetivo": n_iter,
            "projecao_horas_antes_da_busca": round(horas_projetadas, 2),
            "orcamento_horas": ORCAMENTO_BUSCA_HORAS,
            "paralelismo_efetivo_assumido": PARALELISMO_EFETIVO,
            "tempo_um_fit_fold_s": round(t_fit, 1),
            "tempo_busca_s": round(t_busca, 1),
            "criterio_busca": "EER via decision_function (greater_is_better=False), roc_auc ao lado",
            "justificativa_criterio": "mesma do RF — ver rf_random_search.json e docstring de src/models/ajustar_rf.py",
            "espaco_busca": {
                "svc__C": "loguniform(0.1, 100)",
                "svc__gamma": "['scale'] (50%) + loguniform(1e-4, 1e-1) (50%)",
                "svc__class_weight": ["balanced", None],
            },
            "score": "decision_function (probability=False — Platt scaling dispensado; "
                     "escala real centrada em zero, documentada)",
            "paralelismo": "RandomizedSearchCV(n_jobs=-1) com SVC single-thread",
            "melhor": {"params": params,
                       "eer_cv_medio": round(-float(cv.loc[i_melhor, "mean_test_eer"]), 4),
                       "eer_cv_std": round(float(cv.loc[i_melhor, "std_test_eer"]), 4),
                       "roc_auc_cv_medio": round(float(cv.loc[i_melhor, "mean_test_roc_auc"]), 4),
                       # espelha rf_random_search.json: os dois JSONs vão para a
                       # MESMA tabela, então registram os mesmos campos.
                       "roc_auc_cv_std": round(float(cv.loc[i_melhor, "std_test_roc_auc"]), 4)},
            "semente": semente,
            "ambiente": ambiente(n_jobs_inferencia=1),
            "hash_md5_subamostra_csv": hashlib.md5(
                (raiz / cfg["experimento"]["caminho_subamostra"]).read_bytes()).hexdigest(),
        }, f, indent=2, ensure_ascii=False)

    # ---- Treino final nos 30k completos -------------------------------------
    modelo = montar_pipeline(semente)
    modelo.set_params(**params)
    t0 = time.perf_counter()
    modelo.fit(X_tr, y_tr)
    t_treino = time.perf_counter() - t0
    print(f"Treino final em {t_treino:.1f}s")

    scores = modelo.decision_function(X_va)

    # ---- Limiar na validação (escala real, centrada em zero — ok) -----------
    sel = selecionar_limiar(y_va, scores, criterio="f1_macro",
                            conjunto="validacao")
    print(f"Limiar selecionado: {sel['limiar']:.4f} (escala do "
          f"decision_function) -> f1_macro {sel['f1_macro']:.4f}")

    m = avaliar(y_va, scores, nome, limiar=sel["limiar"])
    m["selecao_limiar"] = sel
    m["nota_escala_score"] = (
        "score = decision_function do SVC (real, centrado em zero) — escala "
        "DIFERENTE do predict_proba do RF ([0,1]). Não é inconsistência: a "
        "regra do protocolo (selecionar na validação, aplicar no teste) é "
        "agnóstica de escala, e selecionar_limiar não assume [0,1].")
    m["braco"] = "principal"
    m["hiperparametros"] = params
    m["n_treino"] = int(len(X_tr))
    m["n_validacao"] = int(len(X_va))
    m["n_vetores_suporte"] = [int(v) for v in modelo["svc"].n_support_]
    m["semente"] = semente
    m["treino_deterministico"] = (
        "SVC com probability=False é DETERMINÍSTICO — random_state só afeta o "
        "Platt scaling; a análise de estabilidade usa bootstrap da validação "
        "(scripts/estabilidade_modelos.py), não sementes.")
    m["tempo_treino_s"] = round(t_treino, 2)
    # O SVC é single-thread POR NATUREZA (libsvm não paraleliza o fit) — não é
    # uma escolha de configuração, é o algoritmo. Declarado explicitamente
    # porque o RF treina com n_jobs=-1: sem esta linha, a tabela final
    # compararia 6 núcleos contra 1 sem dizer. Ver `tempo_treino_s_n_jobs_1`
    # em rf_tuned_*.json — é esse o número comparável a este aqui.
    m["n_jobs_treino"] = 1

    cm = confusion_matrix(y_va, (scores >= sel["limiar"]).astype(int),
                          labels=[0, 1])
    m["matriz_confusao"] = cm.tolist()
    m["roc_auc_validacao"] = round(float(roc_auc_score(y_va, scores)), 4)

    # ---- Tempos: MESMO protocolo e MESMO helper do RF ------------------------
    m["tempos_inferencia"] = medir_tempos(modelo.decision_function, X_va,
                                          cfg["tempo"])
    m["ambiente"] = ambiente(n_jobs_inferencia=1)

    m["hash_md5_features_csv"] = hashlib.md5(
        (raiz / "data" / "features" / "features.csv").read_bytes()).hexdigest()
    m["hash_md5_split_csv"] = hashlib.md5(
        (raiz / "data" / "processed" / "split.csv").read_bytes()).hexdigest()
    m["hash_md5_subamostra_csv"] = hashlib.md5(
        (raiz / cfg["experimento"]["caminho_subamostra"]).read_bytes()).hexdigest()

    print(f"  f1_macro : {m['f1_macro']:.4f} | EER {m['eer']:.4f} | "
          f"AUC {m['roc_auc_validacao']:.4f}")
    print(f"  bonafide : recall {m['recall_bonafide']:.4f} | "
          f"precisão {m['precisao_bonafide']:.4f}")
    print(f"  latência : {m['tempos_inferencia']['latencia_ms']['mediana']} ms "
          f"(batch=1) | throughput "
          f"{m['tempos_inferencia']['throughput']['ms_por_audio_mediana']} ms/áudio")

    (raiz / "models").mkdir(exist_ok=True)
    joblib.dump(modelo, raiz / "models" / f"{nome}.joblib")
    with open(dir_met / f"{nome}.json", "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)
    plotar_matriz_confusao(
        cm, raiz / "results" / "figuras" / f"matriz_confusao_{nome}.png",
        f"SVM RBF ajustado (braço principal) — validação, "
        f"limiar {sel['limiar']:.2f}")
    print(f"Salvo: models/{nome}.joblib e results/metricas/{nome}.json")
    return m


if __name__ == "__main__":
    from ..utils.config import carregar_config

    RAIZ = Path(__file__).resolve().parents[2]
    buscar_e_treinar(carregar_config(RAIZ), RAIZ)
