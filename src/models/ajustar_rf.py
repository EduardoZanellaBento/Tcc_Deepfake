"""
Random Forest ajustado — Random Search (B3.2) + treino final nos dois braços (B3.3/B3.4)
========================================================================================

ONDE A BUSCA RODA:
    StratifiedKFold(5, shuffle, seed 42) SOMENTE sobre o treino do braço
    principal (a subamostra de ~30k compartilhada por RF, SVM e CNN). Validação
    e teste NÃO entram na busca — é exatamente o que o diagrama de
    src/data/split.py promete: o 5-fold particiona só o treino.

POR QUAL MÉTRICA BUSCAR (decisão registrada — ver também NOTA_LIMIAR.md):
    Por EER, uma métrica INDEPENDENTE DE LIMIAR (roc_auc reportado ao lado).
    Se a busca pontuasse por f1_macro no limiar fixo 0,50, ela preferiria
    sistematicamente as configurações cuja distribuição de scores por acaso cai
    perto de 0,50 — selecionaria por "score bem centrado", não por "modelo que
    ordena melhor". Como o limiar vai ser movido depois, DE PROPÓSITO (seleção
    na validação, src/models/avaliacao.py), isso otimizaria a coisa errada.
    Separar as duas perguntas — quem ordena melhor? (busca) e onde cortar?
    (validação) — é o que mantém o protocolo coerente.
    Alternativa considerada (f1_macro com seleção de limiar dentro de cada
    fold): mais fiel ao protocolo final, porém embute seleção aninhada, aumenta
    a variância da estimativa e exige um scorer customizado delicado. Registrada
    no JSON de resultados.

PARALELISMO (afeta o TEMPO medido, não o resultado):
    RandomizedSearchCV(n_jobs=-1) com RandomForestClassifier(n_jobs=1). Os dois
    em -1 disputariam os mesmos núcleos e o tempo mediria a briga, não o custo.
    Declarado no JSON.

BRAÇO DE REFERÊNCIA (B3.4):
    treina com os MESMOS hiperparâmetros vencedores do principal, SEM nova
    busca. Ele existe para responder "quanto custou a subamostra?" — se a
    configuração também mudasse, a diferença mediria duas causas ao mesmo tempo
    (tamanho do treino E hiperparâmetros). Limitação declarada: a configuração
    ótima para 30k não é necessariamente a ótima para 103k, então o braço de
    referência é um LIMITE INFERIOR do ganho possível com o treino completo.

Artefatos (nomes novos — rf_baseline_eval_principal.* fica preservado como "antes"):
    results/metricas/rf_random_search.json + rf_random_search_cv.csv
    models/rf_tuned_{braco}.joblib
    results/metricas/rf_tuned_{braco}.json
    results/figuras/matriz_confusao_rf_tuned_{braco}.png

Rode a partir da raiz:  python -m src.models.ajustar_rf
"""

import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, make_scorer, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from ..utils.seeds import fixar_seeds
from ..data.split import (carregar_dados_split, colunas_features,
                          filtrar_treino_braco)
from .avaliacao import avaliar, selecionar_limiar
from .tempo import ambiente, medir_tempos


# ---------------------------------------------------------------------------
# Scorer de EER — função top-level para ser picklável pelos workers do search
# ---------------------------------------------------------------------------
def _eer(y_true, y_score) -> float:
    """EER a partir do score da classe positiva (o make_scorer com
    response_method='predict_proba' já entrega a coluna da classe 1)."""
    from .avaliacao import calcular_eer
    return calcular_eer(y_true, y_score)[0]


# greater_is_better=False: menor EER é melhor; o sklearn NEGA o valor
# internamente, então mean_test_eer sai negativo no cv_results (ex.: -0,18).
SCORER_EER = make_scorer(_eer, response_method="predict_proba",
                         greater_is_better=False)


def espaco_busca(semente: int) -> dict:
    """Espaço do Random Search. min_samples_leaf e class_weight são exigência
    do orientador (config.yaml -> tuning): com folhas puras o class_weight é
    neutralizado no predict_proba (NOTA_LIMIAR.md, §2)."""
    del semente  # distribuições são amostradas pelo random_state do search
    return {
        "min_samples_leaf": randint(5, 21),          # obrigatório: 5..20
        "class_weight": ["balanced", "balanced_subsample"],   # obrigatório
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "max_features": ["sqrt", "log2", 0.3],
    }


def buscar_hiperparametros(cfg: dict, raiz: Path) -> dict:
    """B3.2 — Random Search 5-fold no TREINO do braço principal (30k)."""
    semente = fixar_seeds(cfg["semente"])
    df = carregar_dados_split(raiz)
    cols = colunas_features(df)

    treino = df[df["conjunto"] == "treino"]
    treino = filtrar_treino_braco(treino, "principal", cfg, raiz)
    X_tr, y_tr = treino[cols].values, treino["classe_binaria"].values
    print(f"Busca: {X_tr.shape} (braço principal), 5-fold estratificado, "
          f"n_iter={cfg['tuning']['n_iter']}, scorer=EER (roc_auc ao lado).")

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=semente)
    busca = RandomizedSearchCV(
        RandomForestClassifier(random_state=semente, n_jobs=1),
        espaco_busca(semente),
        n_iter=int(cfg["tuning"]["n_iter"]),
        scoring={"eer": SCORER_EER, "roc_auc": "roc_auc"},
        refit=False,          # o treino final (B3.3) é feito explicitamente
        cv=kfold,
        random_state=semente,
        n_jobs=-1,
        verbose=1,
        return_train_score=False,
    )
    t0 = time.perf_counter()
    busca.fit(X_tr, y_tr)
    t_busca = time.perf_counter() - t0

    cv = pd.DataFrame(busca.cv_results_)
    # mean_test_eer é NEGATIVO (greater_is_better=False); o melhor é o máximo.
    i_melhor = int(cv["mean_test_eer"].idxmax())
    # randint(...) devolve np.int64, que o json.dump não serializa
    params_melhor = {k: (v.item() if isinstance(v, np.generic) else v)
                     for k, v in cv.loc[i_melhor, "params"].items()}
    melhor = {
        "params": params_melhor,
        "eer_cv_medio": round(-float(cv.loc[i_melhor, "mean_test_eer"]), 4),
        "eer_cv_std": round(float(cv.loc[i_melhor, "std_test_eer"]), 4),
        "roc_auc_cv_medio": round(float(cv.loc[i_melhor, "mean_test_roc_auc"]), 4),
        "roc_auc_cv_std": round(float(cv.loc[i_melhor, "std_test_roc_auc"]), 4),
    }
    print(f"\nBusca concluída em {t_busca/60:.1f} min.")
    print(f"Melhor configuração (menor EER médio nos 5 folds): {melhor['params']}")
    print(f"  EER CV: {melhor['eer_cv_medio']:.4f} ± {melhor['eer_cv_std']:.4f} "
          f"| AUC CV: {melhor['roc_auc_cv_medio']:.4f}")

    dir_met = raiz / "results" / "metricas"
    dir_met.mkdir(parents=True, exist_ok=True)
    # cv_results completo, auditável (uma linha por configuração testada)
    cv_salvar = cv.drop(columns=[c for c in cv.columns if c.startswith("split")])
    cv_salvar.to_csv(dir_met / "rf_random_search_cv.csv", index=False)

    registro = {
        "modelo": "rf_random_search",
        "braco": "principal",
        "n_treino": int(len(X_tr)),
        "cv": "StratifiedKFold(5, shuffle=True, random_state=42) sobre o treino do braço principal",
        "n_iter": int(cfg["tuning"]["n_iter"]),
        "criterio_busca": "EER (make_scorer greater_is_better=False), roc_auc reportado ao lado",
        "justificativa_criterio": (
            "métrica independente de limiar: buscar por f1_macro@0,50 "
            "selecionaria configurações com scores centrados em 0,50, não as "
            "que melhor ordenam; o limiar será selecionado depois, na "
            "validação (ver docstring de src/models/ajustar_rf.py e NOTA_LIMIAR.md)"),
        "alternativa_considerada": (
            "f1_macro com seleção de limiar dentro de cada fold — mais fiel ao "
            "protocolo final, porém com seleção aninhada, maior variância e "
            "scorer customizado delicado; não adotada"),
        "espaco_busca": {
            "min_samples_leaf": "randint(5, 21) [obrigatório — orientador]",
            "class_weight": "['balanced', 'balanced_subsample'] [obrigatório — orientador]",
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "max_features": ["sqrt", "log2", 0.3],
        },
        "paralelismo": "RandomizedSearchCV(n_jobs=-1) com RandomForestClassifier(n_jobs=1)",
        "tempo_busca_s": round(t_busca, 1),
        "melhor": melhor,
        "semente": semente,
        "ambiente": ambiente(n_jobs_inferencia=1),
        "hash_md5_subamostra_csv": hashlib.md5(
            (raiz / cfg["experimento"]["caminho_subamostra"]).read_bytes()).hexdigest(),
    }
    with open(dir_met / "rf_random_search.json", "w", encoding="utf-8") as f:
        json.dump(registro, f, indent=2, ensure_ascii=False)
    print(f"Registro da busca salvo em results/metricas/rf_random_search.json")
    return melhor["params"]


def treinar_final(cfg: dict, raiz: Path, params: dict, braco: str) -> dict:
    """B3.3/B3.4 — treina a configuração vencedora no braço pedido, seleciona o
    limiar na VALIDAÇÃO (nunca no teste) e mede tempos pelo protocolo do config."""
    semente = fixar_seeds(cfg["semente"])
    nome = f"rf_tuned_{braco}"

    df = carregar_dados_split(raiz)
    cols = colunas_features(df)
    treino = df[df["conjunto"] == "treino"]
    n_treino_completo = len(treino)
    treino = filtrar_treino_braco(treino, braco, cfg, raiz)
    validacao = df[df["conjunto"] == "validacao"]
    X_tr, y_tr = treino[cols].values, treino["classe_binaria"].values
    X_va, y_va = validacao[cols].values, validacao["classe_binaria"].values
    print(f"\n=== {nome}: treino {X_tr.shape} de {n_treino_completo} | "
          f"validação {X_va.shape} ===")

    modelo = RandomForestClassifier(**params, random_state=semente, n_jobs=-1)
    t0 = time.perf_counter()
    modelo.fit(X_tr, y_tr)
    t_treino = time.perf_counter() - t0
    print(f"Treino em {t_treino:.1f}s (n_jobs=-1)")

    scores = modelo.predict_proba(X_va)[:, 1]

    # ---- Limiar: selecionado na validação, registrado com critério e regra ---
    sel = selecionar_limiar(y_va, scores, criterio="f1_macro",
                            conjunto="validacao")
    print(f"Limiar selecionado (f1_macro na validação): {sel['limiar']:.4f} "
          f"-> f1_macro {sel['f1_macro']:.4f} "
          f"({sel['n_candidatos']} candidatos, "
          f"{sel['n_empates_no_maximo']} empate(s) no máximo)")

    m = avaliar(y_va, scores, nome, limiar=sel["limiar"])
    m["selecao_limiar"] = sel
    m["braco"] = braco
    m["hiperparametros"] = {k: (v if v is None or isinstance(v, (int, float, bool))
                                else str(v)) for k, v in params.items()}
    m["origem_hiperparametros"] = (
        "rf_random_search.json (busca no braço principal)" if braco == "principal"
        else "MESMA configuração vencedora do braço principal, sem nova busca — "
             "um só fator por vez; ver docstring (limitação: limite inferior do "
             "ganho com o treino completo)")
    m["n_treino"] = int(len(X_tr))
    m["n_validacao"] = int(len(X_va))
    m["semente"] = semente
    m["tempo_treino_s"] = round(t_treino, 2)
    m["n_jobs_treino"] = -1

    # ---- Segunda medição de treino, com n_jobs=1 (R2.1) ---------------------
    # POR QUE DUAS MEDIÇÕES: a tabela final compara MODELOS, não configurações
    # de paralelismo. O RF acima treina com n_jobs=-1 — é o custo REAL de uso,
    # e é o que fica em `tempo_treino_s`. Mas o SVC é single-thread por
    # natureza, então comparar 4,3 s de RF em N núcleos contra 11,2 s de SVM em
    # 1 núcleo compara hardware, não algoritmo. `tempo_treino_s_n_jobs_1` é o
    # refit da MESMA configuração e da MESMA semente com n_jobs=1: é o único
    # número comparável ao do SVM. Os dois ficam no JSON, cada um rotulado.
    # O modelo refeito aqui é DESCARTADO — o persistido é o de cima.
    modelo_1t = RandomForestClassifier(**params, random_state=semente, n_jobs=1)
    t0 = time.perf_counter()
    modelo_1t.fit(X_tr, y_tr)
    t_treino_1t = time.perf_counter() - t0
    del modelo_1t
    print(f"Refit cronometrado em {t_treino_1t:.1f}s (n_jobs=1, só para a tabela)")
    m["tempo_treino_s_n_jobs_1"] = round(t_treino_1t, 2)
    m["nota_tempo_treino"] = (
        "tempo_treino_s é com n_jobs=-1 (custo real de uso); "
        "tempo_treino_s_n_jobs_1 é o refit da mesma configuração com n_jobs=1, "
        "o único número comparável ao SVM, que é single-thread por natureza. "
        "n_jobs não altera o resultado do RF, só o tempo.")

    cm = confusion_matrix(y_va,
                          (scores >= sel["limiar"]).astype(int), labels=[0, 1])
    m["matriz_confusao"] = cm.tolist()

    m["roc_auc_validacao"] = round(float(roc_auc_score(y_va, scores)), 4)

    imp = pd.Series(modelo.feature_importances_, index=cols).sort_values(
        ascending=False)
    m["top10_features"] = imp.head(10).round(5).to_dict()

    # ---- Tempos (protocolo config.yaml -> tempo; MESMO helper do SVM) --------
    # Inferência cronometrada com n_jobs=1: é o n_jobs fixado para todos os
    # modelos clássicos (o SVM é single-thread por natureza) — comparação justa.
    modelo.n_jobs = 1
    m["tempos_inferencia"] = medir_tempos(
        lambda X: modelo.predict_proba(X)[:, 1], X_va, cfg["tempo"])
    m["ambiente"] = ambiente(n_jobs_inferencia=1)

    # ---- Rastreabilidade: os três artefatos congelados -----------------------
    m["hash_md5_features_csv"] = hashlib.md5(
        (raiz / "data" / "features" / "features.csv").read_bytes()).hexdigest()
    m["hash_md5_split_csv"] = hashlib.md5(
        (raiz / "data" / "processed" / "split.csv").read_bytes()).hexdigest()
    m["hash_md5_subamostra_csv"] = hashlib.md5(
        (raiz / cfg["experimento"]["caminho_subamostra"]).read_bytes()).hexdigest()

    # ---- Relatório -----------------------------------------------------------
    print(f"  f1_macro   : {m['f1_macro']:.4f}  (baseline argmax do braço: ver "
          f"rf_baseline_eval_{braco}.json)")
    print(f"  EER        : {m['eer']:.4f} | limiar_eer {m['limiar_eer']:.4f} | "
          f"AUC {m['roc_auc_validacao']:.4f}")
    print(f"  bonafide   : recall {m['recall_bonafide']:.4f} | "
          f"precisão {m['precisao_bonafide']:.4f}")
    print(f"  latência   : {m['tempos_inferencia']['latencia_ms']['mediana']} ms "
          f"(batch=1) | throughput "
          f"{m['tempos_inferencia']['throughput']['ms_por_audio_mediana']} ms/áudio")

    # ---- Persistência --------------------------------------------------------
    (raiz / "models").mkdir(exist_ok=True)
    joblib.dump(modelo, raiz / "models" / f"{nome}.joblib")
    dir_met = raiz / "results" / "metricas"
    dir_met.mkdir(parents=True, exist_ok=True)
    with open(dir_met / f"{nome}.json", "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)
    from .avaliacao import plotar_matriz_confusao
    plotar_matriz_confusao(
        cm, raiz / "results" / "figuras" / f"matriz_confusao_{nome}.png",
        f"RF ajustado (braço {braco}) — validação, limiar {sel['limiar']:.2f}")
    print(f"Salvo: models/{nome}.joblib e results/metricas/{nome}.json")
    return m


if __name__ == "__main__":
    from ..utils.config import carregar_config

    RAIZ = Path(__file__).resolve().parents[2]
    cfg = carregar_config(RAIZ)

    params = buscar_hiperparametros(cfg, RAIZ)          # B3.2
    treinar_final(cfg, RAIZ, params, braco="principal")  # B3.3
    treinar_final(cfg, RAIZ, params, braco="referencia")  # B3.4
