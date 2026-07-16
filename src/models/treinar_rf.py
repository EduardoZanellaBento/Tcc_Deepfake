"""
Random Forest
===================================

primeiro RF treinado, matriz de confusão e métricas iniciais.

BASELINE significa: hiperparâmetros razoáveis e honestos, SEM busca. O ajuste fino
(Random Search + 5-fold) é o ajuste dos modelos clássicos. Essa separação é deliberada e é boa ciência:
o baseline é a régua contra a qual o ajuste será medido. Sem ele, você não sabe se
o Random Search melhorou algo ou se só gastou CPU.

DECISÕES DE PROJETO:

1. Sem StandardScaler. O RF decide por LIMIARES ("centroide > 1500?"). Multiplicar
   uma feature por mil não muda a ordem dos valores, logo não muda a árvore: ele é
   invariante a escala monotônica. O SVM VAI precisar de scaler, porque
   depende de distância euclidiana — e aí o scaler entra DENTRO do Pipeline,
   ajustado por fold, para não vazar dados.

2. class_weight="balanced". Com 8,8 spoof : 1 bonafide, o modelo que sempre chuta
   "spoof" acerta ~89,8%. O `balanced` pesa cada classe por n/(k*n_c), penalizando
   mais o erro na classe minoritária (bonafide) — que é justamente a que importa
   não errar: um bonafide classificado como spoof é um usuário legítimo barrado.

3. Avaliação na VALIDAÇÃO, não no teste. O teste continua lacrado até o final do projeto.

4. Comparação contra o CLASSIFICADOR TRIVIAL (sempre "spoof"). Se o RF não superar
   ~89,8% de acurácia, ele não aprendeu nada — apenas descobriu a classe majoritária.
   É por isso que acurácia sozinha, aqui, é uma métrica enganosa.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # backend sem janela: só salva arquivo
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
)
import joblib

from ..utils.seeds import fixar_seeds
from ..data.split import carregar_dados_split, colunas_features, resumo_split


def calcular_eer(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Equal Error Rate — a métrica padrão do ASVspoof.

    O QUE É: acurácia/F1 dependem de um limiar fixo (0,5). O EER remove essa
    arbitrariedade — ele varre TODOS os limiares possíveis e encontra o ponto em
    que a taxa de falsos positivos (FPR) iguala a de falsos negativos (FNR). É esse
    valor comum que se reporta. Menor = melhor. Por ser independente de limiar, é
    o que permite comparar esse número com o da literatura (Yamagishi et al. 2022
    reportam EER de 1,32% em LA).

    Convenção adotada aqui: classe positiva = spoof (classe_binaria = 1), e `scores`
    é a probabilidade predita de ser spoof. Trocar a classe positiva troca o significado de FPR e FNR.

    Returns:
        (eer, limiar_no_eer)
    """
    fpr, tpr, limiares = roc_curve(y_true, scores, pos_label=1)
    fnr = 1 - tpr
    # O ponto onde as duas curvas se cruzam: |FNR - FPR| mínimo.
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = (fpr[i] + fnr[i]) / 2
    return float(eer), float(limiares[i])


def avaliar(y_true, y_pred, scores, nome: str) -> dict:
    """Calcula o conjunto de métricas do config.yaml e imprime a leitura crítica."""
    m = {
        "modelo": nome,
        "acuracia": float(accuracy_score(y_true, y_pred)),
        # zero_division=0: se o modelo NUNCA prevê uma classe, a precisão dela é 0/0.
        # Sem isso o sklearn emite warning e devolve 0 silenciosamente.
        "precisao_spoof": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_spoof": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_spoof": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        # A classe minoritária é a que revela se o modelo realmente aprendeu:
        "precisao_bonafide": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "recall_bonafide": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "f1_bonafide": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        # f1 macro: média não ponderada das duas classes. Com dados desbalanceados, é MUITO mais informativo que a acurácia — não deixa a classe majoritária esconder o fracasso na minoritária.
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    eer, limiar = calcular_eer(y_true, scores)
    m["eer"] = eer
    m["limiar_eer"] = limiar
    return m


def plotar_matriz_confusao(cm: np.ndarray, caminho: Path, titulo: str) -> None:
    """Matriz de confusão com contagens absolutas E percentual por linha.

    Percentual POR LINHA (normalizado pelo total real de cada classe) é o que
    importa aqui: com 8,8:1, os números absolutos da linha 'spoof' esmagam
    visualmente os da 'bonafide' e escondem o erro que interessa.
    """
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    nomes = ["bonafide (0)", "spoof (1)"]
    ax.set_xticks([0, 1], labels=nomes)
    ax.set_yticks([0, 1], labels=nomes)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title(titulo)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n({cm_pct[i, j]:.1f}%)",
                    ha="center", va="center",
                    color="white" if cm_pct[i, j] > 50 else "black",
                    fontsize=11)

    fig.colorbar(im, ax=ax, label="% da classe real")
    fig.tight_layout()
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    print(f"Matriz de confusão salva em {caminho}")


def treinar(cfg: dict, raiz: Path) -> dict:
    semente = fixar_seeds(cfg["semente"])

    # ---- Dados ---------------------------------------------------------------
    df = carregar_dados_split(raiz)
    resumo_split(df)

    cols = colunas_features(df)
    print(f"\n{len(cols)} features em uso (esperado: 44).")

    treino = df[df["conjunto"] == "treino"]
    validacao = df[df["conjunto"] == "validacao"]

    X_tr, y_tr = treino[cols].values, treino["classe_binaria"].values
    X_va, y_va = validacao[cols].values, validacao["classe_binaria"].values
    print(f"treino: {X_tr.shape} | validação: {X_va.shape}")

    # ---- Modelo --------------------------------------------------------------
    # n_estimators=100: baseline. Mais árvores tendem a estabilizar a predição
    #   (menos variância), com retorno decrescente e custo linear de inferência.
    # max_depth=None: árvores crescem até o fim. No RF isso é aceitável — o
    #   ensemble + bagging controlam o overfitting que uma árvore isolada teria.
    # n_jobs=-1: paraleliza o TREINO. Não afeta o resultado, só o tempo.
    modelo = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        class_weight="balanced",
        random_state=semente,   # explícito: não depender do estado global do NumPy
        n_jobs=-1,
    )

    t0 = time.perf_counter()
    modelo.fit(X_tr, y_tr)
    t_treino = time.perf_counter() - t0
    print(f"\nTreino concluído em {t_treino:.1f}s")

    # ---- Inferência + tempo --------------------------------------------------
    # MEDIÇÃO DE TEMPO: medimos a predição do lote inteiro
    # de validação e dividimos pelo nº de amostras. É uma medida de THROUGHPUT.
    # Ela NÃO é a latência de um áudio isolado (que sofre overhead fixo por chamada).
    # Para a comparação final com SVM/CNN valer, os três precisam ser medidos do
    # mesmo jeito, no mesmo hardware, com o mesmo n_jobs. Registre isso.
    t0 = time.perf_counter()
    y_pred = modelo.predict(X_va)
    t_inf = time.perf_counter() - t0
    scores = modelo.predict_proba(X_va)[:, 1]   # P(spoof) — coluna da classe 1

    # ---- Métricas ------------------------------------------------------------
    m = avaliar(y_va, y_pred, scores, "random_forest_baseline")
    m["tempo_treino_s"] = round(t_treino, 2)
    m["tempo_inferencia_total_s"] = round(t_inf, 4)
    m["tempo_inferencia_por_audio_ms"] = round(1000 * t_inf / len(X_va), 4)
    m["n_treino"] = int(len(X_tr))
    m["n_validacao"] = int(len(X_va))
    m["semente"] = semente

    # ---- O baseline trivial: a régua que desmascara a acurácia ---------------
    # Um "modelo" que sempre responde "spoof", sem olhar para o áudio.
    trivial = np.ones_like(y_va)
    m["acuracia_baseline_trivial"] = float(accuracy_score(y_va, trivial))

    cm = confusion_matrix(y_va, y_pred, labels=[0, 1])
    m["matriz_confusao"] = cm.tolist()

    # ---- Relatório -----------------------------------------------------------
    print("\n" + "=" * 64)
    print("RESULTADOS — Random Forest baseline (conjunto de VALIDAÇÃO)")
    print("=" * 64)
    print(f"  acurácia            : {m['acuracia']:.4f}")
    print(f"  acurácia trivial    : {m['acuracia_baseline_trivial']:.4f}  <- sempre 'spoof'")
    print(f"  ganho sobre trivial : {m['acuracia'] - m['acuracia_baseline_trivial']:+.4f}")
    print(f"  f1_macro            : {m['f1_macro']:.4f}")
    print(f"  EER                 : {m['eer']:.4f}  ({100*m['eer']:.2f}%)")
    print("\n  classe SPOOF (majoritária):")
    print(f"    precisão {m['precisao_spoof']:.4f} | recall {m['recall_spoof']:.4f} | f1 {m['f1_spoof']:.4f}")
    print("  classe BONAFIDE (minoritária — a que importa):")
    print(f"    precisão {m['precisao_bonafide']:.4f} | recall {m['recall_bonafide']:.4f} | f1 {m['f1_bonafide']:.4f}")
    print(f"\n  tempo de inferência : {m['tempo_inferencia_por_audio_ms']:.4f} ms/áudio")
    print("\n  matriz de confusão (linhas=real, colunas=predito):")
    print(f"    bonafide -> [{cm[0,0]:>7}, {cm[0,1]:>7}]")
    print(f"    spoof    -> [{cm[1,0]:>7}, {cm[1,1]:>7}]")

    # ---- Features mais importantes -------------------------------------------
    # LEIA COM CUIDADO: a importância do RF é por REDUÇÃO DE IMPUREZA, e ela é
    # enviesada a favor de features contínuas de alta cardinalidade. É uma pista
    # sobre "o que o modelo usou", não prova de causalidade acústica. Para uma
    # afirmação mais forte, usar permutation_importance.
    imp = pd.Series(modelo.feature_importances_, index=cols).sort_values(ascending=False)
    print("\n  top 10 features (importância por impureza — ler com ressalva):")
    for nome, v in imp.head(10).items():
        print(f"    {nome:<20} {v:.4f}")
    m["top10_features"] = imp.head(10).round(5).to_dict()

    # ---- Persistência --------------------------------------------------------
    (raiz / "models").mkdir(exist_ok=True)
    joblib.dump(modelo, raiz / "models" / "rf_baseline.joblib")

    dir_met = raiz / "results" / "metricas"
    dir_met.mkdir(parents=True, exist_ok=True)
    with open(dir_met / "rf_baseline.json", "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)

    plotar_matriz_confusao(
        cm,
        raiz / "results" / "figuras" / "matriz_confusao_rf_baseline.png",
        "Random Forest baseline — validação",
    )
    print(f"\nModelo salvo em models/rf_baseline.joblib")
    print(f"Métricas salvas em results/metricas/rf_baseline.json")
    return m


if __name__ == "__main__":
    import yaml

    RAIZ = Path(__file__).resolve().parents[2]
    cfg = yaml.safe_load(open(RAIZ / "config" / "config.yaml", encoding="utf-8"))
    treinar(cfg, RAIZ)