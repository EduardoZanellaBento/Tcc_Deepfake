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

5. Braço duplo (chave `experimento.braco` do config.yaml, lida via
   filtrar_treino_braco em src/data/split.py):
     braco == 'principal'  -> TREINO filtrado pelos IDs de
                              `experimento.caminho_subamostra` (subamostra ~30k
                              compartilhada por RF, SVM e CNN);
     braco == 'referencia' -> treino no conjunto COMPLETO do eval (103.723),
                              para quantificar o custo da subamostra.
   Em AMBOS os braços, VALIDAÇÃO e TESTE permanecem COMPLETOS (exigência
   textual do orientador). Regra de negócio: o RF roda nos DOIS braços (o
   __main__ faz isso); o SVM, quando existir, roda SÓ no principal. Artefatos
   por braço: rf_baseline_eval_principal.* e rf_baseline_eval_referencia.*.
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

from ..utils.seeds import fixar_seeds
from ..data.split import (carregar_dados_split, colunas_features,
                          filtrar_treino_braco, resumo_split)
# calcular_eer é re-exportado aqui por compatibilidade: consumidores antigos
# importavam de src.models.treinar_rf. O lugar canônico é src.models.avaliacao.
from .avaliacao import avaliar, calcular_eer, plotar_matriz_confusao


def treinar(cfg: dict, raiz: Path, nome: str | None = None,
            braco: str | None = None) -> dict:
    """Treina o RF baseline no braço pedido e salva artefatos com o prefixo `nome`.

    Args:
        cfg: config.yaml carregado.
        raiz: raiz do projeto.
        nome: prefixo dos artefatos. Default: 'rf_baseline_eval_{braco}'.
            Não colide com os baselines já publicados: rf_baseline.* (universo
            histórico de 181.566) e rf_baseline_eval.* (universo eval, treino
            completo, anterior ao braço duplo) ficam preservados como referência
            citada no trabalho e NÃO devem ser sobrescritos.
        braco: 'principal' (subamostra ~30k) ou 'referencia' (treino completo).
            Se None, vale cfg['experimento']['braco'] — config que ninguém lê é
            comentário. Ver filtrar_treino_braco em src/data/split.py.
    """
    if braco is None:
        braco = cfg["experimento"]["braco"]
    if nome is None:
        nome = f"rf_baseline_eval_{braco}"
    semente = fixar_seeds(cfg["semente"])

    # ---- Dados ---------------------------------------------------------------
    df = carregar_dados_split(raiz)
    resumo_split(df)

    cols = colunas_features(df)
    print(f"\n{len(cols)} features em uso (esperado: 44).")

    treino = df[df["conjunto"] == "treino"]
    validacao = df[df["conjunto"] == "validacao"]

    # ---- Braço do experimento (só o TREINO muda; validação/teste completos) --
    n_treino_completo = len(treino)
    treino = filtrar_treino_braco(treino, braco, cfg, raiz)
    print(f"\nBraço '{braco}': treino com {len(treino)} de "
          f"{n_treino_completo} áudios (validação e teste completos).")

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
    m = avaliar(y_va, y_pred, scores, nome)
    m["tempo_treino_s"] = round(t_treino, 2)
    m["tempo_inferencia_total_s"] = round(t_inf, 4)
    m["tempo_inferencia_por_audio_ms"] = round(1000 * t_inf / len(X_va), 4)
    m["braco"] = braco
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
    print(f"RESULTADOS — RF baseline, braço '{braco}' (conjunto de VALIDAÇÃO)")
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
    for feat, v in imp.head(10).items():
        print(f"    {feat:<20} {v:.4f}")
    m["top10_features"] = imp.head(10).round(5).to_dict()

    # ---- Persistência --------------------------------------------------------
    # Rastreabilidade: o hash do split identifica exatamente qual partição gerou
    # estas métricas (mesmo padrão do curva_aprendizado_rf.json).
    split_csv = raiz / "data" / "processed" / "split.csv"
    m["hash_md5_split_csv"] = hashlib.md5(split_csv.read_bytes()).hexdigest()
    if braco == "principal":
        subamostra_csv = raiz / cfg["experimento"]["caminho_subamostra"]
        m["hash_md5_subamostra_csv"] = hashlib.md5(
            subamostra_csv.read_bytes()).hexdigest()

    (raiz / "models").mkdir(exist_ok=True)
    joblib.dump(modelo, raiz / "models" / f"{nome}.joblib")

    dir_met = raiz / "results" / "metricas"
    dir_met.mkdir(parents=True, exist_ok=True)
    with open(dir_met / f"{nome}.json", "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)

    plotar_matriz_confusao(
        cm,
        raiz / "results" / "figuras" / f"matriz_confusao_{nome}.png",
        f"Random Forest baseline (eval, braço {braco}) — validação",
    )
    print(f"\nModelo salvo em models/{nome}.joblib")
    print(f"Métricas salvas em results/metricas/{nome}.json")
    return m


if __name__ == "__main__":
    import yaml

    RAIZ = Path(__file__).resolve().parents[2]
    cfg = yaml.safe_load(open(RAIZ / "config" / "config.yaml", encoding="utf-8"))
    # Regra de negócio do braço duplo: o RF roda nos DOIS braços — 'principal'
    # é o ambiente da comparação RF × SVM × CNN, 'referencia' quantifica o
    # custo da subamostra. O SVM (quando existir) roda SÓ no principal.
    for braco in ("principal", "referencia"):
        treinar(cfg, RAIZ, braco=braco)