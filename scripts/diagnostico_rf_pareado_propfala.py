"""
Diagnóstico de padding (4.2) — RF pareado por faixa de prop_fala
================================================================

SÓ LÊ, TREINA EM SUBCONJUNTO DIAGNÓSTICO E REPORTA. Sem extração de áudio;
o teste continua lacrado (só treino e validação são usados).

PERGUNTA DE BANCA QUE ESTE SCRIPT RESPONDE:
    "Seu modelo aprendeu a detectar voz sintética ou aprendeu a detectar
    silêncio?" — bonafide tem prop_fala média 0,63 e spoof 0,85; o modelo pode
    estar usando a fração de padding como atalho, mesmo com prop_fala fora do X.

PROTOCOLO (pareamento):
    prop_fala é dividida em 10 faixas (bins). Dentro de cada faixa, amostramos
    o MESMO número de bonafide e spoof (o mínimo dos dois lados; faixas sem as
    duas classes são descartadas). No subconjunto pareado, a distribuição de
    prop_fala é equivalente entre as classes -> o atalho de silêncio deixa de
    ter poder discriminativo. Pareamos TREINO (para treinar o RF pareado) e
    VALIDAÇÃO (para avaliar sem que o atalho ajude na própria métrica).

    Avaliamos DOIS modelos no mesmo conjunto pareado de validação:
      (a) RF pareado  — treinado no subconjunto pareado;
      (b) RF baseline — models/rf_baseline_eval.joblib, treinado no treino
          completo (o modelo cuja honestidade está em teste).

INTERPRETAÇÃO (escrita também no JSON de saída):
    - Desempenho DESABA no pareado -> parte relevante do acerto vinha do atalho
      de silêncio, não de artefatos de síntese. Achado forte para a discussão.
    - Desempenho SE MANTÉM -> as features carregam sinal acústico genuíno e
      prop_fala é um confundidor separável. Também defende o trabalho na banca.
    Nenhum dos dois desfechos é "ruim". O ruim é não saber.

SAÍDA:
    results/metricas/rf_pareado_propfala.json
    + conclusão em texto no stdout

Rode a partir da raiz:  python -m scripts.diagnostico_rf_pareado_propfala
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score

from src.utils.config import carregar_config
from src.utils.seeds import fixar_seeds
from src.data.split import carregar_dados_split, colunas_features
from src.models.treinar_rf import calcular_eer

RAIZ = Path(__file__).resolve().parents[1]
N_BINS = 10


def parear(df: pd.DataFrame, semente: int) -> tuple[pd.DataFrame, list[dict]]:
    """Amostra o mesmo nº de bonafide e spoof dentro de cada faixa de prop_fala."""
    bins = np.linspace(0.0, 1.0, N_BINS + 1)
    faixa = pd.cut(df["prop_fala"], bins=bins, include_lowest=True)

    partes, registro = [], []
    for f in faixa.cat.categories:
        sub = df[faixa == f]
        bona = sub[sub["classe_binaria"] == 0]
        spoof = sub[sub["classe_binaria"] == 1]
        n = min(len(bona), len(spoof))
        registro.append({"faixa": str(f), "n_bonafide_disponivel": len(bona),
                         "n_spoof_disponivel": len(spoof), "n_pareado_por_classe": n})
        if n == 0:
            continue
        partes.append(bona.sample(n=n, random_state=semente))
        partes.append(spoof.sample(n=n, random_state=semente))
    return pd.concat(partes), registro


def avaliar(nome: str, y, y_pred, scores) -> dict:
    eer, limiar = calcular_eer(y, scores)
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    return {
        "modelo": nome,
        "f1_macro": round(float(f1_score(y, y_pred, average="macro",
                                         zero_division=0)), 4),
        "eer": round(eer, 4),
        "limiar_eer": round(limiar, 4),
        "matriz_confusao": cm.tolist(),
    }


def main() -> None:
    cfg = carregar_config(RAIZ)
    semente = fixar_seeds(cfg["semente"])

    df = carregar_dados_split(RAIZ)
    cols = colunas_features(df)
    treino = df[df["conjunto"] == "treino"]
    validacao = df[df["conjunto"] == "validacao"]

    tr_par, reg_tr = parear(treino, semente)
    va_par, reg_va = parear(validacao, semente)
    print(f"treino pareado   : {len(tr_par)} ({len(tr_par)//2} por classe)")
    print(f"validação pareada: {len(va_par)} ({len(va_par)//2} por classe)")

    # sanity: no pareado, prop_fala não pode mais separar as classes
    m_bona = tr_par.loc[tr_par["classe_binaria"] == 0, "prop_fala"].mean()
    m_spoof = tr_par.loc[tr_par["classe_binaria"] == 1, "prop_fala"].mean()
    print(f"prop_fala média no treino pareado — bonafide {m_bona:.4f} | "
          f"spoof {m_spoof:.4f} (devem ser ~iguais)")

    X_tr, y_tr = tr_par[cols].values, tr_par["classe_binaria"].values
    X_va, y_va = va_par[cols].values, va_par["classe_binaria"].values

    # ---- (a) RF pareado ------------------------------------------------------
    rf_par = RandomForestClassifier(
        n_estimators=100, max_depth=None, class_weight="balanced",
        random_state=semente, n_jobs=-1,
    )
    rf_par.fit(X_tr, y_tr)
    m_par = avaliar("rf_pareado (treinado e avaliado no pareado)",
                    y_va, rf_par.predict(X_va), rf_par.predict_proba(X_va)[:, 1])

    # ---- (b) RF baseline eval no conjunto pareado ------------------------------
    rf_base = joblib.load(RAIZ / "models" / "rf_baseline_eval.joblib")
    m_base_par = avaliar("rf_baseline_eval (avaliado no pareado)",
                         y_va, rf_base.predict(X_va),
                         rf_base.predict_proba(X_va)[:, 1])

    # referência: o baseline no seu próprio protocolo (validação completa)
    with open(RAIZ / "results" / "metricas" / "rf_baseline_eval.json",
              encoding="utf-8") as f:
        base_json = json.load(f)
    ref = {"f1_macro": base_json["f1_macro"], "eer": base_json["eer"]}

    # ---- Interpretação automática (com os números no texto) -------------------
    delta_eer = m_base_par["eer"] - ref["eer"]
    if delta_eer > 0.10:
        leitura = (
            "DESEMPENHO CAIU MUITO no pareado: parte relevante do acerto do "
            "baseline vinha do atalho de silêncio (fração de padding), não de "
            "artefatos de síntese. Achado forte para a discussão do TC II — o "
            "mascaramento de padding na re-extração ganha prioridade."
        )
    elif delta_eer > 0.03:
        leitura = (
            "QUEDA MODERADA no pareado: o atalho de silêncio contribui para o "
            "desempenho, mas não o explica sozinho — há sinal acústico genuíno "
            "nas features. Reportar os dois números e considerar o mascaramento."
        )
    else:
        leitura = (
            "DESEMPENHO SE MANTEVE no pareado: as features carregam sinal "
            "acústico genuíno e prop_fala é um confundidor separável. Bom "
            "resultado — defende o trabalho na banca."
        )

    resultado = {
        "protocolo": {
            "n_bins": N_BINS,
            "pareamento": "mesmo n de bonafide e spoof por faixa de prop_fala, "
                          "em treino e validação; faixas sem as duas classes "
                          "são descartadas",
            "semente": semente,
            "hiperparametros": "os mesmos do baseline (100 árvores, "
                               "max_depth=None, class_weight='balanced')",
            "n_treino_pareado": int(len(tr_par)),
            "n_validacao_pareada": int(len(va_par)),
            "prop_fala_media_treino_pareado": {
                "bonafide": round(float(m_bona), 4),
                "spoof": round(float(m_spoof), 4),
            },
        },
        "n_por_faixa_treino": reg_tr,
        "n_por_faixa_validacao": reg_va,
        "rf_pareado": m_par,
        "rf_baseline_eval_no_pareado": m_base_par,
        "rf_baseline_eval_referencia_validacao_completa": ref,
        "delta_eer_baseline_pareado_vs_completo": round(float(delta_eer), 4),
        "interpretacao": leitura,
        "nota": "No conjunto pareado as classes são 1:1 por construção; o "
                "f1_macro não é comparável diretamente ao da validação completa "
                "(9:1). A comparação honesta entre protocolos é pelo EER, que "
                "não depende de limiar nem de prevalência.",
    }

    dir_met = RAIZ / "results" / "metricas"
    dir_met.mkdir(parents=True, exist_ok=True)
    with open(dir_met / "rf_pareado_propfala.json", "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("RESULTADOS (validação pareada, 1:1 por faixa de prop_fala)")
    print("=" * 70)
    print(f"RF pareado           : f1_macro {m_par['f1_macro']:.4f} | "
          f"EER {m_par['eer']:.4f}")
    print(f"RF baseline (pareado): f1_macro {m_base_par['f1_macro']:.4f} | "
          f"EER {m_base_par['eer']:.4f}")
    print(f"RF baseline (val completa, referência): f1_macro {ref['f1_macro']:.4f} "
          f"| EER {ref['eer']:.4f}")
    print(f"delta EER (pareado - completo): {delta_eer:+.4f}")
    print(f"\nLEITURA: {leitura}")
    print(f"\nJSON: {dir_met / 'rf_pareado_propfala.json'}")


if __name__ == "__main__":
    main()
