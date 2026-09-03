"""
Diagnóstico por sistema de ataque (A07–A19) — RF e SVM AJUSTADOS
=================================================================

SÓ LÊ E REPORTA. Nenhum modelo é treinado aqui: os dois modelos do braço
principal são CARREGADOS de models/*.joblib e cada um decide com o SEU limiar,
lido de `selecao_limiar.limiar` no JSON companheiro. Avaliação na VALIDAÇÃO.
O split de teste continua LACRADO.

POR QUE ESTA VERSÃO EXISTE (revisão de 03/09/2026):
    A versão de 13/08/2026 (preservada em *_baseline_2026-08-13.csv/png)
    re-treinava um RF *baseline* — n_estimators=100, max_depth=None,
    class_weight='balanced' — no TREINO COMPLETO (103.723) e decidia por
    `modelo.predict()`, isto é, argmax em 0,50. Dois problemas, ambos fatais
    para a leitura:

      1. o modelo diagnosticado NÃO era o modelo do braço principal, então a
         tabela não descrevia nenhum número do README;
      2. em 0,50 o RF diz "spoof" para quase tudo — o resultado foi recall 1,0
         em TODOS os 13 ataques e recall_bonafide entre 0,00 e 0,23. Isso não é
         um diagnóstico por ataque: é a assinatura de um limiar errado. Não
         havia informação nenhuma na tabela.

    Com o limiar do protocolo (selecionado na validação, ~0,65 no RF em
    predict_proba e ~-0,03 no SVM em decision_function) a discriminação por
    ataque aparece de fato.

POR QUE ISTO IMPORTA (não é métrica decorativa):
    O README declara, em "Limitação declarada do split", que o split é
    aleatório POR UTTERANCE — cada ataque aparece em treino E validação —, logo
    o modelo PODE estar decorando a assinatura de cada vocoder. A mitigação
    declarada é justamente esta tabela. A leitura é de mão dupla e as duas
    pontas são citáveis:

      - amplitude GRANDE de EER entre ataques  -> evidência de que o desempenho
        depende do sistema de síntese (o risco declarado);
      - amplitude PEQUENA -> evidência a favor de generalização entre sistemas.

    O que não se pode é seguir sem a medida. Por isso a amplitude sai impressa
    e gravada no JSON, para RF e para SVM.

MÉTRICAS POR ATAQUE:
    Para cada A07–A19: n na validação, recall (fração de spoofs daquele sistema
    detectados SOB O LIMIAR DO PROTOCOLO) e f1 da classe spoof no subconjunto
    {bonafide ∪ ataque} — o f1 de spoof precisa dos bonafide como negativos.
    `eer_vs_bonafide` é calculado no mesmo subconjunto e é INDEPENDENTE DE
    LIMIAR: é ele que ordena os sistemas por dificuldade real.
    Para 'bonafide': recall da classe 0 (especificidade) — linha de referência,
    igual para todos os ataques por construção.

SOBRE A COLUNA `media_prob_spoof`:
    mantida com o mesmo nome da versão baseline, para que as duas tabelas sejam
    comparáveis lado a lado. ATENÇÃO à escala: no RF é probabilidade em [0,1];
    no SVM é a média do decision_function (real, centrado em zero) — NÃO é
    probabilidade. A coluna `escala_score` diz qual é qual em cada linha.

SAÍDAS:
    results/metricas/diagnostico_por_ataque_{rf,svm}_tuned_principal.csv
    results/figuras/diagnostico_por_ataque_{rf,svm}_tuned_principal.png
    results/metricas/diagnostico_por_ataque_resumo.json
    + leitura crítica no stdout

Rode a partir da raiz:  python -m scripts.diagnostico_por_ataque
"""

import json
import platform
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score

from src.utils.config import carregar_config
from src.utils.serializacao import json_seguro
from src.data.split import carregar_dados_split, colunas_features
from src.models.avaliacao import aplicar_limiar, calcular_eer, REGRA_DECISAO
from src.models.modelos_ajustados import (
    MODELOS_PRINCIPAIS, carregar_modelo_ajustado, hashes_congelados, scores_de,
)

RAIZ = Path(__file__).resolve().parents[1]

ESCALA = {"rf": "probabilidade [0,1] (predict_proba)",
          "svm": "decision_function (real, centrado em zero)"}

# Referência de grandeza para a leitura automática: a distância RF x SVM em EER
# agregado é ~0,047 (bootstrap pareado, estabilidade_rf_svm.json). Uma amplitude
# ENTRE ataques bem maior que o dobro disso significa que "qual sistema de
# síntese" pesa mais na dificuldade do que "qual dos dois modelos".
LIMITE_AMPLITUDE_GRANDE = 0.10


def preparar_validacao() -> tuple[pd.DataFrame, list[str]]:
    """Validação com features + rótulo de ataque, e a lista canônica de features.

    `colunas_features(df)` é chamado ANTES do merge com labels.csv, de propósito:
    depois do merge a tabela ganha a coluna 'ataque' e qualquer seleção
    posterior arriscaria mudar a composição/ordem do X. Um modelo carregado que
    recebe X com colunas a mais ou em outra ordem NÃO levanta exceção —
    devolve lixo silenciosamente.
    """
    df = carregar_dados_split(RAIZ)
    cols = colunas_features(df)               # ANTES do merge: só as acústicas

    labels = pd.read_csv(RAIZ / "data" / "processed" / "labels.csv",
                         usecols=["arquivo", "ataque"])
    n_antes = len(df)
    df = df.merge(labels, on="arquivo", how="inner")
    assert len(df) == n_antes, "merge perdeu linhas — investigar"

    # SÓ VALIDAÇÃO. O teste é lacrado (Bloco 5, execução única).
    return df[df["conjunto"] == "validacao"].copy(), cols


def diagnosticar(chave: str, validacao: pd.DataFrame, cols: list[str]) -> dict:
    """Tabela por ataque de UM modelo, sob o limiar do protocolo."""
    carregado = carregar_modelo_ajustado(RAIZ, chave)
    limiar = carregado["limiar"]
    nome = carregado["nome_arquivo"]
    print(f"\n=== {carregado['rotulo']} | limiar {limiar:.4f} "
          f"({REGRA_DECISAO}) | fonte: {carregado['origem_limiar']} ===")

    scores = scores_de(carregado, validacao[cols].values)
    # A regra de decisão é a MESMA de todo o Bloco 3 — nada de modelo.predict().
    va = validacao.assign(score=scores, y_pred=aplicar_limiar(scores, limiar))
    bona = va[va["classe_binaria"] == 0]

    ataques = sorted(a for a in va["ataque"].unique() if a.startswith("A"))
    linhas = []
    for atq in ataques:
        sub = va[va["ataque"] == atq]
        par = pd.concat([bona, sub])          # {bonafide U este ataque}
        eer, _ = calcular_eer(par["classe_binaria"].values, par["score"].values)
        linhas.append({
            "ataque": atq,
            "n_validacao": len(sub),
            "recall": round(recall_score(sub["classe_binaria"], sub["y_pred"],
                                         pos_label=1, zero_division=0), 4),
            "f1_vs_bonafide": round(f1_score(par["classe_binaria"], par["y_pred"],
                                             pos_label=1, zero_division=0), 4),
            "media_prob_spoof": round(float(sub["score"].mean()), 4),
            "eer_vs_bonafide": round(eer, 4),
            "limiar": round(limiar, 6),
            "escala_score": ESCALA[chave],
        })

    recall_bona = float(recall_score(bona["classe_binaria"], bona["y_pred"],
                                     pos_label=0, zero_division=0))
    linhas.append({
        "ataque": "bonafide", "n_validacao": len(bona),
        "recall": round(recall_bona, 4), "f1_vs_bonafide": np.nan,
        "media_prob_spoof": round(float(bona["score"].mean()), 4),
        "eer_vs_bonafide": np.nan, "limiar": round(limiar, 6),
        "escala_score": ESCALA[chave],
    })

    res = pd.DataFrame(linhas)
    dir_met = RAIZ / "results" / "metricas"
    dir_met.mkdir(parents=True, exist_ok=True)
    res.to_csv(dir_met / f"diagnostico_por_ataque_{nome}.csv", index=False)
    print(res.to_string(index=False))

    so_ataques = res[res["ataque"] != "bonafide"]
    pior = so_ataques.loc[so_ataques["eer_vs_bonafide"].idxmax()]
    melhor = so_ataques.loc[so_ataques["eer_vs_bonafide"].idxmin()]
    eer_facil = float(melhor["eer_vs_bonafide"])
    eer_dificil = float(pior["eer_vs_bonafide"])

    _plotar(carregado, so_ataques, recall_bona, limiar)

    return {
        "modelo": nome,
        "rotulo": carregado["rotulo"],
        "limiar": limiar,
        "regra": REGRA_DECISAO,
        "criterio_limiar": carregado["criterio_limiar"],
        "origem_limiar": carregado["origem_limiar"],
        "escala_score": ESCALA[chave],
        "n_validacao": int(len(va)),
        "recall_bonafide": round(recall_bona, 4),
        "ataque_mais_facil": {"ataque": melhor["ataque"],
                              "eer_vs_bonafide": eer_facil},
        "ataque_mais_dificil": {"ataque": pior["ataque"],
                                "eer_vs_bonafide": eer_dificil},
        "amplitude_eer": round(eer_dificil - eer_facil, 4),
        "razao_eer_dificil_facil": (round(eer_dificil / eer_facil, 2)
                                    if eer_facil > 0 else None),
        "eer_por_ataque": {r["ataque"]: r["eer_vs_bonafide"]
                           for r in so_ataques.to_dict("records")},
        "recall_saturado_em_1": bool((so_ataques["recall"] >= 0.999).all()),
        "csv": f"results/metricas/diagnostico_por_ataque_{nome}.csv",
        "figura": f"results/figuras/diagnostico_por_ataque_{nome}.png",
    }


def _plotar(carregado: dict, so_ataques: pd.DataFrame, recall_bona: float,
            limiar: float) -> None:
    nome = carregado["nome_arquivo"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].bar(so_ataques["ataque"], so_ataques["recall"], color="#4c72b0")
    axes[0].axhline(recall_bona, color="gray", ls="--", lw=1,
                    label=f"recall bonafide = {recall_bona:.3f}")
    axes[0].set_ylim(0, 1.02)
    axes[0].set_ylabel(f"recall (limiar {limiar:.4f})")
    axes[0].set_title("recall por ataque — sob o limiar do protocolo")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(so_ataques["ataque"], so_ataques["eer_vs_bonafide"],
                color="#c44e52")
    axes[1].set_ylabel("EER vs. bonafide (menor = melhor)")
    axes[1].set_title("EER por ataque — independente de limiar")
    axes[1].grid(axis="y", alpha=0.3)
    fig.suptitle(f"{carregado['rotulo']} por sistema de ataque — validação, "
                 f"limiar {limiar:.4f}")
    fig.tight_layout()
    dir_fig = RAIZ / "results" / "figuras"
    dir_fig.mkdir(parents=True, exist_ok=True)
    fig.savefig(dir_fig / f"diagnostico_por_ataque_{nome}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    cfg = carregar_config(RAIZ)
    validacao, cols = preparar_validacao()
    print(f"{len(cols)} features | validação {validacao.shape} "
          f"| modelos: {sorted(MODELOS_PRINCIPAIS)}")

    resumos = {c: diagnosticar(c, validacao, cols) for c in ("rf", "svm")}

    # ---- Leitura crítica ----------------------------------------------------
    print("\n" + "=" * 74)
    print("LEITURA CRÍTICA — o desempenho depende do sistema de síntese?")
    print("=" * 74)
    for r in resumos.values():
        saturado = "SIM" if r["recall_saturado_em_1"] else "NÃO"
        print(f"\n{r['rotulo']} (limiar {r['limiar']:.4f}):")
        print(f"  mais fácil   : {r['ataque_mais_facil']['ataque']} — "
              f"EER {r['ataque_mais_facil']['eer_vs_bonafide']:.4f}")
        print(f"  mais difícil : {r['ataque_mais_dificil']['ataque']} — "
              f"EER {r['ataque_mais_dificil']['eer_vs_bonafide']:.4f}")
        print(f"  AMPLITUDE    : {r['amplitude_eer']:.4f} "
              f"(razão {r['razao_eer_dificil_facil']}x)")
        print(f"  recall saturado em 1,0 em todos os ataques? {saturado}")

    amp_rf = resumos["rf"]["amplitude_eer"]
    amp_svm = resumos["svm"]["amplitude_eer"]
    maior = max(amp_rf, amp_svm)
    if maior >= LIMITE_AMPLITUDE_GRANDE:
        veredito = (
            "AMPLITUDE GRANDE: a dificuldade varia muito mais entre sistemas de "
            "síntese (amplitude de EER até "
            f"{maior:.4f}) do que entre os dois modelos (delta EER agregado "
            "0,0466, bootstrap pareado). É evidência A FAVOR do risco declarado "
            "no README: o número agregado depende de QUAIS ataques compõem o "
            "conjunto, e o split por utterance deixa o modelo ver a assinatura "
            "de cada vocoder já no treino. O agregado é, portanto, "
            "potencialmente otimista, e o experimento cross-attack "
            "(leave-one-attack-out) segue sendo a análise complementar "
            "necessária.")
    else:
        veredito = (
            "AMPLITUDE PEQUENA: os 13 sistemas são detectados com dificuldade "
            f"semelhante (amplitude de EER no máximo {maior:.4f}, contra um "
            "delta EER de 0,0466 entre os dois modelos). É evidência A FAVOR de "
            "generalização entre sistemas de síntese e ENFRAQUECE a hipótese de "
            "que o modelo decorou a assinatura de um vocoder específico. A "
            "limitação do split continua declarada — mas agora com uma medida "
            "que a atenua, em vez de uma tabela vazia.")
    print(f"\n{veredito}\n")

    resumo = {
        "analise": "diagnostico_por_ataque",
        "data": date.today().isoformat(),
        "conjunto": "validacao",
        "teste_lacrado": True,
        "regra_decisao": REGRA_DECISAO,
        "observacao_metodo": (
            "nenhum modelo é treinado aqui: os dois são carregados de "
            "models/*.joblib e decidem com o limiar selecionado na validação, "
            "lido do JSON companheiro. A versão de 13/08/2026 re-treinava um RF "
            "baseline e decidia em 0,50 — ver *_baseline_2026-08-13.csv"),
        "limite_amplitude_grande": LIMITE_AMPLITUDE_GRANDE,
        "amplitude_eer_rf": amp_rf,
        "amplitude_eer_svm": amp_svm,
        "leitura_critica": veredito,
        "por_modelo": resumos,
        "referencia_baseline": {
            "csv": "results/metricas/diagnostico_por_ataque_baseline_2026-08-13.csv",
            "nota": ("RF baseline, treino completo, limiar 0,50, features "
                     "PRÉ-mascaramento — mantido como referência 'antes'"),
        },
        "hashes_md5": hashes_congelados(
            RAIZ, cfg["experimento"]["caminho_subamostra"]),
        "ambiente": {
            "python": platform.python_version(),
            "sistema": f"{platform.system()} {platform.release()}",
        },
    }
    caminho = RAIZ / "results" / "metricas" / "diagnostico_por_ataque_resumo.json"
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(resumo, f, indent=2, ensure_ascii=False,
                  default=json_seguro)
    print(f"Resumo salvo em {caminho}")


if __name__ == "__main__":
    main()
