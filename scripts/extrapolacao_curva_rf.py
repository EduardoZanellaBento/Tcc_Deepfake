"""
Extrapolação da curva de aprendizado do RF (R6.1)
==================================================

A PERGUNTA:
    A curva de aprendizado do RF ajustado NÃO satura: o último passo medido
    (80.000 -> 103.723) ainda rende +0,0121 em f1_macro, mais que o dobro da
    tolerância de 0,005 (ver curva_aprendizado_rf_tuned_eval.json, campos
    `saturou` e `ganho_ultimo_passo_f1`). "Não saturou" soa como fraqueza da
    análise. A pergunta que transforma isso em RESULTADO é quantitativa:
    quantos dados o RF precisaria para alcançar o f1_macro que o SVM já atinge
    com 30 mil?

O MÉTODO:
    Curvas de aprendizado são aproximadamente lineares em log(n) na faixa
    intermediária. Ajusta-se, por mínimos quadrados,

        f1_macro = a * ln(n) + b

    sobre os 7 pontos medidos, e resolve-se para o f1_macro do SVM:

        n* = exp((f1_svm - b) / a)

    O R² do ajuste é reportado — é ele que diz se a extrapolação tem alguma
    base. Um R² alto NÃO valida a extrapolação fora da faixa medida; apenas
    mostra que dentro dela a forma log-linear descreve bem os dados.

NADA É HARDCODADO:
    o f1_macro do SVM é lido de svm_tuned_principal.json, os pontos vêm de
    curva_aprendizado_rf_tuned_eval.csv, e o tamanho do treino completo e do
    universo eval saem do split.csv. Se qualquer um desses artefatos mudar, o
    número muda junto — é a mesma disciplina de todo o repositório: script ->
    JSON auditável, nunca número copiado à mão para o texto.

A RESSALVA (gravada DENTRO do JSON, campo `limitacao`):
    isto é uma EXTRAPOLAÇÃO log-linear FORA da faixa medida, não uma medição.
    Curvas de aprendizado costumam ACHATAR à medida que n cresce — o que
    significa que o n estimado aqui é um LIMITE OTIMISTA para o RF: na prática
    ele provavelmente precisaria de MAIS dados que isso, não menos. Apresentar
    um número extrapolado como se fosse medido é exatamente o tipo de coisa que
    a banca pega; por isso a ressalva viaja junto com o número, dentro do
    próprio artefato.

SAÍDA:
    results/metricas/extrapolacao_curva_rf.json

Rode a partir da raiz:  python -m scripts.extrapolacao_curva_rf
"""

import json
import platform
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.serializacao import json_seguro

RAIZ = Path(__file__).resolve().parents[1]
DIR_MET = RAIZ / "results" / "metricas"


def carregar_insumos() -> tuple[pd.DataFrame, float, int, int]:
    """Pontos da curva, f1_macro do SVM e os dois tamanhos de referência."""
    curva = pd.read_csv(DIR_MET / "curva_aprendizado_rf_tuned_eval.csv")

    with open(DIR_MET / "svm_tuned_principal.json", encoding="utf-8") as f:
        svm = json.load(f)
    f1_svm = float(svm["f1_macro"])          # NUNCA hardcodado

    split = pd.read_csv(RAIZ / "data" / "processed" / "split.csv")
    n_treino_completo = int((split["conjunto"] == "treino").sum())
    n_universo = int(len(split))
    return curva, f1_svm, n_treino_completo, n_universo


def ajustar_log(n: np.ndarray, f1: np.ndarray) -> tuple[float, float, float]:
    """Mínimos quadrados de f1 ~ a*ln(n) + b. Devolve (a, b, R2)."""
    x = np.log(n.astype(float))
    a, b = np.polyfit(x, f1, 1)
    previsto = a * x + b
    ss_res = float(np.sum((f1 - previsto) ** 2))
    ss_tot = float(np.sum((f1 - f1.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    return float(a), float(b), float(r2)


def main() -> None:
    curva, f1_svm, n_treino_completo, n_universo = carregar_insumos()
    n = curva["n_treino"].values
    f1 = curva["f1_macro"].values
    a, b, r2 = ajustar_log(n, f1)

    f1_rf_maximo = float(f1.max())
    n_necessario = float(np.exp((f1_svm - b) / a))
    razao_treino = n_necessario / n_treino_completo
    razao_universo = n_necessario / n_universo
    cabe_no_universo = bool(n_necessario <= n_universo)

    print("=" * 74)
    print("EXTRAPOLAÇÃO DA CURVA DE APRENDIZADO DO RF (R6.1)")
    print("=" * 74)
    print(f"pontos medidos     : {len(n)}  (n de {int(n.min())} a {int(n.max())})")
    print(f"ajuste             : f1_macro = {a:.4f} * ln(n) + {b:.4f}")
    print(f"R2                 : {r2:.4f}")
    print(f"f1_macro do RF no treino completo ({n_treino_completo}): {f1_rf_maximo:.4f}")
    print(f"f1_macro do SVM (30k, braço principal)                 : {f1_svm:.4f}")
    print(f"lacuna a fechar                                        : "
          f"{f1_svm - f1_rf_maximo:+.4f}")
    print(f"\nn necessário para o RF alcançar o SVM: {n_necessario:,.0f} áudios")
    print(f"  = {razao_treino:.2f}x o treino completo ({n_treino_completo:,})")
    print(f"  = {razao_universo:.2f}x o universo eval inteiro ({n_universo:,})")
    print(f"  cabe no universo disponível? {'SIM' if cabe_no_universo else 'NÃO'}")

    if cabe_no_universo:
        leitura = (
            f"Pela extrapolação log-linear, o RF alcançaria o f1_macro do SVM "
            f"com cerca de {n_necessario:,.0f} áudios de treino "
            f"({razao_treino:.2f}x o treino completo), volume que AINDA CABE no "
            f"universo eval disponível ({n_universo:,}). O custo da subamostra "
            "seria, portanto, recuperável com mais dados — dentro dos limites "
            "da ressalva log-linear abaixo.")
    else:
        leitura = (
            f"Pela extrapolação log-linear, o RF só alcançaria o f1_macro do SVM "
            f"com cerca de {n_necessario:,.0f} áudios de treino — "
            f"{razao_treino:.2f}x o treino completo ({n_treino_completo:,}) e "
            f"{razao_universo:.2f}x o UNIVERSO EVAL INTEIRO ({n_universo:,}), "
            "que é tudo o que existe neste conjunto. Isto converte 'o RF não "
            "saturou' de fraqueza da análise em achado quantificado: DENTRO DOS "
            "DADOS DISPONÍVEIS, o RF não alcança o SVM nem usando tudo. A "
            "vantagem do SVM não é um artefato do tamanho do treino do braço "
            "principal.")
    print(f"\n{leitura}")

    limitacao = (
        "ESTE NÚMERO É UMA EXTRAPOLAÇÃO, NÃO UMA MEDIÇÃO. O ajuste log-linear "
        f"foi feito sobre a faixa MEDIDA ({int(n.min()):,} a {int(n.max()):,} "
        f"amostras) e a estimativa de n* = {n_necessario:,.0f} cai FORA dessa "
        "faixa. Curvas de aprendizado tipicamente ACHATAM conforme n cresce, "
        "logo a forma log-linear tende a SUPERESTIMAR o ganho por dado adicional "
        "na cauda: n* deve ser lido como um LIMITE OTIMISTA para o RF — na "
        f"prática ele precisaria provavelmente de MAIS que {n_necessario:,.0f} "
        "áudios, não menos. O R² alto atesta o ajuste DENTRO da faixa medida; "
        "não valida a extrapolação fora dela. No texto do TC II este número deve "
        "ser apresentado como ordem de grandeza estimada, jamais como medida.")
    print(f"\nRESSALVA: {limitacao}\n")

    registro = {
        "analise": "extrapolacao_curva_rf",
        "data": date.today().isoformat(),
        "pergunta": ("quantos dados de treino o RF ajustado precisaria para "
                     "alcançar o f1_macro que o SVM atinge com a subamostra de "
                     "30k?"),
        "insumos": {
            "curva": "results/metricas/curva_aprendizado_rf_tuned_eval.csv",
            "f1_macro_svm": {"valor": f1_svm,
                             "fonte": "results/metricas/svm_tuned_principal.json"},
            "n_treino_completo": n_treino_completo,
            "n_universo_eval": n_universo,
        },
        "modelo_de_ajuste": "f1_macro = a * ln(n) + b (mínimos quadrados)",
        "coeficientes": {"a": round(a, 4), "b": round(b, 4), "r2": round(r2, 4)},
        "pontos_usados": [{"n_treino": int(x), "f1_macro": float(y)}
                          for x, y in zip(n, f1)],
        "faixa_medida": {"n_min": int(n.min()), "n_max": int(n.max())},
        "f1_macro_rf_no_treino_completo": f1_rf_maximo,
        "lacuna_f1_rf_para_svm": round(f1_svm - f1_rf_maximo, 4),
        "n_necessario": round(n_necessario, 0),
        "razao_sobre_treino_completo": round(razao_treino, 2),
        "razao_sobre_universo_eval": round(razao_universo, 2),
        "cabe_no_universo_disponivel": cabe_no_universo,
        "leitura": leitura,
        "limitacao": limitacao,
        "ambiente": {
            "python": platform.python_version(),
            "sistema": f"{platform.system()} {platform.release()}",
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    caminho = DIR_MET / "extrapolacao_curva_rf.json"
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(registro, f, indent=2, ensure_ascii=False,
                  default=json_seguro)
    print(f"Salvo em {caminho.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
