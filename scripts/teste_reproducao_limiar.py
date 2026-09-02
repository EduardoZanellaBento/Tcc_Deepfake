"""
Teste de reprodução — a nova `avaliar` tem de reproduzir a divergência conhecida
================================================================================

SÓ LÊ E CONFERE. Nada é treinado nem sobrescrito.

O QUE ESTÁ SENDO TESTADO:
    O Bloco 3 trocou a assinatura de `avaliar` (src/models/avaliacao.py): ela
    agora recebe `limiar` e deriva o y_pred internamente pela regra única
    `score >= limiar`. Antes de qualquer modelo novo usar essa função, ela
    precisa reproduzir números JÁ PUBLICADOS, medidos pela mesma regra:

    - nota_divergencia_f1.md: no modelo rf_baseline.joblib (universo 181k,
      validação de 27.235 do split_181k.csv), `scores >= 0,50` dá f1_macro
      0,5598 e o `predict()` (argmax) dá ≈0,5675 — divergem em 25 amostras
      empatadas exatamente em 0,50.
    - NOTA_LIMIAR.md (tabela da varredura): limiar 0,80 -> f1_macro 0,7730.

    Se avaliar(limiar=0,50) não devolver 0,5598, a implementação está errada.

INSUMOS (todos preservados como evidência histórica — nenhum é regerado):
    models/rf_baseline.joblib
    data/features/features_pre_mascaramento_c01c3c5c.csv  (features ANTIGAS —
        o modelo foi treinado nelas; usar o features.csv congelado aqui seria
        avaliar o modelo em outra definição de feature)
    data/processed/split_181k.csv

Rode a partir da raiz:  python -m scripts.teste_reproducao_limiar
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.models.avaliacao import (REGRA_DECISAO, aplicar_limiar, avaliar,
                                  selecionar_limiar)

RAIZ = Path(__file__).resolve().parents[1]

# (valor_esperado, tolerância) — os alvos vêm de artefatos publicados, com 4
# casas decimais; a tolerância cobre só o arredondamento da publicação.
ESPERADOS = {
    "f1_macro em >= 0,50 (nota_divergencia_f1.md)": (0.5598, 5e-5),
    "f1_macro do predict/argmax (~0,5675 do rf_baseline.json)": (0.5675, 2e-4),
    "f1_macro em >= 0,80 (NOTA_LIMIAR.md)": (0.7730, 5e-5),
}


def conferir(rotulo: str, medido: float) -> bool:
    esperado, tol = ESPERADOS[rotulo]
    ok = abs(medido - esperado) <= tol
    status = "OK " if ok else "FALHOU"
    print(f"  [{status}] {rotulo}: medido {medido:.4f}, esperado {esperado:.4f}")
    return ok


def main() -> None:
    modelo = joblib.load(RAIZ / "models" / "rf_baseline.joblib")
    feats = pd.read_csv(
        RAIZ / "data" / "features" / "features_pre_mascaramento_c01c3c5c.csv")
    split = pd.read_csv(RAIZ / "data" / "processed" / "split_181k.csv")
    df = feats.merge(split, on="arquivo", how="inner")
    validacao = df[df["conjunto"] == "validacao"]
    print(f"validação histórica (universo 181k): {len(validacao)} amostras "
          f"(esperado: 27.235)")

    # As colunas do X na ordem do CSV antigo, exatamente como o modelo foi
    # treinado (48 colunas: arquivo, label, classe_binaria, prop_fala + 44).
    nao_x = {"arquivo", "label", "classe_binaria", "conjunto", "prop_fala"}
    cols = [c for c in feats.columns if c not in nao_x]
    assert len(cols) == 44, f"esperava 44 features, achei {len(cols)}"

    y_va = validacao["classe_binaria"].values
    scores = modelo.predict_proba(validacao[cols].values)[:, 1]

    print(f"\nregra em teste: {REGRA_DECISAO}")
    resultados = []

    m_050 = avaliar(y_va, scores, "teste_reproducao", limiar=0.50)
    resultados.append(conferir(
        "f1_macro em >= 0,50 (nota_divergencia_f1.md)", m_050["f1_macro"]))

    f1_argmax = f1_score(y_va, modelo.predict(validacao[cols].values),
                         average="macro", zero_division=0)
    resultados.append(conferir(
        "f1_macro do predict/argmax (~0,5675 do rf_baseline.json)", f1_argmax))

    m_080 = avaliar(y_va, scores, "teste_reproducao", limiar=0.80)
    resultados.append(conferir(
        "f1_macro em >= 0,80 (NOTA_LIMIAR.md)", m_080["f1_macro"]))

    n_div = int(np.sum(aplicar_limiar(scores, 0.50)
                       != modelo.predict(validacao[cols].values)))
    print(f"  [INFO] divergência >= vs argmax: {n_div} amostras "
          f"(nota_divergencia_f1.md documenta 25)")

    sel = selecionar_limiar(y_va, scores)
    print(f"  [INFO] selecionar_limiar: limiar {sel['limiar']:.4f}, "
          f"f1_macro {sel['f1_macro']:.4f}, {sel['n_candidatos']} candidatos, "
          f"{sel['n_empates_no_maximo']} empate(s) no máximo")
    # O ótimo sobre np.unique(scores) nunca pode ser pior que o melhor ponto da
    # grade de 0,05 (0,7730 em 0,80) — é a razão de ser da decisão de projeto 1.
    assert sel["f1_macro"] >= 0.7730 - 5e-5, \
        "ótimo sobre np.unique ficou abaixo do melhor ponto da grade antiga"

    print()
    if all(resultados):
        print("TESTE DE REPRODUÇÃO: PASSOU — a nova avaliar mede a mesma coisa "
              "que a varredura publicada.")
    else:
        raise SystemExit("TESTE DE REPRODUÇÃO: FALHOU — NÃO seguir para o "
                         "B3.2 até entender a diferença.")


if __name__ == "__main__":
    main()
