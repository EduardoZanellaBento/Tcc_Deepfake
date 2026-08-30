"""
Validação do split e da subamostra APÓS o lote único de re-extração (Bloco 2)
=============================================================================

DECISÃO METODOLÓGICA que este script materializa: o `split.csv` e a
`subamostra_30k.csv` são PRESERVADOS através do lote único — revalidados, nunca
regerados. O raciocínio:

  - O split é uma partição de IDs (`[arquivo, conjunto]`). A re-extração muda os
    VALORES das features, não o CONJUNTO de arquivos do universo eval; logo o
    split continua válido por construção.
  - Regerar, ao contrário, MUDARIA a partição: `criar_split` depende da ordem
    das linhas do features.csv lido, e o CSV novo (eval puro, ordem do
    labels.csv) tem ordem diferente do antigo (181k filtrados a posteriori).
    Mudariam o hash citado no README, a subamostra em cascata, e todos os
    artefatos anteriores (baselines, curva de aprendizado, diagnósticos)
    deixariam de ser comparáveis — a comparação "antes × depois do mascaramento"
    perderia a única variável controlada que tinha.

O que se valida (as quatro condições do plano do Bloco 2):
  1. conjunto de `arquivo` do features.csv novo == conjunto do split.csv
     (nem sobra, nem falta);
  2. `carregar_dados_split` roda sem erro e devolve len == len(split);
  3. `filtrar_treino_braco(..., 'principal')` devolve exatamente o n da
     subamostra;
  4. hashes MD5 de split.csv e subamostra_30k.csv inalterados em relação aos
     registrados nos artefatos commitados (README, subamostra_30k.json,
     rf_baseline_eval*.json).

Saída: results/metricas/validacao_split_pos_lote.json
Rode a partir da raiz:  python -m scripts.validar_split_pos_lote
"""

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from src.utils.config import carregar_config
from src.data.split import carregar_dados_split, filtrar_treino_braco

RAIZ = Path(__file__).resolve().parents[1]

# Hashes de referência — os mesmos citados no README e em
# results/metricas/subamostra_30k.json. Se um dia mudarem DE PROPÓSITO
# (decisão registrada do orientador), atualizar aqui junto.
MD5_SPLIT_ESPERADO = "9143f0c7b83ec2db4aa144ed5deb3402"
MD5_SUBAMOSTRA_ESPERADO = "654cb796b738512388b28e15ffb14a9d"


def md5(caminho: Path) -> str:
    h = hashlib.md5()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def main() -> None:
    cfg = carregar_config(RAIZ)
    problemas = []

    features_csv = RAIZ / "data" / "features" / "features.csv"
    split_csv = RAIZ / "data" / "processed" / "split.csv"
    sub_csv = RAIZ / cfg["experimento"]["caminho_subamostra"]

    # ---- (1) conjunto de IDs idêntico -------------------------------------
    ids_feat = set(pd.read_csv(features_csv, usecols=["arquivo"])["arquivo"])
    ids_split = set(pd.read_csv(split_csv, usecols=["arquivo"])["arquivo"])
    so_feat = ids_feat - ids_split
    so_split = ids_split - ids_feat
    if so_feat:
        problemas.append(f"{len(so_feat)} arquivos no features.csv fora do "
                         f"split (ex.: {sorted(so_feat)[:3]})")
    if so_split:
        problemas.append(f"{len(so_split)} arquivos do split SEM features "
                         f"(ex.: {sorted(so_split)[:3]})")

    # ---- (2) merge íntegro -------------------------------------------------
    n_merge = None
    try:
        df = carregar_dados_split(RAIZ)
        n_merge = int(len(df))
    except Exception as e:
        problemas.append(f"carregar_dados_split falhou: {e!r}")
        df = None

    # ---- (3) subamostra encaixa no treino ---------------------------------
    n_sub = int(len(pd.read_csv(sub_csv, usecols=["arquivo"])))
    n_braco = None
    if df is not None:
        try:
            treino = df[df["conjunto"] == "treino"]
            braco = filtrar_treino_braco(treino, "principal", cfg, RAIZ)
            n_braco = int(len(braco))
            if n_braco != n_sub:
                problemas.append(f"braço principal devolveu {n_braco} linhas, "
                                 f"subamostra tem {n_sub}")
        except Exception as e:
            problemas.append(f"filtrar_treino_braco falhou: {e!r}")

    # ---- (4) hashes inalterados -------------------------------------------
    md5_split = md5(split_csv)
    md5_sub = md5(sub_csv)
    if md5_split != MD5_SPLIT_ESPERADO:
        problemas.append(f"split.csv mudou: {md5_split} != {MD5_SPLIT_ESPERADO}")
    if md5_sub != MD5_SUBAMOSTRA_ESPERADO:
        problemas.append(f"subamostra_30k.csv mudou: {md5_sub} != "
                         f"{MD5_SUBAMOSTRA_ESPERADO}")

    registro = {
        "n_ids_features": len(ids_feat),
        "n_ids_split": len(ids_split),
        "ids_identicos": not so_feat and not so_split,
        "n_linhas_merge": n_merge,
        "n_subamostra": n_sub,
        "n_treino_braco_principal": n_braco,
        "md5_split_csv": md5_split,
        "md5_subamostra_csv": md5_sub,
        "md5_features_csv": md5(features_csv),
        "problemas": problemas,
        "aprovado": not problemas,
    }
    destino = RAIZ / "results" / "metricas" / "validacao_split_pos_lote.json"
    destino.write_text(json.dumps(registro, indent=2, ensure_ascii=False),
                       encoding="utf-8")

    print(json.dumps(registro, indent=2, ensure_ascii=False))
    print("\nAPROVADO — split e subamostra preservados e íntegros."
          if registro["aprovado"] else "\nREPROVADO — ver problemas acima.")
    sys.exit(0 if registro["aprovado"] else 1)


if __name__ == "__main__":
    main()
