"""
Composição do dataset no universo `eval` (148.176)
==================================================

SÓ LÊ E REPORTA. Não altera pipeline, não treina, não toca no teste.

MOTIVAÇÃO:
    O orientador aprovou filtrar o experimento para fase == 'eval' (o conjunto
    oficialmente pontuado do ASVspoof 2021 LA), excluindo 'progress' e 'hidden'.
    Todas as composições existentes (composicao_*.csv) foram calculadas sobre o
    total de 181.566; este script regera as tabelas no recorte aprovado — e
    acrescenta a composição POR ATAQUE, pedida nominalmente pelo orientador
    (o diagnostico_por_ataque.csv existente é DESEMPENHO por ataque, não
    composição).

VERIFICAÇÕES OBRIGATÓRIAS (o script FALHA se alguma não passar):
    1. total do recorte eval == 148.176;
    2. trim == 'notrim' em 100% das linhas (prova de que a exclusão do
       'hidden', que é only_speech, funcionou);
    3. o merge features × labels não perde linhas (mesmo padrão de
       carregar_dados_split) — features.csv NÃO contém codec/ataque/trim/fase,
       então qualquer diagnóstico que cruze features com metadados depende
       deste merge estar íntegro.

SAÍDAS (results/metricas/):
    eval_composicao_classe.csv          contagem e % de bonafide/spoof
    eval_composicao_codec.csv           contagem por codec
    eval_composicao_ataque.csv          contagem por ataque (A07–A19 + bonafide)
    eval_composicao_codec_x_ataque.csv  tabela cruzada (insumo da subamostra 30k)
    eval_composicao_resumo.json         totais, razão spoof:bonafide, checagens

Rode a partir da raiz:  python -m scripts.composicao_eval
"""

import json
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
DIR_MET = RAIZ / "results" / "metricas"

N_EVAL_ESPERADO = 148_176


def tabela_simples(df: pd.DataFrame, coluna: str) -> pd.DataFrame:
    """Contagem + percentual de uma coluna categórica, ordenada por contagem."""
    t = df[coluna].value_counts().rename("n").to_frame()
    t["pct"] = (100 * t["n"] / len(df)).round(2)
    return t.reset_index()


def main() -> None:
    labels = pd.read_csv(RAIZ / "data" / "processed" / "labels.csv")
    DIR_MET.mkdir(parents=True, exist_ok=True)

    eval_ = labels[labels["fase"] == "eval"].copy()
    total = len(eval_)

    print("=" * 70)
    print(f"COMPOSIÇÃO DO UNIVERSO eval — {total} de {len(labels)} linhas")
    print("=" * 70)

    # ---- Verificação 1: tamanho exato do recorte ---------------------------
    if total != N_EVAL_ESPERADO:
        raise SystemExit(
            f"FALHA: recorte fase=='eval' tem {total} linhas; esperado "
            f"{N_EVAL_ESPERADO}. O labels.csv mudou? Parar e reportar."
        )

    # ---- Verificação 2: trim uniformemente 'notrim' ------------------------
    trim_counts = eval_["trim"].value_counts().to_dict()
    if trim_counts != {"notrim": total}:
        raise SystemExit(
            f"FALHA: trim não é uniformemente 'notrim' no eval: {trim_counts}. "
            "A exclusão do 'hidden' não funcionou como esperado."
        )
    print(f"trim == 'notrim' em 100% das {total} linhas: OK")

    # ---- Verificação 3: cobertura do features.csv --------------------------
    # features.csv não tem codec/ataque/trim/fase; todo cruzamento depende do
    # merge por 'arquivo'. Conferimos aqui que o eval inteiro está coberto.
    feats = pd.read_csv(RAIZ / "data" / "features" / "features.csv",
                        usecols=["arquivo"])
    m = eval_.merge(feats, on="arquivo", how="inner")
    if len(m) != total:
        raise SystemExit(
            f"FALHA: merge eval × features.csv resultou em {len(m)} linhas "
            f"(esperado {total}). Há áudios do eval sem features extraídas."
        )
    print(f"merge com features.csv preserva as {total} linhas: OK")

    # ---- Composição por classe ---------------------------------------------
    t_classe = tabela_simples(eval_, "label")
    t_classe.to_csv(DIR_MET / "eval_composicao_classe.csv", index=False)
    print("\n--- classe ---")
    print(t_classe.to_string(index=False))

    # ---- Composição por codec ----------------------------------------------
    t_codec = tabela_simples(eval_, "codec")
    t_codec.to_csv(DIR_MET / "eval_composicao_codec.csv", index=False)
    print("\n--- codec ---")
    print(t_codec.to_string(index=False))

    # ---- Composição por ataque (pedida nominalmente pelo orientador) --------
    # Para bonafide o campo 'ataque' é '-': não é um estrato vazio, é o próprio
    # estrato bonafide. Renomeamos para 'bonafide' para a tabela ser legível.
    eval_["ataque_leg"] = eval_["ataque"].where(eval_["ataque"] != "-", "bonafide")
    t_ataque = tabela_simples(eval_, "ataque_leg").rename(
        columns={"ataque_leg": "ataque"})
    t_ataque = t_ataque.sort_values("ataque").reset_index(drop=True)
    t_ataque.to_csv(DIR_MET / "eval_composicao_ataque.csv", index=False)
    print("\n--- ataque (A07–A19 + bonafide) ---")
    print(t_ataque.to_string(index=False))

    # ---- Cruzamento codec × ataque (insumo da subamostra 30k) ---------------
    cruzada = pd.crosstab(eval_["codec"], eval_["ataque_leg"], margins=True)
    cruzada.to_csv(DIR_MET / "eval_composicao_codec_x_ataque.csv")
    print("\n--- codec × ataque ---")
    print(cruzada.to_string())

    # ---- Resumo em JSON ------------------------------------------------------
    n_bona = int((eval_["classe_binaria"] == 0).sum())
    n_spoof = int((eval_["classe_binaria"] == 1).sum())
    resumo = {
        "universo": "fase == 'eval' (ASVspoof 2021 LA, conjunto pontuado)",
        "total": total,
        "bonafide": n_bona,
        "spoof": n_spoof,
        "razao_spoof_bonafide": round(n_spoof / n_bona, 4),
        "n_codecs": int(eval_["codec"].nunique()),
        "n_ataques": int((eval_["ataque"] != "-").sum() and
                         eval_.loc[eval_["ataque"] != "-", "ataque"].nunique()),
        "trim_uniforme_notrim": trim_counts == {"notrim": total},
        "trim_contagens": trim_counts,
        "excluidos": {
            "progress": int((labels["fase"] == "progress").sum()),
            "hidden": int((labels["fase"] == "hidden").sum()),
        },
        "features_csv_cobre_eval_completo": True,
    }
    with open(DIR_MET / "eval_composicao_resumo.json", "w", encoding="utf-8") as f:
        json.dump(resumo, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("VERIFICAÇÕES: todas passaram.")
    print(f"Total eval = {total} | razão spoof:bonafide = "
          f"{resumo['razao_spoof_bonafide']}:1")
    print(f"CSVs e JSON salvos em {DIR_MET}")


if __name__ == "__main__":
    main()
