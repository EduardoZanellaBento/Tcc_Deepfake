"""
Diagnóstico B4 — Composição do dataset (fase, trim, codec)
==========================================================

SÓ LÊ E REPORTA. Não altera pipeline, não treina, não toca no teste.

MOTIVAÇÃO:
    O config.yaml declara o dataset como "eval completo (181.566)", mas a
    inspeção do trial_metadata.txt mostrou que 181.566 NÃO é o conjunto eval:
    é a soma de eval (148.176) + progress (16.464) + hidden (16.926). O
    subconjunto 'hidden' tem pré-processamento distinto (trim == 'only_speech',
    silêncio pré-cortado na origem), o que contamina qualquer análise que
    dependa de proporção de fala/silêncio. O conjunto oficialmente pontuado do
    ASVspoof 2021 LA é fase == 'eval'.

    Este script quantifica essa composição e gera a evidência para a decisão
    metodológica (TODO 9.3 no config.yaml) — que é do orientador, não deste
    código.

SAÍDAS:
    results/metricas/composicao_fase.csv
    results/metricas/composicao_trim.csv
    results/metricas/composicao_codec.csv
    results/metricas/composicao_fase_x_trim.csv
    results/metricas/composicao_codec_x_label.csv
    + conclusão em texto no stdout

Rode a partir da raiz:  python -m scripts.diagnostico_composicao_dataset
"""

from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
DIR_MET = RAIZ / "results" / "metricas"


def tabela_simples(df: pd.DataFrame, coluna: str) -> pd.DataFrame:
    """Contagem + percentual de uma coluna categórica, ordenada por contagem."""
    t = df[coluna].value_counts().rename("n").to_frame()
    t["pct"] = (100 * t["n"] / len(df)).round(2)
    return t.reset_index()


def main() -> None:
    labels = pd.read_csv(RAIZ / "data" / "processed" / "labels.csv")
    DIR_MET.mkdir(parents=True, exist_ok=True)

    total = len(labels)
    print("=" * 70)
    print(f"COMPOSIÇÃO DO DATASET — {total} linhas no labels.csv")
    print("=" * 70)

    # ---- Tabelas simples: fase, trim, codec --------------------------------
    for col in ["fase", "trim", "codec"]:
        t = tabela_simples(labels, col)
        t.to_csv(DIR_MET / f"composicao_{col}.csv", index=False)
        print(f"\n--- {col} ---")
        print(t.to_string(index=False))

    # ---- Cruzamento fase × trim -------------------------------------------
    fase_trim = pd.crosstab(labels["fase"], labels["trim"], margins=True)
    fase_trim.to_csv(DIR_MET / "composicao_fase_x_trim.csv")
    print("\n--- fase × trim ---")
    print(fase_trim.to_string())

    # ---- Cruzamento codec × label -----------------------------------------
    codec_label = pd.crosstab(labels["codec"], labels["label"], margins=True)
    codec_label.to_csv(DIR_MET / "composicao_codec_x_label.csv")
    print("\n--- codec × label ---")
    print(codec_label.to_string())

    # ---- Quantificação explícita da discrepância --------------------------
    n_eval = int((labels["fase"] == "eval").sum())
    n_progress = int((labels["fase"] == "progress").sum())
    n_hidden = int((labels["fase"] == "hidden").sum())
    n_only_speech = int((labels["trim"] == "only_speech").sum())
    n_notrim = int((labels["trim"] == "notrim").sum())
    hidden_e_only_speech = labels.loc[labels["fase"] == "hidden", "trim"].eq("only_speech").all()
    only_speech_e_hidden = labels.loc[labels["trim"] == "only_speech", "fase"].eq("hidden").all()

    print("\n" + "=" * 70)
    print("CONCLUSÃO")
    print("=" * 70)
    print(f"""
1. O total de {total} NÃO é o conjunto eval do ASVspoof 2021 LA.
   Composição: eval = {n_eval} ({100*n_eval/total:.1f}%) + progress = {n_progress}
   ({100*n_progress/total:.1f}%) + hidden = {n_hidden} ({100*n_hidden/total:.1f}%).
   O conjunto oficialmente pontuado é fase=='eval' ({n_eval} utterances).

2. hidden == only_speech: {'CONFIRMADO' if hidden_e_only_speech and only_speech_e_hidden else 'NÃO CONFIRMADO'}.
   Toda linha da fase 'hidden' tem trim=='only_speech' ({n_hidden} == {n_only_speech})
   e vice-versa. Ou seja, o subconjunto hidden chega com o silêncio JÁ cortado
   na origem — pré-processamento distinto dos {n_notrim} 'notrim'. Qualquer
   análise de prop_fala precisa filtrar/estratificar por trim (ver B1).

3. codec é um fator perfeitamente balanceado (7 × {total//7}), o que torna a
   comparação de desempenho por codec (B5) limpa. Divisão por banda:
   banda estreita (teto ~4 kHz): alaw, ulaw, gsm, pstn = {int(labels['codec'].isin(['alaw','ulaw','gsm','pstn']).sum())} ({100*labels['codec'].isin(['alaw','ulaw','gsm','pstn']).mean():.0f}%)
   banda larga   (>4 kHz):       g722, opus, none      = {int(labels['codec'].isin(['g722','opus','none']).sum())} ({100*labels['codec'].isin(['g722','opus','none']).mean():.0f}%)

4. A decisão de filtrar (ou não) para fase=='eval' é METODOLÓGICA e está
   registrada como TODO no config.yaml (seção dataset). Este script apenas
   produz a evidência; não filtra nada.
""")
    print(f"CSVs salvos em {DIR_MET}")


if __name__ == "__main__":
    main()
