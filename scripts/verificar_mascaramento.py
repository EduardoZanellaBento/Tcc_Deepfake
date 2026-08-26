"""
Checagem do bloco 1 — mascaramento de padding, centróide e esquema do CSV
========================================================================

É a EVIDÊNCIA que o orientador pediu antes de autorizar o lote único de
re-extração. Não encosta em `data/features/features.csv`.

O que este script prova, em duas partes:

PARTE A — o mascaramento está de fato ativo (teste A/B controlado)
    Para uma amostra pequena, extrai o vetor de 44 features DUAS vezes a partir
    do MESMO áudio pré-processado:
        (A) sem máscara  -> extrair_vetor(..., n_amostras_validas=None)
        (B) com máscara  -> extrair_vetor(..., n_amostras_validas=n_validas)
    Como as séries (MFCC/ZCR/centróide) são as mesmas nas duas chamadas, QUALQUER
    diferença vem exclusivamente da agregação. Se A == B, o mascaramento não está
    funcionando. Além disso mede a ASSIMETRIA entre classes (bonafide − spoof),
    que é o que caracteriza o atalho de silêncio: distorção igual nas duas classes
    seria ruído; distorção maior no bonafide é viés.

PARTE B — o runner de produção está consistente
    Roda `executar(..., limite, nome_saida=...)`, isto é, o MESMO código do lote
    único, e confere no CSV gerado:
      - universo: só fase == 'eval';
      - esquema: 3 colunas de identificação + 3 diagnósticas + 44 features;
      - `colunas_features` devolve exatamente 44 e NENHUMA diagnóstica;
      - sem NaN/inf (média/desvio de fatia vazia viraria NaN);
      - 1 <= n_frames_validos <= n_frames_total.

Saídas:
    data/features/features_piloto_mascarado.csv
    results/metricas/checagem_mascaramento.json
    results/metricas/checagem_mascaramento_delta.csv

Rode a partir da raiz:  python -m scripts.verificar_mascaramento
"""

import json
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.config import carregar_config
from src.utils.seeds import fixar_seeds
from src.data.preprocessamento import preprocessar_audio
from src.data.split import colunas_features
from src.features.extrair_features import (COLUNAS_DIAGNOSTICO, executar,
                                           extrair_vetor, nomes_features)

RAIZ = Path(__file__).resolve().parents[1]

N_AB = 200        # parte A: teste controlado com/sem máscara
N_PILOTO = 500    # parte B: passagem pelo runner de produção
NOME_SAIDA = "features_piloto_mascarado.csv"


# ---------------------------------------------------------------------------
# PARTE A — worker: mesmo áudio, duas agregações
# ---------------------------------------------------------------------------
def _ab_um(args: tuple) -> dict:
    arquivo, caminho, classe, cfg = args
    try:
        sr = cfg["audio"]["sample_rate"]
        y, prop, n_validas = preprocessar_audio(caminho, cfg)
        # (A) sem máscara: agrega os 4,0 s inteiros, padding incluído
        va, _, n_tot = extrair_vetor(y, sr, cfg, n_amostras_validas=None)
        # (B) com máscara: agrega só os frames válidos
        vb, n_val, _ = extrair_vetor(y, sr, cfg, n_amostras_validas=n_validas)

        nomes = nomes_features(cfg["features"]["n_mfcc"])
        linha = {"arquivo": arquivo, "classe_binaria": classe,
                 "prop_fala": round(prop, 4),
                 "n_frames_validos": n_val, "n_frames_total": n_tot}
        linha.update({f"{c}__A": float(v) for c, v in zip(nomes, va)})
        linha.update({f"{c}__B": float(v) for c, v in zip(nomes, vb)})
        return linha
    except Exception as e:
        return {"arquivo": arquivo, "erro": repr(e)}


def parte_a(cfg: dict, semente: int) -> dict:
    labels = pd.read_csv(RAIZ / "data" / "processed" / "labels.csv")
    ev = labels[labels["fase"] == cfg["dataset"]["fase"]]

    # metade bonafide, metade spoof: o efeito que interessa é a DIFERENÇA entre
    # as classes, então aqui não se estratifica pela proporção real (9:1) — isso
    # daria só ~20 bonafide e a média da classe ficaria instável.
    n_lado = N_AB // 2
    amostra = pd.concat([
        ev[ev["classe_binaria"] == 0].sample(n=n_lado, random_state=semente),
        ev[ev["classe_binaria"] == 1].sample(n=n_lado, random_state=semente),
    ]).sort_values("arquivo").reset_index(drop=True)

    tarefas = [(r.arquivo, r.caminho, r.classe_binaria, cfg)
               for r in amostra.itertuples()]
    linhas, erros = [], []
    with Pool(cpu_count()) as pool:
        for res in tqdm(pool.imap(_ab_um, tarefas), total=len(tarefas),
                        desc="A/B com e sem máscara"):
            (erros if "erro" in res else linhas).append(res)
    if erros:
        print(f"  {len(erros)} erros na parte A: {erros[:3]}")

    df = pd.DataFrame(linhas)
    nomes = nomes_features(cfg["features"]["n_mfcc"])

    reg = []
    for c in nomes:
        delta = df[f"{c}__A"] - df[f"{c}__B"]
        escala = df[f"{c}__B"].abs().mean() + 1e-12
        d_bona = delta[df["classe_binaria"] == 0]
        d_spoof = delta[df["classe_binaria"] == 1]
        reg.append({
            "feature": c,
            "delta_relativo_pct": round(float(100 * delta.abs().mean() / escala), 2),
            "assimetria_bona_menos_spoof": round(float(d_bona.mean() - d_spoof.mean()), 6),
            # padronizada pelo desvio do próprio delta: comparável entre features
            # de escalas muito diferentes (MFCC em dezenas, ZCR em centésimos)
            "assimetria_padronizada": round(
                float((d_bona.mean() - d_spoof.mean()) / (delta.std() + 1e-12)), 4),
        })
    deltas = pd.DataFrame(reg)
    deltas.to_csv(RAIZ / "results" / "metricas" /
                  "checagem_mascaramento_delta.csv", index=False)

    n_iguais = int((deltas["delta_relativo_pct"] == 0).sum())
    frac_padding = 1 - (df["n_frames_validos"] / df["n_frames_total"])

    resumo = {
        "n_audios": int(len(df)),
        "n_erros": int(len(erros)),
        "features_sem_nenhuma_mudanca": n_iguais,
        "mascaramento_ativo": bool(n_iguais < len(nomes)),
        "delta_relativo_pct_mediano": round(float(deltas["delta_relativo_pct"].median()), 2),
        "delta_relativo_pct_maximo": round(float(deltas["delta_relativo_pct"].max()), 2),
        "top5_assimetria_padronizada": deltas.reindex(
            deltas["assimetria_padronizada"].abs()
            .sort_values(ascending=False).index
        ).head(5)[["feature", "assimetria_padronizada",
                   "delta_relativo_pct"]].to_dict(orient="records"),
        # ATENÇÃO à leitura: `prop_fala` (fração do áudio ORIGINAL mantida pelo
        # VAD) e `fracao_padding` (quanto do tensor de 4,0 s é zero) medem coisas
        # diferentes. Um áudio longo com prop_fala baixa pode terminar com MENOS
        # padding que um curto com prop_fala alta. Reportamos as duas lado a lado
        # justamente para impedir que uma seja usada como proxy da outra.
        "prop_fala_media_bonafide": round(
            float(df.loc[df["classe_binaria"] == 0, "prop_fala"].mean()), 4),
        "prop_fala_media_spoof": round(
            float(df.loc[df["classe_binaria"] == 1, "prop_fala"].mean()), 4),
        "fracao_padding_media_bonafide": round(
            float(frac_padding[df["classe_binaria"] == 0].mean()), 4),
        "fracao_padding_media_spoof": round(
            float(frac_padding[df["classe_binaria"] == 1].mean()), 4),
        "diferenca_fracao_padding_bona_menos_spoof": round(
            float(frac_padding[df["classe_binaria"] == 0].mean()
                  - frac_padding[df["classe_binaria"] == 1].mean()), 4),
    }

    dif = resumo["diferenca_fracao_padding_bona_menos_spoof"]
    print("\n--- PARTE A: o mascaramento mudou as features? ---")
    print(f"  features alteradas         : {len(nomes) - n_iguais}/{len(nomes)}")
    print(f"  delta relativo mediano     : {resumo['delta_relativo_pct_mediano']:.2f}%")
    print(f"  delta relativo máximo      : {resumo['delta_relativo_pct_maximo']:.2f}%")
    print(f"  prop_fala média (bonafide) : {resumo['prop_fala_media_bonafide']:.3f}")
    print(f"  prop_fala média (spoof)    : {resumo['prop_fala_media_spoof']:.3f}")
    print(f"  padding médio (bonafide)   : {100*resumo['fracao_padding_media_bonafide']:.1f}% do tensor")
    print(f"  padding médio (spoof)      : {100*resumo['fracao_padding_media_spoof']:.1f}% do tensor")
    print(f"  diferença (bona − spoof)   : {100*dif:+.1f} p.p.")
    print("  -> DUAS LEITURAS SEPARADAS:")
    print("     (1) validade de medida: o delta acima é grande em TODAS as features,")
    print("         independentemente de classe — sozinho já justifica mascarar;")
    print("     (2) atalho de classe: olhe a assimetria padronizada, NÃO a diferença")
    print("         de padding. Padding igual entre classes NÃO significa distorção")
    print("         igual: o deslocamento depende do conteúdo acústico de cada uma.")
    return resumo


# ---------------------------------------------------------------------------
# PARTE B — o runner de produção
# ---------------------------------------------------------------------------
def parte_b(cfg: dict) -> dict:
    saida = RAIZ / "data" / "features" / NOME_SAIDA
    if saida.exists():
        # o runner tem retomada por nome de arquivo; num piloto de validação
        # queremos SEMPRE uma extração limpa, não uma colcha de retalhos
        saida.unlink()
        print(f"Piloto anterior removido ({NOME_SAIDA}) — extração limpa.")

    executar(cfg, RAIZ, limite=N_PILOTO, nome_saida=NOME_SAIDA)

    df = pd.read_csv(saida)
    nomes = nomes_features(cfg["features"]["n_mfcc"])
    esperadas = ["arquivo", "label", "classe_binaria"] + COLUNAS_DIAGNOSTICO + nomes

    cols_X = colunas_features(df)
    problemas = []

    faltando = [c for c in esperadas if c not in df.columns]
    sobrando = [c for c in df.columns if c not in esperadas]
    if faltando:
        problemas.append(f"colunas faltando no CSV: {faltando}")
    if sobrando:
        problemas.append(f"colunas inesperadas no CSV: {sobrando}")
    if len(cols_X) != 44:
        problemas.append(f"colunas_features devolveu {len(cols_X)}, esperado 44")
    vazadas = [c for c in COLUNAS_DIAGNOSTICO if c in cols_X]
    if vazadas:
        problemas.append(f"coluna diagnóstica DENTRO do X: {vazadas}")

    numericas = df[nomes]
    n_nan = int(numericas.isna().sum().sum())
    n_inf = int(np.isinf(numericas.to_numpy(dtype=float)).sum())
    if n_nan or n_inf:
        problemas.append(f"valores inválidos nas features: {n_nan} NaN, {n_inf} inf")

    if (df["n_frames_validos"] > df["n_frames_total"]).any():
        problemas.append("n_frames_validos > n_frames_total em alguma linha")
    if (df["n_frames_validos"] < 1).any():
        problemas.append("n_frames_validos < 1 em alguma linha")

    # universo: todo arquivo do piloto tem de ser fase == 'eval'
    labels = pd.read_csv(RAIZ / "data" / "processed" / "labels.csv",
                         usecols=["arquivo", "fase"])
    fases = df.merge(labels, on="arquivo", how="left")["fase"].unique().tolist()
    if fases != [cfg["dataset"]["fase"]]:
        problemas.append(f"universo contaminado — fases encontradas: {fases}")

    resumo = {
        "n_linhas": int(len(df)),
        "n_bonafide": int((df["classe_binaria"] == 0).sum()),
        "n_spoof": int((df["classe_binaria"] == 1).sum()),
        "n_colunas": int(df.shape[1]),
        "n_colunas_X": len(cols_X),
        "fases_encontradas": fases,
        "n_frames_total_unico": sorted(df["n_frames_total"].unique().tolist()),
        "n_frames_validos_min": int(df["n_frames_validos"].min()),
        "n_frames_validos_mediana": float(df["n_frames_validos"].median()),
        "n_frames_validos_max": int(df["n_frames_validos"].max()),
        "problemas": problemas,
        "aprovado": not problemas,
    }

    print("\n--- PARTE B: esquema do CSV gerado pelo runner ---")
    print(f"  linhas                : {resumo['n_linhas']} "
          f"({resumo['n_bonafide']} bonafide / {resumo['n_spoof']} spoof)")
    print(f"  colunas no CSV        : {resumo['n_colunas']} (esperado 50 = 3 id + 3 diag + 44 feat)")
    print(f"  colunas no X          : {resumo['n_colunas_X']} (esperado 44)")
    print(f"  fases no piloto       : {fases}")
    print(f"  n_frames_validos      : min {resumo['n_frames_validos_min']}, "
          f"mediana {resumo['n_frames_validos_mediana']:.0f}, "
          f"max {resumo['n_frames_validos_max']} "
          f"(total por áudio: {resumo['n_frames_total_unico']})")
    print("\n  primeiras linhas (colunas selecionadas):")
    print(df[["arquivo", "classe_binaria", "prop_fala", "n_frames_validos",
              "n_frames_total", "mfcc1_media", "centroide_std"]].head().to_string(index=False))
    return resumo


def main() -> None:
    cfg = carregar_config(RAIZ)
    semente = fixar_seeds(cfg["semente"])
    (RAIZ / "results" / "metricas").mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CHECAGEM DO BLOCO 1 — mascaramento de padding")
    print("=" * 70)

    resumo_a = parte_a(cfg, semente)
    resumo_b = parte_b(cfg)

    registro = {
        "semente": semente,
        "universo": cfg["dataset"]["fase"],
        "duracao_segundos": cfg["audio"]["duracao_segundos"],
        "hop_length": cfg["features"]["hop_length"],
        "win_length": cfg["features"]["win_length"],
        "definicao_frame_valido": "centro do frame (i*hop, center=True) dentro do "
                                  "áudio real pós-VAD e pré-padding — idêntica à do "
                                  "piloto scripts/piloto_mascaramento_padding.py",
        "parte_a_ab_controlado": resumo_a,
        "parte_b_runner": resumo_b,
        "aprovado_para_lote_unico": bool(resumo_a["mascaramento_ativo"]
                                         and resumo_b["aprovado"]),
    }
    destino = RAIZ / "results" / "metricas" / "checagem_mascaramento.json"
    with open(destino, "w", encoding="utf-8") as f:
        json.dump(registro, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    if registro["aprovado_para_lote_unico"]:
        print("APROVADO — pipeline pronto para o lote único de re-extração.")
    else:
        print("REPROVADO — corrigir antes de gastar horas no lote único:")
        for p in resumo_b["problemas"]:
            print(f"  - {p}")
        if not resumo_a["mascaramento_ativo"]:
            print("  - o mascaramento NÃO alterou nenhuma feature (parte A)")
    print(f"Registro salvo em {destino}")
    print("=" * 70)

    sys.exit(0 if registro["aprovado_para_lote_unico"] else 1)


if __name__ == "__main__":
    main()
