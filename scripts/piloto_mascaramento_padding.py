"""
Diagnóstico de padding (4.3) — piloto de mascaramento (2.000 áudios)
====================================================================

ÚNICA EXTRAÇÃO DE ÁUDIO AUTORIZADA NESTA RODADA. Grava em arquivo separado
(data/features/piloto_padding.csv) e NÃO encosta no features.csv.

OBJETIVO:
    Medir QUANTO o zero-padding da padronização de duração distorce as 44
    features, para o orientador decidir se o mascaramento da agregação entra
    no lote único de re-extração. A pergunta central não é "o padding muda as
    features?" (muda, por construção), e sim: a distorção é ASSIMÉTRICA entre
    as classes? Se distorce igual, é ruído; se distorce mais o bonafide (que
    perde mais sinal no VAD e recebe mais padding), é o atalho de silêncio.

PROTOCOLO:
    - Amostra: 2.000 áudios do universo eval, estratificados por classe ×
      codec (maior resto), seed 42.
    - Por áudio, as séries (MFCC, ZCR, centróide) são calculadas UMA vez sobre
      os 4,0 s padronizados e agregadas DUAS vezes:
        (A) como hoje  — média/std sobre TODOS os frames (padding incluído);
        (B) mascarado  — média/std SOMENTE sobre os frames válidos.
      Frame válido: aquele cujo centro (i*hop, com center=True do librosa) cai
      dentro do trecho de áudio real (antes do padding). n_frames_validos é
      registrado por áudio.
    - RF (hiperparâmetros do baseline) treinado em (A) e em (B) dentro do
      piloto, com split interno 70/30 estratificado. Com 2.000 áudios os
      números são INSTÁVEIS — tratar como indicativo, e isso está dito no JSON.

SAÍDAS:
    data/features/piloto_padding.csv
    results/metricas/piloto_padding_delta_features.csv
    results/metricas/piloto_padding_rf_ab.json
    results/figuras/piloto_padding_delta.png

Rode a partir da raiz:  python -m scripts.piloto_mascaramento_padding
"""

import json
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import librosa
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.utils.config import carregar_config
from src.utils.seeds import fixar_seeds
from src.data.preprocessamento import (aplicar_vad, carregar_audio,
                                       normalizar_amplitude,
                                       padronizar_duracao)
from src.features.extrair_features import nomes_features
from src.models.treinar_rf import calcular_eer

RAIZ = Path(__file__).resolve().parents[1]
N_PILOTO = 2000


def amostrar_piloto(labels: pd.DataFrame, n: int, semente: int) -> pd.DataFrame:
    """Amostra estratificada por classe × codec (maior resto), universo eval."""
    ev = labels[labels["fase"] == "eval"].copy()
    ev["estrato"] = ev["classe_binaria"].astype(str) + "|" + ev["codec"]
    tam = ev["estrato"].value_counts().sort_index()

    exato = n * tam / tam.sum()
    aloc = np.floor(exato).astype(int)
    restos = (exato - aloc).sort_values(ascending=False, kind="stable")
    for chave in restos.index[:n - int(aloc.sum())]:
        aloc[chave] += 1

    partes = [ev[ev["estrato"] == e].sample(n=int(aloc[e]), random_state=semente)
              for e in sorted(aloc.index)]
    return pd.concat(partes).sort_values("arquivo").reset_index(drop=True)


def _extrair_ab(args: tuple) -> dict:
    """Worker: extrai as 44 features nas versões (A) padding e (B) mascarada."""
    arquivo, caminho, label, classe, cfg = args
    try:
        sr = cfg["audio"]["sample_rate"]
        dur = cfg["audio"]["duracao_segundos"]
        n_mfcc = cfg["features"]["n_mfcc"]
        n_fft = cfg["features"]["n_fft"]
        hop = cfg["features"]["hop_length"]
        win = cfg["features"]["win_length"]

        # pipeline idêntico ao preprocessar_audio, mas expondo o nº de
        # amostras VÁLIDAS (pós-VAD, pré-padding), que ele não devolve
        y = carregar_audio(caminho, sr)
        y = normalizar_amplitude(y)
        prop = 1.0
        if cfg["audio"]["vad"]:
            y, prop = aplicar_vad(y, sr)
        alvo = int(sr * dur)
        n_validas = min(len(y), alvo)
        y = padronizar_duracao(y, sr, dur)

        # séries calculadas UMA vez, sobre o áudio padronizado (como hoje)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft,
                                    hop_length=hop, win_length=win)
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=win,
                                                 hop_length=hop)
        cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft,
                                                 hop_length=hop, win_length=win)

        n_frames_total = mfcc.shape[1]
        # frame i (center=True) tem centro em i*hop -> válido se i*hop < n_validas
        n_frames_validos = min(n_frames_total,
                               int(np.ceil(n_validas / hop)))
        n_frames_validos = max(n_frames_validos, 1)

        def agregar(series: list[np.ndarray], corte: int | None) -> np.ndarray:
            partes = []
            for s in series:
                st = s if corte is None else s[:, :corte]
                partes.append(st.mean(axis=1))
                partes.append(st.std(axis=1))
            return np.concatenate(partes).astype(np.float32)

        va = agregar([mfcc, zcr, cent], None)               # (A) como hoje
        vb = agregar([mfcc, zcr, cent], n_frames_validos)   # (B) mascarado

        nomes = nomes_features(n_mfcc)
        linha = {"arquivo": arquivo, "label": label, "classe_binaria": classe,
                 "prop_fala": round(prop, 4),
                 "n_frames_total": n_frames_total,
                 "n_frames_validos": n_frames_validos}
        linha.update({f"{c}_A": v for c, v in zip(nomes, va)})
        linha.update({f"{c}_B": v for c, v in zip(nomes, vb)})
        return linha
    except Exception as e:
        return {"arquivo": arquivo, "erro": repr(e)}


def main() -> None:
    cfg = carregar_config(RAIZ)
    semente = fixar_seeds(cfg["semente"])

    labels = pd.read_csv(RAIZ / "data" / "processed" / "labels.csv")
    piloto = amostrar_piloto(labels, N_PILOTO, semente)
    print(f"Piloto: {len(piloto)} áudios "
          f"({int((piloto['classe_binaria'] == 0).sum())} bonafide, "
          f"{int((piloto['classe_binaria'] == 1).sum())} spoof)")

    tarefas = [(r.arquivo, r.caminho, r.label, r.classe_binaria, cfg)
               for r in piloto.itertuples()]

    t0 = time.perf_counter()
    linhas, erros = [], []
    with Pool(cpu_count()) as pool:
        for res in tqdm(pool.imap(_extrair_ab, tarefas), total=len(tarefas)):
            (erros if "erro" in res else linhas).append(res)
    print(f"Extração A/B de {len(linhas)} áudios em "
          f"{time.perf_counter() - t0:.0f}s ({len(erros)} erros)")
    if erros:
        print("ERROS:", erros[:5])

    df = pd.DataFrame(linhas)
    saida_csv = RAIZ / "data" / "features" / "piloto_padding.csv"
    df.to_csv(saida_csv, index=False)
    print(f"Features A/B salvas em {saida_csv} (features.csv intocado)")

    nomes = nomes_features(cfg["features"]["n_mfcc"])
    dir_met = RAIZ / "results" / "metricas"
    dir_fig = RAIZ / "results" / "figuras"
    dir_met.mkdir(parents=True, exist_ok=True)
    dir_fig.mkdir(parents=True, exist_ok=True)

    # ---- Delta por feature, separado por classe ------------------------------
    reg = []
    for c in nomes:
        delta = df[f"{c}_A"] - df[f"{c}_B"]
        escala = df[f"{c}_B"].abs().mean() + 1e-12
        d_bona = delta[df["classe_binaria"] == 0]
        d_spoof = delta[df["classe_binaria"] == 1]
        reg.append({
            "feature": c,
            "delta_medio": round(float(delta.mean()), 6),
            "delta_relativo_pct": round(float(100 * delta.abs().mean() / escala), 2),
            "delta_medio_bonafide": round(float(d_bona.mean()), 6),
            "delta_medio_spoof": round(float(d_spoof.mean()), 6),
            "assimetria_bona_menos_spoof": round(float(d_bona.mean() -
                                                       d_spoof.mean()), 6),
            # tamanho de efeito da assimetria (delta padronizado pelo desvio
            # do delta): comparável entre features de escalas diferentes
            "assimetria_padronizada": round(float(
                (d_bona.mean() - d_spoof.mean()) / (delta.std() + 1e-12)), 4),
        })
    deltas = pd.DataFrame(reg)
    deltas.to_csv(dir_met / "piloto_padding_delta_features.csv", index=False)

    # ---- RF A/B dentro do piloto ----------------------------------------------
    y = df["classe_binaria"].values
    idx_tr, idx_te = train_test_split(np.arange(len(df)), train_size=0.7,
                                      stratify=y, random_state=semente)
    res_ab = {}
    for versao in ["A", "B"]:
        X = df[[f"{c}_{versao}" for c in nomes]].values
        rf = RandomForestClassifier(n_estimators=100, max_depth=None,
                                    class_weight="balanced",
                                    random_state=semente, n_jobs=-1)
        rf.fit(X[idx_tr], y[idx_tr])
        y_pred = rf.predict(X[idx_te])
        scores = rf.predict_proba(X[idx_te])[:, 1]
        eer, _ = calcular_eer(y[idx_te], scores)
        res_ab[versao] = {
            "f1_macro": round(float(f1_score(y[idx_te], y_pred, average="macro",
                                             zero_division=0)), 4),
            "eer": round(eer, 4),
        }
        print(f"RF ({versao}): f1_macro {res_ab[versao]['f1_macro']:.4f} | "
              f"EER {res_ab[versao]['eer']:.4f}")

    top_assim = deltas.reindex(
        deltas["assimetria_padronizada"].abs().sort_values(ascending=False).index
    ).head(10)

    registro = {
        "n_piloto": int(len(df)),
        "n_erros": int(len(erros)),
        "semente": semente,
        "estratificacao_amostra": "classe_binaria x codec (maior resto), universo eval",
        "definicao_frame_valido": "centro do frame (i*hop, center=True) dentro "
                                  "do áudio real pós-VAD, pré-padding",
        "aviso": "com 2.000 áudios os números do RF A/B são INSTÁVEIS — "
                 "indicativo, não conclusivo; o delta por feature é mais estável",
        "rf_interno_70_30": res_ab,
        "delta_f1_macro_B_menos_A": round(res_ab["B"]["f1_macro"] -
                                          res_ab["A"]["f1_macro"], 4),
        "delta_eer_B_menos_A": round(res_ab["B"]["eer"] - res_ab["A"]["eer"], 4),
        "top10_features_assimetria_padronizada":
            top_assim[["feature", "assimetria_padronizada",
                       "delta_relativo_pct"]].to_dict(orient="records"),
    }
    with open(dir_met / "piloto_padding_rf_ab.json", "w", encoding="utf-8") as f:
        json.dump(registro, f, indent=2, ensure_ascii=False)

    # ---- Figura -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    ordem = deltas.reindex(deltas["delta_relativo_pct"]
                           .sort_values().index)
    axes[0].barh(ordem["feature"], ordem["delta_relativo_pct"], color="#4c72b0")
    axes[0].set_xlabel("|delta| médio relativo (%) — A vs B")
    axes[0].set_title("Quanto o padding distorce cada feature")
    axes[0].grid(alpha=0.3, axis="x")

    ordem2 = deltas.reindex(deltas["assimetria_padronizada"]
                            .sort_values().index)
    cores = ["#c44e52" if v > 0 else "#4c72b0"
             for v in ordem2["assimetria_padronizada"]]
    axes[1].barh(ordem2["feature"], ordem2["assimetria_padronizada"], color=cores)
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].set_xlabel("assimetria padronizada (bonafide − spoof)")
    axes[1].set_title("A distorção é diferente entre as classes?")
    axes[1].grid(alpha=0.3, axis="x")

    fig.suptitle("Piloto de mascaramento de padding — 2.000 áudios (eval)")
    fig.tight_layout()
    fig.savefig(dir_fig / "piloto_padding_delta.png", dpi=150)
    plt.close(fig)

    print(f"\nDelta por feature: {dir_met / 'piloto_padding_delta_features.csv'}")
    print(f"RF A/B           : {dir_met / 'piloto_padding_rf_ab.json'}")
    print(f"Figura           : {dir_fig / 'piloto_padding_delta.png'}")


if __name__ == "__main__":
    main()
