"""
Extração de features acústicas — MFCC, ZCR, Centróide Espectral
===============================================================

Transforma cada áudio (já pré-processado) num VETOR de tamanho fixo, que é a
entrada tabular do Random Forest e do SVM.

O PROBLEMA que a agregação resolve:
    MFCC, ZCR e centróide são SÉRIES TEMPORAIS (um valor por janela de ~25 ms).
    Áudios diferentes têm nº de janelas diferentes -> vetores de tamanho variável.
    Mas RF e SVM exigem entrada de dimensão FIXA. Solução: resumir cada série no
    tempo por sua MÉDIA (comportamento central) e seu DESVIO-PADRÃO
    (variabilidade/dinâmica). Uma série de N janelas vira 2 números.

Tamanho do vetor final:
    MFCC (20 coef.) × {média, std} = 40
    ZCR              × {média, std} =  2
    Centróide        × {média, std} =  2
    -------------------------------------
    TOTAL                            = 44 features por áudio

IMPORTANTE — as features são salvas CRUAS (não padronizadas).
A padronização (StandardScaler) entra DENTRO do pipeline de cada modelo, ajustada
apenas no fold de treino. Padronizar o CSV inteiro de uma vez faria o scaler
"enxergar" média/desvio das amostras de teste = VAZAMENTO DE DADOS (data leakage),
inflando as métricas artificialmente.

Saída: data/features/features.csv  (uma linha por áudio)
"""

from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import librosa
import pandas as pd
from tqdm import tqdm

# NOTA: o módulo de pré-processamento chama-se `preprocessamento.py` neste repositório.
# Rode a partir da RAIZ do projeto com:  python -m src.features.extrair_features
from ..data.preprocessamento import preprocessar_audio


# ---------------------------------------------------------------------------
# Nomes das colunas — na MESMA ordem em que o vetor é montado
# ---------------------------------------------------------------------------
def nomes_features(n_mfcc: int) -> list[str]:
    cols = [f"mfcc{i+1}_media" for i in range(n_mfcc)]
    cols += [f"mfcc{i+1}_std" for i in range(n_mfcc)]
    cols += ["zcr_media", "zcr_std", "centroide_media", "centroide_std"]
    return cols


# ---------------------------------------------------------------------------
# Extração do vetor de features de UM áudio já pré-processado
# ---------------------------------------------------------------------------
def extrair_vetor(y: np.ndarray, sr: int, cfg: dict) -> np.ndarray:
    """Calcula MFCC, ZCR e centróide e agrega cada um em (média, std) no tempo.

    Parâmetros (todos vindos do config.yaml):
      n_mfcc=20  -> nº de coeficientes cepstrais. Os primeiros capturam o envelope
                    grosso do espectro (ressonância do trato vocal); os mais altos,
                    detalhes finos onde artefatos de síntese tendem a aparecer.
      n_fft=512  -> tamanho da FFT por janela.
      win=400    -> ~25 ms a 16 kHz: a janela curta em que a fala é aproximadamente
                    estacionária (o janelamento discutido na fundamentação).
      hop=256    -> ~16 ms de passo (sobreposição entre janelas consecutivas).
    """
    n_mfcc = cfg["features"]["n_mfcc"]
    n_fft = cfg["features"]["n_fft"]
    hop = cfg["features"]["hop_length"]
    win = cfg["features"]["win_length"]

    # MFCC -> (n_mfcc, n_janelas): textura fina / envelope espectral do trato vocal
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop, win_length=win
    )
    # ZCR -> (1, n_janelas): cruzamentos por zero. Ruído/fricativas de alta freq.
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=win, hop_length=hop)
    # Centróide espectral -> (1, n_janelas): "centro de massa" do espectro.
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop)

    # Agregação temporal: para cada série, média e desvio-padrão ao longo do tempo.
    partes = []
    for serie in (mfcc, zcr, cent):
        partes.append(serie.mean(axis=1))
        partes.append(serie.std(axis=1))
    return np.concatenate(partes).astype(np.float32)


# ---------------------------------------------------------------------------
# Worker: processa UM áudio (top-level para ser "picklável" pelo multiprocessing)
# ---------------------------------------------------------------------------
def _processar_um(args: tuple) -> dict:
    arquivo, caminho, label, classe, cfg = args
    try:
        y, prop = preprocessar_audio(caminho, cfg)
        vetor = extrair_vetor(y, cfg["audio"]["sample_rate"], cfg)
        linha = {
            "arquivo": arquivo,
            "label": label,
            "classe_binaria": classe,
            "prop_fala": round(prop, 4),   # diagnóstico do VAD (ver se comeu demais)
        }
        linha.update(dict(zip(nomes_features(cfg["features"]["n_mfcc"]), vetor)))
        return linha
    except Exception as e:
        # Um .flac corrompido não pode derrubar a varredura inteira de 181k arquivos.
        return {"arquivo": arquivo, "erro": repr(e)}


# ---------------------------------------------------------------------------
# Runner: varre TODOS os áudios em paralelo, com checkpoint/retomada
# ---------------------------------------------------------------------------
def executar(cfg: dict, raiz: Path, n_jobs: int | None = None, flush_a_cada: int = 2000):
    """Extrai features de todos os áudios do labels.csv e grava incrementalmente.

    RESILIÊNCIA (por que checkpoint importa em 181k arquivos):
      - Se `features.csv` já existir, os áudios já extraídos são PULADOS. Uma queda
        no arquivo 170.000 não custa as horas anteriores: é só rodar de novo.
      - Grava em blocos (append) a cada `flush_a_cada` linhas, então o progresso
        fica em disco continuamente, não só no fim.
      - Ordem determinística (imap, não imap_unordered): mesma entrada -> mesmo CSV.
    """
    n_jobs = n_jobs or cpu_count()

    labels_csv = raiz / "data" / "processed" / "labels.csv"
    if not labels_csv.exists():
        raise FileNotFoundError(
            f"{labels_csv} não existe. Rode antes o notebook 02_leitura_dados.ipynb."
        )

    labels = pd.read_csv(labels_csv)
    saida = raiz / "data" / "features" / "features.csv"
    saida.parent.mkdir(parents=True, exist_ok=True)

    # Retomada: descobre o que já foi feito
    ja_feitos = set()
    if saida.exists():
        ja_feitos = set(pd.read_csv(saida, usecols=["arquivo"])["arquivo"])
        print(f"Retomando: {len(ja_feitos)} áudios já extraídos serão pulados.")

    tarefas = [
        (r.arquivo, r.caminho, r.label, r.classe_binaria, cfg)
        for r in labels.itertuples()
        if r.arquivo not in ja_feitos
    ]
    print(f"Total no labels.csv : {len(labels)}")
    print(f"A extrair           : {len(tarefas)} áudios usando {n_jobs} processos.")
    if not tarefas:
        print("Nada a fazer — tudo já extraído.")
        return

    buffer, erros = [], []
    escrever_header = not saida.exists()

    with Pool(n_jobs) as pool:
        for res in tqdm(pool.imap(_processar_um, tarefas), total=len(tarefas)):
            if "erro" in res:
                erros.append(res)
                continue
            buffer.append(res)
            if len(buffer) >= flush_a_cada:
                pd.DataFrame(buffer).to_csv(saida, mode="a", header=escrever_header, index=False)
                escrever_header = False
                buffer = []

    if buffer:  # sobra final
        pd.DataFrame(buffer).to_csv(saida, mode="a", header=escrever_header, index=False)

    erros_csv = raiz / "data" / "features" / "erros.csv"
    if erros:
        pd.DataFrame(erros).to_csv(erros_csv, index=False)
        print(f"ATENÇÃO: {len(erros)} arquivos falharam. Detalhes em {erros_csv}")
    elif erros_csv.exists():
        # Uma rodada sem falhas re-tentou (e venceu) tudo que estava no erros.csv
        # antigo; remove o arquivo para ele não apontar falhas que já não existem.
        erros_csv.unlink()
        print("Nenhuma falha nesta rodada — erros.csv anterior removido.")

    print(f"Concluído. Features salvas em {saida}")


if __name__ == "__main__":
    import os
    import yaml

    RAIZ = Path(__file__).resolve().parents[2]
    cfg = yaml.safe_load(open(RAIZ / "config" / "config.yaml", encoding="utf-8"))

    # nº de processos: use N_JOBS=4 python -m ... para limitar; padrão = todos os núcleos
    n_jobs = int(os.environ.get("N_JOBS", 0)) or None
    executar(cfg, RAIZ, n_jobs=n_jobs)