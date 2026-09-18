"""
Estatísticas de normalização da CNN — média e desvio POR FAIXA MEL (P3)
=======================================================================

O QUE ESTE SCRIPT FAZ:
    Varre os 27.000 tensores do `treino_interno` no memmap do B4.2 e calcula
    DOIS VETORES DE 128 — uma média e um desvio por faixa Mel —, gravando-os em
    `data/espectrogramas/normalizacao_cnn.json`, versionado no git.

O QUE A P3 DECIDIU, E AS TRÊS PRECISÕES QUE DECIDEM A IMPLEMENTAÇÃO:
    "Normalização por estatísticas GLOBAIS DO TREINO, média e desvio POR FAIXA
    MEL, calculadas somente sobre o treino efetivamente disponível naquele
    estágio. Validação e teste NUNCA entram no cálculo e NUNCA recalculam."

    1. "por faixa Mel" = uma média e um desvio POR LINHA do espectrograma: dois
       vetores de 128, não um escalar global e não 128x251 valores. As faixas
       Mel têm energia média muito diferente entre si (as graves concentram a
       energia da fala); normalizar por faixa iguala a escala entre elas sem
       apagar o contraste temporal, que é o que a convolução vai ler.
    2. "somente o treino daquele estágio" = aqui, os 27k. No refit (B4.6),
       recomputa-se nos 30k. São dois artefatos distintos e os dois ficam
       registrados — é o que impede a CNN final de treinar com estatística de
       outro n.
    3. "validação e teste nunca recalculam" = eles são normalizados com estas
       constantes. Mesmo princípio que faz o StandardScaler do SVM viver DENTRO
       do Pipeline: vazamento por normalização global é irreversível num
       artefato gerado uma vez só.

A ARMADILHA ARITMÉTICA — FRAMES VÁLIDOS, NÃO FRAMES TOTAIS:
    A média por faixa tem de sair SÓ dos `n_frames_validos` de cada exemplo. Uma
    varredura dos 251 frames põe ~47% de platô de padding (em `max - 80`) dentro
    da média, e a estatística passa a descrever, em boa parte, a formatação do
    tensor. É exatamente o erro que o Bloco 1 corrigiu para o ramo clássico;
    mascarar aqui é a mesma operação, no mesmo lugar conceitual.

    O `n_frames_validos` vem do índice, direto: o B4.2 conferiu a paridade com o
    `features.csv` congelado LINHA A LINHA nos 74.453 áudios, então o risco de
    ±1 frame entre a máscara da CNN e o mascaramento clássico está encerrado por
    medição — não há o que reconferir aqui.

DUAS PASSADAS, E POR QUÊ:
    Passada 1 acumula soma (e, de graça, soma de quadrados); passada 2 acumula
    `(X - media)^2`. A forma de uma passada, `E[x^2] - E[x]^2`, é numericamente
    instável quando a média é grande frente ao desvio — e aqui as médias em dB
    são da ordem de dezenas. Em float32 isso chega a dar variância NEGATIVA.
    Os acumuladores são float64 e a segunda passada custa uma varredura extra de
    poucos minutos: é o preço certo. O desvio pela forma de uma passada é
    calculado junto, só para REGISTRAR a diferença entre os dois métodos — um
    afastamento grande aí seria sinal de problema numérico.

ONDE A NORMALIZAÇÃO É APLICADA:
    Em tempo de carga, no Dataset (B4.4) — não nos tensores em disco:
        X = (X - media[:, None]) / desvio[:, None]
    Aplicada aos 251 frames, INCLUSIVE ao padding, de propósito: o padding é um
    platô constante que continua constante depois da normalização, e não entra
    em conta nenhuma porque a agregação da rede é mascarada (P2). Normalizar só
    a parte válida criaria uma descontinuidade artificial na fronteira, que a
    convolução leria como borda.

SAÍDA:
    data/espectrogramas/normalizacao_cnn.json   (VERSIONADO — é o que torna a
                                                 inferência reproduzível)

Rode a partir da raiz:  python -m scripts.calcular_normalizacao_cnn
"""

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import carregar_config

RAIZ = Path(__file__).resolve().parents[1]

DIR_ESP = "data/espectrogramas"
MEMMAP = "treino_30k.npy"
INDICE = "indice_treino_30k.csv"
META = "espectrogramas.meta.json"
SPLIT_INTERNO = "data/processed/split_interno_cnn.csv"
SAIDA = "data/espectrogramas/normalizacao_cnn.json"

ESTAGIO = "27k_early_stopping"
EPSILON = 1e-8
# Abaixo disto um desvio não é "pequeno", é suspeito: ver a previsão registrada
# no marco B4.3 — nenhuma faixa deve chegar perto de zero, nem no topo morto dos
# codecs de banda estreita, porque o platô do padding vale `max - 80` DO PRÓPRIO
# EXEMPLO e o `max` varia de exemplo para exemplo.
LIMIAR_SUSPEITO = 1e-6


def _md5(caminho: Path) -> str:
    return hashlib.md5(caminho.read_bytes()).hexdigest()


def _git() -> tuple[str, bool | None]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=RAIZ, text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        commit = "desconhecido"
    try:
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=RAIZ,
            stderr=subprocess.DEVNULL).strip())
    except Exception:
        dirty = None
    return commit, dirty


def main() -> int:
    cfg = carregar_config(RAIZ)
    semente = int(cfg["semente"])
    esp = cfg["espectrograma"]
    n_mels, largura = int(esp["n_mels"]), int(esp["largura"])

    dir_esp = RAIZ / DIR_ESP
    caminho_indice = dir_esp / INDICE
    caminho_split = RAIZ / SPLIT_INTERNO
    caminho_sub = RAIZ / cfg["experimento"]["caminho_subamostra"]

    # ---- Guarda: o índice é o MESMO que descreve o memmap em disco ----------
    # Sem isto, um índice regenerado apontaria `linha` para tensores de outro
    # lote e a estatística sairia de áudios que não são estes — silenciosamente.
    meta = json.loads((dir_esp / META).read_text(encoding="utf-8"))
    md5_indice = _md5(caminho_indice)
    md5_esperado = meta["hash_md5_indice_por_conjunto"][INDICE]
    if md5_indice != md5_esperado:
        print(f"FALHA: {INDICE} tem MD5 {md5_indice}, mas o lote congelado "
              f"registra {md5_esperado}. O índice e o memmap não são o mesmo par.",
              file=sys.stderr)
        return 1

    indice = pd.read_csv(caminho_indice)
    split = pd.read_csv(caminho_split)
    ti = split[split["particao_interna"] == "treino_interno"][["arquivo"]]

    alvo = indice.merge(ti, on="arquivo", how="inner")
    if len(alvo) != len(ti):
        print(f"FALHA: {len(ti) - len(alvo)} arquivos do treino_interno não "
              f"estão no índice do memmap.", file=sys.stderr)
        return 1
    # Ordem por `linha`: leitura sequencial no memmap (rápida) e ordem de
    # acumulação FIXA — soma de float não é associativa, então a ordem faz parte
    # da reprodutibilidade byte a byte deste artefato.
    alvo = alvo.sort_values("linha").reset_index(drop=True)

    mm = np.load(dir_esp / MEMMAP, mmap_mode="r")
    if mm.shape[1:] != (n_mels, largura):
        print(f"FALHA: memmap com tensores {mm.shape[1:]}, esperado "
              f"({n_mels}, {largura}).", file=sys.stderr)
        return 1
    if int(alvo["linha"].max()) >= mm.shape[0]:
        print("FALHA: índice aponta para linha fora do memmap.", file=sys.stderr)
        return 1
    nv = alvo["n_frames_validos"].to_numpy()
    if nv.min() < 1 or nv.max() > largura:
        print(f"FALHA: n_frames_validos fora de [1, {largura}]: "
              f"[{nv.min()}, {nv.max()}].", file=sys.stderr)
        return 1

    linhas = alvo["linha"].to_numpy()
    n_ex = len(alvo)
    print(f"Estágio {ESTAGIO}: {n_ex} exemplos | memmap {mm.shape} | "
          f"n_frames_validos: mín {nv.min()}, mediana {int(np.median(nv))}, "
          f"máx {nv.max()}")
    print(f"frames válidos a acumular: {int(nv.sum())} de {n_ex * largura} "
          f"totais ({100 * nv.sum() / (n_ex * largura):.2f}%) — o resto é "
          f"padding e NÃO entra")

    # ---- Passada 1: soma (e soma de quadrados) por faixa Mel ----------------
    t0 = time.perf_counter()
    soma = np.zeros(n_mels, dtype=np.float64)
    soma_q = np.zeros(n_mels, dtype=np.float64)
    n_frames = 0
    for k, (i, n_valid) in enumerate(zip(linhas, nv), 1):
        X = np.asarray(mm[i], dtype=np.float64)[:, :int(n_valid)]
        soma += X.sum(axis=1)
        soma_q += (X ** 2).sum(axis=1)
        n_frames += int(n_valid)
        if k % 5000 == 0:
            print(f"  passada 1: {k}/{n_ex}")
    media = soma / n_frames

    # desvio pela forma de uma passada — só para comparar com o de duas passadas
    var_1p = soma_q / n_frames - media ** 2
    desvio_1p = np.sqrt(np.maximum(var_1p, 0.0))
    var_negativa_1p = int((var_1p < 0).sum())

    # ---- Passada 2: soma de (X - media)^2 por faixa Mel ---------------------
    soma_d2 = np.zeros(n_mels, dtype=np.float64)
    m_col = media[:, None]
    for k, (i, n_valid) in enumerate(zip(linhas, nv), 1):
        X = np.asarray(mm[i], dtype=np.float64)[:, :int(n_valid)]
        soma_d2 += ((X - m_col) ** 2).sum(axis=1)
        if k % 5000 == 0:
            print(f"  passada 2: {k}/{n_ex}")
    var = soma_d2 / n_frames
    desvio = np.sqrt(np.maximum(var, 0.0))
    dur = time.perf_counter() - t0

    bruto_min = float(desvio.min())
    clampadas = int((desvio < EPSILON).sum())
    desvio = np.maximum(desvio, EPSILON)
    dif_metodos = float(np.abs(desvio_1p - desvio).max())

    print(f"\nduas passadas concluídas em {dur / 60:.1f} min "
          f"({n_frames} frames válidos acumulados)")
    print(f"média  por faixa Mel: [{media.min():.4f}, {media.max():.4f}] dB")
    print(f"desvio por faixa Mel: [{bruto_min:.4f}, {desvio.max():.4f}] dB")
    print(f"faixa 0: media {media[0]:.4f} desvio {desvio[0]:.4f} | "
          f"faixa 127: media {media[-1]:.4f} desvio {desvio[-1]:.4f}")
    print(f"faixas com desvio clampado em {EPSILON}: {clampadas}")
    print(f"variância negativa pela forma E[x^2]-E[x]^2: {var_negativa_1p} "
          f"faixas | maior diferença entre os dois métodos: {dif_metodos:.3e} dB")

    ids = sorted(alvo["arquivo"].astype(str))
    hash_ids = hashlib.sha256(("\n".join(ids) + "\n").encode("utf-8")).hexdigest()
    commit, dirty = _git()

    registro = {
        "estagio": ESTAGIO,
        "conjunto_de_origem": "treino_interno (27.000 de 30.000 da subamostra do "
                              "braco principal)",
        "n_exemplos": n_ex,
        "n_frames_validos_acumulados": n_frames,
        "n_frames_totais_se_nao_mascarasse": int(n_ex * largura),
        "eixo": "por_faixa_mel",
        "mascarado": True,
        "nota_mascaramento": "media e desvio calculados SOMENTE sobre os "
                             "n_frames_validos de cada exemplo; os frames de "
                             "padding (plato em max-top_db) nao entram — mesma "
                             "operacao que o mascaramento da agregacao do Bloco 1",
        "metodo_variancia": "duas passadas: passada 1 acumula a soma e passada 2 "
                            "acumula (X-media)^2, ambas em float64",
        "desvio_minimo_antes_do_clamp": bruto_min,
        "faixas_com_desvio_clampado": clampadas,
        "epsilon_desvio": EPSILON,
        "conferencia_metodo_uma_passada": {
            "nota": "E[x^2]-E[x]^2 em float64, calculado na mesma passada 1 "
                    "apenas para comparacao; NAO e o valor gravado",
            "faixas_com_variancia_negativa": var_negativa_1p,
            "maior_diferenca_absoluta_no_desvio": dif_metodos,
        },
        "media_por_mel": media.tolist(),
        "desvio_por_mel": desvio.tolist(),
        "nota_aplicacao": "aplicar em tempo de carga, no Dataset, sobre os 251 "
                          "frames (inclusive o padding): X = (X - media[:, None]) "
                          "/ desvio[:, None]. Os tensores em disco permanecem crus, "
                          "e por isso o refit do B4.6 recomputa as estatisticas nos "
                          "30k sem regerar 9,57 GB",
        "semente": semente,
        "hash_md5_split_interno_csv": _md5(caminho_split),
        "hash_md5_subamostra_csv": _md5(caminho_sub),
        "hash_md5_indice_treino_csv": md5_indice,
        "n_ids": len(ids),
        "hash_lista_ids": hash_ids,
        "receita_hash_lista_ids": "sha256 de '\\n'.join(sorted(arquivo)) + '\\n', "
                                  "em utf-8",
        "commit_git": commit,
        "git_dirty": dirty,
        "versoes": {"numpy": np.__version__, "pandas": pd.__version__},
    }
    with open(RAIZ / SAIDA, "w", encoding="utf-8") as f:
        json.dump(registro, f, indent=2, ensure_ascii=False)
    print(f"\nArtefato: {RAIZ / SAIDA}")
    print(f"hash_lista_ids: {hash_ids}")

    if clampadas or bruto_min < LIMIAR_SUSPEITO:
        # Previsão registrada no marco: o esperado é ZERO. Desvio ~0 numa faixa
        # não é o codec de banda estreita — é bug. Investigue nesta ordem.
        print(f"\nACHADO: {clampadas} faixa(s) clampada(s) e desvio mínimo "
              f"{bruto_min:.3e} — abaixo do previsto. Investigue nesta ordem: "
              "(a) a leitura do memmap está pegando a linha certa (indice.linha); "
              "(b) o corte [:, :n_valid] usa o n_valid DAQUELE exemplo; "
              "(c) os acumuladores estão em float64. O JSON foi gravado com o "
              "número registrado — não use estas estatísticas antes de resolver.",
              file=sys.stderr)
        return 1
    print("Nenhuma faixa Mel com desvio zero — como previsto no marco.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
