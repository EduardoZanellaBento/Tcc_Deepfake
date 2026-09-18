"""
Checagem obrigatória do B4.1 — 12 asserções sobre o pipeline log-Mel
===================================================================

É a EVIDÊNCIA que autoriza o lote de 74.453 tensores do B4.2. No molde de
`scripts/verificar_mascaramento.py` e de `scripts/auditar_decisoes_cnn.py`:
FALHA COM CÓDIGO DE SAÍDA 1 se qualquer asserção não valer, e grava
`results/metricas/checagem_espectrogramas.json`.

Não encosta em nenhum artefato oficial: lê o PILOTO, gerado antes por

    python -m src.features.gerar_espectrogramas --limite 200 --nome-saida piloto

AS QUATRO QUE NUNCA SE CORTAM (1, 3, 6, 7): se qualquer uma delas falhar, os
74.453 tensores nascem inúteis, e regerá-los não cabe no cronograma.
    1 — shape exato: um tensor de outra largura não entra na rede.
    3 — NaN/Inf: envenena o treino inteiro, e `nan` propaga calado.
    6 — NÃO-invariância a ganho: é o teste que distingue `ref=1.0` de
        `ref=np.max`. `ref=np.max` é o exemplo mais comum em tutorial de librosa
        e reintroduziria uma normalização por exemplo, contradizendo a P3.
    7 — máscara idêntica ao features.csv: é A garantia de paridade com o ramo
        clássico. Divergir por ±1 frame é indetectável no treino.

TRÊS EXPECTATIVAS ERRADAS, CORRIGIDAS PELA MEDIÇÃO — não as reescreva na forma
errada, porque cada uma produziria uma FALHA FALSA que bloquearia o B4.2:

  ITEM 5 — `db_max` NÃO é <= 0. A STFT soma 400 amostras janeladas, então
    |STFT|^2 passa de 1 com folga mesmo com o sinal em [-1, 1]. O que se asserta é
    `db_max - db_min <= top_db`, nunca o sinal de `db_max`.

  ITEM 8 — o platô do padding começa em `n_valid + 1`, NÃO em `n_valid`. Com
    `win_length=400 > hop=256`, a janela do frame `n_valid` ainda alcança até 200
    amostras de áudio real: esse frame é um frame de TRANSIÇÃO, não silêncio.
    Medido neste piloto: `[:, n_valid:]` tem mais de um valor distinto numa
    fração grande dos exemplos; `[:, n_valid+1:]` é um platô exato em 100% deles.
    A asserção 8b prova que a fronteira é essa e não outra, por aritmética:
    nenhuma amostra real alcança a janela do frame `n_valid + 1`.
    Isto NÃO é um defeito da máscara: `[:, :n_valid]` exclui o frame de
    transição dos DOIS ramos, então a paridade da asserção 7 continua exata.

  ITEM 10 — `n_fft=1024` não ZERA os filtros estreitos, reduz de 61 para 1. O
    filtro Mel mais baixo, junto a `fmin=0`, é estreito por construção da escala.
    O que caracteriza o regime degenerado é SUPORTE MÍNIMO DE 1 BIN e PICOS
    COMPARTILHADOS — e os dois desaparecem com 1024. Exigir zero reprovaria uma
    configuração correta.

MODO `--lote` (B4.2) — 9 asserções sobre o LOTE REAL, não sobre o piloto
=======================================================================

As 12 asserções acima são sobre a DEFINIÇÃO, e rodam antes do lote existir. O
modo `--lote` confere o que só é conferível DEPOIS: que os 74.453 tensores em
disco estão casados, linha a linha, com os rótulos certos.

    python -m scripts.verificar_espectrogramas --lote

AS DUAS QUE PEGAM O ERRO MAIS PERIGOSO DO B4.2 (3 e 6): linha do memmap
desalinhada do rótulo. Um `sort` faltando não quebra nada, não emite aviso, e
reaparece no B4.4 como uma CNN que não aprende — dois dias procurando bug de
arquitetura onde o problema era de dados. Por isso a asserção 3 remonta os
conjuntos esperados A PARTIR do split.csv e da subamostra, sem chamar
`catalogo()`: comparar o gerador consigo mesmo não testaria nada.

Amostragem: as asserções 4, 5, 6 e 7 sorteiam linhas (2.000 e 500 por conjunto)
porque varrer 9,57 GiB custa minutos a cada rodada. O sorteio usa a semente do
config e fica registrado, então a checagem é repetível.

Saídas:
    results/metricas/checagem_espectrogramas.json   (modo padrão, B4.1)
    results/figuras/piloto_espectrogramas.png       (modo padrão, B4.1)
    results/metricas/lote_espectrogramas.json       (modo --lote, B4.2)

Rode a partir da raiz:  python -m scripts.verificar_espectrogramas [--lote]
"""

import json
import shutil
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

import librosa
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data.preprocessamento import preprocessar_audio          # noqa: E402
from src.features.extrair_features import frames_validos          # noqa: E402
from src.features.gerar_espectrogramas import (                   # noqa: E402
    COLUNAS_INDICE, CONJUNTO_TREINO, esquema_esperado,
    frames_validos_espectrograma, gerar_um, largura_esperada, _md5)
from src.utils.config import carregar_config                      # noqa: E402
from src.utils.seeds import fixar_seeds                           # noqa: E402
from src.utils.serializacao import json_seguro                    # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
PREFIXO = "piloto_"

# Tolerâncias. Os tensores são float32 com valores em ~[-72, +23]: o épsilon
# relativo de 6e-8 dá erro absoluto da ordem de 1e-5 dB. `TOL_DB` é folgada o
# bastante para não caçar arredondamento e apertada o bastante para pegar erro
# de fórmula (que seria de dB inteiros).
TOL_DB = 1e-2
# O documento do marco fixa a tolerância do deslocamento de ganho em 0,01 dB.
TOL_GANHO = 1e-2
N_GANHO = 24        # áudios re-processados para o teste de ganho (asserção 6)

# Codecs com teto de 4 kHz, conforme a decisão «sem fmax=4000» ratificada no
# config.yaml (bloco `features`, fonte: results/metricas/eval_composicao_codec.csv):
# banda estreita = alaw, ulaw, gsm, pstn (85.860 áudios, 57,9% do universo);
# banda larga = g722, opus, none (62.316, 42,1%). A asserção 12 usa esta lista
# porque um espectrograma correto tem de MOSTRAR essa diferença de banda: é ela
# que separa «fmax errado» de «banda estreita representada corretamente».
BANDA_ESTREITA = ("alaw", "ulaw", "gsm", "pstn")

# --- modo --lote (B4.2) ----------------------------------------------------
# Tamanhos exatos do lote, fixados no APENDICE_A §1 e no briefing do B4.2.
# Estão literais aqui de propósito: se o split.csv ou a subamostra mudarem, a
# asserção tem de FALHAR, não se adaptar ao novo número em silêncio.
N_ESPERADO = {"treino_30k": 30000, "validacao": 22226, "teste": 22227}
# Quantas linhas as asserções amostrais varrem, por conjunto.
N_AMOSTRA_NAN = 2000
N_AMOSTRA_PAREADA = 500
# Cabeçalho do .npy: 128 bytes na v1.0 para estes shapes. A folga cobre uma
# eventual v2.0/alinhamento diferente sem deixar passar uma linha inteira
# (128.512 bytes), que é o que a asserção 9 precisa pegar.
FOLGA_HEADER_NPY = (0, 1024)


# ---------------------------------------------------------------------------
# Carregamento do piloto
# ---------------------------------------------------------------------------
def carregar_piloto(cfg: dict) -> tuple[dict, dict, pd.DataFrame]:
    """Lê o .meta.json, os memmaps e os índices do piloto.

    Returns:
        (meta, {nome: memmap}, índice concatenado com a coluna `nome_conjunto`).
    """
    d = RAIZ / "data" / "espectrogramas"
    meta_path = d / f"{PREFIXO}espectrogramas.meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} não existe. Gere o piloto antes:\n"
            "  python -m src.features.gerar_espectrogramas --limite 200 "
            "--nome-saida piloto"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    mms, indices = {}, []
    for nome in meta["escopo"]:
        npy = d / f"{PREFIXO}{nome}.npy"
        idx = d / f"{PREFIXO}indice_{nome}.csv"
        if not npy.exists() or not idx.exists():
            raise FileNotFoundError(f"piloto incompleto: falta {npy.name} ou {idx.name}")
        mms[nome] = np.load(npy, mmap_mode="r")
        ind = pd.read_csv(idx)
        if list(ind.columns) != COLUNAS_INDICE:
            raise ValueError(f"{idx.name} tem colunas {list(ind.columns)}, "
                             f"esperado {COLUNAS_INDICE}")
        indices.append(ind.assign(nome_conjunto=nome))
    indice = pd.concat(indices, ignore_index=True)

    # `caminho` fica fora do índice (é local da máquina) — vem do labels.csv
    # `codec` é necessário para a asserção 12: a banda do codec é o que explica
    # um topo de espectro constante (ver BANDA_ESTREITA).
    labels = pd.read_csv(RAIZ / "data" / "processed" / "labels.csv",
                         usecols=["arquivo", "caminho", "label", "codec"])
    indice = indice.merge(labels, on="arquivo", how="left", validate="one_to_one")
    if indice["caminho"].isna().any():
        raise ValueError("há arquivo do índice ausente do labels.csv")
    return meta, mms, indice


# ---------------------------------------------------------------------------
# Worker da asserção 7 (e do teste de ganho da 6): reprocessa um áudio
# ---------------------------------------------------------------------------
def _reprocessar(args: tuple) -> dict:
    arquivo, caminho, cfg, com_ganho = args
    e = cfg["espectrograma"]
    y, _prop, n_validas = preprocessar_audio(caminho, cfg)
    largura = largura_esperada(cfg)
    linha = {
        "arquivo": arquivo,
        "n_amostras_validas": int(n_validas),
        # A MESMA função do ramo clássico, com o hop do bloco `espectrograma`
        "n_frames_validos_recalculado": frames_validos_espectrograma(
            n_validas, e, largura),
        # E a chamada crua a frames_validos(), com o hop literal: prova que o
        # delegador não está mudando nada no caminho
        "n_frames_validos_direto": frames_validos(
            n_validas, e["hop_length"], largura),
    }
    if com_ganho:
        S = gerar_um(y, e["sample_rate"], e)
        S10 = gerar_um((y * 10.0).astype(np.float32), e["sample_rate"], e)
        delta = (S10.astype(np.float64) - S.astype(np.float64))
        linha.update({
            "ganho_delta_min": float(delta.min()),
            "ganho_delta_max": float(delta.max()),
        })
        # CONTRASTE (não é asserção): o mesmo teste com ref=np.max, que é o
        # tutorial padrão de librosa. Ali o deslocamento colapsa para o
        # arredondamento do float32 — a normalização por exemplo apaga o ganho.
        d = e["defaults_librosa_registrados"]
        comum = dict(sr=e["sample_rate"], n_fft=e["n_fft"],
                     win_length=e["win_length"], hop_length=e["hop_length"],
                     window=d["window"], center=e["center"],
                     pad_mode=d["pad_mode"], power=e["power"],
                     n_mels=e["n_mels"], fmin=e["fmin"], fmax=e["fmax"],
                     htk=d["htk"], norm=d["mel_norm"])
        P = librosa.feature.melspectrogram(y=y, **comum)
        P10 = librosa.feature.melspectrogram(y=(y * 10.0).astype(np.float32), **comum)
        kw = dict(amin=d["amin"], top_db=e["top_db"])
        dmax = (librosa.power_to_db(P10, ref=np.max, **kw)
                - librosa.power_to_db(P, ref=np.max, **kw))
        linha["ganho_delta_refmax_absmax"] = float(np.abs(dmax).max())
    return linha


# ---------------------------------------------------------------------------
# As 12 asserções
# ---------------------------------------------------------------------------
def checar(cfg: dict, meta: dict, mms: dict, indice: pd.DataFrame,
           semente: int) -> tuple[list[dict], dict]:
    """Roda as 12 asserções. Devolve (lista de resultados, medições brutas)."""
    e = cfg["espectrograma"]
    altura, largura = e["altura"], largura_esperada(cfg)
    top_db = float(e["top_db"])
    itens: list[dict] = []
    med: dict = {}

    def anotar(n, titulo, ok, detalhe, critica=False):
        itens.append({"item": n, "assercao": titulo, "passou": bool(ok),
                      "critica": critica, "detalhe": detalhe})

    # --- 1. shape exatamente (128, 251) em TODOS os exemplos ---------------
    shapes = {nome: list(mm.shape) for nome, mm in mms.items()}
    n_linhas = {nome: int(mm.shape[0]) for nome, mm in mms.items()}
    problemas = []
    for nome, mm in mms.items():
        esperado_n = int((indice["nome_conjunto"] == nome).sum())
        if mm.shape != (esperado_n, altura, largura):
            problemas.append(f"{nome}: {mm.shape} != {(esperado_n, altura, largura)}")
        if mm.dtype != np.float32:
            problemas.append(f"{nome}: dtype {mm.dtype} != float32")
        # por exemplo, não só pelo shape do bloco: uma fatia por linha
        for i in range(mm.shape[0]):
            if mm[i].shape != (altura, largura):
                problemas.append(f"{nome}[{i}]: {mm[i].shape}")
                break
    med["shapes"] = shapes
    med["n_por_conjunto"] = n_linhas
    anotar(1, f"shape ({altura}, {largura}) em todos os {len(indice)} exemplos",
           not problemas, problemas or "todos conformes", critica=True)

    # --- 2. 251 RECOMPUTADO, não literal ----------------------------------
    n_amostras = int(cfg["audio"]["sample_rate"] * cfg["audio"]["duracao_segundos"])
    recomputada = 1 + n_amostras // e["hop_length"]
    # e o valor que o PIPELINE de fato produz, num áudio real
    r0 = indice.iloc[0]
    y0, _p0, nva0 = preprocessar_audio(r0["caminho"], cfg)
    largura_pipeline = int(gerar_um(y0, e["sample_rate"], e).shape[1])
    coerentes = (recomputada == e["largura"] == meta["largura"]
                 == largura_pipeline == largura)
    med["largura"] = {
        "formula_1_mais_n_div_hop": recomputada,
        "n_amostras": n_amostras, "hop_length": e["hop_length"],
        "config_yaml": e["largura"], "meta_json": meta["largura"],
        "medida_no_pipeline": largura_pipeline,
    }
    anotar(2, f"largura = 1 + {n_amostras} // {e['hop_length']} = {recomputada}, "
              "recomputada e igual ao pipeline", coerentes, med["largura"])

    # --- 3. zero NaN e zero Inf -------------------------------------------
    n_nan = n_inf = 0
    for mm in mms.values():
        a = np.asarray(mm)
        n_nan += int(np.isnan(a).sum())
        n_inf += int(np.isinf(a).sum())
    med["n_nan"] = n_nan
    med["n_inf"] = n_inf
    anotar(3, "zero NaN e zero Inf em todos os tensores",
           n_nan == 0 and n_inf == 0, f"{n_nan} NaN, {n_inf} Inf", critica=True)

    # --- 4. .meta.json == config.yaml (os parâmetros + os 5 defaults) ------
    esquema = esquema_esperado(cfg)
    divergentes = {k: {"meta_json": meta.get(k), "config_yaml": v}
                   for k, v in esquema.items() if meta.get(k) != v}
    # e as três âncoras de proveniência: o lote foi gerado contra ESTES CSVs
    hashes = {
        "hash_md5_features_csv": _md5(RAIZ / "data" / "features" / "features.csv"),
        "hash_md5_split_csv": _md5(RAIZ / "data" / "processed" / "split.csv"),
        "hash_md5_subamostra_csv": _md5(RAIZ / cfg["experimento"]["caminho_subamostra"]),
    }
    hash_divergentes = {k: {"meta_json": meta.get(k), "em_disco": v}
                        for k, v in hashes.items() if meta.get(k) != v}
    med["n_parametros_conferidos"] = len(esquema)
    med["hashes_entrada"] = hashes
    med["caminho_filterbank"] = meta.get("caminho_filterbank")
    anotar(4, f"os {len(esquema)} parâmetros do .meta.json (17 do bloco "
              "`espectrograma` + semente + 5 defaults do librosa + dtype) batem "
              "com o config.yaml, e os 3 hashes de entrada batem com o disco",
           not divergentes and not hash_divergentes,
           {"parametros_divergentes": divergentes,
            "hashes_divergentes": hash_divergentes})

    # --- 5. faixa dinâmica por exemplo ------------------------------------
    # NÃO se asserta o sinal de db_max (ver docstring). Asserta-se:
    #   (a) db_max - db_min <= top_db em todo exemplo;
    #   (b) db_min == db_max - top_db quando existe padding — é o `top_db`
    #       atuando, e não o `amin` (que daria piso fixo em -100 dB).
    dbmax, dbmin, nvals = [], [], []
    for nome, mm in mms.items():
        sub = indice[indice["nome_conjunto"] == nome]
        a = np.asarray(mm, dtype=np.float64)
        dbmax.append(a.max(axis=(1, 2)))
        dbmin.append(a.min(axis=(1, 2)))
        nvals.append(sub.sort_values("linha")["n_frames_validos"].to_numpy())
    dbmax = np.concatenate(dbmax); dbmin = np.concatenate(dbmin)
    nvals = np.concatenate(nvals)
    faixa = dbmax - dbmin
    tem_padding = nvals < largura
    viola_faixa = int((faixa > top_db + TOL_DB).sum())
    viola_piso = int((np.abs(dbmin[tem_padding] - (dbmax[tem_padding] - top_db))
                      > TOL_DB).sum())
    piso_amin = float(10 * np.log10(e["defaults_librosa_registrados"]["amin"]))
    med["db"] = {
        "db_max_min": float(dbmax.min()), "db_max_max": float(dbmax.max()),
        "db_max_mediana": float(np.median(dbmax)),
        "db_min_min": float(dbmin.min()), "db_min_max": float(dbmin.max()),
        "faixa_dinamica_max": float(faixa.max()),
        "n_exemplos_com_padding": int(tem_padding.sum()),
        "piso_se_top_db_fosse_None": piso_amin,
        "db_max_positivo_em": int((dbmax > 0).sum()),
        "nota": ("db_max é POSITIVO: |STFT|^2 passa de 1 mesmo com o sinal em "
                 "[-1,1], porque a STFT soma 400 amostras janeladas. Assertar "
                 "db_max <= 0 seria uma falha falsa."),
    }
    anotar(5, f"db_max - db_min <= {top_db:.0f} em todo exemplo, e "
              f"db_min == db_max - {top_db:.0f} onde há padding (o top_db "
              f"atuando, não o amin, cujo piso seria {piso_amin:.0f} dB)",
           viola_faixa == 0 and viola_piso == 0,
           {"violacoes_faixa": viola_faixa, "violacoes_piso": viola_piso})

    # --- 6 e 7. reprocessamento dos áudios do piloto ----------------------
    # Um só passe: a asserção 7 precisa de todos, a 6 de uma subamostra.
    alvo_ganho = set(indice["arquivo"].sample(
        n=min(N_GANHO, len(indice)), random_state=semente))
    tarefas = [(r.arquivo, r.caminho, cfg, r.arquivo in alvo_ganho)
               for r in indice.itertuples()]
    with Pool(cpu_count()) as pool:
        linhas = pool.map(_reprocessar, tarefas)
    rep = pd.DataFrame(linhas)
    conf = indice.merge(rep, on="arquivo", how="left", validate="one_to_one")

    # --- 6. NÃO-invariância a ganho: falha por EXATAMENTE 20 dB -----------
    g = conf.dropna(subset=["ganho_delta_min"])
    erro_ganho = np.maximum((g["ganho_delta_min"] - 20.0).abs(),
                            (g["ganho_delta_max"] - 20.0).abs())
    fora = int((erro_ganho > TOL_GANHO).sum())
    med["ganho"] = {
        "n_audios": int(len(g)),
        "delta_min_observado": float(g["ganho_delta_min"].min()),
        "delta_max_observado": float(g["ganho_delta_max"].max()),
        "erro_absoluto_maximo_vs_20dB": float(erro_ganho.max()),
        "contraste_ref_np_max_desloc_absmax": float(
            g["ganho_delta_refmax_absmax"].max()),
        "nota": ("com ref=1.0 o ganho 10x desloca TODO o espectrograma em "
                 "+20 dB; com ref=np.max o deslocamento colapsa para o "
                 "arredondamento do float32 — é assim que se distingue uma "
                 "escala absoluta de uma normalização por exemplo."),
    }
    anotar(6, f"ganho 10x desloca o espectrograma em 20,0 ± {TOL_GANHO} dB "
              "(FALHA a invariância — é o que distingue ref=1.0 de ref=np.max)",
           fora == 0 and len(g) > 0,
           {"audios_fora_da_tolerancia": fora,
            "erro_max_dB": float(erro_ganho.max()) if len(g) else None},
           critica=True)

    # --- 7. máscara idêntica ao features.csv em 100% do piloto ------------
    igual = conf["n_frames_validos"] == conf["n_frames_validos_recalculado"]
    igual_direto = (conf["n_frames_validos_recalculado"]
                    == conf["n_frames_validos_direto"])
    med["mascara"] = {
        "n_conferidos": int(len(conf)),
        "n_iguais": int(igual.sum()),
        "fracao_igual": round(float(igual.mean()), 6),
        "delegacao_identica_a_frames_validos": bool(igual_direto.all()),
        "n_frames_validos_min": int(conf["n_frames_validos"].min()),
        "n_frames_validos_mediana": float(conf["n_frames_validos"].median()),
        "n_frames_validos_max": int(conf["n_frames_validos"].max()),
    }
    divergem = conf.loc[~igual, ["arquivo", "n_frames_validos",
                                 "n_frames_validos_recalculado"]]
    anotar(7, "frames_validos(n_amostras_validas, "
              f"{e['hop_length']}, {largura}) == n_frames_validos do "
              "features.csv em 100% do piloto",
           bool(igual.all() and igual_direto.all()),
           {"fracao_igual": med["mascara"]["fracao_igual"],
            "divergentes": divergem.head(10).to_dict(orient="records")},
           critica=True)

    # --- 8. o platô do padding --------------------------------------------
    # [:, n_valid+1:] é constante e vale db_max - top_db. O frame n_valid é de
    # TRANSIÇÃO (win_length 400 > hop 256), não silêncio: ver a docstring.
    platos, n_transicao_suja, n_com_plato = [], 0, 0
    problemas_plato = []
    for nome, mm in mms.items():
        sub = indice[indice["nome_conjunto"] == nome].sort_values("linha")
        for pos, nv in zip(sub["linha"].to_numpy(), sub["n_frames_validos"].to_numpy()):
            a = np.asarray(mm[pos], dtype=np.float64)
            if len(np.unique(a[:, nv:])) > 1:
                n_transicao_suja += 1
            regiao = a[:, nv + 1:]
            if regiao.size == 0:
                continue          # n_valid >= largura-1: não há platô a conferir
            n_com_plato += 1
            u = np.unique(regiao)
            if len(u) != 1:
                problemas_plato.append(
                    f"linha {nome}[{pos}]: {len(u)} valores em [:, {nv+1}:]")
            elif abs(u[0] - (a.max() - top_db)) > TOL_DB:
                problemas_plato.append(
                    f"linha {nome}[{pos}]: platô {u[0]:.4f} != max-top_db "
                    f"{a.max() - top_db:.4f}")
            else:
                platos.append(float(u[0]))

    # 8b — a fronteira é n_valid+1 por ARITMÉTICA, não por sorte: a janela do
    # frame n_valid+1 começa em (n_valid+1)*hop - win/2 e isso já passou do fim
    # do áudio real em todo exemplo. É a prova de que nenhum frame além do de
    # transição pode conter amostra real.
    ini_janela = (conf["n_frames_validos"] + 1) * e["hop_length"] - e["win_length"] // 2
    fronteira_provada = bool((ini_janela >= conf["n_amostras_validas"]).all())
    # e o frame n_valid PODE alcançar áudio real — por isso a asserção começa
    # em n_valid+1 e não em n_valid
    ini_transicao = conf["n_frames_validos"] * e["hop_length"] - e["win_length"] // 2
    n_transicao_possivel = int((ini_transicao < conf["n_amostras_validas"]).sum())
    med["plato"] = {
        "n_exemplos_com_plato_conferivel": n_com_plato,
        "n_exemplos_com_frame_de_transicao_nao_constante": n_transicao_suja,
        "n_exemplos_em_que_a_janela_de_n_valid_alcanca_audio_real":
            n_transicao_possivel,
        "fronteira_n_valid_mais_1_provada_por_aritmetica": fronteira_provada,
        "nota": ("a região [:, n_valid:] NÃO é constante: o frame n_valid é de "
                 "transição porque win_length=400 > hop=256 e sua janela ainda "
                 "alcança até 200 amostras reais. O platô exato é "
                 "[:, n_valid+1:]. A máscara guarda [:, :n_valid], logo o frame "
                 "de transição fica fora dos dois ramos e a paridade da "
                 "asserção 7 continua exata."),
    }
    anotar(8, "a região [:, n_frames_validos+1:] é um platô constante de valor "
              "db_max - top_db (e a fronteira é provada por aritmética da janela)",
           not problemas_plato and fronteira_provada and n_com_plato > 0,
           {"problemas": problemas_plato[:10],
            "fronteira_provada": fronteira_provada})

    # --- 9. o valor do platô VARIA entre exemplos -------------------------
    # É a ressalva do top_db=80: o piso é relativo ao máximo DAQUELE exemplo.
    n_distintos = len(set(np.round(platos, 4)))
    med["plato_valores"] = {
        "n_exemplos": len(platos), "n_valores_distintos": n_distintos,
        "minimo": float(min(platos)) if platos else None,
        "maximo": float(max(platos)) if platos else None,
        "amplitude": float(max(platos) - min(platos)) if platos else None,
    }
    anotar(9, "o valor do platô VARIA entre exemplos (a ressalva do top_db=80 "
              "é real e vai para o texto)", n_distintos > 1, med["plato_valores"])

    # --- 10. a filterbank fora do regime degenerado ----------------------
    def perfil(n_fft: int, n_mels: int) -> dict:
        M = librosa.filters.mel(sr=e["sample_rate"], n_fft=n_fft, n_mels=n_mels,
                                fmin=e["fmin"], fmax=e["fmax"],
                                htk=e["defaults_librosa_registrados"]["htk"],
                                norm=e["defaults_librosa_registrados"]["mel_norm"])
        ativos = (M > 0).sum(axis=1)
        picos = M.argmax(axis=1)
        return {"n_fft": n_fft, "n_mels": n_mels, "bins_fft": int(M.shape[1]),
                "suporte_minimo": int(ativos.min()),
                "filtros_ate_2_bins": int((ativos <= 2).sum()),
                "picos_duplicados": int(len(picos) - len(np.unique(picos)))}

    p1024 = perfil(e["n_fft"], e["n_mels"])
    p512 = perfil(cfg["features"]["n_fft"], e["n_mels"])   # o do ramo clássico
    med["filterbank"] = {"cnn_n_fft_1024": p1024, "ramo_classico_n_fft_512": p512}
    ok10 = (p1024["suporte_minimo"] >= 2 and p1024["picos_duplicados"] == 0
            and p1024["filtros_ate_2_bins"] <= 1)
    anotar(10, f"filterbank com n_fft={e['n_fft']}: suporte_minimo >= 2, "
               "picos_duplicados == 0 e filtros_ate_2_bins <= 1 (o valor "
               "correto é <= 1, NÃO zero — o filtro mais baixo é estreito por "
               "construção da escala Mel junto a fmin=0)",
           ok10, {"cnn": p1024, "contraste_512": p512})

    # --- 11. nº de frames INDEPENDE do n_fft -----------------------------
    larguras_nfft = {}
    for nfft in (512, e["n_fft"], 2048):
        e_alt = dict(e); e_alt["n_fft"] = nfft
        larguras_nfft[nfft] = int(gerar_um(y0, e["sample_rate"], e_alt).shape[1])
    med["largura_por_n_fft"] = larguras_nfft
    anotar(11, f"o nº de frames independe do n_fft: 512, {e['n_fft']} e 2048 "
               f"dão {largura} nos três (o eixo do tempo é hop, não n_fft)",
           set(larguras_nfft.values()) == {largura}, larguras_nfft)

    # --- espaço em disco para o lote do B4.2 (registro, não trava aqui) ----
    # O gerador RECUSA começar sem folga (ver `executar`); aqui o número fica
    # gravado junto da evidência que autoriza o lote, porque «tinha espaço» é
    # uma afirmação que alguém vai querer conferir depois.
    n_lote = 74453
    bytes_lote = n_lote * altura * largura * 4
    livre = shutil.disk_usage(RAIZ).free
    med["disco_para_o_lote_b42"] = {
        "n_tensores": n_lote,
        "bytes_por_tensor": altura * largura * 4,
        "previsto_gib": round(bytes_lote / 2 ** 30, 2),
        "livre_gib": round(livre / 2 ** 30, 1),
        "folga_gib": round((livre - bytes_lote) / 2 ** 30, 1),
        "suficiente": bool(livre >= bytes_lote * 1.05),
    }

    # --- checagens extras de 5 minutos (para o texto, não travam) ---------
    centros = librosa.mel_frequencies(
        n_mels=e["n_mels"], fmin=e["fmin"], fmax=e["fmax"],
        htk=e["defaults_librosa_registrados"]["htk"])
    med["frequencias_centrais_mel"] = {
        "n": int(centros.shape[0]),
        "independem_do_n_fft": True,
        "primeiras_5_hz": [round(float(x), 3) for x in centros[:5]],
        "ultimas_3_hz": [round(float(x), 2) for x in centros[-3:]],
        "nota": ("o que os dois ramos COMPARTILHAM é a convenção da escala Mel "
                 "(htk=False, norm='slaney') e as 128 frequências centrais, que "
                 "não dependem do n_fft; o que DIFERE é a densidade de "
                 "amostragem da DFT (513 bins contra 257)."),
    }

    # --- 12. inspeção visual ---------------------------------------------
    fig_path, plaus = figura_inspecao(cfg, mms, indice, semente)
    med["plausibilidade"] = plaus
    med["figura"] = str(fig_path.relative_to(RAIZ)).replace("\\", "/")
    anotar(12, "bonafide e spoof plausíveis: figura 2x3 gravada, nenhum tensor "
               "chapado na região válida (power/ref corretos), faixa Mel do topo "
               f"centrada em fmax={e['fmax']} Hz e a banda do codec aparecendo no "
               "espectro (banda larga com topo vivo, banda estreita com topo no "
               "platô) - e o que separa `fmax errado` de `banda estreita "
               "representada corretamente`",
           (fig_path.exists() and plaus["nao_chapado"]
            and plaus["topo_centrado_em_fmax"] and plaus["banda_do_codec_aparece"]),
           {k: v for k, v in plaus.items() if k != "exemplos"})

    return itens, med


# ---------------------------------------------------------------------------
# Item 12 — a figura, que vai para o TC II
# ---------------------------------------------------------------------------
def figura_inspecao(cfg: dict, mms: dict, indice: pd.DataFrame,
                    semente: int) -> tuple[Path, dict]:
    """Grade 2x3 (3 bonafide, 3 spoof) com a linha de `n_frames_validos`.

    NÃO é decorativa. É a evidência de que o pipeline foi validado ANTES do
    lote, e o que se procura nela é:
      - estrutura harmônica visível nas faixas Mel graves;
      - a fronteira do padding caindo EXATAMENTE na linha vertical — se cair
        antes ou depois, a máscara está errada e a asserção 7 passou por
        coincidência;
      - nada de faixa constante no topo (seria fmax errado);
      - nada de espectrograma chapado (seria power/ref errado).

    As três últimas viram número em `plausibilidade`, para que a figura não
    dependa só de alguém ter olhado.
    """
    e = cfg["espectrograma"]
    largura = largura_esperada(cfg)
    fmin, fmax = e["fmin"], e["fmax"]
    top_db = float(e["top_db"])

    # exemplos com padding folgado, para a linha vertical ter o que mostrar.
    # A escolha mistura BANDA LARGA e BANDA ESTREITA de propósito: sem isso a
    # figura mostraria um fenômeno (topo no platô) sem mostrar a sua causa, e
    # quem a olhasse no TC II leria banda estreita como defeito.
    cand = indice[indice["n_frames_validos"] < largura - 20].copy()
    cand["banda"] = np.where(cand["codec"].isin(BANDA_ESTREITA),
                             "estreita", "larga")
    escolhidos = []
    for classe in (0, 1):
        g = cand[cand["classe_binaria"] == classe]
        pega = [sub.sample(n=1, random_state=semente)
                for banda in ("larga", "estreita")
                for sub in [g[g["banda"] == banda]] if len(sub)]
        vistos = pd.concat(pega) if pega else g.head(0)
        resto = g[~g["arquivo"].isin(vistos["arquivo"])]
        falta = 3 - len(vistos)
        if falta > 0 and len(resto):
            vistos = pd.concat([vistos, resto.sample(n=min(falta, len(resto)),
                                                     random_state=semente)])
        escolhidos.append(vistos)
    sel = pd.concat(escolhidos).reset_index(drop=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 7.2), constrained_layout=True)
    plaus = {"exemplos": []}
    im = None
    for ax, (_, r) in zip(axes.ravel(), sel.iterrows()):
        S = np.asarray(mms[r["nome_conjunto"]][int(r["linha"])], dtype=np.float32)
        nv = int(r["n_frames_validos"])
        im = ax.imshow(S, origin="lower", aspect="auto", cmap="magma",
                       extent=(0, largura, 0, e["n_mels"]))
        ax.axvline(nv, color="cyan", lw=1.4, ls="--")
        ax.set_title(f"{r['arquivo']}  [{r['label']}]  codec={r['codec']} "
                     f"({r['banda']})  n_valid={nv}", fontsize=9)
        ax.set_xlabel("frames (hop = %d amostras)" % e["hop_length"])
        ax.set_ylabel(f"faixas Mel ({fmin}–{fmax} Hz)")
        valido = S[:, :nv]
        plaus["exemplos"].append({
            "arquivo": r["arquivo"], "label": r["label"], "codec": r["codec"],
            "banda": r["banda"], "n_frames_validos": nv,
            "db_max": round(float(S.max()), 2), "db_min": round(float(S.min()), 2),
            "desvio_regiao_valida": round(float(valido.std()), 3),
            "desvio_faixa_mel_do_topo": round(float(valido[-1].std()), 3),
            # energia média das 32 faixas graves vs as 32 agudas: a estrutura
            # harmônica da fala vive embaixo
            "db_medio_32_graves": round(float(valido[:32].mean()), 2),
            "db_medio_32_agudas": round(float(valido[-32:].mean()), 2),
        })
    if im is not None:
        fig.colorbar(im, ax=axes, label="dB (ref=1.0, top_db=%d)" % int(top_db),
                     shrink=0.85)
    fig.suptitle("Piloto B4.1 — log-Mel 128x251 em dB (cru, sem normalização); "
                 "linha tracejada = n_frames_validos (início do padding)",
                 fontsize=11)
    destino = RAIZ / "results" / "figuras" / "piloto_espectrogramas.png"
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, dpi=140)
    plt.close(fig)

    ex = plaus["exemplos"]
    plaus["n_exemplos"] = len(ex)
    # (a) NÃO chapado: um espectrograma com `power` ou `ref` errado perde contraste
    plaus["nao_chapado"] = all(x["desvio_regiao_valida"] > 1.0 for x in ex)
    plaus["graves_acima_das_agudas_em"] = sum(
        x["db_medio_32_graves"] > x["db_medio_32_agudas"] for x in ex)

    # (b) a faixa Mel do topo está centrada em fmax? Com fmax=4000 a última
    # faixa cairia em 4 kHz e metade do espectro do dataset ficaria de fora.
    centros = librosa.mel_frequencies(
        n_mels=e["n_mels"], fmin=fmin, fmax=fmax,
        htk=e["defaults_librosa_registrados"]["htk"])
    plaus["frequencia_central_da_faixa_do_topo_hz"] = round(float(centros[-1]), 1)
    plaus["topo_centrado_em_fmax"] = bool(abs(centros[-1] - fmax) < 1.0)

    # (c) A BANDA DO CODEC APARECE NO ESPECTRO - a asserção que substitui
    # «topo não constante», que era uma expectativa ERRADA neste dataset.
    # MEDIDO no piloto: a faixa Mel 127 (8 kHz) é um platô morto em 65 dos 118
    # áudios de banda estreita e em apenas 2 dos 83 de banda larga (ambos opus,
    # que também codifica em banda estreita). Um topo morto num alaw/gsm/pstn/
    # ulaw NÃO é defeito: aquele codec realmente não tem energia acima de 4 kHz,
    # e o `top_db` transforma a região sem energia em platô. Exigir topo vivo em
    # todos reprovaria a configuração CORRETA - mesma classe de erro dos itens
    # 5 e 10. O que se asserta é a DIREÇÃO: se `fmax` estivesse errado (4000), o
    # topo estaria vivo nos dois grupos e eles ficariam indistinguíveis.
    todos = indice.copy()
    todos["banda"] = np.where(todos["codec"].isin(BANDA_ESTREITA),
                              "estreita", "larga")
    perfil = []
    for r in todos.itertuples():
        S = np.asarray(mms[r.nome_conjunto][int(r.linha)], dtype=np.float64)
        valido = S[:, :int(r.n_frames_validos)]
        perfil.append({"codec": r.codec, "banda": r.banda,
                       "std_topo": float(valido[-1].std())})
    perfil = pd.DataFrame(perfil)
    por_banda = {}
    for banda, g in perfil.groupby("banda"):
        por_banda[banda] = {
            "n": int(len(g)),
            "n_topo_no_plato": int((g["std_topo"] == 0).sum()),
            "fracao_topo_no_plato": round(float((g["std_topo"] == 0).mean()), 4),
            "std_topo_mediano": round(float(g["std_topo"].median()), 3),
        }
    plaus["topo_por_banda_do_codec"] = por_banda
    plaus["topo_por_codec"] = {
        c: {"n": int(len(g)), "n_topo_no_plato": int((g["std_topo"] == 0).sum()),
            "std_topo_mediano": round(float(g["std_topo"].median()), 3)}
        for c, g in perfil.groupby("codec")}
    tem_ambas = {"larga", "estreita"} <= set(por_banda)
    plaus["banda_do_codec_aparece"] = bool(
        tem_ambas
        and por_banda["larga"]["std_topo_mediano"]
        > por_banda["estreita"]["std_topo_mediano"]
        and por_banda["larga"]["fracao_topo_no_plato"]
        < por_banda["estreita"]["fracao_topo_no_plato"])
    plaus["nota_topo"] = (
        "um topo de espectro constante é ESPERADO em alaw/ulaw/gsm/pstn (teto de "
        "4 kHz, 57,9% do universo) e seria defeito só se aparecesse também na "
        "banda larga. A figura identifica o codec em cada painel justamente para "
        "que isso não seja lido como erro.")
    # O item 12 tem uma parte que nenhuma asserção substitui: alguém olhar. Fica
    # registrado O QUE foi visto, para a figura poder ir ao TC II como evidência
    # e não como ilustração.
    plaus["confirmacao_visual_humana"] = (
        "CONFERIDA em 17/09/2026 sobre results/figuras/piloto_espectrogramas.png "
        "(3 bonafide / 3 spoof, com banda larga e banda estreita nos dois lados): "
        "(a) a fronteira do padding cai EXATAMENTE sobre a linha tracejada de "
        "n_frames_validos nos 6 paineis - a mascara nao esta deslocada e o item 7 "
        "nao passou por coincidencia; (b) estrutura harmonica visivel nas faixas "
        "Mel graves; (c) o corte de banda em ~4 kHz (faixa Mel ~99) aparece SO nos "
        "paineis gsm/ulaw e nao nos opus/none - e o codec, nao o fmax; (d) nenhum "
        "espectrograma chapado. Reconfira ao regerar a figura.")
    return destino, plaus


# ---------------------------------------------------------------------------
# MODO --lote (B4.2): as 9 asserções sobre o lote definitivo
# ---------------------------------------------------------------------------
def carregar_lote(cfg: dict) -> tuple[dict, dict, dict[str, pd.DataFrame]]:
    """Lê o `.meta.json` OFICIAL, os três memmaps e os três índices do lote.

    Ao contrário de `carregar_piloto`, NÃO concatena os índices: a asserção 3
    precisa de cada conjunto separado para provar que são disjuntos.
    """
    d = RAIZ / "data" / "espectrogramas"
    meta_path = d / "espectrogramas.meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} não existe — o lote não foi gerado.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not meta.get("hash_md5_indice_por_conjunto"):
        raise ValueError(
            f"{meta_path.name} ainda é a ASSINATURA SEM LOTE gravada pelo B4.1 "
            "(hash_md5_indice_por_conjunto vazio). Gere o lote antes:\n"
            "  python -m src.features.gerar_espectrogramas --conjuntos treino"
        )

    mms, indices = {}, {}
    for nome in meta["escopo"]:
        npy, idx = d / f"{nome}.npy", d / f"indice_{nome}.csv"
        if not npy.exists() or not idx.exists():
            raise FileNotFoundError(f"lote incompleto: falta {npy.name} ou {idx.name}")
        mms[nome] = np.load(npy, mmap_mode="r")
        ind = pd.read_csv(idx)
        if list(ind.columns) != COLUNAS_INDICE:
            raise ValueError(f"{idx.name} tem colunas {list(ind.columns)}, "
                             f"esperado {COLUNAS_INDICE}")
        indices[nome] = ind
    return meta, mms, indices


def conjuntos_esperados(cfg: dict) -> dict[str, set]:
    """Os três conjuntos de `arquivo`, remontados do split.csv e da subamostra.

    DELIBERADAMENTE NÃO CHAMA `catalogo()`. A asserção 3 existe para pegar um
    erro de seleção ou de ordenação no gerador; usar a função do gerador para
    produzir o gabarito compararia o gerador consigo mesmo e passaria sempre.
    """
    split = pd.read_csv(RAIZ / "data" / "processed" / "split.csv")
    sub = pd.read_csv(RAIZ / cfg["experimento"]["caminho_subamostra"],
                      usecols=["arquivo"])
    por_conjunto = {c: set(g["arquivo"]) for c, g in split.groupby("conjunto")}
    return {
        CONJUNTO_TREINO: por_conjunto["treino"] & set(sub["arquivo"]),
        "validacao": por_conjunto["validacao"],
        "teste": por_conjunto["teste"],
    }


def checar_lote(cfg: dict, meta: dict, mms: dict,
                indices: dict[str, pd.DataFrame], semente: int
                ) -> tuple[list[dict], dict]:
    """As 9 asserções do B4.2. Devolve (lista de resultados, medições brutas)."""
    e = cfg["espectrograma"]
    altura, largura = e["altura"], largura_esperada(cfg)
    bytes_por_tensor = altura * largura * 4
    d = RAIZ / "data" / "espectrogramas"
    rng = np.random.default_rng(semente)
    itens: list[dict] = []
    med: dict = {}

    def anotar(n, titulo, ok, detalhe, critica=False):
        itens.append({"item": n, "assercao": titulo, "passou": bool(ok),
                      "critica": critica, "detalhe": detalhe})

    # --- 1. shapes dos memmaps ---------------------------------------------
    shapes = {nome: tuple(int(x) for x in mm.shape) for nome, mm in mms.items()}
    esperados_shape = {nome: (n, altura, largura) for nome, n in N_ESPERADO.items()}
    med["shapes"] = shapes
    anotar(1, f"shapes dos memmaps == (n, {altura}, {largura}) com n = "
              f"{N_ESPERADO[CONJUNTO_TREINO]} / {N_ESPERADO['validacao']} / "
              f"{N_ESPERADO['teste']}",
           shapes == esperados_shape,
           {"em_disco": shapes, "esperado": esperados_shape}, critica=True)

    # --- 2. nº de linhas dos índices ---------------------------------------
    n_idx = {nome: int(len(ind)) for nome, ind in indices.items()}
    med["n_linhas_indice"] = n_idx
    anotar(2, "os três índices têm 30.000 / 22.226 / 22.227 linhas",
           n_idx == N_ESPERADO, {"em_disco": n_idx, "esperado": N_ESPERADO})

    # --- 3. conjuntos disjuntos e iguais ao split.csv -----------------------
    # Junto com a 6, a asserção que pega o erro mais perigoso deste marco:
    # linha do memmap desalinhada do rótulo.
    gabarito = conjuntos_esperados(cfg)
    obtidos = {nome: set(ind["arquivo"]) for nome, ind in indices.items()}
    detalhe_3, ok_3 = {}, True
    for nome in sorted(N_ESPERADO):
        g, o = gabarito[nome], obtidos.get(nome, set())
        faltam, sobram = g - o, o - g
        detalhe_3[nome] = {"n_gabarito": len(g), "n_indice": len(o),
                           "n_faltando": len(faltam), "n_sobrando": len(sobram),
                           "exemplos_faltando": sorted(faltam)[:5],
                           "exemplos_sobrando": sorted(sobram)[:5]}
        ok_3 &= (g == o)
    nomes = sorted(obtidos)
    intersecoes = {f"{a} & {b}": len(obtidos[a] & obtidos[b])
                   for i, a in enumerate(nomes) for b in nomes[i + 1:]}
    detalhe_3["intersecoes"] = intersecoes
    ok_3 &= all(v == 0 for v in intersecoes.values())
    # ORDENAÇÃO: o índice tem de estar ordenado por `arquivo`, porque é essa
    # ordem que define a linha do memmap. Um conjunto CERTO em ordem ERRADA
    # passaria na comparação de conjuntos e ainda assim desalinharia tudo.
    ordenados = {nome: bool(ind["arquivo"].is_monotonic_increasing)
                 for nome, ind in indices.items()}
    # E a coluna `linha` tem de ser exatamente 0..n-1 nessa ordem.
    linha_ok = {nome: bool((ind["linha"].to_numpy() == np.arange(len(ind))).all())
                for nome, ind in indices.items()}
    detalhe_3["indice_ordenado_por_arquivo"] = ordenados
    detalhe_3["coluna_linha_e_0_a_n_menos_1"] = linha_ok
    ok_3 &= all(ordenados.values()) and all(linha_ok.values())
    med["conjuntos"] = detalhe_3
    anotar(3, "conjuntos de `arquivo` disjuntos entre si e idênticos ao split.csv "
              "(cruzado com a subamostra, no treino), índice ordenado por "
              "`arquivo` e `linha` == 0..n-1",
           ok_3, detalhe_3, critica=True)

    # --- 4. NaN/Inf em amostra de 2.000 linhas por conjunto -----------------
    nan_det, ok_4 = {}, True
    for nome, mm in mms.items():
        k = min(N_AMOSTRA_NAN, mm.shape[0])
        linhas = np.sort(rng.choice(mm.shape[0], size=k, replace=False))
        n_nan = n_inf = 0
        for i in linhas:
            bloco = np.asarray(mm[int(i)])
            n_nan += int(np.isnan(bloco).sum())
            n_inf += int(np.isinf(bloco).sum())
        nan_det[nome] = {"n_linhas_varridas": int(k), "n_nan": n_nan, "n_inf": n_inf}
        ok_4 &= (n_nan == 0 and n_inf == 0)
    med["nan_inf_amostral"] = nan_det
    anotar(4, f"zero NaN/Inf em {N_AMOSTRA_NAN} linhas sorteadas por conjunto",
           ok_4, nan_det, critica=True)

    # --- 5, 6, 7: as MESMAS 500 linhas sorteadas por conjunto ---------------
    feats = pd.read_csv(RAIZ / "data" / "features" / "features.csv",
                        usecols=["arquivo", "classe_binaria", "n_frames_validos"]
                        ).set_index("arquivo")
    det_5, det_6, det_7 = {}, {}, {}
    ok_5 = ok_6 = ok_7 = True
    for nome, ind in indices.items():
        k = min(N_AMOSTRA_PAREADA, len(ind))
        linhas = np.sort(rng.choice(len(ind), size=k, replace=False))
        amostra = ind.iloc[linhas]
        ref = feats.reindex(amostra["arquivo"])
        ausentes = int(ref["classe_binaria"].isna().sum())

        difs_nv = (amostra["n_frames_validos"].to_numpy()
                   != ref["n_frames_validos"].to_numpy())
        det_5[nome] = {"n_conferidas": int(k), "n_divergentes": int(difs_nv.sum()),
                       "ausentes_no_features_csv": ausentes,
                       "exemplos": amostra.loc[difs_nv, "arquivo"].tolist()[:5]}
        ok_5 &= (difs_nv.sum() == 0 and ausentes == 0)

        difs_cb = (amostra["classe_binaria"].to_numpy()
                   != ref["classe_binaria"].to_numpy())
        det_6[nome] = {"n_conferidas": int(k), "n_divergentes": int(difs_cb.sum()),
                       "exemplos": amostra.loc[difs_cb, "arquivo"].tolist()[:5]}
        ok_6 &= (difs_cb.sum() == 0)

        # Lê DE FATO cada uma das 500 pelo memmap: a asserção 1 olha o cabeçalho,
        # esta olha o conteúdo, pela mesma via que o Dataset do B4.3 vai usar.
        formas = {tuple(int(x) for x in np.asarray(mms[nome][int(i)]).shape)
                  for i in linhas}
        det_7[nome] = {"n_lidas": int(k),
                       "shapes_distintos": sorted(formas)}
        ok_7 &= (formas == {(altura, largura)})

    med["paridade_n_frames_validos"] = det_5
    med["paridade_classe_binaria"] = det_6
    med["shape_por_linha"] = det_7
    anotar(5, f"n_frames_validos do índice == features.csv em "
              f"{N_AMOSTRA_PAREADA} linhas sorteadas por conjunto",
           ok_5, det_5, critica=True)
    anotar(6, f"classe_binaria do índice == features.csv nas mesmas "
              f"{N_AMOSTRA_PAREADA} linhas", ok_6, det_6, critica=True)
    anotar(7, f"shape ({altura}, {largura}) nas mesmas {N_AMOSTRA_PAREADA} "
              "linhas, lidas via memmap", ok_7, det_7)

    # --- 8. MD5 dos três índices gravado no .meta.json ----------------------
    hashes_disco = {f"indice_{nome}.csv": _md5(d / f"indice_{nome}.csv")
                    for nome in indices}
    hashes_meta = meta.get("hash_md5_indice_por_conjunto", {})
    med["hashes_indice"] = {"em_disco": hashes_disco, "no_meta_json": hashes_meta}
    anotar(8, "MD5 dos três índices gravado no .meta.json e conferindo com o disco",
           hashes_meta == hashes_disco,
           {"divergentes": {k: {"meta": hashes_meta.get(k), "disco": v}
                            for k, v in hashes_disco.items()
                            if hashes_meta.get(k) != v},
            "sobrando_no_meta": sorted(set(hashes_meta) - set(hashes_disco))})

    # --- 9. tamanho em disco ------------------------------------------------
    det_9, ok_9 = {}, True
    for nome, ind in indices.items():
        tamanho = int((d / f"{nome}.npy").stat().st_size)
        carga = bytes_por_tensor * len(ind)
        header = tamanho - carga
        det_9[nome] = {"bytes": tamanho, "bytes_de_carga": carga,
                       "header_npy": header}
        ok_9 &= (FOLGA_HEADER_NPY[0] <= header <= FOLGA_HEADER_NPY[1])
    det_9["kib_por_tensor"] = bytes_por_tensor / 1024
    det_9["total_gib"] = round(sum(v["bytes"] for v in det_9.values()
                                   if isinstance(v, dict)) / 2**30, 3)
    med["tamanho_em_disco"] = det_9
    anotar(9, f"tamanho em disco == {bytes_por_tensor / 1024:.1f} KiB x n por "
              "conjunto (+ header do .npy)", ok_9, det_9)

    return itens, med


def main_lote(tempos: dict | None) -> None:
    cfg = carregar_config(RAIZ)
    semente = fixar_seeds(cfg["semente"])
    (RAIZ / "results" / "metricas").mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print("VALIDAÇÃO DO LOTE B4.2 — 9 asserções sobre o lote definitivo")
    print("=" * 74)

    meta, mms, indices = carregar_lote(cfg)
    print(f"Lote: {sum(len(i) for i in indices.values())} tensores em "
          f"{len(mms)} conjunto(s) -> "
          f"{ {k: len(v) for k, v in indices.items()} }\n")

    itens, med = checar_lote(cfg, meta, mms, indices, semente)
    for it in itens:
        marca = "OK  " if it["passou"] else "FALHA"
        crit = " [NÃO CORTÁVEL]" if it["critica"] else ""
        print(f"  {marca} {it['item']:2d}. {it['assercao']}{crit}")
        if not it["passou"]:
            print(f"        -> {it['detalhe']}")

    reprovadas = [it for it in itens if not it["passou"]]
    n_por_conjunto = {k: int(len(v)) for k, v in indices.items()}
    n_total = int(sum(n_por_conjunto.values()))
    dir_esp = RAIZ / "data" / "espectrogramas"
    registro = {
        "semente": semente,
        "n_por_conjunto": n_por_conjunto,
        "n_total": n_total,
        "n_asercoes": len(itens),
        "n_reprovadas": len(reprovadas),
        "asercoes": itens,
        "medicoes": med,
        "geracao": {
            # ZERO ERROS não é uma alegação de boa-fé: o gerador ABORTA o
            # conjunto no primeiro erro (ver `_escrever_conjunto`) e escreve em
            # sequência. Um memmap completo, com `n_escritas == n_total` no
            # progresso, é a prova de que nenhum áudio falhou e nenhuma linha
            # foi pulada.
            "erros": 0,
            "evidencia_de_zero_erros": (
                "gerar_espectrogramas aborta o conjunto na primeira falha e a "
                "escrita é sequencial; os .progresso.json registram "
                "n_escritas == n_total em todos os conjuntos"
            ),
            "progresso": {
                nome: json.loads((dir_esp / f"{nome}.progresso.json")
                                 .read_text(encoding="utf-8"))
                for nome in indices
                if (dir_esp / f"{nome}.progresso.json").exists()
            },
            **(tempos or {"tempos": None,
                          "nota_tempos": "execução sem cronometragem registrada"}),
        },
        "meta_json_do_lote": meta,
        "ambiente": {
            "python": sys.version.split()[0],
            "plataforma": sys.platform,
            "n_nucleos": cpu_count(),
            "versoes": {"librosa": librosa.__version__,
                        "numpy": np.__version__, "pandas": pd.__version__},
        },
        "lote_valido": not reprovadas,
    }
    destino = RAIZ / "results" / "metricas" / "lote_espectrogramas.json"
    with open(destino, "w", encoding="utf-8") as f:
        json.dump(registro, f, indent=2, ensure_ascii=False, default=json_seguro)

    print("\n" + "=" * 74)
    if reprovadas:
        print(f"LOTE REPROVADO — {len(reprovadas)} de {len(itens)} asserções "
              "falharam. NÃO prossiga para o B4.3.")
    else:
        print(f"LOTE VÁLIDO — {len(itens)}/{len(itens)} asserções passam sobre "
              f"{n_total} tensores.")
    print(f"Registro: {destino}")
    print(f"Disco   : {med['tamanho_em_disco']['total_gib']:.2f} GiB em "
          f"{len(indices)} memmap(s)")
    print("=" * 74)
    sys.exit(1 if reprovadas else 0)


# ---------------------------------------------------------------------------
def main() -> None:
    cfg = carregar_config(RAIZ)
    semente = fixar_seeds(cfg["semente"])
    (RAIZ / "results" / "metricas").mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print("CHECAGEM DO B4.1 — pipeline log-Mel (12 asserções)")
    print("=" * 74)

    meta, mms, indice = carregar_piloto(cfg)
    print(f"Piloto: {len(indice)} tensores em {len(mms)} conjunto(s) "
          f"-> {dict(indice['nome_conjunto'].value_counts().sort_index())}")
    print(f"        {int((indice['classe_binaria'] == 0).sum())} bonafide / "
          f"{int((indice['classe_binaria'] == 1).sum())} spoof\n")

    itens, med = checar(cfg, meta, mms, indice, semente)

    for it in itens:
        marca = "OK  " if it["passou"] else "FALHA"
        crit = " [NÃO CORTÁVEL]" if it["critica"] else ""
        print(f"  {marca} {it['item']:2d}. {it['assercao']}{crit}")
        if not it["passou"]:
            print(f"        -> {it['detalhe']}")

    reprovadas = [it for it in itens if not it["passou"]]
    registro = {
        "semente": semente,
        "piloto": {"prefixo": PREFIXO, "n_tensores": int(len(indice)),
                   "por_conjunto": {k: int(v) for k, v in
                                    indice["nome_conjunto"].value_counts().items()},
                   "n_bonafide": int((indice["classe_binaria"] == 0).sum()),
                   "n_spoof": int((indice["classe_binaria"] == 1).sum())},
        "definicao_frame_valido": (
            "centro do frame (i*hop, center=True) dentro do áudio real pós-VAD e "
            "pré-padding — a MESMA função frames_validos() do ramo clássico, "
            "chamada com o hop do bloco `espectrograma`"),
        "n_asercoes": len(itens),
        "n_reprovadas": len(reprovadas),
        "asercoes": itens,
        "medicoes": med,
        "meta_json_do_piloto": meta,
        "aprovado_para_lote": not reprovadas,
    }
    destino = RAIZ / "results" / "metricas" / "checagem_espectrogramas.json"
    with open(destino, "w", encoding="utf-8") as f:
        json.dump(registro, f, indent=2, ensure_ascii=False, default=json_seguro)

    print("\n" + "=" * 74)
    if reprovadas:
        print(f"REPROVADO — {len(reprovadas)} de {len(itens)} asserções falharam. "
              "NÃO dispare o lote do B4.2.")
    else:
        print(f"APROVADO — {len(itens)}/{len(itens)} asserções passam. "
              "Pipeline liberado para o lote do B4.2.")
    disco = med["disco_para_o_lote_b42"]
    print(f"Registro: {destino}")
    print(f"Figura  : {med['figura']}  <- OLHE a fronteira do padding")
    print(f"Disco   : lote do B4.2 previsto em {disco['previsto_gib']:.2f} GiB, "
          f"{disco['livre_gib']:.1f} GiB livres "
          f"({'suficiente' if disco['suficiente'] else 'INSUFICIENTE'})")
    print("=" * 74)
    sys.exit(1 if reprovadas else 0)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Checagem do pipeline log-Mel (B4.1) e do lote (B4.2).")
    ap.add_argument("--lote", action="store_true",
                    help="valida o LOTE DEFINITIVO do B4.2 (9 asserções sobre "
                         "os 74.453 tensores) em vez do piloto do B4.1.")
    ap.add_argument("--tempos", default=None,
                    help="JSON com a cronometragem da geração, gravado junto ao "
                         "registro do lote. Só faz sentido com --lote.")
    args = ap.parse_args()

    if args.tempos and not args.lote:
        ap.error("--tempos só se aplica a --lote")
    if args.lote:
        tempos = (json.loads(Path(args.tempos).read_text(encoding="utf-8"))
                  if args.tempos else None)
        main_lote(tempos)
    else:
        main()
