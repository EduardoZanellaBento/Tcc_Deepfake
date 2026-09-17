"""
Auditoria numérica do DECISOES_PENDENTES_CNN.md (Bloco 4)
==========================================================

POR QUE ESTE SCRIPT EXISTE:
    O `DECISOES_PENDENTES_CNN.md` é o documento que vai ao orientador para
    destravar o B4.1 — e a geração dos espectrogramas é o "lote único do Bloco
    4": cara, demorada e não repetível dentro do cronograma. Um documento que
    pede decisão irreversível não pode ter número que ninguém recomputou.

    A primeira versão do documento (03/09/2026) trazia um ERRO exatamente
    desse tipo: inferia assimetria de PADDING entre as classes a partir da
    média de `prop_fala`, que mede outra coisa. Este script existe para que
    esse erro — e qualquer outro — seja pego por execução, não por leitura.

    É o mesmo papel de `scripts/validar_split_pos_lote.py` no Bloco 2: a prosa
    vira trava.

O QUE ELE CONFERE (cada bloco corresponde a uma afirmação do documento):
    1. procedência    — MD5 do features.csv congelado, nº de linhas, universo
    2. P1  largura    — n_frames_total é 251 e é único; 1 + 64000//256 == 251
    3. P2  padding    — estatísticas de n_frames_validos no UNIVERSO
    4. P2  assimetria — fração de padding POR CLASSE  (o número que a v1 errou)
                        e prop_fala POR CLASSE (o número que a v1 usou no lugar)
    5. P4  filtros    — degenerescência dos filtros mel via librosa.filters.mel
    6. P4  frames     — nº de frames independe de n_fft
    7. P5  escala     — qual conversão para dB o MFCC do ramo clássico já usa;
                        top_db faz o padding virar platô constante;
                        ref=np.max é normalização POR EXEMPLO;
                        onde ficam máximo e piso, com e sem ganho (nuance do
                        top_db, acrescentada na v4)
    8. eng. escopo    — contagens do split e total necessário para a CNN
    9. P7  classes    — desbalanceamento 9,00:1 nos quatro conjuntos
   10. eng. disco     — bytes por tensor e totais

O QUE ELE *NÃO* CONFERE (e o documento diz isso por escrito, na seção
«Até onde a trava alcança»): os deltas de 30,3% / 47,03% da P2, que vêm do
piloto (`checagem_mascaramento.json → parte_a_ab_controlado`) e não do universo,
e os valores em GiB/float16 do universo inteiro. Esses continuam conferidos à
mão. Qualquer número do documento fora dessas duas exceções tem de ter uma
contrapartida aqui — se um dia não tiver, o documento voltou a afirmar mais do
que este script sustenta.

A DISTINÇÃO QUE O BLOCO 4 EXISTE PARA PROTEGER (ver preprocessamento.py:180):
    `prop_fala`        = fração do áudio ORIGINAL que o VAD manteve;
    `n_frames_validos` = tamanho ABSOLUTO do que sobrou, contra o alvo de 4,0 s.
    Um áudio longo com prop_fala baixa pode ter MENOS padding que um curto com
    prop_fala alta. Uma NÃO prediz a outra — e é isso que o bloco 4 mede.

Saída: results/metricas/auditoria_decisoes_cnn.json
Rode a partir da raiz:  python -m scripts.auditar_decisoes_cnn
"""

import collections
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import librosa

from src.utils.config import carregar_config
from src.utils.serializacao import json_seguro

RAIZ = Path(__file__).resolve().parents[1]

# Assinatura do features.csv congelado pelo lote único (DOSSIE_LOTE_UNICO.md).
# Se este MD5 divergir, a auditoria inteira está falando de outro artefato.
MD5_FEATURES_ESPERADO = "51b2f439bf6f1e10237acbc620bb92d9"

# Razão de desbalanceamento que a P7 do documento afirma valer nos QUATRO
# conjuntos (treino, validação, teste e subamostra de 30k). É a única premissa
# numérica sobre a qual repousa uma recomendação de protocolo — a loss ponderada
# da CNN —, então ela é trava, e não número publicado. Arredondada em 2 casas
# porque as razões reais são 9,001 / 9,003 / 8,999 / 9,000: a divisão de um
# inteiro por outro nunca dá 9 exato com estes tamanhos.
RAZAO_DESBALANCEAMENTO = 9.00

# NOTA HISTÓRICA (v4): existiu aqui uma constante `TOL = 0.05`, descrita como
# "tolerância para comparar um valor recomputado com o publicado no documento".
# Ela nunca foi usada — este script NUNCA leu o .md — e a sua existência
# sustentava, no documento, a promessa falsa de que "todo número é recomputado e
# o script falha se qualquer afirmação deixar de valer". A promessa foi corrigida
# na v3 do documento e a constante morta foi removida na v4. A alternativa —
# fazer o script parsear o .md — foi descartada de propósito: acoplaria a trava à
# formatação da prosa. O caminho escolhido é o inverso e é o que o resto deste
# arquivo faz: cada afirmação que sustenta uma decisão vira uma asserção sobre os
# ARTEFATOS, e o documento cita o JSON.


def md5(caminho: Path) -> str:
    h = hashlib.md5()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def _degenerescencia_mel(sr: int, n_fft: int, n_mels: int) -> dict:
    """Quantos filtros mel são estreitos demais para a resolução da FFT.

    O QUE É MEDIDO, e por quê:
      - `suporte_minimo`: o menor nº de bins de FFT com peso não-nulo entre os
        n_mels filtros. Se for 1, existe filtro que lê UM único bin — ele não
        integra banda nenhuma, só copia um valor.
      - `filtros_ate_2_bins`: quantos filtros leem 2 bins ou menos. É a medida
        de quanto da "imagem" é interpolação dos mesmos poucos bins graves.
      - `picos_duplicados`: quantos GRUPOS de filtros compartilham o mesmo bin
        de pico. Filtros com o mesmo pico são linhas redundantes do
        espectrograma — informação repetida ocupando altura da entrada da CNN.

    O librosa NÃO avisa neste regime (nenhum filtro fica totalmente vazio),
    então a checagem tem de ser explícita.
    """
    M = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    n_bins_por_filtro = (M > 0).sum(axis=1)
    picos = M.argmax(axis=1)
    grupos = collections.Counter(picos)
    return {
        "n_fft": n_fft,
        "n_mels": n_mels,
        "bins_fft": int(M.shape[1]),
        "suporte_minimo": int(n_bins_por_filtro.min()),
        "filtros_ate_2_bins": int((n_bins_por_filtro <= 2).sum()),
        "picos_duplicados": int(sum(1 for v in grupos.values() if v > 1)),
    }


def main() -> None:
    cfg = carregar_config(RAIZ)
    problemas = []
    sr = cfg["audio"]["sample_rate"]
    dur = cfg["audio"]["duracao_segundos"]
    hop = cfg["features"]["hop_length"]
    win = cfg["features"]["win_length"]

    features_csv = RAIZ / "data" / "features" / "features.csv"
    split_csv = RAIZ / "data" / "processed" / "split.csv"
    sub_csv = RAIZ / cfg["experimento"]["caminho_subamostra"]

    # ---- (1) Procedência ---------------------------------------------------
    md5_feat = md5(features_csv)
    if md5_feat != MD5_FEATURES_ESPERADO:
        problemas.append(f"features.csv mudou: {md5_feat} != "
                         f"{MD5_FEATURES_ESPERADO} — as features não estão "
                         f"congeladas ou a auditoria é sobre outro artefato.")

    df = pd.read_csv(features_csv)
    n_linhas = int(len(df))
    if n_linhas != 148176:
        problemas.append(f"features.csv tem {n_linhas} linhas, esperado 148176.")

    # ---- (2) P1: largura do espectrograma ----------------------------------
    totais = sorted(int(v) for v in df["n_frames_total"].unique())
    frames_aritmetica = 1 + int(sr * dur) // hop
    if totais != [frames_aritmetica]:
        problemas.append(f"n_frames_total deveria ser [{frames_aritmetica}] em "
                         f"todas as linhas, veio {totais}.")

    # ---- (3) P2: quanto do tensor é padding, no universo --------------------
    v = df["n_frames_validos"]
    n_total = frames_aritmetica
    stats_validos = {
        "minimo": int(v.min()),
        "mediana": float(v.median()),
        "media": round(float(v.mean()), 2),
        "maximo": int(v.max()),
        "de_um_total_de": n_total,
        "pct_audios_com_mais_da_metade_em_padding":
            round(100 * float((v < n_total / 2).mean()), 2),
    }

    # ---- (4) P2: a assimetria ENTRE CLASSES --------------------------------
    # ESTE é o bloco que a v1 do documento errou. `frac_padding` é a grandeza
    # que de fato descreve "quanto do tensor da CNN é zero"; `prop_fala` é a
    # grandeza que a v1 usou no lugar dela. As duas são medidas e comparadas
    # lado a lado, de propósito, para que a confusão não possa voltar.
    df = df.assign(frac_padding=(n_total - df["n_frames_validos"]) / n_total)
    por_classe = {}
    for coluna in ("frac_padding", "prop_fala", "n_frames_validos"):
        g = df.groupby("label")[coluna].agg(["mean", "median"]).round(4)
        por_classe[coluna] = {cl: {"media": float(r["mean"]),
                                   "mediana": float(r["median"])}
                              for cl, r in g.iterrows()}

    fp = por_classe["frac_padding"]
    dif_padding_pp = round(
        100 * abs(fp["bonafide"]["media"] - fp["spoof"]["media"]), 2)
    pf = por_classe["prop_fala"]
    dif_prop_fala_pp = round(
        100 * abs(pf["bonafide"]["media"] - pf["spoof"]["media"]), 2)
    corr = round(float(df["prop_fala"].corr(df["n_frames_validos"])), 4)

    # A tese do documento corrigido: a QUANTIDADE de padding é simétrica entre
    # as classes (< 1 p.p.), enquanto prop_fala é fortemente assimétrica. Se um
    # dia isso deixar de valer, o texto da P2 precisa ser reescrito de novo.
    if dif_padding_pp >= 1.0:
        problemas.append(
            f"a fração de padding divergiu {dif_padding_pp} p.p. entre as "
            f"classes (>= 1 p.p.): a P2 do documento afirma simetria e "
            f"precisa ser revista.")
    if dif_prop_fala_pp <= 5.0:
        problemas.append(
            f"prop_fala divergiu só {dif_prop_fala_pp} p.p. entre as classes: "
            f"o texto que a descreve como fortemente assimétrica precisa ser "
            f"revisto.")

    # ---- (5) P4: degenerescência dos filtros mel ---------------------------
    grade_mel = [_degenerescencia_mel(sr, n_fft, n_mels)
                 for n_fft, n_mels in [(512, 128), (512, 80), (512, 64),
                                       (1024, 128), (1024, 80), (2048, 128)]]
    atual = next(g for g in grade_mel if g["n_fft"] == 512 and g["n_mels"] == 128)
    if atual["suporte_minimo"] > 1 or atual["filtros_ate_2_bins"] == 0:
        problemas.append("a configuração 512/128 deixou de ser degenerada — a "
                         "P4 do documento precisa ser revista.")

    # ---- (6) P4: o nº de frames não depende do n_fft -----------------------
    # É o que sustenta "subir o n_fft não mexe no eixo temporal": sem isso, a
    # opção A da P4 quebraria a paridade de 251 frames com o ramo clássico.
    silencio = np.zeros(int(sr * dur), dtype=np.float32)
    frames_por_nfft = {}
    for n_fft in (512, 1024, 2048):
        S = librosa.feature.melspectrogram(
            y=silencio, sr=sr, n_fft=n_fft, hop_length=hop, win_length=win,
            n_mels=cfg["espectrograma"]["n_mels"])
        frames_por_nfft[n_fft] = int(S.shape[1])
    if len(set(frames_por_nfft.values())) != 1:
        problemas.append(f"o nº de frames passou a depender do n_fft: "
                         f"{frames_por_nfft} — a P4 do documento precisa ser revista.")

    # ---- (7) P5: o que a escala dB faz com o padding e com o ganho ---------
    # Áudio sintético com padding conhecido: 1,5 s de sinal + zeros até 4,0 s.
    y = np.zeros(int(sr * dur), dtype=np.float32)
    rng = np.random.default_rng(cfg["semente"])
    n_validas = int(sr * 1.5)
    y[:n_validas] = rng.standard_normal(n_validas).astype(np.float32) * 0.3

    def _mel_db(sinal, ref, top_db=80, n_fft=1024):
        S = librosa.feature.melspectrogram(
            y=sinal, sr=sr, n_fft=n_fft, hop_length=hop, win_length=win,
            n_mels=cfg["espectrograma"]["n_mels"])
        return librosa.power_to_db(S, ref=ref, top_db=top_db)

    # `ref=1.0` é o DEFAULT de power_to_db — e é o que librosa.feature.mfcc
    # aplica internamente (`S = power_to_db(melspectrogram(...))`). Portanto é
    # esta, e não ref=np.max, a conversão que dá paridade com o ramo clássico.
    D = _mel_db(y, ref=1.0)
    # Um frame só é totalmente padding depois que a janela (win) deixa de
    # alcançar o áudio real; por isso a margem, em vez de cortar em n_validos.
    inicio_padding = int(np.ceil((n_validas + win) / hop)) + 1
    valores_no_padding = np.unique(D[:, inicio_padding:])
    padding_e_plato = bool(len(valores_no_padding) == 1)
    if not padding_e_plato:
        problemas.append("com top_db o padding deixou de ser platô constante — "
                         "a nota da P2/P5 sobre a máscara ser recuperável da "
                         "própria imagem precisa ser revista.")

    # ref=np.max normaliza cada exemplo pelo próprio máximo: mudar o ganho do
    # áudio não pode mudar nada. É o teste que prova que ref=np.max escolhe,
    # sem dizer, a família "normalização POR EXEMPLO" que a P3 rejeita — foi o
    # que motivou trocar a recomendação da P5 na v2 do documento.
    #
    # CUIDADO COM A PALAVRA "IDÊNTICO" (corrigido na v4). A invariância é medida
    # com `np.allclose`, não com igualdade: em float32 sobra ~1e-6 dB de
    # arredondamento entre as duas conversões. O documento chegou a dizer
    # "byte a byte idêntico", o que é forte demais — daí a diferença máxima ser
    # PUBLICADA abaixo, para que a prosa possa citar a ordem de grandeza certa
    # em vez de arredondá-la para zero.
    #
    # E CUIDADO AO CITAR O DÍGITO. Este valor, como o
    # `mfcc_deslocamento_demais_coeficientes` mais abaixo, é resíduo de
    # arredondamento: NÃO é propriedade do pipeline e muda com a versão de
    # numpy/BLAS. Medido em dois ambientes com o mesmo librosa 0.11.0: 8,34e-06
    # e 7,63e-06 no deslocamento do MFCC. Por isso o bloco `ambiente` do JSON
    # grava as versões ao lado dos números — quem citar o dígito no texto tem de
    # citar o ambiente junto; quem citar só a ordem de grandeza está seguro.
    D10 = _mel_db(y * 10.0, ref=1.0)
    db_max_ref = _mel_db(y, ref=np.max)
    db_max_ref_10x = _mel_db(y * 10.0, ref=np.max)
    ref_max_diferenca_maxima_db = float(np.abs(db_max_ref - db_max_ref_10x).max())
    ref_max_invariante_a_ganho = bool(np.allclose(db_max_ref_10x, db_max_ref))
    ref_abs_preserva_ganho = bool(not np.allclose(D10, D))
    if not (ref_max_invariante_a_ganho and ref_abs_preserva_ganho):
        problemas.append("o comportamento de ref=np.max / ref=1.0 divergiu do "
                         "descrito na P5 do documento.")

    # NUANCE DO top_db (publicada a partir da v4, a pedido da própria P5).
    # `top_db` do librosa corta em `max(S_dB) - top_db` DO PRÓPRIO EXEMPLO, e não
    # num piso absoluto do dataset. Duas consequências, as duas medidas aqui:
    #   (a) um ganho uniforme desloca máximo E piso pelo MESMO nº de dB — logo a
    #       energia global sobrevive, e `ref=1.0 + top_db=80` NÃO é normalização
    #       por exemplo. É o que sustenta a recomendação da P5;
    #   (b) o valor absoluto do platô de padding muda de exemplo para exemplo,
    #       o que interage com as estatísticas globais da P3.
    # Até a v3 estes quatro números estavam escritos no documento sem sair de
    # lugar nenhum — medição de mão não declarada. Agora saem daqui.
    escala_top_db = {
        "db_min": round(float(D.min()), 2),
        "db_max": round(float(D.max()), 2),
        "db_min_com_ganho_10x": round(float(D10.min()), 2),
        "db_max_com_ganho_10x": round(float(D10.max()), 2),
        "deslocamento_do_minimo_db": round(float(D10.min() - D.min()), 3),
        "deslocamento_do_maximo_db": round(float(D10.max() - D.max()), 3),
        "db_min_sem_top_db":
            round(float(_mel_db(y, ref=1.0, top_db=None).min()), 2),
    }

    # PARIDADE COM O RAMO CLÁSSICO: se o MFCC embute dB ABSOLUTO, o ganho do
    # áudio tem de sobreviver nele. Se um dia o MFCC passar a ser invariante a
    # ganho, o argumento de paridade da P5 cai e o texto precisa mudar.
    def _mfcc(sinal):
        return librosa.feature.mfcc(
            y=sinal, sr=sr, n_mfcc=cfg["features"]["n_mfcc"],
            n_fft=cfg["features"]["n_fft"], hop_length=hop, win_length=win)

    mfcc_1x, mfcc_10x = _mfcc(y), _mfcc(y * 10.0)
    # Reportamos a MAGNITUDE do deslocamento, não um booleano de allclose: em
    # float32 os coeficientes altos diferem por ~1e-5 de puro arredondamento, e
    # um allclose com tolerância default chamaria isso de "diferente". O que
    # importa é a razão entre as duas ordens de grandeza.
    desloc_c0 = float(np.abs(mfcc_1x[0] - mfcc_10x[0]).max())
    desloc_demais = float(np.abs(mfcc_1x[1:] - mfcc_10x[1:]).max())
    mfcc_sensivel_a_ganho = bool(desloc_c0 > 1.0)
    # O ganho vive no c0 (energia): um ganho uniforme desloca todas as bandas
    # mel pelo mesmo nº de dB, e a DCT joga um deslocamento constante inteiro
    # no c0. Os demais coeficientes só podem diferir por arredondamento.
    mfcc_ganho_isolado_no_c0 = bool(desloc_demais < 1e-3 < desloc_c0)
    if not (mfcc_sensivel_a_ganho and mfcc_ganho_isolado_no_c0):
        problemas.append(
            f"o MFCC não se comportou como dB absoluto (deslocamento c0="
            f"{desloc_c0:.3g}, demais={desloc_demais:.3g}) — o argumento de "
            f"paridade dB da P5 (ref=1.0) precisa ser revisto.")

    # ---- (8) Escopo: quantos espectrogramas realmente é preciso gerar ------
    split = pd.read_csv(split_csv)
    n_por_conjunto = {k: int(v) for k, v in
                      split["conjunto"].value_counts().items()}
    sub = pd.read_csv(sub_csv, usecols=["arquivo", "classe_binaria"])
    n_sub = int(len(sub))
    n_necessario = n_sub + n_por_conjunto["validacao"] + n_por_conjunto["teste"]

    # ---- (9) P7: desbalanceamento das classes -------------------------------
    # POR QUE ISTO VIROU TRAVA NA v4: a P7 recomenda loss ponderada na CNN, e a
    # recomendação repousa inteira sobre a razão 9,00:1 valer nos QUATRO
    # conjuntos — inclusive na subamostra que a CNN vai usar. Era o único número
    # do documento que sustentava uma decisão de protocolo e não tinha trava
    # nenhuma: estava conferido à mão, e a conferência à mão não sobrevive a uma
    # re-geração do split ou da subamostra.
    #
    # Fonte de cada coluna: `label` vem do features.csv (já carregado em `df`),
    # `conjunto` do split.csv, e a subamostra traz `classe_binaria` (1 = spoof).
    # O merge é INNER de propósito: se um dia o split e o features.csv deixarem
    # de cobrir o mesmo universo, as contagens caem e a trava dispara.
    classes = df[["arquivo", "label"]].merge(
        split[["arquivo", "conjunto"]], on="arquivo", how="inner")
    desbalanceamento = {}
    for conjunto, g in classes.groupby("conjunto"):
        n_spoof = int((g["label"] == "spoof").sum())
        n_bonafide = int((g["label"] == "bonafide").sum())
        desbalanceamento[str(conjunto)] = {
            "n": int(len(g)),
            "spoof": n_spoof,
            "bonafide": n_bonafide,
            "razao_spoof_por_bonafide": round(n_spoof / n_bonafide, 3),
        }
    n_spoof_sub = int((sub["classe_binaria"] == 1).sum())
    n_bonafide_sub = int((sub["classe_binaria"] == 0).sum())
    desbalanceamento["subamostra_30k"] = {
        "n": n_sub,
        "spoof": n_spoof_sub,
        "bonafide": n_bonafide_sub,
        "razao_spoof_por_bonafide": round(n_spoof_sub / n_bonafide_sub, 3),
    }

    fora_da_razao = {
        k: v["razao_spoof_por_bonafide"] for k, v in desbalanceamento.items()
        if round(v["razao_spoof_por_bonafide"], 2) != RAZAO_DESBALANCEAMENTO
    }
    if fora_da_razao:
        problemas.append(
            f"o desbalanceamento deixou de ser {RAZAO_DESBALANCEAMENTO:.2f}:1 "
            f"em {fora_da_razao} — a tabela da P7 e a recomendação de loss "
            f"ponderada precisam ser revistas.")

    # ---- (10) Disco --------------------------------------------------------
    altura = cfg["espectrograma"]["n_mels"]
    bytes_por_tensor = altura * n_total * 4
    disco = {
        "altura_x_largura": f"{altura} x {n_total}",
        "bytes_por_tensor_float32": bytes_por_tensor,
        "kib_por_tensor_float32": round(bytes_por_tensor / 1024, 1),
        "gb_necessario_float32": round(n_necessario * bytes_por_tensor / 1e9, 2),
        "gib_necessario_float32": round(n_necessario * bytes_por_tensor / 2**30, 2),
        "gb_necessario_float16": round(n_necessario * bytes_por_tensor / 2 / 1e9, 2),
        "gb_universo_float32": round(n_linhas * bytes_por_tensor / 1e9, 2),
    }

    registro = {
        "documento_auditado": "results/metricas/DECISOES_PENDENTES_CNN.md",
        "procedencia": {
            "md5_features_csv": md5_feat,
            "md5_confere": md5_feat == MD5_FEATURES_ESPERADO,
            "n_linhas": n_linhas,
            "universo": cfg["dataset"]["fase"],
            "hop_length": hop,
            "win_length": win,
            "duracao_segundos": dur,
            # Sem isto, os valores de ordem 1e-6 deste JSON não têm sentido:
            # eles são resíduo de float32 e mudam com a versão de numpy/BLAS.
            # Gravar o ambiente é o que permite comparar duas execuções — e é o
            # que faltava quando a v3 do documento citou 8,34e-06 como se fosse
            # uma constante do pipeline.
            "ambiente": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "librosa": librosa.__version__,
            },
        },
        "p1_largura": {
            "n_frames_total_unicos": totais,
            "aritmetica_1_mais_amostras_div_hop": frames_aritmetica,
            "largura_no_config": cfg["espectrograma"]["largura"],
            "config_diverge_do_pipeline":
                cfg["espectrograma"]["largura"] != frames_aritmetica,
        },
        "p2_padding_universo": stats_validos,
        "p2_assimetria_entre_classes": {
            "por_classe": por_classe,
            "diferenca_frac_padding_pp": dif_padding_pp,
            "diferenca_prop_fala_pp": dif_prop_fala_pp,
            "correlacao_prop_fala_x_n_frames_validos": corr,
            "leitura": (
                "a QUANTIDADE de padding é praticamente igual nas duas classes "
                f"({dif_padding_pp} p.p.), enquanto prop_fala é fortemente "
                f"assimétrica ({dif_prop_fala_pp} p.p.). Uma não prediz a "
                f"outra (r={corr}). Logo NÃO se pode justificar tratamento de "
                "padding na CNN por assimetria de classe — ver "
                "preprocessamento.py:180 e extrair_features.py:32."),
        },
        "p4_filtros_mel": grade_mel,
        "p4_frames_por_n_fft": frames_por_nfft,
        "p5_escala_db": {
            "padding_vira_plato_constante_com_ref_1_0": padding_e_plato,
            "valor_do_plato_db": round(float(valores_no_padding[0]), 2),
            "ref_max_invariante_a_ganho": ref_max_invariante_a_ganho,
            "ref_absoluto_preserva_ganho": ref_abs_preserva_ganho,
            "mfcc_sensivel_a_ganho": mfcc_sensivel_a_ganho,
            "mfcc_ganho_isolado_no_c0": mfcc_ganho_isolado_no_c0,
            "mfcc_deslocamento_c0_com_ganho_10x": round(desloc_c0, 3),
            "mfcc_deslocamento_demais_coeficientes": float(f"{desloc_demais:.3g}"),
            "ref_max_diferenca_maxima_db_com_ganho_10x":
                float(f"{ref_max_diferenca_maxima_db:.3g}"),
            "nuance_top_db": escala_top_db,
            "leitura": (
                "librosa.feature.mfcc aplica power_to_db com os DEFAULTS "
                "(ref=1.0, top_db=80) — provado por o MFCC mudar com o ganho, "
                "e mudar só no c0. Logo a paridade de não-linearidade com o "
                "ramo clássico se obtém com ref=1.0, NÃO com ref=np.max: este "
                "último é normalização por exemplo (invariante a ganho), a "
                "família que a P3 rejeita. Com ref=1.0 o padding ainda vira "
                "platô constante, então top_db=80 continua resolvendo a faixa "
                "dinâmica sem custar a energia global. NUANCE (nuance_top_db): "
                "o corte é em max-80 DO PRÓPRIO EXEMPLO, então um ganho de 10x "
                "desloca máximo e piso pelo mesmo nº de dB (a energia global "
                "sobrevive, e por isso a recomendação continua de pé), mas o "
                "valor absoluto do platô varia de exemplo para exemplo. E a "
                "invariância de ref=np.max é allclose, não igualdade: sobra "
                "~1e-6 dB de arredondamento de float32 "
                f"({ref_max_diferenca_maxima_db:.3g} dB medidos aqui)."),
        },
        "p7_desbalanceamento": {
            "por_conjunto": desbalanceamento,
            "razao_esperada": RAZAO_DESBALANCEAMENTO,
            "leitura": (
                "a razão spoof:bonafide vale nos quatro conjuntos, inclusive na "
                "subamostra de 30k que a CNN vai usar. É o que sustenta a "
                "recomendação da P7 (loss ponderada) e o que torna o "
                "desbalanceamento uma propriedade do PROTOCOLO, e não de um "
                "conjunto específico."),
        },
        "escopo_geracao": {
            "n_por_conjunto": n_por_conjunto,
            "n_subamostra_treino": n_sub,
            "n_necessario_para_cnn": n_necessario,
            "n_universo": n_linhas,
            "n_que_nao_seria_consumido": n_linhas - n_necessario,
        },
        "disco": disco,
        "problemas": problemas,
        "aprovado": not problemas,
    }

    destino = RAIZ / "results" / "metricas" / "auditoria_decisoes_cnn.json"
    destino.write_text(
        json.dumps(registro, indent=2, ensure_ascii=False, default=json_seguro),
        encoding="utf-8")

    print(json.dumps(registro, indent=2, ensure_ascii=False, default=json_seguro))
    print("\nAPROVADO — os números do DECISOES_PENDENTES_CNN.md conferem."
          if registro["aprovado"] else "\nREPROVADO — ver problemas acima.")
    sys.exit(0 if registro["aprovado"] else 1)


if __name__ == "__main__":
    main()
