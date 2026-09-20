"""
Tempo do pipeline COMPLETO de inferência (R4b) — RF, SVM e CNN
===============================================================

O QUE ESTE SCRIPT RESPONDE, E POR QUE ELE MUDA UMA CONCLUSÃO DO TC II:
    `src/models/tempo.py` cronometra SÓ a predição, a partir do vetor de
    features já extraído — e agora declara isso no JSON (campo
    `protocolo.escopo`, tarefa R4a). Medido assim, o RF prediz em ~0,019 ms por
    áudio e o SVM em ~0,358 ms: o RF parece ~19x mais barato.

    Só que ninguém classifica um vetor de 44 números que caiu do céu. O caminho
    real é: ler o .flac -> normalizar -> VAD -> padding -> MFCC/ZCR/centróide ->
    predizer. Se a featurização custa dezenas de milissegundos por áudio, o
    classificador responde por uma fração desprezível do custo total, e a
    conclusão sobre custo computacional deixa de ser "o RF é mais barato que o
    SVM" e passa a ser "a escolha do classificador é praticamente irrelevante
    para o custo de inferência; o que domina é o pré-processamento".

    Essa hipótese já estava escrita no comentário do config.yaml (bloco
    `tempo`, chave `incluir_extracao_features`) desde o início — declarada e
    nunca medida. Este script mede.

COMO A FIDELIDADE É GARANTIDA (o ponto mais importante):
    as etapas NÃO são reimplementadas aqui. São exatamente as funções do
    pipeline real — `carregar_audio`, `normalizar_amplitude`, `aplicar_vad`,
    `padronizar_duracao` (src/data/preprocessamento.py) e `extrair_vetor`
    (src/features/extrair_features.py) —, encadeadas na mesma ordem em que
    `preprocessar_audio` + `_processar_um` as encadeiam. Um número medido sobre
    uma cópia do pipeline não seria o número do pipeline.

    E há uma GUARDA: o vetor produzido aqui é comparado, feature a feature,
    contra a linha correspondente do features.csv CONGELADO. Se divergir, o
    script aborta — porque então o que está sendo cronometrado não é o caminho
    que gerou os dados do trabalho.

PROTOCOLO:
    o mesmo do config.yaml -> tempo (`repeticoes`, `descartar_aquecimento`),
    pelo mesmo motivo de sempre: comparação de custo só é defensável se todos os
    números vierem do mesmo protocolo. Amostra fixa de N áudios da VALIDAÇÃO
    (semente 42) — jamais do teste, que segue lacrado.

    A medida é por ÁUDIO (batch = 1), que é o cenário de uso real de um detector
    e o único em que somar etapas faz sentido: featurizar é intrinsecamente
    unitário (um arquivo por vez), então compará-lo ao throughput em lote do
    classificador seria comparar coisas diferentes.

A CNN ENTROU NO B4.7 (20/09/2026), PELA AMARRA QUE ESTAVA ESCRITA AQUI:
    `medir_pipeline` é genérica — recebe uma LISTA de etapas nomeadas e cronometra
    cada uma —, e foi exatamente assim que a CNN entrou: como mais uma coluna,
    com as etapas [carregar, VAD+padding, gerar mel-espectrograma, forward]. O
    cronômetro não foi reescrito, e a geração do espectrograma reusa
    `gerar_espectrogramas.gerar_um`, a MESMA função que produziu os tensores do
    B4.2 — com guarda de fidelidade contra o memmap congelado, igual à que já
    existia contra o features.csv.

    A CNN aparece em DOIS cenários, cnn_gpu e cnn_cpu, porque é o par que
    responde à pergunta de pesquisa: a vantagem dos modelos clássicos não é
    serem mais rápidos em igualdade de hardware — é não exigirem GPU.

    A PERGUNTA QUE ESTE NÚMERO RESOLVE: no RF a predição é 57,6% do custo. Na
    CNN, a geração do espectrograma pode dominar. Se dominar, isso é RESULTADO —
    a alavanca de engenharia muda de lugar conforme o ramo.

POR QUE RF E SVM SÃO RE-MEDIDOS JUNTO COM A CNN:
    o artefato é um só, e comparar um tempo de RF medido em agosto com um tempo
    de CNN medido em setembro compara também o estado da máquina. Os três saem
    da MESMA execução, com o mesmo cache, a mesma amostra e o mesmo protocolo.
    O método de medição de RF e SVM não mudou — só a data.

SAÍDA:
    results/metricas/tempo_pipeline_completo.json

Rode a partir da raiz:  python -m scripts.tempo_pipeline_completo
"""

import json
import platform
from datetime import date
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.utils.config import carregar_config
from src.utils.serializacao import json_seguro
from src.data.preprocessamento import (
    aplicar_vad, carregar_audio, normalizar_amplitude, padronizar_duracao,
)
from src.features.extrair_features import extrair_vetor, nomes_features
from src.features.gerar_espectrogramas import (
    frames_validos_espectrograma, gerar_um,
)
from src.models.modelos_ajustados import (
    carregar_modelo_ajustado, fixar_precisao_fp32_cnn, hashes_congelados,
)
from src.models.tempo import ambiente

RAIZ = Path(__file__).resolve().parents[1]

# Amostra pequena e FIXA: o objetivo é a ordem de grandeza relativa entre as
# etapas, não a quarta casa decimal. 200 áudios x 13 execuções já são 2.600
# leituras de disco — o suficiente para a mediana ser estável.
N_AMOSTRA = 200


def amostrar_validacao(cfg: dict, n: int) -> pd.DataFrame:
    """n áudios da VALIDAÇÃO, amostra fixa pela semente do config.

    O teste é lacrado (Bloco 5, execução única) — nem para medir tempo ele é
    tocado. A validação serve perfeitamente: o custo de featurizar não depende
    de qual partição o arquivo caiu.
    """
    split = pd.read_csv(RAIZ / "data" / "processed" / "split.csv")
    labels = pd.read_csv(RAIZ / "data" / "processed" / "labels.csv",
                         usecols=["arquivo", "caminho"])
    val = split[split["conjunto"] == "validacao"].merge(labels, on="arquivo",
                                                        how="inner")
    return val.sample(n=n, random_state=cfg["semente"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# As etapas — funções REAIS do pipeline, encadeadas na ordem real
# ---------------------------------------------------------------------------
def etapa_carregar(caminho: str, cfg: dict):
    """Ler o .flac do disco e normalizar a amplitude (passos 1-2 de preprocessar_audio)."""
    y = carregar_audio(caminho, cfg["audio"]["sample_rate"])
    return normalizar_amplitude(y)


def etapa_vad_padding(y: np.ndarray, cfg: dict):
    """VAD + padronização de duração (passos 3-4), devolvendo também n_amostras_validas.

    A conta de `n_amostras_validas` é a MESMA de preprocessar_audio — ela decide
    o mascaramento do padding na agregação, e sem ela a etapa seguinte mediria
    outra coisa.
    """
    sr, dur = cfg["audio"]["sample_rate"], cfg["audio"]["duracao_segundos"]
    if cfg["audio"]["vad"]:
        y, _ = aplicar_vad(y, sr)
    n_validas = min(len(y), int(sr * dur))
    return padronizar_duracao(y, sr, dur), n_validas


def etapa_extrair(entrada, cfg: dict):
    """MFCC + ZCR + centróide agregados — o vetor de 44 features."""
    y, n_validas = entrada
    if not cfg["features"].get("mascarar_padding", True):
        n_validas = None
    vetor, _, _ = extrair_vetor(y, cfg["audio"]["sample_rate"], cfg,
                                n_amostras_validas=n_validas)
    return vetor


def conferir_fidelidade(cfg: dict, amostra: pd.DataFrame) -> dict:
    """GUARDA: o vetor produzido aqui é o mesmo do features.csv congelado?

    Sem esta checagem, o script poderia estar cronometrando um caminho parecido
    — mas não idêntico — ao que gerou os dados do trabalho, e o número medido
    não descreveria nada.

    A COMPARAÇÃO É FEITA EM float32, E ISSO NÃO É AFROUXAR A GUARDA:
        `extrair_vetor` devolve float32 (é o dtype do librosa). Ao gravar o CSV,
        o pandas escreve o repr CURTO daquele float32 — a menor string que
        volta a ser o MESMO float32. Lendo o CSV, o pandas entrega float64, e aí
        o texto "649.7393" vira 649.7393 exato, enquanto o float32 original vale
        649.7393188476562. A diferença — 1,9e-05 aqui — não é divergência de
        pipeline: é a distância entre o número e a sua própria representação
        decimal curta, e comparar em float64 estaria exigindo do CSV uma
        precisão que ele nunca teve.

        Comparando na precisão que o CSV de fato guarda, a exigência é a mais
        forte possível: IGUALDADE EXATA, feature a feature, zero tolerância.
        (Conferido em 03/09/2026: 440 de 440 features idênticas em 10 arquivos.)
    """
    cols = nomes_features(cfg["features"]["n_mfcc"])
    feats = pd.read_csv(RAIZ / "data" / "features" / "features.csv",
                        usecols=["arquivo"] + cols)
    feats = feats.set_index("arquivo")

    n_conferidos, n_divergentes, maior_desvio = 0, 0, 0.0
    for _, r in amostra.head(10).iterrows():
        if r["arquivo"] not in feats.index:
            continue
        vetor = etapa_extrair(etapa_vad_padding(
            etapa_carregar(r["caminho"], cfg), cfg), cfg).astype(np.float32)
        esperado = feats.loc[r["arquivo"], cols].values.astype(np.float32)
        n_divergentes += int(np.sum(vetor != esperado))
        maior_desvio = max(maior_desvio,
                           float(np.max(np.abs(vetor.astype(float)
                                               - esperado.astype(float)))))
        n_conferidos += 1

    if n_conferidos == 0:
        raise RuntimeError("Nenhum arquivo da amostra foi encontrado em "
                           "features.csv — a guarda de fidelidade não pôde rodar.")
    if n_divergentes:
        raise RuntimeError(
            f"O vetor recalculado DIVERGE do features.csv congelado: "
            f"{n_divergentes} feature(s) diferentes em {n_conferidos} arquivos "
            f"(maior desvio absoluto {maior_desvio:.3e}), comparando em float32. "
            "O que este script cronometraria não é o pipeline que gerou os "
            "dados do trabalho. PARE e investigue antes de citar qualquer tempo.")
    return {"n_arquivos_conferidos": n_conferidos,
            "n_features_por_arquivo": len(cols),
            "n_features_divergentes": n_divergentes,
            "precisao_da_comparacao": (
                "float32 — o dtype que extrair_vetor produz e que o CSV grava; "
                "comparar em float64 exigiria do CSV uma precisão que ele nunca "
                "teve (ver docstring de conferir_fidelidade)"),
            "resultado": ("o vetor recalculado é IDÊNTICO ao features.csv "
                          "congelado, feature a feature, sem tolerância")}


# ---------------------------------------------------------------------------
# As etapas da CNN — `gerar_um` REUSADA, não reimplementada
# ---------------------------------------------------------------------------
def etapa_gerar_espectrograma(entrada, cfg: dict):
    """log-Mel em dB `(128, 251)` + o n_frames_validos da máscara.

    `gerar_um` é a MESMA função que gerou os 74.453 tensores do B4.2, e
    `frames_validos_espectrograma` é a mesma que produziu a coluna
    `n_frames_validos` dos índices. Nenhuma aritmética nova: recontar frames aqui
    produziria uma máscara divergindo por ±1 do que a rede viu no treino.
    """
    y, n_validas = entrada
    e = cfg["espectrograma"]
    S_db = gerar_um(y, e["sample_rate"], e)
    n_frames = frames_validos_espectrograma(n_validas, e, int(S_db.shape[1]))
    return S_db, n_frames


def carregar_normalizacao_cnn(cfg: dict) -> tuple[np.ndarray, np.ndarray, str]:
    """As estatísticas DOS 30k — as do refit, não as dos 27k do early stopping."""
    caminho = RAIZ / "data" / "espectrogramas" / "normalizacao_cnn_30k.json"
    with open(caminho, encoding="utf-8") as f:
        norm = json.load(f)
    if norm["estagio"] != "30k_refit":
        raise RuntimeError(f"{caminho.name} está no estágio {norm['estagio']!r}, "
                           "esperado '30k_refit'.")
    media = np.array(norm["media_por_mel"], dtype=np.float32)[:, None]
    desvio = np.array(norm["desvio_por_mel"], dtype=np.float32)[:, None]
    return media, desvio, f"{caminho.relative_to(RAIZ).as_posix()} ({norm['estagio']})"


def preditor_cnn(modelo, media, desvio, dispositivo):
    """Callable de PREDIÇÃO da CNN para `medir_pipeline` — normalização incluída.

    A NORMALIZAÇÃO ENTRA AQUI, E NÃO NA ETAPA DO ESPECTROGRAMA, porque é aqui que
    ela mora no pipeline real: os tensores em disco NÃO estão normalizados (ver
    espectrogramas.meta.json -> nota_normalizacao), e o `EspectrogramaDataset`
    aplica média/desvio em tempo de carga, imediatamente antes do forward. Pôr a
    normalização na etapa anterior mudaria de lugar um custo de microssegundos e,
    pior, faria a etapa "gerar_melspectrograma" deixar de ser comparável ao que
    `gerar_um` produz e grava.

    `torch.cuda.synchronize()` antes e depois pelo motivo de sempre: sem isso
    mede-se o enfileiramento do kernel, não a execução.
    """
    import torch

    cuda = dispositivo.type == "cuda"

    def predizer(entrada):
        S_db, n_frames = entrada
        X = (S_db - media) / desvio               # normalização em tempo de carga
        x = torch.from_numpy(X)[None, None]       # (1, 1, 128, 251)
        m = torch.zeros(1, X.shape[1], dtype=torch.float32)
        m[0, :n_frames] = 1.0
        if cuda:
            torch.cuda.synchronize()
        with torch.no_grad():
            logits = modelo(x.to(dispositivo), m.to(dispositivo))
            s = torch.softmax(logits, dim=1)[:, 1]
        if cuda:
            torch.cuda.synchronize()
        return s.cpu().numpy()

    return predizer


def conferir_fidelidade_cnn(cfg: dict, amostra: pd.DataFrame) -> dict:
    """GUARDA: o espectrograma produzido aqui é o mesmo do memmap CONGELADO?

    Exatamente a mesma ideia de `conferir_fidelidade`, um ramo adiante: se o
    tensor recalculado divergir do que está em `validacao.npy`, o que este script
    cronometra não é o caminho que gerou os dados com que a CNN foi treinada e
    avaliada.

    Aqui a exigência é IGUALDADE EXATA sem ressalva nenhuma — e a diferença para
    o caso do features.csv é que o memmap guarda os float32 em BINÁRIO, não um
    repr decimal curto. Não há precisão perdida na gravação para desculpar
    divergência alguma.
    """
    dir_esp = RAIZ / "data" / "espectrogramas"
    indice = pd.read_csv(dir_esp / "indice_validacao.csv").set_index("arquivo")
    mm = np.load(dir_esp / "validacao.npy", mmap_mode="r")

    n_conferidos, n_divergentes, n_mascaras_divergentes = 0, 0, 0
    for _, r in amostra.head(10).iterrows():
        if r["arquivo"] not in indice.index:
            continue
        S_db, n_frames = etapa_gerar_espectrograma(
            etapa_vad_padding(etapa_carregar(r["caminho"], cfg), cfg), cfg)
        linha = int(indice.loc[r["arquivo"], "linha"])
        esperado = np.asarray(mm[linha])
        n_divergentes += int(np.sum(S_db != esperado))
        n_mascaras_divergentes += int(
            n_frames != int(indice.loc[r["arquivo"], "n_frames_validos"]))
        n_conferidos += 1

    if n_conferidos == 0:
        raise RuntimeError("Nenhum arquivo da amostra foi encontrado em "
                           "indice_validacao.csv — a guarda de fidelidade da "
                           "CNN não pôde rodar.")
    if n_divergentes or n_mascaras_divergentes:
        raise RuntimeError(
            f"O espectrograma recalculado DIVERGE do memmap congelado: "
            f"{n_divergentes} posição(ões) em {n_conferidos} arquivos, e "
            f"{n_mascaras_divergentes} máscara(s) divergentes. O que este script "
            "cronometraria não é o caminho que gerou os tensores do B4.2. PARE.")
    return {"n_arquivos_conferidos": n_conferidos,
            "formato": "(128, 251) float32",
            "n_posicoes_divergentes": n_divergentes,
            "n_mascaras_divergentes": n_mascaras_divergentes,
            "comparado_contra": "data/espectrogramas/validacao.npy (memmap do B4.2)",
            "precisao_da_comparacao": (
                "igualdade EXATA em float32 — o memmap guarda os bits, não um "
                "repr decimal, então não há perda de gravação a tolerar"),
            "resultado": ("o tensor recalculado é IDÊNTICO ao memmap congelado, "
                          "posição a posição, e a máscara bate com o índice")}


# ---------------------------------------------------------------------------
# Cronômetro genérico — o mesmo que a CNN vai usar no Bloco 4
# ---------------------------------------------------------------------------
def medir_pipeline(etapas: list[tuple[str, Callable]], itens: list,
                   cfg_tempo: dict) -> dict:
    """Cronometra, etapa a etapa, um pipeline aplicado item a item (batch = 1).

    Args:
        etapas: lista de (nome, função). A primeira função recebe o item (ex.: o
            caminho do .flac); cada função seguinte recebe o retorno da anterior.
            Encadear assim é o que permite acrescentar "gerar mel-espectrograma"
            e "forward da CNN" no Bloco 4 sem tocar no cronômetro.
        itens: a amostra (um item por áudio).
        cfg_tempo: bloco `tempo` do config.yaml (repeticoes, descartar_aquecimento).

    Returns:
        dict {nome_da_etapa: {ms_por_audio_mediana, min, max, percentual}}, mais
        o total. As medidas são o tempo MÉDIO POR ÁUDIO em cada repetição; o que
        se reporta é a mediana entre repetições, como manda o protocolo.
    """
    import time

    reps = int(cfg_tempo["repeticoes"])
    aquec = int(cfg_tempo["descartar_aquecimento"])
    nomes = [nome for nome, _ in etapas]
    # medidas[etapa] = uma soma de tempo por repetição
    medidas: dict[str, list[float]] = {nome: [] for nome in nomes}

    for r in range(aquec + reps):
        soma = {nome: 0.0 for nome in nomes}
        for item in itens:
            valor = item
            for nome, func in etapas:
                t0 = time.perf_counter()
                valor = func(valor)
                soma[nome] += time.perf_counter() - t0
        if r >= aquec:                       # aquecimento descartado
            for nome in nomes:
                medidas[nome].append(soma[nome])

    n = len(itens)
    por_etapa = {nome: {
        "ms_por_audio_mediana": round(1000 * float(np.median(medidas[nome])) / n, 4),
        "ms_por_audio_min": round(1000 * min(medidas[nome]) / n, 4),
        "ms_por_audio_max": round(1000 * max(medidas[nome]) / n, 4),
    } for nome in nomes}

    total = sum(v["ms_por_audio_mediana"] for v in por_etapa.values())
    for v in por_etapa.values():
        v["percentual_do_total"] = round(100 * v["ms_por_audio_mediana"] / total, 2)

    return {"etapas": por_etapa, "total_ms_por_audio": round(total, 4),
            "n_audios": n, "protocolo": {"repeticoes": reps,
                                         "descartar_aquecimento": aquec,
                                         "batch": 1,
                                         "fonte": "config.yaml -> tempo"}}


def main() -> None:
    cfg = carregar_config(RAIZ)
    amostra = amostrar_validacao(cfg, N_AMOSTRA)
    print(f"Amostra: {len(amostra)} áudios da VALIDAÇÃO (semente {cfg['semente']}). "
          "O teste continua lacrado.")

    print("Guarda de fidelidade (vetor recalculado x features.csv congelado)...")
    fidelidade = conferir_fidelidade(cfg, amostra)
    print(f"  OK — {fidelidade['n_arquivos_conferidos']} arquivos x "
          f"{fidelidade['n_features_por_arquivo']} features, "
          f"{fidelidade['n_features_divergentes']} divergentes (float32)")

    print("Guarda de fidelidade da CNN (tensor recalculado x validacao.npy)...")
    fidelidade_cnn = conferir_fidelidade_cnn(cfg, amostra)
    print(f"  OK — {fidelidade_cnn['n_arquivos_conferidos']} arquivos, "
          f"{fidelidade_cnn['n_posicoes_divergentes']} posições divergentes, "
          f"{fidelidade_cnn['n_mascaras_divergentes']} máscaras divergentes")

    caminhos = amostra["caminho"].tolist()

    # ---- Base COMPARTILHADA, medida UMA vez ---------------------------------
    # POR QUE UMA VEZ SÓ (e não uma vez por modelo): RF e SVM consomem o MESMO
    # vetor de 44 features — a featurização é literalmente o mesmo trabalho.
    # Medi-la dentro do laço de cada modelo, intercalada com a predição, produz
    # números diferentes para trabalho idêntico: a predição do RF (300 árvores)
    # despeja o cache da CPU e contamina a etapa vizinha. Na primeira versão
    # deste script isso deu 5,51 ms/áudio de featurização no laço do RF contra
    # 3,79 ms no laço do SVM — 45% de diferença para o MESMO código, e o efeito
    # se repetiu entre execuções (não era ruído). Medir a base isolada elimina o
    # artefato e é o que deixa os dois modelos comparáveis: por construção eles
    # diferem só na etapa de predição.
    etapas_base = [
        ("carregar_audio", lambda c: etapa_carregar(c, cfg)),
        ("vad_e_padding", lambda y: etapa_vad_padding(y, cfg)),
        ("extrair_features", lambda e: etapa_extrair(e, cfg)),
    ]
    print("\nMedindo a base compartilhada (carregar -> VAD/padding -> features)...")
    base = medir_pipeline(etapas_base, caminhos, cfg["tempo"])
    for nome, v in base["etapas"].items():
        print(f"  {nome:<18} {v['ms_por_audio_mediana']:>9.4f} ms/áudio")
    print(f"  {'BASE':<18} {base['total_ms_por_audio']:>9.4f} ms/áudio")

    # Vetores pré-calculados: a predição dos dois modelos é medida sobre
    # exatamente a mesma entrada.
    vetores = [etapa_extrair(etapa_vad_padding(etapa_carregar(c, cfg), cfg), cfg)
               for c in caminhos]

    resultados = {}
    for chave in ("rf", "svm"):
        carregado = carregar_modelo_ajustado(RAIZ, chave)
        modelo = carregado["modelo"]
        # Predição unitária, na escala nativa do modelo — o mesmo caminho de
        # decisão que src/models/tempo.py cronometra, só que agora somado às
        # etapas que faltavam.
        if chave == "rf":
            # n_jobs=1: é o valor exigido pelo protocolo de medição de tempo
            # (config.yaml -> tempo, "fixar n_jobs igual para todos os modelos
            # clássicos"), e é também o que torna o score reprodutível bit a bit
            # (ver docstring de src.models.avaliacao.predizer_rf).
            modelo.n_jobs = 1
            predizer = lambda v: modelo.predict_proba(v.reshape(1, -1))[:, 1]
        else:
            predizer = lambda v: modelo.decision_function(v.reshape(1, -1))

        print(f"\nMedindo a predição de {carregado['rotulo']}...")
        r = medir_pipeline([("predizer", predizer)], vetores, cfg["tempo"])
        ms_pred = r["etapas"]["predizer"]["ms_por_audio_mediana"]
        total = round(base["total_ms_por_audio"] + ms_pred, 4)
        # Recompõe o pipeline inteiro: base compartilhada + predição do modelo.
        etapas = {**{k: dict(v) for k, v in base["etapas"].items()},
                  "predizer": dict(r["etapas"]["predizer"])}
        for v in etapas.values():
            v["percentual_do_total"] = round(100 * v["ms_por_audio_mediana"]
                                             / total, 2)
        r = {"etapas": etapas, "total_ms_por_audio": total,
             "n_audios": len(caminhos), "protocolo": base["protocolo"],
             "modelo": carregado["nome_arquivo"], "rotulo": carregado["rotulo"]}
        resultados[chave] = r

        for nome, v in etapas.items():
            print(f"  {nome:<18} {v['ms_por_audio_mediana']:>9.4f} ms/áudio "
                  f"({v['percentual_do_total']:>5.2f}%)")
        print(f"  {'TOTAL':<18} {total:>9.4f} ms/áudio")

    # ---- O ramo da CNN ------------------------------------------------------
    # A base da CNN é MEDIDA À PARTE, e não reaproveitada da base clássica: as
    # duas primeiras etapas são o mesmo código, mas a terceira é outra
    # (espectrograma x 44 features) e a cadeia é cronometrada inteira, no mesmo
    # contexto de cache — que é a razão pela qual a base clássica também é medida
    # isolada (ver nota_metodo). Os dois valores de `carregar_audio` e
    # `vad_e_padding` saem no JSON e devem concordar; a concordância é reportada
    # em `consistencia_das_etapas_comuns`.
    base_cnn, resultados_cnn = None, {}
    carregado_cnn = carregar_modelo_ajustado(RAIZ, "cnn")
    modelo_cnn = carregado_cnn["modelo"]
    media, desvio, origem_norm = carregar_normalizacao_cnn(cfg)
    precisao = fixar_precisao_fp32_cnn()

    etapas_cnn = [
        ("carregar_audio", lambda c: etapa_carregar(c, cfg)),
        ("vad_e_padding", lambda y: etapa_vad_padding(y, cfg)),
        ("gerar_melspectrograma", lambda e: etapa_gerar_espectrograma(e, cfg)),
    ]
    print("\nMedindo a base da CNN (carregar -> VAD/padding -> espectrograma)...")
    base_cnn = medir_pipeline(etapas_cnn, caminhos, cfg["tempo"])
    for nome, v in base_cnn["etapas"].items():
        print(f"  {nome:<22} {v['ms_por_audio_mediana']:>9.4f} ms/áudio")
    print(f"  {'BASE CNN':<22} {base_cnn['total_ms_por_audio']:>9.4f} ms/áudio")

    espectrogramas = [etapa_gerar_espectrograma(
        etapa_vad_padding(etapa_carregar(c, cfg), cfg), cfg) for c in caminhos]

    import torch
    cenarios = [("cnn_gpu", "cuda")] if torch.cuda.is_available() else []
    # torch.set_num_threads(1): paridade com o n_jobs=1 do RF. Uma CNN em CPU com
    # 12 threads contra um RF com n_jobs=1 não é comparação, é ruído.
    cenarios.append(("cnn_cpu", "cpu"))
    threads_originais = torch.get_num_threads()
    for chave, nome_dev in cenarios:
        dispositivo = torch.device(nome_dev)
        torch.set_num_threads(1 if nome_dev == "cpu" else threads_originais)
        modelo_cnn = modelo_cnn.to(dispositivo).eval()
        rotulo = f"{carregado_cnn['rotulo']} — {nome_dev.upper()}"
        print(f"\nMedindo a predição de {rotulo}...")
        r = medir_pipeline([("predizer", preditor_cnn(modelo_cnn, media, desvio,
                                                      dispositivo))],
                           espectrogramas, cfg["tempo"])
        ms_pred = r["etapas"]["predizer"]["ms_por_audio_mediana"]
        total = round(base_cnn["total_ms_por_audio"] + ms_pred, 4)
        etapas = {**{k: dict(v) for k, v in base_cnn["etapas"].items()},
                  "predizer": dict(r["etapas"]["predizer"])}
        for v in etapas.values():
            v["percentual_do_total"] = round(100 * v["ms_por_audio_mediana"]
                                             / total, 2)
        resultados_cnn[chave] = {
            "etapas": etapas, "total_ms_por_audio": total,
            "n_audios": len(caminhos), "protocolo": base_cnn["protocolo"],
            "modelo": carregado_cnn["nome_arquivo"], "rotulo": rotulo,
            "dispositivo": nome_dev,
            "torch_num_threads": 1 if nome_dev == "cpu" else threads_originais,
            "normalizacao": origem_norm,
            "nota_sincronizacao": (
                "torch.cuda.synchronize() antes e depois do trecho cronometrado, "
                "DENTRO do callable — sem isso mede-se o enfileiramento do "
                "kernel, não a execução" if nome_dev == "cuda"
                else "não se aplica: execução em CPU é síncrona"),
            "nota_normalizacao": (
                "a normalização por faixa Mel está DENTRO da etapa `predizer`, "
                "porque é lá que ela mora no pipeline real (os tensores em disco "
                "não estão normalizados; o EspectrogramaDataset aplica "
                "média/desvio em tempo de carga)"),
        }
        for nome, v in etapas.items():
            print(f"  {nome:<22} {v['ms_por_audio_mediana']:>9.4f} ms/áudio "
                  f"({v['percentual_do_total']:>5.2f}%)")
        print(f"  {'TOTAL':<22} {total:>9.4f} ms/áudio")
    torch.set_num_threads(threads_originais)
    resultados.update(resultados_cnn)

    # As duas etapas comuns aos dois ramos foram medidas duas vezes, em contextos
    # de cache diferentes. Se divergirem muito, a comparação entre os totais dos
    # dois ramos está carregando um artefato de medição, não uma diferença real.
    consistencia = {}
    for etapa in ("carregar_audio", "vad_e_padding"):
        a = base["etapas"][etapa]["ms_por_audio_mediana"]
        b = base_cnn["etapas"][etapa]["ms_por_audio_mediana"]
        consistencia[etapa] = {
            "na_base_classica_ms": a, "na_base_cnn_ms": b,
            "diferenca_relativa_pct": round(100 * abs(a - b) / max(a, b), 2),
        }

    # ---- Leitura crítica ----------------------------------------------------
    print("\n" + "=" * 74)
    print("LEITURA CRÍTICA — o classificador importa para o custo de inferência?")
    print("=" * 74)
    linhas = []
    for chave, r in resultados.items():
        pct_pred = r["etapas"]["predizer"]["percentual_do_total"]
        pct_pre = 100 - pct_pred
        # "representação" e não "featurização": no ramo clássico a etapa é o
        # vetor de 44 features, no da CNN é o log-Mel. Dizer "featurização" para
        # os três descreveria errado um terço da tabela.
        linhas.append(
            f"{r['rotulo']}: predizer = {r['etapas']['predizer']['ms_por_audio_mediana']:.4f} "
            f"ms/áudio ({pct_pred:.2f}% do total de "
            f"{r['total_ms_por_audio']:.2f} ms); pré-processamento + "
            f"construção da representação = {pct_pre:.2f}%.")
    for l in linhas:
        print(l)

    # O veredito abaixo é sobre o RAMO CLÁSSICO, e por isso é calculado SÓ sobre
    # rf/svm: ele responde «a escolha entre RF e SVM importa para o custo?». A
    # CNN tem a sua leitura logo a seguir, porque a pergunta dela é outra.
    classicos = {k: resultados[k] for k in ("rf", "svm")}
    pct_max_pred = max(r["etapas"]["predizer"]["percentual_do_total"]
                       for r in classicos.values())
    dif_total = abs(resultados["rf"]["total_ms_por_audio"]
                    - resultados["svm"]["total_ms_por_audio"])
    razao_totais = (max(r["total_ms_por_audio"] for r in classicos.values())
                    / min(r["total_ms_por_audio"] for r in classicos.values()))
    if pct_max_pred < 5.0:
        veredito = (
            f"A predição responde por no máximo {pct_max_pred:.2f}% do custo real "
            "de inferência nos dois modelos: o que domina é o pré-processamento "
            "(carregar + VAD + padding + extração de features). CONSEQUÊNCIA "
            "PARA O TEXTO: a conclusão sobre custo computacional NÃO é 'o RF é "
            "mais barato que o SVM'. Medida ponta a ponta, a diferença entre os "
            f"dois pipelines é de {dif_total:.2f} ms por áudio (razão "
            f"{razao_totais:.2f}x entre os totais), porque os dois pagam o mesmo "
            "custo de featurização. A escolha do classificador é praticamente "
            "irrelevante para o tempo de inferência; a alavanca de engenharia "
            "está no pré-processamento. É este o achado — e ele confirma a "
            "hipótese registrada no comentário do config.yaml (bloco `tempo`).")
    else:
        veredito = (
            f"A predição responde por até {pct_max_pred:.2f}% do custo real de "
            "inferência — fração NÃO desprezível. A hipótese registrada no "
            "config.yaml (de que o pré-processamento dominaria a ponto de tornar "
            "a escolha do classificador irrelevante) NÃO se confirma nesta "
            "medição: a diferença entre RF e SVM sobrevive ao pipeline completo "
            f"({dif_total:.2f} ms por áudio, razão {razao_totais:.2f}x). A "
            "comparação de custo entre modelos continua sendo uma comparação "
            "com significado prático.")
    print(f"\n{veredito}\n")

    # ---- A leitura do ramo da CNN (pergunta diferente) ----------------------
    # No RF a predição é a maior fatia do custo. Na CNN a hipótese do marco é que
    # a GERAÇÃO DO ESPECTROGRAMA domine — e, se dominar, isso é RESULTADO: a
    # alavanca de engenharia muda de lugar conforme o ramo.
    print("=" * 74)
    print("LEITURA CRÍTICA — no ramo da CNN, o que domina o custo?")
    print("=" * 74)
    veredito_cnn = {}
    for chave, r in resultados_cnn.items():
        e = r["etapas"]
        pct_esp = e["gerar_melspectrograma"]["percentual_do_total"]
        pct_pred = e["predizer"]["percentual_do_total"]
        dominante = max(e, key=lambda k: e[k]["ms_por_audio_mediana"])
        texto = (
            f"{r['rotulo']}: total {r['total_ms_por_audio']:.2f} ms/áudio. "
            f"Espectrograma = {pct_esp:.2f}%, forward = {pct_pred:.2f}%. "
            f"Etapa dominante: {dominante}.")
        if dominante == "gerar_melspectrograma":
            texto += (" A hipótese do B4.7 SE CONFIRMA neste cenário: a geração "
                      "da representação domina o custo, e não a rede. É um "
                      "achado de engenharia, não ruído — a alavanca de "
                      "otimização está na featurização, como no ramo clássico.")
        else:
            texto += (" A hipótese do B4.7 NÃO se confirma neste cenário: o "
                      "forward da rede domina o custo. É aqui que a exigência de "
                      "hardware da CNN aparece no tempo, e não só na memória.")
        veredito_cnn[chave] = texto
        print(texto)

    if "cnn_gpu" in resultados_cnn and "cnn_cpu" in resultados_cnn:
        g = resultados_cnn["cnn_gpu"]["etapas"]["predizer"]["ms_por_audio_mediana"]
        c = resultados_cnn["cnn_cpu"]["etapas"]["predizer"]["ms_por_audio_mediana"]
        veredito_cnn["gpu_x_cpu"] = (
            f"O forward da CNN custa {c:.2f} ms/áudio em CPU (1 thread) contra "
            f"{g:.2f} ms em GPU — razão {c / g:.1f}x. Somado ao pipeline, o total "
            f"vai de {resultados_cnn['cnn_gpu']['total_ms_por_audio']:.2f} ms "
            f"para {resultados_cnn['cnn_cpu']['total_ms_por_audio']:.2f} ms. É "
            "ESTE o par que sustenta a afirmação central do trabalho: a vantagem "
            "dos modelos clássicos não é serem mais rápidos em igualdade de "
            "hardware — é não exigirem GPU. Comparar a CNN-GPU com um RF em CPU "
            "compararia hardware, não modelo; a coluna CNN-CPU é o que torna a "
            "comparação honesta.")
        print(f"\n{veredito_cnn['gpu_x_cpu']}")
    print()

    registro = {
        "analise": "tempo_pipeline_completo",
        "data": date.today().isoformat(),
        "pergunta": ("qual a fração do custo real de inferência que cabe ao "
                     "classificador, e qual cabe ao pré-processamento?"),
        "conjunto": "validacao",
        "teste_lacrado": True,
        "n_amostra": N_AMOSTRA,
        "semente": cfg["semente"],
        "escopo": ("pipeline COMPLETO por áudio (batch=1). Ramo clássico: "
                   "carregar .flac + normalizar -> VAD + padding -> extrair 44 "
                   "features -> predizer. Ramo CNN: carregar .flac + normalizar "
                   "-> VAD + padding -> gerar log-Mel (128x251) -> normalizar "
                   "por faixa Mel + forward. Complementa src/models/tempo.py, "
                   "que mede SÓ a última etapa (ver protocolo.escopo nos JSONs "
                   "de RF, SVM e CNN)."),
        "fidelidade_do_pipeline": fidelidade,
        "fidelidade_do_pipeline_cnn": fidelidade_cnn,
        "base_compartilhada": base,
        "base_cnn": base_cnn,
        "consistencia_das_etapas_comuns": {
            "etapas": consistencia,
            "por_que_existe": (
                "carregar_audio e vad_e_padding são o MESMO código nos dois "
                "ramos, mas foram cronometrados duas vezes, em contextos de "
                "cache diferentes. Reportar as duas medidas lado a lado deixa "
                "visível quanto da diferença entre os totais dos dois ramos é "
                "artefato de medição e quanto é diferença real — e é a mesma "
                "precaução que `nota_metodo` descreve para RF x SVM."),
        },
        "nota_metodo": (
            "a base (carregar + VAD/padding + featurização) é medida UMA vez e "
            "somada à predição de cada modelo. Medi-la dentro do laço de cada "
            "modelo dava 5,51 ms/áudio no laço do RF contra 3,79 ms no do SVM "
            "para código IDÊNTICO (efeito reprodutível entre execuções): a "
            "predição do RF, com 300 árvores, despeja o cache da CPU e "
            "contamina a etapa vizinha. RF e SVM consomem o MESMO vetor de "
            "features, então a base é a mesma por construção e toda a diferença "
            "entre eles está na etapa de predição."),
        "por_modelo": resultados,
        "leitura_critica": veredito,
        "leitura_critica_cnn": veredito_cnn,
        "extensao_bloco4": (
            "FEITA no B4.7 (20/09/2026), exatamente como a amarra previa: "
            "medir_pipeline recebe uma lista de etapas nomeadas, e a CNN entrou "
            "trocando ['extrair_features', 'predizer'] por "
            "['gerar_melspectrograma', 'predizer'] — MESMO cronômetro, MESMO "
            "protocolo, MESMA amostra. `gerar_um` de "
            "src/features/gerar_espectrogramas.py é reusada, não reimplementada, "
            "e a guarda de fidelidade compara o tensor recalculado contra o "
            "memmap congelado do B4.2."),
        "precisao_numerica_cnn": precisao,
        "nota_re_medicao": (
            "RF e SVM foram RE-MEDIDOS nesta execução, junto com a CNN. O método "
            "não mudou — mudou a data. Um tempo de RF de agosto comparado a um "
            "tempo de CNN de setembro compararia também o estado da máquina; os "
            "três saem do mesmo processo, com o mesmo cache e a mesma amostra."),
        "hashes_md5": hashes_congelados(
            RAIZ, cfg["experimento"]["caminho_subamostra"]),
        "ambiente": ambiente(n_jobs_inferencia=1),
    }
    registro["ambiente"]["torch"] = torch.__version__
    registro["ambiente"]["gpu"] = (torch.cuda.get_device_name(0)
                                   if torch.cuda.is_available() else None)
    registro["ambiente"]["cuda_versao"] = torch.version.cuda
    caminho = RAIZ / "results" / "metricas" / "tempo_pipeline_completo.json"
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(registro, f, indent=2, ensure_ascii=False,
                  default=json_seguro)
    print(f"Salvo em {caminho.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
