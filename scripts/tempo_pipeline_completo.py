"""
Tempo do pipeline COMPLETO de inferência (R4b) — RF e SVM
==========================================================

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

AMARRA PARA O BLOCO 4 (CNN):
    `medir_pipeline` é genérica: recebe uma LISTA de etapas nomeadas e cronometra
    cada uma. Quando a CNN existir, ela entra como mais uma coluna, com as etapas
    [carregar, VAD+padding, gerar mel-espectrograma, forward da CNN] — MESMO
    código, mesmo protocolo, como manda a docstring de src/models/tempo.py. Não
    reescreva o cronômetro para a CNN: acrescente etapas aqui.

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
from src.models.modelos_ajustados import (
    carregar_modelo_ajustado, hashes_congelados,
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
    não descreveria nada. Compara os primeiros arquivos da amostra.
    """
    cols = nomes_features(cfg["features"]["n_mfcc"])
    feats = pd.read_csv(RAIZ / "data" / "features" / "features.csv",
                        usecols=["arquivo"] + cols)
    feats = feats.set_index("arquivo")

    n_conferidos, maior_desvio = 0, 0.0
    for _, r in amostra.head(10).iterrows():
        if r["arquivo"] not in feats.index:
            continue
        vetor = etapa_extrair(etapa_vad_padding(
            etapa_carregar(r["caminho"], cfg), cfg), cfg)
        esperado = feats.loc[r["arquivo"], cols].values.astype(float)
        desvio = float(np.max(np.abs(vetor - esperado)))
        maior_desvio = max(maior_desvio, desvio)
        n_conferidos += 1

    if n_conferidos == 0:
        raise RuntimeError("Nenhum arquivo da amostra foi encontrado em "
                           "features.csv — a guarda de fidelidade não pôde rodar.")
    if not np.isclose(maior_desvio, 0.0, atol=1e-8):
        raise RuntimeError(
            f"O vetor recalculado DIVERGE do features.csv congelado (maior "
            f"desvio absoluto {maior_desvio:.3e} em {n_conferidos} arquivos). "
            "O que este script cronometraria não é o pipeline que gerou os "
            "dados do trabalho. PARE e investigue antes de citar qualquer tempo.")
    return {"n_arquivos_conferidos": n_conferidos,
            "maior_desvio_absoluto": maior_desvio,
            "resultado": "o vetor recalculado é idêntico ao features.csv congelado"}


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
    print(f"  OK — {fidelidade['n_arquivos_conferidos']} arquivos, maior desvio "
          f"{fidelidade['maior_desvio_absoluto']:.3e}")

    caminhos = amostra["caminho"].tolist()
    etapas_comuns = [
        ("carregar_audio", lambda c: etapa_carregar(c, cfg)),
        ("vad_e_padding", lambda y: etapa_vad_padding(y, cfg)),
        ("extrair_features", lambda e: etapa_extrair(e, cfg)),
    ]

    resultados = {}
    for chave in ("rf", "svm"):
        carregado = carregar_modelo_ajustado(RAIZ, chave)
        modelo = carregado["modelo"]
        # Predição unitária, na escala nativa do modelo — o mesmo caminho de
        # decisão que src/models/tempo.py cronometra, só que agora precedido
        # pelas etapas que faltavam.
        if chave == "rf":
            predizer = lambda v: modelo.predict_proba(v.reshape(1, -1))[:, 1]
        else:
            predizer = lambda v: modelo.decision_function(v.reshape(1, -1))
        etapas = etapas_comuns + [("predizer", predizer)]

        print(f"\nMedindo o pipeline completo com {carregado['rotulo']}...")
        r = medir_pipeline(etapas, caminhos, cfg["tempo"])
        r["modelo"] = carregado["nome_arquivo"]
        r["rotulo"] = carregado["rotulo"]
        resultados[chave] = r

        for nome, v in r["etapas"].items():
            print(f"  {nome:<18} {v['ms_por_audio_mediana']:>9.4f} ms/áudio "
                  f"({v['percentual_do_total']:>5.2f}%)")
        print(f"  {'TOTAL':<18} {r['total_ms_por_audio']:>9.4f} ms/áudio")

    # ---- Leitura crítica ----------------------------------------------------
    print("\n" + "=" * 74)
    print("LEITURA CRÍTICA — o classificador importa para o custo de inferência?")
    print("=" * 74)
    linhas = []
    for chave, r in resultados.items():
        pct_pred = r["etapas"]["predizer"]["percentual_do_total"]
        pct_pre = 100 - pct_pred
        linhas.append(
            f"{r['rotulo']}: predizer = {r['etapas']['predizer']['ms_por_audio_mediana']:.4f} "
            f"ms/áudio ({pct_pred:.2f}% do total de "
            f"{r['total_ms_por_audio']:.2f} ms); pré-processamento + "
            f"featurização = {pct_pre:.2f}%.")
    for l in linhas:
        print(l)

    pct_max_pred = max(r["etapas"]["predizer"]["percentual_do_total"]
                       for r in resultados.values())
    dif_total = abs(resultados["rf"]["total_ms_por_audio"]
                    - resultados["svm"]["total_ms_por_audio"])
    razao_totais = (max(r["total_ms_por_audio"] for r in resultados.values())
                    / min(r["total_ms_por_audio"] for r in resultados.values()))
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

    registro = {
        "analise": "tempo_pipeline_completo",
        "data": date.today().isoformat(),
        "pergunta": ("qual a fração do custo real de inferência que cabe ao "
                     "classificador, e qual cabe ao pré-processamento?"),
        "conjunto": "validacao",
        "teste_lacrado": True,
        "n_amostra": N_AMOSTRA,
        "semente": cfg["semente"],
        "escopo": ("pipeline COMPLETO por áudio (batch=1): carregar .flac + "
                   "normalizar -> VAD + padding -> extrair 44 features -> "
                   "predizer. Complementa src/models/tempo.py, que mede SÓ a "
                   "última etapa (ver protocolo.escopo nos JSONs de RF e SVM)."),
        "fidelidade_do_pipeline": fidelidade,
        "por_modelo": resultados,
        "leitura_critica": veredito,
        "extensao_bloco4": (
            "medir_pipeline recebe uma lista de etapas nomeadas; a CNN entra "
            "acrescentando ['gerar_melspectrograma', 'forward_cnn'] no lugar de "
            "['extrair_features', 'predizer'] — MESMO código, MESMO protocolo, "
            "para que os três modelos sejam comparáveis"),
        "hashes_md5": hashes_congelados(
            RAIZ, cfg["experimento"]["caminho_subamostra"]),
        "ambiente": ambiente(n_jobs_inferencia=1),
    }
    caminho = RAIZ / "results" / "metricas" / "tempo_pipeline_completo.json"
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(registro, f, indent=2, ensure_ascii=False,
                  default=json_seguro)
    print(f"Salvo em {caminho.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
