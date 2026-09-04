"""
Carregamento dos modelos ajustados do braço principal — ponto único
====================================================================

POR QUE ISTO EXISTE (Bloco 3, revisão de 03/09/2026):
    Os diagnósticos por ataque e por codec, na primeira versão (13/08/2026),
    RE-TREINAVAM um RF baseline e decidiam por `modelo.predict()` — isto é,
    argmax em 0,50. O resultado foi um diagnóstico vazio: recall 1,0 em todos
    os 13 ataques, porque naquele limiar o modelo diz "spoof" para quase tudo.
    Pior: o modelo diagnosticado não era o modelo do braço principal, então a
    tabela não descrevia nenhum número do README.

    A correção é estrutural, não cosmética: nenhum script de análise treina
    modelo. Todos CARREGAM o artefato persistido e leem o limiar do JSON que o
    acompanha. Este módulo é o único lugar onde esse par (modelo, limiar) é
    montado — se amanhã o nome do artefato mudar, muda aqui e em lugar nenhum
    mais, e nenhum script pode divergir silenciosamente do outro.

A REGRA DE DECISÃO CONTINUA SENDO UMA SÓ:
    `score >= limiar`, via src.models.avaliacao.aplicar_limiar. Este módulo
    entrega o score na escala CERTA de cada modelo — predict_proba[:, 1] para o
    RF ([0,1]) e decision_function para o SVM (real, centrado em zero) — e o
    limiar SELECIONADO NA VALIDAÇÃO, lido de `selecao_limiar.limiar`. Nunca
    0,50, nunca recalculado no conjunto que está sendo diagnosticado.

O QUE ESTE MÓDULO NÃO FAZ:
    não treina, não seleciona limiar e não toca no conjunto de teste. Quem o
    usar continua responsável por filtrar `conjunto == 'validacao'`.
"""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from .avaliacao import predizer_rf

# Chave curta -> (artefato do modelo, JSON de métricas, rótulo para figura/texto).
# O par joblib/json é fixo: o limiar de um modelo só vale para AQUELE modelo.
MODELOS_PRINCIPAIS = {
    "rf": ("rf_tuned_principal.joblib", "rf_tuned_principal.json",
           "RF ajustado (braço principal)"),
    "svm": ("svm_tuned_principal.joblib", "svm_tuned_principal.json",
            "SVM ajustado (braço principal)"),
}


def carregar_modelo_ajustado(raiz: Path, chave: str) -> dict:
    """Devolve o modelo persistido, o limiar do protocolo e a proveniência.

    Args:
        raiz: raiz do repositório.
        chave: 'rf' ou 'svm'.

    Returns:
        dict com modelo, limiar, rotulo, nome_arquivo e o JSON de métricas
        inteiro (`metricas`), para que o chamador possa registrar de onde o
        número veio sem reabrir o arquivo.

    Raises:
        KeyError: chave desconhecida.
        FileNotFoundError: artefato ausente (rode src.models.ajustar_rf /
            src.models.treinar_svm antes).
        ValueError: o JSON não traz `selecao_limiar.limiar`, ou o limiar não foi
            selecionado na validação — recusar é mais seguro que adivinhar.
    """
    if chave not in MODELOS_PRINCIPAIS:
        raise KeyError(f"Modelo '{chave}' desconhecido. "
                       f"Conhecidos: {sorted(MODELOS_PRINCIPAIS)}")
    arq_modelo, arq_json, rotulo = MODELOS_PRINCIPAIS[chave]

    caminho_modelo = raiz / "models" / arq_modelo
    caminho_json = raiz / "results" / "metricas" / arq_json
    for c in (caminho_modelo, caminho_json):
        if not c.exists():
            raise FileNotFoundError(
                f"Artefato ausente: {c}. Rode o pipeline do Bloco 3 antes "
                "(python -m src.models.ajustar_rf / src.models.treinar_svm).")

    with open(caminho_json, encoding="utf-8") as f:
        metricas = json.load(f)

    sel = metricas.get("selecao_limiar")
    if not sel or "limiar" not in sel:
        raise ValueError(f"{arq_json} não traz selecao_limiar.limiar — o limiar "
                         "do protocolo não pode ser inferido nem substituído "
                         "por 0,50.")
    # Guarda de conjunto: o limiar do protocolo é escolhido na VALIDAÇÃO. Um
    # limiar vindo de outro conjunto entraria aqui sem alarde e contaminaria
    # todo diagnóstico que este módulo alimenta.
    if sel.get("conjunto") != "validacao":
        raise ValueError(f"{arq_json}: limiar selecionado em "
                         f"'{sel.get('conjunto')}', não em 'validacao'.")

    return {
        "chave": chave,
        "modelo": joblib.load(caminho_modelo),
        "limiar": float(sel["limiar"]),
        "rotulo": rotulo,
        "nome_arquivo": f"{chave}_tuned_principal",
        "criterio_limiar": sel.get("criterio"),
        "origem_limiar": f"results/metricas/{arq_json} -> selecao_limiar.limiar",
        "metricas": metricas,
    }


def scores_de(carregado: dict, X: np.ndarray) -> np.ndarray:
    """Score de spoof na escala nativa de cada modelo.

    RF -> predict_proba(X)[:, 1] (probabilidade em [0,1]).
    SVM -> decision_function(X) (real, centrado em zero; o Pipeline já embute o
    StandardScaler, então X entra CRU, exatamente como no treino).

    Misturar as duas escalas é o erro que invalidaria tudo em silêncio: aplicar
    um limiar de 0,65 a um decision_function classificaria quase tudo como
    bonafide sem erro nenhum aparecer.
    """
    modelo = carregado["modelo"]
    if carregado["chave"] == "rf":
        # predizer_rf força n_jobs=1: com n_jobs=-1 a soma das árvores muda de
        # ordem entre execuções e o vetor não reproduz bit a bit (ver docstring
        # de predizer_rf). Num script de diagnóstico isso significaria uma
        # tabela que muda de casa decimal sem nada ter mudado.
        return predizer_rf(modelo, X)
    # decision_function do SVC é single-thread e determinístico.
    return modelo.decision_function(X)


def hashes_congelados(raiz: Path, caminho_subamostra: str) -> dict:
    """MD5 dos três artefatos congelados — mesmo padrão de rf_tuned_principal.json.

    Sem isto, um diagnóstico não diz sobre QUAIS dados foi medido; com features
    congeladas desde 30/08/2026, o hash é a prova de que foi sobre elas.
    """
    return {
        "features": hashlib.md5(
            (raiz / "data" / "features" / "features.csv").read_bytes()).hexdigest(),
        "split": hashlib.md5(
            (raiz / "data" / "processed" / "split.csv").read_bytes()).hexdigest(),
        "subamostra": hashlib.md5(
            (raiz / caminho_subamostra).read_bytes()).hexdigest(),
    }
