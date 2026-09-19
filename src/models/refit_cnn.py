"""
B4.6 — Refit final da CNN nos 30.000 completos (P6)
====================================================

ESTE MARCO NÃO DECIDE NADA. Nenhum hiperparâmetro novo, nenhuma métrica de decisão,
nenhum toque na validação externa. Tudo o que ele faz é APLICAR números que o B4.5 já
fechou: a arquitetura vencedora e a melhor época saem de
`results/metricas/cnn_definida.json` e entram aqui como constantes lidas de arquivo,
nunca redigitadas. Se este módulo precisar escolher alguma coisa, algo do B4.5 ficou
pendente — a correção é fechar lá, não improvisar aqui.

POR QUE O REFIT EXISTE (a resposta de banca, em três frases):
    O *early stopping* precisa de um conjunto que o modelo não veja, e esse conjunto
    saiu dos 30k, deixando 27.000 para treinar. Sem refit, a CNN final teria sido
    treinada em 27.000 exemplos enquanto RF e SVM treinaram em 30.000, e a comparação
    passaria a carregar um fator escondido de 3.000 amostras ALÉM do modelo. O refit
    remove esse privilégio ao contrário: retreina a arquitetura escolhida nos 30.000
    completos, pelo número de épocas já definido, e com isso a CNN final vê as MESMAS
    30 mil amostras que RF e SVM.

    «Por que o refit nos 30k remove o privilégio que RF e SVM não tiveram?» — RF e SVM
    nunca precisaram separar um pedaço do treino para decidir quando parar (o RF não
    tem «quando parar»; o SVM converge). A CNN precisou. O refit paga essa dívida.

POR QUE ESTE MÓDULO É NOVO EM VEZ DE UMA FLAG EM `treinar_cnn.py`:
    `cnn_definida.json` registra o md5 de `cnn.py`, `treinar_cnn.py` e
    `definir_cnn.py` como evidência de que a validação externa ficou isolada durante a
    busca. Editar qualquer um dos três invalidaria aquela evidência e forçaria a
    re-execução da grade — 93 minutos de GPU para não ganhar nada. Então o refit mora
    num módulo próprio e IMPORTA o que não pode divergir:

        `CnnDeepfake` e `pesos_balanced`   de src/models/cnn.py
        `EspectrogramaDataset` e `_uma_epoca`  de src/models/treinar_cnn.py
        os auxiliares da auditoria de isolamento  de src/models/definir_cnn.py

    É a mesma regra que mantém `avaliacao.py` como régua única dos três modelos: num
    trabalho cuja pergunta central é COMPARAR, uma divergência silenciosa entre cópias
    invalidaria a comparação. O que NÃO dá para importar é o preparo dos dados —
    `preparar_treino` está amarrado ao split interno 27k/3k e às estatísticas dos 27k,
    que são exatamente as duas coisas que o refit tem de trocar. Essa parte é escrita
    aqui, e a divergência está declarada em `divergencias_declaradas` no JSON.

O QUE ESTE SCRIPT NÃO FAZ, DE PROPÓSITO:
    - não usa *early stopping*, não monta conjunto de acompanhamento e não para de
      forma adaptativa: o número de épocas é FIXO por decisão do protocolo (P6);
    - não calcula métrica nenhuma, em conjunto nenhum. Ver o comentário em
      `laco_refit()`, que é a parte mais importante deste arquivo.

Rode a partir da raiz:
    python -m scripts.calcular_normalizacao_cnn --estagio 30k_refit   # passo 1, antes
    python -m src.models.refit_cnn                                    # passos 2 a 4
    python -m src.models.refit_cnn --fumaca    # ensaio de 1 epoca, artefatos _FUMACA
"""

import argparse
import ast
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .cnn import CnnDeepfake, pesos_balanced
from .definir_cnn import (NOMES_DE_DADOS_PROIBIDOS, _chamadas_de_limiar,
                          _constantes_de_codigo, _identificadores)
from .tempo import ambiente
from .treinar_cnn import EspectrogramaDataset, _uma_epoca
from ..utils.seeds import fixar_seeds_torch

RAIZ = Path(__file__).resolve().parents[2]
NOME = "refit_cnn"
MODELO = "cnn_final_30k"

DEFINIDA = "results/metricas/cnn_definida.json"
INDICE = "data/espectrogramas/indice_treino_30k.csv"
MEMMAP = "data/espectrogramas/treino_30k.npy"
NORMALIZACAO = "data/espectrogramas/normalizacao_cnn_30k.json"
ESTAGIO_EXIGIDO = "30k_refit"

# Os quatro módulos do caminho de treino deste marco. Os três primeiros são os
# mesmos que `cnn_definida.json` audita — reauditá-los aqui é o item do critério de
# pronto que cobra «o md5 ainda confere». O quarto é este arquivo.
ARQUIVOS_AUDITADOS = (
    "src/models/cnn.py",
    "src/models/treinar_cnn.py",
    "src/models/definir_cnn.py",
    "src/models/refit_cnn.py",
)


# =============================================================================
# 1. Os números que vêm do B4.5 — lidos, nunca redigitados
# =============================================================================
def carregar_decisoes_b45(raiz: Path) -> dict:
    """A arquitetura e a melhor época, de `cnn_definida.json`, com as guardas.

    Um refit que redigita «37» ou «canais 32,64,128,128» é um refit que pode divergir
    do modelo selecionado sem ninguém perceber — e o JSON continuaria dizendo que não
    divergiu. Então tudo sai do artefato, e o que não sair dele levanta erro.

    A guarda do `weight_decay` existe porque este módulo constrói o Adam sem passá-lo,
    exatamente como `preparar_treino` faz. Isso só é fiel enquanto a configuração
    vencedora tiver `weight_decay = 0` (o padrão do Adam). Se um dia não tiver, o
    refit tem de falhar alto em vez de treinar um modelo diferente do escolhido.
    """
    with open(raiz / DEFINIDA, encoding="utf-8") as f:
        d = json.load(f)

    arq, hp = d["arquitetura"], d["hiperparametros"]
    n_epocas = int(d["melhor_epoca"])
    assert n_epocas >= 1, f"melhor_epoca invalida em {DEFINIDA}: {n_epocas}"
    assert float(hp["weight_decay"]) == 0.0, (
        "a configuracao vencedora tem weight_decay != 0, e este modulo constroi o "
        "Adam sem passar weight_decay — corrija antes de rodar")
    assert hp["otimizador"] == "Adam", f"otimizador inesperado: {hp['otimizador']}"
    assert float(arq["p_dropout"]) == float(hp["p_dropout"]), (
        "p_dropout diverge entre os blocos `arquitetura` e `hiperparametros`")

    # As estatisticas dos 27k tem de estar INTACTAS: o artefato do early stopping e
    # o que torna a melhor epoca reproduzivel, e sobrescreve-lo e uma das armadilhas
    # listadas do marco. O md5 registrado no B4.5 e a prova.
    md5_27k = hashlib.md5(
        (raiz / "data" / "espectrogramas" / "normalizacao_cnn.json").read_bytes()
    ).hexdigest()
    assert md5_27k == d["hash_md5_normalizacao_json"], (
        f"as estatisticas dos 27k mudaram (md5 {md5_27k}, registrado no B4.5 "
        f"{d['hash_md5_normalizacao_json']}) — o artefato do early stopping tem de "
        "continuar intacto")

    return {
        "n_epocas": n_epocas,
        "canais": tuple(int(c) for c in arq["canais"]),
        "p_drop": float(arq["p_dropout"]),
        "lr": float(hp["lr"]),
        "batch": int(hp["batch"]),
        "arquitetura": arq,
        "md5_normalizacao_27k": md5_27k,
        "md5_normalizacao_27k_registrado_no_b45": d["hash_md5_normalizacao_json"],
        "config_id_vencedora": d["selecao_arquitetura"]["config_id_vencedora"],
        "limitacao_l3": next(
            (lim for lim in d["limitacoes_registradas"]
             if lim["id"] == "L3-epocas-fixas-nao-sao-passos-fixos"), None),
    }


def _loss_b45(raiz: Path, config_id: str, epoca: int) -> float | None:
    """A loss de TREINO do B4.5, na mesma época, para a única conferência legítima.

    O marco é explícito sobre o que pode e o que não pode ser olhado antes do B4.7:
    a validação externa, não; a loss de treino do refit comparada com a do B4.5 na
    mesma época, sim. Mesma ordem de grandeza é o esperado; muito diferente aponta
    normalização errada no passo 1. E MESMO ASSIM isso vira nota, não re-treino —
    o número que decide sai no B4.7.
    """
    caminho = raiz / "results" / "metricas" / "curvas_busca_cnn.csv"
    if not caminho.exists():
        return None
    curvas = pd.read_csv(caminho)
    linha = curvas[(curvas["config_id"] == config_id) & (curvas["epoca"] == epoca)]
    return float(linha["loss_treino"].iloc[0]) if len(linha) else None


# =============================================================================
# 2. Os dados do refit — as 30.000 linhas inteiras, sem filtrar
# =============================================================================
def preparar_dados_refit(raiz: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
    """O índice INTEIRO e as estatísticas do estágio `30k_refit`.

    NÃO SE FILTRA NADA AQUI, e é o ponto do marco. Os 3.000 que eram o conjunto de
    early stopping voltam para o treino com os `n_frames_validos` que já tinham desde
    o B4.2 — a armadilha listada («esquecer de recomputar o n_valid dos 3.000 novos
    exemplos») se resolve exatamente por não filtrar.

    A guarda do `estagio` é o que impede o erro mais caro deste marco: treinar a CNN
    final com as estatísticas dos 27k, que violaria a P3 («somente o treino
    efetivamente disponível naquele estágio») e normalizaria a rede final por um
    conjunto que não é o seu treino.
    """
    indice = pd.read_csv(raiz / INDICE)

    with open(raiz / NORMALIZACAO, encoding="utf-8") as f:
        norm = json.load(f)
    assert norm["estagio"] == ESTAGIO_EXIGIDO, (
        f"{NORMALIZACAO} esta no estagio {norm['estagio']!r}, esperado "
        f"{ESTAGIO_EXIGIDO!r} — rode "
        "`python -m scripts.calcular_normalizacao_cnn --estagio 30k_refit`")
    assert int(norm["n_exemplos"]) == len(indice), (
        f"as estatisticas sao de {norm['n_exemplos']} exemplos e o indice tem "
        f"{len(indice)} — nao sao do mesmo conjunto")

    media = np.array(norm["media_por_mel"], dtype=np.float64)
    desvio = np.array(norm["desvio_por_mel"], dtype=np.float64)
    assert media.shape == desvio.shape == (128,), "normalizacao nao tem 128 faixas"
    assert float(desvio.min()) > 0, "desvio por faixa Mel com zero -> divisao por zero"

    # As contagens sao DERIVADAS do indice. Reaproveitar os pesos dos 27k manteria a
    # razao 9:1 e nao mudaria nada na pratica — mas o JSON passaria a registrar
    # numeros que nao correspondem ao conjunto, e registro e o que este trabalho tem
    # de melhor.
    n_bonafide = int((indice["classe_binaria"] == 0).sum())
    n_spoof = int((indice["classe_binaria"] == 1).sum())
    assert n_bonafide + n_spoof == len(indice), "classe_binaria tem valor inesperado"

    resumo = {
        "n_treino": int(len(indice)),
        "n_bonafide_treino": n_bonafide,
        "n_spoof_treino": n_spoof,
        "norm": norm,
    }
    return indice, media, desvio, resumo


# =============================================================================
# 3. Auditoria de isolamento — «confirme por leitura do código, não por memória»
# =============================================================================
def auditar_isolamento(raiz: Path, md5_b45: dict) -> dict:
    """Nenhum conjunto de avaliação foi aberto — respondido LENDO O CÓDIGO.

    Reusa os mesmos auxiliares de AST do B4.5 (`_constantes_de_codigo`,
    `_identificadores`, `_chamadas_de_limiar`, `NOMES_DE_DADOS_PROIBIDOS`) em vez de
    recriá-los: uma segunda implementação da guarda poderia ficar mais frouxa que a
    primeira sem ninguém notar, e aí a evidência dos dois marcos deixaria de ser
    comparável.

    Duas coisas são cobradas aqui, e a segunda é exclusiva deste marco:
      1. nenhum dos quatro módulos referencia dado de avaliação, e nenhuma chamada a
         `selecionar_limiar` herda o default da assinatura;
      2. o md5 dos TRÊS módulos auditados pelo B4.5 continua idêntico ao registrado
         em `cnn_definida.json`. Se algum tiver mudado, a evidência de isolamento
         daquele marco caducou e a grade teria de ser re-executada — então o refit
         para aqui em vez de produzir um artefato que se apoia numa prova vencida.
    """
    violacoes, arquivos = [], {}
    for rel in ARQUIVOS_AUDITADOS:
        caminho = raiz / rel
        arvore = ast.parse(caminho.read_text(encoding="utf-8"))

        tokens = _constantes_de_codigo(arvore) + [
            n for n in _identificadores(arvore) if n]
        for proibido in NOMES_DE_DADOS_PROIBIDOS:
            if any(proibido in t for t in tokens):
                violacoes.append(f"{rel}: codigo referencia «{proibido}»")

        chamadas = _chamadas_de_limiar(arvore)
        for chamada in chamadas:
            if not chamada["explicito"]:
                violacoes.append(
                    f"{rel}:{chamada['linha']}: selecionar_limiar sem `conjunto=` "
                    "explicito — herdaria o default 'validacao'")
            elif chamada["conjunto_declarado"] != "early_stopping_interno":
                violacoes.append(
                    f"{rel}:{chamada['linha']}: selecionar_limiar com conjunto="
                    f"{chamada['conjunto_declarado']!r}, esperado "
                    "'early_stopping_interno'")

        md5 = hashlib.md5(caminho.read_bytes()).hexdigest()
        registrado = md5_b45.get(rel, {}).get("md5")
        if registrado is not None and md5 != registrado:
            violacoes.append(
                f"{rel}: md5 {md5} difere do registrado no B4.5 ({registrado}) — a "
                "evidencia de isolamento da grade caducou")
        arquivos[rel] = {
            "md5": md5,
            "md5_registrado_no_b45": registrado,
            "confere_com_o_b45": None if registrado is None else md5 == registrado,
            "chamadas_selecionar_limiar": chamadas,
        }

    assert not violacoes, ("ISOLAMENTO VIOLADO:\n  " + "\n  ".join(violacoes))

    return {
        "verificado_por": ("leitura do codigo-fonte (AST) dos modulos do caminho de "
                           "treino do refit, nao por memoria nem por afirmacao "
                           "escrita a mao"),
        "arquivos_auditados": arquivos,
        "nomes_de_dados_proibidos": list(NOMES_DE_DADOS_PROIBIDOS),
        "o_que_foi_checado": [
            "nenhum identificador nem string de codigo dos modulos auditados contem "
            "qualquer um dos nomes listados em `nomes_de_dados_proibidos`",
            "toda chamada a selecionar_limiar nos modulos auditados declara o "
            "conjunto EXPLICITAMENTE, e declara o conjunto interno — nunca herda o "
            "default da assinatura. Em src/models/refit_cnn.py nao ha nenhuma: o "
            "refit nao seleciona limiar, porque nao avalia nada",
            "o md5 dos tres modulos auditados pelo B4.5 continua identico ao "
            "registrado em cnn_definida.json -> isolamento_validacao",
        ],
        "o_que_NAO_e_checado": (
            "docstrings e comentarios sao ignorados de proposito — e neles que o "
            "projeto DOCUMENTA a proibicao, e uma guarda que dispara com a propria "
            "documentacao seria ruido que ensina a ignorar o alarme"),
        "dados_efetivamente_lidos": [
            "data/espectrogramas/treino_30k.npy (a subamostra de 30k)",
            "data/espectrogramas/indice_treino_30k.csv (as 30.000 linhas, sem filtro)",
            "data/espectrogramas/normalizacao_cnn_30k.json (estatisticas do refit)",
            "results/metricas/cnn_definida.json (arquitetura e melhor epoca)",
            "results/metricas/curvas_busca_cnn.csv (a loss de TREINO do B4.5 na "
            "mesma epoca, unica conferencia permitida antes do B4.7)",
        ],
        "resultado": ("nenhum conjunto de avaliacao foi aberto: nem a validacao "
                      "externa (22.226), que e B4.7, nem o teste (22.227), lacrado "
                      "ate B5.1"),
    }


# =============================================================================
# 4. O laço — épocas fixas, e nada mais
# =============================================================================
def laco_refit(modelo, dl, criterio, otimizador, dispositivo, n_epocas: int) -> list:
    """`n_epocas` épocas, na marra, registrando SÓ a loss de treino.

    A curva deste marco tem UMA série e não duas: não existe conjunto de
    acompanhamento para desenhar a segunda, e isso é de propósito, não esquecimento.
    """
    # O refit NAO avalia nada. Nem na validacao externa (que e B4.7 e serve so
    # para escolher o limiar), nem no teste (lacrado ate B5.1). Um refit que
    # "so da uma olhadinha" na validacao para conferir se ficou bom transforma o
    # limiar do B4.7 num limiar escolhido depois de espiar — e ai o protocolo de
    # RF/SVM e o da CNN deixam de ser o mesmo.
    #
    # Nenhum early stopping. Nenhum conjunto de validacao. Nenhuma parada
    # adaptativa. O numero de epocas vem de cnn_definida.json -> melhor_epoca, e
    # e FIXO por decisao do protocolo (P6).
    historico = []
    for epoca in range(1, n_epocas + 1):
        t0 = time.perf_counter()
        loss_tr = _uma_epoca(modelo, dl, criterio, otimizador, dispositivo)
        dt = time.perf_counter() - t0
        historico.append({"epoca": epoca, "loss_treino": loss_tr,
                          "tempo_s": round(dt, 2)})
        print(f"epoca {epoca:>3}/{n_epocas} | loss_tr {loss_tr:.4f} | {dt:.1f}s")
    return historico


# =============================================================================
# 5. Execução
# =============================================================================
def executar(cfg: dict, raiz: Path, estrito: bool = True, sufixo: str = "",
             epocas_forcadas: int | None = None) -> dict:
    b45 = carregar_decisoes_b45(raiz)
    n_epocas = epocas_forcadas if epocas_forcadas is not None else b45["n_epocas"]

    with open(raiz / DEFINIDA, encoding="utf-8") as f:
        md5_b45 = json.load(f)["isolamento_validacao"]["arquivos_auditados"]
    print("Auditando isolamento (leitura do código) e md5 dos módulos do B4.5...")
    isolamento = auditar_isolamento(raiz, md5_b45)
    print(f"  OK — {isolamento['resultado']}")

    # ---- Sementes e determinismo: a MESMA ordem de `preparar_treino` ---------
    # fixar_seeds_torch define CUBLAS_WORKSPACE_CONFIG, que so tem efeito se nenhum
    # contexto CUDA foi criado ainda — por isso vem antes de tudo. A sondagem existe
    # porque use_deterministic_algorithms(True) nao falha ao ser CHAMADO, falha
    # quando uma operacao sem versao deterministica e EXECUTADA; e a semente e
    # re-fixada DEPOIS dela, porque a passada falsa consome estado do gerador.
    semente = fixar_seeds_torch(int(cfg["semente"]), estrito=estrito)
    determinismo_estrito, limitacao_determinismo = estrito, None
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {dispositivo}")

    if estrito:
        try:
            _m = CnnDeepfake(n_mels=128, canais=b45["canais"],
                             p_drop=b45["p_drop"]).to(dispositivo)
            _x = torch.zeros(2, 1, 128, 251, device=dispositivo)
            _mask = torch.ones(2, 251, device=dispositivo)
            nn.CrossEntropyLoss()(_m(_x, _mask),
                                  torch.zeros(2, dtype=torch.long,
                                              device=dispositivo)).backward()
            if dispositivo.type == "cuda":
                torch.cuda.synchronize()
            del _m, _x, _mask
        except RuntimeError as e:
            limitacao_determinismo = (
                f"use_deterministic_algorithms(True) falhou numa operacao do "
                f"forward/backward: {e}. Reexecutado com estrito=False e REGISTRADO "
                "como limitacao.")
            print(f"AVISO: {limitacao_determinismo}")
            determinismo_estrito = False
        semente = fixar_seeds_torch(int(cfg["semente"]), estrito=determinismo_estrito)

    # ---- Dados ---------------------------------------------------------------
    indice, media, desvio, resumo = preparar_dados_refit(raiz)
    norm = resumo.pop("norm")
    print(f"Treino do refit: {resumo['n_treino']} "
          f"({resumo['n_bonafide_treino']} bonafide / "
          f"{resumo['n_spoof_treino']} spoof)")

    ds = EspectrogramaDataset(raiz / MEMMAP, indice, media, desvio)
    x0, m0, y0 = ds[0]
    assert torch.isfinite(x0).all(), "espectrograma normalizado com NaN/inf"
    assert x0.shape == (1, 128, 251), f"formato inesperado: {tuple(x0.shape)}"
    assert int(y0) == int(indice.loc[0, "classe_binaria"]), "rotulo desalinhado"
    assert float(m0.sum()) == float(indice.loc[0, "n_frames_validos"]), (
        "mascara desalinhada do indice")

    gerador = torch.Generator().manual_seed(semente)
    dl = torch.utils.data.DataLoader(ds, batch_size=b45["batch"], shuffle=True,
                                     num_workers=0, generator=gerador)

    # ---- Modelo, loss (P7) e otimizador -------------------------------------
    modelo = CnnDeepfake(n_mels=128, canais=b45["canais"],
                         p_drop=b45["p_drop"]).to(dispositivo)
    n_por_classe = [resumo["n_bonafide_treino"], resumo["n_spoof_treino"]]
    pesos = pesos_balanced(n_por_classe, dispositivo)
    criterio = nn.CrossEntropyLoss(weight=pesos)
    otimizador = torch.optim.Adam(modelo.parameters(), lr=b45["lr"])
    print(f"Pesos da loss: bonafide {float(pesos[0]):.4f} x spoof {float(pesos[1]):.4f} "
          f"(razao {float(pesos[0] / pesos[1]):.2f}:1 a favor do bonafide)")
    print(f"Parametros: {modelo.descricao()['n_parametros']:,}")
    print(f"Epocas FIXAS: {n_epocas} (de {DEFINIDA} -> melhor_epoca)\n")

    historico = laco_refit(modelo, dl, criterio, otimizador, dispositivo, n_epocas)

    # ---- A única conferência legítima antes do B4.7 -------------------------
    loss_final = historico[-1]["loss_treino"]
    loss_b45 = _loss_b45(raiz, b45["config_id_vencedora"], n_epocas)
    if loss_b45 is not None:
        razao = loss_final / loss_b45 if loss_b45 else float("inf")
        print(f"\nloss de treino final do refit: {loss_final:.4f} | "
              f"B4.5 na epoca {n_epocas}: {loss_b45:.4f} | razao {razao:.2f}x")
        print("  (mesma ordem de grandeza e o esperado. Muito diferente sugere "
              "normalizacao errada no passo 1 — e ainda assim vira NOTA, nao "
              "re-treino: o numero que decide sai no B4.7.)")

    # ---- O registro ---------------------------------------------------------
    m = _montar_registro(raiz, cfg, b45, resumo, norm, historico, isolamento,
                         modelo, pesos, n_epocas, semente, determinismo_estrito,
                         limitacao_determinismo, dispositivo, loss_b45)

    dir_met = raiz / "results" / "metricas"
    dir_met.mkdir(parents=True, exist_ok=True)
    caminho_curva = dir_met / f"curva_{NOME}{sufixo}.csv"
    with open(caminho_curva, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(historico[0].keys()))
        w.writeheader()
        w.writerows(historico)
    print(f"Curva salva em {caminho_curva.relative_to(raiz)}")

    (raiz / "models").mkdir(exist_ok=True)
    caminho_pt = raiz / "models" / f"{MODELO}{sufixo}.pt"
    torch.save({"state_dict": modelo.state_dict(),
                "arquitetura": modelo.descricao(),
                "n_epocas_fixas": n_epocas,
                "estagio_normalizacao": norm["estagio"],
                "semente": semente}, caminho_pt)

    caminho_json = dir_met / f"{NOME}{sufixo}.json"
    with open(caminho_json, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)
    print(f"Salvo: models/{MODELO}{sufixo}.pt e "
          f"results/metricas/{NOME}{sufixo}.json")
    return m


def _montar_registro(raiz, cfg, b45, resumo, norm, historico, isolamento, modelo,
                     pesos, n_epocas, semente, determinismo_estrito,
                     limitacao_determinismo, dispositivo, loss_b45) -> dict:
    """O `refit_cnn.json` — a assinatura do modelo final."""
    m = {
        "modelo": MODELO,
        "marco": "B4.6 — refit final nos 30 mil completos (P6)",
        "braco": "principal",
        "n_treino": resumo["n_treino"],
        "n_bonafide_treino": resumo["n_bonafide_treino"],
        "n_spoof_treino": resumo["n_spoof_treino"],
        "por_que_o_refit_existe": (
            "o early stopping precisa de um conjunto que o modelo nao veja, e ele "
            "saiu dos 30k, deixando 27.000 para treinar. Sem refit, a CNN final "
            "teria sido treinada em 27.000 exemplos enquanto RF e SVM treinaram em "
            "30.000, e a comparacao carregaria um fator escondido de 3.000 amostras "
            "ALEM do modelo. Com o refit, a CNN final ve as MESMAS 30 mil amostras "
            "que RF e SVM. RF e SVM nunca precisaram separar um pedaco do treino "
            "para decidir quando parar (o RF nao tem «quando parar»; o SVM "
            "converge); a CNN precisou, e o refit paga essa divida."),
        "arquitetura": b45["arquitetura"],
        "origem_da_arquitetura": (
            f"{DEFINIDA} -> arquitetura (config vencedora "
            f"{b45['config_id_vencedora']}); copiada integralmente, nao redigitada"),
        "hiperparametros": {
            "lr": b45["lr"],
            "batch": b45["batch"],
            "dropout": b45["p_drop"],
            "otimizador": "Adam",
            "weight_decay": 0.0,
            "scheduler": None,
            "amp_autocast": False,
        },
        "pesos_loss": {
            "bonafide": round(float(pesos[0]), 4),
            "spoof": round(float(pesos[1]), 4),
            "razao": round(float(pesos[0] / pesos[1]), 4),
            "formula": "n / (k * n_c), mesma logica do class_weight='balanced'",
            "contagens_usadas": {"bonafide": resumo["n_bonafide_treino"],
                                 "spoof": resumo["n_spoof_treino"]},
            "nota": ("DERIVADOS das contagens reais do conjunto de refit, nao "
                     "reaproveitados dos 27k. A razao continua 9:1 e na pratica nao "
                     "muda nada — mas reaproveitar faria o JSON registrar numeros "
                     "que nao correspondem ao conjunto, e isso e erro de registro"),
            "reamostragem": "nenhuma — sem undersampling e sem oversampling (P7)",
        },
        "n_epocas_fixas": n_epocas,
        "origem_n_epocas": (
            "results/metricas/cnn_definida.json -> melhor_epoca (early stopping nos "
            "3k internos)"),
        "nota_epocas": (
            "mesmo NUMERO DE EPOCAS da fase de early stopping; como os 30k sao 11,1% "
            "maiores que os 27k, corresponde a 11,1% mais atualizacoes de gradiente "
            "— limitacao registrada"),
        "early_stopping": {
            "usado": False,
            "conjunto_de_acompanhamento": None,
            "nota": ("nenhum. O refit e por numero FIXO de epocas (P6), e por isso a "
                     "curva deste marco tem uma serie so — nao ha conjunto de "
                     "acompanhamento para desenhar a segunda"),
        },
        "normalizacao": f"{NORMALIZACAO} (estagio {norm['estagio']})",
        "normalizacao_detalhe": {
            "estagio": norm["estagio"],
            "n_exemplos": norm["n_exemplos"],
            "eixo": norm["eixo"],
            "faixas_com_desvio_clampado": norm["faixas_com_desvio_clampado"],
            "hash_lista_ids": norm["hash_lista_ids"],
            "artefato_dos_27k_preservado": "data/espectrogramas/normalizacao_cnn.json",
            "md5_dos_27k_agora": b45["md5_normalizacao_27k"],
            "md5_dos_27k_registrado_no_b45": b45["md5_normalizacao_27k_registrado_no_b45"],
            "md5_dos_27k_confere_com_o_b45": (
                b45["md5_normalizacao_27k"]
                == b45["md5_normalizacao_27k_registrado_no_b45"]),
            "por_que_dois_artefatos": (
                "P3: estatisticas globais do treino EFETIVAMENTE DISPONIVEL naquele "
                "estagio. No early stopping eram os 27k; no refit sao os 30k. Os "
                "dois coexistem — o dos 27k documenta a fase que escolheu a melhor "
                "epoca, o dos 30k documenta a CNN final — e o campo `estagio` e o "
                "que impede alguem de pegar o errado. A inferencia do B4.7 e do "
                "B5.1 usa o dos 30k"),
        },
        "sem_avaliacao": (
            "nenhum conjunto de avaliacao foi tocado neste marco: a validacao "
            "externa e B4.7 e o teste e B5.1"),
        "isolamento_validacao": isolamento,
        "conferencia_loss_treino": {
            "loss_treino_final_refit": historico[-1]["loss_treino"],
            "loss_treino_b45_na_mesma_epoca": loss_b45,
            "epoca": n_epocas,
            "config_id_b45": b45["config_id_vencedora"],
            "razao": (historico[-1]["loss_treino"] / loss_b45
                      if loss_b45 else None),
            "por_que_esta_e_a_unica_conferencia_legitima": (
                "a validacao externa nao pode ser olhada antes do B4.7 — «dar uma "
                "olhadinha» transformaria o limiar do B4.7 num limiar escolhido "
                "depois de espiar, e o protocolo de RF/SVM e o da CNN deixariam de "
                "ser o mesmo. A loss de TREINO do refit comparada com a do B4.5 na "
                "mesma epoca nao toca conjunto nenhum de avaliacao. Mesma ordem de "
                "grandeza e o esperado; muito diferente sugere normalizacao errada "
                "no passo 1, e ainda assim vira NOTA, nao re-treino"),
        },
        "loss_treino_final": historico[-1]["loss_treino"],
        "loss_treino_primeira_epoca": historico[0]["loss_treino"],
        "curva_loss_treino": f"results/metricas/curva_{NOME}.csv",
        "nota_curva": ("so a loss de TREINO, porque nao ha conjunto de "
                       "acompanhamento — e isso e de proposito"),
        "tempo_treino_s": round(sum(h["tempo_s"] for h in historico), 2),
        "tempo_por_epoca_s_mediana": round(
            float(np.median([h["tempo_s"] for h in historico])), 2),
        "semente": semente,
        "determinismo": {
            "fixar_seeds_torch": True,
            "estrito": determinismo_estrito,
            "limitacao": limitacao_determinismo,
            "num_workers": 0,
        },
    }

    m["ambiente"] = ambiente(n_jobs_inferencia=1)
    m["ambiente"]["torch"] = torch.__version__
    m["ambiente"]["cuda_disponivel"] = bool(torch.cuda.is_available())
    m["ambiente"]["cuda_versao"] = torch.version.cuda
    m["ambiente"]["gpu"] = (torch.cuda.get_device_name(0)
                            if torch.cuda.is_available() else None)
    m["ambiente"]["compute_capability"] = (list(torch.cuda.get_device_capability(0))
                                           if torch.cuda.is_available() else None)
    m["ambiente"]["dispositivo_treino"] = str(dispositivo)

    # ---- Deltas entre as duas estatisticas (conferencia do passo 1) ---------
    m.update(_deltas_normalizacao(raiz))

    # ---- Limitacoes: a L3 herdada do B4.5, com o complemento do B4.6 -------
    m["limitacoes_registradas"] = [_l3_com_complemento(b45["limitacao_l3"])]

    # ---- Rastreabilidade: sobre QUAIS dados o modelo foi treinado ----------
    dir_esp = raiz / "data" / "espectrogramas"
    m["hash_md5_indice_treino_csv"] = hashlib.md5(
        (dir_esp / "indice_treino_30k.csv").read_bytes()).hexdigest()
    m["hash_md5_split_interno_csv"] = hashlib.md5(
        (raiz / "data" / "processed" / "split_interno_cnn.csv").read_bytes()).hexdigest()
    m["hash_md5_normalizacao_30k_json"] = hashlib.md5(
        (raiz / NORMALIZACAO).read_bytes()).hexdigest()
    m["hash_md5_normalizacao_27k_json"] = b45["md5_normalizacao_27k"]
    m["hash_md5_subamostra_csv"] = hashlib.md5(
        (raiz / cfg["experimento"]["caminho_subamostra"]).read_bytes()).hexdigest()
    m["hash_md5_features_csv"] = hashlib.md5(
        (raiz / "data" / "features" / "features.csv").read_bytes()).hexdigest()
    m["hash_md5_split_csv"] = hashlib.md5(
        (raiz / "data" / "processed" / "split.csv").read_bytes()).hexdigest()
    m["nota_hashes"] = (
        "os hashes dos tres artefatos congelados (features.csv, split.csv, "
        "subamostra_30k.csv) aparecem aqui pelo mesmo motivo que aparecem em "
        "rf_tuned_principal.json: sem eles, o JSON nao diz SOBRE QUAIS DADOS o "
        "modelo foi treinado")

    m["divergencias_declaradas"] = [
        "o preparo dos dados (`preparar_dados_refit`) e proprio deste modulo em vez "
        "de reusar `preparar_treino`, porque aquele esta amarrado ao split interno "
        "27k/3k e as estatisticas dos 27k — as duas coisas que o refit tem de "
        "trocar. O que NAO podia divergir foi importado: a rede (CnnDeepfake), a "
        "formula dos pesos (pesos_balanced), o Dataset com a normalizacao em tempo "
        "de carga (EspectrogramaDataset) e o passo de uma epoca (_uma_epoca). Os "
        "tres modulos do B4.5 nao foram editados, e o md5 deles esta reconferido "
        "em `isolamento_validacao`",
    ]
    return m


def _deltas_normalizacao(raiz: Path) -> dict:
    """O afastamento máximo, por faixa Mel, entre as estatísticas dos 27k e dos 30k.

    Item do critério de pronto. Deve ser pequeno — os 3.000 exemplos a mais são 10%
    do conjunto e vieram de um split estratificado. Se for grande, o problema está no
    split interno ou na leitura do memmap, e não numa diferença real entre os dois
    conjuntos. Recalculado aqui a partir dos DOIS artefatos, em vez de copiado do
    campo que o passo 1 gravou: assim o número no `refit_cnn.json` é verificável
    contra os arquivos que existem em disco agora.
    """
    def _ler(nome):
        d = json.loads((raiz / "data" / "espectrogramas" / nome).read_text("utf-8"))
        return (np.array(d["media_por_mel"], dtype=np.float64),
                np.array(d["desvio_por_mel"], dtype=np.float64))

    m27, d27 = _ler("normalizacao_cnn.json")
    m30, d30 = _ler("normalizacao_cnn_30k.json")
    return {
        "delta_max_media_por_mel": float(np.abs(m30 - m27).max()),
        "delta_max_desvio_por_mel": float(np.abs(d30 - d27).max()),
        "nota_deltas": (
            "em dB, entre as estatisticas dos 27k e as dos 30k, faixa a faixa. "
            "Esperado PEQUENO: os 3.000 exemplos a mais sao 10% do conjunto e "
            "vieram de um split estratificado. Valor grande apontaria erro no split "
            "interno ou na leitura do memmap, nao diferenca real entre os conjuntos"),
    }


def _l3_com_complemento(l3: dict | None) -> dict:
    """A L3 do B4.5, COPIADA, mais o complemento escrito no marco do B4.6.

    Os campos herdados (`descricao`, `alternativa_considerada`,
    `por_que_foi_descartada`, `razao_de_tamanho`, `percentual_a_mais_de_atualizacoes`)
    vêm de `cnn_definida.json` — copiados, não redigitados, que é o que o marco pede.
    O complemento é só o que o B4.6 acrescenta.

    UMA DECISÃO DE MONTAGEM, declarada porque muda o JSON: o complemento tem a sua
    própria razão de descarte, que fala das alternativas DELE (reduzir as épocas
    proporcionalmente, ou reintroduzir dropout no refit) e não das alternativas da L3
    original (fixar os passos). Ela entra como
    `por_que_foi_descartada_o_complemento`, pareada com
    `alternativa_considerada_para_o_complemento`, para não sobrescrever a herdada —
    que o marco manda copiar. Já `destino` e `redacao_para_o_texto` SÃO substituídos:
    as versões do B4.6 contêm as do B4.5 inteiras e as ampliam, e o destino é um
    parágrafo só.
    """
    assert l3 is not None, (
        "L3-epocas-fixas-nao-sao-passos-fixos nao encontrada em cnn_definida.json — "
        "o complemento do B4.6 nao tem do que herdar")
    entrada = dict(l3)
    entrada.update({
        "complemento_b46": (
            "A configuracao vencedora do B4.5 tem dropout 0.0 e weight decay 0, "
            "entao na fase de early stopping o UNICO regularizador em operacao era a "
            "propria parada antecipada, medida nos 3k: na epoca 37 a loss de treino "
            "era 0,0049 contra 0,2536 nos 3k. O refit remove esse mecanismo — treina "
            "por numero fixo de epocas, sem conjunto de acompanhamento — e ao mesmo "
            "tempo aumenta em 11,1% o numero de atualizacoes por epoca. Os dois "
            "efeitos vao na mesma direcao: o modelo final passa um pouco mais de "
            "tempo em regime de ajuste do que o modelo cuja epoca foi selecionada."),
        "por_que_nao_invalida": (
            "A epoca foi selecionada em dado que o modelo NAO viu, e o refit apenas "
            "a aplica — nao ha escolha nova sendo feita com dado contaminado, e a "
            "validacao externa continua intocada. O protocolo e o aprovado na P6, e "
            "o refit existe justamente para que a CNN final veja as mesmas 30 mil "
            "amostras que RF e SVM. O risco residual e de grau, nao de validade: 37 "
            "epocas podem passar um pouco do ponto otimo no conjunto maior."),
        "alternativa_considerada_para_o_complemento": (
            "Reduzir proporcionalmente o numero de epocas (37 / 1,111 = 33,3) para "
            "igualar as atualizacoes, ou reintroduzir dropout no refit."),
        "por_que_foi_descartada_o_complemento": (
            "A primeira usa um numero quebrado de epocas e contraria a letra da P6 "
            "(«numero FIXO de epocas»); a segunda mudaria a arquitetura entre a "
            "selecao e o modelo final, que e exatamente o que o refit existe para "
            "NAO fazer. Ambas custariam re-executar o B4.5."),
        "teste_da_regra_de_escopo": "NAO impede a validade do experimento principal",
        "destino": "limitacao no texto (B6.1), no mesmo paragrafo da L3",
        "redacao_para_o_texto": (
            "O refit foi executado pelo mesmo numero de epocas selecionado na fase "
            "de early stopping. Como o conjunto de refit e 11,1% maior, isso "
            "corresponde a 11,1% mais atualizacoes de gradiente; alem disso, a "
            "configuracao selecionada nao emprega dropout nem weight decay, de modo "
            "que a parada antecipada era o unico mecanismo de regularizacao em "
            "operacao naquela fase e o refit, por definicao, nao o reproduz. A "
            "alternativa — fixar o numero de atualizacoes, ou reintroduzir "
            "regularizacao explicita no refit — foi considerada e descartada por "
            "fidelidade ao protocolo aprovado, e a diferenca fica registrada como "
            "limitacao."),
        "nota_de_montagem": (
            "herdada integralmente de cnn_definida.json -> limitacoes_registradas e "
            "acrescida do complemento do B4.6. `por_que_foi_descartada` e a "
            "herdada, sobre a alternativa da L3 (fixar os passos); a razao de "
            "descarte das alternativas do COMPLEMENTO esta em "
            "`por_que_foi_descartada_o_complemento`. `destino` e "
            "`redacao_para_o_texto` sao os do B4.6, que contem os do B4.5 e os "
            "ampliam. Nada do B4.5 foi editado para isto: o complemento mora aqui, "
            "no artefato do B4.6, e nao em cnn_definida.json, porque editar aquele "
            "arquivo invalidaria a evidencia de md5 do isolamento da grade."),
        "onde_o_complemento_foi_escrito": (
            "docs/execucao/B4.6_refit_final.md, secao «Limitacao a registrar no "
            "refit_cnn.json — complemento da L3» (18/09/2026)"),
    })
    return entrada


def main() -> dict:
    from ..utils.config import carregar_config

    p = argparse.ArgumentParser(description="B4.6 — refit final da CNN nos 30k")
    p.add_argument("--nao-estrito", action="store_true",
                   help="pula a tentativa de determinismo estrito (registra a limitacao)")
    p.add_argument("--fumaca", action="store_true",
                   help="ensaio de 1 epoca, artefatos com sufixo _FUMACA")
    a = p.parse_args()

    sufixo, epocas = "", None
    if a.fumaca:
        sufixo, epocas = "_FUMACA", 1
        print("MODO FUMAÇA: 1 época, artefatos com sufixo _FUMACA. "
              "NÃO é resultado de marco.\n")

    return executar(carregar_config(RAIZ), RAIZ, estrito=not a.nao_estrito,
                    sufixo=sufixo, epocas_forcadas=epocas)


if __name__ == "__main__":
    # Windows usa `spawn`: todo codigo de execucao fica sob este guard.
    main()
