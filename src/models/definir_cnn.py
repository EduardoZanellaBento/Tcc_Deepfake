"""
B4.5 — CNN definida: arquitetura final, *early stopping* nos 3k e MELHOR ÉPOCA
===============================================================================

O QUE ESTE MARCO PRODUZ DE VERDADE É UM NÚMERO: **a melhor época**. Todo o resto —
grade, curvas, figuras — existe para justificá-lo. O B4.6 vai retreinar nos 30k por um
número FIXO de épocas, sem *early stopping*, e esse número nasce aqui. Se ele não
estiver num artefato versionado (`results/metricas/cnn_definida.json`), o B4.6 não roda.

POR QUE ESTE MÓDULO É SEPARADO DE `treinar_cnn.py`:
    Mesma separação que o ramo clássico já usa — `treinar_rf.py` é o baseline e
    `ajustar_rf.py` é a busca mais o modelo final. `treinar_cnn.py` continua sendo o
    B4.4 e continua reproduzindo `cnn_baseline.json`; a busca mora aqui. O que NÃO se
    duplica é o protocolo: `preparar_treino` e `laco_de_treino` são importados, não
    copiados. Num trabalho cuja pergunta central é COMPARAR, uma divergência
    silenciosa entre cópias invalidaria a comparação — a mesma razão que mantém
    `avaliacao.py` como régua única dos três modelos.

POR QUE UMA GRADE CURTA E NÃO RANDOM SEARCH (decisão registrada):
    A regra de escopo do projeto é explícita — «busca de hiperparâmetros ficou cara →
    reduz o espaço de busca». O ramo clássico pôde pagar um `RandomizedSearchCV`
    porque um fit de RF custa segundos; aqui um fit custa entre 6 e 25 minutos. Uma
    grade de 6 configurações DELIBERADAS, com o motivo de cada eixo escrito, é a
    resposta de banca; «testei alguns hiperparâmetros» não é.

    Assimetria proposital com o RF e o SVM, e ela tem de estar dita: a busca do ramo
    clássico pontuou por EER, uma métrica INDEPENDENTE DE LIMIAR, porque ali só se
    escolhia o modelo e o limiar viria depois. Aqui se escolhe ARQUITETURA **e**
    ÉPOCA, e a parada precisa de um número que reflita a decisão binária que o modelo
    vai tomar — daí f1_macro. Os dois são reportados para toda configuração, e se
    discordarem sobre quem vence, isso vai para o texto em vez de ser escondido.

O QUE ESTE SCRIPT NÃO FAZ, DE PROPÓSITO:
    Não abre a validação externa (22.226) nem o teste (22.227). Usar a validação
    externa como conjunto de *early stopping* é a proibição mais importante do
    Bloco 4: o limiar do protocolo nasce no B4.7, e escolhê-lo num conjunto que já
    guiou o treino destruiria o protocolo inteiro. Isso não é confiado à memória —
    `auditar_isolamento_validacao()` lê o código-fonte e cobra.

Rode a partir da raiz:
    python -m src.models.definir_cnn
    python -m src.models.definir_cnn --fumaca        # ensaio de 2 epocas, artefatos _FUMACA
    python -m src.models.definir_cnn --so-fase1      # modo "se atrasar": 4 configuracoes
"""

import argparse
import ast
import csv
import hashlib
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score

from .avaliacao import avaliar, selecionar_limiar
from .tempo import ambiente
from .treinar_cnn import _inferir, laco_de_treino, preparar_treino

RAIZ = Path(__file__).resolve().parents[2]
NOME = "cnn_definida"


# =============================================================================
# 1. O critério de parada — fixado ANTES de olhar qualquer resultado
# =============================================================================
PACIENCIA = 8
TETO_EPOCAS = 60
CRITERIO_PARADA = "f1_macro no conjunto interno de early stopping (3.000)"

# POR QUE PACIENCIA 8 E NAO 3: com apenas 300 bonafide nos 3k, o f1_macro oscila
# entre epocas por VARIANCIA DE ESTIMATIVA, nao por sobreajuste. A curva do B4.4
# mostra isso literalmente — quedas nas epocas 11, 16, 18, 22, 24 e 28, todas
# seguidas de recuperacao para um valor MAIS ALTO que o anterior. Paciencia curta
# pararia na primeira dessas flutuacoes e SUBESTIMARIA a melhor epoca, que e
# justamente o numero que o refit do B4.6 vai usar como fixo. Errar esse numero
# para baixo faz o refit treinar de menos, e o custo nao aparece em lugar nenhum
# do B4.5 — aparece so no B4.7, como um resultado pior sem causa visivel.
#
# POR QUE TETO 60 E NAO 30: no B4.4 a melhor epoca FOI A ULTIMA (30 de 30). A rede
# nao parou de treinar, acabou o orcamento. Repetir 30 aqui seria fazer a mesma
# pergunta que ja se sabe estar mal formulada.

# MARGEM DE EMPATE — declarada aqui, antes de qualquer numero, porque uma margem
# escolhida DEPOIS de ver os resultados e so uma forma educada de escolher o
# vencedor a mao. Com 300 bonafide nos 3k, o erro padrao do recall da minoritaria
# e da ordem de sqrt(0,8 x 0,2 / 300) ~ 0,023; a diferenca em f1_macro que isso
# produz fica na casa do centesimo. 0,005 e conservador: metade de um ponto
# percentual e ruido de estimativa, nao vantagem de arquitetura.
MARGEM_EMPATE_F1 = 0.005

# Janela de estabilidade: media do f1_macro nas epocas [melhor-2, melhor+2]. Serve
# ao desempate e responde a armadilha «escolher a arquitetura pelo melhor pico
# isolado de uma curva ruidosa»: prefere-se a configuracao com curva ESTAVEL em
# torno do maximo, e e esse o criterio declarado.
RAIO_JANELA_ESTABILIDADE = 2


# =============================================================================
# 2. A grade — pequena de propósito, e com o motivo de cada eixo escrito
# =============================================================================
ARQUITETURAS = {
    # rotulo: (canais por bloco)  — o numero de blocos e o comprimento da tupla
    "A4-32x64x128x128": (32, 64, 128, 128),
    "B3-16x32x64": (16, 32, 64),
}

EIXOS = {
    "profundidade_canais": (
        "capacidade x sobreajuste com 27.000 exemplos, dos quais so 2.700 sao "
        "bonafide. A arquitetura A e a do B4.4 (240.866 parametros); a B e mais rasa "
        "e mais estreita. Se a B empatar com a A, a resposta de banca deixa de ser "
        "«escolhi a maior» e passa a ser «a maior nao pagou o que custou»."
    ),
    "dropout": (
        "e o regularizador mais barato de testar e o unico que o B4.4 aplicou sem "
        "nunca ter comparado com a ausencia dele: 0,3 foi escolhido por convencao, "
        "nao por evidencia. Este eixo transforma essa convencao em medida."
    ),
    "learning_rate": (
        "a loss nos 3k do B4.4 oscilou muito (picos de ate 2,92 contra minimo de "
        "0,20) enquanto f1_macro e EER seguiam melhorando. O diagnostico registrado "
        "atribui isso a perda de CALIBRACAO, nao de desempenho — mas um lr menor e o "
        "suspeito alternativo obvio, e sai barato descarta-lo ou confirma-lo."
    ),
}

# EIXOS DEIXADOS DE FORA, E POR QUE: scheduler, weight decay, tamanho do kernel,
# BatchNorm sim/nao e batch size sao TODOS livres pela letra do marco. Ficam fora
# porque a regra de escopo manda reduzir o espaco de busca, e porque 3 eixos x 6
# configuracoes ja e o que cabe no orcamento de 1,5 a 2 dias. Eles vao ao texto
# como espaco NAO explorado — que e diferente de espaco explorado e descartado.
EIXOS_NAO_EXPLORADOS = [
    "scheduler de learning rate (mantido em None, como no B4.4)",
    "weight decay (mantido em 0, o padrao do Adam, como no B4.4)",
    "tamanho do kernel (mantido em 3x3)",
    "BatchNorm sim/nao (mantido: sim, em todos os blocos)",
    "batch size (mantido em 128)",
]

LR_BASE = 1e-3       # o do B4.4
LR_MENOR = 3e-4
DROPOUTS = (0.0, 0.3)
BATCH = 128


def montar_grade(so_fase1: bool = False) -> list:
    """A grade em duas fases — 4 configurações, depois 2 condicionadas à primeira.

    O DESENHO É O DO MARCO, LITERAL: «as duas arquiteturas × os dois dropouts com
    lr=1e-3, mais as duas arquiteturas com lr=3e-4 e o dropout que venceu». 2×2×2 = 8
    e 3×2×2 = 12 estouram o orçamento; 6 cabem numa tarde e sobra tempo para ler as
    curvas, que é o que interessa.

    A REGRA DO «DROPOUT QUE VENCEU» ESTÁ FIXADA AQUI, ANTES DE RODAR: vence o dropout
    com o MAIOR f1_macro MÉDIO entre as duas arquiteturas da fase 1 — não o que
    aparece na melhor configuração isolada. Com 300 bonafide, deixar uma única rodada
    decidir o eixo propagaria um sorteio ruidoso para a fase 2 inteira. A média sobre
    as duas arquiteturas é a estimativa mais estável que a fase 1 oferece de graça.

    A fase 2 só existe depois que a fase 1 rodou, então ela não é montada aqui — ver
    `configuracoes_da_fase_2`.
    """
    grade = [
        {"id": f"{rotulo}|drop{p:.1f}|lr{LR_BASE:g}", "fase": 1, "arquitetura": rotulo,
         "canais": canais, "p_drop": p, "lr": LR_BASE, "batch": BATCH}
        for rotulo, canais in ARQUITETURAS.items()
        for p in DROPOUTS
    ]
    if so_fase1:
        for c in grade:
            c["nota_fase"] = ("modo «se atrasar»: só a fase 1 foi executada, por "
                              "restrição de cronograma")
    return grade


def configuracoes_da_fase_2(registros_fase1: list) -> tuple[list, dict]:
    """As 2 configurações com `lr=3e-4`, no dropout vencedor da fase 1."""
    media_por_dropout = {
        p: float(np.mean([r["f1_macro_3k_melhor_epoca"]
                          for r in registros_fase1 if r["p_dropout"] == p]))
        for p in DROPOUTS
    }
    p_vencedor = max(media_por_dropout, key=media_por_dropout.get)
    decisao = {
        "regra": ("dropout com maior f1_macro MEDIO entre as duas arquiteturas da "
                  "fase 1 — fixada antes de rodar, em montar_grade"),
        "f1_macro_medio_por_dropout": {f"{p:.1f}": v for p, v in media_por_dropout.items()},
        "dropout_escolhido_para_a_fase_2": p_vencedor,
    }
    grade = [
        {"id": f"{rotulo}|drop{p_vencedor:.1f}|lr{LR_MENOR:g}", "fase": 2,
         "arquitetura": rotulo, "canais": canais, "p_drop": p_vencedor,
         "lr": LR_MENOR, "batch": BATCH}
        for rotulo, canais in ARQUITETURAS.items()
    ]
    return grade, decisao


# =============================================================================
# 3. A auditoria de isolamento — «confirme por leitura do código, não por memória»
# =============================================================================
ARQUIVOS_AUDITADOS = (
    "src/models/cnn.py",
    "src/models/treinar_cnn.py",
    "src/models/definir_cnn.py",
)

# Nomes que só apareceriam no código se alguém abrisse a validação externa ou o
# teste. NÃO se procura a palavra «validacao» solta: ela aparece dezenas de vezes
# em prosa que DIZ que a validação não é tocada, e uma guarda que dispara com a
# própria documentação treina o leitor a ignorar a guarda.
NOMES_DE_DADOS_PROIBIDOS = (
    "validacao.npy",
    "indice_validacao",
    "teste.npy",
    "indice_teste",
    "carregar_dados_split",
    "filtrar_treino_braco",
)


def _constantes_de_codigo(arvore: ast.Module) -> list:
    """Strings do módulo EXCETO docstrings e EXCETO a própria lista de proibidos.

    A distinção é o que torna a guarda utilizável. Comentários o `ast` já descarta
    sozinho; docstrings não — e são justamente onde este projeto escreve «não toca em
    validacao.npy». Sem esta exclusão a guarda acusaria a documentação de ser a
    violação. A segunda exclusão é a declaração de `NOMES_DE_DADOS_PROIBIDOS`, que por
    construção contém todos os tokens procurados.
    """
    ignorar = set()
    for no in ast.walk(arvore):
        corpo = getattr(no, "body", None)
        if isinstance(no, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                           ast.ClassDef)) and corpo:
            primeiro = corpo[0]
            if (isinstance(primeiro, ast.Expr)
                    and isinstance(primeiro.value, ast.Constant)
                    and isinstance(primeiro.value.value, str)):
                ignorar.add(id(primeiro.value))
        if isinstance(no, ast.Assign) and any(
                isinstance(a, ast.Name) and a.id == "NOMES_DE_DADOS_PROIBIDOS"
                for a in no.targets):
            ignorar.update(id(n) for n in ast.walk(no.value))

    return [n.value for n in ast.walk(arvore)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and id(n) not in ignorar]


def _identificadores(arvore: ast.Module) -> list:
    nomes = []
    for no in ast.walk(arvore):
        if isinstance(no, ast.Name):
            nomes.append(no.id)
        elif isinstance(no, ast.Attribute):
            nomes.append(no.attr)
        elif isinstance(no, (ast.Import, ast.ImportFrom)):
            nomes.extend(a.name for a in no.names)
            nomes.extend(a.asname for a in no.names if a.asname)
    return nomes


def _chamadas_de_limiar(arvore: ast.Module) -> list:
    """Toda chamada a `selecionar_limiar` e o `conjunto` que ela declara.

    Procurar nomes de arquivo pega quem ABRE a validação externa. Esta checagem pega
    o erro mais sutil e mais provável: chamar `selecionar_limiar` sem passar
    `conjunto`, herdando o default `"validacao"` da assinatura e gravando um JSON que
    ALEGA vir da validação externa. O limiar deste bloco é provisório e tem de sair
    rotulado como tal — a guarda de `carregar_modelo_ajustado` recusa o contrário.
    """
    achados = []
    for no in ast.walk(arvore):
        if not isinstance(no, ast.Call):
            continue
        alvo = no.func.id if isinstance(no.func, ast.Name) else getattr(no.func, "attr", None)
        if alvo != "selecionar_limiar":
            continue
        kw = {k.arg: k.value for k in no.keywords}
        valor = kw.get("conjunto")
        achados.append({
            "linha": no.lineno,
            "conjunto_declarado": (valor.value if isinstance(valor, ast.Constant)
                                   else None),
            "explicito": "conjunto" in kw,
        })
    return achados


def auditar_isolamento_validacao(raiz: Path) -> dict:
    """A validação externa foi tocada? Responde LENDO O CÓDIGO, não a memória.

    Item do critério de pronto, literal: «a validação externa não foi tocada —
    confirme por leitura do código, não por memória». Uma afirmação no JSON escrita à
    mão não é confirmação nenhuma; esta função parseia os três módulos que compõem o
    caminho de treino do B4.5 e devolve evidência, com o md5 de cada arquivo para que
    a evidência fique presa à versão exata do código que rodou.

    Levanta `AssertionError` se achar violação. É deliberado que a busca inteira
    aborte antes de gastar horas de GPU: um resultado obtido com a validação externa
    no laço de treino não tem conserto a posteriori — ele só pode ser jogado fora.
    """
    violacoes, arquivos = [], {}
    for rel in ARQUIVOS_AUDITADOS:
        caminho = raiz / rel
        fonte = caminho.read_text(encoding="utf-8")
        arvore = ast.parse(fonte)

        tokens = _constantes_de_codigo(arvore) + [
            n for n in _identificadores(arvore) if n]
        for proibido in NOMES_DE_DADOS_PROIBIDOS:
            if any(proibido in t for t in tokens):
                violacoes.append(f"{rel}: código referencia «{proibido}»")

        for chamada in _chamadas_de_limiar(arvore):
            if not chamada["explicito"]:
                violacoes.append(
                    f"{rel}:{chamada['linha']}: selecionar_limiar sem `conjunto=` "
                    "explícito — herdaria o default 'validacao'")
            elif chamada["conjunto_declarado"] != "early_stopping_interno":
                violacoes.append(
                    f"{rel}:{chamada['linha']}: selecionar_limiar com conjunto="
                    f"{chamada['conjunto_declarado']!r}, esperado "
                    "'early_stopping_interno'")

        arquivos[rel] = {
            "md5": hashlib.md5(caminho.read_bytes()).hexdigest(),
            "chamadas_selecionar_limiar": _chamadas_de_limiar(arvore),
        }

    assert not violacoes, "ISOLAMENTO DA VALIDAÇÃO VIOLADO:\n  " + "\n  ".join(violacoes)

    return {
        "verificado_por": ("leitura do código-fonte (AST) dos módulos do caminho de "
                           "treino, não por memória nem por afirmação escrita à mão"),
        "arquivos_auditados": arquivos,
        "nomes_de_dados_proibidos": list(NOMES_DE_DADOS_PROIBIDOS),
        "o_que_foi_checado": [
            # Os nomes NAO sao repetidos aqui em prosa: eles ja estao na chave
            # `nomes_de_dados_proibidos`, e escreve-los de novo faria este proprio
            # JSON conter os tokens que a guarda procura — a guarda acusaria a si
            # mesma. Referencia-los pela constante mantem uma fonte unica.
            "nenhum identificador nem string de código dos módulos auditados contém "
            "qualquer um dos nomes listados em `nomes_de_dados_proibidos`",
            "toda chamada a selecionar_limiar declara o conjunto EXPLICITAMENTE, e "
            "declara o conjunto interno — nunca herda o default da assinatura",
        ],
        "o_que_NAO_e_checado": (
            "docstrings e comentários são ignorados de propósito — é neles que o "
            "projeto DOCUMENTA a proibição, e uma guarda que dispara com a própria "
            "documentação seria ruído que ensina a ignorar o alarme"),
        "dados_efetivamente_lidos": [
            "data/espectrogramas/treino_30k.npy (a subamostra de 30k)",
            "data/espectrogramas/indice_treino_30k.csv",
            "data/processed/split_interno_cnn.csv (o split interno 27k/3k do B4.3)",
            "data/espectrogramas/normalizacao_cnn.json (estatísticas dos 27k)",
        ],
        "resultado": "a validação externa (22.226) e o teste (22.227) NÃO foram lidos",
    }


# =============================================================================
# 4. Uma configuração da grade
# =============================================================================
def rodar_configuracao(cfg: dict, raiz: Path, hp: dict, paciencia: int,
                       teto: int, estrito: bool = True) -> dict:
    """Treina uma configuração com *early stopping* de verdade e devolve o registro.

    As métricas finais são recalculadas COM OS PESOS DA MELHOR ÉPOCA recarregados, e
    comparadas com o que o histórico registrou naquela época. A checagem é barata e
    fecha a armadilha «salvar só os pesos e não a época» pelo lado que ninguém olha:
    salvar a época certa e os pesos ERRADOS produz exatamente os mesmos artefatos,
    passa em toda inspeção visual, e só aparece como um refit inexplicavelmente pior
    no B4.6.
    """
    rotulo = f"[{hp['id']}] "
    print(f"\n{'=' * 78}\n{rotulo}canais={hp['canais']} dropout={hp['p_drop']} "
          f"lr={hp['lr']:g} batch={hp['batch']} | teto {teto}, paciência {paciencia}\n"
          f"{'=' * 78}")

    t0 = time.perf_counter()
    ctx = preparar_treino(cfg, raiz, batch=hp["batch"], lr=hp["lr"],
                          canais=hp["canais"], p_drop=hp["p_drop"], estrito=estrito)
    historico, melhor = laco_de_treino(ctx, teto, paciencia=paciencia, rotulo=rotulo)
    tempo_total = time.perf_counter() - t0

    # ---- Métricas com os pesos da melhor época, recarregados ----------------
    modelo, criterio = ctx["modelo"], ctx["criterio"]
    modelo.load_state_dict(melhor["estado"])
    y_es, sc_es, loss_es = _inferir(modelo, ctx["dl_es"], criterio, ctx["dispositivo"])
    sel = selecionar_limiar(y_es, sc_es, criterio="f1_macro",
                            conjunto="early_stopping_interno")
    m = avaliar(y_es, sc_es, NOME, limiar=sel["limiar"])

    h_melhor = next(h for h in historico if h["epoca"] == melhor["epoca"])
    delta = abs(m["f1_macro"] - h_melhor["f1_macro_early_stopping"])
    coerencia = {
        "f1_macro_no_historico": h_melhor["f1_macro_early_stopping"],
        "f1_macro_com_os_pesos_recarregados": m["f1_macro"],
        "delta": float(delta),
        "tolerancia": 1e-6,
        "bate": bool(delta <= 1e-6),
        "o_que_prova": ("os pesos guardados são MESMO os da época registrada como "
                        "melhor — e não os da última época, nem os de uma época "
                        "vizinha"),
    }
    if not coerencia["bate"]:
        print(f"{rotulo}*** ATENÇÃO: pesos da melhor época não reproduzem o "
              f"histórico (delta {delta:.3e})")

    # Janela de estabilidade em torno do máximo (a armadilha do pico isolado).
    f1s = [h["f1_macro_early_stopping"] for h in historico]
    i = melhor["epoca"] - 1
    janela = f1s[max(0, i - RAIO_JANELA_ESTABILIDADE): i + RAIO_JANELA_ESTABILIDADE + 1]

    desc = modelo.descricao()
    registro = {
        "config_id": hp["id"],
        "fase": hp["fase"],
        "arquitetura": hp["arquitetura"],
        "n_blocos": desc["n_blocos"],
        # separador "-" e nao "," de proposito: este valor vai para uma coluna de
        # CSV, e virgula dentro de campo obriga aspas e quebra qualquer leitura
        # rapida da grade no terminal ou numa planilha.
        "canais": "-".join(str(c) for c in hp["canais"]),
        "n_parametros": desc["n_parametros"],
        "p_dropout": hp["p_drop"],
        "lr": hp["lr"],
        "batch": hp["batch"],
        "otimizador": "Adam",
        "weight_decay": 0.0,
        "scheduler": "none",
        "kernel": "3x3",
        "batchnorm": True,
        "paciencia": paciencia,
        "teto_epocas": teto,
        "melhor_epoca": melhor["epoca"],
        "epoca_em_que_parou": melhor["epoca_em_que_parou"],
        "parou_por_paciencia": melhor["parou_por_paciencia"],
        "atingiu_teto": melhor["atingiu_teto"],
        "f1_macro_3k_melhor_epoca": m["f1_macro"],
        "eer_3k_melhor_epoca": m["eer"],
        "roc_auc_3k_melhor_epoca": round(float(roc_auc_score(y_es, sc_es)), 4),
        "recall_bonafide_3k": m["recall_bonafide"],
        "recall_spoof_3k": m["recall_spoof"],
        "limiar_provisorio_3k": sel["limiar"],
        "loss_3k_melhor_epoca": loss_es,
        "loss_treino_melhor_epoca": h_melhor["loss_treino"],
        "f1_macro_janela_estabilidade": float(np.mean(janela)),
        "desvio_janela_estabilidade": float(np.std(janela)),
        "n_epocas_na_janela": len(janela),
        "tempo_total_s": round(tempo_total, 2),
        "tempo_por_epoca_s_mediana": round(
            float(np.median([h["tempo_s"] for h in historico])), 2),
        "semente": ctx["semente"],
        "pesos_batem_com_o_historico": coerencia["bate"],
    }

    print(f"{rotulo}melhor época {melhor['epoca']} (parou na {melhor['epoca_em_que_parou']}"
          f"{', TETO' if melhor['atingiu_teto'] else ''}) | "
          f"f1_macro {m['f1_macro']:.4f} | EER {m['eer']:.4f} | "
          f"estabilidade {registro['f1_macro_janela_estabilidade']:.4f} "
          f"± {registro['desvio_janela_estabilidade']:.4f} | {tempo_total / 60:.1f} min")

    saida = {
        "registro": registro,
        "historico": historico,
        "estado": melhor["estado"],
        "metricas": m,
        "selecao_limiar": sel,
        "coerencia_pesos": coerencia,
        "matriz_confusao": confusion_matrix(
            y_es, (sc_es >= sel["limiar"]).astype(int), labels=[0, 1]).tolist(),
        "descricao_arquitetura": desc,
        "ctx_resumo": {
            "resumo_split": ctx["resumo_split"],
            "norm": ctx["norm"],
            "n_por_classe": ctx["n_por_classe"],
            "pesos": [float(ctx["pesos"][0]), float(ctx["pesos"][1])],
            "semente": ctx["semente"],
            "determinismo_estrito": ctx["determinismo_estrito"],
            "limitacao_determinismo": ctx["limitacao_determinismo"],
            "dispositivo": str(ctx["dispositivo"]),
        },
        "hp": hp,
    }

    del ctx, modelo
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return saida


# =============================================================================
# 5. A seleção — regra declarada antes, aplicada depois
# =============================================================================
def selecionar_vencedora(registros: list) -> dict:
    """f1_macro nos 3k, com desempate por estabilidade dentro da margem declarada.

    O CRITÉRIO PRIMÁRIO É f1_macro NOS 3k, como o marco manda. O desempate por
    estabilidade não é um segundo critério travestido: ele só age entre configurações
    cuja diferença cabe dentro de `MARGEM_EMPATE_F1`, que é ruído de estimativa com
    300 bonafide. Chamar de vencedora a que tem o pico mais alto quando a diferença é
    de 0,002 seria escolher por sorteio e chamar de método.

    O EER é calculado e comparado em paralelo. Se ele apontar outra vencedora, isso
    NÃO é resolvido em silêncio — vai para o JSON e para o texto, porque é uma
    observação honesta sobre o que as duas métricas medem: f1_macro pontua uma
    decisão binária num limiar, EER pontua ordenamento em todos os limiares.
    """
    melhor_f1 = max(r["f1_macro_3k_melhor_epoca"] for r in registros)
    empatadas = [r for r in registros
                 if melhor_f1 - r["f1_macro_3k_melhor_epoca"] <= MARGEM_EMPATE_F1]
    vencedora = max(empatadas, key=lambda r: r["f1_macro_janela_estabilidade"])

    lider_f1 = max(registros, key=lambda r: r["f1_macro_3k_melhor_epoca"])
    lider_eer = min(registros, key=lambda r: r["eer_3k_melhor_epoca"])
    divergem = lider_eer["config_id"] != vencedora["config_id"]

    houve_desempate = vencedora["config_id"] != lider_f1["config_id"]
    return {
        "config_id_vencedora": vencedora["config_id"],
        "criterio": ("f1_macro no conjunto interno de 3.000, na melhor época de cada "
                     "configuração"),
        "por_que_f1_macro_e_nao_eer": (
            "a busca do RF e do SVM pontuou por EER porque ali só se escolhia o "
            "MODELO — o limiar viria depois, na validação. Aqui se escolhe "
            "ARQUITETURA e ÉPOCA ao mesmo tempo, e a parada precisa de um número que "
            "reflita a decisão binária que o modelo vai tomar. A assimetria é "
            "proposital e está registrada."),
        "margem_de_empate": MARGEM_EMPATE_F1,
        "por_que_a_margem": (
            "com 300 bonafide nos 3k, o erro padrão do recall da minoritária é da "
            "ordem de 0,023 e a diferença que isso produz em f1_macro fica no "
            "centésimo. 0,005 é conservador, e foi fixado ANTES de rodar a grade."),
        "n_configuracoes_empatadas": len(empatadas),
        "configuracoes_empatadas": [r["config_id"] for r in empatadas],
        "houve_desempate_por_estabilidade": houve_desempate,
        "lider_por_f1_macro": lider_f1["config_id"],
        "lider_por_eer": lider_eer["config_id"],
        "f1_macro_e_eer_discordam": divergem,
        "nota_divergencia": (
            f"O EER aponta {lider_eer['config_id']} "
            f"(EER {lider_eer['eer_3k_melhor_epoca']:.4f}) e o critério declarado "
            f"aponta {vencedora['config_id']} "
            f"(EER {vencedora['eer_3k_melhor_epoca']:.4f}). As duas métricas medem "
            "coisas diferentes — f1_macro pontua a decisão binária num limiar, EER "
            "pontua o ordenamento dos scores em todos os limiares — e a discordância "
            "está reportada em vez de escondida. O critério declarado antes da busca "
            "prevalece."
            if divergem else
            "f1_macro e EER apontam a MESMA configuração; não há discordância a "
            "reportar."),
        "nota_desempate": (
            f"{lider_f1['config_id']} teve o maior pico de f1_macro "
            f"({lider_f1['f1_macro_3k_melhor_epoca']:.4f}), mas a diferença para "
            f"{vencedora['config_id']} ({vencedora['f1_macro_3k_melhor_epoca']:.4f}) "
            f"cabe na margem de empate de {MARGEM_EMPATE_F1}, e a vencedora tem a "
            "curva mais estável em torno do máximo — que é o critério declarado para "
            "a armadilha do pico isolado numa curva ruidosa."
            if houve_desempate else
            "o maior f1_macro e a maior estabilidade apontam a mesma configuração; "
            "o desempate não precisou ser acionado."),
    }


# =============================================================================
# 6. Limitações que dependem da arquitetura escolhida
# =============================================================================
def limitacoes_da_arquitetura(desc: dict, melhor_epoca: int, n_treino_interno: int,
                              n_refit: int) -> list:
    """As limitações do Bloco 4, com os números da arquitetura que de fato venceu.

    A L2 do B4.4 estava escrita para 4 blocos («cada posição agrega 16 frames»). Se a
    grade eleger 3 blocos, o número vira 8 e a magnitude do erro de fronteira MUDA.
    Copiar o texto antigo seria descrever uma rede que não é a que rodou.
    """
    cadeia = " -> ".join(str(t) for t in desc["reducao_temporal"])
    por_posicao = desc["frames_originais_por_posicao_final"]
    n_final = desc["reducao_temporal"][-1]
    razao = n_refit / n_treino_interno

    return [
        {
            "id": "L1-batchnorm-ve-o-padding",
            "titulo": "O masked pooling protege a AGREGACAO, nao a NORMALIZACAO",
            "descricao": (
                "O BatchNorm2d de cada bloco calcula media e variancia por canal "
                "sobre o LOTE inteiro (N, F, T), com o padding incluido, e essas "
                "estatisticas normalizam tambem as posicoes validas."),
            "por_que_nao_invalida": (
                "A estatistica do BatchNorm e do LOTE, nao do exemplo: desloca e "
                "escala todos os exemplos do lote da mesma forma e NAO cria um canal "
                "por exemplo que codifique onde o audio termina. Alem disso o padding "
                "e um plato constante, produzido pelo mesmo pipeline para bonafide e "
                "para spoof, logo nao e informativo de classe."),
            "teste_da_regra_de_escopo": "NAO impede a validade do experimento principal",
            "destino": "limitacao no texto (B6.1)",
            "trabalho_futuro": "BatchNorm mascarado, ou LayerNorm/GroupNorm",
            "custo_para_corrigir_agora": "re-treino de todo o Bloco 4; nao cabe no cronograma",
            "nota_b45": ("herdada do B4.4 sem mudanca: a grade do B4.5 nao mexeu no "
                         "eixo BatchNorm sim/nao, entao a limitacao vale igual."),
        },
        {
            "id": "L2-semantica-de-teto-na-fronteira",
            "titulo": ("A reducao da mascara usa semantica de TETO, e a posicao de "
                       "fronteira e parcialmente padding"),
            "descricao": (
                f"A mascara e reduzida por F.max_pool1d, entao uma posicao reduzida "
                f"conta como VALIDA se QUALQUER frame original dela era valido. Na "
                f"arquitetura ESCOLHIDA, de {desc['n_blocos']} blocos, o eixo do "
                f"tempo percorre {cadeia} e cada posicao final agrega {por_posicao} "
                f"frames originais — logo a posicao de FRONTEIRA pode ser "
                f"majoritariamente padding e ainda entrar no masked pooling com "
                f"peso 1."),
            "magnitude": (
                f"No maximo UMA posicao entre as validas: 1 de {n_final} (~"
                f"{100 / n_final:.0f}%) num exemplo sem padding. Com MENOS blocos a "
                f"fronteira pesa MENOS, porque cada posicao agrega menos frames — os "
                f"numeros vem de CnnDeepfake.reducao_temporal(), nao de copia do "
                f"texto do B4.4."),
            "por_que_foi_escolhida": (
                "A alternativa e a semantica de PISO, que descartaria a fronteira do "
                "audio — informacao real, nao padding — e encurtaria mais o sinal "
                "justamente nos audios curtos. O teto e coerente com o ceil de "
                "n_frames_validos do Bloco 1."),
            "teste_da_regra_de_escopo": "NAO impede a validade do experimento principal",
            "destino": "convencao declarada no texto (B6.1)",
            "custo_para_corrigir_agora": "re-treinar e re-medir o Bloco 4; nao cabe no cronograma",
        },
        {
            "id": "L3-epocas-fixas-nao-sao-passos-fixos",
            "titulo": (f"{melhor_epoca} epocas em {n_treino_interno} exemplos NAO e "
                       f"{melhor_epoca} epocas em {n_refit} exemplos"),
            "descricao": (
                f"O refit do B4.6 vai treinar pelo MESMO NUMERO DE EPOCAS selecionado "
                f"aqui, mas sobre {n_refit} exemplos em vez de {n_treino_interno}. "
                f"Com o mesmo batch size, uma epoca no conjunto de refit tem "
                f"{(razao - 1) * 100:.1f}% MAIS passos de gradiente que uma epoca "
                f"aqui. «Mesmo numero de epocas» NAO e «mesmo numero de "
                f"atualizacoes»."),
            "alternativa_considerada": (
                f"Fixar os PASSOS: treinar por melhor_epoca x passos_por_epoca do "
                f"conjunto interno, o que igualaria o numero de atualizacoes."),
            "por_que_foi_descartada": (
                "Simplicidade de descricao. Epocas fixas e o que o orientador "
                "escreveu («treina a arquitetura escolhida por numero FIXO de "
                "epocas»), e e a forma como a literatura reporta. A alternativa usa um "
                "numero quebrado de epocas e complica a descricao sem mudar a "
                "conclusao."),
            "razao_de_tamanho": round(razao, 4),
            "percentual_a_mais_de_atualizacoes": round((razao - 1) * 100, 1),
            "teste_da_regra_de_escopo": "NAO impede a validade do experimento principal",
            "destino": "limitacao no texto (B6.1)",
            "redacao_para_o_texto": (
                f"O refit foi executado pelo mesmo numero de epocas selecionado na "
                f"fase de early stopping. Como o conjunto de refit e "
                f"{(razao - 1) * 100:.1f}% maior, isso corresponde a "
                f"{(razao - 1) * 100:.1f}% mais atualizacoes de gradiente; a "
                f"alternativa — fixar o numero de atualizacoes — foi considerada e "
                f"descartada por simplicidade de descricao, e a diferenca fica "
                f"registrada como limitacao."),
        },
    ]


# =============================================================================
# 7. Figuras
# =============================================================================
def plotar_curva_vencedora(historico: list, melhor_epoca: int, titulo: str,
                           destino: Path) -> None:
    """Dois painéis, e a linha vertical é a resposta à pergunta de banca.

    «Por que a CNN parou de treinar onde parou?» tem três partes, e as três precisam
    estar VISÍVEIS: (1) o critério foi f1_macro num conjunto INTERNO de 3k, nunca a
    validação externa — está no rótulo dos eixos e no título; (2) a paciência de 8 foi
    escolhida pela variância de estimativa com 300 bonafide — está na anotação; (3) a
    partir da melhor época a loss de treino continua caindo enquanto a dos 3k
    para/sobe — é o que os dois painéis mostram lado a lado, e é o que a linha
    vertical marca.
    """
    ep = [h["epoca"] for h in historico]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax1.plot(ep, [h["loss_treino"] for h in historico], "-o", ms=3,
             label="treino (27k)")
    ax1.plot(ep, [h["loss_early_stopping"] for h in historico], "-o", ms=3,
             label="early stopping interno (3k)")
    ax1.axvline(melhor_epoca, ls="--", c="gray", lw=1.2)
    ax1.set_xlabel("época")
    ax1.set_ylabel("loss ponderada")
    ax1.set_title("Loss — a de treino cai, a dos 3k não acompanha")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    f1s = [h["f1_macro_early_stopping"] for h in historico]
    ax2.plot(ep, f1s, "-o", ms=3, color="tab:green", label="f1_macro (3k)")
    ax2.plot(ep, [h["eer_early_stopping"] for h in historico], "-o", ms=3,
             color="tab:red", label="EER (3k)")
    ax2.axvline(melhor_epoca, ls="--", c="gray", lw=1.2)
    # A anotação vai para o lado da linha vertical que tiver espaço: com parada
    # antecipada a melhor época cai perto do FIM da curva, e ancorar sempre à
    # direita jogaria o texto para fora do eixo.
    a_direita = melhor_epoca < (ep[0] + ep[-1]) / 2
    ax2.annotate(f"melhor época: {melhor_epoca}\n(paciência {PACIENCIA}, teto {TETO_EPOCAS})",
                 xy=(melhor_epoca, max(f1s)),
                 xytext=(8 if a_direita else -8, -30), textcoords="offset points",
                 ha="left" if a_direita else "right",
                 fontsize=8, color="dimgray")
    ax2.axhline(0.47, ls=":", c="tab:orange", lw=1)
    ax2.annotate("colapso na majoritária (≈0,47)", xy=(ep[0], 0.47),
                 xytext=(2, 4), textcoords="offset points", fontsize=7,
                 color="tab:orange")
    ax2.set_xlabel("época")
    ax2.set_ylabel("métrica nos 3k internos")
    ax2.set_title("Critério de parada: f1_macro nos 3k INTERNOS")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.suptitle(titulo, fontsize=11)
    fig.tight_layout()
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, dpi=150)
    plt.close(fig)
    print(f"Figura salva em {destino}")


def plotar_grade(curvas: dict, registros: list, id_vencedora: str,
                 destino: Path) -> None:
    """Uma linha por configuração — a figura que mostra que a escolha não foi arbitrária."""
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    cores = plt.cm.tab10(np.linspace(0, 1, 10))

    for k, r in enumerate(registros):
        cid = r["config_id"]
        h = curvas[cid]
        ep = [x["epoca"] for x in h]
        f1 = [x["f1_macro_early_stopping"] for x in h]
        venceu = cid == id_vencedora
        ax.plot(ep, f1, lw=2.4 if venceu else 1.2,
                color=cores[k % 10], alpha=1.0 if venceu else 0.75,
                ls="-" if r["n_blocos"] == 4 else "--",
                label=f"{cid}{'  ← escolhida' if venceu else ''}"
                      f"  [melhor ép. {r['melhor_epoca']}, f1 "
                      f"{r['f1_macro_3k_melhor_epoca']:.4f}]",
                zorder=3 if venceu else 2)
        ax.plot([r["melhor_epoca"]], [r["f1_macro_3k_melhor_epoca"]],
                marker="*" if venceu else "o", ms=15 if venceu else 6,
                color=cores[k % 10], zorder=4 if venceu else 2)

    ax.set_xlabel("época")
    ax.set_ylabel("f1_macro no conjunto interno de early stopping (3.000)")
    ax.set_title("B4.5 — grade de configurações da CNN\n"
                 "linha cheia: 4 blocos · tracejada: 3 blocos · "
                 f"early stopping: paciência {PACIENCIA}, teto {TETO_EPOCAS}",
                 fontsize=10)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, dpi=150)
    plt.close(fig)
    print(f"Figura salva em {destino}")


# =============================================================================
# 8. Persistência incremental — a grade inteira sobrevive a uma queda no fim
# =============================================================================
def gravar_grade(registros: list, curvas: dict, dir_met: Path, sufixo: str) -> None:
    """Grava `busca_cnn.csv` e as curvas de todas as configurações.

    Chamada DEPOIS DE CADA configuração, não só no fim. A grade leva horas; perder
    tudo por uma falha no último gráfico seria perder a entrega obrigatória do marco
    («registre a grade inteira») por um motivo decorativo.
    """
    dir_met.mkdir(parents=True, exist_ok=True)

    with open(dir_met / f"busca_cnn{sufixo}.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(registros[0].keys()))
        w.writeheader()
        w.writerows(registros)

    # As curvas por época de TODAS as configuracoes, em formato longo: e o dado
    # que sustenta busca_cnn.png. Sem ele a figura nao seria reproduzivel a partir
    # de artefato nenhum, e uma figura irreproduzivel nao e evidencia.
    linhas = [{"config_id": cid, **h} for cid, hist in curvas.items() for h in hist]
    with open(dir_met / f"curvas_busca_cnn{sufixo}.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
        w.writeheader()
        w.writerows(linhas)


# =============================================================================
# 9. Execução
# =============================================================================
def executar(cfg: dict, raiz: Path, paciencia: int = PACIENCIA,
             teto: int = TETO_EPOCAS, so_fase1: bool = False,
             estrito: bool = True, sufixo: str = "") -> dict:
    dir_met = raiz / "results" / "metricas"
    dir_fig = raiz / "results" / "figuras"

    # A auditoria vem ANTES da GPU: um resultado obtido com a validacao externa no
    # laco nao tem conserto a posteriori, so descarte. Abortar aqui custa segundos.
    print("Auditando isolamento da validação externa (leitura do código)...")
    isolamento = auditar_isolamento_validacao(raiz)
    print(f"  OK — {isolamento['resultado']}")

    t_inicio = time.perf_counter()
    registros, curvas, saidas = [], {}, {}

    grade = montar_grade(so_fase1=so_fase1)
    print(f"\nFase 1 — {len(grade)} configurações (lr={LR_BASE:g}, dropout "
          f"{DROPOUTS[0]} e {DROPOUTS[1]})")
    for hp in grade:
        s = rodar_configuracao(cfg, raiz, hp, paciencia, teto, estrito=estrito)
        registros.append(s["registro"])
        curvas[hp["id"]] = s["historico"]
        saidas[hp["id"]] = s
        gravar_grade(registros, curvas, dir_met, sufixo)

    decisao_fase2 = None
    if not so_fase1:
        grade2, decisao_fase2 = configuracoes_da_fase_2(registros)
        print(f"\nFase 2 — dropout vencedor da fase 1: "
              f"{decisao_fase2['dropout_escolhido_para_a_fase_2']} "
              f"(médias por dropout: {decisao_fase2['f1_macro_medio_por_dropout']})")
        for hp in grade2:
            s = rodar_configuracao(cfg, raiz, hp, paciencia, teto, estrito=estrito)
            registros.append(s["registro"])
            curvas[hp["id"]] = s["historico"]
            saidas[hp["id"]] = s
            gravar_grade(registros, curvas, dir_met, sufixo)

    tempo_busca = time.perf_counter() - t_inicio

    # ---- Seleção ------------------------------------------------------------
    selecao = selecionar_vencedora(registros)
    vid = selecao["config_id_vencedora"]
    venc, reg = saidas[vid], next(r for r in registros if r["config_id"] == vid)
    print(f"\n{'=' * 78}\nVENCEDORA: {vid} — melhor época {reg['melhor_epoca']}, "
          f"f1_macro {reg['f1_macro_3k_melhor_epoca']:.4f}, "
          f"EER {reg['eer_3k_melhor_epoca']:.4f}\n{'=' * 78}")

    # ---- O JSON que o B4.6 consome ------------------------------------------
    resumo_split = venc["ctx_resumo"]["resumo_split"]
    n_treino_interno = resumo_split["n_treino_interno"]
    n_refit = n_treino_interno + resumo_split["n_early_stopping"]
    razao = n_refit / n_treino_interno

    limitacoes = limitacoes_da_arquitetura(
        venc["descricao_arquitetura"], reg["melhor_epoca"], n_treino_interno, n_refit)

    m = dict(venc["metricas"])
    m.update({
        # --- AS CHAVES QUE O B4.6 LÊ. Tudo o mais neste JSON justifica estas. ---
        "melhor_epoca": reg["melhor_epoca"],
        "criterio_parada": CRITERIO_PARADA,
        "paciencia": paciencia,
        "teto_epocas": teto,
        "epoca_em_que_parou": reg["epoca_em_que_parou"],
        "f1_macro_3k_na_melhor_epoca": reg["f1_macro_3k_melhor_epoca"],
        "eer_3k_na_melhor_epoca": reg["eer_3k_melhor_epoca"],
        "nota_epocas_refit": (
            f"o refit do B4.6 treina por {reg['melhor_epoca']} EPOCAS FIXAS nos "
            f"{n_refit} da subamostra, sem early stopping e sem tocar na validacao "
            f"externa. A arquitetura e a da chave `arquitetura` deste JSON "
            f"(canais {reg['canais'].replace('-', ', ')}, dropout "
            f"{reg['p_dropout']}), com lr "
            f"{reg['lr']:g}, batch {reg['batch']}, Adam, e a mesma loss ponderada "
            f"derivada das contagens do conjunto de refit."),

        "marco": "B4.5 — CNN definida (arquitetura final e melhor epoca)",
        "braco": "principal",
        "conjunto_avaliado": "early_stopping_interno (3.000 da subamostra de 30k)",
        "parou_de_verdade": bool(reg["parou_por_paciencia"]),
        "atingiu_teto": bool(reg["atingiu_teto"]),
        "nota_parada": (
            f"A rede parou por PACIENCIA na epoca {reg['epoca_em_que_parou']}: "
            f"{paciencia} epocas sem melhora do f1_macro nos 3k depois da epoca "
            f"{reg['melhor_epoca']}. Diferente do B4.4, onde a melhor epoca foi a "
            f"ultima e a rede nao parou — acabou o orcamento."
            if reg["parou_por_paciencia"] else
            f"ATENCAO: a configuracao vencedora NAO parou por paciencia — atingiu o "
            f"TETO de {teto} epocas com a melhor epoca em {reg['melhor_epoca']}. A "
            f"melhor epoca e portanto um LIMITE INFERIOR: a curva ainda subia quando "
            f"o orcamento acabou. Isso esta reportado em vez de escondido, e e "
            f"limitacao a declarar no texto."),
        "selecao_limiar": venc["selecao_limiar"],
        "limiar": venc["selecao_limiar"]["limiar"],
        "nota_limiar_provisorio": (
            "LIMIAR PROVISORIO. Selecionado nos 3k de early stopping interno, nao na "
            "validacao externa — por isso conjunto='early_stopping_interno'. O limiar "
            "do protocolo nasce no B4.7, na validacao externa."),
        "nota_escala_score": (
            "score = softmax(logits)[:, 1] = P(spoof), em [0,1] — comparavel ao "
            "predict_proba do RF e de escala DIFERENTE do decision_function do SVM."),
        "nota_comparabilidade": (
            "ESTE f1_macro NAO E COMPARAVEL aos 0,7225 do RF e 0,7987 do SVM, que "
            "saem da VALIDACAO EXTERNA (22.226). Este sai dos 3k internos e e "
            "otimista por QUATRO motivos de construcao, um a mais que no B4.4: "
            "(1) CONJUNTO — os 3k vem da mesma subamostra de 30k que gerou o treino; "
            "(2) LIMIAR — foi selecionado nos MESMOS 3k em que a metrica e reportada; "
            "(3) EPOCA — a melhor epoca foi escolhida pelo f1_macro nos MESMOS 3k, "
            "entao o valor e um MAXIMO sobre as epocas rodadas; (4) ARQUITETURA — "
            "NOVO NESTE MARCO: a propria configuracao vencedora foi escolhida por "
            "f1_macro nesses mesmos 3k, entao o valor publicado e tambem um maximo "
            "sobre a grade. Com 300 bonafide a variancia da estimativa e grande por "
            "cima disso tudo. Os numeros comparaveis saem no B4.7. Nada aqui autoriza "
            "a frase «a CNN superou o SVM»."),

        "selecao_arquitetura": selecao,
        "grade": {
            "n_configuracoes": len(registros),
            "desenho": ("grade curta e deliberada em duas fases: 2 arquiteturas x 2 "
                        "dropouts com lr=1e-3, mais as 2 arquiteturas com lr=3e-4 no "
                        "dropout vencedor da fase 1"),
            "por_que_nao_random_search": (
                "regra de escopo do projeto: «busca de hiperparametros ficou cara -> "
                "reduz o espaco de busca». Um fit de RF custa segundos e comportou "
                "RandomizedSearchCV; um fit desta CNN custa entre 6 e 25 minutos. "
                "Random Search amplo estouraria o marco."),
            "eixos": EIXOS,
            "eixos_nao_explorados": EIXOS_NAO_EXPLORADOS,
            "decisao_da_fase_2": decisao_fase2,
            "arquivo": f"results/metricas/busca_cnn{sufixo}.csv",
            "arquivo_curvas": f"results/metricas/curvas_busca_cnn{sufixo}.csv",
            "so_fase1": so_fase1,
            "tempo_busca_s": round(tempo_busca, 2),
            "tempo_busca_min": round(tempo_busca / 60, 1),
        },

        "arquitetura": venc["descricao_arquitetura"],
        "hiperparametros": {
            "epocas_rodadas": reg["epoca_em_que_parou"],
            "teto_epocas": teto,
            "paciencia": paciencia,
            "batch": reg["batch"],
            "otimizador": "Adam",
            "lr": reg["lr"],
            "scheduler": None,
            "weight_decay": 0.0,
            "p_dropout": reg["p_dropout"],
            "canais": [int(c) for c in reg["canais"].split("-")],
            "amp_autocast": False,
            "nota_amp": ("AMP nao usado de proposito: ganho desprezivel numa rede "
                         "pequena e introduz nao-determinismo numerico."),
        },
        "loss": {
            "funcao": "CrossEntropyLoss",
            "n_saidas": 2,
            "formula_pesos": "w_c = n / (k * n_c)  (mesma logica do class_weight='balanced')",
            "contagens_usadas": {"bonafide": venc["ctx_resumo"]["n_por_classe"][0],
                                 "spoof": venc["ctx_resumo"]["n_por_classe"][1]},
            "peso_bonafide": round(venc["ctx_resumo"]["pesos"][0], 4),
            "peso_spoof": round(venc["ctx_resumo"]["pesos"][1], 4),
            "razao": round(venc["ctx_resumo"]["pesos"][0] / venc["ctx_resumo"]["pesos"][1], 4),
            "reamostragem": "nenhuma — sem undersampling e sem oversampling (P7)",
        },
        "split_interno": resumo_split,
        "isolamento_validacao": isolamento,
        "coerencia_pesos_melhor_epoca": venc["coerencia_pesos"],
        "normalizacao": {
            "fonte": "data/espectrogramas/normalizacao_cnn.json (B4.3)",
            "eixo": venc["ctx_resumo"]["norm"]["eixo"],
            "faixas_com_desvio_clampado": venc["ctx_resumo"]["norm"]["faixas_com_desvio_clampado"],
        },
        "semente": venc["ctx_resumo"]["semente"],
        "determinismo": {
            "estrito": venc["ctx_resumo"]["determinismo_estrito"],
            "limitacao": venc["ctx_resumo"]["limitacao_determinismo"],
            "num_workers": 0,
            "nota_grade": ("toda configuracao da grade comeca do MESMO estado de RNG: "
                           "preparar_treino re-fixa a semente depois da sondagem de "
                           "determinismo. Duas configuracoes diferem por "
                           "hiperparametro, e por mais nada."),
        },
        "n_sementes": 1,
        "nota_sementes": (
            "resultado principal em semente 42. A analise de 3 sementes na CNN esta, "
            "por decisao registrada, FORA do caminho critico — limitacao "
            "computacional declarada, e o refit nao foi atrasado por ela."),
        "limitacoes_registradas": limitacoes,
        "matriz_confusao": venc["matriz_confusao"],
        "roc_auc_early_stopping": reg["roc_auc_3k_melhor_epoca"],
        "loss_final_early_stopping": reg["loss_3k_melhor_epoca"],
        "tempo_treino_vencedora_s": reg["tempo_total_s"],
        "tempo_por_epoca_s_mediana": reg["tempo_por_epoca_s_mediana"],
    })

    m["ambiente"] = ambiente(n_jobs_inferencia=1)
    m["ambiente"].update({
        "torch": torch.__version__,
        "cuda_disponivel": bool(torch.cuda.is_available()),
        "cuda_versao": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "dispositivo_treino": venc["ctx_resumo"]["dispositivo"],
    })

    dir_esp = raiz / "data" / "espectrogramas"
    m["hash_md5_split_interno_csv"] = hashlib.md5(
        (raiz / "data" / "processed" / "split_interno_cnn.csv").read_bytes()).hexdigest()
    m["hash_md5_indice_treino_csv"] = hashlib.md5(
        (dir_esp / "indice_treino_30k.csv").read_bytes()).hexdigest()
    m["hash_md5_normalizacao_json"] = hashlib.md5(
        (dir_esp / "normalizacao_cnn.json").read_bytes()).hexdigest()

    # ---- Artefatos ----------------------------------------------------------
    gravar_grade(registros, curvas, dir_met, sufixo)

    with open(dir_met / f"curva_treino_{NOME}{sufixo}.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(venc["historico"][0].keys()))
        w.writeheader()
        w.writerows(venc["historico"])

    plotar_curva_vencedora(
        venc["historico"], reg["melhor_epoca"],
        f"B4.5 — CNN definida ({vid}) · melhor época {reg['melhor_epoca']} "
        f"de {reg['epoca_em_que_parou']} rodadas",
        dir_fig / f"curva_treino_{NOME}{sufixo}.png")
    plotar_grade(curvas, registros, vid, dir_fig / f"busca_cnn{sufixo}.png")

    (raiz / "models").mkdir(exist_ok=True)
    torch.save({"state_dict": venc["estado"],
                "arquitetura": venc["descricao_arquitetura"],
                "epoca": reg["melhor_epoca"],
                "config_id": vid,
                "hiperparametros": m["hiperparametros"],
                "semente": venc["ctx_resumo"]["semente"]},
               raiz / "models" / f"{NOME}{sufixo}.pt")

    with open(dir_met / f"{NOME}{sufixo}.json", "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)

    print(f"\nSalvo: models/{NOME}{sufixo}.pt, results/metricas/{NOME}{sufixo}.json, "
          f"busca_cnn{sufixo}.csv, curva_treino_{NOME}{sufixo}.csv")
    print(f"\n>>> MELHOR EPOCA = {reg['melhor_epoca']} — é este o número que o B4.6 "
          f"usa como FIXO nos {n_refit} ({(razao - 1) * 100:.1f}% mais atualizações "
          f"por época; limitação L3 registrada)")

    falhas = [r["config_id"] for r in registros
              if not r["pesos_batem_com_o_historico"]]
    if falhas:
        print(f"\n*** FALHA: pesos não batem com o histórico em: {falhas}. Os "
              "artefatos foram gravados para inspeção, mas o marco NÃO está fechado.")
    m["_falhas"] = falhas
    return m


def main() -> dict:
    from ..utils.config import carregar_config

    p = argparse.ArgumentParser(description="B4.5 — CNN definida")
    p.add_argument("--paciencia", type=int, default=PACIENCIA)
    p.add_argument("--teto", type=int, default=TETO_EPOCAS)
    p.add_argument("--so-fase1", action="store_true",
                   help="modo «se atrasar»: 4 configuracoes em vez de 6")
    p.add_argument("--fumaca", action="store_true",
                   help="ensaio end-to-end de 2 epocas, artefatos com sufixo _FUMACA")
    p.add_argument("--nao-estrito", action="store_true")
    a = p.parse_args()

    paciencia, teto, sufixo = a.paciencia, a.teto, ""
    if a.fumaca:
        paciencia, teto, sufixo = 1, 2, "_FUMACA"
        print("MODO FUMAÇA: 2 épocas por configuração, artefatos com sufixo _FUMACA. "
              "NÃO é resultado de marco.")

    return executar(carregar_config(RAIZ), RAIZ, paciencia=paciencia, teto=teto,
                    so_fase1=a.so_fase1, estrito=not a.nao_estrito, sufixo=sufixo)


if __name__ == "__main__":
    # Windows usa `spawn`: todo codigo de execucao fica sob este guard.
    resultado = main()
    # Sai com codigo 1 se os pesos salvos nao reproduzirem a epoca registrada. Os
    # artefatos JA foram gravados — a evidencia da falha e mais util que a ausencia
    # dela —, mas o marco nao pode ser dado por fechado. Mesma postura da
    # guarda de reproducao do Bloco 3.
    if resultado.get("_falhas"):
        raise SystemExit(1)
