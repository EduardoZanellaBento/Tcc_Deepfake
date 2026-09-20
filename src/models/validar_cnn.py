"""
B4.7 — Validação externa da CNN: limiar do protocolo, métricas, EER e tempos
=============================================================================

O QUE ESTE MARCO PRODUZ, E POR QUE ELE É O QUE FALTAVA:
    o LIMIAR OFICIAL DA CNN. Depois daqui a CNN tem exatamente o mesmo tipo de
    artefato que RF e SVM têm — um modelo persistido e um JSON com
    `selecao_limiar.conjunto == "validacao"` —, e é isso que torna a tabela
    comparativa do B5.2 uma tabela, e não uma colagem. É este limiar que o B5.1
    aplica no teste lacrado.

NADA DE AVALIAÇÃO É REIMPLEMENTADO AQUI:
    `selecionar_limiar`, `avaliar`, `calcular_eer` e `plotar_matriz_confusao`
    vêm de src/models/avaliacao.py — as MESMAS funções que produziram os números
    de RF e SVM. A régua é uma só; uma divergência silenciosa entre cópias
    invalidaria a comparação, que é a pergunta central do trabalho. Este módulo
    só produz os scores, chama a régua e registra.

    A comparação estatística formal (bootstrap pareado CNN × SVM × RF) é B5.2.
    Aqui só se produzem os números.

AS TRÊS COISAS QUE ESTE MÓDULO TEM DE ACERTAR (e como cada uma é cobrada):

    1. A NORMALIZAÇÃO É A DOS 30k, NÃO A DOS 27k. A CNN final é a do refit; usar
       `normalizacao_cnn.json` (o artefato do early stopping) normalizaria a rede
       final por estatísticas de um conjunto que não é o seu treino. Cobrado por
       `assert norm["estagio"] == "30k_refit"`, e o md5 do artefato vai para o
       JSON de saída.

    2. SCORE E RÓTULO ALINHADOS. `DataLoader(shuffle=False)` é necessário e não
       é suficiente: o que prova o alinhamento é que os rótulos SAÍDOS DO LOADER,
       na ordem em que saíram, são idênticos aos de `indice_validacao.csv`, e que
       reprocessar exemplos isolados reproduz os scores daquelas posições. As
       duas conferências estão em `conferir_alinhamento`. Sem elas, um
       desalinhamento daria EER ~0,5 e o bug seria procurado na rede.

    3. O TEMPO EM GPU TEM DE SER SINCRONIZADO. CUDA é assíncrono: sem
       `torch.cuda.synchronize()`, `perf_counter` mede o ENFILEIRAMENTO do
       kernel, não a execução, e a CNN aparece ordens de grandeza mais rápida do
       que é — uma afirmação FALSA no TC II. A sincronização mora DENTRO do
       callable passado a `medir_tempos` (ver `_preditor`), e `tempo.py` fica
       intocado: ele é compartilhado com RF e SVM e não deve conhecer CUDA. O
       JSON registra, em `demonstracao_sem_sincronizacao`, o número falso que a
       medida sem sincronização teria produzido — para que a armadilha fique
       documentada com evidência, e não só com prosa.

       E HÁ UMA SEGUNDA ARMADILHA DE CUDA, QUE APONTA PARA O LADO CONTRÁRIO: uma
       GPU ociosa entra em baixo consumo, e os 3 descartes do protocolo (~3 ms de
       trabalho) não a trazem de volta ao clock de operação. Medida assim, a
       latência sai ~10x INFLADA — 8,83 ms contra 0,82 ms nesta máquina. A
       primeira execução deste marco publicou justamente o número frio. Corrigido
       por um aquecimento explícito antes do cronômetro, com o número frio
       publicado ao lado (`aquecimento_do_dispositivo`). Ver
       AQUECIMENTO_DISPOSITIVO_S: as duas armadilhas mentem, uma para cada lado,
       e nenhuma das duas se resolve lendo o código — só medindo.

O QUE ESTE MÓDULO NÃO FAZ:
    não treina, não toca no conjunto de teste e não escolhe hiperparâmetro
    nenhum. Tudo que decide veio do B4.5 (arquitetura, época) e do B4.6 (pesos).

Rode a partir da raiz:  python -m src.models.validar_cnn
"""

import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score

from .avaliacao import avaliar, plotar_matriz_confusao, selecionar_limiar
# A auditoria de isolamento reusa os mesmos analisadores de AST do B4.5/B4.6 —
# uma segunda implementação de «o que conta como violação» é a divergência que o
# registro de isolamento existe para tornar impossível.
from .definir_cnn import (NOMES_DE_DADOS_PROIBIDOS, _chamadas_de_limiar,
                          _constantes_de_codigo, _identificadores)
from .modelos_ajustados import (MODELOS_PRINCIPAIS, carregar_cnn_persistida,
                                fixar_precisao_fp32_cnn, hashes_congelados,
                                scores_de)
from .tempo import ambiente, medir_tempos
# O Dataset é IMPORTADO do módulo de treino, não reescrito: a normalização em
# tempo de carga e a construção da máscara a partir de `n_frames_validos` têm de
# ser exatamente as do treino. Importar não altera o md5 auditado em
# refit_cnn.json -> isolamento_validacao; reescrever seria a divergência que
# aquele registro existe para tornar impossível.
from .treinar_cnn import EspectrogramaDataset
from ..utils.seeds import fixar_seeds_torch

RAIZ = Path(__file__).resolve().parents[2]
NOME = "cnn_final_principal"

INDICE = "data/espectrogramas/indice_validacao.csv"
MEMMAP = "data/espectrogramas/validacao.npy"
NORMALIZACAO = "data/espectrogramas/normalizacao_cnn_30k.json"
ESTAGIO_EXIGIDO = "30k_refit"
N_VALIDACAO_ESPERADO = 22226

# Lote da inferência das métricas. Não afeta o resultado (em `eval()` o
# BatchNorm usa as estatísticas do treino, não as do lote), só a memória.
LOTE_INFERENCIA = 256

# Lote do THROUGHPUT — o número declarado no JSON. 128 é o mesmo batch do treino.
LOTE_THROUGHPUT = 128

# Segundos de aquecimento DO DISPOSITIVO antes de cada cenário de tempo.
#
# POR QUE ISTO EXISTE, E POR QUE NÃO É «AFROUXAR A MEDIDA» (medido em 20/09/2026):
#     `descartar_aquecimento: 3` do config descarta três ITERAÇÕES. Para o RF e o
#     SVM isso basta. Para a CNN em GPU, não: a iteração dura ~0,9 ms, então as
#     três descartadas somam ~3 ms — e uma GPU que passou o minuto anterior
#     ociosa (enquanto `selecionar_limiar` varre 9.322 candidatos na CPU) está em
#     estado de baixo consumo e não volta ao clock de operação em 3 ms.
#
#     Medido nesta máquina, com o MESMO protocolo de 3 descartes + 10 medidas:
#         GPU quente ................ mediana 0,82 ms (min 0,80 / max 1,08)
#         GPU após 60 s ociosa ...... mediana 8,83 ms (min 1,03 / max 18,75)
#         após 50 aquecimentos ...... mediana 1,68 ms
#
#     A primeira execução deste marco publicou 8,11 ms — o número FRIO. Ele
#     superestima a latência da CNN em GPU em cerca de 10x, e a distorção aponta
#     na direção OPOSTA à da armadilha do `synchronize()`: uma faz a GPU parecer
#     rápida demais, a outra, lenta demais. As duas são erradas, e as duas ficam
#     documentadas com evidência no artefato (`aquecimento_do_dispositivo` e
#     `demonstracao_sem_sincronizacao`).
#
#     O protocolo compartilhado NÃO foi alterado: `tempo.py` continua descartando
#     exatamente as 3 iterações que o config manda, igual para os três modelos. O
#     que se acrescenta é levar o dispositivo ao regime de operação ANTES de
#     entregar o callable ao cronômetro — e o número frio é medido e publicado ao
#     lado, para que a correção seja verificável em vez de confiável.
AQUECIMENTO_DISPOSITIVO_S = 2.0

# Quantos áudios o throughput em CPU cobre. NÃO são os 22.226, e a razão foi
# MEDIDA antes de escolher o número: num piloto, a CNN em CPU com 1 thread levou
# ~19 ms por áudio, de modo que uma passada completa custaria ~430 s e o
# protocolo manda fazer 13 (10 + 3 de aquecimento) — cerca de 1h33 de CPU girando
# para refinar a terceira casa de um número que é uma TAXA e já está estável em n
# muito menor. (O valor do piloto é o que justificou a ESCOLHA da constante; o
# número que vai para o artefato é o medido na execução, e o campo `cobertura` o
# deriva da medição, não daqui.) A comparabilidade é preservada medindo a GPU
# TAMBÉM sobre esta mesma amostra (campo `comparacao_gpu_cpu`), de modo que a
# razão GPU/CPU sai de trabalho idêntico.
N_THROUGHPUT_CPU = 1280          # 10 lotes de 128


# =============================================================================
# 1. Dados da validação externa
# =============================================================================
def dataset_validacao(raiz: Path) -> tuple[EspectrogramaDataset, pd.DataFrame, dict]:
    """O Dataset da validação (22.226), normalizado pelas estatísticas DOS 30k.

    A guarda do `estagio` é a que impede a primeira armadilha do marco: usar
    `normalizacao_cnn.json`, que é o artefato dos 27k do early stopping. A CNN
    final é a do refit, e as estatísticas têm de ser as do conjunto em que ela
    foi treinada.

    Returns:
        (dataset, índice lido do CSV, dict da normalização).
    """
    indice = pd.read_csv(raiz / INDICE)
    if len(indice) != N_VALIDACAO_ESPERADO:
        raise ValueError(f"{INDICE} tem {len(indice)} linhas, esperado "
                         f"{N_VALIDACAO_ESPERADO}.")

    with open(raiz / NORMALIZACAO, encoding="utf-8") as f:
        norm = json.load(f)
    if norm["estagio"] != ESTAGIO_EXIGIDO:
        raise ValueError(
            f"{NORMALIZACAO} está no estágio {norm['estagio']!r}, esperado "
            f"{ESTAGIO_EXIGIDO!r}. A CNN avaliada aqui é a do REFIT — normalizar "
            "com as estatísticas dos 27k usaria um conjunto que não é o treino "
            "dela.")

    media = np.array(norm["media_por_mel"], dtype=np.float64)
    desvio = np.array(norm["desvio_por_mel"], dtype=np.float64)
    assert media.shape == desvio.shape == (128,), "normalizacao nao tem 128 faixas"
    assert float(desvio.min()) > 0, "desvio por faixa Mel com zero -> divisao por zero"

    ds = EspectrogramaDataset(raiz / MEMMAP, indice, media, desvio)
    assert len(ds) == len(indice)
    return ds, indice, norm


def dataset_para_arquivos(raiz: Path, arquivos) -> EspectrogramaDataset:
    """O mesmo Dataset, porém REORDENADO para casar com uma lista de `arquivo`.

    POR QUE ISTO É NECESSÁRIO — e por que reordenar é seguro:
        os diagnósticos por ataque e por codec montam a validação a partir de
        `features.csv` (via `carregar_dados_split`), cuja ORDEM DE LINHAS não é a
        de `indice_validacao.csv`. Passar um Dataset na ordem do índice, para um
        script que fatia por `validacao["codec"]`, produziria um cruzamento
        silenciosamente errado entre score e metadado — o mesmo tipo de bug que o
        Passo 1 deste marco existe para impedir, só que uma camada acima.

        Reordenar não mexe em dado nenhum: o `EspectrogramaDataset` lê a linha do
        memmap indicada pela coluna `linha` de cada registro, então a ordem das
        LINHAS DO ÍNDICE é livre — o que tem de estar certo é o par
        (arquivo -> linha), e ele vem do próprio índice congelado do B4.2.

    Args:
        arquivos: sequência de nomes de arquivo, na ordem desejada. Todos têm de
            pertencer à validação; um que não pertença aborta em vez de sumir
            num merge silencioso.
    """
    _, indice, _ = dataset_validacao(raiz)
    with open(raiz / NORMALIZACAO, encoding="utf-8") as f:
        norm = json.load(f)
    media = np.array(norm["media_por_mel"], dtype=np.float64)
    desvio = np.array(norm["desvio_por_mel"], dtype=np.float64)

    pedido = pd.DataFrame({"arquivo": list(arquivos)})
    faltando = set(pedido["arquivo"]) - set(indice["arquivo"])
    if faltando:
        raise ValueError(
            f"{len(faltando)} arquivo(s) pedido(s) não estão em {INDICE} "
            f"(ex.: {sorted(faltando)[:3]}). A CNN só pontua o que foi "
            "espectrogramado no B4.2.")
    reordenado = pedido.merge(indice, on="arquivo", how="left",
                              validate="one_to_one")
    assert list(reordenado["arquivo"]) == list(pedido["arquivo"]), (
        "o merge alterou a ordem pedida")
    return EspectrogramaDataset(raiz / MEMMAP, reordenado, media, desvio)


# =============================================================================
# 2. Scores — a passada que produz o número do marco
# =============================================================================
def scores_e_rotulos(modelo, ds, dispositivo, lote: int = LOTE_INFERENCIA) -> dict:
    """Uma passada `shuffle=False` que devolve scores, rótulos e máscaras JUNTOS.

    POR QUE COLETAR O RÓTULO AQUI, se ele já está no índice: porque é justamente
    a correspondência entre os dois que precisa ser provada. Um `y_true` lido do
    CSV e um `scores` produzido pelo loader são dois vetores que PARECEM
    alinhados; coletar o rótulo NA MESMA ORDEM em que o score saiu, e comparar
    com o CSV, é o que transforma a suposição em conferência.

    Os `n_frames_validos` coletados vêm da soma da máscara que a rede de fato
    recebeu — não do CSV —, pelo mesmo motivo.
    """
    modelo = modelo.to(dispositivo)
    modelo.eval()
    dl = torch.utils.data.DataLoader(ds, batch_size=lote, shuffle=False,
                                     num_workers=0)
    scores, rotulos, n_valid = [], [], []
    with torch.no_grad():
        for x, mascara, y in dl:
            logits = modelo(x.to(dispositivo), mascara.to(dispositivo))
            # softmax[:, 1] = P(spoof). Monotonicamente equivalente a
            # logits[:,1] - logits[:,0], logo o limiar selecionado é o MESMO
            # ponto de corte nas duas parametrizações.
            scores.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            rotulos.append(y.numpy())
            n_valid.append(mascara.sum(dim=1).numpy())
    return {
        "scores": np.concatenate(scores).astype(np.float64),
        "y": np.concatenate(rotulos).astype(int),
        "n_frames_validos": np.concatenate(n_valid).astype(int),
    }


def conferir_alinhamento(saida: dict, indice: pd.DataFrame, modelo, ds,
                         dispositivo) -> dict:
    """As três conferências do Passo 1, mais a que prova que o caminho é comum.

    1. `len(scores) == 22.226`.
    2. Os rótulos que saíram do loader, NA ORDEM EM QUE SAÍRAM, são idênticos aos
       de `indice_validacao.csv` — e as somas das máscaras, idênticas aos
       `n_frames_validos` do mesmo índice. Isso cobre o `shuffle=False`: um
       embaralhamento reordenaria os dois vetores e a comparação quebraria.
    3. CONFERÊNCIA POSICIONAL: sete exemplos, reprocessados ISOLADAMENTE (lote de
       1), reproduzem o score daquelas posições. Uma permutação que por azar
       preservasse a sequência de rótulos (permutar DENTRO de uma classe não
       muda o vetor de rótulos!) passaria pela conferência 2 e morreria aqui. É
       esta a conferência que realmente fecha o buraco, e por isso ela existe.
    4. `scores_de(chave='cnn')` — o caminho que o B5.1 e os diagnósticos vão usar
       — reproduz este mesmo vetor. Se as duas implementações divergirem, o
       limiar publicado aqui não será o limiar aplicado lá.

    `eval()` é o que torna (3) quase exato: com BatchNorm em modo de avaliação as
    estatísticas são as acumuladas no treino, então o resultado de um exemplo não
    depende de com quem ele divide o lote. Sobra a reordenação de somas em ponto
    flutuante entre tamanhos de lote, da ordem de 1e-06 com FP32.

    POR QUE O CRITÉRIO NÃO É UMA TOLERÂNCIA FIXA: um limiar absoluto escolhido a
    olho («1e-05») responde à pergunta errada — ele pergunta «o desvio é
    pequeno?» quando a pergunta é «o desvio é de ARREDONDAMENTO ou de
    POSIÇÃO?». O critério aqui é comparativo: mede-se também o desvio contra a
    posição VIZINHA, que é o que um deslocamento de uma casa produziria, e
    exige-se que o desvio real seja pelo menos 100x menor. Assim o teste continua
    válido se a precisão do hardware mudar, e continua falhando se a ordem mudar.
    (Foi essa distinção que, na primeira execução, mostrou que um desvio de
    1,9e-05 era TF32 do cuDNN e não desalinhamento — ver
    `fixar_precisao_fp32_cnn`.)
    """
    scores, y = saida["scores"], saida["y"]
    if len(scores) != N_VALIDACAO_ESPERADO:
        raise ValueError(f"{len(scores)} scores, esperado {N_VALIDACAO_ESPERADO}")

    y_indice = indice["classe_binaria"].to_numpy().astype(int)
    if not np.array_equal(y, y_indice):
        raise ValueError(
            f"os rótulos saídos do DataLoader divergem de {INDICE} em "
            f"{int((y != y_indice).sum())} posições — score e rótulo estão "
            "DESALINHADOS. Confira shuffle=False antes de olhar para a rede: um "
            "EER ~0,5 vindo daqui parece problema do modelo e não é.")

    nv_indice = indice["n_frames_validos"].to_numpy().astype(int)
    if not np.array_equal(saida["n_frames_validos"], nv_indice):
        raise ValueError("as máscaras recebidas pela rede divergem dos "
                         f"n_frames_validos de {INDICE}")

    # ---- conferência posicional, por reprocessamento isolado ----------------
    posicoes = [0, 1, len(ds) // 4, len(ds) // 2, 3 * len(ds) // 4,
                len(ds) - 2, len(ds) - 1]
    desvios, controles = [], []
    modelo.eval()
    with torch.no_grad():
        for i in posicoes:
            x, mascara, yi = ds[i]
            logit = modelo(x[None].to(dispositivo), mascara[None].to(dispositivo))
            s = float(torch.softmax(logit, dim=1)[0, 1])
            desvios.append(abs(s - float(scores[i])))
            # CONTROLE: o desvio que um deslocamento de UMA casa produziria.
            controles.append(abs(s - float(scores[i + 1 if i + 1 < len(scores)
                                                  else i - 1])))
            if int(yi) != int(y_indice[i]):
                raise ValueError(f"rótulo do Dataset na posição {i} diverge do índice")
    maior_desvio = float(max(desvios))
    controle = float(np.median(controles))
    margem = controle / maior_desvio if maior_desvio > 0 else float("inf")
    if margem < 100:
        raise ValueError(
            f"reprocessar exemplos isolados NÃO reproduz os scores das mesmas "
            f"posições: desvio máximo {maior_desvio:.3e} contra um controle de "
            f"deslocamento de {controle:.3e} (margem {margem:.1f}x, exigidas "
            "100x). O desvio tem a ordem de grandeza de um erro de POSIÇÃO, não "
            "de arredondamento — o vetor de scores não corresponde, posição a "
            "posição, às linhas do índice.")

    # ---- o caminho compartilhado devolve o mesmo vetor ----------------------
    carregado = {"chave": "cnn", "tipo": "torch", "modelo": modelo}
    scores_comuns = scores_de(carregado, ds, lote=LOTE_INFERENCIA,
                              dispositivo=dispositivo)
    desvio_comum = float(np.max(np.abs(scores_comuns - scores)))
    if desvio_comum != 0.0:
        raise ValueError(
            f"scores_de(chave='cnn') diverge da passada deste marco em "
            f"{desvio_comum:.3e}. O limiar publicado aqui seria aplicado, no "
            "B5.1, sobre scores produzidos por outro caminho de código.")

    return {
        "n_scores": int(len(scores)),
        "shuffle": False,
        "rotulos_conferidos_contra": f"{INDICE} -> classe_binaria",
        "n_rotulos_divergentes": 0,
        "mascaras_conferidas_contra": f"{INDICE} -> n_frames_validos",
        "n_mascaras_divergentes": 0,
        "conferencia_posicional": {
            "posicoes": posicoes,
            "metodo": ("cada posição reprocessada isoladamente (lote de 1) e "
                       "comparada ao score daquela posição no vetor em lote"),
            "maior_desvio_absoluto": maior_desvio,
            "controle_deslocamento_de_uma_casa": controle,
            "margem": round(margem, 1),
            "margem_exigida": 100,
            "criterio": ("o desvio medido tem de ser pelo menos 100x MENOR que o "
                         "que um deslocamento de uma casa produziria — pergunta "
                         "se o desvio é de arredondamento ou de posição, em vez "
                         "de comparar com uma constante escolhida a olho"),
            "por_que_nao_e_exato": ("somas em ponto flutuante reordenam entre "
                                    "tamanhos de lote; em eval() o BatchNorm usa "
                                    "as estatísticas do treino, então a "
                                    "composição do lote não muda o resultado"),
        },
        "caminho_compartilhado": {
            "funcao": "src.models.modelos_ajustados.scores_de(chave='cnn')",
            "maior_desvio_absoluto": desvio_comum,
            "por_que_importa": ("é esta a função que o B5.1 e os diagnósticos "
                                "chamam; se ela divergisse, o limiar publicado "
                                "aqui seria aplicado a outros scores"),
        },
    }


# =============================================================================
# 2b. «O teste não foi tocado» — confirmado por LEITURA DO CÓDIGO
# =============================================================================
# Os módulos que compõem o caminho deste marco. O critério de pronto do B4.7
# manda confirmar o isolamento do teste POR LEITURA DO CÓDIGO — não por memória
# nem por uma frase escrita à mão no JSON —, e é o mesmo método (AST) que o
# B4.5 e o B4.6 já usam.
ARQUIVOS_AUDITADOS = (
    "src/models/validar_cnn.py",
    "src/models/modelos_ajustados.py",
    "src/models/avaliacao.py",
    "src/models/tempo.py",
)

# SÓ os nomes do conjunto LACRADO. Os da validação externa estão fora de
# propósito: proibi-los aqui seria proibir o objeto deste marco. É a diferença
# entre a auditoria do B4.6 (onde abrir a validação era a violação) e a deste,
# onde abrir a validação é a tarefa — e o teste continua sendo o que não se toca.
#
# DERIVADO por filtro de `NOMES_DE_DADOS_PROIBIDOS`, não redigitado, por DOIS
# motivos. O primeiro é o de sempre: se aquela lista ganhar um nome novo do
# conjunto lacrado, esta acompanha sozinha. O segundo é que escrever os literais
# aqui faria a auditoria acusar a PRÓPRIA DECLARAÇÃO dela — foi o que aconteceu
# na primeira versão. `definir_cnn._constantes_de_codigo` isenta a declaração de
# `NOMES_DE_DADOS_PROIBIDOS` justamente por isso, e a isenção é por nome; um
# segundo literal com outro nome não é isentado. O filtro resolve os dois de uma
# vez, porque "teste" não contém nenhum dos nomes procurados.
NOMES_DO_CONJUNTO_LACRADO = tuple(
    n for n in NOMES_DE_DADOS_PROIBIDOS if "teste" in n)


def auditar_isolamento_do_teste(raiz: Path) -> dict:
    """Nenhum módulo do caminho deste marco nomeia o conjunto lacrado.

    Duas checagens, pelo mesmo método do B4.5/B4.6:

    1. nenhum IDENTIFICADOR e nenhuma STRING DE CÓDIGO dos módulos auditados
       contém um nome do conjunto lacrado. Docstrings e comentários são ignorados
       de propósito: é neles que o projeto DOCUMENTA a proibição, e uma guarda
       que dispara com a própria documentação é ruído que ensina a ignorar o
       alarme;
    2. toda chamada a `selecionar_limiar` declara o `conjunto` EXPLICITAMENTE, e
       declara `"validacao"` — nunca herda o default da assinatura. Um JSON que
       ALEGA vir da validação sem que alguém tenha escrito isso é exatamente o
       que a guarda de `carregar_modelo_ajustado` não consegue pegar.

    O md5 de cada módulo entra no retorno pela mesma razão de sempre: sem ele, a
    auditoria descreve um código que pode já não ser o que está no disco.
    """
    import ast

    auditados, violacoes = {}, []
    for rel in ARQUIVOS_AUDITADOS:
        caminho = raiz / rel
        fonte = caminho.read_text(encoding="utf-8")
        arvore = ast.parse(fonte)
        constantes = _constantes_de_codigo(arvore)
        identificadores = set(_identificadores(arvore))
        achados = [n for n in NOMES_DO_CONJUNTO_LACRADO
                   if any(n in c for c in constantes) or n in identificadores]
        chamadas = _chamadas_de_limiar(arvore)
        for c in chamadas:
            if not c.get("explicito") or c.get("conjunto_declarado") != "validacao":
                violacoes.append(f"{rel}:{c.get('linha')} -> selecionar_limiar "
                                 f"com conjunto {c.get('conjunto_declarado')!r} "
                                 f"(explicito={c.get('explicito')})")
        if achados:
            violacoes.append(f"{rel} nomeia o conjunto lacrado: {achados}")
        auditados[rel] = {
            "md5": hashlib.md5(caminho.read_bytes()).hexdigest(),
            "nomes_do_conjunto_lacrado_encontrados": achados,
            "chamadas_selecionar_limiar": chamadas,
        }
    if violacoes:
        raise RuntimeError("Auditoria de isolamento do teste FALHOU:\n  "
                           + "\n  ".join(violacoes))
    return {
        "verificado_por": ("leitura do código-fonte (AST) dos módulos do caminho "
                           "deste marco, não por memória nem por afirmação "
                           "escrita à mão"),
        "arquivos_auditados": auditados,
        "nomes_do_conjunto_lacrado": list(NOMES_DO_CONJUNTO_LACRADO),
        "por_que_os_nomes_da_validacao_nao_estao_na_lista": (
            "abrir a validação externa é a TAREFA deste marco, ao contrário do "
            "B4.5/B4.6, onde era a violação. O que continua proibido é o "
            "conjunto lacrado, e é só ele que esta auditoria procura"),
        "o_que_NAO_e_checado": ("docstrings e comentários, de propósito — é neles "
                                "que o projeto documenta a proibição, e um alarme "
                                "que dispara com a documentação ensina a "
                                "ignorá-lo"),
        "dados_efetivamente_lidos": [MEMMAP, INDICE, NORMALIZACAO,
                                     "models/cnn_final_30k.pt",
                                     "results/metricas/refit_cnn.json"],
        "resultado": ("nenhum módulo do caminho do B4.7 nomeia o conjunto "
                      "lacrado, e toda seleção de limiar declara "
                      "conjunto='validacao' explicitamente"),
    }


# =============================================================================
# 3. Tempos — GPU e CPU, com a sincronização DENTRO do callable
# =============================================================================
class LoteCnn:
    """Par (tensor, máscara) que se comporta como o `X` que `medir_tempos` espera.

    `medir_tempos` faz `X[:1]` para a latência e `len(X)` para o throughput —
    contrato de matriz. A CNN precisa de DOIS tensores (o espectrograma e a
    máscara temporal), e a alternativa seria mudar `tempo.py` para conhecer esse
    par. Não se muda: `tempo.py` é o protocolo ÚNICO de RF, SVM e CNN, e o dia em
    que ele ganhar um ramo por modelo deixa de ser evidência de que os três foram
    medidos igual. O adaptador mora aqui, do lado do modelo que é diferente.
    """

    def __init__(self, x: torch.Tensor, mascara: torch.Tensor):
        assert len(x) == len(mascara)
        self.x, self.mascara = x, mascara

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, fatia):
        return LoteCnn(self.x[fatia], self.mascara[fatia])


def montar_lote(ds, n: int | None = None) -> LoteCnn:
    """Materializa os espectrogramas JÁ NORMALIZADOS num par de tensores na CPU.

    O escopo da medição é o mesmo de RF e SVM — só a predição, a partir da
    representação JÁ EXTRAÍDA. Gerar o espectrograma faz parte do pipeline
    completo, medido em scripts/tempo_pipeline_completo.py, e não desta conta.
    """
    n = len(ds) if n is None else int(n)
    x = torch.empty(n, 1, 128, 251, dtype=torch.float32)
    mascara = torch.empty(n, 251, dtype=torch.float32)
    for i in range(n):
        x[i], mascara[i], _ = ds[i]
    return LoteCnn(x, mascara)


def _preditor(modelo, dispositivo, lote: int):
    """Callable para `medir_tempos`. A SINCRONIZAÇÃO MORA AQUI, não no helper.

    `torch.cuda.synchronize()` antes e depois: sem isso o `perf_counter` de
    `medir_tempos` mede o ENFILEIRAMENTO dos kernels, não a execução, e a CNN
    aparece ordens de grandeza mais rápida do que é. O `tempo.py` é compartilhado
    com RF e SVM e não deve conhecer CUDA — envolver a sincronização no callable
    mantém UM protocolo de medição para os três modelos.

    A sincronização de ENTRADA não é simetria decorativa: ela drena o que já
    estava na fila, para que este trecho não seja cobrado pelo trabalho do
    trecho anterior.

    A transferência host->device entra no trecho cronometrado, e isso é
    deliberado: os 22.226 espectrogramas não cabem residentes na GPU, então
    qualquer uso real paga a cópia lote a lote. Excluí-la produziria um número
    favorável à GPU que nenhum uso reproduz. Em CPU não há cópia, e o campo
    `inclui_transferencia_host_para_dispositivo` registra a diferença.
    """
    cuda = dispositivo.type == "cuda"

    def predizer(entrada: LoteCnn):
        if cuda:
            torch.cuda.synchronize()
        with torch.no_grad():
            for i in range(0, len(entrada), lote):
                p = entrada[i:i + lote]
                modelo(p.x.to(dispositivo), p.mascara.to(dispositivo))
        if cuda:
            torch.cuda.synchronize()

    return predizer


def _protocolo_cnn(bloco: dict, dispositivo, lote: int, n_threads) -> dict:
    """Corrige o `escopo` herdado de `tempo.py` para o vocabulário da CNN.

    `medir_tempos` escreve «predict_proba / decision_function» porque nasceu para
    RF e SVM. O ESCOPO é o mesmo — só a predição, a partir da representação já
    extraída —, mas deixar o texto como está poria no artefato da CNN uma frase
    que descreve outro modelo. `tempo.py` não foi modificado; a adaptação é do
    chamador e está declarada em `escopo_origem`.
    """
    p = dict(bloco["protocolo"])
    p["escopo"] = ("somente o forward da CNN (softmax sobre os logits) a partir "
                   "do espectrograma JÁ GERADO E NORMALIZADO — NÃO inclui "
                   "carregamento do áudio, VAD, nem geração do espectrograma")
    p["escopo_origem"] = (
        "o texto de src/models/tempo.py nomeia predict_proba/decision_function "
        "porque o helper nasceu para RF e SVM; o escopo é o mesmo (só a "
        "predição, a partir da representação já extraída) e tempo.py NÃO foi "
        "modificado — a adaptação do vocabulário é deste módulo")
    p["dispositivo"] = str(dispositivo)
    p["batch_throughput"] = lote
    p["inclui_transferencia_host_para_dispositivo"] = dispositivo.type == "cuda"
    p["torch_num_threads"] = n_threads
    # Declarado porque ENCARECE a medida: `fixar_seeds_torch` liga
    # use_deterministic_algorithms e desliga cudnn.benchmark, então o cuDNN não
    # autotuna e pode escolher um kernel mais lento. O número reportado é o do
    # regime determinístico em que TODO o trabalho roda — medir fora dele daria
    # um tempo que nenhuma execução deste repositório reproduz.
    p["aquecimento_do_dispositivo_s"] = AQUECIMENTO_DISPOSITIVO_S
    p["regime_deterministico"] = {
        "use_deterministic_algorithms": True,
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "efeito_no_tempo": ("sem autotune do cuDNN a medida tende a ser "
                            "CONSERVADORA (igual ou mais lenta); é o regime em "
                            "que o resto do trabalho roda"),
    }
    bloco["protocolo"] = p
    return bloco


def _aquecer(predizer, entrada, segundos: float = AQUECIMENTO_DISPOSITIVO_S) -> dict:
    """Leva o dispositivo ao regime de operação antes de cronometrar.

    Por tempo de parede, e não por número de iterações, de propósito: o que
    precisa ser amortizado — clock, contexto, kernels do cuDNN — é medido em
    milissegundos de trabalho, não em chamadas. Um número fixo de iterações
    aqueceria de menos em GPU (onde a iteração custa ~0,9 ms) e de mais em CPU
    (onde custa ~15 ms).
    """
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < segundos:
        predizer(entrada)
        n += 1
    return {"segundos_alvo": segundos,
            "segundos_reais": round(time.perf_counter() - t0, 3),
            "n_iteracoes": n}


def medir_cenario(modelo, lote_dados: LoteCnn, dispositivo, cfg_tempo: dict,
                  lote: int = LOTE_THROUGHPUT) -> dict:
    """Latência (batch=1) e throughput (lotes de `lote`) num dispositivo.

    Mede a latência DUAS vezes: uma com o dispositivo como ele chega (fria) e
    outra depois de aquecido. A primeira não é descartada — vai para o artefato
    como `aquecimento_do_dispositivo.latencia_sem_aquecer_ms`, porque é
    exatamente o número que este marco publicaria sem a correção, e publicá-lo ao
    lado é o que torna a correção auditável (ver AQUECIMENTO_DISPOSITIVO_S).
    """
    modelo = modelo.to(dispositivo)
    modelo.eval()
    predizer = _preditor(modelo, dispositivo, lote)

    # Sonda fria: só a latência, sobre um exemplo. Custa ~0,1 s e é o "antes".
    fria = medir_tempos(predizer, lote_dados[:1],
                        {**cfg_tempo, "medir_throughput": False})
    aquecimento = _aquecer(predizer, lote_dados[:1])

    bloco = medir_tempos(predizer, lote_dados, cfg_tempo)
    bloco = _protocolo_cnn(bloco, dispositivo, lote, torch.get_num_threads())
    bloco["aquecimento_do_dispositivo"] = {
        **aquecimento,
        "latencia_sem_aquecer_ms": fria["latencia_ms"],
        "latencia_depois_de_aquecer_ms": bloco.get("latencia_ms"),
        "por_que": (
            "os 3 descartes do config são 3 ITERAÇÕES (~3 ms em GPU), o que não "
            "tira o dispositivo do estado de baixo consumo depois de um trecho "
            "longo de CPU. src/models/tempo.py NÃO foi alterado — o aquecimento "
            "acontece ANTES de o callable ser entregue ao cronômetro, e o número "
            "frio fica publicado ao lado para conferência"),
    }
    return bloco


def demonstrar_sem_sincronizacao(modelo, lote_dados: LoteCnn, n_rep: int = 10) -> dict:
    """Mede o MESMO trabalho com e sem `synchronize()`, e registra o número falso.

    A armadilha fica documentada com evidência desta máquina, e não com uma
    afirmação de manual. Os tensores ficam RESIDENTES na GPU nesta demonstração
    justamente para isolar o efeito: com a cópia host->device no meio, a própria
    cópia já sincroniza em parte e mascara o tamanho do erro.
    """
    if not torch.cuda.is_available():
        return {"aplicavel": False,
                "motivo": "sem GPU nesta execução — nada a demonstrar"}
    dispositivo = torch.device("cuda")
    modelo = modelo.to(dispositivo).eval()
    x = lote_dados.x[:LOTE_THROUGHPUT].to(dispositivo)
    m = lote_dados.mascara[:LOTE_THROUGHPUT].to(dispositivo)

    with torch.no_grad():
        for _ in range(3):                       # aquecimento
            modelo(x, m)
        torch.cuda.synchronize()

        t0 = time.perf_counter()                 # SEM sincronizar: enfileiramento
        for _ in range(n_rep):
            modelo(x, m)
        t_falso = (time.perf_counter() - t0) / n_rep
        torch.cuda.synchronize()                 # drena o que ficou na fila

        torch.cuda.synchronize()
        t0 = time.perf_counter()                 # COM sincronização: execução
        for _ in range(n_rep):
            modelo(x, m)
        torch.cuda.synchronize()
        t_real = (time.perf_counter() - t0) / n_rep

    return {
        "aplicavel": True,
        "batch": LOTE_THROUGHPUT,
        "repeticoes": n_rep,
        "tensores_residentes_na_gpu": True,
        "ms_sem_sincronizacao": round(1000 * t_falso, 4),
        "ms_com_sincronizacao": round(1000 * t_real, 4),
        "fator_de_ilusao": round(t_real / t_falso, 1) if t_falso > 0 else None,
        "leitura": ("o primeiro número NÃO é uma medida incompleta do segundo: é "
                    "o tempo de ENFILEIRAR os kernels. Publicá-lo seria uma "
                    "afirmação falsa sobre o custo da CNN em GPU"),
    }


# =============================================================================
# 4. Execução
# =============================================================================
def executar(cfg: dict, raiz: Path, estrito: bool = True) -> dict:
    # ---- determinismo: mesma ordem e mesma sondagem do refit ----------------
    semente = fixar_seeds_torch(int(cfg["semente"]), estrito=estrito)
    determinismo_estrito, limitacao_determinismo = estrito, None
    # TF32 desligado ANTES de qualquer forward: é o que torna o score
    # independente do tamanho do lote, e portanto o limiar reprodutível. A
    # medição de tempo também roda neste regime, de propósito — reportar o tempo
    # de um cálculo que não é o que produziu os números seria um número certo
    # sobre a coisa errada. Ver fixar_precisao_fp32_cnn.
    precisao = fixar_precisao_fp32_cnn()
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {dispositivo} | TF32 desligado (FP32 pleno)")

    # «O teste não foi tocado — confirme por LEITURA DO CÓDIGO» (critério de
    # pronto do B4.7). Roda ANTES de abrir qualquer dado: se o caminho deste
    # marco nomeasse o conjunto lacrado, nada mais deveria acontecer.
    print("Auditando isolamento do conjunto lacrado (leitura do código)...")
    isolamento = auditar_isolamento_do_teste(raiz)
    print(f"  OK — {isolamento['resultado']}")

    modelo, prov = carregar_cnn_persistida(raiz)
    print(f"CNN carregada: {prov['origem_arquitetura']}")
    print(f"  {prov['arquitetura']['n_parametros']:,} parâmetros | "
          f"{prov['n_epocas_fixas']} épocas fixas | "
          f"normalização do estágio {prov['estagio_normalizacao']}")

    if estrito:
        # use_deterministic_algorithms(True) não falha ao ser CHAMADO, falha
        # quando uma operação sem versão determinística é EXECUTADA. A sondagem
        # é um forward de mentira; a semente é re-fixada depois porque ele
        # consome estado do gerador.
        try:
            with torch.no_grad():
                _m = modelo.to(dispositivo)
                _m(torch.zeros(2, 1, 128, 251, device=dispositivo),
                   torch.ones(2, 251, device=dispositivo))
            if dispositivo.type == "cuda":
                torch.cuda.synchronize()
        except RuntimeError as e:
            limitacao_determinismo = (
                f"use_deterministic_algorithms(True) falhou numa operacao do "
                f"forward: {e}. Reexecutado com estrito=False e REGISTRADO como "
                "limitacao.")
            print(f"AVISO: {limitacao_determinismo}")
            determinismo_estrito = False
        semente = fixar_seeds_torch(int(cfg["semente"]),
                                    estrito=determinismo_estrito)

    # ---- Passo 1: scores na validação externa -------------------------------
    ds, indice, norm = dataset_validacao(raiz)
    print(f"\nValidação externa: {len(ds)} exemplos | normalização "
          f"{NORMALIZACAO} (estágio {norm['estagio']})")

    t0 = time.perf_counter()
    saida = scores_e_rotulos(modelo, ds, dispositivo)
    print(f"Scores em {time.perf_counter() - t0:.1f}s. Conferindo alinhamento...")
    alinhamento = conferir_alinhamento(saida, indice, modelo, ds, dispositivo)
    print(f"  OK — {alinhamento['n_scores']} scores, 0 rótulos divergentes, "
          f"conferência posicional com desvio máximo "
          f"{alinhamento['conferencia_posicional']['maior_desvio_absoluto']:.2e}")

    scores, y_val = saida["scores"], saida["y"]

    # ---- Passo 2: limiar, pela MESMA função dos outros dois -----------------
    sel = selecionar_limiar(y_val, scores, criterio="f1_macro",
                            conjunto="validacao")
    print(f"\nLimiar selecionado: {sel['limiar']:.6f} -> f1_macro "
          f"{sel['f1_macro']:.4f} | {sel['n_candidatos']} candidatos, "
          f"{sel['n_empates_no_maximo']} empate(s) no máximo")
    saturacao = _nota_saturacao(scores, sel)
    print(f"  {saturacao['leitura']}")

    m = avaliar(y_val, scores, nome=NOME, limiar=sel["limiar"])
    m["selecao_limiar"] = sel
    m["nota_escala_score"] = (
        "score = softmax(logits)[:, 1] = P(spoof), em [0,1]. Monotonicamente "
        "equivalente a logits[:,1] - logits[:,0], portanto o limiar selecionado "
        "é o mesmo ponto de corte nas duas parametrizações. Escala comparável ao "
        "predict_proba do RF; diferente do decision_function do SVM, e "
        "selecionar_limiar é agnóstico de escala.")
    m["braco"] = "principal"
    m["arquitetura"] = prov["arquitetura"]
    m["origem_arquitetura"] = prov["origem_arquitetura"]
    m["hiperparametros"] = prov["hiperparametros"]
    m["pesos_loss"] = prov["pesos_loss"]
    m["n_parametros"] = int(prov["arquitetura"]["n_parametros"])
    m["n_epocas_fixas"] = prov["n_epocas_fixas"]
    m["origem_n_epocas"] = prov["origem_n_epocas"]
    m["n_treino"] = prov["n_treino"]
    m["n_validacao"] = int(len(scores))
    m["semente"] = semente
    m["tempo_treino_s"] = prov["tempo_treino_s"]
    m["origem_tempo_treino"] = "results/metricas/refit_cnn.json -> tempo_treino_s"

    cm = confusion_matrix(y_val, (scores >= sel["limiar"]).astype(int),
                          labels=[0, 1])
    m["matriz_confusao"] = cm.tolist()
    m["roc_auc_validacao"] = round(float(roc_auc_score(y_val, scores)), 4)
    m["determinismo"] = {
        "fixar_seeds_torch": True,
        "estrito": determinismo_estrito,
        "limitacao": limitacao_determinismo,
        "num_workers": 0,
        "escopo": ("a INFERÊNCIA deste marco; o determinismo do TREINO está em "
                   "results/metricas/refit_cnn.json -> determinismo"),
        "determinismo_do_treino": prov["determinismo"],
    }
    m["precisao_numerica"] = precisao

    print(f"  f1_macro : {m['f1_macro']:.4f} | EER {m['eer']:.4f} | "
          f"AUC {m['roc_auc_validacao']:.4f}")
    print(f"  bonafide : recall {m['recall_bonafide']:.4f} | "
          f"precisão {m['precisao_bonafide']:.4f}")

    # ---- Passo 3: tempos nos DOIS cenários exigidos pelo config -------------
    m["tempos_inferencia"] = medir_tempos_cnn(modelo, ds, cfg, dispositivo)

    m["ambiente"] = ambiente(n_jobs_inferencia=1)
    m["ambiente"]["torch"] = torch.__version__
    m["ambiente"]["cuda_disponivel"] = bool(torch.cuda.is_available())
    m["ambiente"]["cuda_versao"] = torch.version.cuda
    m["ambiente"]["cudnn"] = torch.backends.cudnn.version()
    m["ambiente"]["gpu"] = (torch.cuda.get_device_name(0)
                            if torch.cuda.is_available() else None)
    m["ambiente"]["compute_capability"] = (
        list(torch.cuda.get_device_capability(0))
        if torch.cuda.is_available() else None)
    m["ambiente"]["dispositivo_metricas"] = str(dispositivo)

    m["normalizacao"] = f"{NORMALIZACAO} (estagio {norm['estagio']})"
    hashes = hashes_congelados(raiz, cfg["experimento"]["caminho_subamostra"])
    m["hash_md5_features_csv"] = hashes["features"]
    m["hash_md5_split_csv"] = hashes["split"]
    m["hash_md5_subamostra_csv"] = hashes["subamostra"]
    m["hash_md5_indice_validacao_csv"] = _md5(raiz / INDICE)
    # O artefato de normalização é a armadilha nº 1 do marco (27k x 30k): o hash
    # é o que permite provar, depois, QUAL dos dois foi usado.
    m["hash_md5_normalizacao_30k_json"] = _md5(raiz / NORMALIZACAO)
    # Os dois campos abaixo vêm DEPOIS dos exigidos, de propósito: o B5.2 monta a
    # tabela comparativa lendo os três JSONs, e os campos obrigatórios ficam na
    # MESMA ordem de svm_tuned_principal.json. O que é específico da CNN entra no
    # fim, sem deslocar nada.
    m["saturacao_do_score"] = saturacao
    m["alinhamento_score_rotulo"] = alinhamento
    m["leitura_do_resultado"] = _leitura_do_resultado(m, raiz)
    print(f"\nLeitura [{m['leitura_do_resultado']['faixa']}]: "
          f"{m['leitura_do_resultado']['leitura']}")
    m["isolamento_teste"] = isolamento
    # NOTA DE REDAÇÃO: esta string deliberadamente NÃO escreve os nomes dos
    # arquivos do conjunto lacrado. A auditoria de isolamento do B4.5/B4.6
    # (definir_cnn._constantes_de_codigo) varre identificadores e STRINGS de
    # código à procura desses nomes, ignorando de propósito docstrings e
    # comentários — que é onde o projeto documenta a proibição. Uma frase de
    # runtime dizendo «não abrimos teste.npy» dispararia o alarme com a própria
    # documentação, e um alarme que soa por bom comportamento ensina a ignorá-lo.
    m["teste_nao_tocado"] = (
        "nenhum caminho deste módulo abre o memmap nem o índice do conjunto "
        "lacrado; os únicos dados lidos estão declarados nas constantes INDICE, "
        "MEMMAP e NORMALIZACAO deste módulo, todas da validação. O conjunto "
        "lacrado segue intocado até o B5.1")

    # ---- Saída --------------------------------------------------------------
    dir_met = raiz / "results" / "metricas"
    dir_met.mkdir(parents=True, exist_ok=True)
    with open(dir_met / f"{NOME}.json", "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)
    # Título na MESMA forma dos de RF e SVM («X (braço principal) — validação,
    # limiar Y»), e por um motivo prático além da simetria: `plotar_matriz_confusao`
    # é compartilhada e tem figsize fixo, então um título mais longo que o dos
    # outros dois é CORTADO na figura. O «refit 30k» vive no JSON e no rótulo de
    # MODELOS_PRINCIPAIS, que é onde ele precisa estar.
    plotar_matriz_confusao(
        cm, raiz / "results" / "figuras" / f"matriz_confusao_{NOME}.png",
        f"CNN final (braço principal) — validação, limiar {sel['limiar']:.2f}")
    print(f"\nSalvo: results/metricas/{NOME}.json")
    return m


def medir_tempos_cnn(modelo, ds, cfg: dict, dispositivo) -> dict:
    """Os quatro números do protocolo que cabem a este marco: CNN-GPU e CNN-CPU.

    A JUSTIFICATIVA DO PAR, que é o coração do diferencial do trabalho: a
    vantagem dos modelos clássicos, na pergunta de pesquisa, NÃO é serem mais
    rápidos em igualdade de hardware — é NÃO EXIGIREM GPU. O par CNN-GPU /
    CNN-CPU é o único arranjo que evidencia isso.
    """
    cfg_tempo = cfg["tempo"]
    tem_gpu = torch.cuda.is_available()
    resultado: dict = {}

    print(f"\nMaterializando os {len(ds)} espectrogramas normalizados "
          "(~2,9 GB em float32)...")
    lote_completo = montar_lote(ds)
    lote_amostra = lote_completo[:N_THROUGHPUT_CPU]

    if tem_gpu:
        gpu = torch.device("cuda")
        print(f"Medindo CNN-GPU (latência batch=1 + throughput sobre "
              f"{len(lote_completo)}, lotes de {LOTE_THROUGHPUT})...")
        resultado["gpu"] = medir_cenario(modelo, lote_completo, gpu, cfg_tempo)
        print(f"  latência {resultado['gpu']['latencia_ms']['mediana']} ms | "
              f"{resultado['gpu']['throughput']['ms_por_audio_mediana']} ms/áudio")
        pareado_gpu = medir_cenario(modelo, lote_amostra, gpu, cfg_tempo)
    else:
        resultado["gpu"] = {"medido": False,
                            "motivo": "nenhuma GPU CUDA disponível nesta execução"}
        pareado_gpu = None

    # torch.set_num_threads(1) É OBRIGATÓRIO: o protocolo exige fixar n_jobs
    # igual para todos os modelos e declarar qual foi, e rf_tuned_principal.json
    # registra n_jobs_inferencia = 1. Uma CNN em CPU com 12 threads contra um RF
    # com n_jobs=1 não é comparação, é ruído.
    threads_originais = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu = torch.device("cpu")
    print(f"Medindo CNN-CPU com torch_num_threads=1 (latência batch=1 + "
          f"throughput sobre {len(lote_amostra)}, lotes de {LOTE_THROUGHPUT})...")
    resultado["cpu"] = medir_cenario(modelo, lote_amostra, cpu, cfg_tempo)
    print(f"  latência {resultado['cpu']['latencia_ms']['mediana']} ms | "
          f"{resultado['cpu']['throughput']['ms_por_audio_mediana']} ms/áudio")

    # Os números desta justificativa saem da MEDIÇÃO desta execução, não de
    # constantes na prosa: uma explicação que cita um tempo diferente do que o
    # artefato reporta duas linhas acima é pior que nenhuma explicação.
    ms_cpu = resultado["cpu"]["throughput"]["ms_por_audio_mediana"]
    s_por_passada = ms_cpu * N_VALIDACAO_ESPERADO / 1000
    reps_totais = int(cfg_tempo["repeticoes"]) + int(
        cfg_tempo["descartar_aquecimento"])
    resultado["cpu"]["throughput"]["cobertura"] = {
        "n_audios": int(len(lote_amostra)),
        "n_audios_da_validacao": N_VALIDACAO_ESPERADO,
        "cobertura_completa": False,
        "por_que": (
            f"a CNN em CPU com 1 thread levou {ms_cpu:.2f} ms por áudio NESTA "
            f"medição; cobrir os 22.226 exigiria {s_por_passada:.0f} s por "
            f"passada, e o protocolo manda {reps_totais} passadas "
            f"({cfg_tempo['repeticoes']} + "
            f"{cfg_tempo['descartar_aquecimento']} de aquecimento) — cerca de "
            f"{reps_totais * s_por_passada / 3600:.1f} h de CPU para refinar a "
            "terceira casa de um número que é uma TAXA e já está estável em n "
            "muito menor. O número reportado é ms por áudio, que não depende de "
            "n uma vez amortizado o início; `comparacao_gpu_cpu` mede a GPU "
            "sobre ESTA MESMA amostra, de modo que a razão entre os dois "
            "cenários sai de trabalho idêntico, e a estabilidade da taxa em n "
            "está conferida ali."),
        "total_s_extrapolado_para_22226": round(s_por_passada, 1),
        "nota_extrapolacao": ("EXTRAPOLAÇÃO, não medição: taxa medida x 22.226. "
                              "Declarado como tal para não virar, no texto, um "
                              "tempo que ninguém cronometrou."),
    }
    if tem_gpu:
        resultado["gpu"]["throughput"]["cobertura"] = {
            "n_audios": int(len(lote_completo)),
            "n_audios_da_validacao": N_VALIDACAO_ESPERADO,
            "cobertura_completa": True,
            "por_que": (
                "em GPU a passada completa custou "
                f"{resultado['gpu']['throughput']['total_s_mediana']:.1f} s "
                f"(mediana); os 22.226 cabem no orçamento das {reps_totais} "
                "passadas do protocolo, então aqui a cobertura é integral"),
        }
    torch.set_num_threads(threads_originais)

    resultado["nota_sincronizacao"] = (
        "torch.cuda.synchronize() antes e depois do trecho cronometrado, DENTRO "
        "do callable passado a medir_tempos; sem isso mede-se o enfileiramento, "
        "nao a execucao. src/models/tempo.py NAO foi modificado — ele e o "
        "protocolo unico de RF, SVM e CNN e nao deve conhecer CUDA.")
    resultado["torch_num_threads_cpu"] = 1

    if pareado_gpu is not None:
        r_lat = (resultado["cpu"]["latencia_ms"]["mediana"]
                 / resultado["gpu"]["latencia_ms"]["mediana"])
        r_thr = (resultado["cpu"]["throughput"]["ms_por_audio_mediana"]
                 / pareado_gpu["throughput"]["ms_por_audio_mediana"])
        resultado["comparacao_gpu_cpu"] = {
            "n_audios": int(len(lote_amostra)),
            "por_que_existe": ("o throughput em GPU cobre os 22.226 e o em CPU "
                               "cobre uma amostra; esta medição roda a GPU sobre "
                               "A MESMA amostra da CPU, para que a razão entre "
                               "os dois cenários venha de trabalho idêntico"),
            "gpu_ms_por_audio": pareado_gpu["throughput"]["ms_por_audio_mediana"],
            "cpu_ms_por_audio":
                resultado["cpu"]["throughput"]["ms_por_audio_mediana"],
            "razao_cpu_sobre_gpu_throughput": round(r_thr, 1),
            "razao_cpu_sobre_gpu_latencia": round(r_lat, 1),
            "gpu_ms_por_audio_na_validacao_completa":
                resultado["gpu"]["throughput"]["ms_por_audio_mediana"],
            "estabilidade_da_taxa_em_n": (
                "a taxa da GPU sobre 22.226 e sobre a amostra estão no mesmo "
                "campo para conferência direta; taxas próximas são o que "
                "autoriza ler o número da CPU como taxa"),
        }
        resultado["demonstracao_sem_sincronizacao"] = (
            demonstrar_sem_sincronizacao(modelo, lote_amostra))
        d = resultado["demonstracao_sem_sincronizacao"]
        if d.get("aplicavel"):
            print(f"  armadilha CUDA: sem synchronize() o mesmo lote 'leva' "
                  f"{d['ms_sem_sincronizacao']} ms contra "
                  f"{d['ms_com_sincronizacao']} ms reais "
                  f"({d['fator_de_ilusao']}x)")

    resultado["leitura_critica"] = (
        "a vantagem dos modelos classicos, na pergunta de pesquisa, NAO e serem "
        "mais rapidos em igualdade de hardware — e NAO EXIGIREM GPU. O par "
        "CNN-GPU / CNN-CPU e o unico arranjo que evidencia isso, e e por isso "
        "que o config.yaml pede os dois cenarios.")
    return resultado


def _nota_saturacao(scores: np.ndarray, sel: dict) -> dict:
    """Quantos scores distintos, e quantos exatamente em 0,0 e 1,0.

    `n_candidatos` muito abaixo de 22.226 significa softmax saturando — a rede
    está superconfiante. Não é erro, mas é a mesma leitura que a NOTA_LIMIAR.md
    faz do RF («o score deixou de ser degrau»), e o número tem de ser registrado
    para que o texto possa fazê-la.
    """
    n = len(scores)
    n_zero = int(np.sum(scores == 0.0))
    n_um = int(np.sum(scores == 1.0))
    frac = sel["n_candidatos"] / n
    if frac > 0.95:
        leitura = (f"{sel['n_candidatos']} valores distintos em {n} scores "
                   f"({100 * frac:.1f}%): o score é praticamente contínuo, sem "
                   "saturação relevante do softmax.")
    else:
        leitura = (f"{sel['n_candidatos']} valores distintos em {n} scores "
                   f"({100 * frac:.1f}%), com {n_zero} exatamente 0,0 e {n_um} "
                   "exatamente 1,0: o softmax SATURA. A rede é superconfiante — "
                   "não é erro, mas o limiar opera sobre um score de "
                   "granularidade grossa e isso tem de estar no texto.")
    return {
        "n_scores": n,
        "n_valores_distintos": sel["n_candidatos"],
        "fracao_distintos": round(frac, 6),
        "n_exatamente_zero": n_zero,
        "n_exatamente_um": n_um,
        "leitura": leitura,
        "referencia": ("a NOTA_LIMIAR.md faz a mesma leitura do RF: 22.207 "
                       "valores distintos em 22.226 amostras foi o que provou "
                       "que «o score deixou de ser degrau»"),
    }


def _leitura_do_resultado(m: dict, raiz: Path) -> dict:
    """Em qual célula da tabela «Como ler o resultado» do B4.7 este número cai.

    OS NÚMEROS DE REFERÊNCIA SÃO LIDOS DOS JSONs DE RF E SVM, não redigitados: a
    tabela do APÊNDICE A os repete, e um número copiado à mão que envelhece é
    como se descreve errado um resultado próprio.

    ESTA FUNÇÃO NÃO COMPARA ESTATISTICAMENTE NADA. O bootstrap pareado
    CNN x SVM x RF é B5.2, e dizer aqui «a CNN é melhor» com base na diferença
    pontual seria exatamente o «diga isso com o bootstrap, não com o olho» que o
    marco proíbe. O que sai daqui é a leitura QUALITATIVA da faixa, com a
    ressalva explícita.
    """
    dir_met = raiz / "results" / "metricas"
    referencias = {}
    for chave, arq in (("rf_principal", "rf_tuned_principal.json"),
                       ("svm_principal", "svm_tuned_principal.json")):
        caminho = dir_met / arq
        if caminho.exists():
            with open(caminho, encoding="utf-8") as f:
                r = json.load(f)
            referencias[chave] = {"f1_macro": r["f1_macro"], "eer": r["eer"],
                                  "origem": f"results/metricas/{arq}"}

    f1, recall_spoof = m["f1_macro"], m["recall_spoof"]
    if f1 < 0.55 and recall_spoof > 0.98:
        faixa = "colapso_na_majoritaria"
        leitura = ("f1_macro perto de 0,47 com recall_spoof perto de 1,0 NÃO é "
                   "resultado, é BUG: a rede colapsou na classe majoritária. "
                   "Confira os pesos da loss e o alinhamento score<->rótulo "
                   "antes de escrever qualquer coisa.")
    elif f1 >= 0.80:
        faixa = "competitiva_ou_melhor"
        leitura = ("a CNN é competitiva ou melhor que os modelos clássicos sob "
                   "ESTE protocolo. A pergunta de pesquisa vira «e a que custo "
                   "computacional?» — e aí os tempos GPU/CPU deste marco são o "
                   "núcleo da discussão, não um apêndice.")
    elif f1 >= 0.72:
        faixa = "classicos_competitivos"
        leitura = ("a faixa mais interessante para a pergunta de pesquisa: "
                   "modelos clássicos com features manuais mantêm desempenho "
                   "competitivo diante da CNN.")
    else:
        faixa = "abaixo_dos_classicos"
        leitura = ("plausível e reportável: 30k exemplos com 3.000 bonafide é "
                   "pouco para uma CNN, que é faminta por dados. Enquadrar como "
                   "limitação DO EXPERIMENTO, não como fracasso do método.")

    return {
        "faixa": faixa,
        "leitura": leitura,
        "referencias_na_mesma_validacao": referencias,
        "ressalva_obrigatoria": (
            "estes são números PONTUAIS na mesma validação de 22.226. A "
            "comparação estatística formal (bootstrap pareado CNN x SVM x RF) é "
            "o B5.2. Nada aqui autoriza afirmar que uma diferença é real — isso "
            "se diz com o intervalo de confiança, não com o olho."),
        "o_que_ja_esta_conferido": (
            "que os três modelos foram avaliados sobre EXATAMENTE o mesmo "
            "conjunto: o índice desta validação e o `conjunto == 'validacao'` de "
            "split.csv têm os mesmos 22.226 arquivos e os mesmos rótulos, e o "
            "hash de split.csv está registrado nos três JSONs."),
    }


def _md5(caminho: Path) -> str:
    return hashlib.md5(caminho.read_bytes()).hexdigest()


def main() -> dict:
    from ..utils.config import carregar_config

    assert "cnn" in MODELOS_PRINCIPAIS, "a CNN precisa estar em MODELOS_PRINCIPAIS"
    print(f"B4.7 — validação externa da CNN | python {platform.python_version()}")
    return executar(carregar_config(RAIZ), RAIZ)


if __name__ == "__main__":
    main()
