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
    RF ([0,1]), decision_function para o SVM (real, centrado em zero) e
    softmax(logits)[:, 1] para a CNN ([0,1]) — e o limiar SELECIONADO NA
    VALIDAÇÃO, lido de `selecao_limiar.limiar`. Nunca 0,50, nunca recalculado no
    conjunto que está sendo diagnosticado.

A CNN ENTROU AQUI NO B4.7 (20/09/2026), E NÃO NUM MÓDULO PARALELO:
    ver o comentário de `MODELOS_PRINCIPAIS`. O resumo: as duas guardas desta
    função — limiar presente, limiar vindo da validação — são o que protege o
    teste lacrado, e o B5.1 carrega os três modelos por aqui. Só o CARREGAMENTO
    ramifica por `tipo` ("sklearn" -> joblib.load; "torch" -> reconstrução da
    arquitetura + load_state_dict). As guardas vêm ANTES do ramo, e por isso
    valem para os três.

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

# Chave curta -> (artefato do modelo, JSON de métricas, rótulo, TIPO do artefato).
# O par artefato/json é fixo: o limiar de um modelo só vale para AQUELE modelo.
#
# POR QUE A CNN ENTRA AQUI, E NÃO NUM LOADER PARALELO (B4.7):
#     `carregar_modelo_ajustado` faz duas coisas que a CNN precisa herdar: recusa
#     um JSON sem `selecao_limiar.limiar`, e recusa um limiar cujo `conjunto` não
#     seja "validacao". Essas duas guardas são a regra que protege o teste lacrado
#     morando no código, não só no texto — e o B5.1 carrega os TRÊS modelos por
#     esta função. Um loader separado para a CNN faria as guardas valerem para
#     dois dos três modelos comparados, e o terceiro seria justamente o mais novo
#     e o menos testado. Por isso a CNN entra na MESMA estrutura, com um campo de
#     `tipo`, e só o CARREGAMENTO ramifica — nunca as guardas.
MODELOS_PRINCIPAIS = {
    "rf": ("rf_tuned_principal.joblib", "rf_tuned_principal.json",
           "RF ajustado (braço principal)", "sklearn"),
    "svm": ("svm_tuned_principal.joblib", "svm_tuned_principal.json",
            "SVM ajustado (braço principal)", "sklearn"),
    "cnn": ("cnn_final_30k.pt", "cnn_final_principal.json",
            "CNN final (braço principal, refit 30k)", "torch"),
}

# A proveniência da arquitetura da CNN. O .pt do B4.6 já carrega `arquitetura`
# junto dos pesos; este JSON é a SEGUNDA fonte, e a reconstrução exige que as
# duas concordem (ver `carregar_cnn_persistida`).
REFIT_CNN_JSON = "results/metricas/refit_cnn.json"


def carregar_cnn_persistida(raiz: Path, arq_modelo: str = "cnn_final_30k.pt"):
    """Reconstrói a CNN do B4.6 a partir do `.pt` e devolve (modelo em eval, proveniência).

    POR QUE A ARQUITETURA É CONFERIDA CONTRA DUAS FONTES:
        o `.pt` gravado pelo refit carrega `arquitetura` ao lado dos pesos, e
        `results/metricas/refit_cnn.json` registra a mesma descrição. Reconstruir
        pela do `.pt` — que é a que acompanha OS PESOS, e portanto não pode ter
        divergido deles — e EXIGIR que ela seja idêntica à do JSON transforma um
        erro silencioso num erro alto: um `.pt` de arquitetura diferente já
        falharia no `load_state_dict`, mas um `.pt` de MESMA forma e proveniência
        diferente passaria batido. `load_state_dict` roda em modo estrito (o
        default), então chave faltando ou sobrando também aborta.

        A reconstrução usa `CnnDeepfake` importada de src/models/cnn.py — o mesmo
        módulo do treino. Nada de redefinir a rede aqui: uma segunda definição é
        exatamente a divergência silenciosa que este módulo existe para evitar.

    `p_drop` é irrelevante em `eval()` (Dropout vira identidade), mas entra na
    reconstrução assim mesmo: o construtor registra `self.p_drop` e é ele que sai
    em `descricao()`, que vai para o JSON de métricas.

    Returns:
        (modelo em eval() na CPU, dict de proveniência). Mover para a GPU é
        responsabilidade do chamador — este módulo não escolhe dispositivo.
    """
    import torch                       # local: quem só usa RF/SVM não paga o import

    from .cnn import CnnDeepfake

    caminho_pt = raiz / "models" / arq_modelo
    caminho_refit = raiz / REFIT_CNN_JSON
    for c in (caminho_pt, caminho_refit):
        if not c.exists():
            raise FileNotFoundError(
                f"Artefato ausente: {c}. Rode o refit do B4.6 antes "
                "(python -m src.models.refit_cnn).")

    ck = torch.load(caminho_pt, map_location="cpu", weights_only=False)
    arq = ck["arquitetura"]
    with open(caminho_refit, encoding="utf-8") as f:
        refit = json.load(f)
    if arq != refit["arquitetura"]:
        difer = sorted(k for k in set(arq) | set(refit["arquitetura"])
                       if arq.get(k) != refit["arquitetura"].get(k))
        raise ValueError(
            f"A arquitetura gravada em models/{arq_modelo} DIVERGE da registrada "
            f"em {REFIT_CNN_JSON} nos campos {difer}. Os pesos e a proveniência "
            "descrevem redes diferentes — pare e investigue antes de citar "
            "qualquer número.")

    modelo = CnnDeepfake(n_mels=int(arq["n_mels"]),
                         canais=tuple(arq["canais"]),
                         p_drop=float(arq["p_dropout"]))
    modelo.load_state_dict(ck["state_dict"])   # strict=True (default)
    modelo.eval()
    return modelo, {
        "arquitetura": arq,
        "hiperparametros": refit["hiperparametros"],
        "pesos_loss": refit["pesos_loss"],
        "n_epocas_fixas": int(ck["n_epocas_fixas"]),
        "origem_n_epocas": refit["origem_n_epocas"],
        "estagio_normalizacao": ck["estagio_normalizacao"],
        "normalizacao": refit["normalizacao"],
        "semente_treino": int(ck["semente"]),
        "n_treino": int(refit["n_treino"]),
        "tempo_treino_s": refit["tempo_treino_s"],
        "determinismo": refit["determinismo"],
        "origem_arquitetura": (f"models/{arq_modelo} -> arquitetura, conferida "
                               f"contra {REFIT_CNN_JSON} -> arquitetura"),
    }


def carregar_modelo_ajustado(raiz: Path, chave: str) -> dict:
    """Devolve o modelo persistido, o limiar do protocolo e a proveniência.

    Args:
        raiz: raiz do repositório.
        chave: 'rf', 'svm' ou 'cnn'.

    Returns:
        dict com modelo, limiar, rotulo, nome_arquivo e o JSON de métricas
        inteiro (`metricas`), para que o chamador possa registrar de onde o
        número veio sem reabrir o arquivo. Para 'cnn', também `proveniencia`
        (arquitetura, épocas, normalização — ver `carregar_cnn_persistida`), e o
        modelo vem em `eval()` na CPU.

    Raises:
        KeyError: chave desconhecida.
        FileNotFoundError: artefato ausente (rode src.models.ajustar_rf /
            src.models.treinar_svm / src.models.validar_cnn antes).
        ValueError: o JSON não traz `selecao_limiar.limiar`, ou o limiar não foi
            selecionado na validação — recusar é mais seguro que adivinhar.
    """
    if chave not in MODELOS_PRINCIPAIS:
        raise KeyError(f"Modelo '{chave}' desconhecido. "
                       f"Conhecidos: {sorted(MODELOS_PRINCIPAIS)}")
    arq_modelo, arq_json, rotulo, tipo = MODELOS_PRINCIPAIS[chave]

    caminho_modelo = raiz / "models" / arq_modelo
    caminho_json = raiz / "results" / "metricas" / arq_json
    for c in (caminho_modelo, caminho_json):
        if not c.exists():
            raise FileNotFoundError(
                f"Artefato ausente: {c}. Rode o pipeline antes "
                "(python -m src.models.ajustar_rf / src.models.treinar_svm / "
                "src.models.validar_cnn).")

    with open(caminho_json, encoding="utf-8") as f:
        metricas = json.load(f)

    # ---- AS DUAS GUARDAS, ANTES DO RAMO DE CARREGAMENTO ---------------------
    # A ORDEM É O PONTO: elas rodam para 'rf', 'svm' e 'cnn' igualmente, e não há
    # caminho de código que carregue um modelo do braço principal sem passar por
    # aqui. Foi por isso que a CNN entrou nesta função em vez de ganhar a sua.
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

    # ---- Só agora o carregamento ramifica, pelo TIPO do artefato ------------
    proveniencia = None
    if tipo == "torch":
        modelo, proveniencia = carregar_cnn_persistida(raiz, arq_modelo)
    else:
        modelo = joblib.load(caminho_modelo)

    carregado = {
        "chave": chave,
        "tipo": tipo,
        "modelo": modelo,
        "limiar": float(sel["limiar"]),
        "rotulo": rotulo,
        # Derivado do nome do JSON, e não montado com f"{chave}_tuned_principal":
        # para rf/svm dá exatamente o mesmo nome de sempre, e para a CNN dá
        # 'cnn_final_principal' em vez de um 'cnn_tuned_principal' que não
        # corresponde a arquivo nenhum. Fonte única: a tabela acima.
        "nome_arquivo": Path(arq_json).stem,
        "criterio_limiar": sel.get("criterio"),
        "origem_limiar": f"results/metricas/{arq_json} -> selecao_limiar.limiar",
        "metricas": metricas,
    }
    if proveniencia is not None:
        carregado["proveniencia"] = proveniencia
    return carregado


def scores_de(carregado: dict, X, lote: int = 256, dispositivo=None) -> np.ndarray:
    """Score de spoof na escala nativa de cada modelo.

    RF -> predict_proba(X)[:, 1] (probabilidade em [0,1]).
    SVM -> decision_function(X) (real, centrado em zero; o Pipeline já embute o
    StandardScaler, então X entra CRU, exatamente como no treino).
    CNN -> softmax(logits)[:, 1] = P(spoof), em [0,1].

    Misturar as escalas é o erro que invalidaria tudo em silêncio: aplicar um
    limiar de 0,65 a um decision_function classificaria quase tudo como bonafide
    sem erro nenhum aparecer. A escala da CNN casa com a do RF e difere da do
    SVM — mas o protocolo não depende disso: `selecionar_limiar` opera sobre
    `np.unique(scores)` e é agnóstico de escala.

    A ÚNICA ASSIMETRIA DE ASSINATURA ENTRE OS TRÊS MODELOS — e por que ela existe:
        para 'rf' e 'svm', `X` é a matriz (n, 44) de features, como sempre.
        Para 'cnn', `X` é um `torch.utils.data.Dataset` que devolve
        `(x, mascara, y)` — NÃO uma matriz.

        A razão é que a CNN precisa da MÁSCARA TEMPORAL além do tensor: sem ela o
        masked global pooling (P2) degenera em média sobre o padding, e o score
        deixa de ser o score do modelo avaliado. Havia duas saídas: devolver do
        índice também o vetor de `n_frames_validos` e aceitar `(X, mascara)`, ou
        aceitar o Dataset inteiro.

        ESCOLHIDO O DATASET, por três motivos: (1) ele já existe e é o MESMO
        `EspectrogramaDataset` do treino, então a normalização em tempo de carga
        é literalmente o mesmo código — um par `(X, mascara)` montado à mão
        convidaria a normalizar por fora, e normalizar por fora com as
        estatísticas erradas é a primeira armadilha do B4.7; (2) os 22.226
        espectrogramas somam ~2,9 GB em float32, e o Dataset lê do memmap sob
        demanda em vez de exigir tudo residente; (3) a máscara sai do índice pela
        mesma conta do treino, sem aritmética nova aqui.

        Quem chama com a chave 'cnn' e uma matriz recebe TypeError com esta
        explicação, em vez de um resultado silenciosamente errado.

    Args:
        carregado: retorno de `carregar_modelo_ajustado`.
        X: matriz (n, d) para rf/svm; Dataset `(x, mascara, y)` para cnn.
        lote: tamanho do lote da inferência da CNN. Não afeta o resultado — em
            `eval()` o BatchNorm usa as estatísticas acumuladas no treino, não as
            do lote —, só a memória usada. Ignorado por rf/svm.
        dispositivo: `torch.device` da inferência da CNN. None -> CUDA se houver.
            Ignorado por rf/svm.

    Returns:
        np.ndarray (n,) float64 de scores, NA ORDEM DE ENTRADA. Para a CNN isso
        depende de `shuffle=False`, que é fixado aqui e não é parâmetro: um
        `shuffle=True` desalinharia score e rótulo e produziria um EER ~0,5 que
        pareceria um problema da rede.
    """
    modelo = carregado["modelo"]
    if carregado["chave"] == "rf":
        # predizer_rf força n_jobs=1: com n_jobs=-1 a soma das árvores muda de
        # ordem entre execuções e o vetor não reproduz bit a bit (ver docstring
        # de predizer_rf). Num script de diagnóstico isso significaria uma
        # tabela que muda de casa decimal sem nada ter mudado.
        return predizer_rf(modelo, X)
    if carregado.get("tipo") == "torch":
        return _scores_cnn(modelo, X, lote=lote, dispositivo=dispositivo)
    # decision_function do SVC é single-thread e determinístico.
    return modelo.decision_function(X)


def fixar_precisao_fp32_cnn() -> dict:
    """Desliga o TF32 nas convoluções da GPU antes de qualquer score da CNN.

    POR QUE ISTO EXISTE (medido no B4.7, 20/09/2026 — RTX 5060 Ti, torch 2.11):
        por padrão o PyTorch deixa o cuDNN usar TF32 nas convoluções. TF32 tem
        mantissa de 10 bits: é uma precisão MENOR que float32, e o resultado
        passa a depender de qual kernel o cuDNN escolheu — que varia com o
        TAMANHO DO LOTE. Medido sobre os primeiros 256 exemplos da validação:

            TF32 ligado : |score(lote 256) - score(lote 1)| até 8,6e-05
            TF32 deslig.: |score(lote 256) - score(lote 1)| até 2,7e-06
            TF32 x FP32 no MESMO lote:                      até 2,3e-03

        O último número é o que decide. A distância mediana entre scores VIZINHOS
        na validação é ~3,5e-03 — ou seja, o erro do TF32 é da mesma ordem do
        espaçamento entre candidatos a limiar. Como `selecionar_limiar` usa
        `np.unique(scores)`, o TF32 mexe em `n_candidatos` e pode deslocar o
        limiar publicado, que é o número que o B5.1 aplica no teste lacrado.

        É o mesmo raciocínio de `predizer_rf` e o mesmo tipo de correção: custa
        alguns milissegundos, remove a dúvida, e o efeito colateral é um cálculo
        MAIS preciso, não menos.

    Chamada de dentro de `_scores_cnn` — e não deixada a cargo do chamador — pelo
    mesmo motivo que `predizer_rf` zera o `n_jobs` sozinho: uma garantia que
    depende de alguém lembrar não é garantia. Vale para o B4.7, para o B5.1 e
    para os diagnósticos, porque os três passam por aqui.

    Returns:
        dict com o estado ANTERIOR e o atual, para registro no artefato.
    """
    import torch

    anterior = {"cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
                "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32)}
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    return {
        "tf32_desligado": True,
        "estado_anterior": anterior,
        "motivo": ("TF32 (mantissa de 10 bits) faz o score depender do tamanho "
                   "do lote e difere do float32 em até 2,3e-03 — mesma ordem do "
                   "espaçamento entre scores vizinhos, logo mexe em "
                   "n_candidatos e pode deslocar o limiar publicado"),
        "medido_em": ("B4.7, 20/09/2026, RTX 5060 Ti / torch 2.11 / cuDNN 9.19, "
                      "sobre os 256 primeiros exemplos da validação"),
    }


def _scores_cnn(modelo, ds, lote: int = 256, dispositivo=None) -> np.ndarray:
    """P(spoof) da CNN sobre um Dataset, em `eval()`, `no_grad()` e `shuffle=False`.

    As três travas do Passo 1 do B4.7 estão aqui, e não no chamador, justamente
    para que nenhum chamador consiga esquecê-las. A quarta — FP32 em vez de TF32
    — entra por `fixar_precisao_fp32_cnn`, pela mesma razão.
    """
    import torch

    if not isinstance(ds, torch.utils.data.Dataset):
        raise TypeError(
            "scores_de(chave='cnn') espera um torch Dataset que devolva "
            "(x, mascara, y), não uma matriz de features — a CNN precisa da "
            "máscara temporal além do tensor (ver docstring de scores_de). Use "
            "src.models.validar_cnn.dataset_validacao para montá-lo.")

    fixar_precisao_fp32_cnn()
    if dispositivo is None:
        dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = modelo.to(dispositivo)
    modelo.eval()
    dl = torch.utils.data.DataLoader(ds, batch_size=lote, shuffle=False,
                                     num_workers=0)
    saida = []
    with torch.no_grad():
        for x, mascara, _y in dl:
            logits = modelo(x.to(dispositivo), mascara.to(dispositivo))
            saida.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(saida).astype(np.float64)


def entrada_de_validacao(carregado: dict, raiz: Path, validacao, cols: list[str]):
    """O que CADA modelo come, a partir da tabela de validação de um diagnóstico.

    Existe para que os scripts de diagnóstico não precisem de um `if chave ==
    'cnn'` cada um. A assimetria de entrada entre os três modelos (ver
    `scores_de`) é real, mas é DE UM TIPO SÓ, e o lugar de resolvê-la é aqui —
    duas cópias do mesmo `if` em dois scripts são duas chances de uma divergir.

    RF/SVM -> a matriz (n, 44) na ordem canônica de colunas.
    CNN    -> um Dataset de espectrogramas REORDENADO para casar, linha a linha,
              com `validacao["arquivo"]`. A reordenação é o ponto: a tabela dos
              diagnósticos vem de features.csv, cuja ordem de linhas não é a de
              indice_validacao.csv, e um score na ordem errada cruzaria com o
              metadado errado (o ataque, o codec) sem levantar exceção nenhuma.

    O import de `validar_cnn` é LOCAL de propósito: aquele módulo importa este no
    topo, e um import mútuo em tempo de carga quebraria os dois. Resolver dentro
    da função é o que mantém `modelos_ajustados` como a base da pilha.
    """
    if carregado.get("tipo") != "torch":
        return validacao[cols].values
    from .validar_cnn import dataset_para_arquivos

    return dataset_para_arquivos(raiz, validacao["arquivo"])


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
