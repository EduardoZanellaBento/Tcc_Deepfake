"""Avaliacao final no teste lacrado — EXECUCAO UNICA.

POR QUE ESTE SCRIPT NAO TREINA NADA:
    o teste e a prova final. Treinar aqui, ainda que "so para conferir",
    significaria que os hiperparametros e os limiares foram escolhidos com o
    teste em vista. Todos os modelos sao CARREGADOS, e o limiar de cada um vem
    do JSON da VALIDACAO — a guarda de conjunto em carregar_modelo_ajustado
    recusa qualquer outra procedencia.

GUARDA DE EXECUCAO UNICA:
    se results/metricas/teste_lacrado.json JA EXISTIR, o script aborta com erro
    explicito, exigindo --forcar e a razao por escrito. Nao e burocracia: uma
    reexecucao distraida sobrescreveria a unica medida honesta do trabalho, e o
    Bloco 3 ja perdeu um bloco de limitacao por edicao silenciosa de artefato
    gerado.

    Com --forcar, --razao "..." e obrigatorio e vai para o proprio artefato, em
    `reexecucoes: [{data, razao, md5_do_artefato_sobrescrito}]`, acumulando as
    anteriores — quem ler o JSON ve quantas vezes o teste foi aberto e por que.

AS QUATRO LINHAS (B5.1, regra 4):
    RF principal, RF referencia, SVM principal, CNN principal. O braco de
    referencia NAO e concorrente direto de SVM/CNN — ele quantifica o custo da
    subamostragem, e a ressalva sai gravada na linha dele.

    O braco de referencia nao esta em MODELOS_PRINCIPAIS, e nao deve estar: o
    nome da tabela diz o que ela e. Mas as DUAS guardas de
    carregar_modelo_ajustado (limiar presente; limiar vindo da validacao) tem de
    valer para ele tambem. Em vez de copia-las aqui — uma segunda copia das
    guardas e exatamente a divergencia que aquele modulo existe para impedir —,
    o braco e registrado na tabela SO durante o carregamento (`patch.dict`, que
    restaura o dict ao sair) e passa pela MESMA funcao. modelos_ajustados.py fica
    intocado: o md5 dele esta gravado em cnn_final_principal.json como evidencia
    do isolamento do teste no B4.7.

A ORDEM DOS PASSOS E A PROTECAO DO USO UNICO:
    1. guarda de execucao unica — antes de qualquer outra coisa;
    2. auditoria por LEITURA DO CODIGO (AST): nenhuma chamada a
       selecionar_limiar e nenhuma chamada de treino (fit, backward, step...)
       neste script nem nos modulos que ele chama para pontuar e avaliar;
    3. hashes dos artefatos congelados conferidos contra os quatro JSONs;
    4. PRE-VOO NA VALIDACAO: cada modelo carregado, com o seu limiar, tem de
       REPRODUZIR EXATAMENTE o JSON de validacao dele — as metricas de avaliar(),
       a matriz de confusao, o ROC-AUC e o numero de scores distintos
       (`selecao_limiar.n_candidatos`, que e uma contagem e por isso enxerga o
       ultimo bit: foi ele que denunciou o n_jobs do RF e o TF32 da CNN). Isso
       prova que o par (modelo, limiar) aplicado ao teste e o par publicado. Se
       qualquer campo divergir, o script aborta SEM ABRIR O TESTE;
    5. so entao o teste: scores, conferencias de alinhamento da CNN, avaliar(),
       gravacao. Nada depois do passo 5 depende de decisao.

MODO ENSAIO (--ensaio-na-validacao DIR):
    roda o script INTEIRO trocando o conjunto avaliado pela validacao e gravando
    em DIR, fora de results/. Existe porque "escreva o script inteiro, revise-o,
    e so entao rode" pede um jeito de exercitar cada linha sem gastar o uso
    unico. O ensaio tem resposta conhecida: o JSON dele tem de trazer delta
    validacao-teste IGUAL A ZERO nos quatro modelos. Os caminhos do teste e da
    validacao diferem so nas constantes de ESPECTROGRAMAS; todo o resto e o
    mesmo codigo.

POR QUE A CNN E PONTUADA NA ORDEM DO SEU PROPRIO INDICE:
    o limiar da CNN e um score observado (np.unique), isto e, e EXATAMENTE o
    score de um exemplo da validacao. Em FP32 a composicao do lote mexe no score
    na setima casa (3,6e-07 medido no B4.7), e um exemplo sobre o limiar pode
    trocar de lado com isso. Pontuar na ordem do indice, em lotes de
    LOTE_INFERENCIA, reproduz a MESMA composicao de lotes do B4.7 — e e o que
    permite exigir igualdade exata no pre-voo. A reordenacao para a ordem da
    tabela (a de features.csv, a mesma de RF e SVM) acontece depois, sobre o
    vetor de scores, por `arquivo` — nao sobre o Dataset.

SAIDAS:
    results/metricas/teste_lacrado.json
    results/metricas/scores_teste_lacrado.csv   (scores dos 4 modelos no teste)
    results/metricas/scores_validacao.csv       (os mesmos, na validacao)
    results/figuras/matriz_confusao_teste_lacrado_{modelo}.png  (quatro)

    Os dois CSVs de scores existem para que o B5.2 (bootstrap pareado, curvas)
    NUNCA precise abrir o teste de novo: o teste e pontuado uma vez, aqui.

Rode a partir da raiz:
    python -m scripts.avaliar_teste_lacrado --ensaio-na-validacao <dir fora de results/>
    python -m scripts.avaliar_teste_lacrado            # o uso unico
"""

import argparse
import ast
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score

from src.data.split import carregar_dados_split, colunas_features
from src.models import modelos_ajustados
from src.models.avaliacao import (REGRA_DECISAO, aplicar_limiar, avaliar,
                                  plotar_matriz_confusao, predizer_rf)
from src.models.definir_cnn import _chamadas_de_limiar
from src.models.modelos_ajustados import (carregar_modelo_ajustado,
                                          fixar_precisao_fp32_cnn,
                                          hashes_congelados, scores_de)
from src.models.tempo import ambiente
# O Dataset e as constantes da inferencia vem dos modulos do B4.6/B4.7, nao
# reescritos: normalizacao em tempo de carga, mascara e lote TEM de ser os que
# produziram o limiar. Importar nao altera o md5 de nenhum deles.
from src.models.treinar_cnn import EspectrogramaDataset
from src.models.validar_cnn import (ESTAGIO_EXIGIDO, LOTE_INFERENCIA,
                                    NORMALIZACAO)
from src.utils.config import carregar_config
from src.utils.seeds import fixar_seeds_torch
from src.utils.serializacao import json_seguro

RAIZ = Path(__file__).resolve().parents[1]
DIR_MET = RAIZ / "results" / "metricas"
DIR_FIG = RAIZ / "results" / "figuras"
SAIDA = "teste_lacrado.json"

CONJUNTO_LACRADO = "teste"

# Ordem das linhas em todas as saidas. 'rf', 'svm' e 'cnn' sao as chaves de
# MODELOS_PRINCIPAIS; 'rf_ref' so existe durante o carregamento (ver docstring).
ORDEM = ("rf", "rf_ref", "svm", "cnn")
BRACO_REFERENCIA = {
    "rf_ref": ("rf_tuned_referencia.joblib", "rf_tuned_referencia.json",
               "RF ajustado (braço de referência, treino completo)", "sklearn"),
}
RESSALVA_REFERENCIA = (
    "NÃO é concorrente direto de SVM e CNN: foi treinado no treino completo "
    "(103.723) e existe para quantificar o custo da subamostragem de 30k, "
    "imposta pela complexidade O(n²)–O(n³) do SVM-RBF. A comparação entre "
    "modelos é a do braço principal, em que os três viram as mesmas 30.000 "
    "amostras.")

ESCALA = {"rf": "probabilidade [0,1] (predict_proba)",
          "rf_ref": "probabilidade [0,1] (predict_proba)",
          "svm": "decision_function (real, centrado em zero)",
          "cnn": "probabilidade [0,1] (softmax(logits)[:, 1] = P(spoof))"}

# Titulo da matriz de confusao: o MESMO molde das de validacao
# (scripts/replotar_matrizes_confusao.py), trocando só o conjunto.
TITULO = {"rf": "RF ajustado (braço principal)",
          "rf_ref": "RF ajustado (braço referencia)",
          "svm": "SVM RBF ajustado (braço principal)",
          "cnn": "CNN final (braço principal)"}

# Espectrogramas da CNN por conjunto. 'validacao' só é usado pelo pre-voo e
# pelo ensaio; o uso real le o conjunto lacrado.
ESPECTROGRAMAS = {
    "teste": ("data/espectrogramas/indice_teste.csv",
              "data/espectrogramas/teste.npy"),
    "validacao": ("data/espectrogramas/indice_validacao.csv",
                  "data/espectrogramas/validacao.npy"),
}
META_ESPECTROGRAMAS = "data/espectrogramas/espectrogramas.meta.json"
N_ESPERADO = {"teste": 22227, "validacao": 22226}

# Campos de avaliar() que o pre-voo exige iguais ao JSON de validacao.
CAMPOS_PRE_VOO = ("acuracia", "precisao_spoof", "recall_spoof", "f1_spoof",
                  "precisao_bonafide", "recall_bonafide", "f1_bonafide",
                  "f1_macro", "eer", "limiar_eer")
# A mesma tolerancia da guarda de reproducao: absorve serializacao, nao
# absorve diferenca de resultado.
TOL_FLOAT = 1e-9

# Auditoria por leitura do codigo: os modulos cujas funcoes este script CHAMA
# para pontuar e avaliar. De treinar_cnn.py e validar_cnn.py só se importam a
# classe do Dataset e constantes — nenhuma funcao deles roda aqui.
MODULOS_AUDITADOS = ("scripts/avaliar_teste_lacrado.py",
                     "src/models/modelos_ajustados.py",
                     "src/models/avaliacao.py")
CHAMADAS_DE_TREINO = ("fit", "fit_transform", "partial_fit", "backward",
                      "step", "zero_grad")


def _md5(caminho: Path) -> str:
    return hashlib.md5(caminho.read_bytes()).hexdigest()


# =============================================================================
# 1. Guarda de execucao unica e auditoria do codigo
# =============================================================================
def guarda_execucao_unica(args) -> list:
    """Devolve a lista de reexecucoes a gravar, ou encerra o processo.

    Roda antes de qualquer dado ser lido e de qualquer modelo ser carregado: se o
    teste já foi avaliado, nada mais deve acontecer.
    """
    caminho = DIR_MET / SAIDA
    if args.razao is not None and not args.forcar:
        sys.exit("ERRO: --razao só tem sentido junto com --forcar.")
    if not caminho.exists():
        if args.forcar:
            print("AVISO: --forcar ignorado — não há execução anterior.")
        return []
    if not args.forcar:
        sys.exit(
            f"ERRO: {caminho.relative_to(RAIZ)} JÁ EXISTE. O teste lacrado é "
            "avaliado UMA ÚNICA VEZ; reexecutar sobrescreveria a única medida "
            "honesta de generalização do trabalho. Se houver erro grave que "
            "exija reexecução, use --forcar --razao \"<o erro, por escrito>\" — "
            "a razão fica gravada no próprio artefato.")
    if not args.razao or not args.razao.strip():
        sys.exit("ERRO: --forcar exige --razao \"...\" não vazia.")
    with open(caminho, encoding="utf-8") as f:
        anteriores = json.load(f).get("reexecucoes", [])
    return anteriores + [{
        "data": datetime.now().isoformat(timespec="seconds"),
        "razao": args.razao.strip(),
        "md5_do_artefato_sobrescrito": _md5(caminho),
    }]


def auditar_codigo() -> dict:
    """Nenhum limiar recalculado, nenhum modelo treinado — por LEITURA DO CÓDIGO.

    Mesmo método do B4.7 (AST), pela mesma razão: uma frase no JSON dizendo
    «nenhum modelo foi treinado» não é evidência; a árvore sintática dos
    módulos que rodaram, com o md5 de cada um, é.
    """
    registro, violacoes = {}, []
    for rel in MODULOS_AUDITADOS:
        caminho = RAIZ / rel
        arvore = ast.parse(caminho.read_text(encoding="utf-8"))
        limiar = _chamadas_de_limiar(arvore)
        treino = []
        for no in ast.walk(arvore):
            if isinstance(no, ast.Call):
                alvo = (no.func.id if isinstance(no.func, ast.Name)
                        else getattr(no.func, "attr", None))
                if alvo in CHAMADAS_DE_TREINO:
                    treino.append({"linha": no.lineno, "chamada": alvo})
        registro[rel] = {"md5": _md5(caminho),
                         "chamadas_selecionar_limiar": limiar,
                         "chamadas_de_treino": treino}
        if limiar or treino:
            violacoes.append(rel)
    if violacoes:
        raise RuntimeError(f"auditoria do código falhou em {violacoes}: "
                           f"{json.dumps(registro, indent=1)}")
    return {
        "metodo": "AST dos módulos que pontuam e avaliam (mesmo método do B4.7)",
        "chamadas_procuradas": ["selecionar_limiar", *CHAMADAS_DE_TREINO],
        "arquivos": registro,
        "resultado": ("nenhuma chamada a selecionar_limiar e nenhuma chamada de "
                      "treino nos módulos do caminho"),
    }


# =============================================================================
# 2. Modelos e dados
# =============================================================================
def carregar_bracos() -> dict:
    """Os quatro braços, todos pela MESMA carregar_modelo_ajustado."""
    carregados = {c: carregar_modelo_ajustado(RAIZ, c) for c in ("rf", "svm", "cnn")}
    with patch.dict(modelos_ajustados.MODELOS_PRINCIPAIS, BRACO_REFERENCIA):
        carregados["rf_ref"] = carregar_modelo_ajustado(RAIZ, "rf_ref")
    assert "rf_ref" not in modelos_ajustados.MODELOS_PRINCIPAIS, (
        "patch.dict não restaurou MODELOS_PRINCIPAIS")
    return {c: carregados[c] for c in ORDEM}


def conferir_hashes(carregados: dict, hashes: dict) -> None:
    """Os artefatos congelados em disco são os que treinaram os quatro modelos."""
    for c, carregado in carregados.items():
        js = carregado["metricas"]
        gravados = {"features": js["hash_md5_features_csv"],
                    "split": js["hash_md5_split_csv"],
                    "subamostra": js["hash_md5_subamostra_csv"]}
        if gravados != hashes:
            raise RuntimeError(
                f"hashes em disco {hashes} divergem dos gravados em "
                f"{carregado['nome_arquivo']}.json {gravados} — os artefatos "
                "congelados mudaram desde o treino; a avaliação seria inválida.")


def dataset_cnn(conjunto: str, tabela: pd.DataFrame,
                hash_norm_esperado: str) -> tuple:
    """Dataset da CNN NA ORDEM DO ÍNDICE + a permutação índice -> tabela.

    Cinco conferências antes de a rede ver um tensor: md5 do índice contra o
    registrado no B4.2; n e conjunto de arquivos iguais aos da tabela (split);
    normalização do estágio 30k com o md5 que o B4.7 registrou; forma do memmap.
    A quinta — rótulo e n_frames_validos do índice, depois de reordenados,
    iguais aos de features.csv/split.csv — é a que prova que a permutação casa
    cada score com a linha certa da tabela.
    """
    rel_indice, rel_memmap = ESPECTROGRAMAS[conjunto]
    indice = pd.read_csv(RAIZ / rel_indice)
    with open(RAIZ / META_ESPECTROGRAMAS, encoding="utf-8") as f:
        meta = json.load(f)
    md5_indice = _md5(RAIZ / rel_indice)
    esperado = meta["hash_md5_indice_por_conjunto"][Path(rel_indice).name]
    if md5_indice != esperado:
        raise RuntimeError(f"{rel_indice}: md5 {md5_indice} != {esperado} "
                           f"registrado em {META_ESPECTROGRAMAS}")
    if len(indice) != N_ESPERADO[conjunto] or len(tabela) != N_ESPERADO[conjunto]:
        raise RuntimeError(f"{conjunto}: índice {len(indice)}, tabela "
                           f"{len(tabela)}, esperado {N_ESPERADO[conjunto]}")
    if set(indice["arquivo"]) != set(tabela["arquivo"]):
        raise RuntimeError(f"{rel_indice} e split.csv ({conjunto}) não têm os "
                           "mesmos arquivos")

    with open(RAIZ / NORMALIZACAO, encoding="utf-8") as f:
        norm = json.load(f)
    md5_norm = _md5(RAIZ / NORMALIZACAO)
    if norm["estagio"] != ESTAGIO_EXIGIDO or md5_norm != hash_norm_esperado:
        raise RuntimeError(f"{NORMALIZACAO}: estágio {norm['estagio']!r}, md5 "
                           f"{md5_norm} — esperado {ESTAGIO_EXIGIDO!r} e o md5 "
                           "gravado em cnn_final_principal.json")
    media = np.array(norm["media_por_mel"], dtype=np.float64)
    desvio = np.array(norm["desvio_por_mel"], dtype=np.float64)

    ds = EspectrogramaDataset(RAIZ / rel_memmap, indice, media, desvio)
    if ds.mm.shape[1:] != (128, 251) or ds.mm.shape[0] <= indice["linha"].max():
        raise RuntimeError(f"{rel_memmap}: forma {ds.mm.shape} incompatível")

    pos = pd.Series(np.arange(len(indice)), index=indice["arquivo"].to_numpy())
    ordem = pos.loc[tabela["arquivo"].to_numpy()].to_numpy()
    rot = indice["classe_binaria"].to_numpy()[ordem]
    nv = indice["n_frames_validos"].to_numpy()[ordem]
    n_rot = int((rot != tabela["classe_binaria"].to_numpy()).sum())
    n_nv = int((nv != tabela["n_frames_validos"].to_numpy()).sum())
    if n_rot or n_nv:
        raise RuntimeError(f"{conjunto}: depois de reordenar, {n_rot} rótulos e "
                           f"{n_nv} n_frames_validos divergem da tabela — a "
                           "permutação não casa score com linha")
    return ds, ordem, {
        "indice": rel_indice, "md5_indice": md5_indice,
        "memmap": rel_memmap, "forma_memmap": list(ds.mm.shape),
        "normalizacao": f"{NORMALIZACAO} (estagio {norm['estagio']})",
        "md5_normalizacao": md5_norm,
        "n_rotulos_divergentes_apos_reordenar": n_rot,
        "n_frames_validos_divergentes_apos_reordenar": n_nv,
    }


def conferencia_posicional(modelo, ds, s_indice: np.ndarray) -> dict:
    """Sete posições reprocessadas isoladamente reproduzem o vetor em lote.

    Mesmo critério do B4.7: o desvio tem de ser >= 100x menor que o de um
    deslocamento de posição. Uma diferença: o controle usa o vizinho mais
    próximo cujo score DIFERE do da posição. Com ~23% dos scores saturados em
    exatamente 1,0, o vizinho imediato pode ter o mesmo valor, e um controle
    zero reprovaria um alinhamento correto — no conjunto que não pode ser
    reaberto.
    """
    n = len(s_indice)
    posicoes = [0, 1, n // 4, n // 2, 3 * n // 4, n - 2, n - 1]
    dispositivo = next(modelo.parameters()).device
    desvios, controles = [], []
    modelo.eval()
    with torch.no_grad():
        for i in posicoes:
            x, mascara, _ = ds[i]
            logit = modelo(x[None].to(dispositivo), mascara[None].to(dispositivo))
            s = float(torch.softmax(logit, dim=1)[0, 1])
            desvios.append(abs(s - float(s_indice[i])))
            vizinhos = [j for j in list(range(i + 1, n)) + list(range(i - 1, -1, -1))
                        if s_indice[j] != s_indice[i]]
            controles.append(abs(s - float(s_indice[vizinhos[0]])))
    maior = float(max(desvios))
    controle = float(np.median(controles))
    margem = controle / maior if maior > 0 else float("inf")
    if margem < 100:
        raise RuntimeError(f"conferência posicional: desvio {maior:.3e} contra "
                           f"controle {controle:.3e} (margem {margem:.1f}x < 100x)")
    return {"posicoes": posicoes, "maior_desvio_absoluto": maior,
            "controle_deslocamento": controle,
            "margem": round(margem, 1) if np.isfinite(margem) else "inf",
            "margem_exigida": 100}


def pontuar(carregados: dict, tabela: pd.DataFrame, cols: list[str],
            conjunto: str) -> tuple[dict, dict]:
    """Scores dos quatro modelos, TODOS na ordem das linhas de `tabela`."""
    X = tabela[cols].values
    scores, registro_cnn = {}, None
    for c, carregado in carregados.items():
        if carregado["tipo"] == "torch":
            ds, ordem, registro_cnn = dataset_cnn(
                conjunto, tabela,
                carregado["metricas"]["hash_md5_normalizacao_30k_json"])
            s_indice = scores_de(carregado, ds, lote=LOTE_INFERENCIA)
            registro_cnn["conferencia_posicional"] = conferencia_posicional(
                carregado["modelo"], ds, s_indice)
            registro_cnn["lote_inferencia"] = LOTE_INFERENCIA
            registro_cnn["ordem_de_pontuacao"] = (
                "a do índice (mesma composição de lotes do B4.7); o vetor é "
                "reordenado depois, por arquivo, para a ordem da tabela")
            scores[c] = s_indice[ordem]
        elif c == "rf_ref":
            # Exatamente o ramo de scores_de para 'rf' (predict_proba com
            # n_jobs=1). scores_de despacha pela chave, e 'rf_ref' não é 'rf'.
            scores[c] = predizer_rf(carregado["modelo"], X)
        else:
            scores[c] = scores_de(carregado, X)
        if len(scores[c]) != len(tabela):
            raise RuntimeError(f"{c}: {len(scores[c])} scores para "
                               f"{len(tabela)} linhas")
    return scores, registro_cnn


# =============================================================================
# 3. Pre-voo e avaliacao
# =============================================================================
def pre_voo(carregados: dict, y: np.ndarray, scores: dict) -> dict:
    """Cada modelo, com o seu limiar, reproduz EXATAMENTE o seu JSON de validação."""
    registro, falhas = {}, []
    for c, carregado in carregados.items():
        js, limiar = carregado["metricas"], carregado["limiar"]
        s = scores[c]
        m = avaliar(y, s, nome=carregado["nome_arquivo"], limiar=limiar)
        div = {k: [js[k], m[k]] for k in CAMPOS_PRE_VOO
               if abs(m[k] - js[k]) > TOL_FLOAT}
        cm = confusion_matrix(y, aplicar_limiar(s, limiar), labels=[0, 1]).tolist()
        if cm != js["matriz_confusao"]:
            div["matriz_confusao"] = [js["matriz_confusao"], cm]
        auc = round(float(roc_auc_score(y, s)), 4)
        if auc != js["roc_auc_validacao"]:
            div["roc_auc_validacao"] = [js["roc_auc_validacao"], auc]
        n_dist = int(np.unique(s).size)
        if n_dist != js["selecao_limiar"]["n_candidatos"]:
            div["selecao_limiar.n_candidatos"] = [
                js["selecao_limiar"]["n_candidatos"], n_dist]
        if len(y) != js["n_validacao"]:
            div["n_validacao"] = [js["n_validacao"], len(y)]
        registro[carregado["nome_arquivo"]] = {
            "campos_conferidos": [*CAMPOS_PRE_VOO, "matriz_confusao",
                                  "roc_auc_validacao",
                                  "selecao_limiar.n_candidatos", "n_validacao"],
            "divergencias": div,
            "reproduziu": not div,
        }
        estado = "OK " if not div else "!! "
        print(f"  [{estado}] {carregado['nome_arquivo']}: f1_macro "
              f"{m['f1_macro']:.4f} (JSON {js['f1_macro']:.4f}) | "
              f"{n_dist} scores distintos (JSON "
              f"{js['selecao_limiar']['n_candidatos']})")
        if div:
            falhas.append(carregado["nome_arquivo"])
    if falhas:
        raise RuntimeError(
            f"PRE-VOO FALHOU em {falhas}: o modelo carregado não reproduz o JSON "
            f"de validação — {json.dumps(registro, indent=1, default=json_seguro)}"
            "\nO TESTE NÃO FOI ABERTO. Investigue antes de qualquer outra coisa.")
    return {
        "o_que_prova": ("o par (modelo, limiar) aplicado ao teste é o par "
                        "publicado: cada modelo carregado reproduz, na "
                        "validação, as métricas, a matriz, o ROC-AUC e o número "
                        "de scores distintos do seu JSON"),
        "tolerancia_float": TOL_FLOAT,
        "resultado": "os quatro reproduziram — só então o teste foi aberto",
        "por_modelo": registro,
    }


def avaliar_conjunto(carregados: dict, y: np.ndarray, scores: dict) -> dict:
    """avaliar() no conjunto avaliado, com o limiar lido do JSON de validação."""
    modelos = {}
    for c, carregado in carregados.items():
        js, limiar, s = carregado["metricas"], carregado["limiar"], scores[c]
        m = avaliar(y, s, nome=carregado["nome_arquivo"], limiar=limiar)
        linha = {
            "limiar": limiar,
            "origem_limiar": carregado["origem_limiar"],
            "conjunto_do_limiar": js["selecao_limiar"]["conjunto"],
            "criterio_limiar": carregado["criterio_limiar"],
            **m,
            "braco": js["braco"],
            "n_treino": js["n_treino"],
            "n_avaliado": int(len(y)),
            "matriz_confusao": confusion_matrix(
                y, aplicar_limiar(s, limiar), labels=[0, 1]).tolist(),
            "roc_auc": round(float(roc_auc_score(y, s)), 4),
            "escala_score": ESCALA[c],
        }
        if c == "rf_ref":
            linha["ressalva"] = RESSALVA_REFERENCIA
        modelos[carregado["nome_arquivo"]] = linha
    return modelos


# =============================================================================
# 4. Execucao
# =============================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--forcar", action="store_true",
                    help="reexecuta mesmo com teste_lacrado.json existente "
                         "(exige --razao)")
    ap.add_argument("--razao", default=None,
                    help="por que o teste precisa ser reaberto (gravada no JSON)")
    ap.add_argument("--ensaio-na-validacao", metavar="DIR", default=None,
                    help="roda o script inteiro sobre a VALIDAÇÃO, gravando em "
                         "DIR (fora de results/); não toca no teste")
    args = ap.parse_args()

    ensaio = args.ensaio_na_validacao is not None
    if ensaio:
        dir_saida = Path(args.ensaio_na_validacao).resolve()
        if (RAIZ / "results") in [dir_saida, *dir_saida.parents]:
            sys.exit("ERRO: o ensaio grava FORA de results/ — escolha outro DIR.")
        dir_met = dir_fig = dir_saida
        conjunto = "validacao"
        reexecucoes = []
        print(f"*** ENSAIO: o conjunto avaliado é a VALIDAÇÃO; saída em {dir_saida} ***")
    else:
        reexecucoes = guarda_execucao_unica(args)
        dir_met, dir_fig = DIR_MET, DIR_FIG
        conjunto = CONJUNTO_LACRADO
    dir_met.mkdir(parents=True, exist_ok=True)
    dir_fig.mkdir(parents=True, exist_ok=True)

    cfg = carregar_config(RAIZ)
    # Antes de qualquer chamada CUDA: CUBLAS_WORKSPACE_CONFIG e o modo
    # determinístico só valem se vierem primeiro. TF32 desligado — é o regime em
    # que o limiar da CNN foi selecionado.
    semente = fixar_seeds_torch(int(cfg["semente"]), estrito=True)
    precisao = fixar_precisao_fp32_cnn()
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"

    print("Auditando o código (AST)...")
    auditoria = auditar_codigo()
    print(f"  OK — {auditoria['resultado']}")

    hashes = hashes_congelados(RAIZ, cfg["experimento"]["caminho_subamostra"])
    carregados = carregar_bracos()
    conferir_hashes(carregados, hashes)
    print("Modelos carregados; limiares lidos dos JSONs de VALIDAÇÃO:")
    for carregado in carregados.values():
        print(f"  {carregado['rotulo']:<52} limiar {carregado['limiar']: .6f}  "
              f"<- {carregado['origem_limiar']}")

    df = carregar_dados_split(RAIZ)
    cols = colunas_features(df)
    validacao = df[df["conjunto"] == "validacao"].reset_index(drop=True)

    # ---- PRE-VOO: nada do conjunto lacrado foi lido até aqui -----------------
    print(f"\nPré-voo na validação ({len(validacao)}), dispositivo {dispositivo}:")
    scores_va, _ = pontuar(carregados, validacao, cols, "validacao")
    y_va = validacao["classe_binaria"].to_numpy().astype(int)
    registro_pre_voo = pre_voo(carregados, y_va, scores_va)

    # ---- O CONJUNTO AVALIADO (o teste lacrado, salvo no ensaio) --------------
    avaliado = df[df["conjunto"] == conjunto].reset_index(drop=True)
    print(f"\nAvaliando '{conjunto}' ({len(avaliado)} linhas)...")
    scores_av, registro_cnn = pontuar(carregados, avaliado, cols, conjunto)
    y_av = avaliado["classe_binaria"].to_numpy().astype(int)
    modelos = avaliar_conjunto(carregados, y_av, scores_av)

    delta_f1, delta_eer = {}, {}
    for carregado in carregados.values():
        nome, js = carregado["nome_arquivo"], carregado["metricas"]
        delta_f1[nome] = modelos[nome]["f1_macro"] - js["f1_macro"]
        delta_eer[nome] = modelos[nome]["eer"] - js["eer"]

    # ---- Saídas --------------------------------------------------------------
    sufixo = "teste_lacrado" if not ensaio else "ensaio_validacao"
    nomes = [carregados[c]["nome_arquivo"] for c in ORDEM]
    for rotulo, tabela, sc in (("validacao", validacao, scores_va),
                               (sufixo, avaliado, scores_av)):
        pd.DataFrame({"arquivo": tabela["arquivo"],
                      "classe_binaria": tabela["classe_binaria"],
                      **{carregados[c]["nome_arquivo"]: sc[c] for c in ORDEM}}
                     ).to_csv(dir_met / f"scores_{rotulo}.csv", index=False,
                              float_format="%.17g")

    figuras = []
    for c in ORDEM:
        nome = carregados[c]["nome_arquivo"]
        caminho = dir_fig / f"matriz_confusao_{sufixo}_{nome}.png"
        plotar_matriz_confusao(
            modelos[nome]["matriz_confusao"], caminho,
            f"{TITULO[c]} — {'teste lacrado' if not ensaio else 'ENSAIO na validação'}, "
            f"limiar {carregados[c]['limiar']:.2f}")
        figuras.append(str(caminho.relative_to(RAIZ)) if not ensaio else str(caminho))

    amb = ambiente(n_jobs_inferencia=1)
    amb.update({"torch": torch.__version__, "cuda_versao": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "gpu": (torch.cuda.get_device_name(0)
                        if torch.cuda.is_available() else None),
                "dispositivo_cnn": dispositivo})

    saida = {
        "conjunto": conjunto if not ensaio else "validacao (ENSAIO — não é o teste)",
        "n": int(len(avaliado)),
        "ensaio": ensaio,
        "data_execucao": datetime.now().isoformat(timespec="seconds"),
        "protocolo": {
            "regra": REGRA_DECISAO,
            "origem_dos_limiares": ("JSON de validacao de cada modelo, campo "
                                    "selecao_limiar.limiar"),
            "nenhum_modelo_treinado_aqui": True,
            "execucao": "unica",
            "guarda_de_conjunto": ("carregar_modelo_ajustado recusa JSON cujo "
                                   "selecao_limiar.conjunto != 'validacao'; os "
                                   "quatro braços passaram por ela"),
        },
        "modelos": modelos,
        "delta_validacao_teste": {
            "nota": ("diferenca f1_macro(teste) - f1_macro(validacao) por modelo; "
                     "|delta| pequeno indica que a validacao nao foi sobreajustada "
                     "pela selecao de limiar"),
            **delta_f1,
        },
        "delta_eer_validacao_teste": {
            "nota": ("diferenca eer(teste) - eer(validacao). O EER não depende do "
                     "limiar: este delta mede só a diferença de dificuldade entre "
                     "os dois conjuntos. O de f1_macro soma a isso o efeito de "
                     "transportar o limiar — ler os dois lado a lado separa as "
                     "duas causas"),
            **delta_eer,
        },
        "conferencia_pre_teste": registro_pre_voo,
        "cnn": {
            "dados": registro_cnn,
            "determinismo": {"fixar_seeds_torch": True, "estrito": True,
                             "semente": semente, "num_workers": 0},
            "precisao_numerica": precisao,
        },
        "auditoria_do_codigo": auditoria,
        "hashes": {**hashes,
                   "indice_espectrogramas": registro_cnn["md5_indice"],
                   "normalizacao_30k": registro_cnn["md5_normalizacao"]},
        "saidas": {
            "scores": ["scores_validacao.csv", f"scores_{sufixo}.csv"],
            "nota_scores": ("scores dos quatro modelos, na ordem de features.csv "
                            "x split.csv; é deles que o B5.2 tira o bootstrap "
                            "pareado — o teste não é pontuado de novo"),
            "matrizes_confusao": figuras,
        },
        "reexecucoes": reexecucoes,
        "ambiente": amb,
    }
    with open(dir_met / (SAIDA if not ensaio else f"{sufixo}.json"), "w",
              encoding="utf-8") as f:
        json.dump(saida, f, indent=2, ensure_ascii=False, default=json_seguro)

    # ---- Resumo --------------------------------------------------------------
    print(f"\n{'modelo':<22} {'limiar':>10} {'f1_macro':>9} {'Δ val':>8} "
          f"{'EER':>7} {'Δ val':>8} {'AUC':>7} {'rec_bona':>8}")
    for nome in nomes:
        m = modelos[nome]
        print(f"{nome:<22} {m['limiar']:>10.4f} {m['f1_macro']:>9.4f} "
              f"{delta_f1[nome]:>+8.4f} {m['eer']:>7.4f} {delta_eer[nome]:>+8.4f} "
              f"{m['roc_auc']:>7.4f} {m['recall_bonafide']:>8.4f}")
    print(f"\nSalvo: {dir_met / (SAIDA if not ensaio else sufixo + '.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
