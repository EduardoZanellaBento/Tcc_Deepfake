"""
B5.2 — Comparação experimental fechada: RF × SVM × CNN
=======================================================

O QUE ESTE SCRIPT FAZ, E O QUE ELE NÃO FAZ:
    NÃO carrega modelo e NÃO abre o conjunto de teste. Tudo o que ele usa já
    está em disco:
      - os quatro JSONs de validação (rf_tuned_principal, rf_tuned_referencia,
        svm_tuned_principal, cnn_final_principal) e o teste_lacrado.json do B5.1;
      - os vetores de score que o B5.1 gravou (scores_validacao.csv e
        scores_teste_lacrado.csv) — o teste foi pontuado UMA vez, lá;
      - tempo_pipeline_completo.json, estabilidade_rf_svm.json e os resumos dos
        diagnósticos por ataque e por codec.

A TABELA É MONTADA POR LEITURA, NUNCA DIGITADA:
    comparacao_final.csv e COMPARACAO_FINAL.md (a versão para o texto) saem dos
    MESMOS dicionários lidos dos JSONs. Tabela do texto e artefato não têm como
    divergir. E, antes de montar qualquer coisa, os vetores de score são
    conferidos contra os JSONs: f1_macro, EER e matriz de confusão recalculados
    a partir de cada vetor têm de bater EXATAMENTE com o número publicado — é o
    que garante que o bootstrap abaixo reamostra os scores que produziram a
    tabela, e não outros.

O TESTE ESTATÍSTICO — bootstrap PAREADO, molde de scripts/estabilidade_modelos.py:
    os três modelos são avaliados exatamente nas mesmas 22.226 (validação) /
    22.227 (teste) amostras, então os erros deles são correlacionados. A cada
    uma das 1.000 reamostragens, UM vetor de índices é sorteado e aplicado aos
    scores de TODOS os modelos ao mesmo tempo; guardam-se as diferenças
    SVM−RF, CNN−RF e CNN−SVM de f1_macro e de EER. O limiar de cada modelo fica
    FIXO no valor do protocolo (reamostra-se a avaliação, não a seleção).

    Mesma semente, mesmo número de reamostragens e mesma ordem de linhas do
    estabilidade_rf_svm.json. Consequência verificável, e verificada aqui: o
    par SVM−RF na validação REPRODUZ o número do Bloco 3 casa por casa
    (+0,0762 [0,0663; 0,0856]; −0,0466 [−0,0560; −0,0377]; 100%). Se não
    reproduzir, o script aborta — ou a ordem das linhas ou os scores mudaram.

    Leitura: IC95 da diferença sem zero => diferença real neste protocolo. IC95
    com zero => «não distinguíveis neste protocolo» — nunca «empataram», nunca
    «X é ligeiramente melhor». A frase sai DERIVADA do IC, não escrita à mão.

    O braço de referência entra no mesmo laço (mesmo vetor de índices) só para
    dar IC ao CUSTO DA SUBAMOSTRAGEM (referência − principal). Ele não entra em
    nenhum dos três pares da comparação: não é concorrente de SVM e CNN.

HEURÍSTICAS, ROTULADAS COMO HEURÍSTICAS:
    IC individual de cada modelo, variância de treino do RF entre 5 sementes
    (lida de estabilidade_rf_svm.json) e a comparação «diferença × maior
    dispersão individual». Úteis como contexto; NÃO são o teste — ignoram a
    correlação entre os erros. Ficam num bloco próprio, com esse rótulo.

POR QUE A LEITURA É GERADA PELO CÓDIGO:
    o Bloco 3 perdeu um bloco de limitação escrito à mão dentro de um artefato
    gerado. Aqui toda frase de leitura (as três frases obrigatórias inclusive) é
    montada a partir dos números — sobrevive à reexecução e muda sozinha se um
    número mudar.

SAÍDAS:
    results/metricas/comparacao_final.csv        (validação e teste, 4 linhas cada)
    results/metricas/comparacao_tempos.csv       (RF-CPU, SVM-CPU, CNN-GPU, CNN-CPU)
    results/metricas/comparacao_estatistica.json (bootstrap pareado + leituras)
    results/metricas/COMPARACAO_FINAL.md         (versão para o texto — GERADA)
    results/figuras/comparacao_{f1_eer,roc,tempos,por_ataque,por_codec}.png

Rode a partir da raiz (depois do B5.1 e dos dois diagnósticos):
    python -m scripts.diagnostico_por_ataque
    python -m scripts.diagnostico_por_codec
    python -m scripts.comparacao_final
"""

import json
import platform
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import confusion_matrix, f1_score, roc_curve

from scripts.estabilidade_modelos import N_BOOTSTRAP, SEMENTE_BOOTSTRAP
from src.data.split import carregar_dados_split
from src.models.avaliacao import aplicar_limiar, calcular_eer
from src.utils.serializacao import json_seguro

RAIZ = Path(__file__).resolve().parents[1]
DIR_MET = RAIZ / "results" / "metricas"
DIR_FIG = RAIZ / "results" / "figuras"

# chave -> (JSON de validação, rótulo curto, braço). Ordem = ordem das linhas.
MODELOS = {
    "rf": ("rf_tuned_principal", "RF", "principal"),
    "rf_ref": ("rf_tuned_referencia", "RF (referência)", "referência"),
    "svm": ("svm_tuned_principal", "SVM", "principal"),
    "cnn": ("cnn_final_principal", "CNN", "principal"),
}
PRINCIPAIS = ("rf", "svm", "cnn")
# (a, b) -> diferença a − b. Os três pares da comparação, na ordem do marco.
PARES = (("svm", "rf"), ("cnn", "rf"), ("cnn", "svm"))
PAR_CUSTO_SUBAMOSTRAGEM = ("rf_ref", "rf")

CONJUNTOS = {"validacao": "scores_validacao.csv",
             "teste": "scores_teste_lacrado.csv"}
NOME_CONJUNTO = {"validacao": "validação (22.226)", "teste": "teste lacrado (22.227)"}

# Referência do Bloco 3 que o par SVM−RF na validação tem de reproduzir.
REFERENCIA_BLOCO3 = "estabilidade_rf_svm.json -> bootstrap_pareado_svm_menos_rf"

# ---- Figuras: cor segue a ENTIDADE, a mesma nas cinco --------------------------
# Três primeiros slots da paleta de referência (validados em todos os pares,
# fundo branco: pior CVD ΔE 9,2; normal 24,0). O aqua fica abaixo de 3:1 no
# branco — por isso toda barra leva rótulo visível e a tabela existe ao lado.
# O braço de referência vai em cinza neutro: não é concorrente, e a figura não
# deve sugerir que é.
COR = {"rf": "#2a78d6", "svm": "#eb6834", "cnn": "#1baf7a", "rf_ref": "#898781"}
TINTA, TINTA_2, EIXO, GRADE = "#0b0b0b", "#52514e", "#c3c2b7", "#e1e0d9"
ALPHA_CLARO = 0.42      # validação / predição pura: o mesmo tom, mais claro


def _ler(nome: str) -> dict:
    with open(DIR_MET / nome, encoding="utf-8") as f:
        return json.load(f)


def _fmt(x, casas: int = 4) -> str:
    """Número no padrão do texto (vírgula decimal)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.{casas}f}".replace(".", ",")


ROTULO_DO_ARQUIVO = {arq: rot for arq, rot, _ in MODELOS.values()}


def _lista(itens) -> str:
    """['a', 'b', 'c'] -> 'a, b e c' — nada de repr de lista Python no texto."""
    itens = [ROTULO_DO_ARQUIVO.get(i, i) for i in itens]
    return itens[0] if len(itens) == 1 else ", ".join(itens[:-1]) + " e " + itens[-1]


def _ic(v: np.ndarray) -> dict:
    """Mesmo resumo de estabilidade_modelos.py (média, desvio ddof=1, IC95)."""
    return {"media": round(float(v.mean()), 4),
            "desvio": round(float(v.std(ddof=1)), 4),
            "ic95": [round(float(np.percentile(v, 2.5)), 4),
                     round(float(np.percentile(v, 97.5)), 4)]}


# =============================================================================
# 1. Leitura e conferência das fontes
# =============================================================================
def carregar_fontes() -> dict:
    fontes = {
        "val": {c: _ler(f"{arq}.json") for c, (arq, _, _) in MODELOS.items()},
        "teste": _ler("teste_lacrado.json"),
        "pipeline": _ler("tempo_pipeline_completo.json"),
        "estabilidade": _ler("estabilidade_rf_svm.json"),
        "ataque": _ler("diagnostico_por_ataque_resumo.json"),
        "codec": _ler("diagnostico_por_codec_resumo.json"),
    }
    if fontes["teste"].get("ensaio") or fontes["teste"]["conjunto"] != "teste":
        raise RuntimeError("teste_lacrado.json não é a avaliação real do teste")
    for c, (arq, _, _) in MODELOS.items():
        sel = fontes["val"][c]["selecao_limiar"]
        lim_teste = fontes["teste"]["modelos"][arq]["limiar"]
        if sel["conjunto"] != "validacao" or lim_teste != sel["limiar"]:
            raise RuntimeError(f"{arq}: o limiar aplicado no teste ({lim_teste}) "
                               f"não é o da validação ({sel['limiar']})")
    return fontes


def carregar_scores(fontes: dict) -> dict:
    """Vetores do B5.1, conferidos contra os números publicados.

    A validação é conferida também contra a ORDEM de linhas de features.csv x
    split.csv — é ela que faz o par SVM−RF reproduzir o Bloco 3.
    """
    df = carregar_dados_split(RAIZ)
    ordem_val = df.loc[df["conjunto"] == "validacao", ["arquivo", "classe_binaria"]]
    saida = {}
    for conj, arquivo in CONJUNTOS.items():
        t = pd.read_csv(DIR_MET / arquivo, float_precision="round_trip")
        if conj == "validacao" and not (
                t["arquivo"].tolist() == ordem_val["arquivo"].tolist()
                and t["classe_binaria"].tolist() == ordem_val["classe_binaria"].tolist()):
            raise RuntimeError(f"{arquivo}: ordem/rótulos diferentes de "
                               "features.csv x split.csv")
        y = t["classe_binaria"].to_numpy().astype(int)
        s = {c: t[arq].to_numpy() for c, (arq, _, _) in MODELOS.items()}
        for c, (arq, _, _) in MODELOS.items():
            pub = (fontes["val"][c] if conj == "validacao"
                   else fontes["teste"]["modelos"][arq])
            lim = fontes["val"][c]["selecao_limiar"]["limiar"]
            f1 = f1_score(y, aplicar_limiar(s[c], lim), average="macro",
                          zero_division=0)
            eer = calcular_eer(y, s[c])[0]
            cm = confusion_matrix(y, aplicar_limiar(s[c], lim), labels=[0, 1]).tolist()
            if f1 != pub["f1_macro"] or eer != pub["eer"] or cm != pub["matriz_confusao"]:
                raise RuntimeError(f"{arquivo} [{arq}]: o vetor não reproduz o "
                                   "número publicado — o bootstrap reamostraria "
                                   "outros scores")
        saida[conj] = {"y": y, "scores": s}
    return saida


# =============================================================================
# 2. Tabelas lidas dos JSONs
# =============================================================================
def _hardware(c: str, js: dict, cenario_cnn: str = "gpu") -> str:
    if c == "cnn":
        if cenario_cnn == "gpu":
            return f"GPU {js['ambiente']['gpu']}"
        n = js["tempos_inferencia"]["cpu"]["protocolo"]["torch_num_threads"]
        return f"CPU, {n} thread"
    return f"CPU, {js['ambiente']['n_jobs_inferencia']} thread"


def _tempos(c: str, js: dict, pipeline: dict, cenario_cnn: str = "gpu") -> dict:
    """Latência (predição pura), ms/áudio em lote e pipeline completo."""
    if c == "cnn":
        t = js["tempos_inferencia"][cenario_cnn]
        chave_pipe = f"cnn_{cenario_cnn}"
    else:
        t = js["tempos_inferencia"]
        chave_pipe = c
    pipe = pipeline["por_modelo"].get(chave_pipe)
    return {
        "latencia_ms": t["latencia_ms"]["mediana"],
        "ms_por_audio_lote": t["throughput"]["ms_por_audio_mediana"],
        # total de áudios da medição em lote (RF/SVM: uma chamada com todos;
        # CNN: lotes internos de 128 somando este total)
        "n_audios_lote": t["throughput"]["batch"],
        # O braço de referência não foi cronometrado ponta a ponta
        # (tempo_pipeline_completo.json mede rf, svm, cnn_gpu, cnn_cpu).
        "pipeline_completo_ms": pipe["total_ms_por_audio"] if pipe else np.nan,
        "hardware": _hardware(c, js, cenario_cnn),
    }


def montar_tabela(fontes: dict) -> pd.DataFrame:
    linhas = []
    for conj in ("validacao", "teste"):
        for c, (arq, rotulo, _) in MODELOS.items():
            js = fontes["val"][c]
            m = js if conj == "validacao" else fontes["teste"]["modelos"][arq]
            linhas.append({
                "conjunto": conj, "modelo": arq, "rotulo": rotulo,
                "braco": js["braco"],          # lido do JSON, não da tabela acima
                "n_treino": js["n_treino"], "limiar": js["selecao_limiar"]["limiar"],
                "acuracia": m["acuracia"], "f1_bonafide": m["f1_bonafide"],
                "f1_spoof": m["f1_spoof"], "f1_macro": m["f1_macro"],
                "eer": m["eer"],
                "roc_auc": m["roc_auc_validacao"] if conj == "validacao" else m["roc_auc"],
                "recall_bonafide": m["recall_bonafide"],
                **_tempos(c, js, fontes["pipeline"]),
            })
    return pd.DataFrame(linhas)


def montar_tempos(fontes: dict) -> pd.DataFrame:
    linhas = []
    for cenario, c, cen in (("rf_cpu", "rf", "gpu"), ("svm_cpu", "svm", "gpu"),
                            ("cnn_gpu", "cnn", "gpu"), ("cnn_cpu", "cnn", "cpu")):
        js = fontes["val"][c]
        t = _tempos(c, js, fontes["pipeline"], cen)
        pipe = fontes["pipeline"]["por_modelo"][cenario if c == "cnn" else c]
        linhas.append({"cenario": cenario, "modelo": MODELOS[c][0], **t,
                       "predicao_no_pipeline_ms":
                           pipe["etapas"]["predizer"]["ms_por_audio_mediana"]})
    return pd.DataFrame(linhas)


# =============================================================================
# 3. Bootstrap pareado — um vetor de índices, todos os modelos
# =============================================================================
def reamostrar(y: np.ndarray, scores: dict, limiares: dict) -> tuple[dict, dict]:
    rng = np.random.default_rng(SEMENTE_BOOTSTRAP)
    n = len(y)
    f1 = {c: np.empty(N_BOOTSTRAP) for c in scores}
    eer = {c: np.empty(N_BOOTSTRAP) for c in scores}
    for b in range(N_BOOTSTRAP):
        i = rng.integers(0, n, size=n)          # UM sorteio, TODOS os modelos
        y_b = y[i]
        for c, s in scores.items():
            s_b = s[i]
            f1[c][b] = f1_score(y_b, aplicar_limiar(s_b, limiares[c]),
                                average="macro", zero_division=0)
            eer[c][b] = calcular_eer(y_b, s_b)[0]
    return f1, eer


def _leitura_metrica(a: str, b: str, metrica: str, ic95: list,
                     a_melhor_se_positivo: bool) -> str:
    lo, hi = ic95
    ra, rb = MODELOS[a][1], MODELOS[b][1]
    if lo > 0 or hi < 0:
        vencedor = ra if (lo > 0) == a_melhor_se_positivo else rb
        return (f"{vencedor} melhor em {metrica}: o IC95 da diferença "
                f"[{_fmt(lo)}; {_fmt(hi)}] não contém zero — diferença real "
                "neste protocolo")
    return (f"{ra} e {rb} não distinguíveis em {metrica} neste protocolo: o IC95 "
            f"da diferença [{_fmt(lo)}; {_fmt(hi)}] contém zero")


def resumir_par(a: str, b: str, f1: dict, eer: dict, obs: dict) -> dict:
    d_f1, d_eer = f1[a] - f1[b], eer[a] - eer[b]
    r = {
        "diferenca": f"{MODELOS[a][0]} - {MODELOS[b][0]}",
        "rotulo": f"{MODELOS[a][1]} − {MODELOS[b][1]}",
        "delta_observado": {"f1_macro": obs[a]["f1_macro"] - obs[b]["f1_macro"],
                            "eer": obs[a]["eer"] - obs[b]["eer"]},
        "delta_f1_macro": _ic(d_f1),
        "delta_eer": _ic(d_eer),
        f"fracao_reamostragens_{a}_melhor_f1": round(float((d_f1 > 0).mean()), 4),
        f"fracao_reamostragens_{a}_melhor_eer": round(float((d_eer < 0).mean()), 4),
    }
    r["leitura_f1_macro"] = _leitura_metrica(a, b, "f1_macro",
                                             r["delta_f1_macro"]["ic95"], True)
    r["leitura_eer"] = _leitura_metrica(a, b, "EER", r["delta_eer"]["ic95"], False)
    # Forma curta, para a coluna da tabela: o mesmo veredito, sem repetir o IC.
    for met in ("f1_macro", "eer"):
        texto = r[f"leitura_{met}"]
        r[f"veredito_{met}"] = (texto.split(":")[0].replace(" em f1_macro", "")
                                .replace(" em EER", "") + " (real)"
                                if "não contém zero" in texto
                                else "não distinguíveis neste protocolo")
    return r


def bootstrap_conjunto(conj: str, dados: dict, fontes: dict) -> dict:
    y, scores = dados["y"], dados["scores"]
    limiares = {c: fontes["val"][c]["selecao_limiar"]["limiar"] for c in MODELOS}
    obs = ({c: fontes["val"][c] for c in MODELOS} if conj == "validacao" else
           {c: fontes["teste"]["modelos"][MODELOS[c][0]] for c in MODELOS})
    print(f"  {conj}: {N_BOOTSTRAP} reamostragens x {len(MODELOS)} modelos...")
    f1, eer = reamostrar(y, scores, limiares)
    pares = {f"{a}_menos_{b}": resumir_par(a, b, f1, eer, obs) for a, b in PARES}
    a, b = PAR_CUSTO_SUBAMOSTRAGEM
    custo = resumir_par(a, b, f1, eer, obs)
    custo["o_que_mede"] = ("custo da subamostragem de 30k para o RF: mesmo "
                           "algoritmo e hiperparâmetros, treino completo "
                           "(103.723) contra a subamostra (30.000). NÃO é um "
                           "par da comparação entre modelos")
    return {
        "n": int(len(y)),
        "n_reamostragens": N_BOOTSTRAP,
        "semente": SEMENTE_BOOTSTRAP,
        "limiares_fixos": {MODELOS[c][0]: limiares[c] for c in MODELOS},
        "pares": pares,
        "custo_da_subamostragem": custo,
        "_individual": {c: {"f1_macro": _ic(f1[c]), "eer": _ic(eer[c])}
                        for c in MODELOS},
    }


def conferir_bloco3(boot_val: dict, estab: dict) -> dict:
    """O par SVM−RF na validação tem de reproduzir o Bloco 3 casa por casa."""
    ref = estab["bootstrap_pareado_svm_menos_rf"]
    novo = boot_val["pares"]["svm_menos_rf"]
    iguais = (novo["delta_f1_macro"] == ref["delta_f1_macro"]
              and novo["delta_eer"] == ref["delta_eer"]
              and novo["fracao_reamostragens_svm_melhor_f1"]
              == ref["fracao_reamostragens_svm_melhor_f1"]
              and novo["fracao_reamostragens_svm_melhor_eer"]
              == ref["fracao_reamostragens_svm_melhor_eer"])
    if not iguais:
        raise RuntimeError(
            f"o par SVM−RF na validação NÃO reproduz {REFERENCIA_BLOCO3}:\n"
            f"  Bloco 3: {ref['delta_f1_macro']} / {ref['delta_eer']}\n"
            f"  agora  : {novo['delta_f1_macro']} / {novo['delta_eer']}\n"
            "Mesma semente, mesmas linhas e mesmos scores têm de dar o mesmo "
            "número — investigue antes de citar qualquer IC.")
    return {"referencia": REFERENCIA_BLOCO3, "reproduziu": True,
            "o_que_prova": ("a extensão para três modelos não mudou o teste: "
                            "mesmo sorteio de índices, mesmas linhas, mesmos "
                            "scores — o par SVM−RF sai idêntico ao do Bloco 3")}


# =============================================================================
# 4. Leituras derivadas
# =============================================================================
def heuristicas(boots: dict, fontes: dict) -> dict:
    estab = fontes["estabilidade"]
    indiv = {conj: {MODELOS[c][0]: v for c, v in b["_individual"].items()}
             for conj, b in boots.items()}
    comp = {}
    for conj, b in boots.items():
        for nome_par, p in b["pares"].items():
            a, bb = nome_par.split("_menos_")
            disp_f1 = max(b["_individual"][a]["f1_macro"]["desvio"],
                          b["_individual"][bb]["f1_macro"]["desvio"])
            disp_eer = max(b["_individual"][a]["eer"]["desvio"],
                           b["_individual"][bb]["eer"]["desvio"])
            comp[f"{conj}.{nome_par}"] = {
                "abs_delta_f1_macro": round(abs(p["delta_observado"]["f1_macro"]), 4),
                "maior_dispersao_individual_f1": disp_f1,
                "abs_delta_eer": round(abs(p["delta_observado"]["eer"]), 4),
                "maior_dispersao_individual_eer": disp_eer,
            }
    return {
        "rotulo": "HEURÍSTICA — contexto, NÃO é o teste",
        "por_que_nao_e_o_teste": (
            "IC individual e dispersão por semente ignoram a correlação entre os "
            "erros dos modelos nas MESMAS amostras; comparar a diferença contra a "
            "maior dispersão individual superestima o ruído da comparação. O "
            "teste é o IC da diferença pareada (bloco `bootstrap_pareado`)"),
        "ic_individual": indiv,
        "nota_ic_individual": (
            "calculados a partir das MESMAS 1.000 reamostragens do pareado; por "
            "isso diferem na 4ª casa dos IC individuais de "
            "estabilidade_rf_svm.json, que usou outra sequência de índices"),
        "variancia_de_treino": {
            "rf": {"f1_macro_5_sementes": estab["rf_sementes"]["f1_macro"],
                   "eer_5_sementes": estab["rf_sementes"]["eer"],
                   "fonte": "estabilidade_rf_svm.json -> rf_sementes"},
            "svm": "não existe: SVC com probability=False é determinístico",
            "cnn": ("não medida: a CNN final é um treino único (refit do B4.6, "
                    f"semente {_ler('refit_cnn.json')['semente']}); re-treinar "
                    "com outras sementes está fora do escopo do B5"),
        },
        "diferenca_x_maior_dispersao_individual": comp,
    }


def leitura_delta(fontes: dict, boots: dict) -> dict:
    """Validação -> teste: o limiar sobreajustou a validação?

    A RÉGUA: validação e teste são amostras INDEPENDENTES do mesmo universo, e
    cada uma tem o seu ruído. Comparar o f1 do teste com o IC95 da validação
    ignoraria o ruído do próprio teste (e acusaria sobreajuste onde há só
    amostragem). O ruído da DIFERENÇA de duas estimativas independentes é
    1,96·√(dp_val² + dp_teste²), com os desvios do bootstrap de cada conjunto.
    Aproximação normal — heurística, e rotulada assim.
    """
    t = fontes["teste"]
    ind = {conj: boots[conj]["_individual"] for conj in boots}
    por_modelo = {}
    for c, (arq, _, _) in MODELOS.items():
        d_f1 = t["delta_validacao_teste"][arq]
        d_eer = t["delta_eer_validacao_teste"][arq]
        ruido = {met: round(1.96 * float(np.hypot(ind["validacao"][c][met]["desvio"],
                                                   ind["teste"][c][met]["desvio"])), 4)
                 for met in ("f1_macro", "eer")}
        por_modelo[arq] = {
            "f1_macro_validacao": fontes["val"][c]["f1_macro"],
            "f1_macro_teste": t["modelos"][arq]["f1_macro"],
            "delta_f1_macro": d_f1,
            "delta_eer": d_eer,
            "ruido_da_diferenca_f1_macro": ruido["f1_macro"],
            "ruido_da_diferenca_eer": ruido["eer"],
            "delta_f1_dentro_do_ruido": bool(abs(d_f1) < ruido["f1_macro"]),
            "delta_eer_dentro_do_ruido": bool(abs(d_eer) < ruido["eer"]),
            "fracao_do_ruido_f1": round(abs(d_f1) / ruido["f1_macro"], 2),
        }
    ordem = {conj: sorted(PRINCIPAIS, key=lambda c: -(
        fontes["val"][c]["f1_macro"] if conj == "validacao"
        else t["modelos"][MODELOS[c][0]]["f1_macro"])) for conj in ("validacao", "teste")}
    fora = [a for a, v in por_modelo.items()
            if not (v["delta_f1_dentro_do_ruido"] and v["delta_eer_dentro_do_ruido"])]
    perto = [a for a, v in por_modelo.items() if v["fracao_do_ruido_f1"] >= 0.8]
    # Queda de f1 sem queda de EER: o conjunto não ficou mais difícil; o que
    # perdeu foi o ponto de corte transportado.
    so_limiar = [a for a, v in por_modelo.items()
                 if abs(v["delta_f1_macro"]) >= 0.005 and abs(v["delta_eer"]) < 0.001]
    return {
        "rotulo": ("HEURÍSTICA — ruído da diferença entre duas amostras "
                   "independentes, 1,96·√(dp_val² + dp_teste²), aproximação normal"),
        "por_modelo": por_modelo,
        "ordem_f1_macro_validacao": [MODELOS[c][1] for c in ordem["validacao"]],
        "ordem_f1_macro_teste": [MODELOS[c][1] for c in ordem["teste"]],
        "ordem_preservada": ordem["validacao"] == ordem["teste"],
        "leitura": (
            ("Os quatro deltas validação→teste (f1_macro e EER) cabem no ruído "
             "de amostragem dos dois conjuntos: a seleção de limiar sobre 22 mil "
             "amostras não sobreajustou a validação — evidência de que o "
             "protocolo funcionou. " if not fora else
             f"Em {_lista(fora)} o delta validação→teste EXCEDE o ruído de amostragem "
             "dos dois conjuntos: para esses, o limiar transportado perdeu parte "
             "do ótimo e isso vai para o texto como observação. ")
            + (f"Mais perto do limite do ruído: {_lista(perto)} (Δf1 "
               + _lista([f"{_fmt(por_modelo[a]['delta_f1_macro'])} para um ruído de "
                         f"±{_fmt(por_modelo[a]['ruido_da_diferenca_f1_macro'])}"
                         for a in perto]) + "). "
               if perto else "")
            + (f"Para {_lista(so_limiar)}, o EER praticamente não muda "
               f"({_lista([_fmt(por_modelo[a]['delta_eer']) for a in so_limiar])}) "
               "enquanto o f1_macro cai: como o EER não depende de limiar, o "
               "teste não ficou mais difícil para esse modelo — o que se perdeu "
               "foi um pouco do ponto de corte transportado. " if so_limiar else "")
            + "A ordem dos três modelos principais "
            + ("é a mesma na validação e no teste." if ordem["validacao"] == ordem["teste"]
               else "MUDA entre validação e teste.")),
    }


def leitura_ataque(fontes: dict) -> dict:
    res = fontes["ataque"]["por_modelo"]
    delta = {f"{a}_menos_{b}": fontes["val"][a]["eer"] - fontes["val"][b]["eer"]
             for a, b in PARES}
    maior_delta = max(abs(v) for v in delta.values())
    por_modelo, frases = {}, []
    for c in PRINCIPAIS:
        r = res[c]
        amp = r["amplitude_eer"]
        pares_c = {k: abs(v) for k, v in delta.items() if c in k.split("_menos_")}
        supera = {k: amp > v for k, v in pares_c.items()}
        top3 = sorted(r["eer_por_ataque"], key=lambda k: -r["eer_por_ataque"][k])[:3]
        por_modelo[MODELOS[c][0]] = {
            "amplitude_eer_entre_ataques": amp,
            "razao_eer_dificil_facil": r["razao_eer_dificil_facil"],
            "ataque_mais_facil": r["ataque_mais_facil"],
            "ataque_mais_dificil": r["ataque_mais_dificil"],
            "tres_mais_dificeis": top3,
            "abs_delta_eer_para_os_outros_modelos": pares_c,
            "amplitude_supera_a_distancia_para": supera,
        }
        if all(supera.values()):
            frases.append(f"{MODELOS[c][1]}: amplitude {_fmt(amp)} supera a "
                          "distância agregada para os dois outros modelos")
        else:
            def _outro(k):
                return MODELOS[[m for m in k.split("_menos_") if m != c][0]][1]
            nao = [k for k, v in supera.items() if not v]
            sim = [k for k, v in supera.items() if v]
            frases.append(f"{MODELOS[c][1]}: amplitude {_fmt(amp)} NÃO supera a "
                          "distância agregada para "
                          + _lista([f"o {_outro(k)} ({_fmt(pares_c[k])})" for k in nao])
                          + (", embora supere a distância para "
                             + _lista([f"o {_outro(k)} ({_fmt(pares_c[k])})" for k in sim])
                             if sim else ""))
    comuns = set.intersection(*(set(v["tres_mais_dificeis"]) for v in por_modelo.values()))
    vale_para_todos = all(all(v["amplitude_supera_a_distancia_para"].values())
                          for v in por_modelo.values())
    amps = [por_modelo[MODELOS[c][0]]["amplitude_eer_entre_ataques"] for c in PRINCIPAIS]
    return {
        "conjunto": "validacao (o diagnóstico por ataque não abre o teste)",
        "delta_eer_agregado_entre_modelos": delta,
        "maior_abs_delta_eer_entre_modelos": maior_delta,
        "por_modelo": por_modelo,
        "ataques_entre_os_3_mais_dificeis_nos_tres_modelos": sorted(comuns),
        "leitura_registrada_bloco3": (
            "a dificuldade varia muito mais entre sistemas de síntese (até 0,29 "
            "de EER) do que entre modelos (ΔEER 0,0466)"),
        "vale_para_os_tres": vale_para_todos,
        "leitura": (
            "; ".join(frases) + ". "
            + ("A leitura do Bloco 3 vale para os três modelos: " if vale_para_todos
               else "A leitura do Bloco 3 vale integralmente para RF e SVM e só "
                    "em parte para a CNN: ")
            + f"a amplitude entre ataques cai de {_fmt(amps[0])} (RF) para "
            f"{_fmt(amps[1])} (SVM) e {_fmt(amps[2])} (CNN), enquanto a maior "
            f"distância entre modelos é {_fmt(maior_delta)}. "
            + ((f"O ataque {_lista(sorted(comuns))} está" if len(comuns) == 1 else
                f"Os ataques {_lista(sorted(comuns))} estão")
               + " entre os três mais difíceis nos três modelos — a dificuldade "
               "é, em parte, do sistema de síntese e não do classificador. "
               if comuns else "")
            + "Em razão (EER do mais difícil / do mais fácil) a CNN é a mais "
            f"desigual ({_fmt(por_modelo['cnn_final_principal']['razao_eer_dificil_facil'], 2)}x), "
            "porque o seu ataque mais fácil fica perto de zero. A amplitude "
            "grande em qualquer dos modelos segue sendo evidência a favor do "
            "risco declarado na limitação do split (por utterance, os mesmos "
            "vocoders em treino e avaliação)."),
    }


def leitura_codec(fontes: dict) -> dict:
    res = fontes["codec"]["por_modelo"]
    por_modelo = {}
    for c in PRINCIPAIS:
        pb = res[c]["por_banda"]
        est, lar = pb["estreita"], pb["larga"]
        por_modelo[MODELOS[c][0]] = {
            "f1_macro": {"estreita": est["f1_macro"], "larga": lar["f1_macro"],
                         "ganho_absoluto": round(lar["f1_macro"] - est["f1_macro"], 4)},
            "eer": {"estreita": est["eer"], "larga": lar["eer"],
                    "ganho_absoluto": round(est["eer"] - lar["eer"], 4),
                    "reducao_relativa": round(1 - lar["eer"] / est["eer"], 4)},
            "hipotese_banda_alta_sustentada": res[c]["hipotese_banda_alta_sustentada"],
        }
    cnn = por_modelo["cnn_final_principal"]
    outros = [por_modelo[MODELOS[c][0]] for c in ("rf", "svm")]
    maior_abs_f1 = all(cnn["f1_macro"]["ganho_absoluto"] > o["f1_macro"]["ganho_absoluto"] for o in outros)
    maior_abs_eer = all(cnn["eer"]["ganho_absoluto"] > o["eer"]["ganho_absoluto"] for o in outros)
    maior_rel_eer = all(cnn["eer"]["reducao_relativa"] > o["eer"]["reducao_relativa"] for o in outros)
    todos = all(v["hipotese_banda_alta_sustentada"] for v in por_modelo.values())
    ganhos = "; ".join(
        f"{MODELOS[c][1]} Δf1 {_fmt(por_modelo[MODELOS[c][0]]['f1_macro']['ganho_absoluto'])}, "
        f"ΔEER {_fmt(-por_modelo[MODELOS[c][0]]['eer']['ganho_absoluto'])} "
        f"({_fmt(100 * por_modelo[MODELOS[c][0]]['eer']['reducao_relativa'], 1)}% do EER)"
        for c in PRINCIPAIS)
    return {
        "conjunto": "validacao (o diagnóstico por codec não abre o teste)",
        "hipotese_do_marco": ("se o ganho da banda larga for MAIOR na CNN, a "
                              "representação tempo-frequência preservada (fmax "
                              "8 kHz) capta melhor os artefatos acima de 4 kHz"),
        "por_modelo": por_modelo,
        "cnn_ganho_maior_absoluto_f1": maior_abs_f1,
        "cnn_ganho_maior_absoluto_eer": maior_abs_eer,
        "cnn_ganho_maior_relativo_eer": maior_rel_eer,
        "rotulo": "leitura DESCRITIVA — não há IC para a diferença entre ganhos",
        "leitura": (
            ("A banda larga é melhor nos três modelos (f1_macro maior E EER "
             "menor), e o EER é independente de limiar — o contraste não é "
             "artefato do limiar. " if todos else
             "A hipótese da banda alta NÃO se sustenta em todos os modelos. ")
            + f"Ganho larga − estreita: {ganhos}. "
            + ("Em termos ABSOLUTOS o ganho da CNN não é o maior "
               if not (maior_abs_f1 or maior_abs_eer) else
               "Em termos absolutos o ganho da CNN é o maior ")
            + ("e em termos RELATIVOS (fração do EER da banda estreita removida) "
               "é o maior dos três. " if maior_rel_eer else
               "e em termos relativos também não é o maior. ")
            + ("As duas escalas apontam em sentidos opostos, e sem IC para a "
               "diferença entre ganhos a hipótese do marco fica NÃO DECIDIDA: "
               "não se sustenta em escala absoluta e é compatível com os dados "
               "em escala relativa. O que a CNN acrescenta com segurança é que "
               "o contraste aparece também num modelo que não usa ZCR nem "
               "centróide — a evidência é sobre o sinal, não sobre aquelas "
               "duas features."
               if (maior_rel_eer != (maior_abs_f1 or maior_abs_eer)) else
               "Sem IC para a diferença entre ganhos, é leitura descritiva.")),
    }


def frases_obrigatorias(fontes: dict, boots: dict, tempos: pd.DataFrame) -> dict:
    v, t = fontes["val"], fontes["teste"]["modelos"]
    ref, rf = MODELOS["rf_ref"][0], MODELOS["rf"][0]
    custo_te = boots["teste"]["custo_da_subamostragem"]
    custo_va = boots["validacao"]["custo_da_subamostragem"]
    T = tempos.set_index("cenario")
    r_audio = T.loc["rf_cpu", "pipeline_completo_ms"] / T.loc["svm_cpu", "pipeline_completo_ms"]
    r_lote = T.loc["svm_cpu", "ms_por_audio_lote"] / T.loc["rf_cpu", "ms_por_audio_lote"]
    r_cnn_lote = T.loc["cnn_cpu", "ms_por_audio_lote"] / T.loc["cnn_gpu", "ms_por_audio_lote"]
    r_cnn_pipe = T.loc["cnn_cpu", "pipeline_completo_ms"] / T.loc["cnn_gpu", "pipeline_completo_ms"]
    eers_te = ", ".join(f"{MODELOS[c][1]} {_fmt(100 * t[MODELOS[c][0]]['eer'], 2)}%"
                        for c in PRINCIPAIS)
    pipe = {c: T.loc[c, "pipeline_completo_ms"] for c in T.index}
    mais_barata_pipe = min(pipe, key=pipe.get)
    mais_cara_pipe = max(pipe, key=pipe.get)
    razao_dados = v["rf_ref"]["n_treino"] / v["rf"]["n_treino"]
    f1_ref = t[ref]["f1_macro"]
    abaixo = ("ainda fica abaixo"
              if f1_ref < min(t[MODELOS[c][0]]["f1_macro"] for c in ("svm", "cnn"))
              else "NÃO fica abaixo de ambos")
    # Igualdade de hardware = CPU nos três. A frase herdada do B4.7 («a
    # vantagem dos clássicos não é serem mais rápidos em igualdade de
    # hardware») só se sustenta dita com os números: em CPU eles SÃO mais
    # baratos; o que muda o quadro é a CNN ganhar uma GPU.
    classicos_mais_baratos_em_cpu = max(pipe["rf_cpu"], pipe["svm_cpu"]) < pipe["cnn_cpu"]
    return {
        "1_braco_de_referencia": (
            "O braço de referência (RF no treino completo de 103.723) não é "
            "concorrente direto de SVM e CNN: ele existe para quantificar o custo "
            "da subamostragem de 30k, que foi imposta pela complexidade "
            "O(n²)–O(n³) do SVM-RBF. A comparação entre modelos é a do braço "
            "principal, em que os três viram as mesmas 30.000 amostras. "
            f"Medido: no teste o RF de referência chega a f1_macro "
            f"{_fmt(t[ref]['f1_macro'])} contra {_fmt(t[rf]['f1_macro'])} do RF "
            f"principal (Δ {_fmt(custo_te['delta_observado']['f1_macro'])}, IC95 "
            f"[{_fmt(custo_te['delta_f1_macro']['ic95'][0])}; "
            f"{_fmt(custo_te['delta_f1_macro']['ic95'][1])}]); na validação, Δ "
            f"{_fmt(custo_va['delta_observado']['f1_macro'])}. É esse o custo da "
            f"subamostragem para o RF — e, com {_fmt(razao_dados, 1)}× mais dados, "
            f"ele {abaixo} do SVM ({_fmt(t[MODELOS['svm'][0]]['f1_macro'])}) e da "
            f"CNN ({_fmt(t[MODELOS['cnn'][0]]['f1_macro'])}) treinados em 30.000."),
        "2_eer_e_literatura": (
            f"EER no teste: {eers_te}. Esses valores NÃO são comparáveis ao EER "
            "de 1,32% de Yamagishi et al. (2022): o protocolo aqui é um split "
            "interno aleatório POR UTTERANCE — os mesmos ataques (A07–A19), "
            "codecs e locutores aparecem em treino e avaliação —, enquanto o "
            "protocolo oficial do ASVspoof é deliberadamente CROSS-ATTACK. Ser "
            "independente de limiar remove a arbitrariedade do 0,50; NÃO remove "
            "a diferença de protocolo."),
        "3_custo_por_regime": (
            "Não existe «o modelo mais barato» sem dizer o regime: por áudio, "
            f"ponta a ponta, o SVM é ~{_fmt(r_audio, 1)}× mais barato que o RF "
            f"({_fmt(pipe['svm_cpu'], 2)} contra {_fmt(pipe['rf_cpu'], 2)} ms), "
            f"mas em lote o RF é ~{_fmt(r_lote, 1)}× melhor "
            f"({_fmt(T.loc['rf_cpu', 'ms_por_audio_lote'])} contra "
            f"{_fmt(T.loc['svm_cpu', 'ms_por_audio_lote'])} ms por áudio). A CNN "
            "acrescenta a dimensão que de fato importa para a pergunta de "
            "pesquisa — exige GPU ou não? — e é o par CNN-GPU / CNN-CPU que "
            f"responde: ponta a ponta, a CNN em GPU ({_fmt(pipe['cnn_gpu'], 2)} ms) "
            f"é {'a mais barata' if mais_barata_pipe == 'cnn_gpu' else 'competitiva'} "
            f"dos quatro cenários e a CNN em CPU ({_fmt(pipe['cnn_cpu'], 2)} ms) "
            f"{'a mais cara' if mais_cara_pipe == 'cnn_cpu' else 'não é a mais cara'} "
            f"({_fmt(r_cnn_pipe, 1)}× a própria versão em GPU); em lote a "
            f"diferença CPU/GPU chega a {_fmt(r_cnn_lote, 1)}×. "
            + ("Em igualdade de hardware (CPU) os dois clássicos são mais baratos "
               f"ponta a ponta que a CNN ({_fmt(pipe['svm_cpu'], 2)} e "
               f"{_fmt(pipe['rf_cpu'], 2)} contra {_fmt(pipe['cnn_cpu'], 2)} ms); "
               "o quadro só se inverte quando a CNN ganha uma GPU. A vantagem "
               "de custo dos clássicos, portanto, é NÃO EXIGIREM GPU — e é "
               "contra essa vantagem que o ganho de desempenho da CNN tem de "
               "ser pesado."
               if classicos_mais_baratos_em_cpu else
               "Em CPU os clássicos NÃO são ambos mais baratos que a CNN — a "
               "leitura de custo não depende só de haver GPU.")),
    }


# =============================================================================
# 5. Figuras
# =============================================================================
def _estilo(ax) -> None:
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(EIXO)
    ax.tick_params(colors=TINTA_2, labelsize=9)
    ax.yaxis.grid(True, color=GRADE, lw=0.6)
    ax.set_axisbelow(True)


def _rotular(ax, barras, valores, fmt, log=False, extra=None) -> None:
    for i, (b, v) in enumerate(zip(barras, valores)):
        if np.isnan(v):
            continue
        texto = fmt(v) + (f"\n{extra[i]}" if extra else "")
        y = b.get_height() * (1.15 if log else 1) + (0 if log else 0.006)
        ax.text(b.get_x() + b.get_width() / 2, y, texto, ha="center",
                va="bottom", fontsize=7, color=TINTA_2)


def fig_f1_eer(tab: pd.DataFrame, caminho: Path) -> None:
    chaves = list(MODELOS)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    x = np.arange(len(chaves))
    w = 0.38
    for ax, met, titulo in ((axes[0], "f1_macro", "f1_macro (maior = melhor)"),
                            (axes[1], "eer", "EER (menor = melhor)")):
        for k, (conj, dx, alpha) in enumerate((("validacao", -w / 2, ALPHA_CLARO),
                                               ("teste", w / 2, 1.0))):
            sub = tab[tab["conjunto"] == conj].set_index("modelo")
            vals = [sub.loc[MODELOS[c][0], met] for c in chaves]
            barras = ax.bar(x + dx, vals, w, color=[COR[c] for c in chaves],
                            alpha=alpha, edgecolor="white", linewidth=1)
            _rotular(ax, barras, vals, lambda v: _fmt(v, 3))
        ax.set_xticks(x, [MODELOS[c][1].replace(" (", "\n(") for c in chaves])
        ax.set_title(titulo, fontsize=10, color=TINTA)
        ax.set_ylim(0, 1.05 if met == "f1_macro" else tab["eer"].max() * 1.25)
        _estilo(ax)
    legenda = [Patch(facecolor=TINTA_2, alpha=ALPHA_CLARO, label="validação (tom claro)"),
               Patch(facecolor=TINTA_2, label="teste lacrado (tom cheio)")]
    axes[0].legend(handles=legenda, fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("Comparação final — validação × teste lacrado, limiar do "
                 "protocolo (selecionado na validação)", fontsize=11, color=TINTA)
    fig.text(0.5, 0.005, "RF (referência): treino completo (103.723) — quantifica "
             "o custo da subamostragem; não é concorrente direto de SVM e CNN, "
             "que treinaram nos mesmos 30.000.", ha="center", fontsize=8,
             color=TINTA_2)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


def fig_roc(dados_val: dict, fontes: dict, caminho: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.plot([0, 1], [1, 0], color=EIXO, lw=0.8)
    ax.text(0.62, 0.40, "FPR = FNR", color=TINTA_2, fontsize=8, rotation=-41)
    y = dados_val["y"]
    for c in PRINCIPAIS:
        fpr, tpr, _ = roc_curve(y, dados_val["scores"][c], pos_label=1)
        js = fontes["val"][c]
        eer = js["eer"]
        ax.plot(fpr, tpr, color=COR[c], lw=2,
                label=f"{MODELOS[c][1]} — AUC {_fmt(js['roc_auc_validacao'])}, "
                      f"EER {_fmt(eer)}")
        ax.plot([eer], [1 - eer], "o", ms=8, color=COR[c], mec="white", mew=2)
        # À direita do ponto, na mesma altura: cai no vão entre esta curva e a
        # do modelo seguinte, sem cobrir o ponto de nenhum outro modelo.
        ax.annotate(f"EER {_fmt(eer, 3)}", (eer, 1 - eer), xytext=(9, 0),
                    textcoords="offset points", fontsize=8, color=TINTA,
                    ha="left", va="center")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("taxa de falsos positivos (bonafide aceito como spoof)",
                  fontsize=9, color=TINTA_2)
    ax.set_ylabel("taxa de verdadeiros positivos (spoof detectado)",
                  fontsize=9, color=TINTA_2)
    ax.set_title("Curvas ROC na validação (22.226) — EER no cruzamento com "
                 "FPR = FNR", fontsize=10, color=TINTA)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    _estilo(ax)
    ax.xaxis.grid(True, color=GRADE, lw=0.6)
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


def fig_tempos(tempos: pd.DataFrame, n_pipeline: int, caminho: Path) -> None:
    cen = tempos.set_index("cenario")
    ordem = ["rf_cpu", "svm_cpu", "cnn_gpu", "cnn_cpu"]
    chave = {"rf_cpu": "rf", "svm_cpu": "svm", "cnn_gpu": "cnn", "cnn_cpu": "cnn"}
    curto = {"rf_cpu": "CPU", "svm_cpu": "CPU", "cnn_gpu": "GPU", "cnn_cpu": "CPU"}
    rot = [f"{MODELOS[chave[k]][1]}\n"
           f"{cen.loc[k, 'hardware'].replace('NVIDIA GeForce ', '')}" for k in ordem]
    cores = [COR[chave[k]] for k in ordem]
    x = np.arange(len(ordem))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    w = 0.38
    ax = axes[0]
    for dx, col, alpha in ((-w / 2, "latencia_ms", ALPHA_CLARO),
                           (w / 2, "pipeline_completo_ms", 1.0)):
        vals = [cen.loc[k, col] for k in ordem]
        barras = ax.bar(x + dx, vals, w, color=cores, alpha=alpha,
                        edgecolor="white", linewidth=1)
        _rotular(ax, barras, vals, lambda v: _fmt(v, 2), log=True,
                 extra=[curto[k] for k in ordem])
    ax.set_yscale("log")
    ax.set_ylim(0.1, 100)
    ax.set_ylabel("ms por áudio (escala log)", fontsize=9, color=TINTA_2)
    ax.set_title("Por áudio (batch = 1)", fontsize=10, color=TINTA)
    ax.legend(handles=[Patch(facecolor=TINTA_2, alpha=ALPHA_CLARO,
                             label="só a predição (latência)"),
                       Patch(facecolor=TINTA_2,
                             label="pipeline completo (carregar + VAD + "
                                   "representação + predição)")],
              fontsize=8, frameon=False, loc="upper left")

    ax = axes[1]
    vals = [cen.loc[k, "ms_por_audio_lote"] for k in ordem]
    barras = ax.bar(x, vals, 0.55, color=cores, edgecolor="white", linewidth=1)
    _rotular(ax, barras, vals, lambda v: _fmt(v, 4), log=True,
             extra=[f"{curto[k]} · n = {int(cen.loc[k, 'n_audios_lote']):,}".replace(",", ".")
                    for k in ordem])
    ax.set_yscale("log")
    ax.set_ylim(0.01, 100)
    ax.set_ylabel("ms por áudio (escala log)", fontsize=9, color=TINTA_2)
    ax.set_title("Em lote (vazão) — só a predição", fontsize=10, color=TINTA)

    for ax in axes:
        ax.set_xticks(x, rot, fontsize=8)
        _estilo(ax)
    fig.suptitle("Custo de inferência — o hardware está em cada barra; latência e "
                 "vazão NÃO ordenam os modelos do mesmo jeito", fontsize=11,
                 color=TINTA)
    n_cpu = int(cen.loc["cnn_cpu", "n_audios_lote"])
    fig.text(0.5, 0.005, "Predição pura: tempos_inferencia de cada JSON de "
             "validação. Pipeline completo: tempo_pipeline_completo.json (amostra "
             f"de {n_pipeline} áudios). Medições distintas — não somar. CNN-CPU em "
             f"lote: amostra de {n_cpu:,} áudios (taxa).".replace(",", "."),
             ha="center", fontsize=7.5, color=TINTA_2)
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


def fig_por_ataque(fontes: dict, caminho: Path) -> None:
    res = fontes["ataque"]["por_modelo"]
    ataques = sorted(res["rf"]["eer_por_ataque"])
    x = np.arange(len(ataques))
    w = 0.27
    fig, ax = plt.subplots(figsize=(13, 4.8))
    for k, c in enumerate(PRINCIPAIS):
        vals = [res[c]["eer_por_ataque"][a] for a in ataques]
        ax.bar(x + (k - 1) * w, vals, w, color=COR[c], edgecolor="white",
               linewidth=1,
               label=f"{MODELOS[c][1]} — amplitude {_fmt(res[c]['amplitude_eer'])} "
                     f"({res[c]['ataque_mais_facil']['ataque']} → "
                     f"{res[c]['ataque_mais_dificil']['ataque']}), EER agregado "
                     f"{_fmt(fontes['val'][c]['eer'])}")
        i_max = int(np.argmax(vals))
        ax.text(x[i_max] + (k - 1) * w, vals[i_max] + 0.004, _fmt(vals[i_max], 3),
                ha="center", va="bottom", fontsize=7, color=TINTA_2)
    ax.set_xticks(x, ataques)
    ax.set_ylabel("EER vs. bonafide (menor = melhor)", fontsize=9, color=TINTA_2)
    ax.set_title("EER por sistema de ataque (A07–A19) — validação, independente "
                 "de limiar; rótulo no pior ataque de cada modelo", fontsize=10,
                 color=TINTA)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    _estilo(ax)
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


def fig_por_codec(fontes: dict, caminho: Path) -> None:
    ordem = ["alaw", "ulaw", "gsm", "pstn", "g722", "opus", "none"]
    tabs = {c: pd.read_csv(DIR_MET / f"diagnostico_por_codec_{MODELOS[c][0]}.csv"
                           ).set_index("codec") for c in PRINCIPAIS}
    x = np.arange(len(ordem))
    w = 0.27
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    for ax, met, titulo in ((axes[0], "f1_macro", "f1_macro (maior = melhor)"),
                            (axes[1], "eer", "EER (menor = melhor)")):
        for k, c in enumerate(PRINCIPAIS):
            vals = [tabs[c].loc[cod, met] for cod in ordem]
            pb = fontes["codec"]["por_modelo"][c]["por_banda"]
            ax.bar(x + (k - 1) * w, vals, w, color=COR[c], edgecolor="white",
                   linewidth=1,
                   label=(f"{MODELOS[c][1]} — estreita {_fmt(pb['estreita'][met])} "
                          f"→ larga {_fmt(pb['larga'][met])}"))
        ax.axvline(3.5, color=EIXO, lw=0.8)
        topo = 1.12 if met == "f1_macro" else max(
            tabs[c]["eer"].max() for c in PRINCIPAIS) * 1.3
        ax.set_ylim(0, topo)
        ax.text(1.5, topo * 0.97, "banda estreita (~4 kHz)", ha="center",
                va="top", fontsize=8, color=TINTA_2)
        ax.text(5, topo * 0.97, "banda larga", ha="center", va="top",
                fontsize=8, color=TINTA_2)
        ax.set_xticks(x, ordem)
        ax.set_title(titulo, fontsize=10, color=TINTA)
        # Abaixo do eixo: dentro dele, qualquer canto cobre barras.
        ax.legend(fontsize=8, frameon=False, loc="upper center",
                  bbox_to_anchor=(0.5, -0.09), ncol=1)
        _estilo(ax)
    fig.suptitle("Desempenho por codec — validação, limiar do protocolo "
                 "(o EER é independente de limiar); legenda: métrica sobre a "
                 "banda inteira",
                 fontsize=11, color=TINTA)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


# =============================================================================
# 6. Versão para o texto (GERADA)
# =============================================================================
CABECALHO_MD = ["modelo", "braço", "n treino", "limiar", "acurácia", "f1_bonafide",
                "f1_spoof", "**f1_macro**", "**EER**", "ROC-AUC", "recall bonafide",
                "latência ms", "ms/áudio (lote)", "pipeline completo ms", "hardware"]


def _linha_md(r: pd.Series, cpu_cnn: pd.Series | None) -> str:
    def par(g, c, casas):
        return f"{_fmt(g, casas)} (GPU) / {_fmt(c, casas)} (CPU)"
    if r["modelo"] == MODELOS["cnn"][0] and cpu_cnn is not None:
        lat = par(r["latencia_ms"], cpu_cnn["latencia_ms"], 4)
        lote = par(r["ms_por_audio_lote"], cpu_cnn["ms_por_audio_lote"], 4)
        pipe = par(r["pipeline_completo_ms"], cpu_cnn["pipeline_completo_ms"], 2)
        hw = f"{r['hardware']} / {cpu_cnn['hardware']}"
    else:
        lat, lote = _fmt(r["latencia_ms"]), _fmt(r["ms_por_audio_lote"])
        pipe = (_fmt(r["pipeline_completo_ms"], 2) if not np.isnan(r["pipeline_completo_ms"])
                else "não medido")
        hw = r["hardware"]
    braco = {"referencia": "referência"}.get(r["braco"], r["braco"])
    celulas = [r["rotulo"], braco, f"{r['n_treino']:,}".replace(",", "."),
               _fmt(r["limiar"]), _fmt(r["acuracia"]), _fmt(r["f1_bonafide"]),
               _fmt(r["f1_spoof"]), f"**{_fmt(r['f1_macro'])}**",
               f"**{_fmt(r['eer'])}**", _fmt(r["roc_auc"]), _fmt(r["recall_bonafide"]),
               lat, lote, pipe, hw]
    return "| " + " | ".join(celulas) + " |"


def escrever_md(tab, tempos, estat, caminho: Path) -> None:
    cpu_cnn = tempos.set_index("cenario").loc["cnn_cpu"]
    L = ["# Comparação final — RF × SVM × CNN (B5.2)", "",
         f"> **GERADO** por `scripts/comparacao_final.py` em {estat['data']} a partir "
         "dos JSONs de métricas — **não editar à mão**: a reexecução sobrescreve. "
         "Toda frase de leitura abaixo é montada a partir dos números.", ""]
    for conj, titulo in (("teste", "1. Teste lacrado (22.227) — execução única, B5.1"),
                         ("validacao", "2. Validação (22.226) — onde os limiares foram escolhidos")):
        L += [f"## {titulo}", "", "| " + " | ".join(CABECALHO_MD) + " |",
              "|" + "---|" * len(CABECALHO_MD)]
        L += [_linha_md(r, cpu_cnn) for _, r in tab[tab["conjunto"] == conj].iterrows()]
        L += [""]
    L += ["Regra de decisão `score >= limiar`, com o limiar selecionado na validação "
          "e só aplicado no teste. Latência e ms/áudio (lote) medem só a predição, a "
          "partir da representação já extraída (`tempos_inferencia` de cada JSON); "
          "pipeline completo é `tempo_pipeline_completo.json` — medições distintas. "
          "RF (referência) não é concorrente direto — ver frase 1.", ""]

    d = estat["delta_validacao_teste"]
    L += ["## 3. Validação → teste", "", "| modelo | f1_macro val | f1_macro teste | "
          "Δf1 | ruído Δf1 | ΔEER | ruído ΔEER | dentro do ruído? |",
          "|---|---|---|---|---|---|---|---|"]
    for arq, v in d["por_modelo"].items():
        dentro = v["delta_f1_dentro_do_ruido"] and v["delta_eer_dentro_do_ruido"]
        L.append(f"| {ROTULO_DO_ARQUIVO[arq]} | {_fmt(v['f1_macro_validacao'])} | "
                 f"{_fmt(v['f1_macro_teste'])} | "
                 f"{_fmt(v['delta_f1_macro'])} | ±{_fmt(v['ruido_da_diferenca_f1_macro'])} | "
                 f"{_fmt(v['delta_eer'])} | ±{_fmt(v['ruido_da_diferenca_eer'])} | "
                 f"{'sim' if dentro else 'NÃO'} |")
    L += ["", f"*{d['rotulo']}.* " + d["leitura"], ""]

    L += ["## 4. Bootstrap pareado — o teste que decide", "",
          f"{N_BOOTSTRAP:,}".replace(",", ".")
          + f" reamostragens, semente {SEMENTE_BOOTSTRAP}; um vetor de "
          "índices por reamostragem aplicado aos scores de todos os modelos; limiar "
          "fixo no do protocolo. Conferido: "
          + estat["conferencia_bloco3"]["o_que_prova"] + ".", "",
          "| conjunto | diferença | Δf1_macro | IC95 | ΔEER | IC95 | 1º vence (f1 / EER) | f1_macro | EER |",
          "|---|---|---|---|---|---|---|---|---|"]
    for conj in ("validacao", "teste"):
        b = estat["bootstrap_pareado"][conj]
        for nome_par, p in list(b["pares"].items()) + [("custo_da_subamostragem",
                                                        b["custo_da_subamostragem"])]:
            a = nome_par.split("_menos_")[0] if "_menos_" in nome_par else "rf_ref"
            fr_f1 = p[f"fracao_reamostragens_{a}_melhor_f1"]
            fr_eer = p[f"fracao_reamostragens_{a}_melhor_eer"]
            rotulo = (p["rotulo"] if nome_par != "custo_da_subamostragem"
                      else f"{p['rotulo']} *(custo da subamostragem — não é par da comparação)*")
            L.append(f"| {conj} | {rotulo} | {_fmt(p['delta_observado']['f1_macro'])} | "
                     f"[{_fmt(p['delta_f1_macro']['ic95'][0])}; {_fmt(p['delta_f1_macro']['ic95'][1])}] | "
                     f"{_fmt(p['delta_observado']['eer'])} | "
                     f"[{_fmt(p['delta_eer']['ic95'][0])}; {_fmt(p['delta_eer']['ic95'][1])}] | "
                     f"{_fmt(100 * fr_f1, 1)}% / {_fmt(100 * fr_eer, 1)}% | "
                     f"{p['veredito_f1_macro']} | {p['veredito_eer']} |")
    L += ["", "IC que não contém zero ⇒ a diferença é real neste protocolo. IC que "
          "contém zero ⇒ «não distinguíveis neste protocolo».", ""]

    titulos = {"1_braco_de_referencia": "1. Sobre o braço de referência.",
               "2_eer_e_literatura": "2. Sobre o EER e a literatura.",
               "3_custo_por_regime": "3. Sobre o custo."}
    L += ["## 5. As três frases obrigatórias", ""]
    for k, frase in estat["frases_obrigatorias"].items():
        L += [f"**{titulos[k]}** {frase}", ""]

    L += ["## 6. Por ataque e por codec (validação)", "",
          "**Por ataque.** " + estat["por_ataque"]["leitura"], "",
          "**Por codec.** *(" + estat["por_codec"]["rotulo"] + ")* "
          + estat["por_codec"]["leitura"], ""]

    h = estat["heuristicas"]
    L += ["## 7. Heurísticas — contexto, NÃO o teste", "", h["por_que_nao_e_o_teste"] + ".", "",
          f"- RF, variância de treino (5 sementes): f1_macro "
          f"{_fmt(h['variancia_de_treino']['rf']['f1_macro_5_sementes']['media'])} ± "
          f"{_fmt(h['variancia_de_treino']['rf']['f1_macro_5_sementes']['desvio'])}.",
          f"- SVM: {h['variancia_de_treino']['svm']}.",
          f"- CNN: {h['variancia_de_treino']['cnn']}.", "",
          "| conjunto | modelo | f1_macro IC95 individual | EER IC95 individual |",
          "|---|---|---|---|"]
    for conj, por in h["ic_individual"].items():
        for arq, v in por.items():
            L.append(f"| {conj} | {ROTULO_DO_ARQUIVO[arq]} | [{_fmt(v['f1_macro']['ic95'][0])}; "
                     f"{_fmt(v['f1_macro']['ic95'][1])}] | [{_fmt(v['eer']['ic95'][0])}; "
                     f"{_fmt(v['eer']['ic95'][1])}] |")
    L += ["", "## Fontes", ""] + [f"- `{f}`" for f in estat["fontes"]] + [""]
    caminho.write_text("\n".join(L), encoding="utf-8")


# =============================================================================
# 7. Execução
# =============================================================================
def main() -> None:
    fontes = carregar_fontes()
    print("Conferindo os vetores de score contra os números publicados...")
    dados = carregar_scores(fontes)
    print("  OK — os 8 vetores (4 modelos x 2 conjuntos) reproduzem os JSONs")

    tab = montar_tabela(fontes)
    tempos = montar_tempos(fontes)
    colunas_csv = ["conjunto", "modelo", "braco", "n_treino", "limiar", "acuracia",
                   "f1_bonafide", "f1_spoof", "f1_macro", "eer", "roc_auc",
                   "recall_bonafide", "latencia_ms", "ms_por_audio_lote",
                   "pipeline_completo_ms", "hardware"]
    tab[colunas_csv].to_csv(DIR_MET / "comparacao_final.csv", index=False)
    tempos.to_csv(DIR_MET / "comparacao_tempos.csv", index=False)

    print("Bootstrap pareado:")
    boots = {conj: bootstrap_conjunto(conj, dados[conj], fontes) for conj in CONJUNTOS}
    bloco3 = conferir_bloco3(boots["validacao"], fontes["estabilidade"])
    print("  OK — SVM−RF na validação reproduz o Bloco 3")

    estat = {
        "analise": "comparacao_final",
        "data": date.today().isoformat(),
        "conjuntos": {"validacao": 22226, "teste": 22227},
        "teste_pontuado_aqui": False,
        "origem_dos_scores": {c: f"results/metricas/{a}" for c, a in CONJUNTOS.items()},
        "delta_validacao_teste": leitura_delta(fontes, boots),
        "heuristicas": heuristicas(boots, fontes),
    }
    estat["bootstrap_pareado"] = {
        conj: {k: v for k, v in b.items() if k != "_individual"}
        for conj, b in boots.items()}
    estat["conferencia_bloco3"] = bloco3
    estat["por_ataque"] = leitura_ataque(fontes)
    estat["por_codec"] = leitura_codec(fontes)
    estat["frases_obrigatorias"] = frases_obrigatorias(fontes, boots, tempos)
    estat["fontes"] = ([f"results/metricas/{a}.json" for a, _, _ in MODELOS.values()]
                       + ["results/metricas/teste_lacrado.json",
                          "results/metricas/tempo_pipeline_completo.json",
                          "results/metricas/estabilidade_rf_svm.json",
                          "results/metricas/diagnostico_por_ataque_resumo.json",
                          "results/metricas/diagnostico_por_codec_resumo.json",
                          *estat["origem_dos_scores"].values()])
    estat["ambiente"] = {"python": platform.python_version(),
                         "sistema": f"{platform.system()} {platform.release()}"}
    with open(DIR_MET / "comparacao_estatistica.json", "w", encoding="utf-8") as f:
        json.dump(estat, f, indent=2, ensure_ascii=False, default=json_seguro)
    escrever_md(tab, tempos, estat, DIR_MET / "COMPARACAO_FINAL.md")

    DIR_FIG.mkdir(parents=True, exist_ok=True)
    fig_f1_eer(tab, DIR_FIG / "comparacao_f1_eer.png")
    fig_roc(dados["validacao"], fontes, DIR_FIG / "comparacao_roc.png")
    fig_tempos(tempos, fontes["pipeline"]["n_amostra"], DIR_FIG / "comparacao_tempos.png")
    fig_por_ataque(fontes, DIR_FIG / "comparacao_por_ataque.png")
    fig_por_codec(fontes, DIR_FIG / "comparacao_por_codec.png")

    # ---- Resumo no stdout ---------------------------------------------------
    for conj in ("validacao", "teste"):
        print(f"\n== {NOME_CONJUNTO[conj]} ==")
        for nome_par, p in estat["bootstrap_pareado"][conj]["pares"].items():
            print(f"  {p['diferenca']:<42} Δf1 {p['delta_observado']['f1_macro']:+.4f} "
                  f"IC95 {p['delta_f1_macro']['ic95']} | ΔEER "
                  f"{p['delta_observado']['eer']:+.4f} IC95 {p['delta_eer']['ic95']}")
            print(f"    {p['leitura_f1_macro']}")
            print(f"    {p['leitura_eer']}")
    print(f"\n{estat['delta_validacao_teste']['leitura']}")
    print(f"\nPor ataque: {estat['por_ataque']['leitura']}")
    print(f"\nPor codec: {estat['por_codec']['leitura']}")
    print("\nSalvo: comparacao_final.csv, comparacao_tempos.csv, "
          "comparacao_estatistica.json, COMPARACAO_FINAL.md + 5 figuras")


if __name__ == "__main__":
    main()
