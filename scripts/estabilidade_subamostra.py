"""
Estabilidade entre SUBAMOSTRAS (R6.2) — a fonte de variação que faltava medir
=============================================================================

A PERGUNTA DE FUNDO DO ORIENTADOR, RESPONDIDA NO EIXO CERTO:
    O item 6 do protocolo pede variância entre sementes como análise
    complementar. `scripts/estabilidade_modelos.py` já a mediu no eixo do
    TREINO: 5 sementes do RF dão ±0,0004 em f1_macro — praticamente zero. E a
    resposta ao pedido de "3 sementes no SVM" ("o SVC com probability=False é
    determinístico") está tecnicamente certa.

    Só que as duas respostas juntas deixam a pergunta de fundo em aberto, porque
    a fonte de incerteza que DOMINA o braço principal não é a semente das
    árvores: é QUAL SUBAMOSTRA DE 30k CAIU. Essa fonte nunca foi variada — para
    modelo nenhum. RF, SVM (e amanhã a CNN) foram todos treinados na única
    subamostra de semente 42.

    Este script varia exatamente isso: 3 subamostras alternativas (sementes 43,
    44, 45), MESMA regra de estratificação, MESMOS hiperparâmetros vencedores,
    MESMO protocolo de limiar. Um fator por vez — a mesma disciplina do braço de
    referência (ver docstring de src/models/ajustar_rf.py). Custa menos de dois
    minutos de CPU e fecha a única frente de banca do Bloco 3 sem medida.

O QUE NÃO SE FAZ AQUI, E POR QUÊ:
    - NÃO se refaz a busca de hiperparâmetros. Se a configuração também mudasse,
      a dispersão mediria duas causas ao mesmo tempo (qual subamostra caiu E
      qual configuração venceu) e não responderia a pergunta.
    - NÃO se sobrescreve data/processed/subamostra_30k.csv. A subamostra oficial
      (MD5 654cb796...) é artefato CONGELADO; as alternativas vão para
      data/processed/subamostras_estabilidade/.
    - NÃO se toca no teste. Limiar selecionado na validação, avaliação na
      validação.

A GUARDA QUE LEGITIMA TUDO:
    antes de gerar qualquer alternativa, o script re-gera a subamostra com
    semente 42 pela função COMPARTILHADA `montar_subamostra` e confere o MD5
    contra o artefato congelado. Se não bater, a regra que produz as
    alternativas não é a regra que produziu a oficial — e comparar as
    dispersões seria comparar coisas diferentes. Nesse caso o script ABORTA.

A LEITURA CRÍTICA (o ponto todo do exercício):
    o JSON compara três grandezas na MESMA unidade (f1_macro):
      (a) desvio entre SUBAMOSTRAS      — medido aqui;
      (b) desvio entre SEMENTES DE TREINO — ±0,0004, estabilidade_rf_svm.json;
      (c) distância RF x SVM             — Δf1 ≈ 0,0762, bootstrap pareado.
    Se (a) >> (b), isso é resultado a declarar: a semente das árvores era a
    fonte de variação errada de se olhar. Se, ainda assim, (c) > (a), a
    conclusão do Bloco 3 fica MAIS forte do que está hoje. E se (c) < (a), isso
    TEM de ser dito no texto — é literalmente a regra do item 6 do protocolo,
    já ecoada em estabilidade_rf_svm.json -> leitura_critica.regra.

SAÍDAS:
    data/processed/subamostras_estabilidade/subamostra_30k_seed{43,44,45}.csv
    results/metricas/estabilidade_subamostra.json

Rode a partir da raiz:  python -m scripts.estabilidade_subamostra
"""

import hashlib
import json
import platform
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from src.utils.config import carregar_config
from src.utils.serializacao import json_seguro
from src.utils.seeds import fixar_seeds
from src.data.split import carregar_dados_split, colunas_features
from src.models.avaliacao import avaliar, selecionar_limiar
from src.models.treinar_svm import montar_pipeline
from scripts.gerar_subamostra import montar_subamostra

RAIZ = Path(__file__).resolve().parents[1]
DIR_MET = RAIZ / "results" / "metricas"
DIR_ALT = RAIZ / "data" / "processed" / "subamostras_estabilidade"

SEMENTES_ALTERNATIVAS = [43, 44, 45]

# Referências para a leitura crítica, lidas de artefatos e não chutadas.
FONTE_ESTABILIDADE = DIR_MET / "estabilidade_rf_svm.json"


def preparar_treino_e_validacao(cfg: dict):
    """Treino completo (com coluna `estrato`) e validação, prontos para uso."""
    colunas_estrato = list(cfg["experimento"]["estratificacao_subamostra"])
    df = carregar_dados_split(RAIZ)
    cols = colunas_features(df)

    labels = pd.read_csv(RAIZ / "data" / "processed" / "labels.csv",
                         usecols=["arquivo", *colunas_estrato])
    treino = df[df["conjunto"] == "treino"].merge(labels, on="arquivo",
                                                  how="inner")
    treino["estrato"] = treino[colunas_estrato].astype(str).agg("|".join, axis=1)
    validacao = df[df["conjunto"] == "validacao"]   # NUNCA 'teste'
    return treino, validacao, cols, colunas_estrato


def guarda_semente_42(treino, colunas_estrato, alvo, cfg) -> dict:
    """A regra compartilhada reproduz, byte a byte, a subamostra congelada?"""
    oficial = RAIZ / cfg["experimento"]["caminho_subamostra"]
    md5_oficial = hashlib.md5(oficial.read_bytes()).hexdigest()

    amostra, _, _ = montar_subamostra(treino, colunas_estrato, alvo,
                                      semente=cfg["semente"])
    tmp = DIR_ALT / "_conferencia_seed42.csv"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    amostra.to_csv(tmp, index=False)
    md5_regerado = hashlib.md5(tmp.read_bytes()).hexdigest()
    tmp.unlink()

    if md5_regerado != md5_oficial:
        raise RuntimeError(
            "GUARDA FALHOU: montar_subamostra com semente 42 NÃO reproduz "
            f"{oficial.name}.\n  oficial : {md5_oficial}\n  regerado: "
            f"{md5_regerado}\nA regra que geraria as alternativas não é a regra "
            "que gerou a subamostra oficial — comparar as dispersões seria "
            "comparar coisas diferentes. PARE e investigue.")
    print(f"[OK ] guarda: semente 42 reproduz {oficial.name} (MD5 {md5_oficial})")
    return {"md5_oficial": md5_oficial, "md5_regerado": md5_regerado,
            "reproduz": True}


def treinar_e_avaliar(chave: str, params: dict, ids: pd.Series,
                      treino: pd.DataFrame, validacao: pd.DataFrame,
                      cols: list[str], semente_treino: int) -> dict:
    """Treina UM modelo numa subamostra e mede na validação, pelo protocolo."""
    sub = treino[treino["arquivo"].isin(set(ids))]
    X_tr, y_tr = sub[cols].values, sub["classe_binaria"].values
    X_va, y_va = validacao[cols].values, validacao["classe_binaria"].values

    t0 = time.perf_counter()
    if chave == "rf":
        modelo = RandomForestClassifier(**params, random_state=semente_treino,
                                        n_jobs=-1)
        modelo.fit(X_tr, y_tr)
        scores = modelo.predict_proba(X_va)[:, 1]
    else:
        modelo = montar_pipeline(semente_treino)
        modelo.set_params(**params)
        modelo.fit(X_tr, y_tr)
        scores = modelo.decision_function(X_va)
    t_treino = time.perf_counter() - t0

    # Protocolo de sempre: limiar selecionado na VALIDAÇÃO, regra `score >=`.
    sel = selecionar_limiar(y_va, scores, criterio="f1_macro",
                            conjunto="validacao")
    m = avaliar(y_va, scores, f"{chave}_subamostra", limiar=sel["limiar"])
    return {
        "n_treino": int(len(sub)),
        "limiar": round(float(sel["limiar"]), 6),
        "f1_macro": round(float(m["f1_macro"]), 4),
        "eer": round(float(m["eer"]), 4),
        "roc_auc": round(float(roc_auc_score(y_va, scores)), 4),
        "recall_bonafide": round(float(m["recall_bonafide"]), 4),
        "tempo_treino_s": round(t_treino, 2),
    }


def resumir(pontos: list[dict], metrica: str) -> dict:
    v = np.array([p[metrica] for p in pontos], dtype=float)
    return {"media": round(float(v.mean()), 4),
            "desvio": round(float(v.std(ddof=1)), 4),
            "min": round(float(v.min()), 4),
            "max": round(float(v.max()), 4),
            "amplitude": round(float(v.max() - v.min()), 4)}


def main() -> None:
    cfg = carregar_config(RAIZ)
    semente_treino = fixar_seeds(cfg["semente"])
    alvo = int(cfg["experimento"]["tamanho_subamostra"])

    treino, validacao, cols, colunas_estrato = preparar_treino_e_validacao(cfg)
    print(f"treino completo {treino.shape} | validação {validacao.shape} | "
          f"{len(cols)} features")

    guarda = guarda_semente_42(treino, colunas_estrato, alvo, cfg)

    # ---- Hiperparâmetros vencedores: LIDOS, nunca rebuscados ----------------
    with open(DIR_MET / "rf_random_search.json", encoding="utf-8") as f:
        params_rf = json.load(f)["melhor"]["params"]
    with open(DIR_MET / "svm_random_search.json", encoding="utf-8") as f:
        params_svm = json.load(f)["melhor"]["params"]
    print(f"RF : {params_rf}")
    print(f"SVM: {params_svm}")

    # ---- Subamostra oficial (semente 42) como 1º ponto ----------------------
    oficial = pd.read_csv(RAIZ / cfg["experimento"]["caminho_subamostra"])
    subamostras = [{"semente": cfg["semente"], "oficial": True,
                    "arquivo": cfg["experimento"]["caminho_subamostra"],
                    "ids": oficial["arquivo"]}]

    # ---- Alternativas: só a semente muda ------------------------------------
    DIR_ALT.mkdir(parents=True, exist_ok=True)
    for s in SEMENTES_ALTERNATIVAS:
        amostra, _, _ = montar_subamostra(treino, colunas_estrato, alvo,
                                          semente=s)
        destino = DIR_ALT / f"subamostra_30k_seed{s}.csv"
        amostra.to_csv(destino, index=False)
        subamostras.append({
            "semente": s, "oficial": False,
            "arquivo": str(destino.relative_to(RAIZ)).replace("\\", "/"),
            "ids": amostra["arquivo"],
            "md5": hashlib.md5(destino.read_bytes()).hexdigest(),
        })
        print(f"gerada: {destino.name} ({len(amostra)} IDs)")

    # ---- Treinar RF e SVM em cada subamostra --------------------------------
    pontos = {"rf": [], "svm": []}
    for s in subamostras:
        print(f"\n--- subamostra semente {s['semente']}"
              f"{' (OFICIAL)' if s['oficial'] else ''} ---")
        for chave, params in (("rf", params_rf), ("svm", params_svm)):
            r = treinar_e_avaliar(chave, params, s["ids"], treino, validacao,
                                  cols, semente_treino)
            r["semente_subamostra"] = s["semente"]
            r["subamostra_oficial"] = s["oficial"]
            pontos[chave].append(r)
            print(f"  {chave.upper():<4} f1_macro {r['f1_macro']:.4f} | "
                  f"EER {r['eer']:.4f} | AUC {r['roc_auc']:.4f} | "
                  f"limiar {r['limiar']:.4f} | {r['tempo_treino_s']:.1f}s")

    resumos = {chave: {met: resumir(pontos[chave], met)
                       for met in ("f1_macro", "eer", "roc_auc", "limiar")}
               for chave in pontos}

    # ---- Leitura crítica ----------------------------------------------------
    with open(FONTE_ESTABILIDADE, encoding="utf-8") as f:
        estab = json.load(f)
    # Os números (b) e (c) são LIDOS do artefato de estabilidade, por caminho
    # explícito: os três termos da comparação têm de vir de medições, não de
    # memória, e um caminho errado tem de estourar em vez de devolver o número
    # de outra coisa.
    desvio_sementes = _ler_caminho(estab, "rf_sementes.f1_macro.desvio")
    delta_rf_svm = abs(_ler_caminho(
        estab, "bootstrap_pareado_svm_menos_rf.delta_f1_macro.media"))

    desvio_sub_rf = resumos["rf"]["f1_macro"]["desvio"]
    desvio_sub_svm = resumos["svm"]["f1_macro"]["desvio"]
    desvio_sub_max = max(desvio_sub_rf, desvio_sub_svm)

    print("\n" + "=" * 74)
    print("LEITURA CRÍTICA — qual fonte de variação domina o braço principal?")
    print("=" * 74)
    print(f"(a) desvio entre SUBAMOSTRAS (f1_macro): RF {desvio_sub_rf:.4f} | "
          f"SVM {desvio_sub_svm:.4f}")
    print(f"(b) desvio entre SEMENTES DE TREINO do RF (f1_macro): "
          f"{desvio_sementes:.4f}   [estabilidade_rf_svm.json]")
    print(f"(c) distância RF x SVM (delta f1_macro): "
          f"{delta_rf_svm:.4f}   [bootstrap pareado]")

    partes = []
    if True:
        razao = desvio_sub_max / float(desvio_sementes)
        if razao >= 2:
            partes.append(
                f"A dispersão entre subamostras ({desvio_sub_max:.4f}) é "
                f"{razao:.1f}x a dispersão entre sementes de treino "
                f"({float(desvio_sementes):.4f}). RESULTADO A DECLARAR: a semente "
                "das árvores era a fonte de variação ERRADA de se olhar; a "
                "incerteza real do braço principal vem de qual subamostra de 30k "
                "caiu. A análise por sementes continua válida — ela apenas "
                "responde a outra pergunta.")
        else:
            partes.append(
                f"A dispersão entre subamostras ({desvio_sub_max:.4f}) é da mesma "
                f"ordem da dispersão entre sementes de treino "
                f"({float(desvio_sementes):.4f}): a estratificação por "
                "classe x codec x ataque tornou a escolha da subamostra quase "
                "irrelevante, o que é, por si só, um resultado sobre a qualidade "
                "do desenho amostral.")
    if True:
        if float(delta_rf_svm) > desvio_sub_max:
            partes.append(
                f"E a distância RF x SVM ({float(delta_rf_svm):.4f}) CONTINUA "
                f"MAIOR que a dispersão entre subamostras ({desvio_sub_max:.4f}): "
                "a vantagem do SVM não é um acidente de qual subamostra caiu. A "
                "conclusão do Bloco 3 fica mais forte, não mais fraca.")
        else:
            partes.append(
                f"ATENÇÃO: a distância RF x SVM ({float(delta_rf_svm):.4f}) é "
                f"MENOR que a dispersão entre subamostras ({desvio_sub_max:.4f}). "
                "Pela regra do item 6 do protocolo do orientador, isto TEM de ser "
                "dito no texto: a diferença entre os modelos não sobrevive à "
                "variação da subamostra, e a conclusão do Bloco 3 precisa ser "
                "reformulada com essa ressalva explícita.")
    veredito = " ".join(partes)
    print(f"\n{veredito}\n")

    registro = {
        "analise": "estabilidade_subamostra",
        "data": date.today().isoformat(),
        "pergunta": ("quanto do resultado do braço principal depende de QUAL "
                     "subamostra de 30k caiu?"),
        "conjunto": "validacao",
        "teste_lacrado": True,
        "metodo": (
            "3 subamostras alternativas (sementes 43, 44, 45) geradas pela MESMA "
            "função montar_subamostra de scripts/gerar_subamostra.py — mesma "
            "estratificação classe x codec x ataque, mesma regra de maior resto, "
            "mesmo piso. Só a semente varia. RF e SVM treinados com os MESMOS "
            "hiperparâmetros vencedores, sem nova busca (um fator por vez). "
            "Limiar selecionado na validação em cada ponto, pelo protocolo."),
        "guarda_semente_42": guarda,
        "sementes": [cfg["semente"], *SEMENTES_ALTERNATIVAS],
        "hiperparametros": {"rf": params_rf, "svm": params_svm},
        "subamostras": [{k: v for k, v in s.items() if k != "ids"}
                        for s in subamostras],
        "pontos": pontos,
        "resumo": resumos,
        "comparacao_de_fontes_de_variacao": {
            "desvio_entre_subamostras_f1_macro": {"rf": desvio_sub_rf,
                                                  "svm": desvio_sub_svm},
            "desvio_entre_sementes_de_treino_rf_f1_macro": desvio_sementes,
            "distancia_rf_svm_f1_macro": delta_rf_svm,
            "fonte_b_e_c": "results/metricas/estabilidade_rf_svm.json",
        },
        "leitura_critica": veredito,
        "regra_do_protocolo": (
            "item 6 do protocolo do orientador: se a diferença entre os modelos "
            "for menor que a dispersão medida, isso tem de ser DISCUTIDO no "
            "texto, não escondido"),
        "hashes_md5": {
            "features": hashlib.md5(
                (RAIZ / "data" / "features" / "features.csv").read_bytes()).hexdigest(),
            "split": hashlib.md5(
                (RAIZ / "data" / "processed" / "split.csv").read_bytes()).hexdigest(),
            "subamostra_oficial": guarda["md5_oficial"],
        },
        "ambiente": {"python": platform.python_version(),
                     "sistema": f"{platform.system()} {platform.release()}"},
    }
    caminho = DIR_MET / "estabilidade_subamostra.json"
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(registro, f, indent=2, ensure_ascii=False,
                  default=json_seguro)
    print(f"Salvo em {caminho.relative_to(RAIZ)}")


def _ler_caminho(obj: dict, caminho: str) -> float:
    """Lê um número por caminho pontilhado, estourando se o caminho não existir.

    Existe para que os três termos da comparação venham de artefatos medidos.
    Um KeyError explícito é MUITO melhor que uma busca heurística que devolve o
    número de outro campo e produz uma frase confiante e errada no texto.
    """
    atual = obj
    for parte in caminho.split("."):
        if not isinstance(atual, dict) or parte not in atual:
            raise KeyError(
                f"'{caminho}' não existe em {FONTE_ESTABILIDADE.name} (parou em "
                f"'{parte}'). O esquema do artefato mudou: ajuste o caminho em "
                "vez de deixar a leitura crítica adivinhar.")
        atual = atual[parte]
    if isinstance(atual, bool) or not isinstance(atual, (int, float)):
        raise TypeError(f"'{caminho}' não é um número: {atual!r}")
    return float(atual)


if __name__ == "__main__":
    main()
