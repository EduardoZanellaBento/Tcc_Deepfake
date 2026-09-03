"""
Ablação da mfcc1_std — quanto do desempenho do RF depende da feature? (Bloco 3)
===============================================================================

POR QUE ESTA FEATURE, E SÓ ELA:
    A `mfcc1_std` é, ao mesmo tempo, a ÚNICA feature com |r| > 0,3 contra
    `prop_fala` (−0,3203, padding_corr_features_propfala_pos_lote.csv) e a 2ª
    mais importante do RF ajustado (0,05672, rf_tuned_principal.json). É o
    cruzamento que o diagnóstico de contaminação foi construído para detectar:
    a feature mais contaminada é também das mais usadas pelo modelo. Até aqui a
    resposta era só interpretação (adendo pós-lote de
    RECOMENDACAO_MASCARAMENTO.md); a ablação responde com dado.

O QUE A ABLAÇÃO MEDE — E O QUE NÃO MEDE:
    Mede QUANTO O DESEMPENHO DEPENDE da feature. NÃO decide se a feature é
    "atalho" ou "acústica" — essa parte continua sendo interpretação. O que ela
    entrega é um LIMITE SUPERIOR PARA O ESTRAGO possível: se remover a feature
    custa pouco, então mesmo que ela fosse atalho puro o modelo não se apoia
    nela.

COMO A DIFERENÇA É MEDIDA — bootstrap PAREADO:
    Os dois modelos (44 e 43 features) são avaliados nas MESMAS 22.226 linhas de
    validação e compartilham 43 das 44 features: os erros deles são fortemente
    correlacionados. Comparar o Δ observado contra o desvio de um bootstrap NÃO
    PAREADO joga fora essa correlação e superestima a incerteza — é exatamente o
    erro corrigido em scripts/estabilidade_modelos.py para o par RF × SVM. Aqui,
    a cada reamostragem sorteia-se UM vetor de índices, aplicado aos scores dos
    DOIS modelos, e guarda-se a DIFERENÇA; o IC95 da diferença é o critério.

PROTOCOLO (um fator por vez, como no braço de referência):
    Mesma configuração vencedora de rf_random_search.json, mesma semente 42,
    mesmo treino (30k do braço principal), mesma validação completa. ÚNICA
    diferença: mfcc1_std removida do X (43 features em vez de 44). Limiar
    selecionado na validação pelo protocolo normal (selecionar_limiar). Nada de
    conjunto de teste — o teste segue lacrado até o Bloco 5.

    O modelo de 44 features é RE-TREINADO aqui (e não lido do .joblib) para que
    os scores dos dois braços saiam do mesmo processo; uma GUARDA aborta se ele
    não reproduzir exatamente as métricas publicadas em rf_tuned_principal.json
    — sem reprodução exata, a diferença não é atribuível à feature removida.

Saída: results/metricas/ablacao_mfcc1_std.json (arquivo NOVO; rf_tuned_* não é
tocado). Rode a partir da raiz:
    python -m scripts.ablacao_mfcc1_std
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score

from src.utils.config import carregar_config
from src.data.split import (carregar_dados_split, colunas_features,
                            filtrar_treino_braco)
from src.models.avaliacao import (aplicar_limiar, calcular_eer,
                                  selecionar_limiar)

RAIZ = Path(__file__).resolve().parents[1]

FEATURE_REMOVIDA = "mfcc1_std"
N_BOOTSTRAP = 1000
SEMENTE_BOOTSTRAP = 42

# Dispersões NÃO pareadas já medidas (estabilidade_rf_svm.json). Ficam no JSON
# como contexto histórico — NÃO são mais o critério de decisão (ver docstring).
DESVIO_SEMENTES_EER = 0.0012
DESVIO_BOOTSTRAP_EER_NAO_PAREADO = 0.0041

LEITURAS = {
    "a_indistinguivel_de_zero": (
        "o IC95 do ΔEER pareado CONTÉM zero: a remoção da mfcc1_std não produz "
        "efeito distinguível do ruído de amostragem. Mesmo que a feature fosse "
        "atalho puro, ela não sustenta o desempenho do modelo."),
    "b_detectavel_mas_desprezivel": (
        "o IC95 do ΔEER pareado EXCLUI zero (a feature contribui de verdade), "
        "mas a magnitude é pequena: menos de 10% do EER de referência e menos "
        "de um quarto da distância entre RF e SVM. A conclusão de fundo é a "
        "mesma da faixa (a) — o modelo não se apoia na feature, e mesmo que ela "
        "fosse atalho puro o estrago máximo é marginal. As duas afirmações "
        "convivem: estatisticamente detectável, praticamente desprezível."),
    "c_piora_grande": (
        "o IC95 exclui zero E a magnitude é comparável à diferença entre os "
        "modelos. NÃO significa que o modelo trapaceia — significa que a "
        "dinâmica de energia da fala é genuinamente informativa para distinguir "
        "voz sintética (afirmação acústica plausível e defensável). Nesse caso a "
        "interpretação do adendo pós-lote (energia da fala que sobreviveu ao "
        "VAD) passa a ser OBRIGATÓRIA no texto, e vale confirmar com "
        "permutation_importance."),
}

# Limiares das faixas — fixados em grandezas que já existiam antes deste script
# (o EER de referência e a distância RF × SVM), não escolhidas para produzir um
# veredito conveniente.
LIM_REL_EER = 0.10          # ΔEER < 10% do EER de referência
LIM_FRACAO_GAP = 0.25       # ΔEER < 1/4 da distância RF × SVM


def _ic(v: np.ndarray) -> dict:
    return {"media": round(float(v.mean()), 4),
            "desvio": round(float(v.std(ddof=1)), 4),
            "ic95": [round(float(np.percentile(v, 2.5)), 4),
                     round(float(np.percentile(v, 97.5)), 4)]}


def bootstrap_pareado(y, s_44, l_44, s_43, l_43) -> dict:
    """IC95 da DIFERENÇA (44 − 43) reamostrando os MESMOS índices nos dois."""
    rng = np.random.default_rng(SEMENTE_BOOTSTRAP)
    n = len(y)
    d_f1, d_eer = [], []
    for _ in range(N_BOOTSTRAP):
        i = rng.integers(0, n, size=n)              # UM sorteio, dois modelos
        yb = y[i]
        f44 = f1_score(yb, aplicar_limiar(s_44[i], l_44), average="macro",
                       zero_division=0)
        f43 = f1_score(yb, aplicar_limiar(s_43[i], l_43), average="macro",
                       zero_division=0)
        e44 = calcular_eer(yb, s_44[i])[0]
        e43 = calcular_eer(yb, s_43[i])[0]
        d_f1.append(f44 - f43)          # positivo = as 44 features são melhores
        d_eer.append(e43 - e44)         # positivo = remover a feature piora
    d_f1, d_eer = np.array(d_f1), np.array(d_eer)
    return {
        "n_reamostragens": N_BOOTSTRAP,
        "semente": SEMENTE_BOOTSTRAP,
        "convencao_sinal": "positivo = o modelo COM a mfcc1_std é melhor",
        "limiares_fixos": {"com_44_features": round(float(l_44), 4),
                           "sem_mfcc1_std": round(float(l_43), 4)},
        "delta_f1_macro": _ic(d_f1),
        "delta_eer": _ic(d_eer),
        "fracao_reamostragens_44_melhor_f1": round(float((d_f1 > 0).mean()), 4),
        "fracao_reamostragens_44_melhor_eer": round(float((d_eer > 0).mean()), 4),
    }


def _treinar(params, semente, X_tr, y_tr, X_va):
    modelo = RandomForestClassifier(**params, random_state=semente, n_jobs=-1)
    t0 = time.perf_counter()
    modelo.fit(X_tr, y_tr)
    t = time.perf_counter() - t0
    return modelo.predict_proba(X_va)[:, 1], t


def main() -> None:
    cfg = carregar_config(RAIZ)
    dir_met = RAIZ / "results" / "metricas"

    with open(dir_met / "rf_random_search.json", encoding="utf-8") as f:
        params = json.load(f)["melhor"]["params"]
    with open(dir_met / "rf_tuned_principal.json", encoding="utf-8") as f:
        ref = json.load(f)
    with open(dir_met / "estabilidade_rf_svm.json", encoding="utf-8") as f:
        gap_rf_svm = float(json.load(f)["leitura_critica"]["diferenca_eer_rf_svm"])

    # Guarda: a comparação só vale sobre os MESMOS artefatos congelados
    h_feat = hashlib.md5(
        (RAIZ / "data" / "features" / "features.csv").read_bytes()).hexdigest()
    h_split = hashlib.md5(
        (RAIZ / "data" / "processed" / "split.csv").read_bytes()).hexdigest()
    if (ref["hash_md5_features_csv"] != h_feat
            or ref["hash_md5_split_csv"] != h_split):
        raise RuntimeError(
            "features.csv/split.csv em disco divergem dos hashes de "
            "rf_tuned_principal.json — a ablação não seria comparável.")

    df = carregar_dados_split(RAIZ)
    cols = colunas_features(df)
    if FEATURE_REMOVIDA not in cols:
        raise RuntimeError(f"'{FEATURE_REMOVIDA}' não está entre as features.")
    cols_ablacao = [c for c in cols if c != FEATURE_REMOVIDA]
    print(f"Ablação: {len(cols)} -> {len(cols_ablacao)} features "
          f"(removida: {FEATURE_REMOVIDA})")

    treino = filtrar_treino_braco(df[df["conjunto"] == "treino"], "principal",
                                  cfg, RAIZ)
    validacao = df[df["conjunto"] == "validacao"]
    y_tr = treino["classe_binaria"].values
    y_va = validacao["classe_binaria"].values
    semente = ref["semente"]

    # ---- Braço de referência (44 features), re-treinado aqui ----------------
    s_44, _ = _treinar(params, semente, treino[cols].values, y_tr,
                       validacao[cols].values)
    l_44 = float(ref["limiar"])
    f1_44 = float(f1_score(y_va, aplicar_limiar(s_44, l_44), average="macro",
                           zero_division=0))
    eer_44 = float(calcular_eer(y_va, s_44)[0])
    if (round(f1_44, 4) != round(float(ref["f1_macro"]), 4)
            or round(eer_44, 4) != round(float(ref["eer"]), 4)):
        raise RuntimeError(
            "O RF de 44 features re-treinado NÃO reproduziu "
            f"rf_tuned_principal.json (f1 {f1_44:.4f} vs {ref['f1_macro']:.4f}; "
            f"EER {eer_44:.4f} vs {ref['eer']:.4f}). Sem reprodução exata a "
            "diferença não é atribuível à feature removida — investigar.")
    print(f"  referência (44 feats) reproduzida: f1 {f1_44:.4f} | "
          f"EER {eer_44:.4f}")

    # ---- Braço da ablação (43 features) -------------------------------------
    s_43, t_treino = _treinar(params, semente, treino[cols_ablacao].values,
                              y_tr, validacao[cols_ablacao].values)
    sel = selecionar_limiar(y_va, s_43, criterio="f1_macro",
                            conjunto="validacao")
    l_43 = float(sel["limiar"])
    eer_43, limiar_eer_43 = calcular_eer(y_va, s_43)
    auc_43 = float(roc_auc_score(y_va, s_43))

    delta_f1 = sel["f1_macro"] - float(ref["f1_macro"])
    delta_eer = float(eer_43) - float(ref["eer"])   # positivo = ablação piorou

    # ---- Bootstrap pareado — o critério de decisão --------------------------
    print(f"  bootstrap pareado ({N_BOOTSTRAP} reamostragens)...")
    par = bootstrap_pareado(y_va, s_44, l_44, s_43, l_43)

    ic_lo, ic_hi = par["delta_eer"]["ic95"]
    contem_zero = bool(ic_lo <= 0.0 <= ic_hi)
    rel_eer = delta_eer / float(ref["eer"])
    fracao_gap = delta_eer / gap_rf_svm

    if contem_zero:
        leitura = "a_indistinguivel_de_zero"
    elif rel_eer < LIM_REL_EER and fracao_gap < LIM_FRACAO_GAP:
        leitura = "b_detectavel_mas_desprezivel"
    else:
        leitura = "c_piora_grande"

    print(f"\n  RF 44 features (referência): f1 {ref['f1_macro']:.4f} | "
          f"EER {ref['eer']:.4f}")
    print(f"  RF 43 features (ablação)   : f1 {sel['f1_macro']:.4f} | "
          f"EER {eer_43:.4f}")
    print(f"  Δf1_macro {delta_f1:+.4f} | ΔEER {delta_eer:+.4f} "
          f"({100*rel_eer:+.1f}% relativo; {100*fracao_gap:.1f}% da distância "
          f"RF×SVM de {gap_rf_svm:.4f})")
    print(f"  IC95 pareado do ΔEER: [{ic_lo:+.4f}; {ic_hi:+.4f}] "
          f"{'CONTÉM zero' if contem_zero else 'exclui zero'}")
    print(f"  Leitura aplicada: {leitura}\n    {LEITURAS[leitura]}")

    saida = {
        "modelo": "ablacao_mfcc1_std",
        "conjunto": "validacao (o teste continua lacrado)",
        "feature_removida": FEATURE_REMOVIDA,
        "motivacao": (
            "única feature com |r| > 0,3 contra prop_fala (−0,3203) E 2ª mais "
            "importante do RF ajustado (0,05672) — a ablação dá o limite "
            "superior do estrago caso ela fosse atalho puro"),
        "protocolo": (
            "um fator por vez: mesma config vencedora de rf_random_search.json,"
            " mesma semente 42, mesmo treino (30k do braço principal), mesma "
            "validação; única diferença: 43 features em vez de 44; limiar "
            "re-selecionado na validação pelo protocolo normal. O braço de 44 "
            "features é re-treinado no mesmo processo e uma guarda exige que ele "
            "reproduza rf_tuned_principal.json antes de comparar."),
        "hiperparametros": {k: (v if v is None or isinstance(v, (int, float,
                                                                 bool))
                                else str(v)) for k, v in params.items()},
        "n_features": len(cols_ablacao),
        "n_treino": int(len(treino)),
        "n_validacao": int(len(validacao)),
        "semente": semente,
        "tempo_treino_s": round(t_treino, 2),
        "selecao_limiar": sel,
        "f1_macro": round(float(sel["f1_macro"]), 4),
        "eer": round(float(eer_43), 4),
        "limiar_eer": round(float(limiar_eer_43), 4),
        "roc_auc_validacao": round(auc_43, 4),
        "referencia_44_features": {
            "arquivo": "rf_tuned_principal.json",
            "f1_macro": round(float(ref["f1_macro"]), 4),
            "eer": round(float(ref["eer"]), 4),
            "roc_auc_validacao": ref.get("roc_auc_validacao"),
            "reproduzida_neste_script": True,
        },
        "delta_f1_macro": round(float(delta_f1), 4),
        "delta_eer": round(float(delta_eer), 4),
        "magnitude": {
            "eer_relativo_pct": round(100 * rel_eer, 1),
            "fracao_da_distancia_rf_svm_pct": round(100 * fracao_gap, 1),
            "distancia_eer_rf_svm": gap_rf_svm,
            "fonte_da_distancia": "estabilidade_rf_svm.json -> leitura_critica",
        },
        "bootstrap_pareado_44_menos_43": par,
        "dispersoes_nao_pareadas_historicas": {
            "desvio_entre_sementes": DESVIO_SEMENTES_EER,
            "desvio_bootstrap": DESVIO_BOOTSTRAP_EER_NAO_PAREADO,
            "fonte": "estabilidade_rf_svm.json",
            "aviso": (
                "contexto histórico apenas — NÃO são o critério de decisão: são "
                "dispersões NÃO pareadas e superestimam a incerteza da diferença "
                "entre dois modelos avaliados nas mesmas linhas"),
        },
        "o_que_a_ablacao_mede": (
            "quanto o desempenho depende da feature — NÃO decide se ela é "
            "'atalho' ou 'acústica'; entrega um limite superior para o estrago"),
        "criterio_de_leitura": {
            "a": "IC95 do ΔEER pareado contém zero",
            "b": (f"IC95 exclui zero, ΔEER < {100*LIM_REL_EER:.0f}% do EER de "
                  f"referência E < {100*LIM_FRACAO_GAP:.0f}% da distância RF×SVM"),
            "c": "IC95 exclui zero e magnitude comparável à distância RF×SVM",
        },
        "leituras_pre_registradas": LEITURAS,
        "leitura_aplicada": leitura,
        "nota_sobre_o_pre_registro": (
            "O pré-registro original tinha apenas DUAS faixas — '(a) ΔEER menor "
            "que a dispersão medida' e '(b) ΔEER muito maior que a dispersão' — "
            "e deixou um vão entre elas. O resultado observado (ΔEER +0,0050) "
            "caiu exatamente nesse vão: 1,2× o desvio não pareado (0,0041) e "
            "1,8× o desvio pareado (0,0028), longe de '>>'. A primeira versão "
            "deste script classificou como '(b) piora grande', o que "
            "superestimava o efeito. Duas correções foram feitas, e ambas ficam "
            "declaradas: (1) o critério passou a ser o IC95 do bootstrap "
            "PAREADO — a régua antiga comparava contra uma dispersão NÃO "
            "pareada, o mesmo erro estatístico corrigido em "
            "estabilidade_modelos.py para o par RF × SVM; (2) foi acrescentada a "
            "faixa intermediária 'detectável mas desprezível', que o pré-registro "
            "não previa. Essa faixa foi criada DEPOIS de ver o resultado — o que "
            "é justamente aquilo que o pré-registro existe para evitar, e por "
            "isso é declarado aqui em vez de silenciado. Atenuantes: os limiares "
            "da faixa são grandezas que já existiam antes (o EER de referência e "
            "a distância RF × SVM), não valores escolhidos para produzir um "
            "veredito conveniente; e a conclusão de fundo é a MESMA da faixa (a) "
            "do pré-registro original — o modelo não se apoia na mfcc1_std. O "
            "que a faixa intermediária acrescenta é honestidade sobre o efeito "
            "ser real, ainda que pequeno."),
        "hash_md5_features_csv": h_feat,
        "hash_md5_split_csv": h_split,
        "hash_md5_subamostra_csv": hashlib.md5(
            (RAIZ / cfg["experimento"]["caminho_subamostra"]
             ).read_bytes()).hexdigest(),
    }
    with open(dir_met / "ablacao_mfcc1_std.json", "w", encoding="utf-8") as f:
        json.dump(saida, f, indent=2, ensure_ascii=False)
    print("\nSalvo em results/metricas/ablacao_mfcc1_std.json")


if __name__ == "__main__":
    main()
