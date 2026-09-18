"""
Split interno 27k/3k da CNN — treino interno × early stopping (P6)
==================================================================

O QUE ESTE SCRIPT FAZ:
    Parte a subamostra de 30.000 do braço principal em 27.000 (`treino_interno`)
    e 3.000 (`early_stopping`), pela MESMA regra de alocação da subamostra, e
    grava o resultado em disco para que ele seja auditável.

A DECISÃO QUE ELE IMPLEMENTA (P6, fechada em 17/09/2026):
    30k -> split interno FIXO 27k/3k, seed 42, estratificado por
    classe × codec × ataque -> estatísticas de normalização nos 27k -> treina e
    seleciona arquitetura, registrando a MELHOR ÉPOCA -> REFIT FINAL nos 30k
    completos, por nº FIXO de épocas. O refit existe para que a CNN final não
    treine em 27k enquanto RF e SVM treinaram em 30k: sem ele, a comparação —
    que é o coração do trabalho — ganharia um fator escondido de 3.000 amostras
    além do modelo.

POR QUE OS 3k NÃO SÃO A VALIDAÇÃO EXTERNA:
    A validação externa (22.226) é usada SÓ para escolher o limiar — é o
    protocolo que RF e SVM já seguem. Se o early stopping olhasse esses mesmos
    22.226, o limiar escolhido depois estaria sendo selecionado sobre um
    conjunto que já influenciou o treino. A validação externa NÃO é usada para
    early stopping em fase nenhuma.

POR QUE SALVAR EM DISCO E VERSIONAR:
    Mesmo motivo do `split.csv`: um split que mora só na memória do script
    transforma "a CNN parou na época 14" numa afirmação não verificável. Com o
    arquivo versionado, a época 14 é reproduzível.

A REGRA DE ALOCAÇÃO NÃO É COPIADA — É IMPORTADA:
    `montar_subamostra` (que por sua vez chama `alocar_maior_resto`) vem de
    scripts/gerar_subamostra.py. Proporcional por maior resto, piso de 1 por
    estrato, amostragem por estrato com random_state=42. Ela vale aqui pelo
    mesmo motivo de lá: 3.000 divididos em 98 estratos dá ~30 por estrato EM
    MÉDIA, mas os estratos são muito desiguais, e o arredondamento ingênuo
    (`round(len(g) * 0.1)`) zera os pequenos — justamente os ataques raros, que
    são os mais informativos para o early stopping. Duas cópias da mesma regra
    seriam a divergência silenciosa que o projeto vem evitando.

SAÍDAS:
    data/processed/split_interno_cnn.csv    [arquivo, particao_interna],
                                            ordenado por arquivo (determinístico
                                            byte a byte)
    results/metricas/split_interno_cnn.json n por partição, hashes, regra

Rode a partir da raiz:  python -m scripts.gerar_split_interno_cnn
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.gerar_subamostra import montar_subamostra
from src.utils.config import carregar_config
from src.utils.seeds import fixar_seeds

RAIZ = Path(__file__).resolve().parents[1]

# P6, fechada: 3.000 para early stopping, o resto (27.000) para treino interno.
# NÃO é parâmetro de ajuste — é decisão metodológica registrada. Fica aqui, e
# não no config.yaml, pelo mesmo motivo do PISO de gerar_subamostra.py: o config
# é para o que se escolhe, não para o que já foi fechado.
N_EARLY_STOPPING = 3000
SAIDA_CSV = "data/processed/split_interno_cnn.csv"
SAIDA_JSON = "results/metricas/split_interno_cnn.json"


def _md5(caminho: Path) -> str:
    return hashlib.md5(caminho.read_bytes()).hexdigest()


def _razao(df: pd.DataFrame) -> float:
    """spoof:bonafide — a razão 9,0:1 que é propriedade do protocolo (P7)."""
    return (df["classe_binaria"] == 1).sum() / (df["classe_binaria"] == 0).sum()


def main() -> None:
    cfg = carregar_config(RAIZ)
    semente = fixar_seeds(cfg["semente"])
    colunas_estrato = list(cfg["experimento"]["estratificacao_subamostra"])
    caminho_sub = RAIZ / cfg["experimento"]["caminho_subamostra"]

    sub = pd.read_csv(caminho_sub)

    # `codec` e `ataque` vêm do labels.csv (merge por `arquivo`), como manda o
    # marco. A subamostra já carrega as mesmas colunas — então o merge é, de
    # graça, uma CONFERÊNCIA: se as duas fontes discordarem, os estratos daqui
    # não seriam os mesmos 98 de lá, e isso tem de explodir, não passar batido.
    labels = pd.read_csv(RAIZ / "data" / "processed" / "labels.csv",
                         usecols=["arquivo", "fase", *colunas_estrato])
    m = sub[["arquivo"]].merge(labels, on="arquivo", how="inner")
    if len(m) != len(sub):
        raise SystemExit("FALHA: merge subamostra x labels perdeu linhas.")
    if not (m["fase"] == "eval").all():
        raise SystemExit("FALHA: subamostra contém linhas fora do universo eval.")
    conferencia = (m[["arquivo", *colunas_estrato]]
                   .sort_values("arquivo").reset_index(drop=True))
    esperado = (sub[["arquivo", *colunas_estrato]]
                .sort_values("arquivo").reset_index(drop=True))
    if not conferencia.equals(esperado):
        raise SystemExit("FALHA: estratos do labels.csv divergem da subamostra.")

    print(f"Subamostra do braço principal: {len(sub)} | "
          f"alvo do early stopping: {N_EARLY_STOPPING}")

    # ---- Estratos: os MESMOS 98 da subamostra --------------------------------
    sub["estrato"] = sub[colunas_estrato].astype(str).agg("|".join, axis=1)

    # A regra (maior resto + piso 1 + amostragem por estrato) é importada.
    # Aqui ela aloca os 3.000 do early stopping; o complemento é o treino interno.
    es, aloc, tamanhos = montar_subamostra(sub, colunas_estrato,
                                           N_EARLY_STOPPING, semente)
    ids_es = set(es["arquivo"])

    split = pd.DataFrame({
        "arquivo": sub["arquivo"],
        "particao_interna": np.where(sub["arquivo"].isin(ids_es),
                                     "early_stopping", "treino_interno"),
    }).sort_values("arquivo").reset_index(drop=True)

    n_es = int((split["particao_interna"] == "early_stopping").sum())
    n_ti = int((split["particao_interna"] == "treino_interno").sum())
    print(f"Estratos: {len(tamanhos)} (menor: {tamanhos.min()}, "
          f"maior: {tamanhos.max()}) | alocação no early stopping: "
          f"menor {aloc.min()}, maior {aloc.max()}")

    # ---- Conferências do marco (falham alto, não em silêncio) ----------------
    if n_ti + n_es != len(sub):
        raise SystemExit(f"FALHA: {n_ti} + {n_es} != {len(sub)}.")
    if n_es != N_EARLY_STOPPING:
        raise SystemExit(f"FALHA: early stopping ficou com {n_es}, "
                         f"esperado {N_EARLY_STOPPING}.")
    ids_ti = set(split.loc[split["particao_interna"] == "treino_interno",
                           "arquivo"])
    if ids_es & ids_ti:
        raise SystemExit("FALHA: as partições se sobrepõem.")
    if set(split["arquivo"]) - set(sub["arquivo"]):
        raise SystemExit("FALHA: há arquivo no split interno fora da subamostra.")
    if split["arquivo"].duplicated().any():
        raise SystemExit("FALHA: `arquivo` duplicado no split interno.")

    # ---- Composição por estrato ----------------------------------------------
    rotulado = sub.merge(split, on="arquivo")
    p_sub = 100 * tamanhos / tamanhos.sum()
    desvios = {}
    for part in ("treino_interno", "early_stopping"):
        cont = (rotulado[rotulado["particao_interna"] == part]["estrato"]
                .value_counts().reindex(tamanhos.index).fillna(0))
        desvios[part] = float((100 * cont / cont.sum() - p_sub).abs().max())
        print(f"  {part}: n={int(cont.sum())} | maior desvio de proporção "
              f"por estrato: {desvios[part]:.4f} p.p. | estratos vazios: "
              f"{int((cont == 0).sum())}")

    r_sub = _razao(sub)
    r_ti = _razao(rotulado[rotulado["particao_interna"] == "treino_interno"])
    r_es = _razao(rotulado[rotulado["particao_interna"] == "early_stopping"])
    print(f"razão spoof:bonafide — subamostra {r_sub:.4f}:1 | "
          f"treino_interno {r_ti:.4f}:1 | early_stopping {r_es:.4f}:1")

    saida = RAIZ / SAIDA_CSV
    split.to_csv(saida, index=False)
    print(f"Split interno salvo em {saida}")

    # ---- JSON de rastreabilidade (formato de subamostra_30k.json) ------------
    dir_met = RAIZ / "results" / "metricas"
    dir_met.mkdir(parents=True, exist_ok=True)
    registro = {
        "alvo_early_stopping": N_EARLY_STOPPING,
        "n_treino_interno": n_ti,
        "n_early_stopping": n_es,
        "n_total": int(len(split)),
        "semente": semente,
        "escopo": "somente a subamostra de 30.000 do braço principal; a validação "
                  "externa (22.226) NÃO é usada para early stopping em fase "
                  "nenhuma — ela serve só para escolher o limiar",
        "estratificacao": "classe_binaria x codec x ataque "
                          "(bonafide: ataque=='-' é o próprio estrato bonafide)",
        "n_estratos": int(len(tamanhos)),
        "regra_alocacao": "proporcional por maior resto (largest remainder); "
                          "piso de 1 por estrato, excedente retirado dos maiores; "
                          "amostragem por estrato com random_state=42 — importada "
                          "de scripts.gerar_subamostra.montar_subamostra, não copiada",
        "piso_por_estrato": 1,
        "piso_acionado": bool((np.floor(N_EARLY_STOPPING * tamanhos
                                        / tamanhos.sum()) == 0).any()),
        "maior_desvio_pct_pontos": round(max(desvios.values()), 4),
        "maior_desvio_pct_pontos_treino_interno": round(desvios["treino_interno"], 4),
        "maior_desvio_pct_pontos_early_stopping": round(desvios["early_stopping"], 4),
        "razao_spoof_bonafide_subamostra": round(r_sub, 4),
        "razao_spoof_bonafide_treino_interno": round(r_ti, 4),
        "razao_spoof_bonafide_early_stopping": round(r_es, 4),
        "hash_md5_split_interno_csv": _md5(saida),
        "hash_md5_subamostra_csv_origem": _md5(caminho_sub),
    }
    with open(RAIZ / SAIDA_JSON, "w", encoding="utf-8") as f:
        json.dump(registro, f, indent=2, ensure_ascii=False)
    print(f"JSON            : {RAIZ / SAIDA_JSON}")
    print(f"MD5 do split    : {registro['hash_md5_split_interno_csv']}")
    print("\nNENHUM modelo foi treinado. O refit do B4.6 volta aos 30k completos.")


if __name__ == "__main__":
    main()
