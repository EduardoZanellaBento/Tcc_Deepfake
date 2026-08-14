"""
Subamostra estratificada de ~30k do TREINO — insumo para RF × SVM × CNN
=======================================================================

SÓ GERA A LISTA DE IDs. Não treina nada: a lista é um dos itens que o
orientador pediu para revisar antes de autorizar os próximos passos.

DECISÃO DO ORIENTADOR QUE ESTE SCRIPT IMPLEMENTA:
    Subamostra estratificada ÚNICA (~30k), seed 42, salva em disco e
    COMPARTILHADA por RF, SVM e CNN — o SVM-RBF é O(n²)–O(n³) e não roda nos
    ~103k de treino; treinar os três modelos no MESMO subconjunto preserva o
    "mesmo ambiente experimental" da pergunta de pesquisa.

ESCOPO (explícito no texto do orientador):
    A subamostra se aplica SOMENTE ao conjunto de treino. "Validação e teste
    devem permanecer completos, sem redução, para que a avaliação final
    continue robusta."

ESTRATIFICAÇÃO (todos os requisitos textuais do orientador):
    classe_binaria × codec × ataque. Para bonafide, ataque == '-' — não é um
    estrato vazio, é o próprio estrato bonafide (7 codecs × bonafide + 7 codecs
    × 13 ataques de spoof = 98 estratos no universo eval).

PARÂMETROS (config.yaml, bloco `experimento`):
    alvo    <- experimento.tamanho_subamostra
    saída   <- experimento.caminho_subamostra
    estratos<- experimento.estratificacao_subamostra
    A constante PISO permanece no código de propósito: é regra de IMPLEMENTAÇÃO
    da alocação (nenhum estrato pode zerar), não parâmetro metodológico.

REGRA DE ALOCAÇÃO (documentação obrigatória):
    Proporcional ao tamanho de cada estrato no treino completo, pelo método do
    MAIOR RESTO (largest remainder): aloca floor(alvo * n_estrato / n_treino) a
    todos e distribui as sobras, uma a uma, aos estratos com maior parte
    fracionária — assim a soma fecha exatamente no alvo. PISO DE SEGURANÇA:
    nenhum estrato pode ficar com 0 amostras; se o arredondamento zerar algum,
    ele é forçado a 1 e o excedente é retirado, uma unidade por vez, dos
    estratos com maior alocação. O n final é registrado no JSON — pode divergir
    do alvo redondo e isso é reportado, não maquiado.

SAÍDAS:
    data/processed/subamostra_30k.csv               [arquivo, classe_binaria,
                                                     codec, ataque] ordenado
                                                     por arquivo (determinístico
                                                     byte a byte)
    results/metricas/subamostra_30k_composicao.csv  contagem por estrato, lado
                                                     a lado com o treino completo
    results/metricas/subamostra_30k.json            n, seed, hashes, regra
    results/figuras/subamostra_30k_composicao.png   distribuições comparadas

Rode a partir da raiz:  python -m scripts.gerar_subamostra
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.config import carregar_config
from src.utils.seeds import fixar_seeds

RAIZ = Path(__file__).resolve().parents[1]
PISO = 1   # regra de implementação (ver docstring) — não é parâmetro do config


def alocar_maior_resto(tamanhos: pd.Series, alvo: int, piso: int) -> pd.Series:
    """Alocação proporcional pelo maior resto, com piso por estrato."""
    exato = alvo * tamanhos / tamanhos.sum()
    aloc = np.floor(exato).astype(int)
    sobra = alvo - int(aloc.sum())
    # distribui as sobras aos maiores restos (desempate determinístico: índice)
    restos = (exato - aloc).sort_values(ascending=False, kind="stable")
    for chave in restos.index[:sobra]:
        aloc[chave] += 1

    # piso de segurança: nenhum estrato com menos que `piso`
    deficit = int((piso - aloc[aloc < piso]).sum())
    aloc[aloc < piso] = piso
    # retira o excedente, uma unidade por vez, de quem tem mais
    while deficit > 0:
        maior = aloc.idxmax()
        aloc[maior] -= 1
        deficit -= 1

    # nunca pedir mais do que o estrato tem
    estourados = aloc[aloc > tamanhos]
    if len(estourados):
        raise ValueError(f"Alocação excede o tamanho do estrato: {estourados}")
    return aloc


def main() -> None:
    cfg = carregar_config(RAIZ)
    semente = fixar_seeds(cfg["semente"])
    alvo = int(cfg["experimento"]["tamanho_subamostra"])
    colunas_estrato = list(cfg["experimento"]["estratificacao_subamostra"])
    saida = RAIZ / cfg["experimento"]["caminho_subamostra"]

    split = pd.read_csv(RAIZ / "data" / "processed" / "split.csv")
    labels = pd.read_csv(RAIZ / "data" / "processed" / "labels.csv",
                         usecols=["arquivo", "fase", *colunas_estrato])

    treino = split[split["conjunto"] == "treino"].merge(labels, on="arquivo",
                                                        how="inner")
    if len(treino) != int((split["conjunto"] == "treino").sum()):
        raise SystemExit("FALHA: merge treino × labels perdeu linhas — investigar.")
    if not (treino["fase"] == "eval").all():
        raise SystemExit("FALHA: split de treino contém linhas fora do universo eval.")

    print(f"Treino completo: {len(treino)} | alvo da subamostra: {alvo}")

    # ---- Estratos (config: experimento.estratificacao_subamostra) ------------
    treino["estrato"] = treino[colunas_estrato].astype(str).agg("|".join, axis=1)
    tamanhos = treino["estrato"].value_counts().sort_index()
    print(f"Estratos: {len(tamanhos)} (menor: {tamanhos.min()}, "
          f"maior: {tamanhos.max()})")

    aloc = alocar_maior_resto(tamanhos, alvo, PISO)

    # ---- Amostragem determinística por estrato -------------------------------
    partes = []
    for estrato in sorted(aloc.index):
        sub = treino[treino["estrato"] == estrato]
        partes.append(sub.sample(n=int(aloc[estrato]), random_state=semente))
    amostra = pd.concat(partes)

    amostra = amostra[["arquivo", *colunas_estrato]] \
        .sort_values("arquivo").reset_index(drop=True)
    n_final = len(amostra)
    print(f"n final da subamostra: {n_final} "
          f"({'exato' if n_final == alvo else 'diverge do alvo — registrado'})")

    amostra.to_csv(saida, index=False)
    print(f"IDs salvos em {saida}")

    # ---- Composição lado a lado (verificação da estratificação) --------------
    comp = pd.DataFrame({
        "n_treino_completo": tamanhos,
        "pct_treino_completo": (100 * tamanhos / tamanhos.sum()).round(4),
        "n_subamostra": aloc,
        "pct_subamostra": (100 * aloc / aloc.sum()).round(4),
    })
    comp.index.name = "estrato (classe|codec|ataque)"
    comp["delta_pct"] = (comp["pct_subamostra"] -
                         comp["pct_treino_completo"]).round(4)
    dir_met = RAIZ / "results" / "metricas"
    dir_met.mkdir(parents=True, exist_ok=True)
    comp.to_csv(dir_met / "subamostra_30k_composicao.csv")

    desvio_max = comp["delta_pct"].abs().max()
    print(f"maior desvio de proporção por estrato: {desvio_max:.4f} p.p.")

    # razões spoof:bonafide
    def razao(df):
        return (df["classe_binaria"] == 1).sum() / (df["classe_binaria"] == 0).sum()
    print(f"razão spoof:bonafide — treino completo {razao(treino):.4f}:1 | "
          f"subamostra {razao(amostra):.4f}:1")

    # ---- JSON de rastreabilidade ---------------------------------------------
    split_csv = RAIZ / "data" / "processed" / "split.csv"
    registro = {
        "alvo": alvo,
        "n_final": n_final,
        "semente": semente,
        "escopo": "somente conjunto=='treino'; validação e teste permanecem completos",
        "estratificacao": "classe_binaria x codec x ataque "
                          "(bonafide: ataque=='-' é o próprio estrato bonafide)",
        "n_estratos": int(len(tamanhos)),
        "regra_alocacao": "proporcional por maior resto (largest remainder); "
                          "piso de 1 por estrato, excedente retirado dos maiores; "
                          "amostragem por estrato com random_state=42",
        "piso_por_estrato": PISO,
        "piso_acionado": bool((np.floor(alvo * tamanhos / tamanhos.sum()) == 0).any()),
        "maior_desvio_pct_pontos": float(desvio_max),
        "razao_spoof_bonafide_treino": round(razao(treino), 4),
        "razao_spoof_bonafide_subamostra": round(razao(amostra), 4),
        "hash_md5_subamostra_csv": hashlib.md5(saida.read_bytes()).hexdigest(),
        "hash_md5_split_csv_origem": hashlib.md5(split_csv.read_bytes()).hexdigest(),
    }
    with open(dir_met / "subamostra_30k.json", "w", encoding="utf-8") as f:
        json.dump(registro, f, indent=2, ensure_ascii=False)

    # ---- Figura: distribuições comparadas -------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    for ax, coluna, titulo in [
        (axes[0], "codec", "por codec"),
        (axes[1], "ataque", "por ataque (- = bonafide)"),
    ]:
        p_full = 100 * treino[coluna].value_counts(normalize=True).sort_index()
        p_sub = 100 * amostra[coluna].value_counts(normalize=True) \
            .reindex(p_full.index).fillna(0)
        x = np.arange(len(p_full))
        ax.bar(x - 0.2, p_full.values, width=0.4, label="treino completo",
               color="#4c72b0")
        ax.bar(x + 0.2, p_sub.values, width=0.4, label=f"subamostra ({n_final})",
               color="#c44e52")
        ax.set_xticks(x, p_full.index, rotation=45, ha="right")
        ax.set_ylabel("% do conjunto")
        ax.set_title(f"Composição {titulo}")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Subamostra 30k × treino completo — estratificação preservada")
    fig.tight_layout()
    dir_fig = RAIZ / "results" / "figuras"
    dir_fig.mkdir(parents=True, exist_ok=True)
    fig.savefig(dir_fig / "subamostra_30k_composicao.png", dpi=150)
    plt.close(fig)

    print(f"\nComposição: {dir_met / 'subamostra_30k_composicao.csv'}")
    print(f"JSON      : {dir_met / 'subamostra_30k.json'}")
    print(f"Figura    : {dir_fig / 'subamostra_30k_composicao.png'}")
    print("\nNENHUM modelo foi treinado — a lista de IDs vai para revisão do "
          "orientador.")


if __name__ == "__main__":
    main()
