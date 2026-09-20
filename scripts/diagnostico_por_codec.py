"""
Diagnóstico por codec — RF, SVM e CNN AJUSTADOS
================================================

SÓ LÊ E REPORTA. Nenhum modelo é treinado aqui: os três modelos do braço
principal são CARREGADOS de models/ (dois .joblib e um .pt) e cada um decide com
o SEU limiar, lido de `selecao_limiar.limiar` no JSON companheiro. Avaliação na
VALIDAÇÃO. O split de teste continua LACRADO.

HIPÓTESE (a mesma desde 13/08/2026):
    Os 7 codecs do ASVspoof 2021 LA dividem-se em banda estreita (alaw, ulaw,
    gsm, pstn — teto ~4 kHz) e banda larga (g722, opus, none — preservam
    >4 kHz). Os artefatos de síntese que ZCR e centróide espectral capturam
    vivem principalmente na banda alta; codecs telefônicos a destroem. Logo,
    espera-se desempenho MELHOR nos codecs de banda larga e PIOR nos de banda
    estreita. Como codec é fator perfeitamente balanceado (25.938 cada, mesma
    proporção spoof/bonafide), a comparação é limpa.

POR QUE ESTA VERSÃO EXISTE (revisão de 03/09/2026):
    A versão anterior (preservada em *_baseline_2026-08-13.csv/png) re-treinava
    um RF *baseline* no treino completo e decidia por `modelo.predict()` —
    argmax em 0,50. Naquele limiar o modelo prevê "spoof" para quase tudo:
    recall_bonafide perto de zero em todos os codecs. Um contraste medido sob um
    limiar que satura pode ser artefato DO LIMIAR e não do codec. Esta versão
    responde à pergunta que a anterior não conseguia: a hipótese da banda alta
    se sustenta com o MODELO DO BRAÇO PRINCIPAL, sob o limiar do protocolo?

    Além disso, a versão anterior não tocava no SVM — que é o modelo VENCEDOR
    do braço principal entre os clássicos. Agora os três são medidos com o mesmo
    código.

A CNN ENTROU NO B4.7 (20/09/2026), SEM TRABALHO NOVO:
    ela passou a existir em `MODELOS_PRINCIPAIS`, e este script a carrega pelo
    mesmo `carregar_modelo_ajustado` — com as mesmas duas guardas (limiar
    presente; limiar vindo da validação). A única diferença é a ENTRADA: a CNN
    come um Dataset de espectrogramas em vez da matriz de 44 features, e
    `entrada_de_validacao` resolve isso num lugar só, reordenando o Dataset para
    casar linha a linha com a tabela deste script.

    E a CNN torna a hipótese da banda ALTA testável de um jeito que RF e SVM não
    tornavam: ela não usa ZCR nem centróide — vê o espectrograma inteiro até
    8 kHz. Se o contraste por banda aparecer nela também, a evidência passa a ser
    sobre o SINAL, não sobre aquelas duas features.

SAÍDAS:
    results/metricas/diagnostico_por_codec_{rf_tuned,svm_tuned,cnn_final}_principal.csv
    results/figuras/diagnostico_por_codec_{rf_tuned,svm_tuned,cnn_final}_principal.png
    results/metricas/diagnostico_por_codec_resumo.json
    + leitura crítica no stdout

Rode a partir da raiz:  python -m scripts.diagnostico_por_codec
"""

import json
import platform
from datetime import date
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score

from src.utils.config import carregar_config
from src.utils.serializacao import json_seguro
from src.data.split import carregar_dados_split, colunas_features
from src.models.avaliacao import aplicar_limiar, calcular_eer, REGRA_DECISAO
from src.models.modelos_ajustados import (
    MODELOS_PRINCIPAIS, carregar_modelo_ajustado, entrada_de_validacao,
    hashes_congelados, scores_de,
)

RAIZ = Path(__file__).resolve().parents[1]

BANDA = {
    "alaw": "estreita", "ulaw": "estreita", "gsm": "estreita", "pstn": "estreita",
    "g722": "larga", "opus": "larga", "none": "larga",
}
ORDEM_CODECS = ["alaw", "ulaw", "gsm", "pstn", "g722", "opus", "none"]

ESCALA = {"rf": "probabilidade [0,1] (predict_proba)",
          "svm": "decision_function (real, centrado em zero)",
          "cnn": "probabilidade [0,1] (softmax(logits)[:, 1] = P(spoof))"}


def preparar_validacao() -> tuple[pd.DataFrame, list[str]]:
    """Validação com features + codec, e a lista canônica de features.

    `colunas_features(df)` é calculado ANTES do merge com labels.csv (a mesma
    ordem da versão de 13/08, e pela mesma razão): depois do merge a tabela
    ganha colunas de metadado, e um X com colunas a mais ou em outra ordem faz o
    modelo carregado devolver lixo SEM levantar exceção.
    """
    df = carregar_dados_split(RAIZ)
    cols = colunas_features(df)   # ANTES do merge: só as 44 acústicas

    labels = pd.read_csv(RAIZ / "data" / "processed" / "labels.csv",
                         usecols=["arquivo", "codec", "trim", "fase"])
    n_antes = len(df)
    df = df.merge(labels, on="arquivo", how="inner")
    assert len(df) == n_antes, "merge perdeu linhas — investigar"

    # SÓ VALIDAÇÃO. O teste é lacrado (Bloco 5, execução única).
    return df[df["conjunto"] == "validacao"].copy(), cols


def diagnosticar(chave: str, validacao: pd.DataFrame, cols: list[str]) -> dict:
    """Tabela por codec de UM modelo, sob o limiar do protocolo."""
    carregado = carregar_modelo_ajustado(RAIZ, chave)
    limiar = carregado["limiar"]
    nome = carregado["nome_arquivo"]
    print(f"\n=== {carregado['rotulo']} | limiar {limiar:.4f} "
          f"({REGRA_DECISAO}) | fonte: {carregado['origem_limiar']} ===")

    # `entrada_de_validacao` entrega a matriz de features para RF/SVM e o
    # Dataset de espectrogramas REORDENADO para a CNN — ver a docstring dela.
    scores = scores_de(carregado, entrada_de_validacao(carregado, RAIZ,
                                                       validacao, cols))
    # Regra única do protocolo — nada de modelo.predict().
    y_pred = aplicar_limiar(scores, limiar)
    y_va = validacao["classe_binaria"].values

    linhas = []
    for codec in ORDEM_CODECS:
        m = (validacao["codec"] == codec).values
        yt, yp, sc = y_va[m], y_pred[m], scores[m]
        eer, _ = calcular_eer(yt, sc)
        linhas.append({
            "codec": codec,
            "banda": BANDA[codec],
            "n": int(m.sum()),
            "n_bonafide": int((yt == 0).sum()),
            "f1_macro": round(f1_score(yt, yp, average="macro", zero_division=0), 4),
            "recall_bonafide": round(recall_score(yt, yp, pos_label=0,
                                                  zero_division=0), 4),
            "recall_spoof": round(recall_score(yt, yp, pos_label=1,
                                               zero_division=0), 4),
            "eer": round(eer, 4),
            "limiar": round(limiar, 6),
            "escala_score": ESCALA[chave],
        })

    res = pd.DataFrame(linhas)
    # Média simples entre codecs: os grupos são balanceados por construção.
    agg = res.groupby("banda")[["f1_macro", "recall_bonafide", "eer"]].mean().round(4)

    dir_met = RAIZ / "results" / "metricas"
    dir_met.mkdir(parents=True, exist_ok=True)
    res.to_csv(dir_met / f"diagnostico_por_codec_{nome}.csv", index=False)
    print(res.to_string(index=False))
    print("\n--- média por banda ---")
    print(agg.to_string())

    _plotar(carregado, res, limiar)

    estreita, larga = agg.loc["estreita"], agg.loc["larga"]
    delta_f1 = float(larga["f1_macro"] - estreita["f1_macro"])
    delta_eer = float(larga["eer"] - estreita["eer"])
    # Hipótese sustentada = banda larga melhor nas DUAS métricas: f1 maior E EER
    # menor. Exigir as duas evita ler ruído de uma só métrica como confirmação.
    sustenta = bool(delta_f1 > 0 and delta_eer < 0)

    return {
        "modelo": nome,
        "rotulo": carregado["rotulo"],
        "limiar": limiar,
        "regra": REGRA_DECISAO,
        "criterio_limiar": carregado["criterio_limiar"],
        "origem_limiar": carregado["origem_limiar"],
        "escala_score": ESCALA[chave],
        "n_validacao": int(len(validacao)),
        "por_banda": {
            "estreita": {k: float(v) for k, v in estreita.items()},
            "larga": {k: float(v) for k, v in larga.items()},
        },
        "delta_larga_menos_estreita": {
            "f1_macro": round(delta_f1, 4),
            "recall_bonafide": round(float(larga["recall_bonafide"]
                                           - estreita["recall_bonafide"]), 4),
            "eer": round(delta_eer, 4),
        },
        "hipotese_banda_alta_sustentada": sustenta,
        "eer_por_codec": dict(zip(res["codec"], res["eer"].astype(float))),
        "codec_melhor_eer": res.loc[res["eer"].idxmin(), "codec"],
        "codec_pior_eer": res.loc[res["eer"].idxmax(), "codec"],
        "amplitude_eer": round(float(res["eer"].max() - res["eer"].min()), 4),
        "csv": f"results/metricas/diagnostico_por_codec_{nome}.csv",
        "figura": f"results/figuras/diagnostico_por_codec_{nome}.png",
    }


def _plotar(carregado: dict, res: pd.DataFrame, limiar: float) -> None:
    nome = carregado["nome_arquivo"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True)
    cores = ["#c44e52" if BANDA[c] == "estreita" else "#4c72b0"
             for c in res["codec"]]
    for ax, met, titulo in zip(
        axes,
        ["f1_macro", "recall_bonafide", "eer"],
        ["f1_macro", "recall bonafide", "EER (menor = melhor)"],
    ):
        ax.bar(res["codec"], res[met], color=cores)
        ax.set_title(titulo)
        ax.set_xlabel("codec")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"{carregado['rotulo']} por codec (limiar {limiar:.4f}) — "
                 "vermelho: banda estreita | azul: banda larga")
    fig.tight_layout()
    dir_fig = RAIZ / "results" / "figuras"
    dir_fig.mkdir(parents=True, exist_ok=True)
    fig.savefig(dir_fig / f"diagnostico_por_codec_{nome}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    cfg = carregar_config(RAIZ)
    validacao, cols = preparar_validacao()
    print(f"{len(cols)} features | validação {validacao.shape} "
          f"| modelos: {sorted(MODELOS_PRINCIPAIS)}")

    # Os TRÊS modelos do braço principal. A CNN entrou em MODELOS_PRINCIPAIS
    # no B4.7, e este script não mudou de natureza por isso: continua CARREGANDO
    # o artefato e lendo o limiar do JSON que o acompanha — nada é re-treinado, e
    # o limiar da CNN veio da validação como o dos outros dois.
    resumos = {c: diagnosticar(c, validacao, cols) for c in ("rf", "svm", "cnn")}

    # ---- Leitura crítica ----------------------------------------------------
    print("\n" + "=" * 74)
    print("LEITURA CRÍTICA — banda larga vence banda estreita?")
    print("=" * 74)
    for r in resumos.values():
        e, l = r["por_banda"]["estreita"], r["por_banda"]["larga"]
        d = r["delta_larga_menos_estreita"]
        print(f"\n{r['rotulo']} (limiar {r['limiar']:.4f}):")
        print(f"  banda estreita (alaw, ulaw, gsm, pstn): "
              f"f1_macro {e['f1_macro']:.4f} | recall bonafide "
              f"{e['recall_bonafide']:.4f} | EER {e['eer']:.4f}")
        print(f"  banda larga    (g722, opus, none)     : "
              f"f1_macro {l['f1_macro']:.4f} | recall bonafide "
              f"{l['recall_bonafide']:.4f} | EER {l['eer']:.4f}")
        print(f"  delta (larga - estreita): f1_macro {d['f1_macro']:+.4f} | "
              f"recall bonafide {d['recall_bonafide']:+.4f} | "
              f"EER {d['eer']:+.4f}")
        print(f"  hipótese da banda alta sustentada? "
              f"{'SIM' if r['hipotese_banda_alta_sustentada'] else 'NÃO'}")

    todos = all(r["hipotese_banda_alta_sustentada"] for r in resumos.values())
    algum = any(r["hipotese_banda_alta_sustentada"] for r in resumos.values())
    if todos:
        veredito = (
            "HIPÓTESE SUSTENTADA NOS TRÊS MODELOS DO BRAÇO PRINCIPAL, com o limiar do "
            "protocolo e não em 0,50: o desempenho é sistematicamente melhor nos "
            "codecs de banda larga (f1_macro maior E EER menor). Como o EER é "
            "independente de limiar, o contraste NÃO era artefato do limiar 0,50 "
            "da versão baseline. A evidência sustenta que os artefatos "
            "discriminativos vivem acima de 4 kHz, faixa que os codecs telefônicos "
            "removem — e REJEITA um fmax=4000 global, que destruiria os 43% de "
            "áudio em banda larga (decisão ratificada pelo orientador; ver o "
            "bloco `features` do config.yaml). QUE A CNN ACOMPANHE RF E SVM "
            "REFORÇA a leitura em vez de repeti-la: a CNN não usa ZCR nem "
            "centróide — ela vê o espectrograma inteiro até 8 kHz. O contraste "
            "aparecer nos dois ramos aponta para uma propriedade DO SINAL "
            "(a banda que o codec remove), e não para uma peculiaridade "
            "daquelas duas features.")
    elif algum:
        veredito = (
            "HIPÓTESE SUSTENTADA EM APENAS PARTE DOS MODELOS. O contraste por "
            "banda não é estável entre os três sob o limiar do protocolo — "
            "logo ele não pode ser apresentado como propriedade DO SINAL nem "
            "DAS FEATURES sem ressalva. Reportar por modelo, e não como "
            "conclusão geral.")
    else:
        veredito = (
            "HIPÓTESE NÃO SUSTENTADA em nenhum dos três modelos do braço "
            "principal. O "
            "contraste por banda observado no baseline de 13/08/2026 era, "
            "portanto, artefato do limiar 0,50 (que satura o modelo em 'spoof' "
            "e faz o recall_bonafide colapsar de modo desigual entre codecs), e "
            "não uma propriedade das features. Isto TEM de ser dito no texto: a "
            "justificativa de rejeitar fmax=4000 continua válida por outros "
            "motivos (perder 43% do áudio em banda larga), mas não pode mais se "
            "apoiar neste diagnóstico.")
    print(f"\n{veredito}\n")

    resumo = {
        "analise": "diagnostico_por_codec",
        "data": date.today().isoformat(),
        "conjunto": "validacao",
        "teste_lacrado": True,
        "regra_decisao": REGRA_DECISAO,
        "hipotese": ("artefatos de síntese vivem acima de 4 kHz; codecs de banda "
                     "estreita (alaw, ulaw, gsm, pstn) os destroem, logo o "
                     "desempenho deve ser melhor em banda larga (g722, opus, none)"),
        "observacao_metodo": (
            "nenhum modelo é treinado aqui: os três são carregados de models/ "
            "(rf/svm em .joblib, cnn em .pt) e decidem com o limiar selecionado "
            "na validação. A "
            "versão de 13/08/2026 re-treinava um RF baseline e decidia em 0,50 — "
            "ver *_baseline_2026-08-13.csv"),
        "leitura_critica": veredito,
        "por_modelo": resumos,
        "referencia_baseline": {
            "csv": "results/metricas/diagnostico_por_codec_baseline_2026-08-13.csv",
            "nota": ("RF baseline, treino completo, limiar 0,50, features "
                     "PRÉ-mascaramento — mantido como referência 'antes'"),
        },
        "hashes_md5": hashes_congelados(
            RAIZ, cfg["experimento"]["caminho_subamostra"]),
        "ambiente": {
            "python": platform.python_version(),
            "sistema": f"{platform.system()} {platform.release()}",
        },
    }
    caminho = RAIZ / "results" / "metricas" / "diagnostico_por_codec_resumo.json"
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(resumo, f, indent=2, ensure_ascii=False,
                  default=json_seguro)
    print(f"Resumo salvo em {caminho}")


if __name__ == "__main__":
    main()
