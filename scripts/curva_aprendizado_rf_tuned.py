"""
Curva de aprendizado do RF AJUSTADO sobre as features CONGELADAS (Bloco 3)
==========================================================================

POR QUE ESTA CURVA EXISTE, SE JÁ HAVIA UMA:
    `curva_aprendizado_rf_eval.{csv,json,png}` foi medida ANTES do lote único,
    sobre as features pré-mascaramento e com o RF *baseline* (100 árvores,
    max_depth=None, decisão por argmax em 0,50). Ela sustenta dois argumentos
    que o trabalho usa até hoje:
      (1) o RF NÃO satura antes do treino completo -> justifica o braço de
          referência (quantificar o custo da subamostra de 30k);
      (2) "o RF ainda está com fome de dados no ponto em que o SVM já extraiu o
          que precisava" (NOTA_RF_VS_SVM.md, seção 2).
    Ambos passaram a apoiar-se em número velho. Esta versão refaz a curva com o
    artefato e o protocolo vigentes; a curva antiga fica PRESERVADA como
    referência "antes" (nomes de arquivo diferentes, nada é sobrescrito).

O QUE MUDA EM RELAÇÃO À CURVA ANTIGA:
    - features CONGELADAS (features.csv MD5 51b2f439…), não as pré-mascaramento;
    - hiperparâmetros AJUSTADOS, lidos de rf_random_search.json (fonte única) —
      não os do baseline;
    - decisão pelo protocolo do Bloco 3: limiar SELECIONADO NA VALIDAÇÃO em cada
      ponto (`selecionar_limiar`, regra `score >= limiar`), não `argmax` em 0,50.
      O limiar escolhido em cada ponto entra no CSV — ele próprio é um resultado:
      mostra se o ponto de corte migra com o tamanho do treino.
    - EER continua sendo reportado e é a métrica que responde à pergunta da
      saturação sem depender de limiar nenhum.

CHECAGEM EMBUTIDA (a que dá confiança no resto da curva):
    O último ponto (treino inteiro, sem subamostrar) é, por construção, o BRAÇO
    DE REFERÊNCIA do Bloco 3: mesma config, mesma semente, mesmo treino de
    103.723. Logo ele TEM de reproduzir rf_tuned_referencia.json (f1_macro
    0,7723 / EER 0,1579). O script compara e AVISA se divergir — se o extremo da
    curva não bate com um artefato já publicado, os pontos intermediários também
    não merecem crédito.

CUIDADO DE LEITURA — o ponto de 30.000 NÃO é a subamostra_30k:
    Os pontos desta curva são estratificados por classe_binaria × codec; a
    `subamostra_30k.csv` do braço principal é estratificada por
    classe_binaria × codec × ataque (98 estratos) e é um ARQUIVO FIXO. O ponto de
    30k serve para ler a curva na altura certa, não para substituir a subamostra.
    A comparação legítima "subamostra × treino completo" continua sendo
    rf_tuned_principal.json × rf_tuned_referencia.json.

CUSTO: 7 pontos × (treino + seleção de limiar sobre ~22k candidatos). A seleção
    é O(n_candidatos × n_validacao) e leva ~40-60 s por ponto; conte alguns
    minutos no total. Nada de re-extração, nada de conjunto de teste.

SAÍDAS (arquivos NOVOS — as curvas antigas não são tocadas):
    results/metricas/curva_aprendizado_rf_tuned_eval.csv
    results/metricas/curva_aprendizado_rf_tuned_eval.json
    results/figuras/curva_aprendizado_rf_tuned_eval.png

Rode a partir da raiz:  python -m scripts.curva_aprendizado_rf_tuned
"""

import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.utils.config import carregar_config
from src.utils.seeds import fixar_seeds
from src.data.split import carregar_dados_split, colunas_features
from src.models.avaliacao import calcular_eer, predizer_rf, selecionar_limiar

RAIZ = Path(__file__).resolve().parents[1]
NOME = "curva_aprendizado_rf_tuned_eval"

# 30.000 entra para permitir ler a curva na altura do braço principal — com a
# ressalva da docstring: NÃO é a subamostra_30k. None = treino inteiro.
TAMANHOS = [5000, 10000, 20000, 30000, 40000, 80000, None]
MIN_ESTRATO = 10          # estratos menores que isto são colapsados em 'outros'
TOL_SATURACAO = 0.005     # satura no 1º tamanho a menos de 0,005 do f1 máximo


def subamostrar(treino: pd.DataFrame, n: int, semente: int) -> pd.DataFrame:
    """Subamostra estratificada por classe_binaria × codec do split de treino."""
    estrato = treino["classe_binaria"].astype(str) + "_" + treino["codec"]
    contagem = estrato.value_counts()
    pequenos = contagem[contagem < MIN_ESTRATO].index
    if len(pequenos):
        estrato = estrato.where(~estrato.isin(pequenos), "outros")
    sub, _ = train_test_split(
        treino, train_size=n, stratify=estrato,
        random_state=semente, shuffle=True,
    )
    return sub


def main() -> None:
    cfg = carregar_config(RAIZ)
    semente = fixar_seeds(cfg["semente"])
    dir_met = RAIZ / "results" / "metricas"

    # ---- Fonte única dos hiperparâmetros: a busca do Bloco 3 ----------------
    with open(dir_met / "rf_random_search.json", encoding="utf-8") as f:
        params = json.load(f)["melhor"]["params"]
    with open(dir_met / "rf_tuned_referencia.json", encoding="utf-8") as f:
        ref_full = json.load(f)

    # ---- Guarda: só vale sobre os artefatos congelados ----------------------
    h_feat = hashlib.md5(
        (RAIZ / "data" / "features" / "features.csv").read_bytes()).hexdigest()
    h_split = hashlib.md5(
        (RAIZ / "data" / "processed" / "split.csv").read_bytes()).hexdigest()
    if (ref_full["hash_md5_features_csv"] != h_feat
            or ref_full["hash_md5_split_csv"] != h_split):
        raise RuntimeError(
            "features.csv/split.csv em disco divergem dos hashes de "
            "rf_tuned_referencia.json — a curva não seria comparável aos "
            "resultados publicados do Bloco 3.")

    df = carregar_dados_split(RAIZ)
    cols = colunas_features(df)
    labels = pd.read_csv(RAIZ / "data" / "processed" / "labels.csv",
                         usecols=["arquivo", "codec"])
    n_antes = len(df)
    df = df.merge(labels, on="arquivo", how="inner")
    assert len(df) == n_antes, "merge com labels perdeu linhas — investigar"

    treino = df[df["conjunto"] == "treino"]
    validacao = df[df["conjunto"] == "validacao"]
    X_va, y_va = validacao[cols].values, validacao["classe_binaria"].values
    print(f"{len(cols)} features | treino {len(treino)} | "
          f"validação {len(validacao)}")
    print(f"config ajustada (rf_random_search.json): {params}\n")

    linhas = []
    for n in TAMANHOS:
        sub = treino if n is None else subamostrar(treino, n, semente)
        X_tr, y_tr = sub[cols].values, sub["classe_binaria"].values

        modelo = RandomForestClassifier(**params, random_state=semente,
                                        n_jobs=-1)
        t0 = time.perf_counter()
        modelo.fit(X_tr, y_tr)
        t_treino = time.perf_counter() - t0

        scores = predizer_rf(modelo, X_va)   # n_jobs=1: reprodutível bit a bit
        # Protocolo do Bloco 3: limiar selecionado NA VALIDAÇÃO, regra `>=`.
        sel = selecionar_limiar(y_va, scores, criterio="f1_macro",
                                conjunto="validacao")
        eer, _ = calcular_eer(y_va, scores)

        linhas.append({
            "n_treino": len(sub),
            "limiar": round(float(sel["limiar"]), 4),
            "f1_macro": round(float(sel["f1_macro"]), 4),
            "eer": round(float(eer), 4),
            "tempo_treino_s": round(t_treino, 2),
        })
        print(f"n={len(sub):>6}: f1_macro={sel['f1_macro']:.4f}  EER={eer:.4f}  "
              f"limiar={sel['limiar']:.4f}  treino={t_treino:.1f}s")

    res = pd.DataFrame(linhas)
    dir_met.mkdir(parents=True, exist_ok=True)
    res.to_csv(dir_met / f"{NOME}.csv", index=False)

    # ---- Checagem embutida: o último ponto é o braço de referência ----------
    ultimo = res.iloc[-1]
    bate = (round(float(ultimo["f1_macro"]), 4)
            == round(float(ref_full["f1_macro"]), 4)
            and round(float(ultimo["eer"]), 4)
            == round(float(ref_full["eer"]), 4))
    checagem = {
        "o_que_e": "o último ponto (treino inteiro) é, por construção, o braço "
                   "de referência do Bloco 3 — mesma config, mesma semente, "
                   "mesmo treino",
        "curva_f1_macro": float(ultimo["f1_macro"]),
        "curva_eer": float(ultimo["eer"]),
        "rf_tuned_referencia_f1_macro": float(ref_full["f1_macro"]),
        "rf_tuned_referencia_eer": float(ref_full["eer"]),
        "reproduz": bool(bate),
    }
    if not bate:
        print("\n*** ATENÇÃO: o último ponto NÃO reproduziu "
              "rf_tuned_referencia.json. Os pontos intermediários não merecem "
              "crédito enquanto isso não for explicado. ***")

    # ---- Ponto de saturação -------------------------------------------------
    # POR QUE MUDOU (R3, revisão de 03/09/2026): a versão anterior fazia
    #     saturado = res[res["f1_macro"] >= f1_max - TOL].iloc[0]
    # e publicava `satura_em_n`. Numa curva MONÓTONA CRESCENTE o máximo é o
    # último ponto, e essa expressão devolve o último ponto sempre que nenhum
    # anterior esteja a menos de TOL dele — isto é, o campo era
    # ESTRUTURALMENTE INCAPAZ de distinguir "saturou no fim" de "NÃO saturou".
    # O JSON e a legenda da figura afirmavam saturação enquanto o próprio dado
    # mostrava o contrário (o salto 80k -> 103.723 ainda rende mais que o dobro
    # da tolerância). Agora a saturação é um booleano de definição explícita: o
    # ganho do ÚLTIMO passo é menor que a tolerância? `satura_em_n` só é
    # emitido quando a resposta é sim.
    f1_max = res["f1_macro"].max()
    ganho_ultimo_passo = float(res["f1_macro"].iloc[-1] - res["f1_macro"].iloc[-2])
    saturou = bool(ganho_ultimo_passo < TOL_SATURACAO)
    # Só faz sentido perguntar ONDE saturou se de fato saturou.
    saturado = (res[res["f1_macro"] >= f1_max - TOL_SATURACAO].iloc[0]
                if saturou else None)

    registro = {
        "modelo": NOME,
        "conjunto": "validacao (o teste continua lacrado)",
        "config": "hiperparâmetros AJUSTADOS (rf_random_search.json), features "
                  "CONGELADAS, limiar selecionado na validação em cada ponto",
        "hiperparametros": {k: (v if v is None or isinstance(v, (int, float,
                                                                 bool))
                                else str(v)) for k, v in params.items()},
        "semente": semente,
        "estratificacao_dos_pontos": "classe_binaria x codec",
        "aviso_ponto_30k": (
            "o ponto de 30.000 NÃO é a subamostra_30k do braço principal "
            "(estratificada por classe x codec x ataque, arquivo fixo); serve "
            "para ler a curva na altura certa. A comparação legítima "
            "subamostra × treino completo continua sendo rf_tuned_principal "
            "× rf_tuned_referencia."),
        "tamanhos": [int(x) for x in res["n_treino"]],
        "resultados": res.to_dict(orient="records"),
        "f1_macro_maximo": float(f1_max),
        "saturou": saturou,
        "maior_n_medido": int(res["n_treino"].iloc[-1]),
        "ganho_ultimo_passo_f1": round(ganho_ultimo_passo, 4),
        "definicao_saturacao": (
            "saturou = (f1_macro do maior n) - (f1_macro do n anterior) < "
            f"{TOL_SATURACAO}. Se saturou for false, a curva AINDA SOBE no maior "
            "treino disponível e o campo satura_em_n NÃO é emitido: nenhum n "
            "medido satisfaz a definição."),
        "tolerancia_saturacao": TOL_SATURACAO,
        "checagem_ultimo_ponto": checagem,
        "curva_anterior_preservada": {
            "arquivos": ["curva_aprendizado_rf_eval.{csv,json,png}",
                         "curva_aprendizado_rf.{csv,json,png}"],
            "o_que_sao": "curvas medidas com o RF baseline; a _eval sobre as "
                         "features PRÉ-mascaramento, a sem sufixo sobre o "
                         "universo histórico de 181.566. Referência 'antes'.",
        },
        "ambiente": {
            "cpu": platform.processor(),
            "maquina": platform.machine(),
            "sistema": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(),
            "versoes": {"numpy": np.__version__, "sklearn": sklearn.__version__,
                        "pandas": pd.__version__},
        },
        "hash_md5_features_csv": h_feat,
        "hash_md5_split_csv": h_split,
    }
    if saturou:
        registro["satura_em_n"] = int(saturado["n_treino"])
    with open(dir_met / f"{NOME}.json", "w", encoding="utf-8") as f:
        json.dump(registro, f, indent=2, ensure_ascii=False)

    # ---- Figura -------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(res["n_treino"], res["f1_macro"], marker="o", color="#4c72b0",
             label="f1_macro (limiar selecionado)")
    if saturou:
        ax1.axvline(saturado["n_treino"], ls="--", color="gray", lw=1,
                    label=f"saturação ≈ {int(saturado['n_treino'])}")
    else:
        # Sem linha de saturação: desenhá-la seria afirmar na figura o que o
        # dado nega. A anotação diz o que de fato se sabe.
        ax1.annotate(
            "não satura no maior n disponível\n"
            f"(último passo: +{ganho_ultimo_passo:.4f} em f1_macro,\n"
            f"tolerância {TOL_SATURACAO})",
            xy=(0.97, 0.05), xycoords="axes fraction", ha="right", va="bottom",
            fontsize=8.5, color="#444444",
            bbox=dict(boxstyle="round,pad=0.4", fc="#f5f5f5", ec="#bbbbbb"))
    ax1.axvline(30000, ls=":", color="#55a868", lw=1.2,
                label="30k (altura do braço principal)")
    ax1.set_xscale("log")
    ax1.set_xlabel("nº de amostras de treino (escala log)")
    ax1.set_ylabel("f1_macro (validação)", color="#4c72b0")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(res["n_treino"], res["eer"], marker="s", color="#c44e52",
             label="EER")
    ax2.set_ylabel("EER (validação)", color="#c44e52")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center right", fontsize=8)
    ax1.set_title("Curva de aprendizado — RF AJUSTADO, features congeladas")
    fig.tight_layout()
    dir_fig = RAIZ / "results" / "figuras"
    dir_fig.mkdir(parents=True, exist_ok=True)
    fig.savefig(dir_fig / f"{NOME}.png", dpi=150)
    plt.close(fig)

    # ---- Conclusão ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("CONCLUSÃO")
    print("=" * 70)
    # As duas ramificações são escritas por extenso e sem hedge: o texto do
    # TC II vai citar uma delas, e ela precisa ser afirmativa sobre o dado.
    n_penultimo = int(res["n_treino"].iloc[-2])
    n_ultimo = int(res["n_treino"].iloc[-1])
    if saturou:
        leitura_saturacao = (
            f"f1_macro máximo = {f1_max:.4f}. A curva SATURA: o último passo "
            f"({n_penultimo} -> {n_ultimo}) rende apenas "
            f"+{ganho_ultimo_passo:.4f} em f1_macro, abaixo da tolerância de "
            f"{TOL_SATURACAO}. Pela definição, a saturação começa em "
            f"n = {int(saturado['n_treino'])} (f1_macro "
            f"{saturado['f1_macro']:.4f}, EER {saturado['eer']:.4f}).\n"
            "CONSEQUÊNCIA: mais dados do mesmo tipo não compram desempenho, e a "
            "afirmação da NOTA_RF_VS_SVM.md de que o RF está 'com fome de "
            "dados' precisa ser REESCRITA — é um resultado novo.")
    else:
        leitura_saturacao = (
            f"f1_macro máximo = {f1_max:.4f}. A curva NÃO SATURA no maior treino "
            f"disponível: o último passo ({n_penultimo} -> {n_ultimo}) ainda "
            f"rende +{ganho_ultimo_passo:.4f} em f1_macro — "
            f"{ganho_ultimo_passo / TOL_SATURACAO:.1f}x a tolerância de "
            f"{TOL_SATURACAO}. Nenhum n medido satisfaz a definição de "
            "saturação, então satura_em_n NÃO é emitido e a figura não traz "
            "linha de saturação.\n"
            "CONSEQUÊNCIA: a subamostra de 30k custa desempenho real ao RF — é "
            "o que justifica o braço de referência e sustenta a afirmação da "
            "NOTA_RF_VS_SVM.md de que o RF ainda está com fome de dados no "
            "ponto em que o SVM já extraiu o que precisava. Quanto faltaria "
            "para o RF alcançar o SVM é quantificado em "
            "scripts/extrapolacao_curva_rf.py.")
    print(f"""
{leitura_saturacao}

Checagem do último ponto contra rf_tuned_referencia.json: \
{'REPRODUZIU' if bate else 'DIVERGIU — investigar antes de citar a curva'}.

CSV : {dir_met / (NOME + '.csv')}
JSON: {dir_met / (NOME + '.json')}
PNG : {dir_fig / (NOME + '.png')}
""")


if __name__ == "__main__":
    main()
