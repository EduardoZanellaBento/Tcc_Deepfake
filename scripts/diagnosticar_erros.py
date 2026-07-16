"""
Diagnóstico das falhas da extração de features
==============================================

Responde a três perguntas, nesta ordem:
  1) QUAL foi o erro? (tipo de exceção — arquivo ausente? decodificação? outro?)
  2) Os arquivos que falharam EXISTEM no disco?
  3) A falha é ENVIESADA por classe? (bonafide vs spoof)  <- a pergunta que importa

A (3) é a decisiva: se as falhas atingirem desproporcionalmente uma classe, o
conjunto sobrevivente deixa de ser o ASVspoof 2021 LA e vira um subconjunto
enviesado — o que quebra a comparabilidade com a literatura.

Uso, a partir da RAIZ do projeto:
    python scripts/diagnosticar_erros.py
"""

from pathlib import Path
from collections import Counter

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
ERROS = RAIZ / "data" / "features" / "erros.csv"
FEATURES = RAIZ / "data" / "features" / "features.csv"
LABELS = RAIZ / "data" / "processed" / "labels.csv"


def cabecalho(txt):
    print("\n" + "=" * 70)
    print(txt)
    print("=" * 70)


# ---------------------------------------------------------------------------
cabecalho("0) INVENTÁRIO")

labels = pd.read_csv(LABELS)
print(f"labels.csv         : {len(labels):>7} provas listadas no trial_metadata")

if FEATURES.exists():
    feats = pd.read_csv(FEATURES)
    print(f"features.csv       : {len(feats):>7} extraídos com sucesso")
else:
    feats = None
    print("features.csv       : NÃO EXISTE")

if not ERROS.exists():
    print("\nerros.csv não existe — nada a diagnosticar.")
    raise SystemExit(0)

erros = pd.read_csv(ERROS)
print(f"erros.csv          : {len(erros):>7} falhas")
print(f"soma               : {len(erros) + (len(feats) if feats is not None else 0):>7} "
      f"(deve bater com labels.csv)")


# ---------------------------------------------------------------------------
cabecalho("1) QUAL FOI O ERRO?")

# A coluna 'erro' guarda o repr() da exceção. O tipo é o que vem antes do '('.
tipos = erros["erro"].astype(str).str.split("(").str[0]
for tipo, n in Counter(tipos).most_common(10):
    pct = 100 * n / len(erros)
    print(f"  {n:>7} ({pct:5.1f}%)  {tipo}")

print("\nExemplos de mensagem completa (3 primeiras):")
for msg in erros["erro"].astype(str).head(3):
    print(f"  - {msg[:160]}")


# ---------------------------------------------------------------------------
cabecalho("2) OS ARQUIVOS QUE FALHARAM EXISTEM NO DISCO?")

# Recupera o caminho a partir do labels.csv (o erros.csv só guarda 'arquivo')
mapa = labels.set_index("arquivo")["caminho"].to_dict()
amostra = erros["arquivo"].head(200)

existem = sum(1 for a in amostra if a in mapa and Path(mapa[a]).exists())
print(f"  Amostra de {len(amostra)} arquivos que falharam:")
print(f"    existem no disco  : {existem}")
print(f"    NÃO existem       : {len(amostra) - existem}")

if existem == 0:
    print("\n  >>> DIAGNÓSTICO: os arquivos NÃO ESTÃO NO DISCO.")
    print("      Causa provável: o LA eval do ASVspoof 2021 é distribuído em")
    print("      VÁRIAS PARTES (part00, part01, ...). Faltou baixar/extrair alguma.")
elif existem == len(amostra):
    print("\n  >>> Os arquivos EXISTEM. A falha é de LEITURA/DECODIFICAÇÃO,")
    print("      não de ausência. Ver o tipo de exceção na seção 1.")


# ---------------------------------------------------------------------------
cabecalho("3) A FALHA É ENVIESADA POR CLASSE?  (A PERGUNTA DECISIVA)")

erros_lbl = erros.merge(labels[["arquivo", "label"]], on="arquivo", how="left")

print("Distribuição ORIGINAL (labels.csv):")
orig = labels["label"].value_counts()
for k, v in orig.items():
    print(f"  {k:10s}: {v:>7} ({100*v/len(labels):5.2f}%)")

print("\nDistribuição das FALHAS (erros.csv):")
fal = erros_lbl["label"].value_counts()
for k, v in fal.items():
    print(f"  {k:10s}: {v:>7} ({100*v/len(erros_lbl):5.2f}%)")

if feats is not None:
    print("\nDistribuição SOBREVIVENTE (features.csv) — é COM ISTO que você treinaria:")
    sob = feats["label"].value_counts()
    for k, v in sob.items():
        print(f"  {k:10s}: {v:>7} ({100*v/len(feats):5.2f}%)")

    if "bonafide" in sob and "spoof" in sob:
        razao_orig = orig.get("spoof", 0) / max(orig.get("bonafide", 1), 1)
        razao_sob = sob["spoof"] / max(sob["bonafide"], 1)
        print(f"\n  razão spoof:bonafide ORIGINAL    = {razao_orig:.2f} : 1")
        print(f"  razão spoof:bonafide SOBREVIVENTE = {razao_sob:.2f} : 1")

        desvio = abs(razao_sob - razao_orig) / razao_orig * 100
        if desvio < 5:
            print(f"\n  >>> As falhas são ~UNIFORMES entre classes (desvio {desvio:.1f}%).")
            print("      O desbalanceamento foi preservado. Menos grave — mas ainda")
            print("      é preciso explicar por que 44% do dataset sumiu.")
        else:
            print(f"\n  >>> ALERTA: a proporção MUDOU {desvio:.1f}%. As falhas são")
            print("      ENVIESADAS por classe. Treinar assim compromete a validade")
            print("      do experimento e a comparabilidade com a literatura.")

# ---------------------------------------------------------------------------
cabecalho("4) VERIFICAÇÃO DO VAD (nos que deram certo)")

if feats is not None and "prop_fala" in feats.columns:
    p = feats["prop_fala"]
    print(p.describe().to_string())
    zerados = (p == 0).sum()
    print(f"\n  áudios com prop_fala == 0 (VAD zerou tudo): {zerados} "
          f"({100*zerados/len(feats):.2f}%)")
    if p.mean() < 0.3:
        print("  >>> VAD parece AGRESSIVO demais. Considerar agressividade=1 ou 0.")
    else:
        print("  >>> VAD com comportamento plausível.")