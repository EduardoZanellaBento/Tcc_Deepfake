"""
Re-renderiza as matrizes de confusão JÁ PUBLICADAS, sem retreinar nada.
=======================================================================

POR QUE ISTO EXISTE:
    `plotar_matriz_confusao` (src/models/avaliacao.py) é compartilhada por RF,
    SVM e CNN — de propósito: num trabalho cuja pergunta é COMPARAR modelos, a
    figura da comparação precisa ser a mesma régua, como `avaliar` é a mesma
    régua de métrica. A consequência é que um ajuste nessa função deixa TODAS as
    figuras já gravadas desatualizadas ao mesmo tempo.

    A alternativa seria reexecutar os pipelines que produziram cada figura. Isso
    custaria horas de treino e — pior — reescreveria JSONs congelados só para
    mexer num detalhe de desenho. Este script separa as duas coisas: a matriz de
    confusão já está gravada dentro de cada JSON de métricas, então redesenhar é
    ler o array e chamar a função. Nenhum modelo é carregado, nenhum áudio é
    lido, nenhum JSON é escrito.

O QUE ELE NÃO FAZ, E POR QUÊ:
    - não recalcula a matriz: o array vem do JSON, então a figura nova mostra
      EXATAMENTE os mesmos números da antiga. Se recalculasse, o script viraria
      um segundo caminho de avaliação — e dois caminhos divergem;
    - não toca no conjunto lacrado: só lê JSONs de validação, listados um a um
      abaixo. Nada aqui varre diretório em busca de arquivos.

A DUPLICAÇÃO DOS TÍTULOS É DELIBERADA, E VIGIADA:
    o título de cada figura é construído no módulo que treina o modelo, dentro
    da mesma f-string que grava o arquivo. Repeti-los aqui cria uma segunda
    fonte da verdade, que pode divergir. A defesa é a coluna `origem` da tabela:
    ela nomeia o arquivo e a f-string de onde o título foi copiado. Ao mudar um
    título no módulo de origem, mude aqui também — ou a figura volta ao texto
    com um rótulo que o pipeline não produz mais.

USO:
    python -m scripts.replotar_matrizes_confusao
    python -m scripts.replotar_matrizes_confusao --conferir   # só verifica

`--conferir` não escreve nada: lista o que seria regerado e sai com código 1 se
algum JSON esperado não existir. Serve para rodar antes de um commit de figuras.
"""

import argparse
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.models.avaliacao import plotar_matriz_confusao  # noqa: E402


# (nome do artefato, molde do título, origem do molde)
#
# O molde recebe o dicionário do JSON. `limiar` é formatado com :.2f — a MESMA
# precisão dos módulos de origem, que é o que mantém o rótulo idêntico ao que o
# pipeline grava. Os dois baselines de RF do eval não têm limiar selecionado
# (são anteriores ao protocolo do Bloco 3) e por isso o título deles não o cita.
FIGURAS = [
    (
        "rf_baseline_eval_principal",
        lambda d: f"Random Forest baseline (eval, braço {d['braco']}) — validação",
        "src/models/treinar_rf.py",
    ),
    (
        "rf_baseline_eval_referencia",
        lambda d: f"Random Forest baseline (eval, braço {d['braco']}) — validação",
        "src/models/treinar_rf.py",
    ),
    (
        "rf_tuned_principal",
        lambda d: f"RF ajustado (braço {d['braco']}) — validação, limiar {d['limiar']:.2f}",
        "src/models/ajustar_rf.py",
    ),
    (
        "rf_tuned_referencia",
        lambda d: f"RF ajustado (braço {d['braco']}) — validação, limiar {d['limiar']:.2f}",
        "src/models/ajustar_rf.py",
    ),
    (
        "svm_tuned_principal",
        lambda d: ("SVM RBF ajustado (braço principal) — validação, "
                   f"limiar {d['limiar']:.2f}"),
        "src/models/treinar_svm.py",
    ),
    (
        "cnn_baseline",
        lambda d: ("CNN baseline (B4.4) — early stopping interno (3k), "
                   f"limiar {d['limiar']:.2f}"),
        "src/models/treinar_cnn.py",
    ),
    (
        "cnn_final_principal",
        lambda d: ("CNN final (braço principal) — validação, "
                   f"limiar {d['limiar']:.2f}"),
        "src/models/validar_cnn.py",
    ),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--conferir", action="store_true",
                    help="não escreve figura nenhuma; só confere e lista")
    args = ap.parse_args()

    dir_met = RAIZ / "results" / "metricas"
    dir_fig = RAIZ / "results" / "figuras"
    faltando = []

    for nome, molde, origem in FIGURAS:
        caminho_json = dir_met / f"{nome}.json"
        if not caminho_json.exists():
            faltando.append(str(caminho_json.relative_to(RAIZ)))
            continue

        with open(caminho_json, encoding="utf-8") as f:
            d = json.load(f)
        cm = d.get("matriz_confusao")
        if cm is None:
            faltando.append(f"{caminho_json.relative_to(RAIZ)} (sem matriz_confusao)")
            continue

        titulo = molde(d)
        if args.conferir:
            print(f"[ok] {nome}: «{titulo}»  (título de {origem})")
            continue
        plotar_matriz_confusao(cm, dir_fig / f"matriz_confusao_{nome}.png", titulo)

    if faltando:
        print("\nARTEFATOS ESPERADOS E NÃO ENCONTRADOS:", file=sys.stderr)
        for item in faltando:
            print(f"  - {item}", file=sys.stderr)
        return 1

    if not args.conferir:
        print(f"\n{len(FIGURAS)} matrizes regeradas a partir dos JSONs de métricas. "
              "Nenhum modelo foi carregado e nenhum JSON foi reescrito.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
