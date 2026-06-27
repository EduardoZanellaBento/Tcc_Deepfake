"""
Leitura dos labels e metadados do ASVspoof 2021 LA

Filosofia deste módulo: NÃO confiar cegamente na posição das colunas. O código
inspeciona o arquivo, detecta sozinho onde está o rótulo (bonafide/spoof) e o ID
do áudio, e monta uma tabela limpa. Se o formato do arquivo variar um pouco, o
código continua funcionando.

Tabela de saída:
    arquivo | caminho | label | classe_binaria
    onde classe_binaria: 0 = bonafide (real), 1 = spoof (sintético)
"""

import re
from pathlib import Path
import pandas as pd

# Anchors que usamos para detectar as colunas independentemente da posição:
RE_FILE_ID = re.compile(r"^LA_[A-Z]_\d+$")   # ex.: LA_E_9332881
LABELS_VALIDOS = {"bonafide", "spoof"}


def inspecionar_arquivo(caminho_chaves: str, n_linhas: int = 5) -> None:
    """Imprime as primeiras linhas cruas e o nº de colunas. SEMPRE rode isto
    primeiro, antes de parsear qualquer dataset novo."""
    caminho = Path(caminho_chaves)
    print(f"Arquivo: {caminho}")
    with open(caminho, "r", encoding="utf-8") as f:
        for i, linha in enumerate(f):
            if i >= n_linhas:
                break
            tokens = linha.split()
            print(f"  linha {i}: {len(tokens)} colunas -> {tokens}")


def _detectar_indices(caminho_chaves: Path, max_linhas: int = 200) -> dict:
    """Varre as primeiras linhas para descobrir qual coluna é o ID do áudio,
    qual é o rótulo e qual é o ataque. Varre várias linhas (não só a primeira)
    porque o ataque só aparece em linhas 'spoof' — em 'bonafide' ele é '-'."""
    idx = {"id": None, "label": None, "ataque": None}
    with open(caminho_chaves, "r", encoding="utf-8") as f:
        for n, linha in enumerate(f):
            if n >= max_linhas or all(v is not None for v in idx.values()):
                break
            for j, tok in enumerate(linha.split()):
                if idx["id"] is None and RE_FILE_ID.match(tok):
                    idx["id"] = j
                elif idx["label"] is None and tok.lower() in LABELS_VALIDOS:
                    idx["label"] = j
                elif idx["ataque"] is None and re.match(r"^A\d{2}$", tok):
                    idx["ataque"] = j
    return idx


def carregar_labels(caminho_chaves: str, caminho_audios: str) -> pd.DataFrame:
    """Lê o trial_metadata.txt e devolve um DataFrame com:
    [arquivo, caminho, ataque, label, classe_binaria].
    """
    caminho_chaves = Path(caminho_chaves)
    pasta_audios = Path(caminho_audios)

    if not caminho_chaves.exists():
        raise FileNotFoundError(
            f"Não achei o arquivo de chaves em {caminho_chaves}. "
            "Você já baixou e descompactou as keys do ASVspoof 2021 LA?"
        )

    # Descobre os índices das colunas varrendo as primeiras linhas.
    idx = _detectar_indices(caminho_chaves)
    if idx["id"] is None or idx["label"] is None:
        raise ValueError(
            "Não consegui detectar as colunas de ID e/ou label automaticamente. "
            "Rode inspecionar_arquivo() e me mostre as primeiras linhas."
        )

    linhas = []
    with open(caminho_chaves, "r", encoding="utf-8") as f:
        for linha in f:
            t = linha.split()
            if not t:
                continue
            file_id = t[idx["id"]]
            label = t[idx["label"]].lower()
            ataque = t[idx["ataque"]] if idx["ataque"] is not None else "-"
            linhas.append({
                "arquivo": file_id,
                "caminho": str(pasta_audios / f"{file_id}.flac"),
                "ataque": ataque,
                "label": label,
                "classe_binaria": 0 if label == "bonafide" else 1,
            })

    df = pd.DataFrame(linhas)
    return df


def resumo_distribuicao(df: pd.DataFrame) -> None:
    """Mostra o desbalanceamento real/sintético — base da justificativa para
    usar class_weights (pergunta certeira de banca)."""
    cont = df["label"].value_counts()
    total = len(df)
    print(f"\nTotal de áudios: {total}")
    for label, n in cont.items():
        print(f"  {label:<9}: {n:>7}  ({100*n/total:.1f}%)")
    if {"bonafide", "spoof"}.issubset(cont.index):
        razao = cont["spoof"] / cont["bonafide"]
        print(f"  Razão spoof:bonafide = {razao:.1f} : 1")


if __name__ == "__main__":
    print("Módulo de carregamento de labels — ASVspoof 2021 LA.")
