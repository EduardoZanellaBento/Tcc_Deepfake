"""
Leitura dos labels e metadados do ASVspoof 2021 LA

Filosofia deste módulo: NÃO confiar cegamente na posição das colunas. O código
inspeciona o arquivo, detecta sozinho onde está o rótulo (bonafide/spoof) e o ID
do áudio, e monta uma tabela limpa. Se o formato do arquivo variar um pouco, o
código continua funcionando.

Tabela de saída:
    arquivo | caminho | codec | transmissao | ataque | label | trim | fase | classe_binaria
    onde classe_binaria: 0 = bonafide (real), 1 = spoof (sintético)

Sobre as 8 colunas do trial_metadata.txt (ASVspoof 2021 LA):
    locutor  arquivo  codec  transmissao  ataque  label  trim  fase
    - codec: alaw/ulaw/gsm/pstn (banda estreita, teto ~4 kHz) e g722/opus/none
      (banda larga, preserva >4 kHz). Fator perfeitamente balanceado (25.938 cada).
    - trim: notrim (eval+progress) ou only_speech (hidden, silêncio pré-cortado).
    - fase: eval (148.176) / progress (16.464) / hidden (16.926). O conjunto
      oficialmente pontuado do ASVspoof 2021 LA é fase=='eval'.
    Esses metadados alimentam os diagnósticos por codec/fase/trim (scripts/).
"""

import re
from pathlib import Path
import pandas as pd

# Anchors que usamos para detectar as colunas independentemente da posição:
RE_FILE_ID = re.compile(r"^LA_[A-Z]_\d+$")   # ex.: LA_E_9332881
LABELS_VALIDOS = {"bonafide", "spoof"}

# Ordem das 8 colunas do trial_metadata.txt do ASVspoof 2021 LA:
COLUNAS_META = ["locutor", "arquivo", "codec", "transmissao",
                "ataque", "label", "trim", "fase"]


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
    [arquivo, caminho, codec, transmissao, ataque, label, trim, fase,
    classe_binaria].

    Parseia as 8 colunas do arquivo (ver COLUNAS_META) em vez de só ID/label/
    ataque: codec, transmissao, trim e fase são necessários para os diagnósticos
    por codec e por fase (scripts/diagnostico_por_codec.py e
    scripts/diagnostico_composicao_dataset.py). A filosofia de "não confiar
    cegamente na posição" é mantida via validação pós-leitura: o ID precisa
    casar com RE_FILE_ID e o label precisa estar em LABELS_VALIDOS — se o
    formato do arquivo mudar, o erro é explícito, não silencioso.
    """
    caminho_chaves = Path(caminho_chaves)
    pasta_audios = Path(caminho_audios)

    if not caminho_chaves.exists():
        raise FileNotFoundError(
            f"Não achei o arquivo de chaves em {caminho_chaves}. "
            "Você já baixou e descompactou as keys do ASVspoof 2021 LA?"
        )

    meta = pd.read_csv(caminho_chaves, sep=r"\s+", header=None, names=COLUNAS_META)

    # Validação (substitui a confiança cega na posição das colunas): se o
    # formato do arquivo não for o esperado, paramos aqui com erro claro.
    if not meta["arquivo"].str.match(RE_FILE_ID).all():
        raise ValueError(
            "Coluna 'arquivo' não bate com o padrão LA_X_NNNNNNN. O formato do "
            "trial_metadata.txt mudou? Rode inspecionar_arquivo() e confira."
        )
    labels_encontrados = set(meta["label"].str.lower().unique())
    if not labels_encontrados.issubset(LABELS_VALIDOS):
        raise ValueError(
            f"Labels inesperados no arquivo: {labels_encontrados - LABELS_VALIDOS}. "
            "Rode inspecionar_arquivo() e confira as colunas."
        )

    df = pd.DataFrame({
        "arquivo": meta["arquivo"],
        "caminho": [str(pasta_audios / f"{a}.flac") for a in meta["arquivo"]],
        "codec": meta["codec"],
        "transmissao": meta["transmissao"],
        "ataque": meta["ataque"],
        "label": meta["label"].str.lower(),
        "trim": meta["trim"],
        "fase": meta["fase"],
    })
    df["classe_binaria"] = (df["label"] != "bonafide").astype(int)
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
    # Regenera data/processed/labels.csv a partir do trial_metadata.txt.
    # Operação barata (lê só TEXTO, nenhum áudio) e idempotente.
    # Rode a partir da raiz:  python -m src.data.carregar_dados
    import yaml

    RAIZ = Path(__file__).resolve().parents[2]
    cfg = yaml.safe_load(open(RAIZ / "config" / "config.yaml", encoding="utf-8"))

    print("Módulo de carregamento de labels — ASVspoof 2021 LA.")
    df = carregar_labels(
        RAIZ / cfg["dataset"]["caminho_chaves"],
        RAIZ / cfg["dataset"]["caminho_audios"],
    )
    resumo_distribuicao(df)

    saida = RAIZ / "data" / "processed" / "labels.csv"
    saida.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(saida, index=False)
    print(f"\nlabels.csv regenerado em {saida} com colunas: {list(df.columns)}")
