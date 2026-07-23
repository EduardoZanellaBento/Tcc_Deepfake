import pandas as pd
from pathlib import Path

COLUNAS = ["locutor", "arquivo", "codec", "transmissao",
           "ataque", "label", "trim", "fase"]

meta = pd.read_csv(
    Path("data/raw/keys/LA/CM/trial_metadata.txt"),
    sep=r"\s+", header=None, names=COLUNAS,
)

print(meta.shape)
for c in ["codec", "transmissao", "ataque", "label", "trim", "fase"]:
    print(f"\n=== {c} ===")
    print(meta[c].value_counts())
print("\n=== codec x label ===")
print(pd.crosstab(meta["codec"], meta["label"]))