"""
Carregamento do config.yaml — ponto único de leitura
====================================================

POR QUE ISTO EXISTE:
    Config que ninguém lê é comentário com sintaxe YAML. Cada script que
    hard-coda um parâmetro (ex.: `fixar_seeds(42)`) cria um ponto onde mudar o
    config.yaml NÃO muda o experimento — exatamente o furo de reprodutibilidade
    que o arquivo central deveria fechar. Este helper garante que módulos e
    scripts leiam o MESMO arquivo, do MESMO jeito.

USO (sempre a partir da raiz, com `python -m`):
    RAIZ = Path(__file__).resolve().parents[1]   # em scripts/  -> parents[1]
                                                 # em src/x/    -> parents[2]
    cfg = carregar_config(RAIZ)
    semente = fixar_seeds(cfg["semente"])
"""

from pathlib import Path

import yaml


def carregar_config(raiz: Path) -> dict:
    """Lê e devolve config/config.yaml como dict.

    Args:
        raiz: raiz do projeto (a pasta que contém config/).

    Returns:
        Conteúdo do config.yaml (yaml.safe_load).
    """
    with open(raiz / "config" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)
