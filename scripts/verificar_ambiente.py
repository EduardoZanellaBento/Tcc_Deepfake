"""
Verificação de ambiente

Roda este script depois de instalar o requirements.txt. Ele confirma que todas
as bibliotecas estão instaladas, mostra as versões, checa se há GPU disponível
para a CNN e verifica se o dataset já foi organizado nas pastas certas.

Uso:
    python scripts/verificar_ambiente.py
"""

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent


def checar(nome, fn):
    try:
        versao = fn()
        print(f"  [OK]   {nome:<14} {versao}")
        return True
    except Exception as e:
        print(f"  [FALHA] {nome:<14} -> {e}")
        return False


def main():
    print("=" * 60)
    print("VERIFICAÇÃO DE AMBIENTE — Detecção de Voz Sintética")
    print("=" * 60)

    print(f"\nPython: {sys.version.split()[0]}")
    if sys.version_info < (3, 9):
        print("  AVISO: recomenda-se Python 3.9 ou superior.")

    print("\nBibliotecas:")
    ok = True
    ok &= checar("numpy",        lambda: __import__("numpy").__version__)
    ok &= checar("pandas",       lambda: __import__("pandas").__version__)
    ok &= checar("scipy",        lambda: __import__("scipy").__version__)
    ok &= checar("librosa",      lambda: __import__("librosa").__version__)
    ok &= checar("soundfile",    lambda: __import__("soundfile").__version__)
    ok &= checar("sklearn",      lambda: __import__("sklearn").__version__)
    ok &= checar("matplotlib",   lambda: __import__("matplotlib").__version__)
    ok &= checar("seaborn",      lambda: __import__("seaborn").__version__)
    ok &= checar("yaml",         lambda: __import__("yaml").__version__)
    ok &= checar("tqdm",         lambda: __import__("tqdm").__version__)
    ok &= checar("webrtcvad",    lambda: __import__("webrtcvad").__version__)

    # TensorFlow é opcional nesta fase (só a CNN precisa). Não derruba o check.
    print("\nDeep learning (necessário somente para CNN):")
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        print(f"  [OK]   tensorflow     {tf.__version__}")
        print(f"         GPU disponível: {'SIM (' + str(len(gpus)) + ')' if gpus else 'NÃO (vai usar CPU)'}")
    except Exception as e:
        print(f"  [INFO] tensorflow ainda não instalado/configurado -> {e}")

    print("\nEstrutura de dados:")
    for p in ["data/raw", "data/processed", "data/features"]:
        existe = (RAIZ / p).is_dir()
        print(f"  [{'OK' if existe else '--'}]   {p}")

    audios = RAIZ / "data" / "raw"
    flacs = list(audios.rglob("*.flac"))[:1] if audios.is_dir() else []
    if flacs:
        print(f"\n  Dataset detectado: encontrei pelo menos 1 arquivo .flac em data/raw/")
    else:
        print(f"\n  AINDA SEM ÁUDIOS: baixe e descompacte o ASVspoof 2021 LA em data/raw/")

    print("\n" + "=" * 60)
    print("Ambiente OK!" if ok else "Há pendências — verifique as falhas acima.")
    print("=" * 60)


if __name__ == "__main__":
    main()
