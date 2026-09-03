"""
Serialização segura de JSON — tipos do NumPy/pandas viram tipos do Python
=========================================================================

POR QUE ISTO EXISTE:
    `json.dump` não serializa `np.float64`, `np.int64` nem `np.bool_`, e um
    DataFrame do pandas devolve esses tipos com facilidade (`.to_dict()`,
    `.idxmax()`, agregações). O erro só aparece na ÚLTIMA linha do script — ou
    seja, depois de a análise inteira ter rodado. Num repositório em que cada
    análise custa minutos de CPU, perder o resultado na hora de gravar é caro e
    perfeitamente evitável.

    Usar como `json.dump(obj, f, indent=2, ensure_ascii=False,
    default=json_seguro)`.

O QUE ELE NÃO FAZ, DE PROPÓSITO:
    não converte silenciosamente objetos arbitrários em string. Se o tipo não
    for um escalar/array do NumPy, ele levanta TypeError como o json faria —
    gravar `"<Modelo object at 0x...>"` num artefato de resultado seria pior
    que falhar.
"""

import numpy as np


def json_seguro(obj):
    """`default` do json.dump: converte escalares e arrays do NumPy."""
    if isinstance(obj, np.generic):      # np.float64, np.int64, np.bool_, ...
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Tipo não serializável em JSON: {type(obj).__name__}")
