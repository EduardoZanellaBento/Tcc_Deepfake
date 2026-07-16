"""
Reprodutibilidade — controle de aleatoriedade
=============================================

POR QUE ISTO EXISTE:
    Vários pontos do pipeline usam números aleatórios: o embaralhamento do split,
    o bootstrap das árvores do Random Forest, a amostragem do Random Search, a
    inicialização dos pesos da CNN. Sem semente fixa, cada execução produz um
    resultado diferente — e "meu modelo deu 94%" vira uma afirmação não verificável.
    Fixar a semente torna o experimento DETERMINÍSTICO: mesma entrada, mesmo
    resultado, em qualquer máquina.

O QUE FIXAR SEMENTE **NÃO** RESOLVE:
    Determinismo != robustez. Um resultado estável não é necessariamente um
    resultado bom: você pode ter fixado a semente numa partição "sortuda". É por
    isso que existe a validação cruzada — ela mede a VARIÂNCIA entre partições.
    Semente fixa dá reprodutibilidade; k-fold dá confiança estatística. São coisas
    diferentes e complementares.

A semente vem de config.yaml (`semente: 42`), nunca hard-coded no meio do código.
"""

import os
import random

import numpy as np


def fixar_seeds(semente: int = 42) -> int:
    """Fixa todas as fontes de aleatoriedade do processo Python.

    Cobre três geradores distintos, porque bibliotecas diferentes usam fontes
    diferentes:
      1. PYTHONHASHSEED -> a aleatoriedade de hash de str/bytes do próprio Python.
         Afeta a ordem de iteração de sets/dicts. Precisa ser definido ANTES do
         interpretador iniciar para ter efeito pleno; definir aqui cobre os
         subprocessos criados depois (ex.: workers do multiprocessing).
      2. random.seed      -> a biblioteca padrão do Python.
      3. np.random.seed   -> o gerador global do NumPy, que é o que o scikit-learn
         consulta quando `random_state=None`.

    IMPORTANTE — isto NÃO substitui `random_state`:
        No scikit-learn, sempre passe `random_state=semente` explicitamente
        (RandomForestClassifier, train_test_split, StratifiedKFold...). Depender do
        estado global é frágil: a ordem em que o código roda passa a alterar o
        resultado. `random_state` explícito torna cada componente reprodutível de
        forma independente.

    Args:
        semente: valor lido de config.yaml (`semente`).

    Returns:
        A semente usada, para poder registrá-la junto com os resultados.
    """
    os.environ["PYTHONHASHSEED"] = str(semente)
    random.seed(semente)
    np.random.seed(semente)
    return semente


def fixar_seeds_tensorflow(semente: int = 42) -> None:
    """Extensão para a CNN. Só importa o TF se ele for usado.

    Nota: mesmo com semente fixa, o treino em GPU pode ser NÃO-determinístico,
    porque operações cuDNN (redução em paralelo, atomics) somam em float em ordem
    variável. `enable_op_determinism()` força versões determinísticas dessas
    operações — ao custo de algum desempenho. Para um TCC, vale a troca: um
    resultado reproduzível vale mais do que alguns minutos de treino.
    """
    import tensorflow as tf

    fixar_seeds(semente)
    tf.random.set_seed(semente)
    tf.config.experimental.enable_op_determinism()