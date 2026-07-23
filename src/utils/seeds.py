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


def fixar_seeds_torch(semente: int = 42, estrito: bool = True) -> int:
    """Extensão para a CNN em PyTorch. Só importa o torch se ele for usado.

    POR QUE NÃO BASTA `torch.manual_seed`:
        O treino em GPU tem fontes de não-determinismo que a semente não alcança.
        Operações do cuDNN (convolução, redução em paralelo, atomics) somam em
        ponto flutuante numa ORDEM que varia entre execuções. Como soma de float
        não é associativa, (a+b)+c != a+(b+c) nos últimos bits — e essa diferença
        se amplifica ao longo do treino. Resultado: mesma semente, pesos finais
        diferentes. Por isso são necessárias quatro travas, não uma.

      1. manual_seed / manual_seed_all -> pesos iniciais, dropout, embaralhamento.
      2. cudnn.deterministic = True    -> força algoritmos determinísticos.
      3. cudnn.benchmark = False       -> o benchmark escolhe o algoritmo mais
         rápido medindo em tempo de execução; essa escolha VARIA conforme a carga
         da máquina. Rápido, porém não reprodutível.
      4. CUBLAS_WORKSPACE_CONFIG       -> exigido pelo cuBLAS (CUDA >= 10.2) para
         que operações de matriz sejam determinísticas. Precisa ser definido ANTES
         da primeira chamada CUDA, por isso vem no topo da função.

    ATENÇÃO: o DataLoader com num_workers > 0 cria processos filhos com
    sementes próprias. Se houver embaralhamento lá dentro, será preciso passar
    `generator=torch.Generator().manual_seed(semente)` e um `worker_init_fn`.
    Esta função NÃO cobre esse caso.

    Args:
        semente: valor lido de config.yaml.
        estrito: se True, `use_deterministic_algorithms` levanta erro ao encontrar
            uma operação sem versão determinística. Se False, apenas avisa. Comece
            com True; se alguma camada da CNN não tiver implementação determinística,
            passe False e REGISTRE isso na metodologia — é uma limitação honesta,
            bem melhor do que silenciar o problema.

    Returns:
        A semente usada, para registrar junto com os resultados.
    """
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    import torch

    fixar_seeds(semente)
    torch.manual_seed(semente)
    torch.cuda.manual_seed_all(semente)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=not estrito)
    return semente