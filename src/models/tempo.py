"""
Medição de tempo de inferência — protocolo único para RF, SVM (e CNN em CPU)
============================================================================

POR QUE ISTO EXISTE:
    O protocolo de medição mora no config.yaml (bloco `tempo`) e a comparação de
    custo entre modelos só é defensável se TODOS forem medidos pelo MESMO código,
    no mesmo hardware, com o mesmo n_jobs declarado. Se cada pipeline cronometrar
    do seu jeito (lote vs unitário, com/sem aquecimento), a tabela final compara
    protocolos, não modelos.

O QUE O PROTOCOLO MANDA (config.yaml -> tempo):
    - latência (batch = 1, cenário de uso real) e throughput (lote) SEPARADOS:
      o predict de 1 amostra paga o overhead fixo da chamada; dividir o tempo do
      lote pelo n subestima esse custo em ordens de grandeza;
    - descartar as primeiras execuções (aquecimento de cache/alocador);
    - reportar mediana + mínimo + máximo de N execuções, nunca medição única;
    - registrar hardware, versões e n_jobs no JSON de resultados.

O QUE ESTE HELPER **NÃO** MEDE (declarado no JSON, campo protocolo.escopo):
    só a PREDIÇÃO, a partir do vetor de features já extraído. Carregar o áudio,
    VAD, padding e extração de features ficam de fora — a chave
    `incluir_extracao_features` do config.yaml NÃO é lida aqui. O custo do
    pipeline completo é medido por scripts/tempo_pipeline_completo.py, que
    reaproveita as MESMAS funções de src/features/extrair_features.py.
"""

import platform
import time

import numpy as np


def medir_tempos(predizer, X: np.ndarray, cfg_tempo: dict) -> dict:
    """Cronometra `predizer` (callable X -> scores) sob o protocolo do config.

    Args:
        predizer: função que recebe uma matriz (n, d) e devolve os scores — a
            MESMA usada nas métricas (predict_proba no RF, decision_function no
            SVM), para que o tempo medido seja o do caminho real de decisão.
        X: matriz de avaliação (a latência usa X[:1]; o throughput, X inteiro).
        cfg_tempo: bloco `tempo` do config.yaml (repeticoes,
            descartar_aquecimento, medir_latencia, medir_throughput).

    Returns:
        dict com latencia_ms e throughput (mediana/min/max), pronto para o JSON.
    """
    reps = int(cfg_tempo["repeticoes"])
    aquec = int(cfg_tempo["descartar_aquecimento"])

    def _cronometrar(entrada: np.ndarray) -> list[float]:
        medidas = []
        for _ in range(aquec + reps):
            t0 = time.perf_counter()
            predizer(entrada)
            medidas.append(time.perf_counter() - t0)
        return medidas[aquec:]          # aquecimento descartado

    resultado: dict = {
        "protocolo": {
            "repeticoes": reps,
            "descartar_aquecimento": aquec,
            "fonte": "config.yaml -> tempo",
            # HONESTIDADE DE ESCOPO (R4a): "fonte: config.yaml -> tempo" sozinho
            # sugeria conformidade com TODO o bloco `tempo`, inclusive a chave
            # `incluir_extracao_features: true` — que este helper NÃO lê e nunca
            # leu. Os dois campos abaixo dizem, no próprio artefato, o que ficou
            # de fora. O custo do pré-processamento é medido em
            # scripts/tempo_pipeline_completo.py.
            "escopo": ("somente a predição do modelo (predict_proba / "
                       "decision_function) a partir do vetor de features JÁ "
                       "EXTRAÍDO — NÃO inclui carregamento do áudio, VAD, nem "
                       "extração de features"),
            "incluir_extracao_features": False,
            "onde_o_pipeline_completo_e_medido":
                "results/metricas/tempo_pipeline_completo.json "
                "(scripts/tempo_pipeline_completo.py)",
        },
    }
    if cfg_tempo.get("medir_latencia", True):
        lat = _cronometrar(X[:1])
        resultado["latencia_ms"] = {
            "batch": 1,
            "mediana": round(1000 * float(np.median(lat)), 4),
            "min": round(1000 * min(lat), 4),
            "max": round(1000 * max(lat), 4),
        }
    if cfg_tempo.get("medir_throughput", True):
        thr = _cronometrar(X)
        mediana_s = float(np.median(thr))
        resultado["throughput"] = {
            "batch": int(len(X)),
            "total_s_mediana": round(mediana_s, 4),
            "total_s_min": round(min(thr), 4),
            "total_s_max": round(max(thr), 4),
            "ms_por_audio_mediana": round(1000 * mediana_s / len(X), 4),
            "audios_por_segundo_mediana": round(len(X) / mediana_s, 1),
        }
    return resultado


def ambiente(n_jobs_inferencia: int) -> dict:
    """Hardware, versões e n_jobs — o contexto sem o qual os tempos não significam nada."""
    import sklearn
    import pandas as pd

    return {
        "cpu": platform.processor(),
        "maquina": platform.machine(),
        "sistema": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "versoes": {
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
        },
        # n_jobs da INFERÊNCIA cronometrada — fixado igual para todos os
        # modelos clássicos (exigência do protocolo de medição do config.yaml).
        "n_jobs_inferencia": n_jobs_inferencia,
    }
