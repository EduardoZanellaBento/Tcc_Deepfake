"""
Extração de features acústicas — MFCC, ZCR, Centróide Espectral
===============================================================

Transforma cada áudio (já pré-processado) num VETOR de tamanho fixo, que é a
entrada tabular do Random Forest e do SVM.

O PROBLEMA que a agregação resolve:
    MFCC, ZCR e centróide são SÉRIES TEMPORAIS (um valor por janela de ~25 ms).
    Áudios diferentes têm nº de janelas diferentes -> vetores de tamanho variável.
    Mas RF e SVM exigem entrada de dimensão FIXA. Solução: resumir cada série no
    tempo por sua MÉDIA (comportamento central) e seu DESVIO-PADRÃO
    (variabilidade/dinâmica). Uma série de N janelas vira 2 números.

MASCARAMENTO DE PADDING [decisão aprovada pelo orientador]:
    O pré-processamento é `VAD -> padronizar para 4,0 s com zero-padding`. Se a
    agregação varresse os 4,0 s inteiros, média e desvio seriam calculados também
    sobre frames de silêncio artificial. Aqui a agregação usa SOMENTE os frames
    válidos: um frame i é válido quando seu centro (i*hop, convenção center=True
    do librosa) cai dentro do trecho de áudio real, antes do padding.

    DOIS ARGUMENTOS, e é importante não confundi-los (ver
    results/metricas/checagem_mascaramento.json):

    (1) VALIDADE DE MEDIDA — o argumento principal, que não depende de viés
        nenhum. Metade do tensor é padding na prática (mediana de 121 frames
        válidos em 251), e incluí-lo desloca cada feature em 30% na mediana,
        até 47% no pior caso. Uma feature assim descreve, em boa parte, a
        formatação do vetor, não o áudio. Mascarar restaura o significado da
        medida — por construção, e não por resultado empírico.

    (2) ASSIMETRIA ENTRE CLASSES — o canal de atalho. CUIDADO com a intuição
        fácil: a FRAÇÃO de padding é praticamente IGUAL nas duas classes
        (44,8% bonafide vs 45,2% spoof, n=200 balanceado), então NÃO se pode
        dizer que "o bonafide recebe mais padding". Ainda assim a distorção é
        assimétrica (assimetria padronizada até |0,47| em mfcc1_std,
        centroide_std e mfcc5_media), porque o deslocamento que o silêncio
        causa depende do CONTEÚDO ACÚSTICO de cada classe, não só da
        quantidade de zeros: misturar zeros a espectros diferentes não move
        as médias na mesma proporção.

    Contexto adicional: results/metricas/RECOMENDACAO_MASCARAMENTO.md (piloto
    A/B de 2.000 áudios; RF neutro-para-melhor com mascaramento) — ver o adendo
    de 26/08 nesse arquivo, que corrige o mecanismo alegado na versão original.

Tamanho do vetor final:
    MFCC (20 coef.) × {média, std} = 40
    ZCR              × {média, std} =  2
    Centróide        × {média, std} =  2
    -------------------------------------
    TOTAL                            = 44 features por áudio

COLUNAS DIAGNÓSTICAS (NÃO são features — ficam FORA do X):
    prop_fala, n_frames_validos, n_frames_total. Servem para auditar o VAD e o
    mascaramento; entrar com elas no modelo seria treinar sobre variáveis que não
    estão na fundamentação teórica do TC I. A exclusão é feita em
    `colunas_features` (src/data/split.py), que é o ponto ÚNICO que define o X.

IMPORTANTE — as features são salvas CRUAS (não padronizadas).
A padronização (StandardScaler) entra DENTRO do pipeline de cada modelo, ajustada
apenas no fold de treino. Padronizar o CSV inteiro de uma vez faria o scaler
"enxergar" média/desvio das amostras de teste = VAZAMENTO DE DADOS (data leakage),
inflando as métricas artificialmente.

Saída: data/features/features.csv  (uma linha por áudio)
"""

from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import librosa
import pandas as pd
from tqdm import tqdm

# NOTA: o módulo de pré-processamento chama-se `preprocessamento.py` neste repositório.
# Rode a partir da RAIZ do projeto com:  python -m src.features.extrair_features
from ..data.preprocessamento import preprocessar_audio


# ---------------------------------------------------------------------------
# Nomes das colunas — na MESMA ordem em que o vetor é montado
# ---------------------------------------------------------------------------
def nomes_features(n_mfcc: int) -> list[str]:
    cols = [f"mfcc{i+1}_media" for i in range(n_mfcc)]
    cols += [f"mfcc{i+1}_std" for i in range(n_mfcc)]
    cols += ["zcr_media", "zcr_std", "centroide_media", "centroide_std"]
    return cols


# Colunas gravadas no CSV que NÃO são features (diagnóstico). Fica aqui, ao lado
# de nomes_features, para haver UMA fonte do que é feature e do que é
# diagnóstico; `colunas_features` (src/data/split.py) consome esta lista.
COLUNAS_DIAGNOSTICO = ["prop_fala", "n_frames_validos", "n_frames_total"]


# ---------------------------------------------------------------------------
# Quantos frames de uma série caem dentro do áudio REAL (antes do padding)
# ---------------------------------------------------------------------------
def frames_validos(n_amostras_validas: int, hop: int, n_frames_total: int) -> int:
    """Converte "amostras válidas" em "frames válidos".

    Com `center=True` (default do librosa), o frame i está centrado na amostra
    i*hop. Logo o frame i é válido enquanto i*hop < n_amostras_validas, o que dá
    ceil(n_amostras_validas / hop) frames válidos. É EXATAMENTE a definição usada
    no piloto (scripts/piloto_mascaramento_padding.py) — manter idêntica é o que
    permite citar a evidência do piloto como justificativa desta implementação.

    Dois cuidados:
      - `min` com n_frames_total: a conta não pode prometer mais frames do que a
        série tem (o áudio já foi cortado no alvo).
      - `max` com 1: mesmo um áudio degenerado precisa de ao menos um frame,
        senão média/desvio de uma fatia vazia viram NaN e contaminam o CSV.
    """
    n = int(np.ceil(n_amostras_validas / hop))
    return max(min(n, n_frames_total), 1)


# ---------------------------------------------------------------------------
# Extração do vetor de features de UM áudio já pré-processado
# ---------------------------------------------------------------------------
def extrair_vetor(y: np.ndarray, sr: int, cfg: dict,
                  n_amostras_validas: int | None = None
                  ) -> tuple[np.ndarray, int, int]:
    """Calcula MFCC, ZCR e centróide e agrega cada um em (média, std) no tempo.

    Parâmetros (todos vindos do config.yaml):
      n_mfcc=20  -> nº de coeficientes cepstrais. Os primeiros capturam o envelope
                    grosso do espectro (ressonância do trato vocal); os mais altos,
                    detalhes finos onde artefatos de síntese tendem a aparecer.
      n_fft=512  -> tamanho da FFT por janela.
      win=400    -> ~25 ms a 16 kHz: a janela curta em que a fala é aproximadamente
                    estacionária (o janelamento discutido na fundamentação).
      hop=256    -> ~16 ms de passo (sobreposição entre janelas consecutivas).

    Args:
        n_amostras_validas: nº de amostras de áudio REAL (pós-VAD, pré-padding),
            devolvido por `preprocessar_audio`. Se None, a agregação varre a série
            inteira — comportamento ANTIGO, mantido só para reproduzir o
            features.csv pré-mascaramento em comparações A/B.

    Returns:
        (vetor_de_44_features, n_frames_validos, n_frames_total)
    """
    n_mfcc = cfg["features"]["n_mfcc"]
    n_fft = cfg["features"]["n_fft"]
    hop = cfg["features"]["hop_length"]
    win = cfg["features"]["win_length"]

    # MFCC -> (n_mfcc, n_janelas): textura fina / envelope espectral do trato vocal
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop, win_length=win
    )
    # ZCR -> (1, n_janelas): cruzamentos por zero. Ruído/fricativas de alta freq.
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=win, hop_length=hop)
    # Centróide espectral -> (1, n_janelas): "centro de massa" do espectro.
    # `win_length=400` explícito: o features.csv antigo foi extraído SEM este
    # parâmetro e o librosa caiu no default (janela = n_fft = 512), deixando o
    # centróide com resolução temporal diferente da do MFCC e do ZCR. Corrigido
    # a partir do lote único de re-extração.
    cent = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop, win_length=win
    )

    n_frames_total = int(mfcc.shape[1])
    if n_amostras_validas is None:
        n_valid = n_frames_total          # sem mascaramento (modo A/B histórico)
    else:
        n_valid = frames_validos(n_amostras_validas, hop, n_frames_total)

    # Agregação temporal MASCARADA: média e desvio-padrão apenas sobre os frames
    # válidos. O corte [:, :n_valid] É o mascaramento — os frames de padding
    # simplesmente não entram na conta.
    partes = []
    for serie in (mfcc, zcr, cent):
        janela = serie[:, :n_valid]
        partes.append(janela.mean(axis=1))
        partes.append(janela.std(axis=1))
    vetor = np.concatenate(partes).astype(np.float32)
    return vetor, n_valid, n_frames_total


# ---------------------------------------------------------------------------
# Worker: processa UM áudio (top-level para ser "picklável" pelo multiprocessing)
# ---------------------------------------------------------------------------
def _processar_um(args: tuple) -> dict:
    arquivo, caminho, label, classe, cfg = args
    try:
        y, prop, n_validas = preprocessar_audio(caminho, cfg)
        # `features.mascarar_padding` (config.yaml) governa a agregação. Default
        # True: se a chave sumir do config, o comportamento aprovado prevalece —
        # um config incompleto não pode reverter silenciosamente uma decisão
        # metodológica. False só existe para reproduzir o CSV antigo em A/B.
        if not cfg["features"].get("mascarar_padding", True):
            n_validas = None
        vetor, n_frames_val, n_frames_tot = extrair_vetor(
            y, cfg["audio"]["sample_rate"], cfg, n_amostras_validas=n_validas
        )
        linha = {
            "arquivo": arquivo,
            "label": label,
            "classe_binaria": classe,
            # --- diagnóstico (FORA do X — ver COLUNAS_DIAGNOSTICO) ---
            "prop_fala": round(prop, 4),       # quanto o VAD manteve
            "n_frames_validos": n_frames_val,  # quanto entrou na agregação
            "n_frames_total": n_frames_tot,    # quanto haveria sem mascarar
        }
        linha.update(dict(zip(nomes_features(cfg["features"]["n_mfcc"]), vetor)))
        return linha
    except Exception as e:
        # Um .flac corrompido não pode derrubar a varredura inteira.
        return {"arquivo": arquivo, "erro": repr(e)}


# ---------------------------------------------------------------------------
# Runner: varre TODOS os áudios em paralelo, com checkpoint/retomada
# ---------------------------------------------------------------------------
def executar(cfg: dict, raiz: Path, n_jobs: int | None = None,
             flush_a_cada: int = 2000, fase: str | None = None,
             limite: int | None = None, nome_saida: str = "features.csv"):
    """Extrai features dos áudios do universo do experimento e grava incrementalmente.

    UNIVERSO [decisão aprovada]: só `fase == cfg['dataset']['fase']` ('eval',
    148.176 áudios). O labels.csv tem 181.566 linhas porque inclui 'progress' e
    'hidden'; extrair os três seria gastar horas de CPU em áudios que o split
    descarta — e o 'hidden' vem com silêncio pré-cortado na origem
    (trim == 'only_speech'), contaminando qualquer análise de prop_fala.

    RESILIÊNCIA (por que checkpoint importa em ~150k arquivos):
      - Se o CSV de saída já existir, os áudios já extraídos são PULADOS. Uma
        queda no arquivo 140.000 não custa as horas anteriores: é só rodar de novo.
      - Grava em blocos (append) a cada `flush_a_cada` linhas, então o progresso
        fica em disco continuamente, não só no fim.
      - Ordem determinística (imap, não imap_unordered): mesma entrada -> mesmo CSV.

    ATENÇÃO À RETOMADA APÓS MUDANÇA DE CÓDIGO: o checkpoint pula por NOME de
    arquivo, não por versão da feature. Retomar sobre um CSV gerado por uma
    definição ANTIGA de feature produziria um arquivo meio antigo, meio novo —
    o pior artefato possível. Por isso o lote único começa com o features.csv
    anterior renomeado, nunca com ele no lugar.

    Args:
        fase: override pontual do universo (diagnósticos). None = vale o config.
        limite: se informado, extrai apenas uma AMOSTRA estratificada por classe
            desse tamanho — é o modo piloto, para validar o pipeline antes de
            gastar horas no lote único. Recusa escrever em features.csv.
        nome_saida: arquivo dentro de data/features/. O piloto deve escrever em
            outro nome para não encostar no artefato oficial.
    """
    n_jobs = n_jobs or cpu_count()
    if fase is None:
        fase = cfg["dataset"]["fase"]

    labels_csv = raiz / "data" / "processed" / "labels.csv"
    if not labels_csv.exists():
        raise FileNotFoundError(
            f"{labels_csv} não existe. Rode antes o notebook 02_leitura_dados.ipynb."
        )

    labels = pd.read_csv(labels_csv)
    n_bruto = len(labels)

    # ---- Filtro de universo ---------------------------------------------------
    if "fase" not in labels.columns:
        raise ValueError(
            "labels.csv não tem a coluna `fase`. Regere com "
            "`python -m src.data.carregar_dados` antes de extrair."
        )
    labels = labels[labels["fase"] == fase].reset_index(drop=True)
    if labels.empty:
        raise ValueError(f"Nenhum áudio com fase == '{fase}' no labels.csv.")
    print(f"Universo da extração: fase == '{fase}' -> {len(labels)} "
          f"de {n_bruto} áudios do labels.csv.")

    # ---- Modo piloto (amostra estratificada por classe) -----------------------
    if limite is not None:
        if nome_saida == "features.csv":
            raise ValueError(
                "Modo piloto (limite) não escreve em features.csv. "
                "Passe nome_saida='features_piloto.csv' ou similar."
            )
        semente = cfg["semente"]
        frac = limite / len(labels)
        partes = []
        for _, g in labels.groupby("classe_binaria"):
            # `max(1, ...)`: com classes muito desbalanceadas e limite pequeno, a
            # minoritária poderia arredondar para zero e o piloto deixaria de ter
            # bonafide — justamente a classe cujo padding queremos auditar.
            partes.append(g.sample(n=max(1, round(len(g) * frac)),
                                   random_state=semente))
        labels = (pd.concat(partes)
                    .sort_values("arquivo").reset_index(drop=True))
        print(f"MODO PILOTO: {len(labels)} áudios (amostra estratificada por "
              f"classe, semente {semente}) -> {nome_saida}")

    saida = raiz / "data" / "features" / nome_saida
    saida.parent.mkdir(parents=True, exist_ok=True)

    # Retomada: descobre o que já foi feito
    ja_feitos = set()
    if saida.exists():
        ja_feitos = set(pd.read_csv(saida, usecols=["arquivo"])["arquivo"])
        print(f"Retomando: {len(ja_feitos)} áudios já extraídos serão pulados.")

    tarefas = [
        (r.arquivo, r.caminho, r.label, r.classe_binaria, cfg)
        for r in labels.itertuples()
        if r.arquivo not in ja_feitos
    ]
    print(f"A extrair           : {len(tarefas)} áudios usando {n_jobs} processos.")
    if not tarefas:
        print("Nada a fazer — tudo já extraído.")
        return

    buffer, erros = [], []
    escrever_header = not saida.exists()

    with Pool(n_jobs) as pool:
        for res in tqdm(pool.imap(_processar_um, tarefas), total=len(tarefas)):
            if "erro" in res:
                erros.append(res)
                continue
            buffer.append(res)
            if len(buffer) >= flush_a_cada:
                pd.DataFrame(buffer).to_csv(saida, mode="a", header=escrever_header, index=False)
                escrever_header = False
                buffer = []

    if buffer:  # sobra final
        pd.DataFrame(buffer).to_csv(saida, mode="a", header=escrever_header, index=False)

    erros_csv = raiz / "data" / "features" / f"erros_{Path(nome_saida).stem}.csv"
    if erros:
        pd.DataFrame(erros).to_csv(erros_csv, index=False)
        print(f"ATENÇÃO: {len(erros)} arquivos falharam. Detalhes em {erros_csv}")
    elif erros_csv.exists():
        # Uma rodada sem falhas re-tentou (e venceu) tudo que estava no erros.csv
        # antigo; remove o arquivo para ele não apontar falhas que já não existem.
        erros_csv.unlink()
        print("Nenhuma falha nesta rodada — erros.csv anterior removido.")

    print(f"Concluído. Features salvas em {saida}")


if __name__ == "__main__":
    import os
    import yaml

    RAIZ = Path(__file__).resolve().parents[2]
    cfg = yaml.safe_load(open(RAIZ / "config" / "config.yaml", encoding="utf-8"))

    # nº de processos: use N_JOBS=4 python -m ... para limitar; padrão = todos os núcleos
    n_jobs = int(os.environ.get("N_JOBS", 0)) or None
    executar(cfg, RAIZ, n_jobs=n_jobs)
