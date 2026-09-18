"""
Geração dos espectrogramas log-Mel — a entrada da CNN
=====================================================

Transforma cada áudio (já pré-processado) numa "imagem" de tamanho FIXO
`(128, 251)` em dB, que é a entrada da rede convolucional. É o análogo, para a
CNN, do que `extrair_features.py` faz para o RF e o SVM — e é de propósito que
os dois módulos se pareçam: o padrão de checkpoint/retomada, a guarda de
esquema e o `.meta.json` já foram pagos no Bloco 2 e são reaproveitados aqui.

O QUE ESTE MÓDULO NÃO FAZ, E POR QUÊ (a regra que sustenta o trabalho inteiro):

    NÃO reimplementa carregamento, VAD nem padding. Ele CHAMA
    `preprocessar_audio` de src/data/preprocessamento.py — a mesmíssima função
    que o ramo clássico usa.

    NÃO recalcula "frames válidos". Ele delega a `frames_validos()` de
    src/features/extrair_features.py — a mesmíssima função que o mascaramento da
    agregação usa.

    A frase «os três modelos recebem o mesmo pré-processamento», que é a base da
    pergunta de pesquisa, só é verdadeira enquanto isto valer. Duas
    implementações da mesma intenção divergem em SILÊNCIO: nada quebra, nada
    avisa, e a comparação deixa de significar o que o texto diz que significa.

TODO PARÂMETRO É PASSADO EXPLICITAMENTE AO LIBROSA, inclusive os que coincidem
com o default da versão instalada (`window`, `pad_mode`, `htk`, `norm`, `amin` —
registrados em `config.yaml -> espectrograma.defaults_librosa_registrados`).
Registrar um valor no YAML e confiar no default no código cria a APARÊNCIA de
controle sem o controle: uma atualização de biblioteca mudaria a definição do
experimento sem uma linha de diff.

ESCALA EM dB COM `ref=1.0` — NUNCA `ref=np.max` [decisão P5, aprovada]:
    `ref=np.max` é o exemplo mais comum em tutorial de librosa e seria um erro
    aqui: é uma normalização POR EXEMPLO, relativa ao máximo do próprio
    espectrograma, que contradiria a normalização global por faixa Mel da P3.
    A asserção 6 de `scripts/verificar_espectrogramas.py` existe só para pegar
    isto: com `ref=1.0`, multiplicar o áudio por 10 desloca o espectrograma
    inteiro em exatamente 20 dB; com `ref=np.max`, o deslocamento seria ~0.

TENSORES GRAVADOS CRUS (sem normalização) [decisão de projeto — registre-a]:
    a normalização da P3 (média/desvio por faixa Mel, estatísticas só do treino)
    é aplicada EM TEMPO DE CARGA, a partir de `normalizacao_cnn.json`. Razão de
    cronograma, não de método: o refit final da P6 RECOMPUTA média e desvio nos
    30k, e se a normalização estivesse assada nos tensores o refit exigiria
    regerar 9,57 GB — dois dias que o cronograma não tem. Separar o tensor cru
    da estatística é o que torna o refit barato.

ARMAZENAMENTO: UM MEMMAP POR CONJUNTO, NÃO 74.453 ARQUIVOS SOLTOS
    Decisão de ENGENHARIA (não metodológica): o volume é o mesmo (9,57 GB), mas
    o NTFS sofre com 74 mil arquivos pequenos e abrir 30 mil deles por época faz
    o `DataLoader` virar o gargalo do treino. Três `.npy` lidos por
    `np.load(..., mmap_mode="r")` dão leitura preguiçosa por índice e I/O
    sequencial.

    O PREÇO do memmap é que a linha `i` só significa algo ATRAVÉS DO ÍNDICE, e
    é aí que mora o erro mais caro deste bloco: uma ordem não determinística
    (`imap_unordered`, `glob`) desalinha tensor e rótulo, nada quebra, e o
    sintoma só aparece no B4.4 como acurácia ~50%. Contra isso, três travas:
      1. o índice é escrito ANTES de gerar, ordenado por `arquivo`;
      2. o `Pool` usa `imap` (ordenado), e as linhas são escritas em sequência;
      3. um áudio que falha ABORTA o conjunto (ver `_escrever_conjunto`) — ao
         contrário do CSV de features, onde uma linha a menos é só uma linha a
         menos, aqui uma linha pulada deixaria zeros sob um rótulo real.

Rode a partir da RAIZ do projeto:
    python -m src.features.gerar_espectrogramas --limite 200 --nome-saida piloto
    python -m src.features.gerar_espectrogramas          # lote oficial (B4.2)
"""

import hashlib
import json
import shutil
import subprocess
from multiprocessing import Pool, cpu_count
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile
from tqdm import tqdm

from ..data.preprocessamento import preprocessar_audio
from ..data.split import filtrar_treino_braco
from .extrair_features import frames_validos

# Nome do conjunto de treino nos artefatos. Carrega o "_30k" porque o treino da
# CNN é a SUBAMOSTRA do braço principal, não o treino completo de 103.723 — e o
# nome do arquivo é o lugar mais difícil de ignorar essa distinção.
CONJUNTO_TREINO = "treino_30k"

# Colunas do índice, na ordem exata em que são gravadas. A linha `i` do memmap
# corresponde à linha `i` deste CSV — é o contrato inteiro do armazenamento.
COLUNAS_INDICE = ["arquivo", "linha", "classe_binaria", "n_frames_validos"]


# ---------------------------------------------------------------------------
# O núcleo: um áudio pré-processado -> um log-Mel (128, 251) em dB
# ---------------------------------------------------------------------------
def gerar_um(y: np.ndarray, sr: int, e: dict) -> np.ndarray:
    """log-Mel em dB, `(n_mels, largura)` float32, SEM normalização.

    Args:
        y: forma de onda de tamanho fixo devolvida por `preprocessar_audio`.
        sr: taxa de amostragem (16 kHz).
        e: o bloco `espectrograma:` do config.yaml.

    Returns:
        `(128, 251)` float32 — log-Mel em dB, cru.
    """
    d = e["defaults_librosa_registrados"]

    # Passo 1 — mel-espectrograma de POTÊNCIA.
    # `htk` e `norm` NÃO são parâmetros nominais de `melspectrogram`: chegam por
    # **kwargs e são repassados a `librosa.filters.mel`. Isso é verificado em
    # tempo de execução por `verificar_repasse_filterbank()` (chamada por
    # `executar`) e o caminho efetivamente usado vai para o .meta.json, em
    # `caminho_filterbank` — porque uma versão de librosa que deixasse de
    # repassá-los trocaria a convenção da escala Mel (slaney/htk) em silêncio, e
    # com ela a paridade de frequências centrais com o ramo clássico.
    S = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_fft=e["n_fft"],               # 1024 — densidade da DFT, não resolução
        win_length=e["win_length"],     # 400 (~25 ms) — a resolução real
        hop_length=e["hop_length"],     # 256 -> 251 frames em 64.000 amostras
        window=d["window"],             # "hann"
        center=e["center"],             # True -> frame i centrado na amostra i*hop
        pad_mode=d["pad_mode"],         # "constant"
        power=e["power"],               # 2.0 -> potência (|STFT|^2)
        n_mels=e["n_mels"],             # 128
        fmin=e["fmin"], fmax=e["fmax"],  # 0 .. 8000 (banda cheia; sem fmax=4000)
        htk=d["htk"],                   # False -> paridade com librosa.feature.mfcc
        norm=d["mel_norm"],             # "slaney" -> idem
    )

    # Passo 2 — dB com ref=1.0. NUNCA np.max: `ref=np.max` É normalização por
    # exemplo e contradiria a normalização global da P3 (ver docstring do módulo).
    S_db = librosa.power_to_db(
        S, ref=e["db_ref"], amin=d["amin"], top_db=e["top_db"]
    )
    return S_db.astype(np.float32)


def verificar_repasse_filterbank(e: dict) -> str:
    """Confere que `htk` e `norm` chegam de fato à filterbank e devolve o caminho.

    Em librosa 0.11 `melspectrogram` repassa `htk`/`norm` por **kwargs a
    `librosa.filters.mel`. Uma versão que os IGNORASSE aceitaria os argumentos
    calados e produziria a filterbank com os defaults — trocando a convenção da
    escala Mel sem erro nenhum. Este teste de 30 ms compara o resultado de
    `htk=False` com `htk=True` num ruído sintético: se forem iguais, o repasse
    não aconteceu.

    Returns:
        A string gravada em `caminho_filterbank` do .meta.json.
    """
    d = e["defaults_librosa_registrados"]
    rng = np.random.default_rng(0)
    y = (rng.standard_normal(4096) * 0.1).astype(np.float32)
    comum = dict(y=y, sr=e["sample_rate"], n_fft=e["n_fft"],
                 win_length=e["win_length"], hop_length=e["hop_length"],
                 window=d["window"], center=e["center"], pad_mode=d["pad_mode"],
                 power=e["power"], n_mels=e["n_mels"],
                 fmin=e["fmin"], fmax=e["fmax"])
    a = librosa.feature.melspectrogram(**comum, htk=d["htk"], norm=d["mel_norm"])
    b = librosa.feature.melspectrogram(**comum, htk=not d["htk"], norm=d["mel_norm"])
    c = librosa.feature.melspectrogram(**comum, htk=d["htk"], norm=None)
    if np.allclose(a, b) or np.allclose(a, c):
        raise RuntimeError(
            f"librosa {librosa.__version__} NÃO repassa htk/norm de "
            "melspectrogram para filters.mel: trocar htk/norm não mudou a saída. "
            "A filterbank sairia com os defaults da biblioteca, quebrando a "
            "paridade de escala Mel com librosa.feature.mfcc. Monte a filterbank "
            "à mão (librosa.filters.mel(...) @ np.abs(stft)**2) e registre o "
            "caminho novo em caminho_filterbank."
        )
    return "melspectrogram(htk,norm)"


# ---------------------------------------------------------------------------
# Máscara: a MESMA definição de frame válido do ramo clássico
# ---------------------------------------------------------------------------
def frames_validos_espectrograma(n_amostras_validas: int, cfg_esp: dict,
                                 n_frames_total: int) -> int:
    """Delega a `frames_validos()` com o hop do bloco `espectrograma`.

    Existe como função nomeada (em vez de uma chamada solta) para deixar num só
    lugar a resposta à pergunta «de onde vem a máscara da CNN?»: dela, que só
    repassa. Não há aritmética nova aqui, de propósito — recontar frames é a
    armadilha que produziria uma máscara divergindo do ramo clássico por ±1
    frame, indetectável no treino.

    ATENÇÃO À FRONTEIRA (medido, não suposto): com `win_length=400 > hop=256`,
    a janela do frame `n_valid` ainda alcança até 200 amostras de áudio real,
    então esse frame — o PRIMEIRO inválido — é um frame de TRANSIÇÃO, não
    silêncio puro. O platô exato do padding começa em `n_valid + 1`. Como a
    máscara guarda `[:, :n_valid]`, o frame de transição fica fora dos dois
    ramos e não contamina nada; mas quem for assertar «a região inválida é
    constante» tem de começar em `n_valid + 1` (asserção 8 da checagem).
    """
    return frames_validos(n_amostras_validas, cfg_esp["hop_length"], n_frames_total)


# ---------------------------------------------------------------------------
# Esquema / assinatura da definição vigente
# ---------------------------------------------------------------------------
def esquema_esperado(cfg: dict) -> dict:
    """Assinatura da DEFINIÇÃO do espectrograma vigente (sem escopo nem hashes).

    É a definição ÚNICA do que caracteriza um tensor deste experimento: a
    guarda de retomada compara este dicionário contra o gravado no .meta.json, e
    `scripts/verificar_espectrogramas.py` compara o .meta.json contra o
    config.yaml (asserção 4). Os 16 parâmetros do bloco `espectrograma:` mais os
    5 defaults do librosa registrados.
    """
    e = cfg["espectrograma"]
    d = e["defaults_librosa_registrados"]
    largura = largura_esperada(cfg)
    return {
        "semente": cfg["semente"],
        "tipo": e["tipo"],
        "sample_rate": e["sample_rate"],
        "n_fft": e["n_fft"],
        "win_length": e["win_length"],
        "hop_length": e["hop_length"],
        "n_mels": e["n_mels"],
        "fmin": e["fmin"],
        "fmax": e["fmax"],
        "power": e["power"],
        "escala": e["escala"],
        "db_ref": e["db_ref"],
        "top_db": e["top_db"],
        "center": e["center"],
        "altura": e["altura"],
        "largura": largura,
        "normalizacao": e["normalizacao"],
        "mascarar_agregacao": e["mascarar_agregacao"],
        # os 5 defaults do librosa, registrados para que uma atualização de
        # biblioteca não mude a definição do experimento em silêncio
        "window": d["window"],
        "pad_mode": d["pad_mode"],
        "htk": d["htk"],
        "mel_norm": d["mel_norm"],
        "amin": d["amin"],
        "dtype": "float32",
    }


def largura_esperada(cfg: dict) -> int:
    """`1 + n_amostras // hop` — o 251 RECALCULADO, nunca literal.

    Com `center=True` o librosa devolve `1 + n // hop` frames para `n` amostras.
    Este número é derivado de `audio.duracao_segundos`, `audio.sample_rate` e
    `espectrograma.hop_length`; se ele divergir de `espectrograma.largura`, o
    config está inconsistente e o gerador se recusa a rodar (ver `executar`) —
    era exatamente a divergência que o rascunho da v1 carregava.
    """
    n_amostras = int(cfg["audio"]["sample_rate"] * cfg["audio"]["duracao_segundos"])
    return 1 + n_amostras // cfg["espectrograma"]["hop_length"]


def _md5(caminho: Path) -> str:
    h = hashlib.md5()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


# Ordem canônica dos conjuntos nos artefatos — a mesma que `catalogo` produz.
ORDEM_CONJUNTOS = (CONJUNTO_TREINO, "validacao", "teste")


def _lote_em_disco(saida_dir: Path, prefixo: str
                   ) -> tuple[dict[str, int], dict[str, str]]:
    """Escopo e MD5 dos índices REALMENTE em disco para este prefixo.

    O B4.2 gera os três conjuntos em execuções SEPARADAS (uma por conjunto,
    para que uma queda no meio do terceiro não custe os dois primeiros), e cada
    execução reescreve o `.meta.json`. Montar o escopo só com o que a execução
    corrente gerou apagaria o registro das anteriores: depois de rodar `teste`
    por último, o `.meta.json` alegaria um lote de 22.227 tensores. Reler o
    disco acerta os dois lados — o registro acumula entre execuções e nunca
    afirma a existência de um índice que não está lá.

    Levanta se um índice e seu memmap discordarem no nº de linhas: isso é um
    conjunto interrompido, e registrá-lo como completo no `.meta.json` seria
    exatamente o tipo de alegação falsa que este arquivo existe para impedir.
    """
    achados = {}
    for idx_path in saida_dir.glob(f"{prefixo}indice_*.csv"):
        nome = idx_path.name[len(prefixo) + len("indice_"):-len(".csv")]
        achados[nome] = idx_path
    ordem = ([n for n in ORDEM_CONJUNTOS if n in achados]
             + sorted(n for n in achados if n not in ORDEM_CONJUNTOS))

    escopo, hashes = {}, {}
    for nome in ordem:
        idx_path = achados[nome]
        n_idx = len(pd.read_csv(idx_path, usecols=["linha"]))
        npy = saida_dir / f"{prefixo}{nome}.npy"
        if not npy.exists():
            raise ValueError(
                f"{idx_path.name} existe mas {npy.name} não. Um índice sem "
                "memmap não descreve lote nenhum: arquive-o antes de gerar."
            )
        n_npy = int(np.load(npy, mmap_mode="r").shape[0])
        if n_npy != n_idx:
            raise ValueError(
                f"{npy.name} tem {n_npy} linhas e {idx_path.name} tem {n_idx}. "
                "O conjunto ficou incompleto; rode de novo o comando daquele "
                "conjunto (a retomada continua de onde parou) antes de fechar "
                "o .meta.json."
            )
        escopo[nome] = n_idx
        hashes[idx_path.name] = _md5(idx_path)
    return escopo, hashes


def _meta_geracao(cfg: dict, raiz: Path, escopo: dict[str, int],
                  hashes_indice: dict[str, str] | None = None) -> dict:
    """Assinatura completa gravada em `<prefixo>espectrogramas.meta.json`.

    Mesmo mecanismo do `features.meta.json`, que salvou o Bloco 2: transforma
    «confio no nome do arquivo» em «confiro a definição do tensor».

    Args:
        escopo: `{nome_conjunto: n_audios}`, na ordem de geração.
        hashes_indice: MD5 de cada índice CSV, quando já existirem.
    """
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=raiz, text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        commit = "desconhecido"
    try:
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=raiz,
            stderr=subprocess.DEVNULL).strip())
    except Exception:
        dirty = None

    meta = dict(esquema_esperado(cfg))
    meta.update({
        "escopo": list(escopo.keys()),
        "n_por_conjunto": dict(escopo),
        "caminho_filterbank": verificar_repasse_filterbank(cfg["espectrograma"]),
        # Os tensores em disco são CRUS. Esta nota existe no artefato porque
        # quem abrir o .npy daqui a seis meses precisa saber que falta um passo.
        "nota_normalizacao": (
            "os tensores em disco NAO estao normalizados; media/desvio por faixa "
            "Mel sao aplicados em tempo de carga, a partir de "
            "normalizacao_cnn.json (B4.3)"
        ),
        "hash_md5_features_csv": _md5(raiz / "data" / "features" / "features.csv"),
        "hash_md5_split_csv": _md5(raiz / "data" / "processed" / "split.csv"),
        "hash_md5_subamostra_csv": _md5(raiz / cfg["experimento"]["caminho_subamostra"]),
        "hash_md5_indice_por_conjunto": dict(hashes_indice or {}),
        "versoes": {
            "librosa": librosa.__version__,
            "numpy": np.__version__,
            "soundfile": soundfile.__version__,
        },
        "commit_git": commit,
        "git_dirty": dirty,
    })
    return meta


# chaves que NÃO entram na comparação da guarda de retomada, e por quê:
#   escopo / n_por_conjunto / hash_md5_indice_por_conjunto -> mudam entre piloto
#     e lote e entre conjuntos; a integridade do escopo é conferida pelo próprio
#     índice, linha a linha;
#   commit_git / git_dirty -> um commit de documentação não muda um tensor;
#   versoes -> só `librosa` define a transformada e é conferida à parte
#     (_CHAVE_VERSAO_CRITICA); bloquear a retomada de um lote de 74.453 tensores
#     por um bump de patch do numpy seria custo sem ganho.
#   estado -> rotulo do ciclo de vida do artefato (assinatura sem lote / lote
#     completo e congelado). Descreve em que ponto o lote esta, nao o tensor;
#     compara-lo travaria a PRIMEIRA geracao sobre a assinatura do B4.1.
_CHAVES_INFORMATIVAS = ("escopo", "n_por_conjunto", "hash_md5_indice_por_conjunto",
                        "commit_git", "git_dirty", "versoes", "nota_normalizacao",
                        "estado")
_CHAVE_VERSAO_CRITICA = "librosa"


def _validar_retomada(saida: Path, meta_atual: dict) -> None:
    """Trava contra corrupção silenciosa ao retomar sobre um lote pré-existente.

    Args:
        saida: o `<prefixo>espectrogramas.meta.json` do lote. Aqui a assinatura
            é o PRÓPRIO artefato conferido — ao contrário de
            `extrair_features._validar_retomada`, que recebe o CSV porque
            também precisa ler o cabeçalho dele. Um memmap não tem cabeçalho de
            esquema para ler: o que define o tensor está todo no .meta.json.

    POR QUE ESTA GUARDA EXISTE (a versão espectrograma do problema que o Bloco 2
    já pagou): a retomada pula linhas por POSIÇÃO, não por versão da definição.
    Se o memmap em disco veio de outra definição — outro `n_fft`, outro `top_db`,
    `ref=np.max` — a retomada produziria um tensor meio antigo, meio novo, no
    MESMO arquivo, sob o MESMO índice. Nada quebraria: a CNN treinaria em cima de
    duas definições misturadas e o resultado seria irreproduzível sem que
    ninguém pudesse notar. Um aviso em prosa não é uma trava; esta função é.
    """
    meta_path = saida
    if not meta_path.exists():
        raise ValueError(
            f"Há .npy em disco mas {meta_path.name} não existe. Um lote sem "
            "assinatura não é retomável: não há como saber qual definição o "
            "gerou. Renomeie/arquive os .npy e os índices antes de rodar."
        )
    gravada = json.loads(meta_path.read_text(encoding="utf-8"))
    chaves = [k for k in meta_atual if k not in _CHAVES_INFORMATIVAS]
    divergentes = {k: (gravada.get(k), meta_atual[k])
                   for k in chaves if gravada.get(k) != meta_atual[k]}
    v_gravada = (gravada.get("versoes") or {}).get(_CHAVE_VERSAO_CRITICA)
    v_atual = meta_atual["versoes"][_CHAVE_VERSAO_CRITICA]
    if v_gravada != v_atual:
        divergentes[f"versoes.{_CHAVE_VERSAO_CRITICA}"] = (v_gravada, v_atual)
    if divergentes:
        raise ValueError(
            f"{meta_path} registra outra definição de espectrograma "
            f"(gravado vs atual): {divergentes}.\n"
            "Se já há .npy em disco, retomar produziria um memmap meio antigo, "
            "meio novo, sob o mesmo índice. Se ainda não há, esta é a trava de "
            "CONGELAMENTO: a definição foi validada e commitada com estes "
            "valores, e gerar o lote com outros tornaria a checagem do B4.1 "
            "inválida para os tensores que a CNN vai consumir.\n"
            "Ou reverta o config, ou reabra a decisão por escrito e arquive os "
            ".npy, os índices e o .meta.json antes de regerar."
        )


# ---------------------------------------------------------------------------
# Catálogo: quais áudios entram, em que conjunto, e com que n_frames_validos
# ---------------------------------------------------------------------------
def catalogo(cfg: dict, raiz: Path,
             conjuntos: tuple[str, ...] = ("treino", "validacao", "teste")
             ) -> pd.DataFrame:
    """Monta a lista de áudios a gerar, ORDENADA POR `arquivo` dentro de cada conjunto.

    ESCOPO: 74.453, não 148.176 [decisão de escopo do B4.1]. O braço de
    referência é RF-only e nunca precisa de espectrograma; gerar os 148.176
    produziria 73.723 tensores que nada consome, mais ~9,5 GB e horas de CPU.

        split.csv -> conjunto == 'treino'    ∩ subamostra_30k  -> 30.000
                  -> conjunto == 'validacao' (completo)        -> 22.226
                  -> conjunto == 'teste'     (completo)        -> 22.227

    O filtro do treino é `filtrar_treino_braco(..., "principal", ...)` de
    src/data/split.py, que já carrega a guarda de integridade («subamostra tem N
    IDs, mas só M estão no treino atual»). Não se refaz o merge à mão.

    Colunas devolvidas: conjunto, nome_conjunto, arquivo, caminho,
    classe_binaria, n_frames_validos.
    """
    e = cfg["espectrograma"]

    # ---- ASSERÇÃO DE HOP: a que impede reutilizar a coluna errada -----------
    # `n_frames_validos` vem do features.csv, onde foi calculado com o hop do
    # bloco `features:`. O bloco `espectrograma:` usa o MESMO 256 — mas por
    # coincidência de parâmetro, não por construção. Se um dia divergirem, a
    # coluna do features.csv passa a descrever outra grade temporal e a máscara
    # da CNN fica errada sem sintoma. Falha explícita em vez de `assert` porque
    # `python -O` remove asserts, e esta é uma trava de integridade de dados.
    if e["hop_length"] != cfg["features"]["hop_length"]:
        raise ValueError(
            f"espectrograma.hop_length={e['hop_length']} != "
            f"features.hop_length={cfg['features']['hop_length']}. O índice traz "
            "n_frames_validos do features.csv, calculado com o hop de `features:`; "
            "com hops diferentes esse número descreve OUTRA grade temporal e a "
            "máscara da CNN divergiria do ramo clássico. Recalcule "
            "n_frames_validos com o hop do espectrograma antes de prosseguir."
        )

    split = pd.read_csv(raiz / "data" / "processed" / "split.csv")
    desconhecidos = set(conjuntos) - set(split["conjunto"].unique())
    if desconhecidos:
        raise ValueError(
            f"conjunto(s) inexistente(s) no split.csv: {sorted(desconhecidos)}. "
            f"Válidos: {sorted(split['conjunto'].unique())}."
        )

    # n_frames_validos vem do features.csv CONGELADO, por merge em `arquivo` —
    # nunca recalculado. É o que garante que a máscara da CNN e o mascaramento do
    # ramo clássico usam O MESMO NÚMERO para o mesmo áudio. `classe_binaria` vai
    # para o índice para o Dataset não reabrir o features.csv a cada época.
    feats = pd.read_csv(raiz / "data" / "features" / "features.csv",
                        usecols=["arquivo", "classe_binaria",
                                 "n_frames_validos", "n_frames_total"])
    # `caminho` só existe no labels.csv.
    labels = pd.read_csv(raiz / "data" / "processed" / "labels.csv",
                         usecols=["arquivo", "caminho"])

    base = split.merge(feats, on="arquivo", how="inner")
    if len(base) != len(split):
        raise ValueError(
            f"Merge split×features perdeu linhas: split={len(split)}, "
            f"resultado={len(base)}. Há áudio do split sem features — o "
            "split.csv foi gerado a partir de outro features.csv."
        )
    base = base.merge(labels, on="arquivo", how="inner")
    if len(base) != len(split):
        raise ValueError(
            f"Merge com labels.csv perdeu linhas: {len(split)} -> {len(base)}. "
            "Há áudio do split ausente do labels.csv."
        )

    partes = []
    for conj in conjuntos:
        linhas = base[base["conjunto"] == conj]
        if conj == "treino":
            # braço PRINCIPAL: a CNN treina na subamostra de 30k, a mesma do RF
            # e do SVM — é o ambiente experimental da comparação.
            linhas = filtrar_treino_braco(linhas, "principal", cfg, raiz)
            nome = CONJUNTO_TREINO
        else:
            nome = conj
        # ORDEM DETERMINÍSTICA POR `arquivo`: é o contrato do memmap. Sem isto,
        # a linha `i` deixa de corresponder ao rótulo `i` e o sintoma só aparece
        # no B4.4, como acurácia ~50%.
        linhas = linhas.sort_values("arquivo").reset_index(drop=True)
        linhas = linhas.assign(nome_conjunto=nome)
        partes.append(linhas)

    cols = ["conjunto", "nome_conjunto", "arquivo", "caminho",
            "classe_binaria", "n_frames_validos", "n_frames_total"]
    return pd.concat(partes, ignore_index=True)[cols]


def _amostra_piloto(cat: pd.DataFrame, limite: int, semente: int) -> pd.DataFrame:
    """Amostra estratificada por (conjunto × classe), no padrão de `extrair_features`.

    Estratificar também por CONJUNTO (e não só por classe) garante que o piloto
    exercite os três caminhos do escopo — inclusive o filtro da subamostra no
    treino — em vez de só o maior deles.
    """
    frac = limite / len(cat)
    partes = []
    for _, g in cat.groupby(["nome_conjunto", "classe_binaria"], sort=True):
        # `max(1, ...)`: com 9:1 de desbalanceamento e limite pequeno, a
        # minoritária poderia arredondar para zero e o piloto deixaria de ter
        # bonafide — justamente a classe que a inspeção visual precisa ver.
        partes.append(g.sample(n=max(1, round(len(g) * frac)), random_state=semente))
    return (pd.concat(partes)
              .sort_values(["nome_conjunto", "arquivo"])
              .reset_index(drop=True))


# ---------------------------------------------------------------------------
# Worker: gera o tensor de UM áudio (top-level para ser "picklável")
# ---------------------------------------------------------------------------
def _gerar_um_arquivo(args: tuple) -> dict:
    arquivo, caminho, cfg = args
    try:
        e = cfg["espectrograma"]
        # MESMO pré-processamento do ramo clássico — não reimplementado.
        y, _prop, n_validas = preprocessar_audio(caminho, cfg)
        S_db = gerar_um(y, e["sample_rate"], e)
        # máscara pela MESMA função do ramo clássico, para conferência contra o
        # valor que veio do features.csv (ver `_escrever_conjunto`)
        n_val = frames_validos_espectrograma(n_validas, e, int(S_db.shape[1]))
        return {"arquivo": arquivo, "tensor": S_db, "n_frames_validos": n_val}
    except Exception as ex:
        return {"arquivo": arquivo, "erro": repr(ex)}


# ---------------------------------------------------------------------------
# Escrita de um conjunto: índice primeiro, memmap depois, com retomada
# ---------------------------------------------------------------------------
def _escrever_conjunto(destino: Path, indice: pd.DataFrame, cfg: dict,
                       n_jobs: int, flush_a_cada: int) -> dict:
    """Gera e grava um conjunto inteiro num memmap `(n, altura, largura)`.

    O índice já está em disco e ordenado; esta função só preenche o memmap na
    MESMA ordem. Retomada por `<destino>.progresso.json`: como as linhas são
    escritas em sequência, «escrevi k linhas» é toda a informação necessária —
    as linhas `[0, k)` estão prontas e a geração recomeça em `k`.

    UM ERRO ABORTA O CONJUNTO, de propósito. No CSV de features uma linha
    perdida é só uma linha perdida; aqui a linha é POSICIONAL, e pular um áudio
    deixaria uma fatia de zeros casada com um rótulo real — o pior artefato
    possível, invisível até o treino. Melhor parar em `k`, gravar o motivo e
    deixar a retomada continuar de onde deu.
    """
    e = cfg["espectrograma"]
    altura, largura = e["altura"], largura_esperada(cfg)
    n = len(indice)

    progresso_path = destino.with_suffix(".progresso.json")
    feitas = 0
    if destino.exists() and not progresso_path.exists():
        # Abrir com mode="w+" aqui ZERARIA um lote possivelmente inteiro, em
        # silêncio. Sem o progresso não há como saber quantas linhas valem, e
        # "recomeçar do zero" é uma decisão de horas de CPU: quem a toma é o
        # operador, apagando o arquivo à mão.
        raise ValueError(
            f"{destino.name} existe mas {progresso_path.name} não. Sem o "
            "progresso não se sabe quantas linhas do memmap são válidas, e "
            "recriá-lo apagaria o que já foi gerado. Arquive ou apague o .npy e "
            "o índice deste prefixo para regerar."
        )
    if destino.exists():
        feitas = int(json.loads(progresso_path.read_text(encoding="utf-8"))["n_escritas"])
        mm = np.lib.format.open_memmap(destino, mode="r+")
        if mm.shape != (n, altura, largura):
            raise ValueError(
                f"{destino.name} tem shape {mm.shape}, esperado "
                f"{(n, altura, largura)}. O índice mudou depois da geração — "
                "apague o .npy, o índice e o .progresso.json e regere."
            )
        print(f"  retomando {destino.name}: {feitas}/{n} linhas já escritas.")
    else:
        # open_memmap grava um cabeçalho .npy de verdade, então o arquivo pode
        # ser lido depois com np.load(..., mmap_mode="r") no Dataset do PyTorch.
        mm = np.lib.format.open_memmap(destino, mode="w+", dtype=np.float32,
                                       shape=(n, altura, largura))

    if feitas >= n:
        print(f"  {destino.name}: nada a fazer, {n} linhas já escritas.")
        mm.flush()
        del mm
        return {"n": n, "n_escritas": feitas, "erro": None}

    pendentes = indice.iloc[feitas:]
    tarefas = [(r.arquivo, r.caminho, cfg) for r in pendentes.itertuples()]
    esperado_n_val = pendentes["n_frames_validos"].tolist()
    esperado_arq = pendentes["arquivo"].tolist()

    erro = None
    i = feitas
    with Pool(n_jobs) as pool:
        # `imap`, NUNCA `imap_unordered`: a ordem é o contrato do memmap.
        iterador = pool.imap(_gerar_um_arquivo, tarefas)
        for k, res in enumerate(tqdm(iterador, total=len(tarefas),
                                     desc=f"  {destino.stem}")):
            if "erro" in res:
                erro = f"{res['arquivo']}: {res['erro']}"
                break
            # Três conferências por linha, todas baratas e todas capazes de
            # pegar um desalinhamento antes de ele virar um lote inútil:
            if res["arquivo"] != esperado_arq[k]:
                erro = (f"ordem violada na linha {i}: esperado "
                        f"{esperado_arq[k]}, veio {res['arquivo']}")
                break
            t = res["tensor"]
            if t.shape != (altura, largura) or t.dtype != np.float32:
                erro = (f"{res['arquivo']}: tensor {t.shape} {t.dtype}, "
                        f"esperado {(altura, largura)} float32")
                break
            # PARIDADE DA MÁSCARA: o valor recalculado por frames_validos() tem
            # de bater com o que veio do features.csv congelado. Divergir aqui
            # significa que os dois ramos mascaram grades diferentes — a
            # armadilha de ±1 frame, que de outro modo seria indetectável.
            if res["n_frames_validos"] != esperado_n_val[k]:
                erro = (f"{res['arquivo']}: n_frames_validos recalculado "
                        f"{res['n_frames_validos']} != features.csv "
                        f"{esperado_n_val[k]} — máscara da CNN divergindo do "
                        "ramo clássico.")
                break
            mm[i] = t
            i += 1
            if (i - feitas) % flush_a_cada == 0:
                mm.flush()
                progresso_path.write_text(
                    json.dumps({"n_escritas": i, "n_total": n}), encoding="utf-8")

    mm.flush()
    del mm
    progresso_path.write_text(json.dumps({"n_escritas": i, "n_total": n}),
                              encoding="utf-8")
    if erro:
        raise RuntimeError(
            f"Geração de {destino.name} ABORTADA na linha {i} de {n}: {erro}\n"
            "As linhas [0, %d) estão íntegras em disco; corrija a causa e rode "
            "de novo para retomar daqui." % i
        )
    print(f"  {destino.name}: {i}/{n} linhas ({destino.stat().st_size / 2**30:.2f} GiB)")
    return {"n": n, "n_escritas": i, "erro": None}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def executar(cfg: dict, raiz: Path,
             conjuntos: tuple[str, ...] = ("treino", "validacao", "teste"),
             limite: int | None = None, nome_saida: str | None = None,
             n_jobs: int | None = None, flush_a_cada: int = 500,
             somente_assinatura: bool = False) -> dict:
    """Gera os espectrogramas dos conjuntos pedidos, um memmap por conjunto.

    Args:
        conjuntos: conjuntos do split.csv a gerar. O default são os três que a
            CNN consome; o braço de referência (RF-only) nunca precisa destes
            tensores.
        limite: modo PILOTO — gera apenas uma amostra estratificada por
            (conjunto × classe) desse tamanho TOTAL. Recusa escrever nos nomes
            oficiais, como em `extrair_features.executar`.
        nome_saida: prefixo dos artefatos dentro de data/espectrogramas/. None =
            nomes oficiais (`treino_30k.npy`, ...). O piloto precisa de um
            prefixo ("piloto") para não encostar no lote oficial.
        n_jobs: processos. Default = todos os núcleos.
        somente_assinatura: grava o `.meta.json` da definição vigente e PARA,
            sem gerar tensor nenhum. É como o B4.1 deixa
            `espectrogramas.meta.json` versionado no git antes de o lote do B4.2
            existir: a assinatura descreve a DEFINIÇÃO (parâmetros, escopo
            planejado, hashes das entradas congeladas), e o que só o lote pode
            preencher — os hashes dos índices — fica explicitamente vazio.

    Returns:
        O `.meta.json` gravado.
    """
    n_jobs = n_jobs or cpu_count()
    e = cfg["espectrograma"]

    # ---- coerência do config: 251 é DERIVADO, não literal -------------------
    largura = largura_esperada(cfg)
    if largura != e["largura"]:
        raise ValueError(
            f"espectrograma.largura={e['largura']} mas o pipeline produz "
            f"{largura} frames (= 1 + {int(cfg['audio']['sample_rate'] * cfg['audio']['duracao_segundos'])} "
            f"// {e['hop_length']}). Corrija o config antes de gerar: um lote com "
            "largura diferente da registrada é um lote irreproduzível."
        )
    if e["escala"] != "db":
        raise ValueError(f"espectrograma.escala='{e['escala']}': este gerador "
                         "implementa apenas a escala 'db' (decisão P5).")
    if e["tipo"] != "mel":
        raise ValueError(f"espectrograma.tipo='{e['tipo']}': este gerador "
                         "implementa apenas 'mel' (decisão P4).")

    # ---- piloto não encosta nos nomes oficiais ------------------------------
    if limite is not None and not nome_saida:
        raise ValueError(
            "Modo piloto (limite) não escreve nos nomes oficiais. Passe "
            "nome_saida='piloto' (ou similar)."
        )
    prefixo = f"{nome_saida}_" if nome_saida else ""

    cat = catalogo(cfg, raiz, conjuntos)
    print(f"Escopo: {len(cat)} áudios em {cat['nome_conjunto'].nunique()} conjunto(s) "
          f"-> {dict(cat['nome_conjunto'].value_counts().sort_index())}")

    if limite is not None:
        cat = _amostra_piloto(cat, limite, cfg["semente"])
        print(f"MODO PILOTO: {len(cat)} áudios (estratificado por conjunto × "
              f"classe, semente {cfg['semente']}) -> prefixo '{prefixo}'")

    saida_dir = raiz / "data" / "espectrogramas"
    saida_dir.mkdir(parents=True, exist_ok=True)

    # Ordem de GERAÇÃO (a do catálogo: treino, validação, teste), não alfabética
    # — é a ordem que vai para `escopo` no .meta.json.
    ordem = cat["nome_conjunto"].drop_duplicates().tolist()
    contagem = cat["nome_conjunto"].value_counts()
    escopo = {nome: int(contagem[nome]) for nome in ordem}

    meta_atual = _meta_geracao(cfg, raiz, escopo)
    meta_path = saida_dir / f"{prefixo}espectrogramas.meta.json"

    if somente_assinatura:
        meta_atual["hash_md5_indice_por_conjunto"] = {}
        meta_atual["estado"] = (
            "ASSINATURA SEM LOTE: os parametros, o escopo planejado e os hashes "
            "das entradas congeladas estao fechados; os tensores e os indices "
            "ainda nao foram gerados (e o B4.2 que os gera, preenchendo "
            "hash_md5_indice_por_conjunto)."
        )
        meta_path.write_text(json.dumps(meta_atual, indent=2, ensure_ascii=False),
                             encoding="utf-8")
        print(f"Assinatura (sem lote) gravada em {meta_path}")
        return meta_atual

    # ---- espaço em disco: medir ANTES, não descobrir no meio ----------------
    bytes_previstos = int(len(cat) * e["altura"] * largura * 4)
    livre = shutil.disk_usage(saida_dir).free
    print(f"Disco: previstos {bytes_previstos / 2**30:.2f} GiB, "
          f"livres {livre / 2**30:.1f} GiB")
    if livre < bytes_previstos * 1.05:
        raise RuntimeError(
            f"Espaço insuficiente: {bytes_previstos / 2**30:.2f} GiB previstos, "
            f"{livre / 2**30:.2f} GiB livres. Libere espaço antes de começar — "
            "um memmap truncado no meio não é retomável."
        )

    # ---- índice PRIMEIRO, memmap depois ------------------------------------
    resultados = {}
    for nome in ordem:
        linhas = cat[cat["nome_conjunto"] == nome].reset_index(drop=True)
        npy = saida_dir / f"{prefixo}{nome}.npy"
        idx_path = saida_dir / f"{prefixo}indice_{nome}.csv"

        # A guarda roda assim que EXISTE assinatura em disco, mesmo sem nenhum
        # .npy ainda. É o que torna a assinatura gravada pelo B4.1
        # (--somente-assinatura) operante e não decorativa: se o config mudar
        # entre o B4.1 e o lote do B4.2, o lote se recusa a começar em vez de
        # nascer com outra definição da que foi validada e commitada.
        if meta_path.exists():
            _validar_retomada(meta_path, meta_atual)

        indice = linhas.assign(linha=np.arange(len(linhas)))[COLUNAS_INDICE]
        if idx_path.exists():
            antigo = pd.read_csv(idx_path)
            if (list(antigo.columns) != COLUNAS_INDICE
                    or not antigo.reset_index(drop=True).equals(indice)):
                raise ValueError(
                    f"{idx_path.name} em disco difere do índice que o catálogo "
                    "produz agora. O memmap existente está casado com o índice "
                    "ANTIGO: sobrescrevê-lo desalinharia tensor e rótulo. "
                    "Arquive os artefatos deste prefixo antes de regerar."
                )
        else:
            indice.to_csv(idx_path, index=False)

        # O caminho do .flac fica só no catálogo (é local da máquina); o índice
        # guarda o que o Dataset precisa: arquivo, linha, classe,
        # n_frames_validos. `how="left"` + `validate="one_to_one"`: o merge tem
        # de PRESERVAR a ordem do índice — é ela que define a linha do memmap —
        # e não pode duplicar nem perder linha nenhuma.
        plano = indice.merge(linhas[["arquivo", "caminho"]], on="arquivo",
                             how="left", validate="one_to_one")
        if plano["caminho"].isna().any() or len(plano) != len(indice):
            raise ValueError(
                f"merge do índice de {nome} com os caminhos falhou "
                f"({len(indice)} -> {len(plano)}, "
                f"{int(plano['caminho'].isna().sum())} sem caminho)."
            )
        resultados[nome] = _escrever_conjunto(npy, plano, cfg, n_jobs, flush_a_cada)

    # O escopo e os hashes gravados descrevem o LOTE EM DISCO, não apenas o que
    # esta execução gerou — o B4.2 roda um conjunto por vez.
    escopo_disco, hashes = _lote_em_disco(saida_dir, prefixo)
    meta_atual = _meta_geracao(cfg, raiz, escopo_disco, hashes_indice=hashes)
    # O rotulo de congelamento vale para os artefatos OFICIAIS. Um piloto tem os
    # tres conjuntos tambem (estratificado por conjunto x classe) e receberia o
    # mesmo carimbo, alegando lote definitivo sobre 201 tensores.
    completo = (not prefixo) and all(escopo_disco.get(n) for n in ORDEM_CONJUNTOS)
    meta_atual["estado"] = (
        "LOTE COMPLETO - DEFINICAO CONGELADA (B4.2): os tres conjuntos estao em "
        "disco com seus indices e hashes. A partir daqui o bloco `espectrograma` "
        "do config.yaml entra no mesmo regime de `features`: nao se altera. "
        "Regeracao apenas por erro grave, com decisao registrada - e a guarda "
        "_validar_retomada recusa gerar com outros parametros ate que o "
        ".meta.json, os indices e os .npy sejam arquivados."
        if completo else
        f"LOTE NAO OFICIAL (prefixo '{prefixo}'): "
        + ", ".join(f"{k} ({v})" for k, v in escopo_disco.items())
        + ". Nao congela definicao nenhuma."
        if prefixo else
        "LOTE PARCIAL: gerado(s) " + ", ".join(f"{k} ({v})" for k, v in escopo_disco.items())
        + ". Faltam " + ", ".join(n for n in ORDEM_CONJUNTOS if not escopo_disco.get(n))
        + " - a definicao so se declara congelada com os tres em disco."
    )
    meta_path.write_text(json.dumps(meta_atual, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    print(f"\nAssinatura gravada em {meta_path}")
    total = sum(r["n_escritas"] for r in resultados.values())
    print(f"Concluído: {total} tensores em {len(resultados)} memmap(s).")
    return meta_atual


if __name__ == "__main__":
    import argparse

    from ..utils.config import carregar_config
    from ..utils.seeds import fixar_seeds

    RAIZ = Path(__file__).resolve().parents[2]

    ap = argparse.ArgumentParser(
        description="Gera os espectrogramas log-Mel da CNN (128 x 251, dB, crus).")
    ap.add_argument("--limite", type=int, default=None,
                    help="modo piloto: nº TOTAL de áudios, estratificado por "
                         "conjunto x classe. Exige --nome-saida.")
    ap.add_argument("--nome-saida", default=None,
                    help="prefixo dos artefatos (ex.: 'piloto'). Sem ele, "
                         "escreve nos nomes oficiais.")
    ap.add_argument("--conjuntos", default="treino,validacao,teste",
                    help="conjuntos do split.csv, separados por vírgula.")
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--somente-assinatura", action="store_true",
                    help="grava o .meta.json da definição vigente e para, sem "
                         "gerar tensor nenhum (usado pelo B4.1).")
    args = ap.parse_args()

    cfg = carregar_config(RAIZ)
    fixar_seeds(cfg["semente"])
    executar(cfg, RAIZ,
             conjuntos=tuple(c.strip() for c in args.conjuntos.split(",") if c.strip()),
             limite=args.limite, nome_saida=args.nome_saida, n_jobs=args.n_jobs,
             somente_assinatura=args.somente_assinatura)
