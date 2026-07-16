"""
Pré-processamento de áudio — ASVspoof 2021 LA
==============================================

Este módulo transforma um .flac cru numa forma de onda LIMPA e de TAMANHO FIXO,
pronta tanto para a extração de features (RF/SVM) quanto para a geração de
espectrogramas (CNN).

ORDEM DAS OPERAÇÕES:

    carregar -> normalizar amplitude -> VAD (remove silêncio) -> padronizar duração

Por que essa ordem, e não outra:
  * Normalizar ANTES do VAD faz a conversão para int16 usar toda a faixa dinâmica,
    o que dá decisões de fala/silêncio mais confiáveis em gravações baixas.
  * VAD ANTES de fixar a duração é o ponto central: primeiro removemos o silêncio
    do áudio natural, e SÓ DEPOIS cortamos/preenchemos para o tamanho alvo. Se
    fizéssemos o inverso (fixar duração e depois VAD), acabaríamos com um clipe de
    tamanho variável de novo, e o padding de zeros seria estranhamente picotado.

Saída: vetor float32, mono, 16 kHz, com exatamente `sample_rate * duracao_segundos`
amostras.
"""

from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
import webrtcvad


# ---------------------------------------------------------------------------
# 1) Carregamento + padronização de sample rate e canais
# ---------------------------------------------------------------------------
def carregar_audio(caminho: str, sr: int = 16000) -> np.ndarray:
    """Carrega um áudio como mono float32 em [-1, 1], reamostrando para `sr`.

    Lemos com soundfile DIRETO, e não com librosa.load, por um motivo de
    diagnóstico: quando o soundfile falha, o librosa.load engole a exceção real
    e tenta o fallback via audioread — que no Windows sem ffmpeg só produz um
    NoBackendError() vazio. Com a leitura direta, a exceção verdadeira do
    decodificador (ex.: LibsndfileError) sobe intacta até o erros.csv.

    Padronizações aplicadas:
      - mono: eventuais canais extras são misturados pela média (o ASVspoof é
        mono; isto é só robustez).
      - sr=16000: no ASVspoof já é 16 kHz, então a reamostragem é um passa-direto;
        só reamostramos de fato se o arquivo vier com taxa diferente.
    """
    y, sr_orig = sf.read(caminho, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr_orig != sr:
        y = librosa.resample(y, orig_sr=sr_orig, target_sr=sr)
    return y.astype(np.float32)


# ---------------------------------------------------------------------------
# 2) Normalização de amplitude
# ---------------------------------------------------------------------------
def normalizar_amplitude(y: np.ndarray) -> np.ndarray:
    """Normalização de PICO: divide pelo maior valor absoluto -> faixa [-1, 1].

    Por que normalizar: gravações têm ganhos diferentes (microfone, distância,
    volume). Sem normalizar, o modelo pode aprender o VOLUME como atalho em vez de
    aprender o que interessa (os artefatos de síntese). Normalizar remove essa
    variável espúria e coloca todo mundo na mesma escala.

    Escolha de pico (e não RMS): é simples, determinística e não altera a forma da
    onda (só a escala). RMS/LUFS seria uma alternativa mais "perceptual", mas
    adiciona parâmetros e não traz ganho claro para esta tarefa.
    """
    pico = np.max(np.abs(y))
    if pico < 1e-8:            # áudio praticamente silencioso: evita divisão por zero
        return y
    return y / pico


# ---------------------------------------------------------------------------
# 3) VAD — Voice Activity Detection (remoção de silêncio)
# ---------------------------------------------------------------------------
def aplicar_vad(
    y: np.ndarray,
    sr: int,
    agressividade: int = 2,
    frame_ms: int = 30,
) -> tuple[np.ndarray, float]:
    """Remove trechos de silêncio usando o webrtcvad.

    ARMADILHA que este código trata (e por que o VAD é chato):
      O webrtcvad NÃO aceita o float32 que o librosa devolve. Ele exige:
        - PCM int16 (não float),
        - mono,
        - sr em {8000, 16000, 32000, 48000},
        - frames de exatamente 10, 20 ou 30 ms.
      Então convertemos float [-1,1] -> int16 SÓ para o VAD DECIDIR o que é fala,
      mas remontamos o áudio final a partir do array float ORIGINAL, preservando a
      precisão para o cálculo posterior de MFCC.

    Parâmetros:
      agressividade (0-3): quão agressivo é ao classificar algo como NÃO-fala.
        0 = conservador (mantém mais), 3 = agressivo (corta mais). 2 é um meio-termo
        seguro para começar.

    Retorna:
      (y_sem_silencio, proporcao_de_fala) — a proporção é um diagnóstico útil:
      se cair muito perto de 0, o VAD comeu o áudio (bandeira vermelha).
    """
    if sr not in (8000, 16000, 32000, 48000):
        raise ValueError(f"webrtcvad não suporta sr={sr}. Use 8/16/32/48 kHz.")

    vad = webrtcvad.Vad(agressividade)

    # float [-1,1] -> int16 (apenas para as DECISÕES do VAD)
    y16 = (np.clip(y, -1.0, 1.0) * 32767).astype(np.int16)

    n_frame = int(sr * frame_ms / 1000)     # ex.: 16000 * 0.03 = 480 amostras/frame
    n_frames = len(y16) // n_frame           # ignora o resto (< um frame) no final

    manter = []                              # índices (início, fim) dos frames de fala
    for i in range(n_frames):
        ini, fim = i * n_frame, (i + 1) * n_frame
        if vad.is_speech(y16[ini:fim].tobytes(), sr):
            manter.append((ini, fim))

    if not manter:
        # VAD zerou tudo (áudio curtíssimo ou muito ruidoso): devolve o original
        # para não quebrar a etapa seguinte. proporcao=0.0 sinaliza o caso.
        return y, 0.0

    # Remonta a partir do array FLOAT original (precisão preservada p/ MFCC)
    y_vad = np.concatenate([y[ini:fim] for ini, fim in manter])
    return y_vad, len(y_vad) / len(y)


# ---------------------------------------------------------------------------
# 4) Padronização de duração (tamanho fixo)
# ---------------------------------------------------------------------------
def padronizar_duracao(y: np.ndarray, sr: int, dur_segundos: float) -> np.ndarray:
    """Força o áudio a ter EXATAMENTE `dur_segundos` (corta ou preenche com zeros).

    Por que tamanho fixo é obrigatório:
      - Para a CNN: o espectrograma vira uma "imagem" e a rede exige dimensão fixa.
      - Para RF/SVM: agregamos as features no tempo (média/desvio), então o
        comprimento some no vetor final — mas manter a mesma janela para todos deixa
        a comparação justa.

    Removemos silêncio no VAD e aqui, se o áudio ficou
    curto, ADICIONAMOS zeros de volta. Não é contraditório: esses zeros são padding
    determinístico só para igualar o formato do tensor, não "silêncio natural" — e
    são idênticos para todas as amostras, logo não informam nada ao modelo.
    """
    alvo = int(sr * dur_segundos)
    if len(y) >= alvo:
        return y[:alvo]                                  # corta o excesso (pega o início)
    return np.pad(y, (0, alvo - len(y)), mode="constant")  # preenche com zeros no fim


# ---------------------------------------------------------------------------
# 5) Orquestrador: o pipeline completo por áudio
# ---------------------------------------------------------------------------
def preprocessar_audio(caminho: str, cfg: dict) -> tuple[np.ndarray, float]:
    """Aplica a cadeia inteira a um arquivo, lendo os parâmetros do config.yaml.

    Retorna (forma_de_onda_final, proporcao_de_fala).
    """
    sr = cfg["audio"]["sample_rate"]
    dur = cfg["audio"]["duracao_segundos"]
    usar_vad = cfg["audio"]["vad"]

    y = carregar_audio(caminho, sr)
    y = normalizar_amplitude(y)

    proporcao_fala = 1.0
    if usar_vad:
        y, proporcao_fala = aplicar_vad(y, sr)

    y = padronizar_duracao(y, sr, dur)
    return y, proporcao_fala


# ---------------------------------------------------------------------------
# TESTE INICIAL: roda a cadeia em alguns arquivos
# e imprime o que aconteceu em cada estágio.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import yaml
    import pandas as pd

    RAIZ = Path(__file__).resolve().parents[2]
    cfg = yaml.safe_load(open(RAIZ / "config" / "config.yaml", encoding="utf-8"))

    labels_csv = RAIZ / "data" / "processed" / "labels.csv"
    if not labels_csv.exists():
        print("labels.csv não encontrado. Rode antes o notebook 02_leitura_dados.ipynb.")
        sys.exit(0)

    df = pd.read_csv(labels_csv)
    sr = cfg["audio"]["sample_rate"]
    dur = cfg["audio"]["duracao_segundos"]
    alvo = int(sr * dur)

    print("=" * 68)
    print(f"TESTE DE PRÉ-PROCESSAMENTO  |  alvo = {dur}s = {alvo} amostras @ {sr} Hz")
    print("=" * 68)

    testados = 0
    for _, linha in df.iterrows():
        caminho = linha["caminho"]
        if not Path(caminho).exists():
            continue

        y_bruto = carregar_audio(caminho, sr)
        dur_bruta = len(y_bruto) / sr

        y_final, prop = preprocessar_audio(caminho, cfg)

        print(f"\n{Path(caminho).name}  [{linha['label']}]")
        print(f"  bruto     : {len(y_bruto):>7} amostras  ({dur_bruta:5.2f}s)")
        print(f"  pós-VAD   : {prop*100:5.1f}% de fala mantida")
        print(f"  final     : {len(y_final):>7} amostras  "
              f"({'OK' if len(y_final) == alvo else 'ERRO DE TAMANHO'})")

        testados += 1
        if testados >= 5:
            break

    if testados == 0:
        print("\nNenhum .flac encontrado no disco. Baixe/organize o dataset em data/raw/.")
    else:
        print(f"\n{testados} áudios testados. Se todos deram 'OK' no tamanho final, "
              "Pré-processamento OK.")