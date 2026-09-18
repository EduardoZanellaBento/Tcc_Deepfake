"""
B4.4 — CNN baseline: primeira rede treinada de ponta a ponta
=============================================================

O OBJETIVO DESTE MARCO É DESTRAVAR, NÃO OTIMIZAR. «Primeira CNN treinada de ponta a
ponta, com curvas de treino/validação interna e primeiras métricas. Não precisa ser a
final; precisa funcionar corretamente.» A definição final da arquitetura é o B4.5.

O QUE ESTE SCRIPT NÃO FAZ, DE PROPÓSITO:
  - não toca na validação externa (22.226) nem no teste (22.227). O limiar do
    protocolo nasce no B4.7; aqui ele é PROVISÓRIO e sai marcado como tal;
  - não para de verdade no early stopping — apenas REGISTRA a melhor época pelo
    f1_macro nos 3k. O critério de parada fecha no B4.5;
  - não usa AMP/autocast. Ganho desprezível numa rede pequena, e introduz
    não-determinismo numérico exatamente quando se está tentando provar que o
    pipeline está correto.

A RÉGUA É A MESMA DOS OUTROS DOIS: `selecionar_limiar`, `calcular_eer` e `avaliar`
vêm de src/models/avaliacao.py, sem cópia local. Num trabalho cuja pergunta central é
COMPARAR modelos, uma divergência silenciosa entre cópias invalidaria a comparação.

Uso:
    python -m src.models.treinar_cnn
    python -m src.models.treinar_cnn --epocas 15 --canais 16,32,64   # modo "se atrasar"
"""

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, roc_auc_score

from .avaliacao import avaliar, plotar_matriz_confusao, selecionar_limiar
from .cnn import CnnDeepfake, pesos_balanced
from .tempo import ambiente
from ..utils.seeds import fixar_seeds_torch

RAIZ = Path(__file__).resolve().parents[2]
NOME = "cnn_baseline"


# =============================================================================
# Limitacoes registradas do marco
# =============================================================================
# Regra de escopo (17/09): achado que NAO impede a validade do experimento
# principal vira limitacao registrada, nao investigacao. As duas abaixo foram
# levantadas na revisao do B4.4, reprovam o teste "isso impede a validade do
# experimento principal?" com um NAO, e vao para o texto no B6.1. Ficam no
# codigo — e nao so no JSON — para que qualquer reexecucao do marco as carregue.
LIMITACOES_REGISTRADAS = [
    {
        "id": "L1-batchnorm-ve-o-padding",
        "titulo": "O masked pooling protege a AGREGACAO, nao a NORMALIZACAO",
        "descricao": (
            "O BatchNorm2d de cada bloco calcula media e variancia por canal sobre o "
            "LOTE inteiro (N, F, T), com o padding incluido, e essas estatisticas "
            "normalizam tambem as posicoes validas. O masked global pooling garante "
            "que o padding nao entra na AGREGACAO final; ele nao impede que o padding "
            "participe da NORMALIZACAO intermediaria."
        ),
        "por_que_nao_invalida": (
            "A estatistica do BatchNorm e do LOTE, nao do exemplo: desloca e escala "
            "todos os exemplos do lote da mesma forma e, portanto, NAO cria um canal "
            "por exemplo que codifique onde o audio termina — que e exatamente a "
            "variavel espuria (prop_fala) que o Bloco 1 tirou do X do ramo classico. "
            "Alem disso o padding e um plato constante, produzido pelo mesmo pipeline "
            "para bonafide e para spoof, logo nao e informativo de classe. O unico "
            "vazamento POR EXEMPLO que de fato existe e o da borda da convolucao, "
            "medido em ate 27 frames originais na checagem 3 de "
            "checagem_mascara_cnn.json, e ja esta documentado."
        ),
        "consequencia_para_a_checagem_da_mascara": (
            "A checagem 4 roda com o modelo em eval(), onde o BatchNorm usa running "
            "stats fixas — e por isso que ali a diferenca COM mascara e ZERO EXATO. "
            "Em train() a mesma invariancia bit a bit nao vale. A afirmacao precisa e: "
            "sob masked pooling e com as estatisticas de normalizacao fixas, o "
            "conteudo do padding fora do alcance da convolucao tem influencia "
            "exatamente nula na saida."
        ),
        "teste_da_regra_de_escopo": "NAO impede a validade do experimento principal",
        "destino": "limitacao no texto (B6.1)",
        "trabalho_futuro": (
            "BatchNorm mascarado (estatisticas so sobre frames validos), ou "
            "LayerNorm/GroupNorm no lugar do BatchNorm"
        ),
        "custo_para_corrigir_agora": (
            "re-treino de todo o Bloco 4 (B4.4 a B4.7); nao cabe no cronograma"
        ),
    },
    {
        "id": "L2-semantica-de-teto-na-fronteira",
        "titulo": "A reducao da mascara usa semantica de TETO, e a posicao de fronteira e parcialmente padding",
        "descricao": (
            "A mascara e reduzida por F.max_pool1d, entao uma posicao reduzida conta "
            "como VALIDA se QUALQUER frame original dela era valido. Depois dos 4 "
            "blocos cada posicao reduzida agrega 16 frames originais "
            "(251 -> 125 -> 62 -> 31 -> 15), logo a posicao de FRONTEIRA pode ser "
            "majoritariamente padding e ainda assim entrar no masked pooling com "
            "peso 1."
        ),
        "magnitude": (
            "No maximo UMA posicao entre as validas, e o erro nao cresce com a "
            "profundidade: 1 de 15 (~7%) num exemplo sem padding e 1 de 7 (~14%) num "
            "exemplo com 100 frames validos, que e o caso usado nas checagens."
        ),
        "por_que_foi_escolhida": (
            "A alternativa e a semantica de PISO (a posicao reduzida so conta se "
            "TODOS os frames originais dela eram validos), que descartaria a fronteira "
            "do audio — informacao real, nao padding — e encurtaria mais o sinal "
            "justamente nos audios curtos, que ja sao os que tem menos frames "
            "validos. O teto e coerente com o ceil de n_frames_validos do Bloco 1."
        ),
        "teste_da_regra_de_escopo": "NAO impede a validade do experimento principal",
        "destino": (
            "convencao declarada no texto (B6.1), junto com a resposta de banca sobre "
            "a mascara temporal"
        ),
        "custo_para_corrigir_agora": (
            "trocar a semantica exigiria re-treinar e re-medir o Bloco 4; nao cabe no "
            "cronograma"
        ),
    },
]



# =============================================================================
# Dataset
# =============================================================================
class EspectrogramaDataset(torch.utils.data.Dataset):
    """Le do memmap, normaliza em tempo de carga, devolve (x, mascara, y).

    A normalizacao vem de normalizacao_cnn.json (B4.3) e e aplicada aos 251
    frames, INCLUSIVE ao padding. De proposito: o padding e um plato constante
    que, normalizado, continua constante — e nao entra em conta nenhuma, porque
    a agregacao e mascarada. Normalizar so a parte valida criaria uma
    descontinuidade artificial na fronteira, que a convolucao leria como borda.
    """

    def __init__(self, memmap_path, indice_df, media, desvio):
        self.mm = np.load(memmap_path, mmap_mode="r")      # (N, 128, 251)
        self.linhas = indice_df["linha"].to_numpy()
        self.y = indice_df["classe_binaria"].to_numpy().astype(np.int64)
        self.n_valid = indice_df["n_frames_validos"].to_numpy().astype(np.int64)
        self.media = media[:, None].astype(np.float32)     # (128, 1)
        self.desvio = desvio[:, None].astype(np.float32)

    def __len__(self):
        return len(self.linhas)

    def __getitem__(self, i):
        X = np.asarray(self.mm[self.linhas[i]], dtype=np.float32)
        X = (X - self.media) / self.desvio
        m = np.zeros(X.shape[1], dtype=np.float32)
        m[: self.n_valid[i]] = 1.0
        return torch.from_numpy(X)[None], torch.from_numpy(m), int(self.y[i])


# =============================================================================
# Preparo dos dados
# =============================================================================
def preparar_indices(raiz: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Junta o índice do memmap (B4.2) com o split interno 27k/3k (B4.3).

    A DISJUNÇÃO É VERIFICADA AQUI, não assumida: «treino perfeito, 3k terrível desde
    a época 2» e «métricas boas demais» são o mesmo bug visto de dois ângulos, e os
    dois se previnem com um `assert` de interseção vazia que custa microssegundos.
    """
    indice = pd.read_csv(raiz / "data" / "espectrogramas" / "indice_treino_30k.csv")
    split = pd.read_csv(raiz / "data" / "processed" / "split_interno_cnn.csv")

    df = indice.merge(split, on="arquivo", how="inner", validate="one_to_one")
    assert len(df) == len(indice) == len(split), (
        f"merge perdeu linhas: indice={len(indice)}, split={len(split)}, merge={len(df)}")

    tr = df[df["particao_interna"] == "treino_interno"].reset_index(drop=True)
    es = df[df["particao_interna"] == "early_stopping"].reset_index(drop=True)

    # Disjunção — a guarda contra o vazamento e contra o vazamento AO CONTRÁRIO.
    inter = set(tr["arquivo"]) & set(es["arquivo"])
    assert not inter, f"treino e early stopping se cruzam em {len(inter)} arquivos"
    assert len(tr) + len(es) == len(df), "particao_interna tem valor inesperado"
    # Linhas do memmap também têm de ser disjuntas (o que de fato é lido).
    assert not (set(tr["linha"]) & set(es["linha"])), "linhas do memmap se cruzam"

    resumo = {
        "n_treino_interno": int(len(tr)),
        "n_early_stopping": int(len(es)),
        "bonafide_treino": int((tr["classe_binaria"] == 0).sum()),
        "spoof_treino": int((tr["classe_binaria"] == 1).sum()),
        "bonafide_early_stopping": int((es["classe_binaria"] == 0).sum()),
        "spoof_early_stopping": int((es["classe_binaria"] == 1).sum()),
        "intersecao_arquivos": 0,
        "intersecao_linhas_memmap": 0,
    }
    return tr, es, resumo


def carregar_normalizacao(raiz: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    caminho = raiz / "data" / "espectrogramas" / "normalizacao_cnn.json"
    with open(caminho, encoding="utf-8") as f:
        norm = json.load(f)
    media = np.array(norm["media_por_mel"], dtype=np.float64)
    desvio = np.array(norm["desvio_por_mel"], dtype=np.float64)
    assert media.shape == desvio.shape == (128,), "normalizacao nao tem 128 faixas"
    # «loss nao cai nada» / «NaN na 1a epoca» = desvio zero. Cobrado aqui, uma vez.
    assert float(desvio.min()) > 0, "desvio por faixa Mel com zero -> divisao por zero"
    return media, desvio, norm


# =============================================================================
# Laços de treino e avaliação
# =============================================================================
def _uma_epoca(modelo, loader, criterio, otimizador, dispositivo) -> float:
    modelo.train()
    soma, n = 0.0, 0
    for x, m, y in loader:
        x, m, y = x.to(dispositivo), m.to(dispositivo), y.to(dispositivo)
        otimizador.zero_grad(set_to_none=True)
        logits = modelo(x, m)
        perda = criterio(logits, y)
        perda.backward()
        otimizador.step()
        soma += float(perda.detach()) * len(y)   # detach: `perda` ainda tem grafo
        n += len(y)
    return soma / n


@torch.no_grad()
def _inferir(modelo, loader, criterio, dispositivo) -> tuple[np.ndarray, np.ndarray, float]:
    """Devolve (y_true, scores, loss). score = P(spoof) — convenção do projeto."""
    modelo.eval()
    ys, scores, soma, n = [], [], 0.0, 0
    for x, m, y in loader:
        x, m, y = x.to(dispositivo), m.to(dispositivo), y.to(dispositivo)
        logits = modelo(x, m)
        soma += float(criterio(logits, y)) * len(y)
        n += len(y)
        # softmax[:,1] e logits[:,1]-logits[:,0] sao monotonicamente equivalentes,
        # logo o limiar selecionado e o MESMO ponto de corte nos dois. O softmax
        # vence por ser legivel em [0,1] e comparavel ao predict_proba do RF.
        scores.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        ys.append(y.cpu().numpy())
    return np.concatenate(ys), np.concatenate(scores), soma / n


def _plotar_curva(historico: list, destino: Path) -> None:
    """Dois painéis: loss (treino × 3k) e f1_macro/EER nos 3k."""
    ep = [h["epoca"] for h in historico]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.plot(ep, [h["loss_treino"] for h in historico], "-o", ms=3, label="treino (27k)")
    ax1.plot(ep, [h["loss_early_stopping"] for h in historico], "-o", ms=3,
             label="early stopping (3k)")
    ax1.set_xlabel("época")
    ax1.set_ylabel("loss ponderada")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(ep, [h["f1_macro_early_stopping"] for h in historico], "-o", ms=3,
             color="tab:green", label="f1_macro (3k)")
    ax2.plot(ep, [h["eer_early_stopping"] for h in historico], "-o", ms=3,
             color="tab:red", label="EER (3k)")
    melhor = max(historico, key=lambda h: h["f1_macro_early_stopping"])
    ax2.axvline(melhor["epoca"], ls="--", c="gray", lw=1)
    ax2.annotate(f"melhor época: {melhor['epoca']}",
                 xy=(melhor["epoca"], melhor["f1_macro_early_stopping"]),
                 xytext=(4, -12), textcoords="offset points", fontsize=8, color="gray")
    # 0,47 e o colapso na majoritaria — a linha que separa "aprendeu" de "desistiu".
    ax2.axhline(0.47, ls=":", c="tab:orange", lw=1)
    ax2.annotate("colapso na majoritária (≈0,47)", xy=(ep[0], 0.47),
                 xytext=(2, 4), textcoords="offset points", fontsize=7,
                 color="tab:orange")
    ax2.set_xlabel("época")
    ax2.set_ylabel("métrica nos 3k")
    ax2.set_title("f1_macro e EER no early stopping interno")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle("CNN baseline (B4.4) — curva de treino", fontsize=11)
    fig.tight_layout()
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, dpi=150)
    plt.close(fig)
    print(f"Figura salva em {destino}")


# =============================================================================
# Execução
# =============================================================================
def treinar(cfg: dict, raiz: Path, epocas: int = 30, batch: int = 128,
            lr: float = 1e-3, canais=(32, 64, 128, 128), p_drop: float = 0.3,
            estrito: bool = True) -> dict:
    # fixar_seeds_torch define CUBLAS_WORKSPACE_CONFIG, que so tem efeito se
    # nenhum contexto CUDA foi criado ainda. `import torch` sozinho nao cria
    # contexto; `torch.zeros(1, device="cuda")` cria. Por isso vem ANTES de tudo.
    semente = fixar_seeds_torch(cfg["semente"], estrito=estrito)
    determinismo_estrito = estrito
    limitacao_determinismo = None

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {dispositivo}")

    # SONDAGEM DO DETERMINISMO ESTRITO — precisa ser aqui, e nao num try/except em
    # volta de fixar_seeds_torch: `use_deterministic_algorithms(True)` nao falha ao
    # ser CHAMADO, falha quando uma operacao sem versao deterministica e EXECUTADA.
    # Envolver so a chamada da semente daria um fallback decorativo, que nunca
    # dispararia e deixaria o treino de 30 epocas morrer no meio do backward.
    # Uma passada fwd+bwd num lote falso custa milissegundos e resolve.
    if estrito:
        try:
            _m = CnnDeepfake(n_mels=128, canais=canais, p_drop=p_drop).to(dispositivo)
            _x = torch.zeros(2, 1, 128, 251, device=dispositivo)
            _mask = torch.ones(2, 251, device=dispositivo)
            nn.CrossEntropyLoss()(_m(_x, _mask),
                                  torch.zeros(2, dtype=torch.long,
                                              device=dispositivo)).backward()
            if dispositivo.type == "cuda":
                torch.cuda.synchronize()
            del _m, _x, _mask
        except RuntimeError as e:
            limitacao_determinismo = (
                f"use_deterministic_algorithms(True) falhou numa operacao do "
                f"forward/backward: {e}. Reexecutado com estrito=False e REGISTRADO "
                "como limitacao — bem melhor do que silenciar o problema (docstring "
                "de fixar_seeds_torch).")
            print(f"AVISO: {limitacao_determinismo}")
            determinismo_estrito = False
        # Re-fixa a semente DEPOIS da sondagem, em qualquer caso: a passada falsa
        # consumiu estado do gerador, e sem isto os pesos iniciais dependeriam de
        # a sondagem ter rodado ou nao.
        semente = fixar_seeds_torch(cfg["semente"], estrito=determinismo_estrito)

    # ---- Dados ---------------------------------------------------------------
    tr_df, es_df, resumo_split = preparar_indices(raiz)
    media, desvio, norm = carregar_normalizacao(raiz)
    print(f"Treino interno: {resumo_split['n_treino_interno']} "
          f"({resumo_split['bonafide_treino']} bonafide / "
          f"{resumo_split['spoof_treino']} spoof)")
    print(f"Early stopping: {resumo_split['n_early_stopping']} "
          f"({resumo_split['bonafide_early_stopping']} bonafide / "
          f"{resumo_split['spoof_early_stopping']} spoof)")

    memmap = raiz / "data" / "espectrogramas" / "treino_30k.npy"
    ds_tr = EspectrogramaDataset(memmap, tr_df, media, desvio)
    ds_es = EspectrogramaDataset(memmap, es_df, media, desvio)

    # «acuracia ~50% com loss caindo» = rotulos desalinhados do memmap. Antes de
    # mexer na rede, verifica-se ISTO: o que o Dataset entrega tem de bater com o
    # indice, e o tensor tem de ser finito («loss nao cai» / «NaN na 1a epoca»).
    x0, m0, y0 = ds_tr[0]
    assert torch.isfinite(x0).all(), "espectrograma normalizado com NaN/inf"
    assert x0.shape == (1, 128, 251), f"formato inesperado: {tuple(x0.shape)}"
    assert int(y0) == int(tr_df.loc[0, "classe_binaria"]), "rotulo desalinhado do memmap"
    assert float(m0.sum()) == float(tr_df.loc[0, "n_frames_validos"]), "mascara desalinhada"

    # Windows + DataLoader: num_workers=0. O np.load(mmap_mode="r") ja e preguicoso
    # e o gargalo aqui nao e I/O; subir workers exigiria `generator=` e
    # `worker_init_fn` (fixar_seeds_torch NAO cobre esse caso) e o `spawn` do
    # Windows, sem ganho que justifique o risco de reprodutibilidade neste marco.
    gerador = torch.Generator().manual_seed(semente)
    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=batch, shuffle=True,
                                        num_workers=0, generator=gerador)
    dl_es = torch.utils.data.DataLoader(ds_es, batch_size=batch, shuffle=False,
                                        num_workers=0)

    # ---- Modelo, loss (P7) e otimizador -------------------------------------
    modelo = CnnDeepfake(n_mels=128, canais=canais, p_drop=p_drop).to(dispositivo)
    n_por_classe = [resumo_split["bonafide_treino"], resumo_split["spoof_treino"]]
    pesos = pesos_balanced(n_por_classe, dispositivo)
    criterio = nn.CrossEntropyLoss(weight=pesos)
    otimizador = torch.optim.Adam(modelo.parameters(), lr=lr)
    print(f"Pesos da loss: bonafide {float(pesos[0]):.4f} x spoof {float(pesos[1]):.4f} "
          f"(razao {float(pesos[0] / pesos[1]):.2f}:1 a favor do bonafide)")
    print(f"Parametros: {modelo.descricao()['n_parametros']:,}")

    # ---- Laço ---------------------------------------------------------------
    historico = []
    melhor = {"f1": -1.0, "epoca": -1, "estado": None}
    for epoca in range(1, epocas + 1):
        t0 = time.perf_counter()
        loss_tr = _uma_epoca(modelo, dl_tr, criterio, otimizador, dispositivo)
        y_es, sc_es, loss_es = _inferir(modelo, dl_es, criterio, dispositivo)

        # Métrica da época: limiar re-selecionado nos 3k a cada época. É a leitura
        # honesta do que a rede sabe naquele ponto — um limiar fixo em 0,5 mediria
        # calibração, não separabilidade, e a curva ficaria ilegível.
        sel_ep = selecionar_limiar(y_es, sc_es, criterio="f1_macro",
                                   conjunto="early_stopping_interno")
        m_ep = avaliar(y_es, sc_es, NOME, limiar=sel_ep["limiar"])
        dt = time.perf_counter() - t0

        historico.append({
            "epoca": epoca,
            "loss_treino": loss_tr,
            "loss_early_stopping": loss_es,
            "f1_macro_early_stopping": m_ep["f1_macro"],
            "eer_early_stopping": m_ep["eer"],
            "recall_bonafide_early_stopping": m_ep["recall_bonafide"],
            "recall_spoof_early_stopping": m_ep["recall_spoof"],
            "limiar_epoca": sel_ep["limiar"],
            "tempo_s": round(dt, 2),
        })
        print(f"epoca {epoca:>3}/{epocas} | loss_tr {loss_tr:.4f} | loss_3k {loss_es:.4f} "
              f"| f1_macro {m_ep['f1_macro']:.4f} | EER {m_ep['eer']:.4f} | {dt:.1f}s")

        if m_ep["f1_macro"] > melhor["f1"]:
            # APENAS REGISTRAR a melhor época — nao parar de verdade. O criterio
            # de parada fecha no B4.5. O estado e guardado para que o .pt salvo
            # seja o da melhor epoca, nao o da ultima.
            melhor = {
                "f1": m_ep["f1_macro"],
                "epoca": epoca,
                "estado": {k: v.detach().cpu().clone()
                           for k, v in modelo.state_dict().items()},
            }

    print(f"\nMelhor epoca (registrada, sem parada): {melhor['epoca']} "
          f"-> f1_macro {melhor['f1']:.4f}")

    # ---- Métricas finais: os pesos da MELHOR época --------------------------
    modelo.load_state_dict(melhor["estado"])
    y_es, sc_es, loss_es = _inferir(modelo, dl_es, criterio, dispositivo)

    # NESTE MARCO o limiar sai dos 3k de early stopping, NAO da validacao externa.
    # Ele e PROVISORIO e vai para o JSON com conjunto="early_stopping_interno" —
    # nunca "validacao". A guarda de carregar_modelo_ajustado recusa limiar que nao
    # venha de "validacao", e e assim que tem de ser: o limiar oficial nasce no B4.7.
    sel = selecionar_limiar(y_es, sc_es, criterio="f1_macro",
                            conjunto="early_stopping_interno")
    m = avaliar(y_es, sc_es, NOME, limiar=sel["limiar"])
    m["selecao_limiar"] = sel
    m["nota_escala_score"] = (
        "score = softmax(logits)[:, 1] = P(spoof), em [0,1] — comparavel ao "
        "predict_proba do RF e de escala DIFERENTE do decision_function do SVM. "
        "softmax[:,1] e logits[:,1]-logits[:,0] sao monotonicamente equivalentes, "
        "logo o limiar selecionado e o MESMO ponto de corte nos dois; o softmax foi "
        "escolhido por legibilidade. A regra do protocolo e agnostica de escala.")
    m["nota_limiar_provisorio"] = (
        "LIMIAR PROVISORIO. Selecionado nos 3k de early stopping interno, nao na "
        "validacao externa — por isso conjunto='early_stopping_interno'. Serve para "
        "ler as primeiras metricas do B4.4 e NAO e o limiar do protocolo, que nasce "
        "na validacao externa no B4.7.")
    m["marco"] = "B4.4 — CNN baseline (nao e a arquitetura final; o B4.5 a define)"
    m["braco"] = "principal"
    m["conjunto_avaliado"] = "early_stopping_interno (3.000 da subamostra de 30k)"
    m["nota_comparabilidade"] = (
        "ESTE f1_macro NAO E COMPARAVEL aos 0,7225 do RF e 0,7987 do SVM. Aqueles "
        "saem da VALIDACAO EXTERNA (22.226); este sai dos 3k internos, e e otimista "
        "por TRES motivos independentes, todos de construcao: (1) CONJUNTO — os 3k "
        "vem da mesma subamostra estratificada de 30k que gerou o treino, enquanto a "
        "validacao externa e o conjunto completo; (2) LIMIAR — foi selecionado nos "
        "MESMOS 3k em que a metrica e reportada, e nao num conjunto a parte; "
        "(3) EPOCA — a melhor epoca foi escolhida pelo f1_macro nos MESMOS 3k, entao "
        "o valor publicado e um MAXIMO sobre 30 epocas naquele conjunto. Com apenas "
        "300 bonafide, a variancia da estimativa ainda e grande por cima disso. "
        "Os numeros comparaveis saem no B4.7, na validacao externa. Nada aqui "
        "autoriza a frase «a CNN superou o SVM».")

    f1s = [h["f1_macro_early_stopping"] for h in historico]
    losses_es = [h["loss_early_stopping"] for h in historico]
    m["diagnostico_curva"] = {
        "melhor_epoca_e_a_ultima": bool(melhor["epoca"] == epocas),
        "f1_macro_primeira_epoca": f1s[0],
        "f1_macro_ultima_epoca": f1s[-1],
        "f1_macro_media_ultimas_5": float(np.mean(f1s[-5:])),
        "loss_early_stopping_min": float(min(losses_es)),
        "loss_early_stopping_max": float(max(losses_es)),
        "loss_early_stopping_ultima": losses_es[-1],
        "loss_treino_ultima": historico[-1]["loss_treino"],
        "leitura": (
            "Se `melhor_epoca_e_a_ultima` for True, a rede NAO PAROU de treinar: "
            "acabou o orcamento de epocas. E a resposta honesta a pergunta de banca "
            "«por que a CNN parou de treinar onde parou?» — nao parou, e o B4.5 "
            "precisa decidir o criterio de parada sabendo disso. Separadamente: a "
            "loss nos 3k e MUITO mais ruidosa que a de treino e chega a dar picos de "
            "uma ordem de grandeza, enquanto f1_macro e EER continuam melhorando. "
            "Isso NAO e perda de desempenho — e perda de CALIBRACAO: com peso 5,0 na "
            "bonafide, poucos exemplos minoritarios classificados com alta confianca "
            "no lado errado dominam a CrossEntropy, mas nao mudam o ORDENAMENTO dos "
            "scores, que e o que EER e AUC medem. Por isso o criterio de parada deste "
            "projeto e o f1_macro, nao a loss."),
    }

    m["limitacoes_registradas"] = LIMITACOES_REGISTRADAS

    m["arquitetura"] = modelo.descricao()
    m["hiperparametros"] = {
        "epocas": epocas,
        "batch": batch,
        "otimizador": "Adam",
        "lr": lr,
        "scheduler": None,
        "p_dropout": p_drop,
        "canais": list(canais),
        "amp_autocast": False,
        "nota_amp": ("AMP nao usado de proposito: ganho desprezivel numa rede pequena "
                     "e introduz nao-determinismo numerico exatamente quando se esta "
                     "tentando provar que o pipeline esta correto"),
    }
    m["loss"] = {
        "funcao": "CrossEntropyLoss",
        "n_saidas": 2,
        "formula_pesos": "w_c = n / (k * n_c)  (mesma logica do class_weight='balanced')",
        "contagens_usadas": {"bonafide": n_por_classe[0], "spoof": n_por_classe[1]},
        "peso_bonafide": round(float(pesos[0]), 4),
        "peso_spoof": round(float(pesos[1]), 4),
        "razao": round(float(pesos[0] / pesos[1]), 4),
        "nota": ("pesos DERIVADOS das contagens reais do split interno, nunca "
                 "hard-coded: se o split mudar, os pesos acompanham. NAO "
                 "BCEWithLogitsLoss com pos_weight=9 — spoof e a MAJORITARIA, e "
                 "isso inverteria a intencao aprovada (P7)."),
        "reamostragem": ("nenhuma — sem undersampling e sem oversampling; os 30 mil "
                         "permanecem intactos (P7)"),
    }
    m["early_stopping"] = {
        "melhor_epoca": melhor["epoca"],
        "f1_macro_na_melhor_epoca": melhor["f1"],
        "criterio": "f1_macro nos 3k",
        "parou_de_verdade": False,
        "nota": ("a melhor epoca e APENAS REGISTRADA neste marco; o criterio de "
                 "parada fecha no B4.5. Os pesos salvos em models/cnn_baseline.pt "
                 "sao os da melhor epoca."),
    }
    m["split_interno"] = resumo_split
    m["normalizacao"] = {
        "fonte": "data/espectrogramas/normalizacao_cnn.json (B4.3)",
        "eixo": norm["eixo"],
        "aplicada_em": ("tempo de carga, no Dataset, sobre os 251 frames INCLUSIVE o "
                        "padding — o padding e plato constante que, normalizado, "
                        "continua constante, e nao entra em conta nenhuma porque a "
                        "agregacao e mascarada. Normalizar so a parte valida criaria "
                        "uma descontinuidade artificial na fronteira, que a "
                        "convolucao leria como borda"),
        "faixas_com_desvio_clampado": norm["faixas_com_desvio_clampado"],
    }
    m["semente"] = semente
    m["determinismo"] = {
        "fixar_seeds_torch_chamado_antes_de_qualquer_cuda": True,
        "estrito": determinismo_estrito,
        "limitacao": limitacao_determinismo,
        "num_workers": 0,
        "nota_pooling": ("o masked global pooling e implementado com sum/div "
                         "explicitos e NAO usa adaptive_avg_pool2d, cujo backward em "
                         "CUDA e nao-deterministico — a arquitetura aprovada e, por "
                         "acidente feliz, mais amigavel ao determinismo estrito do "
                         "que a alternativa obvia"),
    }
    m["loss_final_early_stopping"] = loss_es

    cm = confusion_matrix(y_es, (sc_es >= sel["limiar"]).astype(int), labels=[0, 1])
    m["matriz_confusao"] = cm.tolist()
    m["roc_auc_early_stopping"] = round(float(roc_auc_score(y_es, sc_es)), 4)

    m["ambiente"] = ambiente(n_jobs_inferencia=1)
    m["ambiente"]["torch"] = torch.__version__
    m["ambiente"]["cuda_disponivel"] = bool(torch.cuda.is_available())
    m["ambiente"]["cuda_versao"] = torch.version.cuda
    m["ambiente"]["gpu"] = (torch.cuda.get_device_name(0)
                            if torch.cuda.is_available() else None)
    m["ambiente"]["compute_capability"] = (list(torch.cuda.get_device_capability(0))
                                           if torch.cuda.is_available() else None)
    m["ambiente"]["dispositivo_treino"] = str(dispositivo)

    m["tempo_treino_s"] = round(sum(h["tempo_s"] for h in historico), 2)
    m["tempo_por_epoca_s_mediana"] = round(
        float(np.median([h["tempo_s"] for h in historico])), 2)

    dir_esp = raiz / "data" / "espectrogramas"
    m["hash_md5_split_interno_csv"] = hashlib.md5(
        (raiz / "data" / "processed" / "split_interno_cnn.csv").read_bytes()).hexdigest()
    m["hash_md5_indice_treino_csv"] = hashlib.md5(
        (dir_esp / "indice_treino_30k.csv").read_bytes()).hexdigest()
    m["hash_md5_normalizacao_json"] = hashlib.md5(
        (dir_esp / "normalizacao_cnn.json").read_bytes()).hexdigest()
    m["hash_md5_subamostra_csv"] = hashlib.md5(
        (raiz / cfg["experimento"]["caminho_subamostra"]).read_bytes()).hexdigest()

    print(f"\n  f1_macro : {m['f1_macro']:.4f} | EER {m['eer']:.4f} | "
          f"AUC {m['roc_auc_early_stopping']:.4f}")
    print(f"  bonafide : recall {m['recall_bonafide']:.4f} | "
          f"precisao {m['precisao_bonafide']:.4f}")
    print(f"  spoof    : recall {m['recall_spoof']:.4f} | "
          f"precisao {m['precisao_spoof']:.4f}")

    # ---- Artefatos ----------------------------------------------------------
    dir_met = raiz / "results" / "metricas"
    dir_met.mkdir(parents=True, exist_ok=True)

    with open(dir_met / f"curva_treino_{NOME}.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(historico[0].keys()))
        w.writeheader()
        w.writerows(historico)
    print(f"Curva salva em results/metricas/curva_treino_{NOME}.csv")

    _plotar_curva(historico, raiz / "results" / "figuras" / f"curva_treino_{NOME}.png")
    plotar_matriz_confusao(
        cm, raiz / "results" / "figuras" / f"matriz_confusao_{NOME}.png",
        f"CNN baseline (B4.4) — early stopping interno (3k), "
        f"limiar {sel['limiar']:.2f}")

    (raiz / "models").mkdir(exist_ok=True)
    torch.save({"state_dict": melhor["estado"],
                "arquitetura": modelo.descricao(),
                "epoca": melhor["epoca"],
                "semente": semente},
               raiz / "models" / f"{NOME}.pt")

    with open(dir_met / f"{NOME}.json", "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)
    print(f"Salvo: models/{NOME}.pt e results/metricas/{NOME}.json")
    return m


def main() -> dict:
    from ..utils.config import carregar_config

    p = argparse.ArgumentParser(description="B4.4 — CNN baseline")
    p.add_argument("--epocas", type=int, default=30)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--canais", type=str, default="32,64,128,128")
    p.add_argument("--p-drop", type=float, default=0.3)
    p.add_argument("--nao-estrito", action="store_true",
                   help="pula a tentativa de determinismo estrito (registra a limitacao)")
    a = p.parse_args()

    canais = tuple(int(c) for c in a.canais.split(","))
    return treinar(carregar_config(RAIZ), RAIZ, epocas=a.epocas, batch=a.batch,
                   lr=a.lr, canais=canais, p_drop=a.p_drop, estrito=not a.nao_estrito)


if __name__ == "__main__":
    # Windows usa `spawn`: todo codigo de execucao tem de estar sob este guard —
    # obrigatorio se algum dia num_workers > 0 (ver docstring de fixar_seeds_torch).
    main()
