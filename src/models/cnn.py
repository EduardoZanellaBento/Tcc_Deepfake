"""
Arquitetura da CNN — espectrograma cru + agregação global MASCARADA (P2, opção D)
=================================================================================

POR QUE ESTE MÓDULO É SEPARADO DO TREINO:
    A arquitetura é a peça que o B4.5 vai mexer; o protocolo de treino, não. Mantê-la
    num módulo próprio deixa o B4.5 trocar blocos/canais sem tocar em nada que decida
    limiar, métrica ou split — e deixa a checagem da máscara (scripts/checar_mascara_cnn.py)
    importar a rede sem arrastar o laço de treino junto.

O REFINAMENTO OBRIGATÓRIO DA P2 — a máscara temporal:
    O enunciado do orientador, literal: a máscara precisa ACOMPANHAR A REDUÇÃO
    TEMPORAL DA REDE. Havendo MaxPool, stride ou qualquer camada que encurte o eixo
    do tempo, a máscara é reduzida de forma coerente. O resultado final é um MASKED
    GLOBAL POOLING — soma apenas das posições válidas, dividida pelo número de
    posições válidas —, nunca um GlobalAveragePooling comum sobre regiões inválidas.
    A máscara atua SÓ no tempo; o eixo de frequência continua integral.
    `Flatten` está VETADO.

    A implementação abaixo satisfaz isso POR CONSTRUÇÃO: a máscara não é reduzida por
    uma fórmula aritmética escrita à mão (que ficaria desatualizada assim que o B4.5
    mudasse a arquitetura), e sim PELA MESMA OPERAÇÃO DE POOLING que reduz o mapa de
    ativação. Os comprimentos coincidem porque são produzidos pela mesma conta, e um
    `assert` dentro do `forward` cobra isso a cada passada.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CnnDeepfake(nn.Module):
    """CNN pequena + agregacao global mascarada (P2, opcao D).

    DECISOES DE DESIGN:

    1. `Flatten` esta VETADO (exigencia do orientador). Duas razoes: infla os
       parametros (128x251 achatado sao ~32k entradas na densa) e amarra a rede
       as POSICOES de padding — a rede aprenderia "onde" o audio termina, que e
       a variavel espuria que o Bloco 1 tirou do X do ramo classico.

    2. A mascara e reduzida pela MESMA operacao que reduz o mapa: `max_pool1d`
       com o mesmo kernel/stride do eixo do tempo do `MaxPool2d`. Se o B4.5
       trocar a arquitetura, os comprimentos continuam batendo sem ninguem
       recalcular nada. Convencao: `max_pool` sobre mascara binaria marca a
       posicao reduzida como valida se QUALQUER posicao original dela era
       valida — semantica de teto, coerente com o `ceil` de `frames_validos`.

    3. A mascara atua SO no tempo. O eixo de frequencia e agregado por media
       integral (todas as 128 faixas Mel sao validas em todo exemplo) — nao ha
       padding em frequencia, e mascarar ali seria inventar um problema.

    4. Duas saidas + CrossEntropyLoss (P7). NAO BCEWithLogitsLoss: com
       `bonafide=0` / `spoof=1` e spoof MAJORITARIA (9:1),
       `pos_weight=9` daria peso maior a majoritaria — inverteria a intencao.

    5. `mascarar=False` no `forward` desliga SO o masked pooling (vira `mean`
       comum sobre os 251 frames), mantendo pesos e todo o resto identicos. E o
       que torna o teste de invariancia ao padding CONCLUSIVO: sem esse modo, o
       teste mediria um numero pequeno sem ter contra o que compara-lo. Ver
       `scripts/checar_mascara_cnn.py`.

    LIMITACOES CONHECIDAS E REGISTRADAS (nenhuma impede a validade do
    experimento principal — ver `limitacoes_registradas` em
    results/metricas/cnn_baseline.json):

    6. O masked pooling protege a AGREGACAO, nao a NORMALIZACAO. O
       `BatchNorm2d` de cada bloco calcula media e variancia por canal sobre o
       LOTE inteiro (N, F, T), padding incluido, e normaliza com isso tambem as
       posicoes validas. Nao e vazamento POR EXEMPLO — a estatistica e do lote e
       desloca todos os exemplos igualmente, logo nao codifica onde o audio
       termina. O unico vazamento por exemplo e o da borda da convolucao, medido
       em ate 27 frames originais (checagem 3 de checagem_mascara_cnn.json).

    7. A reducao da mascara usa semantica de TETO: `max_pool1d` marca a posicao
       reduzida como valida se QUALQUER frame original dela era valido. Depois
       dos 4 blocos cada posicao reduzida agrega 16 frames (251 -> 125 -> 62 ->
       31 -> 15), entao a posicao de FRONTEIRA pode ser majoritariamente padding
       e ainda entrar no pooling com peso 1 — no maximo 1 posicao entre as
       validas. A alternativa (semantica de piso) descartaria a fronteira do
       audio, que e informacao real. Convencao declarada, coerente com o `ceil`
       de `n_frames_validos` do Bloco 1.
    """

    def __init__(self, n_mels: int = 128, canais=(32, 64, 128, 128),
                 p_drop: float = 0.3):
        super().__init__()
        blocos, c_in = [], 1
        for c_out in canais:
            blocos.append(nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),   # reduz freq E tempo por 2
            ))
            c_in = c_out
        self.blocos = nn.ModuleList(blocos)
        self.dropout = nn.Dropout(p_drop)
        self.cabeca = nn.Linear(c_in, 2)                 # 2 saidas (P7)

        self.n_mels = n_mels
        self.canais = tuple(canais)
        self.p_drop = p_drop

    def forward(self, x, mascara, mascarar: bool = True):
        """x: (B, 1, 128, 251) ja normalizado. mascara: (B, 251) float {0,1}."""
        m = mascara.unsqueeze(1)                         # (B, 1, T)
        for bloco in self.blocos:
            x = bloco(x)                                 # (B, C, F', T')
            # a mascara sofre a MESMA reducao temporal do bloco
            m = F.max_pool1d(m, kernel_size=2, stride=2) # (B, 1, T')
        assert m.shape[-1] == x.shape[-1], (
            f"mascara ({m.shape[-1]}) e mapa ({x.shape[-1]}) divergiram no tempo")

        # frequencia: media INTEGRAL (nao ha padding em frequencia)
        h = x.mean(dim=2)                                # (B, C, T')

        if mascarar:
            # MASKED GLOBAL POOLING no tempo: soma so das posicoes validas,
            # dividida pelo NUMERO de posicoes validas daquele exemplo.
            n_valid = m.sum(dim=2).clamp(min=1.0)        # (B, 1)
            h = (h * m).sum(dim=2) / n_valid             # (B, C)
        else:
            # SO PARA DIAGNOSTICO (ver decisao 5): media sobre TUDO, inclusive
            # as posicoes de padding. Nunca usar no treino.
            h = h.mean(dim=2)                            # (B, C)

        return self.cabeca(self.dropout(h))              # (B, 2) logits

    def descricao(self) -> dict:
        """Resumo da arquitetura para o JSON de métricas (sem `Flatten` em lugar nenhum)."""
        n_par = sum(p.numel() for p in self.parameters())
        return {
            "nome": "CnnDeepfake",
            "n_blocos": len(self.blocos),
            "canais": list(self.canais),
            "bloco": "Conv2d(3x3, padding=1, bias=False) -> BatchNorm2d -> ReLU -> MaxPool2d(2,2)",
            "n_mels": self.n_mels,
            "p_dropout": self.p_drop,
            "cabeca": "Linear(C_final -> 2)",
            "agregacao": ("media integral no eixo de frequencia + masked global "
                          "pooling no eixo do tempo (soma das posicoes validas / "
                          "numero de posicoes validas)"),
            "reducao_da_mascara": ("F.max_pool1d(kernel=2, stride=2) por bloco — a "
                                   "MESMA operacao que reduz o eixo do tempo do mapa; "
                                   "comprimentos batem por construcao, com assert no forward"),
            "flatten": False,
            "n_parametros": int(n_par),
        }


def pesos_balanced(n_por_classe, dispositivo=None) -> torch.Tensor:
    """Pesos da CrossEntropyLoss pela MESMA fórmula do `class_weight='balanced'`.

        w_c = n / (k * n_c)

    Derivados das CONTAGENS REAIS do split interno, nunca hard-coded: se o split
    mudar (p.ex. o refit nos 30k do B4.6), os pesos acompanham sozinhos.

    Por que a forma `balanced` e não `[9.0, 1.0]`: a razão é a mesma (9:1 a favor do
    bonafide, que é a MINORITÁRIA), mas a `balanced` mantém a MAGNITUDE MÉDIA da loss
    comparável à não-ponderada — o que torna a curva legível e comparável entre
    configurações.

    Args:
        n_por_classe: sequência (n_classe_0, n_classe_1) — bonafide, spoof.
    """
    n_por_classe = [float(n) for n in n_por_classe]
    k = len(n_por_classe)
    n = sum(n_por_classe)
    pesos = [n / (k * n_c) for n_c in n_por_classe]
    return torch.tensor(pesos, dtype=torch.float32, device=dispositivo)
