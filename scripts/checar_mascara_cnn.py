"""
Prova de que a máscara temporal da CNN está correta — RODAR ANTES DE TREINAR
============================================================================

POR QUE ISTO EXISTE:
    É a evidência direta da pergunta de banca mais previsível do Bloco 4:
    «Como a máscara temporal acompanhou a redução do eixo do tempo dentro da CNN?»
    São checagens que custam segundos e que, se faltarem, custam dois dias
    procurando bug de arquitetura.

AS TRÊS CHECAGENS:

  1. COMPRIMENTOS — a máscara acompanha a redução em toda a profundidade. O
     `assert` mora dentro do `forward`, então basta uma passada para cobrá-lo; aqui
     ele é exercitado em várias larguras de entrada, para provar que vale por
     CONSTRUÇÃO e não por coincidência com o 251.

  2. INVARIÂNCIA AO PADDING, forma direta — dois exemplos IDÊNTICOS na parte
     válida, com padding DIFERENTE (-3 e +7), COM e SEM máscara. É o teste do
     B4.4 na letra. Os dois números são gravados.

  3. INVARIÂNCIA AO PADDING COM ZONA DE GUARDA — a mesma ideia, mas variando o
     padding só ALÉM DO ALCANCE DA CONVOLUÇÃO. É esta que DECIDE. Ver abaixo.

POR QUE A CHECAGEM 2 NÃO DECIDE SOZINHA (achado desta execução, 18/09/2026):
    O B4.4 previa que a diferença SEM máscara «tem de explodir» em relação à
    diferença COM máscara, e que uma razão próxima de 1 significaria que «o masked
    pooling não está fazendo nada». MEDIDO: a razão fica em 3–4x na maioria das
    sementes e em 0,70x na semente 42 — a do projeto. E, ainda assim, o masked
    pooling está CORRETO. O diagnóstico do documento não se aplica aqui; a causa
    real é outra.

    A causa: com `padding=1` e `kernel=3`, cada bloco vaza para dentro da região
    válida, e o vazamento ACUMULADO chega a 27 frames originais (medido, não
    estimado — ver `alcance_medido` no JSON). Com `n_valid = 100`, o eixo do tempo
    reduzido tem 15 posições, das quais 7 são válidas — e o vazamento contamina 1 a 2
    dessas 7. Ou seja, a diferença COM máscara não é pequena porque a região válida
    é curta demais em relação ao alcance da convolução. Some-se a isso o fato de a
    cabeça ser `Linear(128 -> 2)`, que soma 128 canais com pesos aleatórios de sinal
    variado: os deltas dos canais se cancelam parcialmente, comprimindo justamente o
    número SEM máscara. As duas coisas juntas deixam os dois valores na mesma ordem
    de grandeza, e a razão vira ruído de semente.

    Portanto a checagem 2 é registrada como o B4.4 pede, em VÁRIAS sementes para não
    virar anedota de uma só, mas NÃO é ela que aprova o marco.

POR QUE A CHECAGEM 3 DECIDE:
    Se o padding só varia DEPOIS de uma zona de guarda maior que o alcance da
    convolução, então nenhuma posição reduzida marcada como VÁLIDA consegue enxergar
    a parte que variou. Sob masked pooling, a saída tem de ser IDÊNTICA — não
    «pequena», e sim BIT A BIT IGUAL. E sem máscara ela tem de mudar, porque a média
    comum engole o padding inteiro.

    MEDIDO: com zona de guarda de 32 frames, a diferença COM máscara é ZERO EXATO em
    todas as sementes testadas, e a diferença SEM máscara fica entre 1,7e-2 e 5,8e-2.
    Zero exato contra diferença mensurável é uma separação categórica, não uma razão
    que dependa de sorte de inicialização — é esta a evidência que responde à banca.

Uso:
    python -m scripts.checar_mascara_cnn

Sai com código 1 se qualquer invariante for violada (mesmo padrão de
scripts/auditar_decisoes_cnn.py: a auditoria FALHA, não avisa).
"""

import json
import sys
from pathlib import Path

import torch

from src.models.cnn import CnnDeepfake
from src.utils.seeds import fixar_seeds_torch

RAIZ = Path(__file__).resolve().parents[1]

LARGURA = 251
N_VALID = 100
PAD_A, PAD_B = -3.0, +7.0
# 32 = 2 x 16. O bloco mais profundo agrega 16 frames originais por posicao
# reduzida, e o vazamento da convolucao alcanca ate 27 frames (medido). 32 cobre os
# dois com folga e nao depende do alinhamento de `n_valid` a grade de pooling.
ZONA_GUARDA = 32
SEMENTES = (42, 1, 7, 123, 2026)


def _construir_par(modelo, seed: int, guarda: int):
    """Dois exemplos idênticos na parte válida, com padding diferente após `guarda`."""
    torch.manual_seed(seed)
    base = torch.randn(1, 1, modelo.n_mels, N_VALID)
    comum = torch.full((1, 1, modelo.n_mels, guarda), PAD_A)
    resto = LARGURA - N_VALID - guarda
    a = torch.cat([base, comum, torch.full((1, 1, modelo.n_mels, resto), PAD_A)], dim=3)
    b = torch.cat([base, comum, torch.full((1, 1, modelo.n_mels, resto), PAD_B)], dim=3)
    masc = torch.zeros(1, LARGURA)
    masc[:, :N_VALID] = 1.0
    return a, b, masc


def _par_de_diferencas(seed: int, guarda: int) -> tuple[float, float]:
    """(diferença COM máscara, diferença SEM máscara) nos logits."""
    torch.manual_seed(seed)
    modelo = CnnDeepfake().eval()          # pesos desta semente
    a, b, masc = _construir_par(modelo, seed, guarda)
    with torch.no_grad():
        com = float((modelo(a, masc) - modelo(b, masc)).abs().max())
        sem = float((modelo(a, masc, mascarar=False)
                     - modelo(b, masc, mascarar=False)).abs().max())
    return com, sem


def checar_comprimentos(modelo, larguras=(251, 128, 64, 32)) -> list:
    """Checagem 1 — o `assert` do forward tem de passar em toda largura."""
    registros = []
    for T in larguras:
        x = torch.randn(4, 1, modelo.n_mels, T)
        m = torch.ones(4, T)
        with torch.no_grad():
            saida = modelo(x, m)                       # o assert interno cobra aqui
        assert saida.shape == (4, 2), f"saida inesperada em T={T}: {tuple(saida.shape)}"
        registros.append({"largura_entrada": int(T), "saida": list(saida.shape)})
        print(f"  T={T:>4} -> logits {tuple(saida.shape)}  (assert do forward passou)")
    return registros


def checar_invariancia_direta() -> dict:
    """Checagem 2 — o teste do B4.4 na letra, em várias sementes."""
    por_semente = []
    for s in SEMENTES:
        com, sem = _par_de_diferencas(s, guarda=0)
        razao = sem / com if com > 0 else float("inf")
        por_semente.append({"semente": s, "diferenca_com_mascara": com,
                            "diferenca_sem_mascara": sem, "razao_sem_sobre_com": razao})
        print(f"  semente {s:>4}: com={com:.4e}  sem={sem:.4e}  razao={razao:6.2f}x")
    return {"por_semente": por_semente}


def medir_alcance_convolucao() -> dict:
    """Mede a menor zona de guarda que zera a diferença — o alcance real da convolução.

    Número MEDIDO, não estimado: a aritmética do receptive field é fácil de errar
    (o `max_pool` da máscara tem semântica de teto, então uma posição reduzida válida
    já cobre 16 frames originais ANTES de somar o vazamento da convolução).
    """
    alcances = {}
    for n_valid_teste in (100, 120, 137):
        global N_VALID
        anterior, N_VALID = N_VALID, n_valid_teste
        try:
            minimo = next(
                g for g in range(0, 64)
                if all(_par_de_diferencas(s, g)[0] == 0.0 for s in SEMENTES))
        finally:
            N_VALID = anterior
        alcances[f"n_valid_{n_valid_teste}"] = int(minimo)
        print(f"  n_valid={n_valid_teste}: menor guarda com zero exato = {minimo} frames")
    return {
        "nota": ("menor zona de guarda que torna a diferenca COM mascara ZERO EXATO. "
                 "Varia com o alinhamento de n_valid a grade de pooling (o bloco mais "
                 "profundo agrega 16 frames originais por posicao reduzida)"),
        "por_n_valid": alcances,
        "maior_observado": int(max(alcances.values())),
        "zona_guarda_adotada": ZONA_GUARDA,
    }


def checar_invariancia_com_guarda() -> dict:
    """Checagem 3 — a que decide: zero exato com máscara, diferença sem."""
    por_semente, ok = [], True
    for s in SEMENTES:
        com, sem = _par_de_diferencas(s, guarda=ZONA_GUARDA)
        aprovado = (com == 0.0) and (sem > 0.0)
        ok = ok and aprovado
        por_semente.append({"semente": s, "diferenca_com_mascara": com,
                            "diferenca_sem_mascara": sem,
                            "com_mascara_e_zero_exato": com == 0.0})
        print(f"  semente {s:>4}: com={com:.4e} {'(ZERO EXATO)' if com == 0.0 else '(NAO ZERO!)'}"
              f"  sem={sem:.4e}")
    return {"zona_guarda": ZONA_GUARDA, "por_semente": por_semente, "aprovado": bool(ok)}


def executar() -> dict:
    # estrito=False aqui de proposito: esta checagem so faz forward em CPU com pesos
    # aleatorios, e nao ha treino para o determinismo estrito proteger. O treino
    # (treinar_cnn.py) chama com estrito=True.
    semente = fixar_seeds_torch(42, estrito=False)
    modelo = CnnDeepfake().eval()

    print("\n[1/4] COMPRIMENTOS — a mascara acompanha a reducao do eixo do tempo")
    comprimentos = checar_comprimentos(modelo)

    print("\n[2/4] INVARIANCIA AO PADDING, forma direta (o teste do B4.4 na letra)")
    direta = checar_invariancia_direta()

    print("\n[3/4] ALCANCE DA CONVOLUCAO — medido, nao estimado")
    alcance = medir_alcance_convolucao()

    print(f"\n[4/4] INVARIANCIA COM ZONA DE GUARDA DE {ZONA_GUARDA} — a que decide")
    guarda = checar_invariancia_com_guarda()

    resultado = {
        "proposito": ("evidencia direta da pergunta de banca «como a mascara temporal "
                      "acompanhou a reducao do eixo do tempo dentro da CNN?» — "
                      "rodado ANTES do treino do B4.4"),
        "arquitetura": modelo.descricao(),
        "checagem_1_comprimentos": {
            "nota": ("o assert `mascara x mapa` mora DENTRO do forward; aqui ele e "
                     "exercitado em varias larguras para mostrar que os comprimentos "
                     "batem por CONSTRUCAO (a mascara passa pelo mesmo max_pool1d que "
                     "reduz o mapa), nao por coincidencia com o 251"),
            "larguras_testadas": comprimentos,
            "aprovado": True,
        },
        "checagem_2_invariancia_direta": {
            "nota": ("o teste do B4.4 na letra: dois exemplos identicos na parte valida "
                     f"({N_VALID} frames), padding {PAD_A} contra {PAD_B} em TODO o resto, "
                     "com e sem mascara, nos logits"),
            "achado": (
                "A razao sem/com NAO explode: fica em 3-4x na maioria das sementes e em "
                "0,70x na semente 42, a do projeto. O B4.4 previa que razao ~1 "
                "significaria «o masked pooling nao esta fazendo nada» — e esse "
                "diagnostico NAO se aplica aqui (a checagem 3 prova o contrario, com "
                "zero exato). A causa real e dupla: (a) o vazamento da convolucao "
                "alcanca ate 27 frames originais e, com n_valid=100, o eixo reduzido "
                "tem 15 posicoes das quais so 7 sao validas — o vazamento contamina 1 "
                "a 2 dessas 7, entao a diferenca COM mascara nao tem como ser pequena; "
                "(b) a cabeca Linear(128->2) soma 128 canais com pesos aleatorios de "
                "sinal variado, e os deltas se cancelam parcialmente, comprimindo o "
                "numero SEM mascara. Por isso esta checagem e REGISTRADA mas NAO "
                "aprova o marco."),
            **direta,
        },
        "checagem_3_alcance_medido": alcance,
        "checagem_4_invariancia_com_zona_de_guarda": {
            "nota": ("variando o padding SO depois de uma zona de guarda maior que o "
                     "alcance da convolucao, nenhuma posicao reduzida marcada como "
                     "VALIDA consegue enxergar a parte que variou. Sob masked pooling "
                     "a saida tem de ser BIT A BIT IGUAL — nao «pequena». Sem mascara "
                     "ela tem de mudar, porque a media comum engole o padding inteiro. "
                     "Zero exato contra diferenca mensuravel e uma separacao "
                     "categorica, que nao depende de sorte de inicializacao"),
            "criterio": "diferenca COM mascara == 0.0 exato E diferenca SEM mascara > 0",
            **guarda,
        },
        "resposta_a_banca": (
            "A mascara e reduzida pela MESMA operacao que reduz o mapa de ativacao "
            "(F.max_pool1d com o kernel/stride do eixo do tempo do MaxPool2d), com "
            "assert de comprimento dentro do forward — os tamanhos batem por "
            "construcao, nao por formula. A agregacao final e um masked global "
            "pooling: soma das posicoes validas dividida pelo numero de posicoes "
            "validas. A mascara atua so no tempo; a frequencia e agregada por media "
            "integral, porque nao ha padding em frequencia. Flatten nao aparece em "
            "lugar nenhum. A prova empirica esta na checagem 4: conteudo de padding "
            "fora do alcance da convolucao tem influencia EXATAMENTE NULA na saida "
            "mascarada, e influencia mensuravel na nao-mascarada."),
        "semente": semente,
        "versoes": {"torch": torch.__version__},
    }

    aprovado = (resultado["checagem_1_comprimentos"]["aprovado"]
                and resultado["checagem_4_invariancia_com_zona_de_guarda"]["aprovado"])
    resultado["aprovado"] = bool(aprovado)

    destino = RAIZ / "results" / "metricas" / "checagem_mascara_cnn.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    with open(destino, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)
    print(f"\nSalvo: {destino.relative_to(RAIZ)}")
    print("APROVADO" if aprovado else "REPROVADO")
    return resultado


if __name__ == "__main__":
    r = executar()
    sys.exit(0 if r["aprovado"] else 1)
