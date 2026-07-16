"""
Partição dos dados — split estratificado 70/15/15
=================================================

"onde exatamente o split 70/15/15 e o Stratified 5-Fold acontecem, e em qual subconjunto cada um é aplicado?"

A resposta, e a arquitetura deste módulo:

    181.566 áudios
        |
        +-- TREINO 70% (127.096) --> é AQUI que o Stratified 5-Fold roda,
        |                            durante a busca de hiperparâmetros.
        |                            O 5-fold particiona SÓ o treino.
        |
        +-- VALIDAÇÃO 15% (27.235) -> comparar modelos / decisões de projeto
        |
        +-- TESTE 15% (27.235) ----> INTOCADO até o resultado final. Usado UMA vez.

Por que três conjuntos e não dois:
    Se você escolhe hiperparâmetros olhando o teste, o teste deixa de ser uma
    estimativa honesta de generalização — você o contaminou com decisões. A
    validação existe para absorver esse desgaste. O teste é a "prova final".

Por que ESTRATIFICADO:
    O ASVspoof LA é desbalanceado (~8,8 spoof : 1 bonafide). Num split aleatório
    simples, a proporção oscila entre os subconjuntos por acaso, e a comparação
    entre eles deixa de ser justa. A estratificação FORÇA cada subconjunto a
    manter a proporção original.

Saída: data/processed/split.csv  com [arquivo, conjunto], conjunto ∈
{treino, validacao, teste}. É salvo em disco DE PROPÓSITO: o split passa a ser um
artefato versionável e auditável, idêntico para RF, SVM e CNN. Sem isso, cada
modelo poderia (por descuido) treinar numa partição diferente, e a comparação —
que é o coração deste trabalho — perderia o sentido.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ..utils.seeds import fixar_seeds


def criar_split(cfg: dict, raiz: Path, forcar: bool = False) -> pd.DataFrame:
    """Gera (ou recarrega) a partição estratificada 70/15/15.

    Args:
        cfg: config.yaml carregado.
        raiz: raiz do projeto.
        forcar: se True, regera mesmo que split.csv já exista.

    Returns:
        DataFrame [arquivo, conjunto].
    """
    semente = fixar_seeds(cfg["semente"])
    saida = raiz / "data" / "processed" / "split.csv"

    # Idempotência: se o split já existe, RECARREGA em vez de regerar.
    # Isso garante que RF, SVM e CNN usem exatamente a mesma partição, mesmo rodando em dias diferentes.
    if saida.exists() and not forcar:
        print(f"Split já existe em {saida} — recarregando (use forcar=True para regerar).")
        return pd.read_csv(saida)

    feats = pd.read_csv(raiz / "data" / "features" / "features.csv",
                        usecols=["arquivo", "classe_binaria"])

    p_treino = cfg["split"]["treino"]
    p_val = cfg["split"]["validacao"]
    p_teste = cfg["split"]["teste"]
    assert abs((p_treino + p_val + p_teste) - 1.0) < 1e-9, "As proporções devem somar 1."

    # ---- 1º corte: treino (70%) vs. resto (30%) --------------------------------
    # `stratify=y` é o que mantém a proporção spoof/bonafide idêntica dos dois lados.
    treino, resto = train_test_split(
        feats,
        train_size=p_treino,
        stratify=feats["classe_binaria"],
        random_state=semente,
        shuffle=True,
    )

    # ---- 2º corte: o resto (30%) vira validação (15%) + teste (15%) ------------
    # Atenção à aritmética: queremos 15% do TOTAL, mas estamos cortando um bloco
    # que é 30% do total. Então a fração DENTRO do resto é 0.15/0.30 = 0.50.
    # Errar isso é um bug silencioso clássico (dá 15%/15% do resto = 4,5%/25,5%).
    frac_val = p_val / (p_val + p_teste)
    validacao, teste = train_test_split(
        resto,
        train_size=frac_val,
        stratify=resto["classe_binaria"],
        random_state=semente,
        shuffle=True,
    )

    split = pd.concat([
        pd.DataFrame({"arquivo": treino["arquivo"], "conjunto": "treino"}),
        pd.DataFrame({"arquivo": validacao["arquivo"], "conjunto": "validacao"}),
        pd.DataFrame({"arquivo": teste["arquivo"], "conjunto": "teste"}),
    ])
    # Ordena por 'arquivo': deixa o CSV determinístico byte a byte, o que torna
    # diffs de git legíveis e o artefato auditável.
    split = split.sort_values("arquivo").reset_index(drop=True)

    saida.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(saida, index=False)
    print(f"Split salvo em {saida}")
    return split


def carregar_dados_split(raiz: Path) -> pd.DataFrame:
    """Junta features.csv + split.csv numa única tabela pronta para treinar."""
    feats = pd.read_csv(raiz / "data" / "features" / "features.csv")
    split = pd.read_csv(raiz / "data" / "processed" / "split.csv")
    df = feats.merge(split, on="arquivo", how="inner")

    if len(df) != len(feats):
        raise ValueError(
            f"Merge perdeu linhas: features={len(feats)}, resultado={len(df)}. "
            "O split.csv provavelmente foi gerado a partir de outro features.csv. "
            "Regere com forcar=True."
        )
    return df


def colunas_features(df: pd.DataFrame) -> list[str]:
    """Devolve SÓ as 44 colunas de features acústicas.

    CUIDADO (bug fácil de cometer): o features.csv também contém `prop_fala`, que é
    um DIAGNÓSTICO do VAD, não uma feature declarada na metodologia. Se ela entrar
    no X, estará treinando com uma variável que não está na fundamentação
    teórica. Fora também `arquivo`, `label` e `classe_binaria` (esta última é o alvo: incluí-la seria vazamento
    total, o modelo acertaria 100%).
    """
    excluir = {"arquivo", "label", "classe_binaria", "prop_fala", "conjunto", "ataque"}
    return [c for c in df.columns if c not in excluir]


def resumo_split(df: pd.DataFrame) -> None:
    """Imprime a conferência que prova que a estratificação funcionou."""
    print("\n" + "=" * 64)
    print("CONFERÊNCIA DO SPLIT")
    print("=" * 64)
    print(f"{'conjunto':<12} {'total':>8} {'bonafide':>9} {'spoof':>8} {'% spoof':>9} {'razão':>8}")
    for nome in ["treino", "validacao", "teste"]:
        sub = df[df["conjunto"] == nome]
        n_bona = int((sub["classe_binaria"] == 0).sum())
        n_spoof = int((sub["classe_binaria"] == 1).sum())
        pct = 100 * n_spoof / len(sub) if len(sub) else 0
        razao = n_spoof / n_bona if n_bona else float("nan")
        print(f"{nome:<12} {len(sub):>8} {n_bona:>9} {n_spoof:>8} {pct:>8.2f}% {razao:>7.2f}:1")

    total = len(df)
    n_bona = int((df["classe_binaria"] == 0).sum())
    n_spoof = int((df["classe_binaria"] == 1).sum())
    print("-" * 64)
    print(f"{'TOTAL':<12} {total:>8} {n_bona:>9} {n_spoof:>8} "
          f"{100*n_spoof/total:>8.2f}% {n_spoof/n_bona:>7.2f}:1")
    print("\nSe as razões das 3 linhas baterem com a do TOTAL, a estratificação funcionou.")


if __name__ == "__main__":
    import yaml

    RAIZ = Path(__file__).resolve().parents[2]
    cfg = yaml.safe_load(open(RAIZ / "config" / "config.yaml", encoding="utf-8"))

    criar_split(cfg, RAIZ)
    resumo_split(carregar_dados_split(RAIZ))