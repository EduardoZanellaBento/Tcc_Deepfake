"""
Partição dos dados — split estratificado 70/15/15
=================================================

"onde exatamente o split 70/15/15 e o Stratified 5-Fold acontecem, e em qual subconjunto cada um é aplicado?"

A resposta, e a arquitetura deste módulo:

    148.176 áudios (fase == 'eval' — decisão do orientador; ver abaixo)
        |
        +-- TREINO 70% (103.723) --> é AQUI que o Stratified 5-Fold roda,
        |                            durante a busca de hiperparâmetros.
        |                            O 5-fold particiona SÓ o treino.
        |
        +-- VALIDAÇÃO 15% (22.226) -> comparar modelos / decisões de projeto
        |
        +-- TESTE 15% (22.227) ----> INTOCADO até o resultado final. Usado UMA vez.

UNIVERSO DO EXPERIMENTO (decisão metodológica aprovada pelo orientador):
    fase == 'eval' (148.176) — o conjunto oficialmente pontuado do ASVspoof 2021
    LA. 'progress' (16.464) e 'hidden' (16.926) são EXCLUÍDOS: o hidden tem
    silêncio pré-cortado na origem (trim == 'only_speech'), pré-processamento
    distinto que contamina qualquer análise de proporção de fala. O filtro vem
    da coluna `fase` do labels.csv, não de uma lista hard-coded de arquivos.
    O split anterior, no universo de 181.566, está preservado em
    data/processed/split_181k.csv (é ele que gerou o rf_baseline.json histórico).

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

TODO [LIMITAÇÃO CONHECIDA — documentar no TC II]
Split aleatório por utterance, estratificado só por classe_binaria. Cada ataque
(A07-A19), codec e LOCUTOR se repete em milhares de arquivos, aparecendo em treino E
teste. O modelo pode memorizar a assinatura de um vocoder/locutor específico.
Implicações: (1) métricas otimistas vs. cenário cross-attack; (2) NÃO são
comparáveis ao EER de 1,32% de Yamagishi et al. (2022), cujo protocolo é
deliberadamente cross-attack. Mitigação mínima: reportar métricas por ataque
(B2) e por codec (B5). Ideal: experimento leave-one-attack-out (decisão do orientador).
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ..utils.seeds import fixar_seeds


def criar_split(cfg: dict, raiz: Path, forcar: bool = False,
                fase: str | None = None) -> pd.DataFrame:
    """Gera (ou recarrega) a partição estratificada 70/15/15 no universo `fase`.

    Args:
        cfg: config.yaml carregado.
        raiz: raiz do projeto.
        forcar: se True, regera mesmo que split.csv já exista.
        fase: override PONTUAL do universo, para scripts de diagnóstico.
            Se None (default), vale `cfg["dataset"]["fase"]` — 'eval'
            (148.176), decisão do orientador. O comportamento padrão é
            governado pelo config, não por este parâmetro.

    Returns:
        DataFrame [arquivo, conjunto].
    """
    if fase is None:
        fase = cfg["dataset"]["fase"]
    semente = fixar_seeds(cfg["semente"])
    saida = raiz / "data" / "processed" / "split.csv"

    # Idempotência: se o split já existe, RECARREGA em vez de regerar.
    # Isso garante que RF, SVM e CNN usem exatamente a mesma partição, mesmo rodando em dias diferentes.
    if saida.exists() and not forcar:
        print(f"Split já existe em {saida} — recarregando (use forcar=True para regerar).")
        return pd.read_csv(saida)

    feats = pd.read_csv(raiz / "data" / "features" / "features.csv",
                        usecols=["arquivo", "classe_binaria"])

    # ---- Filtro de universo: fase == 'eval' (vem do labels.csv) ---------------
    # O features.csv não tem a coluna `fase`; ela mora no labels.csv. O merge é
    # validado: se algum áudio do universo escolhido não tiver features, o erro
    # é explícito (mesma filosofia de carregar_dados_split).
    labels = pd.read_csv(raiz / "data" / "processed" / "labels.csv",
                         usecols=["arquivo", "fase"])
    n_universo = int((labels["fase"] == fase).sum())
    feats = feats.merge(labels.loc[labels["fase"] == fase, ["arquivo"]],
                        on="arquivo", how="inner")
    if len(feats) != n_universo:
        raise ValueError(
            f"Universo fase=='{fase}' tem {n_universo} áudios no labels.csv, mas "
            f"só {len(feats)} têm features extraídas. Investigar antes de regerar."
        )
    print(f"Universo do split: fase=='{fase}' -> {len(feats)} áudios.")

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
    """Junta features.csv + split.csv numa única tabela pronta para treinar.

    O universo do experimento é o do split.csv (fase=='eval', 148.176), que é um
    SUBCONJUNTO do features.csv (181.566, extraído antes da decisão de filtrar).
    A integridade que importa: TODA linha do split precisa encontrar suas
    features — se o merge devolver menos linhas que o split, há áudio sem
    features e o erro é explícito.
    """
    feats = pd.read_csv(raiz / "data" / "features" / "features.csv")
    split = pd.read_csv(raiz / "data" / "processed" / "split.csv")
    df = feats.merge(split, on="arquivo", how="inner")

    if len(df) != len(split):
        raise ValueError(
            f"Merge perdeu linhas: split={len(split)}, resultado={len(df)}. "
            "O split.csv provavelmente foi gerado a partir de outro features.csv. "
            "Regere com forcar=True."
        )
    return df


def filtrar_treino_braco(treino: pd.DataFrame, braco: str, cfg: dict,
                         raiz: Path) -> pd.DataFrame:
    """Aplica o braço do experimento (config: bloco `experimento`) ao TREINO.

    Os dois braços (decisão do orientador, documentada no config.yaml):
        'principal'  -> filtra o treino pelos IDs da subamostra estratificada
                        (~30k) de `experimento.caminho_subamostra`, compartilhada
                        por RF, SVM e CNN — o ambiente experimental da comparação.
        'referencia' -> devolve o treino completo, sem alteração; existe para
                        quantificar o custo da subamostra.

    Aplica-se SOMENTE ao conjunto de treino: validação e teste permanecem
    COMPLETOS nos dois braços (exigência textual do orientador). Quem chama é
    responsável por passar apenas as linhas com conjunto == 'treino'.

    Args:
        treino: linhas de treino já unidas às features (carregar_dados_split).
        braco: 'principal' ou 'referencia'. O default do experimento vem de
            cfg['experimento']['braco']; o parâmetro é explícito para o RF
            poder rodar os DOIS braços e o SVM (futuro) rodar só o principal.
        cfg: config.yaml carregado.
        raiz: raiz do projeto.

    Returns:
        DataFrame de treino do braço pedido.
    """
    if braco == "referencia":
        return treino
    if braco != "principal":
        raise ValueError(
            f"Braço desconhecido: '{braco}'. Válidos: 'principal', 'referencia' "
            "(config.yaml, chave experimento.braco)."
        )

    caminho = raiz / cfg["experimento"]["caminho_subamostra"]
    if not caminho.exists():
        raise FileNotFoundError(
            f"Subamostra não encontrada em {caminho}. "
            "Gere-a com: python -m scripts.gerar_subamostra"
        )
    ids = pd.read_csv(caminho, usecols=["arquivo"])

    filtrado = treino.merge(ids, on="arquivo", how="inner")
    # Integridade: TODO ID da subamostra precisa existir no treino atual. Menos
    # linhas que a subamostra = ela foi gerada a partir de OUTRO split.
    if len(filtrado) != len(ids):
        raise ValueError(
            f"Subamostra tem {len(ids)} IDs, mas só {len(filtrado)} estão no "
            "treino atual. A subamostra foi gerada de outro split.csv — "
            "regere com scripts/gerar_subamostra.py antes de treinar."
        )
    return filtrado


def colunas_features(df: pd.DataFrame) -> list[str]:
    """Devolve SÓ as 44 colunas de features acústicas. PONTO ÚNICO que define o X.

    CUIDADO (bug fácil de cometer): o features.csv também contém colunas de
    DIAGNÓSTICO, que não são features declaradas na metodologia:

      - `prop_fala`        -> quanto do áudio o VAD manteve;
      - `n_frames_validos` -> quantos frames entraram na agregação mascarada;
      - `n_frames_total`   -> quantos frames existiriam sem mascarar.

    As três medem QUANTIDADE DE FALA/SILÊNCIO do arquivo, não timbre. Como o
    bonafide perde mais sinal no VAD que o spoof, qualquer uma delas dentro do X
    daria ao modelo um atalho estatístico: ele acertaria classificando pela
    duração da fala, não por artefato de síntese, e a conclusão do trabalho
    desabaria na banca. Elas ficam no CSV porque são auditáveis e citáveis; ficam
    fora do X porque não são o objeto de estudo.

    Fora também `arquivo`, `label` e `classe_binaria` (esta última é o alvo:
    incluí-la seria vazamento total, o modelo acertaria 100%), além de `conjunto`
    e `ataque`, que entram por merge com split.csv / labels.csv.
    """
    # Import local (e não no topo) de propósito: extrair_features puxa o librosa,
    # pesado e inútil para quem só vai treinar sobre o CSV. A lista mora lá
    # porque é lá que essas colunas são ESCRITAS — uma cópia aqui seria a fonte
    # clássica de divergência silenciosa entre quem grava e quem lê.
    from ..features.extrair_features import COLUNAS_DIAGNOSTICO

    excluir = {"arquivo", "label", "classe_binaria", "conjunto", "ataque"}
    excluir.update(COLUNAS_DIAGNOSTICO)
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