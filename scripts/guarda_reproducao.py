"""
Guarda de reprodução do Bloco 3 (R2.3)
=======================================

O QUE ESTE SCRIPT RESPONDE:
    "Rodar de novo dá o mesmo número?" — a pergunta de reprodutibilidade que a
    banca faz e que o B6.1 previa fazer no fim. As mudanças de esquema dos JSONs
    (R2.1 n_jobs_treino, R2.2 roc_auc_cv_std, R4a escopo do tempo) obrigaram a
    re-executar `src.models.ajustar_rf` e `src.models.treinar_svm`. Como os dois
    pipelines são DETERMINÍSTICOS — semente 42 fixada, RandomizedSearchCV com
    random_state=42, SVC com probability=False, e o n_jobs do RF afeta o tempo e
    não o resultado —, a re-execução TEM de reproduzir os números anteriores.
    Ou seja: a re-execução forçada por uma mudança de esquema é, de graça, a
    verificação de reprodutibilidade do trabalho.

    Divergência aqui NÃO é obstáculo a contornar: é ACHADO, e tem de ser
    resolvido antes do Bloco 5. Por isso o script termina com exit code 1 e não
    grava nenhum "reproduziu" quando encontra diferença.

O QUE É COMPARADO:
    Todos os caminhos (chaves aninhadas) comuns aos dois JSONs, EXCETO os
    listados em CHAVES_IGNORADAS. A comparação é de valores exatos para strings,
    inteiros, booleanos e listas; para floats há tolerância de TOL_FLOAT, que
    absorve ruído de arredondamento sem absorver diferença de resultado.

    Campos que existem só no JSON NOVO (os acrescentados por R2.1/R2.2/R4a) são
    reportados como ACRÉSCIMOS, não como divergência — é exatamente o que se
    esperava mudar. Campos que sumiram são reportados como REMOÇÕES e contam
    como divergência: perder rastreabilidade é regressão.

O QUE É IGNORADO, E POR QUÊ:
    tudo que é medida de RELÓGIO (tempos_inferencia, tempo_treino_s,
    tempo_busca_s, tempo_um_fit_fold_s, tempo_treino_s_n_jobs_1) e o bloco
    `ambiente`. Tempo VARIA POR DEFINIÇÃO entre execuções — compará-lo aqui
    geraria alarme falso e treinaria o leitor a ignorar o alarme, que é o pior
    resultado possível para uma guarda.

Rode a partir da raiz:  python -m scripts.guarda_reproducao
"""

import json
import platform
import sys
from datetime import date
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DIR_MET = RAIZ / "results" / "metricas"
DIR_ANTES = DIR_MET / "_pre_revisao"

# Tolerância de float: absorve arredondamento de serialização, não absorve
# diferença de resultado (a 4ª casa decimal é a que os JSONs publicam).
TOL_FLOAT = 1e-9

# Só medida de relógio e contexto de máquina. `n_jobs_treino` e
# `nota_tempo_treino` NÃO entram aqui de propósito: o do RF já existia (-1) e
# tem de continuar igual; o do SVM é novo e deve aparecer como ACRÉSCIMO no
# relatório — é parte do que a revisão mudou, e esconder isso empobreceria a
# evidência.
CHAVES_IGNORADAS = {
    "tempo_treino_s", "tempo_treino_s_n_jobs_1", "tempo_busca_s",
    "tempo_um_fit_fold_s", "tempos_inferencia", "ambiente",
}

# Os campos que a ordem de serviço exige conferir explicitamente. Se algum
# sumir do JSON, a guarda falha: não basta "não divergiu", tem de ter sido
# comparado. Caminhos com '.' são aninhados; nem todo JSON tem todos.
CAMPOS_EXIGIDOS = [
    "acuracia", "f1_macro", "eer", "limiar", "roc_auc_validacao",
    "matriz_confusao", "hiperparametros", "n_vetores_suporte",
    "selecao_limiar.limiar", "melhor.params",
]

ARQUIVOS = [
    "rf_random_search.json",
    "rf_tuned_principal.json",
    "rf_tuned_referencia.json",
    "svm_random_search.json",
    "svm_tuned_principal.json",
]


def achatar(obj, prefixo: str = "") -> dict:
    """Achata um JSON em {caminho: valor}, podando as chaves ignoradas.

    Listas de números (matriz de confusão, n_vetores_suporte) são mantidas
    INTEIRAS como valor único — comparar a lista de uma vez é mais legível no
    relatório do que 4 caminhos separados para uma matriz 2x2.
    """
    plano = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in CHAVES_IGNORADAS:
                continue
            plano.update(achatar(v, f"{prefixo}.{k}" if prefixo else k))
    else:
        plano[prefixo] = obj
    return plano


def igual(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= TOL_FLOAT
    return a == b


def comparar(nome: str) -> dict:
    """Compara um JSON contra a sua cópia em _pre_revisao/."""
    novo_p, antes_p = DIR_MET / nome, DIR_ANTES / nome
    if not antes_p.exists():
        return {"arquivo": nome, "estado": "SEM_REFERENCIA",
                "detalhe": f"{antes_p} não existe — nada a comparar"}
    if not novo_p.exists():
        return {"arquivo": nome, "estado": "AUSENTE",
                "detalhe": f"{novo_p} não existe — o pipeline não foi re-executado"}

    with open(antes_p, encoding="utf-8") as f:
        antes = achatar(json.load(f))
    with open(novo_p, encoding="utf-8") as f:
        novo = achatar(json.load(f))

    comuns = sorted(set(antes) & set(novo))
    divergencias = [{"campo": c, "antes": antes[c], "depois": novo[c]}
                    for c in comuns if not igual(antes[c], novo[c])]
    removidos = sorted(set(antes) - set(novo))
    acrescentados = sorted(set(novo) - set(antes))

    # Os campos exigidos precisam ter sido efetivamente comparados.
    conferidos = [c for c in CAMPOS_EXIGIDOS
                  if any(p == c or p.startswith(c + ".") for p in comuns)]
    faltando = [c for c in CAMPOS_EXIGIDOS
                if c not in conferidos and any(
                    p == c or p.startswith(c + ".") for p in set(antes) | set(novo))]

    ok = not divergencias and not removidos and not faltando
    return {
        "arquivo": nome,
        "estado": "REPRODUZIU" if ok else "DIVERGIU",
        "n_campos_comparados": len(comuns),
        "campos_exigidos_conferidos": conferidos,
        "campos_exigidos_nao_conferidos": faltando,
        "divergencias": divergencias,
        "campos_removidos": removidos,
        "campos_acrescentados": acrescentados,
    }


def main() -> int:
    print("=" * 74)
    print("GUARDA DE REPRODUÇÃO — Bloco 3 (R2.3)")
    print("=" * 74)
    print(f"referência: {DIR_ANTES.relative_to(RAIZ)}")
    print(f"ignorados : {', '.join(sorted(CHAVES_IGNORADAS))}")
    print("            (medidas de relógio variam por definição)\n")

    relatorios = [comparar(nome) for nome in ARQUIVOS]
    falhou = False
    for r in relatorios:
        marca = {"REPRODUZIU": "[OK ]", "DIVERGIU": "[!! ]"}.get(r["estado"], "[?? ]")
        print(f"{marca} {r['arquivo']}: {r['estado']}"
              + (f" ({r['n_campos_comparados']} campos comparados)"
                 if "n_campos_comparados" in r else f" — {r.get('detalhe','')}"))
        for d in r.get("divergencias", []):
            falhou = True
            print(f"       DIVERGÊNCIA em {d['campo']}:")
            print(f"         antes : {d['antes']}")
            print(f"         depois: {d['depois']}")
        for c in r.get("campos_removidos", []):
            falhou = True
            print(f"       REMOVIDO: {c} (regressão de rastreabilidade)")
        for c in r.get("campos_exigidos_nao_conferidos", []):
            falhou = True
            print(f"       EXIGIDO NÃO CONFERIDO: {c}")
        if r["estado"] not in ("REPRODUZIU", "DIVERGIU"):
            falhou = True
        acres = r.get("campos_acrescentados", [])
        if acres:
            print(f"       acrescentados (esperado): {', '.join(acres)}")

    print()
    if falhou:
        print("=" * 74)
        print("RESULTADO: NÃO REPRODUZIU. PARE — não commite.")
        print("Divergência de reprodução é ACHADO, não obstáculo a contornar:")
        print("os dois pipelines são determinísticos, então uma diferença aqui")
        print("aponta mudança real de comportamento e tem de ser explicada")
        print("ANTES do Bloco 5 (avaliação única do teste lacrado).")
        print("=" * 74)
        return 1

    registro = {
        "verificacao": "reproducao_bloco3",
        "data": date.today().isoformat(),
        "resultado": "REPRODUZIU",
        "pergunta": ("a re-execução de src.models.ajustar_rf e "
                     "src.models.treinar_svm, forçada pelas mudanças de esquema "
                     "R2.1/R2.2/R4a, reproduz os resultados publicados antes da "
                     "revisão de 03/09/2026?"),
        "por_que_tem_de_reproduzir": (
            "os dois pipelines são determinísticos: semente 42 fixada, "
            "RandomizedSearchCV com random_state=42, SVC com probability=False "
            "(determinístico), e o n_jobs do RF altera o tempo, não o resultado"),
        "referencia": "results/metricas/_pre_revisao/ (cópia dos JSONs pré-revisão)",
        "campos_exigidos": CAMPOS_EXIGIDOS,
        "chaves_ignoradas": sorted(CHAVES_IGNORADAS),
        "por_que_ignorar_tempo": (
            "medidas de relógio variam por definição entre execuções; "
            "compará-las produziria alarme falso e desmoralizaria a guarda"),
        "tolerancia_float": TOL_FLOAT,
        "arquivos": relatorios,
        "ambiente": {
            "python": platform.python_version(),
            "sistema": f"{platform.system()} {platform.release()}",
        },
    }
    caminho = DIR_MET / "reproducao_bloco3.json"
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(registro, f, indent=2, ensure_ascii=False)
    print("=" * 74)
    print("RESULTADO: REPRODUZIU em todos os artefatos comparados.")
    print(f"Evidência gravada em {caminho.relative_to(RAIZ)} — é este o arquivo")
    print("a citar na resposta de banca sobre reprodutibilidade.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
