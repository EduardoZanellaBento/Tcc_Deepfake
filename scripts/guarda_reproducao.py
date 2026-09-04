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
    tempo_busca_s, tempo_um_fit_fold_s, tempo_treino_s_n_jobs_1), o que é
    DERIVADO de relógio (projecao_horas_antes_da_busca) e o bloco `ambiente`.
    Tempo VARIA POR DEFINIÇÃO entre execuções — compará-lo aqui geraria alarme
    falso e treinaria o leitor a ignorar o alarme, que é o pior resultado
    possível para uma guarda.

    O corte é entre MEDIDA e CONSEQUÊNCIA: a projeção de horas é ignorada, mas
    o `n_iter_efetivo` que ela decide continua sendo comparado. Se um dia a
    projeção estourar o orçamento e cortar o n_iter, isso muda o resultado — e
    a guarda tem de gritar.

DUAS EXCEÇÕES DECLARADAS, E POR QUE ELAS NÃO SÃO "DESLIGAR O ALARME":
    `REESTRUTURACOES_DECLARADAS` e `DIVERGENCIAS_EXPLICADAS` registram mudanças
    deliberadas desta revisão. As duas são apertadas de propósito: a
    reestruturação só é aceita se o bloco NOVO existir de fato no JSON
    re-gerado, e a divergência explicada só é aceita para o par (antes, depois)
    EXATO declarado. Qualquer outro valor no mesmo campo continua derrubando a
    verificação. Cada entrada carrega o motivo, e o motivo vai para o JSON de
    evidência — quem auditar lê a exceção junto com o resultado.

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
    # DERIVADO de relógio: projecao_horas_antes_da_busca = tempo_um_fit_fold_s
    # x n_iter x 5 / paralelismo (treinar_svm.py). Varia com o t_fit medido
    # (4,6 s -> 5,3 s entre execuções) sem que nada de resultado mude. O que NÃO
    # é ignorado é `n_iter_efetivo` — a DECISÃO que essa projeção alimenta: se a
    # projeção estourasse o orçamento, o n_iter cairia, e aí o resultado mudaria
    # de verdade. Ignora-se a medida, compara-se a consequência.
    "projecao_horas_antes_da_busca",
}

# Os campos que a ordem de serviço exige conferir explicitamente. Se algum
# sumir do JSON, a guarda falha: não basta "não divergiu", tem de ter sido
# comparado. Caminhos com '.' são aninhados; nem todo JSON tem todos.
CAMPOS_EXIGIDOS = [
    "acuracia", "f1_macro", "eer", "limiar", "roc_auc_validacao",
    "matriz_confusao", "hiperparametros", "n_vetores_suporte",
    "selecao_limiar.limiar", "melhor.params",
]

# Reestruturações DECLARADAS: mudanças de esquema deliberadas, feitas nesta
# revisão, em que um campo antigo deixou de existir porque foi SUBSTITUÍDO por
# outro — não porque a informação se perdeu. Cada entrada exige um motivo e o
# caminho novo que passa a carregar a informação; a guarda confere que o caminho
# novo REALMENTE existe no JSON re-gerado antes de aceitar a remoção. Sem essa
# conferência, isto seria uma lista de desculpas em vez de uma declaração.
REESTRUTURACOES_DECLARADAS = {
    "rf_random_search.json": {
        "prefixo_antigo": "limitacao_otimo_na_borda.",
        "prefixo_novo": "limitacao_otimo_na_borda.hiperparametros_na_borda.",
        "motivo": (
            "o bloco limitacao_otimo_na_borda era escrito À MÃO dentro de um "
            "artefato GERADO e sumiu na re-execução (achado da própria guarda, "
            "03/09/2026). Agora é produzido por analisar_bordas() em "
            "src/models/ajustar_rf.py, a partir do espaço de busca e da "
            "configuração vencedora: as notas em prosa viraram campos "
            "estruturados por hiperparâmetro, mais `censurados` e "
            "`consequencia`. A informação foi PRESERVADA e passou a ser "
            "verificável — ver REVISAO_BLOCO3.md, R2.3."),
    },
}

# Divergências EXPLICADAS: campos cujo valor mudou por uma razão conhecida,
# documentada e VERIFICADA — não por mudança de comportamento do modelo. A
# tolerância é apertada de propósito: a guarda só aceita o par (antes, depois)
# EXATO declarado aqui. Qualquer outro valor no mesmo campo continua sendo
# divergência e derruba a verificação. É a diferença entre "explicamos esta
# diferença" e "desligamos o alarme deste campo".
# Remoções DELIBERADAS de campo: o campo sumiu porque o código passou a
# EMITI-LO CONDICIONALMENTE, e a condição não vale mais. Aceita só se os campos
# de `exige` existirem no JSON novo — isto é, se a informação continua lá, em
# forma melhor. Sem esse teste, seria licença para apagar campo.
REMOCOES_DECLARADAS = {
    "curva_aprendizado_rf_tuned_eval.json": {
        "satura_em_n": {
            "exige": ["saturou", "maior_n_medido", "ganho_ultimo_passo_f1",
                      "definicao_saturacao"],
            "motivo": (
                "R3: `satura_em_n` era estruturalmente incapaz de distinguir "
                "'saturou no fim' de 'nunca saturou' — numa curva monótona "
                "crescente a expressão devolvia sempre o último ponto. O campo "
                "passou a ser emitido SOMENTE quando `saturou` é verdadeiro, e a "
                "curva ajustada NÃO satura (o último passo ainda rende +0,0121, "
                "2,4x a tolerância). A ausência do campo é, portanto, a "
                "informação — e ela agora vem explícita em `saturou`, "
                "`ganho_ultimo_passo_f1` e `definicao_saturacao`."),
        },
    },
}

DIVERGENCIAS_EXPLICADAS = {
    "rf_tuned_referencia.json": {
        "selecao_limiar.n_candidatos": {
            "antes": 22103,
            "depois": 22104,
            "explicacao": (
                "o valor ANTIGO foi medido com predict_proba(n_jobs=-1), que NÃO "
                "é reprodutível bit a bit: a soma das 300 árvores num array "
                "compartilhado muda de ordem entre execuções e soma de ponto "
                "flutuante não é associativa. Quatro predições do MESMO .joblib "
                "sobre a MESMA validação deram 22104/22102/22104/22103 com "
                "n_jobs=-1, e 22104 nas quatro (idênticas bit a bit) com "
                "n_jobs=1. Maior diferença de score: 4,4e-16; f1_macro e EER "
                "batem até a décima casa decimal. O valor NOVO é o determinístico "
                "— a correção de uma medição instável, não uma mudança de "
                "resultado. Ver predizer_rf em src/models/avaliacao.py e "
                "REVISAO_BLOCO3.md, R2.3, achado (3)."),
        },
    },
}

ARQUIVOS = [
    "rf_random_search.json",
    "rf_tuned_principal.json",
    "rf_tuned_referencia.json",
    "svm_random_search.json",
    "svm_tuned_principal.json",
    # Dependem dos .joblib re-gerados, então entram na mesma guarda (R2.3,
    # passo 6): se o modelo mudou, é aqui que a análise que o consome denuncia.
    "estabilidade_rf_svm.json",
    "ablacao_mfcc1_std.json",
    "curva_aprendizado_rf_tuned_eval.json",
]


def achatar(obj, prefixo: str = "") -> dict:
    """Achata um JSON em {caminho: valor}, podando as chaves ignoradas.

    Listas de NÚMEROS (matriz de confusão, n_vetores_suporte) ficam INTEIRAS
    como valor único — comparar a matriz 2x2 de uma vez é mais legível no
    relatório do que quatro caminhos separados.

    Listas de DICTS (os pontos da curva de aprendizado, por exemplo) são
    DESCIDAS, item a item. Tratá-las como valor único faria a comparação falhar
    por causa de um `tempo_treino_s` guardado dentro de cada ponto — um campo
    que a lista de ignorados existe justamente para excluir. Sem descer, a
    poda não alcança o que está lá dentro e a guarda dispara alarme falso.
    """
    plano = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in CHAVES_IGNORADAS:
                continue
            plano.update(achatar(v, f"{prefixo}.{k}" if prefixo else k))
    elif isinstance(obj, list) and any(isinstance(v, dict) for v in obj):
        for i, v in enumerate(obj):
            plano.update(achatar(v, f"{prefixo}[{i}]"))
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

    # ---- Divergências explicadas (ver DIVERGENCIAS_EXPLICADAS) -------------
    explicadas = []
    for campo, decl in DIVERGENCIAS_EXPLICADAS.get(nome, {}).items():
        for d in list(divergencias):
            # Casa o par EXATO. Um valor diferente do declarado continua sendo
            # divergência — a explicação vale para aquele fato, não para o campo.
            if (d["campo"] == campo and igual(d["antes"], decl["antes"])
                    and igual(d["depois"], decl["depois"])):
                divergencias.remove(d)
                explicadas.append({**d, "explicacao": decl["explicacao"]})

    # ---- Remoções declaradas (ver REMOCOES_DECLARADAS) ---------------------
    removidos_declarados = []
    for campo, decl in REMOCOES_DECLARADAS.get(nome, {}).items():
        if campo in removidos and all(e in novo for e in decl["exige"]):
            removidos.remove(campo)
            removidos_declarados.append({"campo": campo,
                                         "substituido_por": decl["exige"],
                                         "motivo": decl["motivo"]})

    # ---- Reestruturação declarada (ver REESTRUTURACOES_DECLARADAS) ----------
    reestruturado = []
    decl = REESTRUTURACOES_DECLARADAS.get(nome)
    if decl:
        antigo, novo_pref = decl["prefixo_antigo"], decl["prefixo_novo"]
        # Só vale se o bloco NOVO existe de fato. Uma declaração sem substituto
        # seria autorização para perder rastreabilidade — o oposto da guarda.
        if any(p.startswith(novo_pref) for p in novo):
            movidos = [c for c in removidos if c.startswith(antigo)]
            divergiu_no_bloco = [d for d in divergencias
                                 if d["campo"].startswith(antigo)]
            removidos = [c for c in removidos if not c.startswith(antigo)]
            divergencias = [d for d in divergencias
                            if not d["campo"].startswith(antigo)]
            reestruturado = {
                "campos_absorvidos": movidos,
                "campos_reescritos": [d["campo"] for d in divergiu_no_bloco],
                "prefixo_novo": novo_pref,
                "motivo": decl["motivo"],
            }

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
        "divergencias_explicadas": explicadas,
        "remocoes_declaradas": removidos_declarados,
        "reestruturacao_declarada": reestruturado,
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
        for e in r.get("remocoes_declaradas", []):
            print(f"       REMOÇÃO DECLARADA: {e['campo']} -> substituído por "
                  f"{', '.join(e['substituido_por'])}")
        for e in r.get("divergencias_explicadas", []):
            print(f"       DIVERGÊNCIA EXPLICADA em {e['campo']}: "
                  f"{e['antes']} -> {e['depois']}")
            print(f"         motivo: {e['explicacao'][:120]}...")
        reest = r.get("reestruturacao_declarada")
        if reest:
            print(f"       REESTRUTURADO (declarado): "
                  f"{len(reest['campos_absorvidos'])} campo(s) absorvido(s) por "
                  f"{reest['prefixo_novo']}")
        acres = r.get("campos_acrescentados", [])
        if acres:
            print(f"       acrescentados (esperado): {len(acres)} campo(s)")

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
        "reestruturacoes_declaradas": REESTRUTURACOES_DECLARADAS,
        "divergencias_explicadas_declaradas": DIVERGENCIAS_EXPLICADAS,
        "remocoes_declaradas": REMOCOES_DECLARADAS,
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
