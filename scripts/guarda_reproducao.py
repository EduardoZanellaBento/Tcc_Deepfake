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

EXTENSÃO DO B5.2 (23/09/2026) — OS ARTEFATOS DA CNN:
    um segundo grupo, com evidência própria (reproducao_cnn.json), cobre o que
    a CNN acrescentou e pode ser regenerado SEM treinar nada:

      cnn_final_principal.json           limiar, n_candidatos, métricas (B4.7)
      diagnostico_por_ataque_resumo.json os três modelos, CNN inclusive
      diagnostico_por_codec_resumo.json  idem
      comparacao_estatistica.json        bootstrap pareado do B5.2

    A referência de cada um é a cópia em _pre_revisao/ tirada ANTES da
    regeneração (as três primeiras são as versões publicadas no B4.7).

    O B4.7 NÃO é regenerado por cima do publicado. validar_cnn re-mede os
    tempos a cada execução, e os tempos publicados (1,1093 ms, 0,2383 ms/áudio…)
    já estão citados no README e nos apêndices; sobrescrevê-los para verificar
    OUTRA coisa trocaria números citados sem motivo. Por isso
    `--regenerar-cnn` roda o PRÓPRIO validar_cnn.main() — o mesmo código, não
    uma reimplementação — com o nome de saída trocado (NOME_REEXECUCAO), e a
    guarda compara a referência contra essa regeneração. A única diferença de
    conteúdo que isso introduz é o campo `modelo`, declarada abaixo pelo par
    exato. E a guarda confere, à parte, que o publicado continua byte a byte
    igual à referência.

    É a pegadinha que o marco manda vigiar: soma paralela em ponto flutuante
    (no RF, o n_jobs; na CNN, o cuDNN e o TF32). Se `n_candidatos` da CNN
    variar entre execuções, é aqui que aparece.

    FICAM DE FORA, e o motivo vai para a evidência (NAO_REGENERAVEIS):
    teste_lacrado.json (execução única — reexecutar é justamente o que o B5.1
    proíbe; a reprodução dele está no pré-voo e no ensaio do próprio script) e
    os artefatos de TREINO da CNN (refit_cnn.json, cnn_definida.json,
    cnn_baseline.json): regenerá-los re-treinaria e sobrescreveria os .pt, o
    que o B5 proíbe.

    "data" entrou em CHAVES_IGNORADAS: é a data da execução (relógio), e só os
    resumos de diagnóstico a carregam — nenhum artefato do Bloco 3 tem o campo.

Rode a partir da raiz:
    python -m scripts.guarda_reproducao                  # compara os dois grupos
    python -m scripts.guarda_reproducao --regenerar-cnn  # antes, regenera o B4.7
                                                         # (~7 min; re-mede tempos)
Sequência completa do grupo CNN: diagnostico_por_ataque, diagnostico_por_codec
e comparacao_final regeneram em lugar; depois --regenerar-cnn.
"""

import argparse
import hashlib
import json
import platform
import sys
from datetime import date
from pathlib import Path
from unittest.mock import patch

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
    # Data da execução (B5.2): relógio, não resultado. Só os resumos de
    # diagnóstico a carregam; nenhum artefato do Bloco 3 tem este campo.
    "data",
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
    "cnn_final_principal.json": {
        "modelo": {
            "antes": "cnn_final_principal",
            "depois": "cnn_final_principal_reexecucao",
            "explicacao": (
                "a regeneração roda validar_cnn.main() com NOME trocado para não "
                "sobrescrever o artefato publicado (que re-mediria os tempos já "
                "citados no texto); `modelo` é o único campo que recebe NOME. "
                "Ver regenerar_cnn()."),
        },
        # ACHADO DA PRÓPRIA GUARDA (23/09/2026): o md5 gravado no B4.7 é o do
        # código que RODOU o B4.7 (commit fbdba7d). Depois dele, o commit
        # a37eed3 corrigiu o título cortado das matrizes de confusão — mexeu só
        # no LAYOUT de plotar_matriz_confusao e num comentário de validar_cnn.
        # Nenhum campo de resultado mudou (a guarda compara todos os outros e
        # eles reproduzem), e a auditoria de isolamento rodou de novo sobre o
        # código NOVO na regeneração, sem achar o conjunto lacrado.
        "isolamento_teste.arquivos_auditados.src/models/avaliacao.py.md5": {
            "antes": "b803cd4ce89346396b910650142ea301",
            "depois": "0c0e4cabee55d891ba4c2ec84c89494c",
            "explicacao": (
                "commit a37eed3 (fix do título cortado): plotar_matriz_confusao "
                "passou a usar fig.suptitle + textwrap.fill + "
                "layout='constrained', com o parâmetro opcional largura_titulo. "
                "Nenhuma linha de avaliar, calcular_eer, selecionar_limiar, "
                "aplicar_limiar ou predizer_rf mudou — `git diff fbdba7d a37eed3 "
                "-- src/models/avaliacao.py` toca só a função de figura. O md5 "
                "antigo descreve o código que rodou o B4.7; o novo, o código "
                "que rodou a regeneração, e os resultados das duas execuções "
                "são idênticos."),
        },
        "isolamento_teste.arquivos_auditados.src/models/validar_cnn.py.md5": {
            "antes": "c8a3c9c8bde15393ba4732f0b2e100b4",
            "depois": "de4980f42c1d822c3968afa4dd9d1e7d",
            "explicacao": (
                "commit a37eed3: só o COMENTÁRIO acima da chamada de "
                "plotar_matriz_confusao foi reescrito (o motivo do título curto "
                "deixou de ser técnico). Nenhuma linha executável mudou — `git "
                "diff fbdba7d a37eed3 -- src/models/validar_cnn.py` tem 4 "
                "linhas removidas e 7 acrescentadas, todas de comentário."),
        },
    },
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

# ---- Grupo CNN (B5.2) -------------------------------------------------------
NOME_REEXECUCAO = "cnn_final_principal_reexecucao"
# arquivo de referência (em _pre_revisao/) -> arquivo regenerado (em DIR_MET)
ARQUIVOS_CNN = {
    "cnn_final_principal.json": f"{NOME_REEXECUCAO}.json",
    "diagnostico_por_ataque_resumo.json": "diagnostico_por_ataque_resumo.json",
    "diagnostico_por_codec_resumo.json": "diagnostico_por_codec_resumo.json",
    "comparacao_estatistica.json": "comparacao_estatistica.json",
}
# Publicados que a regeneração NÃO pode ter tocado: byte a byte iguais à
# referência. É o que prova que a verificação não trocou número citado.
PUBLICADOS_INTOCADOS = ["cnn_final_principal.json"]
CAMPOS_EXIGIDOS_CNN = [
    "acuracia", "f1_macro", "eer", "limiar", "roc_auc_validacao",
    "matriz_confusao", "selecao_limiar.limiar", "selecao_limiar.n_candidatos",
    "saturacao_do_score.n_valores_distintos", "por_modelo",
    "bootstrap_pareado",
]
NAO_REGENERAVEIS = {
    "teste_lacrado.json": (
        "execução única do teste lacrado: reexecutar é o que o B5.1 proíbe. A "
        "reprodução dele está no próprio artefato — `conferencia_pre_teste` "
        "(os quatro modelos reproduzem exatamente os JSONs de validação antes "
        "de o teste ser aberto) — e no ensaio na validação (delta = 0)."),
    "refit_cnn.json / cnn_definida.json / cnn_baseline.json": (
        "artefatos de TREINO da CNN: regenerá-los re-treinaria a rede e "
        "sobrescreveria os .pt, o que o B5 proíbe (nenhuma mudança nos "
        "modelos). O determinismo do treino está registrado em "
        "refit_cnn.json -> determinismo."),
}

GRUPOS = {
    "bloco3": {
        "pares": {n: n for n in ARQUIVOS},
        "exigidos": CAMPOS_EXIGIDOS,
        "saida": "reproducao_bloco3.json",
    },
    "cnn": {
        "pares": ARQUIVOS_CNN,
        "exigidos": CAMPOS_EXIGIDOS_CNN,
        "saida": "reproducao_cnn.json",
    },
}


def regenerar_cnn() -> None:
    """Re-executa o B4.7 pelo PRÓPRIO validar_cnn, gravando com outro nome.

    `patch.object` troca `NOME` só durante a chamada; nada no módulo é editado
    (o md5 de validar_cnn.py é evidência do isolamento do teste no B4.7). A
    matriz de confusão que o B4.7 grava junto é subproduto idêntico à
    publicada, e é removida.
    """
    from src.models import validar_cnn

    with patch.object(validar_cnn, "NOME", NOME_REEXECUCAO):
        validar_cnn.main()
    fig = RAIZ / "results" / "figuras" / f"matriz_confusao_{NOME_REEXECUCAO}.png"
    fig.unlink(missing_ok=True)


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


def comparar(nome: str, nome_novo: str | None = None,
             exigidos: list | None = None) -> dict:
    """Compara um JSON (ou a sua regeneração `nome_novo`) contra a cópia em _pre_revisao/."""
    novo_p, antes_p = DIR_MET / (nome_novo or nome), DIR_ANTES / nome
    exigidos = CAMPOS_EXIGIDOS if exigidos is None else exigidos
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
    conferidos = [c for c in exigidos
                  if any(p == c or p.startswith(c + ".") for p in comuns)]
    faltando = [c for c in exigidos
                if c not in conferidos and any(
                    p == c or p.startswith(c + ".") for p in set(antes) | set(novo))]

    ok = not divergencias and not removidos and not faltando
    return {
        "arquivo": nome,
        **({"regenerado_em": nome_novo} if nome_novo and nome_novo != nome else {}),
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


def _md5(caminho: Path) -> str:
    return hashlib.md5(caminho.read_bytes()).hexdigest()


def conferir_publicados_intocados() -> list:
    """O publicado continua byte a byte igual à referência tirada antes."""
    return [{"arquivo": n,
             "md5_publicado": _md5(DIR_MET / n),
             "md5_referencia": _md5(DIR_ANTES / n),
             "intocado": _md5(DIR_MET / n) == _md5(DIR_ANTES / n)}
            for n in PUBLICADOS_INTOCADOS]


def imprimir(relatorios: list) -> bool:
    """Imprime o relatório de um grupo; devolve True se algo falhou."""
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
    return falhou


def _declarados(tabela: dict, pares: dict) -> dict:
    """Só as declarações dos arquivos do grupo — cada evidência lista as suas."""
    return {k: v for k, v in tabela.items() if k in pares}


def registro_bloco3(relatorios: list, g: dict) -> dict:
    return {
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
        "campos_exigidos": g["exigidos"],
        "reestruturacoes_declaradas": _declarados(REESTRUTURACOES_DECLARADAS, g["pares"]),
        "divergencias_explicadas_declaradas": _declarados(DIVERGENCIAS_EXPLICADAS, g["pares"]),
        "remocoes_declaradas": _declarados(REMOCOES_DECLARADAS, g["pares"]),
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


def variacao_dos_tempos() -> dict:
    """Tempos do B4.7 publicado x regenerado: IGNORADOS pela guarda, mas medidos.

    Não é comparação (relógio varia por definição); é o registro de QUANTO ele
    varia entre sessões — o que diz com que precisão um tempo absoluto pode
    ser citado no texto.
    """
    with open(DIR_ANTES / "cnn_final_principal.json", encoding="utf-8") as f:
        antes = json.load(f)["tempos_inferencia"]
    with open(DIR_MET / f"{NOME_REEXECUCAO}.json", encoding="utf-8") as f:
        depois = json.load(f)["tempos_inferencia"]
    saida = {}
    for cen in ("gpu", "cpu"):
        for campo, chave in (("latencia_ms", "mediana"),
                             ("throughput", "ms_por_audio_mediana")):
            a, d = antes[cen][campo][chave], depois[cen][campo][chave]
            saida[f"{cen}.{campo}.{chave}"] = {
                "publicado": a, "regenerado": d,
                "variacao_relativa": round(d / a - 1, 4)}
    return {
        "o_que_e": ("registro, não verificação: os tempos são ignorados pela "
                    "guarda; aqui só se mede quanto variaram entre a sessão do "
                    "B4.7 e a da regeneração"),
        "por_campo": saida,
        "maior_variacao_relativa_abs": max(abs(v["variacao_relativa"])
                                           for v in saida.values()),
    }


def registro_cnn(relatorios: list, g: dict, intocados: list) -> dict:
    return {
        "verificacao": "reproducao_cnn",
        "data": date.today().isoformat(),
        "resultado": "REPRODUZIU",
        "pergunta": ("regenerar os artefatos da CNN que dependem só de INFERÊNCIA "
                     "(B4.7, diagnósticos por ataque e por codec, bootstrap "
                     "pareado do B5.2) reproduz os resultados publicados?"),
        "por_que_tem_de_reproduzir": (
            "a inferência da CNN roda com fixar_seeds_torch(estrito=True), "
            "cudnn.deterministic, TF32 desligado (fixar_precisao_fp32_cnn) e "
            "shuffle=False; o bootstrap tem semente fixa. A soma paralela do "
            "cuDNN é a fonte de não-determinismo que o marco manda vigiar — se "
            "n_candidatos variar, é ela"),
        "como_o_b47_foi_regenerado": (
            "validar_cnn.main() com NOME trocado para "
            f"'{NOME_REEXECUCAO}' (patch.object): o MESMO código do B4.7, "
            "gravando ao lado do publicado em vez de por cima — os tempos "
            "publicados, citados no texto, não são re-medidos por esta "
            "verificação. Os tempos da regeneração estão no arquivo regenerado "
            "e NÃO são os oficiais"),
        "publicados_intocados": intocados,
        "variacao_dos_tempos_entre_sessoes": variacao_dos_tempos(),
        "referencia": ("results/metricas/_pre_revisao/ (cópias tiradas em "
                       "23/09/2026, antes da regeneração; as do B4.7 idênticas "
                       "às publicadas no commit fbdba7d)"),
        "nao_regeneraveis": NAO_REGENERAVEIS,
        "campos_exigidos": g["exigidos"],
        "divergencias_explicadas_declaradas": _declarados(DIVERGENCIAS_EXPLICADAS, g["pares"]),
        "chaves_ignoradas": sorted(CHAVES_IGNORADAS),
        "tolerancia_float": TOL_FLOAT,
        "arquivos": relatorios,
        "ambiente": {
            "python": platform.python_version(),
            "sistema": f"{platform.system()} {platform.release()}",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regenerar-cnn", action="store_true",
                    help=f"roda validar_cnn gravando {NOME_REEXECUCAO}.json "
                         "antes de comparar (~7 min; não toca no publicado)")
    args = ap.parse_args()
    if args.regenerar_cnn:
        print(f"Regenerando o B4.7 em {NOME_REEXECUCAO}.json ...")
        regenerar_cnn()

    print("=" * 74)
    print("GUARDA DE REPRODUÇÃO — Bloco 3 (R2.3) + CNN (B5.2)")
    print("=" * 74)
    print(f"referência: {DIR_ANTES.relative_to(RAIZ)}")
    print(f"ignorados : {', '.join(sorted(CHAVES_IGNORADAS))}")
    print("            (medidas de relógio variam por definição)")

    falhou, registros = False, {}
    for grupo, g in GRUPOS.items():
        print(f"\n--- grupo {grupo} ---")
        relatorios = [comparar(ref, novo, g["exigidos"])
                      for ref, novo in g["pares"].items()]
        falhou |= imprimir(relatorios)
        if grupo == "cnn":
            intocados = conferir_publicados_intocados()
            for i in intocados:
                print(f"{'[OK ]' if i['intocado'] else '[!! ]'} {i['arquivo']}: "
                      f"publicado {'intocado' if i['intocado'] else 'ALTERADO'} "
                      "(md5 igual à referência)")
                falhou |= not i["intocado"]
            for n, motivo in NAO_REGENERAVEIS.items():
                print(f"[-- ] {n}: fora da guarda — {motivo[:70]}...")
            registros[grupo] = registro_cnn(relatorios, g, intocados)
        else:
            registros[grupo] = registro_bloco3(relatorios, g)

    print()
    if falhou:
        print("=" * 74)
        print("RESULTADO: NÃO REPRODUZIU. PARE — não commite.")
        print("Divergência de reprodução é ACHADO, não obstáculo a contornar:")
        print("os pipelines são determinísticos, então uma diferença aqui")
        print("aponta mudança real de comportamento e tem de ser explicada")
        print("antes de qualquer número ser citado.")
        print("=" * 74)
        return 1

    print("=" * 74)
    print("RESULTADO: REPRODUZIU em todos os artefatos comparados.")
    for grupo, g in GRUPOS.items():
        caminho = DIR_MET / g["saida"]
        with open(caminho, "w", encoding="utf-8") as f:
            json.dump(registros[grupo], f, indent=2, ensure_ascii=False)
        print(f"Evidência [{grupo}] gravada em {caminho.relative_to(RAIZ)}")
    print("São estes os arquivos a citar na resposta de banca sobre")
    print("reprodutibilidade.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
