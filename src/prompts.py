import json

from src.chunking import Chunk

SISTEMA = (
    "Você é um professor que cria questões de múltipla escolha a partir de "
    "material de aula. Responda sempre em português do Brasil e SOMENTE com JSON."
)

FORMATO = {
    "enunciado": "string",
    "alternativas": ["string", "string", "string", "string"],
    "correta": "int (0 a 3, índice da alternativa correta)",
    "justificativa": "string explicando por que a correta está certa",
    "evidencia": "trecho LITERAL copiado do material",
    "pagina": "int (página onde está a evidência)",
}


def montar_contexto(trechos: list[Chunk]) -> str:
    blocos = []
    for c in trechos:
        pagina = c.paginas[0] if c.paginas else "?"
        blocos.append(f"[página {pagina}]\n{c.texto}")
    return "\n\n".join(blocos)


def prompt_v1(contexto: str, evita: list[str] | None = None) -> list[dict]:
    """Versão simples: só o contexto e o formato."""
    user = (
        "Crie UMA questão de múltipla escolha sobre o material abaixo.\n\n"
        f"MATERIAL:\n{contexto}\n\n"
        f"Formato do JSON: {json.dumps(FORMATO, ensure_ascii=False)}"
    )
    return [
        {"role": "system", "content": SISTEMA},
        {"role": "user", "content": user},
    ]


def prompt_v2(contexto: str, evita: list[str] | None = None) -> list[dict]:
    """Versão com regras, formato estrito e exemplo."""
    evitar = ""
    if evita:
        lista = "\n".join(f"- {e}" for e in evita)
        evitar = f"\n\nNÃO repita o tema destas questões já criadas:\n{lista}"

    exemplo = (
        '{"enunciado": "Qual organela realiza a fotossíntese?", '
        '"alternativas": ["Mitocôndria", "Cloroplasto", "Ribossomo", "Lisossomo"], '
        '"correta": 1, '
        '"justificativa": "A clorofila fica nos cloroplastos, onde ocorre a fotossíntese.", '
        '"evidencia": "A clorofila nos cloroplastos absorve a luz solar.", '
        '"pagina": 1}'
    )

    user = (
        "Crie UMA questão de múltipla escolha usando EXCLUSIVAMENTE o material abaixo. "
        "A resposta deve estar sustentada pelo material.\n\n"
        f"MATERIAL:\n{contexto}\n\n"
        "REGRAS:\n"
        "- exatamente 4 alternativas distintas e plausíveis\n"
        "- apenas 1 correta; as outras 3 devem ser incorretas\n"
        "- 'correta' é o índice (0 a 3) da alternativa correta\n"
        "- 'evidencia' deve ser um trecho copiado LITERALMENTE do material\n"
        "- 'pagina' deve ser a página indicada no material onde está a evidência\n"
        "- não invente informação que não esteja no material\n"
        "- sem markdown, sem comentários: apenas o objeto JSON\n" + evitar + "\n\n"
        f"EXEMPLO DE FORMATO:\n{exemplo}\n\n"
        "Agora responda com o JSON da sua questão."
    )
    return [
        {"role": "system", "content": SISTEMA},
        {"role": "user", "content": user},
    ]
