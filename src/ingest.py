import re

import pymupdf

MIN_CHARS = 200
_RE_ESPACOS = re.compile(r"[ \t]+")
_RE_LINHAS = re.compile(r"\n{3,}")


def limpar(texto: str) -> str:
    texto = texto.replace("\u00ad", "")
    texto = _RE_ESPACOS.sub(" ", texto)
    texto = _RE_LINHAS.sub("\n\n", texto)
    return texto.strip()


def extrair_paginas(fonte) -> list[str]:
    """Devolve o texto limpo de cada página (índice 0 = página 1).

    `fonte` pode ser um caminho ou um bytes/bytearray com o conteúdo do PDF.
    """
    if isinstance(fonte, (bytes, bytearray)):
        doc = pymupdf.open(stream=fonte, filetype="pdf")
    else:
        doc = pymupdf.open(fonte)
    try:
        if doc.page_count == 0:
            raise ValueError("O PDF não tem páginas.")
        return [limpar(pagina.get_text("text")) for pagina in doc]
    finally:
        doc.close()


def validar_texto(paginas: list[str]) -> None:
    total = sum(len(p) for p in paginas)
    if total < MIN_CHARS:
        raise ValueError(
            f"O PDF tem apenas {total} caracteres de texto extraído "
            f"(mínimo {MIN_CHARS}). Provavelmente é um PDF digitalizado "
            "(imagem) ou vazio. Envie um PDF com texto selecionável."
        )
