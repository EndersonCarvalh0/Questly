from dataclasses import dataclass

from sentence_transformers import SentenceTransformer

MODELO_PADRAO = "intfloat/multilingual-e5-small"
PREFIXO_PASSAGE = "passage: "
PREFIXO_QUERY = "query: "

_re_cache: dict[str, SentenceTransformer] = {}


def get_model(nome: str = MODELO_PADRAO) -> SentenceTransformer:
    if nome not in _re_cache:
        _re_cache[nome] = SentenceTransformer(nome)
    return _re_cache[nome]


@dataclass
class Chunk:
    texto: str
    paginas: list[int]
    n_tokens: int

    @property
    def faixa_paginas(self) -> str:
        p = self.paginas
        if not p:
            return "?"
        if len(p) == 1:
            return f"p. {p[0]}"
        return f"p. {p[0]}-{p[-1]}"


def contar_tokens(model: SentenceTransformer, texto: str, prefixo: str = "") -> int:
    # conta com o prefixo e os tokens especiais, como o modelo realmente recebe
    ids = model.tokenizer.encode(prefixo + texto, add_special_tokens=True)
    return len(ids)


def _dividir_frases(texto: str) -> list[str]:
    partes: list[str] = []
    for bloco in texto.split("\n"):
        bloco = bloco.strip()
        if not bloco:
            continue
        partes.extend(f.strip() for f in bloco.replace("\n", " ").split(". ") if f.strip())
    return [f if f.endswith(".") else f + "." for f in partes]


def chunk_por_paginas(
    paginas: list[str],
    max_tokens: int,
    sobreposicao: int,
    nome_modelo: str = MODELO_PADRAO,
    prefixo: str = PREFIXO_PASSAGE,
) -> tuple[list[Chunk], list[str]]:
    """Divide o material em chunks que respeitam max_seq_length.

    Acumula frases (sem cortar no meio) até o limite de tokens e mantém
    a lista de páginas de cada chunk. Devolve (chunks, avisos).
    """
    model = get_model(nome_modelo)
    limit = min(max_tokens, model.max_seq_length - contar_tokens(model, "", prefixo))
    avisos: list[str] = []
    chunks: list[Chunk] = []

    for pagina_num, texto in enumerate(paginas, start=1):
        if not texto:
            continue
        frases = _dividir_frases(texto)
        atual: list[str] = []
        atual_tokens = 0

        for frase in frases:
            t = contar_tokens(model, frase, prefixo)
            if t > limit:
                if atual:
                    chunks.append(_fechar(atual, atual_tokens, [pagina_num], model, prefixo))
                    atual, atual_tokens = [], 0
                avisos.append(
                    f"p. {pagina_num}: uma frase excede o limite de {limit} tokens "
                    "e foi mantida isolada (pode ser truncada pelo modelo)."
                )
                chunks.append(Chunk(frase, [pagina_num], t))
                continue

            if atual and atual_tokens + t > limit:
                chunks.append(
                    _fechar(atual, atual_tokens, [pagina_num], model, prefixo)
                )
                atual, atual_tokens = _sobrepor(atual, model, prefixo, sobreposicao)
            atual.append(frase)
            atual_tokens += t

        if atual:
            chunks.append(_fechar(atual, atual_tokens, [pagina_num], model, prefixo))

    checagem = _checar_limite(chunks, model, prefixo)
    avisos.extend(checagem)
    return chunks, avisos


def _fechar(frases, tokens, paginas, model, prefixo) -> Chunk:
    texto = " ".join(frases)
    return Chunk(texto=texto, paginas=paginas, n_tokens=contar_tokens(model, texto, prefixo))


def _sobrepor(frases, model, prefixo, sobreposicao) -> tuple[list[str], int]:
    if sobreposicao <= 0:
        return [], 0
    mantidas: list[str] = []
    total = 0
    for frase in reversed(frases):
        t = contar_tokens(model, frase, prefixo)
        if total + t > sobreposicao:
            break
        mantidas.insert(0, frase)
        total += t
    return mantidas, total


def _checar_limite(chunks: list[Chunk], model, prefixo) -> list[str]:
    avisos: list[str] = []
    limite = model.max_seq_length
    for c in chunks:
        if c.n_tokens > limite:
            avisos.append(
                f"{c.faixa_paginas}: chunk com {c.n_tokens} tokens excede "
                f"max_seq_length={limite} — será truncado silenciosamente."
            )
    return avisos
