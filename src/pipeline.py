import time
from dataclasses import dataclass, field
from threading import Event

from src.chunking import MODELO_PADRAO, Chunk, chunk_por_paginas, get_model
from src.geracao import Checagem, gerar_questoes
from src.ingest import extrair_paginas, validar_texto
from src.llm import get_modelo
from src.retrieval import Indice, Trecho
from src.schema import Questao


@dataclass
class Simulado:
    questoes: list[Questao] = field(default_factory=list)
    trechos: list[Trecho] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    avisos: list[str] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)
    descartadas: int = 0
    n_paginas: int = 0
    paginas_filtro: tuple[int, int] | None = None
    provedor: str | None = None
    modelo: str = ""
    cancelado: bool = False
    duracao: float = 0.0
    checagens: list[Checagem] = field(default_factory=list)

    @property
    def n_chunks(self) -> int:
        return len(self.chunks)


def _avisar(etapa, mensagem: str) -> None:
    if etapa:
        etapa(mensagem)


def filtrar_chunks_por_paginas(
    chunks: list[Chunk], pagina_inicio: int, pagina_fim: int
) -> list[Chunk]:
    return [
        chunk
        for chunk in chunks
        if any(pagina_inicio <= p <= pagina_fim for p in chunk.paginas)
    ]


def gerar_simulado(
    fonte,
    n: int,
    tema: str = "",
    versao: str = "v2",
    temperatura: float = 0.2,
    modelo: str | None = None,
    provedor: str | None = None,
    progresso=None,
    etapa=None,
    max_tokens: int = 250,
    sobreposicao: int = 40,
    top_k: int = 5,
    pagina_inicio: int | None = None,
    pagina_fim: int | None = None,
    cancelar: Event | None = None,
) -> Simulado:
    """Executa o fluxo completo: PDF -> chunks -> embeddings -> RAG -> questões."""
    inicio = time.perf_counter()
    modelo_uso = modelo or get_modelo(provedor)

    _avisar(etapa, "Lendo o PDF...")
    paginas = extrair_paginas(fonte)
    validar_texto(paginas)
    _avisar(etapa, f"{len(paginas)} páginas lidas")

    total_paginas = len(paginas)
    filtro_ativo = pagina_inicio is not None or pagina_fim is not None
    primeiro = 1 if pagina_inicio is None else int(pagina_inicio)
    ultimo = total_paginas if pagina_fim is None else int(pagina_fim)
    if not 1 <= primeiro <= ultimo <= total_paginas:
        raise ValueError(
            f"Intervalo de páginas inválido: {primeiro}-{ultimo}. "
            f"O PDF tem {total_paginas} páginas."
        )

    _avisar(etapa, "Gerando embeddings e recuperando trechos...")
    model = get_model(MODELO_PADRAO)
    chunks, avisos = chunk_por_paginas(paginas, max_tokens, sobreposicao)
    if not chunks:
        raise ValueError("Nenhum texto aproveitável foi extraído do PDF.")

    if filtro_ativo:
        _avisar(etapa, f"Filtrando páginas {primeiro}-{ultimo}...")
        chunks = filtrar_chunks_por_paginas(chunks, primeiro, ultimo)
        if not chunks:
            raise ValueError(
                f"Nenhum texto aproveitável nas páginas {primeiro}-{ultimo}."
            )

    indice = Indice(model, chunks)
    if tema.strip():
        trechos = indice.buscar(tema, top_k=min(top_k, len(chunks)))
    else:
        trechos = indice.mais_central()[: min(top_k, len(chunks))]
    if filtro_ativo:
        _avisar(
            etapa,
            f"{len(chunks)} chunks nas páginas {primeiro}-{ultimo}, "
            f"{len(trechos)} trechos usados",
        )
    else:
        _avisar(etapa, f"{len(chunks)} chunks, {len(trechos)} trechos usados")

    _avisar(etapa, "Gerando e validando questões...")
    res = gerar_questoes(
        [t.chunk for t in trechos],
        n,
        versao=versao,
        temperatura=temperatura,
        modelo=modelo_uso,
        progresso=progresso,
        provedor=provedor,
        cancelar=cancelar,
    )

    return Simulado(
        questoes=res.questoes,
        trechos=trechos,
        chunks=chunks,
        avisos=avisos,
        logs=res.logs,
        descartadas=res.descartadas,
        n_paginas=len(paginas),
        paginas_filtro=(primeiro, ultimo) if filtro_ativo else None,
        provedor=provedor,
        modelo=modelo_uso,
        cancelado=res.cancelado,
        duracao=time.perf_counter() - inicio,
        checagens=res.checagens,
    )
