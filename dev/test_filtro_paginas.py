import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, ".")

import src.pipeline as pipeline
from src.chunking import Chunk
from src.ingest import contar_paginas
from src.pipeline import filtrar_chunks_por_paginas
from src.retrieval import Trecho

RAIZ = Path(__file__).resolve().parent.parent
PDF = RAIZ / "eval" / "pdfs" / "Aula01_ConceitosBasicos.pdf"


def testar_filtro_de_chunks() -> None:
    chunks = [
        Chunk("página 1", [1], 5),
        Chunk("páginas 2 e 3", [2, 3], 5),
        Chunk("página 4", [4], 5),
        Chunk("sem página", [], 5),
    ]
    filtrados = filtrar_chunks_por_paginas(chunks, 2, 3)
    assert [c.texto for c in filtrados] == ["páginas 2 e 3"]
    assert filtrar_chunks_por_paginas(chunks, 5, 6) == []
    print("[OK] chunks filtrados pelo intervalo de páginas")


def testar_pdf_real() -> None:
    assert contar_paginas(PDF) == 14
    print("[OK] contagem de páginas do PDF de aula")


def testar_pipeline_com_filtro() -> None:
    chunks = [
        Chunk("página 1", [1], 5),
        Chunk("página 2", [2], 5),
        Chunk("página 3", [3], 5),
    ]

    class IndiceFalso:
        def __init__(self, model, chunks):
            self.chunks = chunks

        def buscar(self, consulta, top_k=3):
            return [Trecho(c, 0.9) for c in self.chunks[:top_k]]

        def mais_central(self):
            return [Trecho(c, 0.9) for c in self.chunks]

    def gerar_falso(*args, **kwargs):
        return SimpleNamespace(
            questoes=[],
            logs=[],
            descartadas=0,
            cancelado=False,
            checagens=[],
        )

    with (
        patch.object(pipeline, "extrair_paginas", return_value=["a", "b", "c"]),
        patch.object(pipeline, "validar_texto"),
        patch.object(pipeline, "get_modelo", return_value="modelo-teste"),
        patch.object(pipeline, "get_model", return_value=object()),
        patch.object(
            pipeline, "chunk_por_paginas", return_value=(chunks, [])
        ),
        patch.object(pipeline, "Indice", IndiceFalso),
        patch.object(pipeline, "gerar_questoes", side_effect=gerar_falso),
    ):
        sim = pipeline.gerar_simulado(
            "pdf-falso",
            1,
            modelo="modelo-teste",
            pagina_inicio=2,
            pagina_fim=3,
        )

    assert [c.texto for c in sim.chunks] == ["página 2", "página 3"]
    assert [t.chunk.texto for t in sim.trechos] == ["página 2", "página 3"]
    assert sim.paginas_filtro == (2, 3)
    assert sim.n_paginas == 3
    print("[OK] pipeline aplica o filtro antes do índice")


def main() -> None:
    testar_filtro_de_chunks()
    testar_pdf_real()
    testar_pipeline_com_filtro()
    print("[OK] filtro por páginas")


if __name__ == "__main__":
    main()
