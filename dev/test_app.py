import queue
import sys
import threading
from pathlib import Path

sys.path.insert(0, ".")

from streamlit.testing.v1 import AppTest

from src.chunking import Chunk
from src.geracao import Checagem
from src.pipeline import Simulado
from src.retrieval import Trecho
from src.schema import Questao

RAIZ = Path(__file__).resolve().parent.parent

CHUNK = Chunk(
    "A clorofila fica nos cloroplastos e absorve a luz solar.",
    [1],
    12,
)
QUESTAO = Questao(
    enunciado="Onde fica a clorofila que absorve a luz solar?",
    alternativas=["Mitocôndria", "Cloroplasto", "Ribossomo", "Lisossomo"],
    correta=1,
    justificativa="A clorofila está nos cloroplastos.",
    evidencia="A clorofila fica nos cloroplastos",
    pagina=1,
)


def _app() -> AppTest:
    return AppTest.from_file(RAIZ / "app.py")


def _job(id: int = 1, progresso: float = 0.0) -> dict:
    return {
        "id": id,
        "fila": queue.Queue(),
        "cancelar": threading.Event(),
        "parando": False,
        "etapa": "Gerando e validando questões...",
        "progresso": progresso,
    }


def testar_render_inicial() -> None:
    at = _app()
    at.run(timeout=60)
    assert not at.exception, at.exception
    assert len(at.tabs) == 3
    assert at.title
    print("[OK] app inicial renderizado")


def testar_finalizacao_via_fila() -> None:
    sim = Simulado(
        questoes=[QUESTAO],
        trechos=[Trecho(chunk=CHUNK, score=0.9)],
        chunks=[CHUNK],
        logs=["Q1 tentativa 1: ok"],
        n_paginas=1,
        provedor="nvidia",
        modelo="teste",
        cancelado=True,
        duracao=1.23,
        checagens=[
            Checagem(1, 1, "Chamada ao LLM", "ok"),
            Checagem(1, 1, "Formato JSON", "ok"),
            Checagem(1, 1, "Estrutura, alternativas e índice", "ok"),
            Checagem(1, 1, "Evidência no contexto", "ok"),
            Checagem(1, 1, "Página da evidência", "corrigida", "de 2 para 1"),
        ],
    )
    job = _job()
    job["fila"].put(("ok", sim))
    at = _app()
    at.session_state["geracao"] = job
    at.run(timeout=60)
    assert not at.exception, at.exception
    assert len(at.session_state["questoes"]) == 1
    assert at.session_state["duracao"] == 1.23
    assert at.session_state["cancelado"] is True
    assert len(at.session_state["checagens"]) == 5
    assert "Testes executados" in [s.value for s in at.subheader]
    print("[OK] fila, fragmento e finalização")


def testar_botao_parar() -> None:
    job = _job(id=2, progresso=0.33)
    at = _app()
    at.session_state["geracao"] = job
    at.run(timeout=60)
    assert not at.exception, at.exception
    botao = next(b for b in at.button if b.key == "parar_geracao_2")
    botao.click().run(timeout=60)
    assert not at.exception, at.exception
    assert job["cancelar"].is_set()
    assert job["parando"] is True
    print("[OK] botão Parar aciona o evento")


def testar_toggle_catalogo() -> None:
    at = _app()
    at.run(timeout=60)
    assert not at.exception, at.exception
    provedor = next(s.value for s in at.selectbox if s.label == "Provedor")
    at.session_state["catalogo"] = [
        "modelo-completo-a",
        "modelo-completo-b",
    ]
    at.session_state["catalogo_provedor"] = provedor
    at.run(timeout=60)
    assert not at.exception, at.exception
    assert any(b.label == "Usar modelos conhecidos" for b in at.button)
    assert not any(
        b.label == "Carregar catálogo completo" for b in at.button
    )
    botao = next(b for b in at.button if b.label == "Usar modelos conhecidos")
    botao.click().run(timeout=60)
    assert not at.exception, at.exception
    assert at.session_state["catalogo"] is None
    assert at.session_state["catalogo_provedor"] is None
    assert any(b.label == "Carregar catálogo completo" for b in at.button)
    print("[OK] catálogo alterna entre completo e modelos conhecidos")


def main() -> None:
    testar_render_inicial()
    testar_finalizacao_via_fila()
    testar_botao_parar()
    testar_toggle_catalogo()


if __name__ == "__main__":
    main()
