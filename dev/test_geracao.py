import json
import sys

sys.path.insert(0, ".")

from src import geracao
from src.chunking import Chunk

CHUNK = Chunk(
    "A clorofila fica nos cloroplastos e absorve a luz solar.",
    [1],
    12,
)
CONTEXTO = CHUNK.texto
QUESTAO = {
    "enunciado": "Onde fica a clorofila que absorve a luz solar?",
    "alternativas": ["Mitocôndria", "Cloroplasto", "Ribossomo", "Lisossomo"],
    "correta": 1,
    "justificativa": "A clorofila está nos cloroplastos, onde ocorre a fotossíntese.",
    "evidencia": "A clorofila fica nos cloroplastos",
    "pagina": 2,
}


class ErroHTTP(Exception):
    def __init__(self, status_code: int):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def testar_pagina_corrigida() -> None:
    checagens = []
    original = geracao.chat
    geracao.chat = lambda *args, **kwargs: json.dumps(QUESTAO, ensure_ascii=False)
    try:
        questao, motivo = geracao._gerar_uma(
            CONTEXTO,
            "v2",
            0.2,
            [],
            provedor="nvidia",
            trechos=[CHUNK],
            checagens=checagens,
        )
    finally:
        geracao.chat = original

    assert questao is not None
    assert questao.pagina == 1
    assert "página corrigida" in motivo
    status = {titulo: status for titulo, status, _ in checagens}
    detalhe = {titulo: detalhe for titulo, _, detalhe in checagens}
    assert status[geracao.CHECAGEM_LLM] == "ok"
    assert status[geracao.CHECAGEM_JSON] == "ok"
    assert status[geracao.CHECAGEM_ESTRUTURA] == "ok"
    assert status[geracao.CHECAGEM_EVIDENCIA] == "ok"
    assert status[geracao.CHECAGEM_PAGINA] == "corrigida"
    assert detalhe[geracao.CHECAGEM_PAGINA] == "de 2 para 1"


def testar_checagens_json_invalido() -> None:
    checagens = []
    original = geracao.chat
    geracao.chat = lambda *args, **kwargs: "resposta sem JSON"
    try:
        questao, _ = geracao._gerar_uma(
            CONTEXTO,
            "v2",
            0.2,
            [],
            provedor="nvidia",
            trechos=[CHUNK],
            checagens=checagens,
        )
    finally:
        geracao.chat = original

    assert questao is None
    status = {titulo: status for titulo, status, _ in checagens}
    assert status[geracao.CHECAGEM_LLM] == "ok"
    assert status[geracao.CHECAGEM_JSON] == "falhou"
    assert status[geracao.CHECAGEM_ESTRUTURA] == "nao_executado"
    assert status[geracao.CHECAGEM_EVIDENCIA] == "nao_executado"
    assert status[geracao.CHECAGEM_PAGINA] == "nao_executado"


def testar_checagens_evidencia() -> None:
    invalida = dict(
        QUESTAO,
        evidencia="texto que não existe no material",
    )
    checagens = []
    original = geracao.chat
    geracao.chat = lambda *args, **kwargs: json.dumps(
        invalida, ensure_ascii=False
    )
    try:
        questao, _ = geracao._gerar_uma(
            CONTEXTO,
            "v2",
            0.2,
            [],
            provedor="nvidia",
            trechos=[CHUNK],
            checagens=checagens,
        )
    finally:
        geracao.chat = original

    assert questao is None
    status = {titulo: status for titulo, status, _ in checagens}
    assert status[geracao.CHECAGEM_EVIDENCIA] == "falhou"
    assert status[geracao.CHECAGEM_PAGINA] == "nao_executado"


def testar_checagens_em_gerar_questoes() -> None:
    def chat_fake(*args, **kwargs):
        return json.dumps(QUESTAO, ensure_ascii=False)

    chat_fake.ultimo_usage = {}
    original = geracao.chat
    geracao.chat = chat_fake
    try:
        res = geracao.gerar_questoes([CHUNK], 1, provedor="nvidia")
    finally:
        geracao.chat = original

    assert len(res.checagens) == 5
    assert all(c.questao == 1 and c.tentativa == 1 for c in res.checagens)
    assert res.checagens[0].titulo == geracao.CHECAGEM_LLM
    assert res.checagens[-1].status == "corrigida"


def testar_checagens_em_retry() -> None:
    respostas = iter(["sem JSON", json.dumps(QUESTAO, ensure_ascii=False)])

    def chat_fake(*args, **kwargs):
        chat_fake.ultimo_usage = {}
        return next(respostas)

    original = geracao.chat
    geracao.chat = chat_fake
    try:
        res = geracao.gerar_questoes([CHUNK], 1, provedor="nvidia")
    finally:
        geracao.chat = original

    assert len(res.questoes) == 1
    assert len(res.checagens) == 10
    primeira = {
        c.titulo: c.status
        for c in res.checagens
        if c.tentativa == 1
    }
    segunda = {
        c.titulo: c.status
        for c in res.checagens
        if c.tentativa == 2
    }
    assert primeira[geracao.CHECAGEM_JSON] == "falhou"
    assert primeira[geracao.CHECAGEM_EVIDENCIA] == "nao_executado"
    assert segunda[geracao.CHECAGEM_JSON] == "ok"
    assert segunda[geracao.CHECAGEM_PAGINA] == "corrigida"


def testar_erro_fatal() -> None:
    original = geracao.chat

    def falhar(*args, **kwargs):
        raise ErroHTTP(401)

    geracao.chat = falhar
    try:
        try:
            geracao._gerar_uma(CONTEXTO, "v2", 0.2, [], provedor="nvidia")
        except RuntimeError as e:
            assert "401" in str(e)
        else:
            raise AssertionError("401 deveria ser fatal")
    finally:
        geracao.chat = original


def testar_limite_429() -> None:
    original = geracao.chat

    def falhar(*args, **kwargs):
        raise ErroHTTP(429)

    geracao.chat = falhar
    try:
        try:
            geracao._gerar_uma(CONTEXTO, "v2", 0.2, [], provedor="groq")
        except RuntimeError as e:
            assert "429" in str(e) or "Limite" in str(e)
        else:
            raise AssertionError("429 deveria parar o provedor experimental")
    finally:
        geracao.chat = original


def main() -> None:
    testar_pagina_corrigida()
    testar_checagens_json_invalido()
    testar_checagens_evidencia()
    testar_checagens_em_gerar_questoes()
    testar_checagens_em_retry()
    testar_erro_fatal()
    testar_limite_429()
    print("[OK] geração, página, checagens e erros fatais")


if __name__ == "__main__":
    main()
