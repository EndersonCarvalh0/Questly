import json
import sys
from threading import Event

sys.path.insert(0, ".")

from src import geracao
from src.chunking import Chunk

CHUNK = Chunk(
    "A clorofila fica nos cloroplastos e absorve a luz solar.",
    [1],
    12,
)
QUESTAO = {
    "enunciado": "Onde fica a clorofila que absorve a luz solar?",
    "alternativas": ["Mitocôndria", "Cloroplasto", "Ribossomo", "Lisossomo"],
    "correta": 1,
    "justificativa": "A clorofila está nos cloroplastos, onde ocorre a fotossíntese.",
    "evidencia": "A clorofila fica nos cloroplastos",
    "pagina": 2,
}


def _instalar_chat(fake):
    fake.ultimo_usage = {}
    original = geracao.chat
    geracao.chat = fake
    return original


def testar_cancelado_antes_de_comecar() -> None:
    chamadas = []

    def chat_fake(*args, **kwargs):
        chamadas.append(1)
        return json.dumps(QUESTAO, ensure_ascii=False)

    original = _instalar_chat(chat_fake)
    evento = Event()
    evento.set()
    try:
        res = geracao.gerar_questoes(
            [CHUNK], 3, provedor="nvidia", cancelar=evento
        )
    finally:
        geracao.chat = original

    assert res.cancelado
    assert res.questoes == []
    assert chamadas == []


def testar_cancelamento_apos_primeira_questao() -> None:
    chamadas = []
    evento = Event()

    def chat_fake(*args, **kwargs):
        chamadas.append(1)
        evento.set()
        return json.dumps(QUESTAO, ensure_ascii=False)

    original = _instalar_chat(chat_fake)
    try:
        res = geracao.gerar_questoes(
            [CHUNK], 3, provedor="nvidia", cancelar=evento
        )
    finally:
        geracao.chat = original

    assert len(res.questoes) == 1
    assert res.cancelado
    assert res.descartadas == 0
    assert len(chamadas) == 1
    assert len(res.checagens) == 5


def testar_cancelamento_em_questao_invalida_nao_retenta() -> None:
    chamadas = []
    evento = Event()

    def chat_fake(*args, **kwargs):
        chamadas.append(1)
        evento.set()
        return "resposta sem JSON"

    original = _instalar_chat(chat_fake)
    try:
        res = geracao.gerar_questoes(
            [CHUNK], 3, provedor="nvidia", cancelar=evento
        )
    finally:
        geracao.chat = original

    assert res.questoes == []
    assert res.descartadas == 1
    assert res.cancelado
    assert len(chamadas) == 1


def main() -> None:
    testar_cancelado_antes_de_comecar()
    testar_cancelamento_apos_primeira_questao()
    testar_cancelamento_em_questao_invalida_nao_retenta()
    print("[OK] cancelamento cooperativo")


if __name__ == "__main__":
    main()
