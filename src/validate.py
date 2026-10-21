import re

from src.schema import Questao

_RE_ESPACOS = re.compile(r"\s+")
_RE_HIFEN = re.compile(r"-\s*\n\s*")


def normalizar(texto: str) -> str:
    texto = texto.replace("\u00ad", "")
    texto = _RE_HIFEN.sub("", texto)  # junta palavra hifenizada quebrada em linha
    texto = texto.lower()
    texto = _RE_ESPACOS.sub(" ", texto)
    return texto.strip()


def evidencia_no_contexto(questao: Questao, contexto: str) -> bool:
    """Checagem mecânica: o trecho de evidência existe no contexto recuperado."""
    ev = normalizar(questao.evidencia)
    ctx = normalizar(contexto)
    if len(ev) < 10:
        return False
    return ev in ctx
