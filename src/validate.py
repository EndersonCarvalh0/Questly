import re
import unicodedata

from src.schema import Questao

_RE_ESPACOS = re.compile(r"\s+")
_RE_HIFEN = re.compile(r"-\s*\n\s*")


def normalizar(texto: str) -> str:
    texto = texto.replace("\u00ad", "")
    texto = _RE_HIFEN.sub("", texto)  # junta palavra hifenizada quebrada em linha
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    # Pontuação pode variar entre o PDF e a citação do modelo; as palavras
    # da evidência ainda precisam aparecer na mesma ordem e sem paráfrase.
    texto = re.sub(r"[^\w\s]", " ", texto)
    texto = _RE_ESPACOS.sub(" ", texto)
    return texto.strip()


def evidencia_no_contexto(questao: Questao, contexto: str) -> bool:
    """Checagem mecânica: o trecho de evidência existe no contexto recuperado."""
    ev = normalizar(questao.evidencia)
    ctx = normalizar(contexto)
    if len(ev) < 10:
        return False
    return ev in ctx


def pagina_da_evidencia(evidencia: str, trechos) -> int | None:
    """Descobre a página real do trecho recuperado que contém a evidência."""
    ev = normalizar(evidencia)
    if len(ev) < 10:
        return None
    for t in trechos:
        if ev in normalizar(t.texto):
            paginas = getattr(t, "paginas", None) or []
            return paginas[0] if paginas else None
    return None
