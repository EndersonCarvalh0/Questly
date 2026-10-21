import numpy as np
from sentence_transformers import SentenceTransformer

from src.chunking import PREFIXO_PASSAGE, PREFIXO_QUERY


def gerar_embeddings(
    model: SentenceTransformer, textos: list[str]
) -> np.ndarray:
    vetores = model.encode(
        [PREFIXO_PASSAGE + t for t in textos],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vetores.astype(np.float32)


def gerar_embedding_consulta(model: SentenceTransformer, consulta: str) -> np.ndarray:
    vetor = model.encode(
        PREFIXO_QUERY + consulta,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vetor.astype(np.float32)


def similaridade_cosseno(matriz: np.ndarray, vetor: np.ndarray) -> np.ndarray:
    # vetores já normalizados -> produto escalar == cosseno
    return matriz @ vetor
