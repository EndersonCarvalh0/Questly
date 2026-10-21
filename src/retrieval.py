from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

from src.chunking import Chunk
from src.embeddings import gerar_embedding_consulta, gerar_embeddings, similaridade_cosseno


@dataclass
class Trecho:
    chunk: Chunk
    score: float


class Indice:
    def __init__(self, model: SentenceTransformer, chunks: list[Chunk]):
        self.model = model
        self.chunks = chunks
        self.matriz = gerar_embeddings(model, [c.texto for c in chunks])

    def buscar(self, consulta: str, top_k: int = 3) -> list[Trecho]:
        vetor = gerar_embedding_consulta(self.model, consulta)
        scores = similaridade_cosseno(self.matriz, vetor)
        ordem = np.argsort(-scores)[:top_k]
        return [Trecho(chunk=self.chunks[i], score=float(scores[i])) for i in ordem]

    def mais_central(self) -> list[Trecho]:
        """Fallback sem tema: usa a média dos vetores como consulta."""
        if not self.chunks:
            return []
        centro = self.matriz.mean(axis=0)
        norma = np.linalg.norm(centro)
        if norma > 0:
            centro = centro / norma
        scores = similaridade_cosseno(self.matriz, centro.astype(np.float32))
        ordem = np.argsort(-scores)
        return [Trecho(chunk=self.chunks[i], score=float(scores[i])) for i in ordem]
