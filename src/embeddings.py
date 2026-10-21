"""Lightweight multilingual embeddings shared by local and Vercel deployments."""

import os
import tempfile
from typing import List, Optional

import numpy as np

try:
    from fastembed import TextEmbedding
except ImportError:  # pragma: no cover - gives a useful setup error at first use
    TextEmbedding = None


DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class MultilingualEmbeddingFunction:
    """FastEmbed wrapper that returns normalized 384-dimensional vectors."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_MODEL)
        self.cache_dir = os.getenv(
            "FASTEMBED_CACHE_PATH",
            os.path.join(tempfile.gettempdir(), "fastembed"),
        )
        self._model = None

    @property
    def model(self):
        if self._model is None:
            if TextEmbedding is None:
                raise ImportError("FastEmbed is required. Install dependencies from requirements.txt.")
            self._model = TextEmbedding(
                model_name=self.model_name,
                cache_dir=self.cache_dir,
                threads=2,
                providers=["CPUExecutionProvider"],
            )
        return self._model

    @staticmethod
    def _normalize(vectors) -> List[List[float]]:
        normalized = []
        for vector in vectors:
            values = np.asarray(vector, dtype=np.float32)
            norm = float(np.linalg.norm(values))
            if norm:
                values = values / norm
            normalized.append(values.tolist())
        return normalized

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.embed_documents(input)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return self._normalize(self.model.embed(texts, batch_size=32))

    def embed_query(self, query: str) -> List[float]:
        vectors = self._normalize(self.model.query_embed([query]))
        return vectors[0]
