"""
Multilingual Embedding Model Interface.

Uses sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 to project
both Hindi (Devanagari) and English queries and passages into a shared 384-dimensional
semantic vector space, enabling native cross-lingual retrieval without machine translation.
"""

import os
from typing import List, Optional
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    HAS_CHROMA_ONNX = True
except ImportError:
    HAS_CHROMA_ONNX = False

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class MultilingualEmbeddingFunction:
    """
    Embedding function that projects both English and Hindi text into a 384-d semantic vector space.
    Compatible with ChromaDB's EmbeddingFunction protocol.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_MODEL)
        self._st_model = None
        self._fallback_onnx = None

    @property
    def model(self):
        if self._st_model is None:
            if HAS_SENTENCE_TRANSFORMERS:
                self._st_model = SentenceTransformer(self.model_name)
            elif HAS_CHROMA_ONNX:
                self._fallback_onnx = DefaultEmbeddingFunction()
            else:
                raise ImportError(
                    "Neither sentence-transformers nor chromadb embedding utilities are available."
                )
        return self._st_model if self._st_model is not None else self._fallback_onnx

    def __call__(self, input: List[str]) -> List[List[float]]:
        """ChromaDB EmbeddingFunction signature."""
        return self.embed_documents(input)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Computes normalized embeddings for a list of document chunks."""
        if not texts:
            return []

        # If sentence-transformers model is active
        if HAS_SENTENCE_TRANSFORMERS and (self._st_model is not None or self.model is self._st_model):
            embeddings = self._st_model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            return embeddings.tolist()

        # Fallback path if sentence-transformers is missing
        raw_embs = self.model(texts)
        norm_embs = []
        for vec in raw_embs:
            arr = np.array(vec, dtype=np.float32)
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
            norm_embs.append(arr.tolist())
        return norm_embs

    def embed_query(self, query: str) -> List[float]:
        """Computes a normalized embedding for a single user query."""
        if HAS_SENTENCE_TRANSFORMERS and (self._st_model is not None or self.model is self._st_model):
            emb = self._st_model.encode(
                [query],
                normalize_embeddings=True
            )
            return emb[0].tolist()

        res = self.embed_documents([query])
        return res[0]
