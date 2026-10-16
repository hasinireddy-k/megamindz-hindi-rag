"""
Transparent Multilingual Retriever Module.

Performs query normalization, language identification, vector search,
similarity score computation, confidence threshold gating, and formatted retrieval logging.
"""

import os
from typing import Dict, Any, List
from src.normalizer import normalize_text, detect_language
from src.vector_store import ChromaVectorStore


class TransparentRetriever:
    """
    Retriever with end-to-end transparency: exposes normalized queries,
    detected scripts, cosine similarities, and ranked chunk snippets.
    """

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        confidence_threshold: float = None
    ):
        self.vector_store = vector_store
        threshold_env = os.getenv("CONFIDENCE_THRESHOLD", "0.35")
        self.confidence_threshold = confidence_threshold if confidence_threshold is not None else float(threshold_env)

    def retrieve(
        self,
        query: str,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Executes normalized multilingual retrieval.
        """
        normalized_query = normalize_text(query, preserve_newlines=False)
        lang = detect_language(normalized_query)

        raw_results = self.vector_store.query(normalized_query, top_k=top_k)

        # Confidence gating
        is_confident = False
        if raw_results and raw_results[0]["similarity_score"] >= self.confidence_threshold:
            is_confident = True

        return {
            "query_original": query,
            "query_normalized": normalized_query,
            "detected_language": lang,
            "top_k": top_k,
            "confidence_threshold": self.confidence_threshold,
            "is_confident": is_confident,
            "retrieved_chunks": raw_results
        }
