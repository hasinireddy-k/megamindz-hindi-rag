"""
ChromaDB Vector Store Interface.

Provides persistent, local vector database storage with cosine distance metric,
rich metadata tagging, collection re-indexing, and transparent query retrieval.
"""

import os
import shutil
from typing import List, Dict, Any, Optional

try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False

from src.embeddings import MultilingualEmbeddingFunction


class ChromaVectorStore:
    """
    Manages local ChromaDB vector collections for multilingual document retrieval.
    """

    def __init__(
        self,
        persist_dir: str = None,
        collection_name: str = None,
        embedding_fn: Optional[MultilingualEmbeddingFunction] = None
    ):
        if not HAS_CHROMADB:
            raise ImportError(
                "chromadb is not installed. Please run: pip install chromadb"
            )

        self.persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
        self.collection_name = collection_name or os.getenv("CHROMA_COLLECTION_NAME", "multilingual_rag")
        self.embedding_fn = embedding_fn or MultilingualEmbeddingFunction()

        # Initialize persistent ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def reset_collection(self):
        """Deletes the current collection and recreates an empty one."""
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 64):
        """
        Adds chunk records to ChromaDB with normalized embeddings and metadata.
        """
        if not chunks:
            return

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            ids = [c["chunk_id"] for c in batch]
            documents = [c["text"] for c in batch]

            # Sanitize metadata (ChromaDB allows only str, int, float, bool)
            metadatas = []
            for c in batch:
                meta = {
                    "source": str(c.get("source", "")),
                    "page": str(c.get("page", "")),
                    "section": str(c.get("section", "")),
                    "chunk_index": int(c.get("chunk_index", 0)),
                    "start_char": int(c.get("start_char", 0)),
                    "end_char": int(c.get("end_char", 0)),
                    "char_length": int(c.get("char_length", 0)),
                    "token_estimate": int(c.get("token_estimate", 0)),
                }
                metadatas.append(meta)

            embeddings = self.embedding_fn.embed_documents(documents)

            self.collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )

    def query(
        self,
        query_text: str,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Performs vector similarity search and returns ranked chunks with similarity scores.
        Cosine distance in ChromaDB is in [0, 2]. Cosine similarity = 1 - (distance / 2) or 1 - distance.
        When embeddings are normalized, cosine distance = 1 - cosine_similarity.
        Therefore: similarity_score = 1.0 - distance.
        """
        query_vector = self.embedding_fn.embed_query(query_text)

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        ranked_results: List[Dict[str, Any]] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return ranked_results

        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        for rank, (cid, doc, meta, dist) in enumerate(zip(ids, docs, metas, distances), start=1):
            # Compute bounded similarity score [0, 1]
            similarity = max(0.0, min(1.0, 1.0 - dist))
            ranked_results.append({
                "rank": rank,
                "chunk_id": cid,
                "text": doc,
                "source": meta.get("source", "unknown"),
                "chunk_index": meta.get("chunk_index", 0),
                "distance": round(float(dist), 4),
                "similarity_score": round(float(similarity), 4),
                "metadata": meta
            })

        return ranked_results

    def count(self) -> int:
        """Returns the number of indexed chunks."""
        return self.collection.count()
