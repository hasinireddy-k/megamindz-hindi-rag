"""
ChromaDB Vector Store Interface.

Provides persistent, local vector database storage with cosine distance metric,
rich metadata tagging, collection re-indexing, and transparent query retrieval.
"""

import os
import re
from typing import List, Dict, Any, Optional
from pathlib import Path

import numpy as np

try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False

from src.embeddings import MultilingualEmbeddingFunction
from src.chunker import MultilingualChunker
from src.normalizer import normalize_text


class ReadOnlyVectorStore:
    """In-memory semantic index built from repository text files for Vercel."""

    PRIMARY_SOURCE = "Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf"
    PAGE_HEADING = re.compile(r"^##\s+पृष्ठ\s+(\d+)\s*:\s*(.*)$", re.MULTILINE)

    def __init__(
        self,
        project_root: str,
        collection_name: Optional[str] = None,
        embedding_fn: Optional[MultilingualEmbeddingFunction] = None,
    ):
        self.project_root = Path(project_root)
        self.collection_name = collection_name or os.getenv("CHROMA_COLLECTION_NAME", "multilingual_rag")
        self.embedding_fn = embedding_fn or MultilingualEmbeddingFunction()
        self.chunker = MultilingualChunker(
            chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "100")),
        )
        self._chunks = self._load_chunks()
        self._embeddings = None

    def _load_chunks(self) -> List[Dict[str, Any]]:
        raw_dir = self.project_root / "data" / "raw"
        primary_path = raw_dir / "agni_ki_udaan_hi.txt"
        if not primary_path.is_file():
            raise FileNotFoundError(
                "Vercel's read-only index source is missing: data/raw/agni_ki_udaan_hi.txt"
            )

        primary_text = primary_path.read_text(encoding="utf-8-sig")
        headings = list(self.PAGE_HEADING.finditer(primary_text))
        if not headings:
            raise ValueError(
                "The Hindi source text has no '## पृष्ठ N: heading' markers; cannot preserve page citations."
            )

        chunks: List[Dict[str, Any]] = []
        for index, heading in enumerate(headings):
            page = int(heading.group(1))
            end = headings[index + 1].start() if index + 1 < len(headings) else len(primary_text)
            content = normalize_text(primary_text[heading.end():end], preserve_newlines=True)
            if content:
                chunks.extend(self.chunker.chunk_document(
                    content,
                    source=f"Agni_Ki_Udaan_p{page:02d}",
                    doc_metadata={
                        "source": self.PRIMARY_SOURCE,
                        "page": str(page),
                        "section": heading.group(2).strip() or f"Page {page}",
                    },
                ))

        for text_path in sorted(raw_dir.glob("*.txt")):
            if text_path.name.casefold() == primary_path.name.casefold():
                continue
            content = normalize_text(text_path.read_text(encoding="utf-8-sig"), preserve_newlines=True)
            if content:
                chunks.extend(self.chunker.chunk_document(
                    content,
                    source=text_path.name,
                    doc_metadata={"source": text_path.name, "page": "1", "section": "General"},
                ))

        if not chunks:
            raise ValueError("No searchable text was found in data/raw.")
        return chunks

    def _ensure_embeddings(self) -> None:
        if self._embeddings is None:
            vectors = self.embedding_fn.embed_documents([chunk["text"] for chunk in self._chunks])
            self._embeddings = np.asarray(vectors, dtype=np.float32)

    def query(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if top_k <= 0 or not self._chunks:
            return []
        self._ensure_embeddings()
        query_vector = np.asarray(self.embedding_fn.embed_query(query_text), dtype=np.float32)
        similarities = self._embeddings @ query_vector
        best_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for rank, index in enumerate(best_indices, start=1):
            chunk = self._chunks[int(index)]
            metadata = {key: value for key, value in chunk.items() if key not in {
                "chunk_id", "text", "chunk_index", "start_char", "end_char", "char_length", "token_estimate"
            }}
            similarity = float(np.clip(similarities[index], 0.0, 1.0))
            results.append({
                "rank": rank,
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "source": metadata.get("source", "unknown"),
                "chunk_index": chunk.get("chunk_index", 0),
                "distance": round(1.0 - similarity, 4),
                "similarity_score": round(similarity, 4),
                "metadata": metadata,
            })
        return results

    def count(self) -> int:
        return len(self._chunks)

    def get_metadatas(self) -> List[Dict[str, Any]]:
        fields = {"source", "page", "section", "chunk_index", "start_char", "end_char", "char_length", "token_estimate"}
        return [{key: value for key, value in chunk.items() if key in fields} for chunk in self._chunks]

    def reset_collection(self) -> None:
        raise RuntimeError("The Vercel index is read-only. Update data/raw and redeploy to change documents.")

    def add_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 64) -> None:
        raise RuntimeError("The Vercel index is read-only. Update data/raw and redeploy to change documents.")


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
                "chromadb is not installed. Please run: pip install -r requirements-local.txt"
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

    def get_metadatas(self) -> List[Dict[str, Any]]:
        """Returns metadata for all chunks without exposing Chroma internals to the API."""
        stored = self.collection.get(include=["metadatas"])
        return stored.get("metadatas") or []
