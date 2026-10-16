"""
End-to-End Multilingual RAG Pipeline.

Orchestrates document loading, Hindi normalization, recursive chunking,
ChromaDB indexing, transparent multilingual retrieval, and citation-enforced generation.
Reuses loaded models and vector store connections for optimal runtime performance.
"""

import os
import glob
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from src.normalizer import normalize_text, detect_language
from src.chunker import MultilingualChunker
from src.embeddings import DEFAULT_MODEL, MultilingualEmbeddingFunction
from src.vector_store import ChromaVectorStore
from src.retriever import TransparentRetriever
from src.generator import GroundedGenerator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


class IndicRagPipeline:
    """
    Singleton/Reused pipeline for IndicRAG (Hindi Document Intelligence).
    """

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        confidence_threshold: Optional[float] = None,
        gemini_api_key: Optional[str] = None
    ):
        persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
        persist_path = Path(persist_dir)
        self.persist_dir = str(persist_path if persist_path.is_absolute() else PROJECT_ROOT / persist_path)
        self.collection_name = collection_name or os.getenv("CHROMA_COLLECTION_NAME", "multilingual_rag")
        self.embedding_model_name = embedding_model_name or os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_MODEL)
        self.chunk_size = chunk_size if chunk_size is not None else int(os.getenv("CHUNK_SIZE", "500"))
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else int(os.getenv("CHUNK_OVERLAP", "100"))

        # Default confidence threshold 0.55 tuned for paraphrase-multilingual-MiniLM-L12-v2
        env_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.55"))
        self.confidence_threshold = confidence_threshold if confidence_threshold is not None else env_threshold

        # Initialize shared components once
        self.embedding_fn = MultilingualEmbeddingFunction(model_name=self.embedding_model_name)
        self.vector_store = ChromaVectorStore(
            persist_dir=self.persist_dir,
            collection_name=self.collection_name,
            embedding_fn=self.embedding_fn
        )
        self.chunker = MultilingualChunker(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        self.retriever = TransparentRetriever(
            vector_store=self.vector_store,
            confidence_threshold=self.confidence_threshold
        )
        self.generator = GroundedGenerator(api_key=gemini_api_key)

    def ingest_pdf(self, pdf_path: str = "Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf") -> List[Dict[str, Any]]:
        """
        Ingests a Hindi PDF document using PyMuPDF (pymupdf).
        Extracts pages with accurate 1-indexed page numbers, normalizes Hindi Devanagari text,
        identifies chapter/section headings, chunks with danda awareness,
        and assigns deterministic chunk IDs.
        """
        try:
            import pymupdf
        except ImportError as exc:
            raise ImportError(
                "PyMuPDF is required for PDF ingestion. Install project dependencies with "
                "`python -m pip install -r requirements.txt`."
            ) from exc

        pdf_candidate = Path(pdf_path)
        if not pdf_candidate.is_absolute() and not pdf_candidate.exists():
            pdf_candidate = PROJECT_ROOT / pdf_candidate
        if not pdf_candidate.exists():
            alt = PROJECT_ROOT / "data" / "raw" / pdf_candidate.name
            if alt.exists():
                pdf_candidate = alt
            else:
                raise FileNotFoundError(
                    f"PDF document not found: {pdf_path}. Place the supplied PDF in the project root "
                    "or pass its path to the ingestion command."
                )

        doc = pymupdf.open(str(pdf_candidate))
        pdf_name = pdf_candidate.name
        all_chunks: List[Dict[str, Any]] = []

        for p_idx, page in enumerate(doc):
            page_num = p_idx + 1
            raw_text = page.get_text()
            if not raw_text or not raw_text.strip():
                continue

            normalized_text = normalize_text(raw_text, preserve_newlines=True)
            lines = [l.strip() for l in normalized_text.splitlines() if l.strip()]
            sec_heading = lines[0] if lines else f"Page {page_num}"
            sec_heading = re.sub(r"^\s*\d+\s*[·.]\s*", "", sec_heading).strip()

            meta = {
                "source": pdf_name,
                "page": str(page_num),
                "section": sec_heading,
            }

            p_chunks = self.chunker.chunk_document(
                normalized_text,
                source=f"Agni_Ki_Udaan_p{page_num:02d}",
                doc_metadata=meta
            )
            all_chunks.extend(p_chunks)

        return all_chunks

    def ingest_directory(
        self,
        raw_dir: str = "data/raw",
        pdf_file: str = "Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf"
    ) -> Dict[str, Any]:
        """
        Unified ingestion: Ingests primary PDF document using PyMuPDF and any supplemental text files.
        Resets and populates persistent ChromaDB collection.
        """
        all_chunks: List[Dict[str, Any]] = []
        docs_loaded = 0
        total_pages = 0

        # 1. Ingest primary PDF
        project_raw_dir = Path(raw_dir)
        if not project_raw_dir.is_absolute():
            project_raw_dir = PROJECT_ROOT / project_raw_dir
        pdf_candidates = [
            Path(pdf_file),
            project_raw_dir / Path(pdf_file).name,
            PROJECT_ROOT / Path(pdf_file).name,
        ]
        target_pdf = next(
            (candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
             for candidate in pdf_candidates
             if (candidate if candidate.is_absolute() else PROJECT_ROOT / candidate).exists()),
            None,
        )
        if target_pdf is None:
            raise FileNotFoundError(
                f"Required assignment PDF not found: {Path(pdf_file).name}. "
                f"Expected it in {PROJECT_ROOT} or {project_raw_dir}."
            )

        pdf_chunks = self.ingest_pdf(str(target_pdf))
        if not pdf_chunks:
            raise ValueError(
                f"No searchable text was extracted from {target_pdf}. The PDF may be image-only; "
                "OCR is not currently supported."
            )
        all_chunks.extend(pdf_chunks)
        docs_loaded += 1
        total_pages += len({chunk["page"] for chunk in pdf_chunks})

        # 2. Ingest supplemental text documents if raw_dir exists (excluding old mock txt)
        if project_raw_dir.is_dir():
            txt_files = [
                f for f in glob.glob(str(project_raw_dir / "*.txt"))
                if "agni_ki_udaan" not in os.path.basename(f).lower()
            ]
            for file_path in txt_files:
                fname = os.path.basename(file_path)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                norm = normalize_text(content, preserve_newlines=True)
                docs_loaded += 1
                total_pages += 1
                meta = {"source": fname, "page": "1", "section": "General"}
                sec_chunks = self.chunker.chunk_document(norm, source=fname, doc_metadata=meta)
                all_chunks.extend(sec_chunks)

        # Replace the collection only after every source document has loaded successfully.
        self.vector_store.reset_collection()
        self.vector_store.add_chunks(all_chunks)

        return {
            "documents_loaded": docs_loaded,
            "pages_processed": total_pages,
            "chunks_created": len(all_chunks),
            "vectors_stored": self.vector_store.count(),
            "status": "success"
        }

    def ask(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Executes end-to-end question answering with full retrieval trace.
        """
        if not question or not question.strip():
            return {
                "error": "Please enter a question.",
                "question": "",
                "answer": "",
                "language": "en",
                "retrieval_trace": [],
                "citations": []
            }

        retrieval = self.retriever.retrieve(question, top_k=top_k)
        generation = self.generator.generate(question, retrieval)

        return {
            "question": question,
            "normalized_question": retrieval["query_normalized"],
            "detected_language": retrieval["detected_language"],
            "language_label": "Hindi" if retrieval["detected_language"] == "hi" else ("English" if retrieval["detected_language"] == "en" else "Mixed"),
            "answer": generation["answer"],
            "is_grounded": generation["is_grounded"],
            "citations": generation["citations"],
            "retrieval_trace": retrieval["retrieved_chunks"],
            "confidence_threshold": self.confidence_threshold,
            "is_confident": retrieval["is_confident"]
        }
