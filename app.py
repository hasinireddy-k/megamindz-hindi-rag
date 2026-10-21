"""
IndicRAG - FastAPI Backend.
Exposes clean REST endpoints for Ask, Retrieval Trace, Evaluation, and Document Stats.
Uses the identical RAG pipeline as the CLI for 100% parity.
"""

import os
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.pipeline import IndicRagPipeline
from src.evaluation import evaluate_cases

PROJECT_ROOT = Path(__file__).resolve().parent

app = FastAPI(
    title="IndicRAG — Hindi Document Intelligence",
    description="Multilingual Retrieval-Augmented Generation over Hindi Documents",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton pipeline instance reused across requests
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = IndicRagPipeline()
    return _pipeline


class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=3, ge=1, le=20)


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = PROJECT_ROOT / "templates" / "index.html"
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Frontend template not found")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()


@app.post("/api/ask")
async def api_ask(req: QueryRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Please enter a question.")
    pipeline = get_pipeline()
    if pipeline.vector_store.count() == 0:
        raise HTTPException(status_code=409, detail="The search index is empty. Ingest the source documents before asking questions.")
    result = pipeline.ask(req.query, top_k=req.top_k)
    return JSONResponse(content=result)


@app.get("/api/evaluate")
async def api_evaluate():
    pipeline = get_pipeline()
    if pipeline.vector_store.count() == 0:
        raise HTTPException(status_code=409, detail="The search index is empty. Ingest the source documents before evaluating.")
    eval_path = PROJECT_ROOT / "data" / "eval_dataset.json"
    if not os.path.exists(eval_path):
        raise HTTPException(status_code=404, detail="Evaluation dataset not found")

    with open(eval_path, "r", encoding="utf-8") as f:
        eval_cases = json.load(f)

    return JSONResponse(content=evaluate_cases(pipeline, eval_cases, top_k=5))


@app.get("/api/stats")
async def api_stats():
    pipeline = get_pipeline()
    count = pipeline.vector_store.count()
    metadatas = pipeline.vector_store.get_metadatas()
    pages = {
        metadata.get("page", "")
        for metadata in metadatas
        if metadata.get("page")
        and metadata.get("source") == "Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf"
    }
    return JSONResponse(content={
        "document_name": "Agni Ki Udaan — Hindi Study Reference",
        "pages_processed": len(pages),
        "language": "Hindi / Devanagari",
        "vector_db": "Read-only in-memory index" if pipeline.read_only else "ChromaDB (Persistent)",
        "vectors_stored": count,
        "embedding_model": pipeline.embedding_model_name,
        "chunk_size": pipeline.chunk_size,
        "chunk_overlap": pipeline.chunk_overlap,
        "retrieval_method": "Semantic Cosine Distance",
        "generator_mode": "Gemini with source-checked citations" if pipeline.generator._llm_client else "Grounded extractive",
        "status": "ready" if count > 0 else "needs_ingestion"
    })


@app.post("/api/ingest")
async def api_ingest():
    pipeline = get_pipeline()
    if pipeline.read_only:
        raise HTTPException(
            status_code=405,
            detail="The Vercel index is read-only. Update tracked files in data/raw and redeploy to change documents.",
        )
    try:
        stats = pipeline.ingest_directory("data/raw")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=405, detail=str(exc)) from exc
    return JSONResponse(content=stats)
