"""
Unified CLI and Server Entrypoint for IndicRAG.

Commands:
    python run.py ingest       - Index documents into ChromaDB
    python run.py query <q>    - Run a query through the pipeline
    python run.py evaluate     - Run the evaluation benchmark
    python run.py serve        - Launch the IndicRAG Web Dashboard
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Fix Windows console UTF-8 encoding for Hindi Devanagari rendering
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from tabulate import tabulate
from src.pipeline import IndicRagPipeline
from src.evaluation import evaluate_cases

PROJECT_ROOT = Path(__file__).resolve().parent


def get_pipeline():
    return IndicRagPipeline()


def run_ingest(args):
    print("\n--- Ingesting Documents into ChromaDB ---")
    pipeline = get_pipeline()
    stats = pipeline.ingest_directory("data/raw")
    print(f"✓ Documents loaded:  {stats['documents_loaded']}")
    print(f"✓ Pages processed:   {stats['pages_processed']}")
    print(f"✓ Chunks created:    {stats['chunks_created']}")
    print(f"✓ Embeddings generated ({pipeline.embedding_model_name})")
    print(f"✓ ChromaDB indexed:  {stats['vectors_stored']} vectors stored successfully.\n")


def run_query(args):
    if not args.query:
        print("Error: Please provide a question. Example: python run.py query 'SLV-III ने किस उपग्रह को कक्षा में स्थापित किया?'")
        sys.exit(1)
    if not 1 <= args.top_k <= 20:
        print("Error: --top-k must be between 1 and 20.")
        sys.exit(1)

    pipeline = get_pipeline()
    if pipeline.vector_store.count() == 0:
        print("Error: ChromaDB is empty. Run `python run.py ingest` first.")
        sys.exit(1)
    result = pipeline.ask(args.query, top_k=args.top_k)

    print("\n=======================================================")
    print(" IndicRAG — Retrieval & Answer Trace")
    print("=======================================================")
    print(f"QUESTION: {result['question']}")
    print(f"DETECTED LANGUAGE: {result['language_label']} ({result['detected_language']})")
    print(f"CONFIDENCE STATUS: {'Grounded in context' if result['is_confident'] else 'Insufficient evidence'}")
    print("\n--- RETRIEVAL TRACE (ChromaDB Top-K) ---")

    table_data = []
    for chunk in result["retrieval_trace"]:
        meta = chunk.get("metadata", {})
        table_data.append([
            chunk.get("rank"),
            meta.get("page", "-"),
            meta.get("section", "-")[:28],
            chunk.get("chunk_id"),
            f"{chunk.get('similarity_score', 0):.4f}",
            chunk.get("text", "")[:60] + "..."
        ])
    print(tabulate(table_data, headers=["Rank", "Page", "Section", "Chunk ID", "Score", "Preview"], tablefmt="grid"))

    print("\n--- GROUNDED ANSWER ---")
    print(result["answer"])

    if result["citations"]:
        print("\n--- SOURCE CITATIONS ---")
        for idx, cite in enumerate(result["citations"], 1):
            print(f"[{idx}] Page {cite['page']} | Section: {cite['section']} | Chunk: {cite['chunk_id']} | Score: {cite['score']}")
    print("=======================================================\n")


def run_evaluate(args):
    print("\n--- Running IndicRAG Benchmark Evaluation ---")
    pipeline = get_pipeline()

    eval_path = PROJECT_ROOT / "data" / "eval_dataset.json"
    if not os.path.exists(eval_path):
        print(f"Error: Evaluation dataset not found at {eval_path}")
        sys.exit(1)
    if pipeline.vector_store.count() == 0:
        print("Error: ChromaDB is empty. Run `python run.py ingest` first.")
        sys.exit(1)

    with open(eval_path, "r", encoding="utf-8") as f:
        eval_cases = json.load(f)
    report = evaluate_cases(pipeline, eval_cases, top_k=5)
    table_rows = [[
        item["query"][:38] + ("..." if len(item["query"]) > 38 else ""),
        item["language"],
        item["expected_source"],
        item["retrieved_status"],
        item["citation_correct"],
    ] for item in report["details"]]

    print("\nDetailed Test Query Results:")
    print(tabulate(
        table_rows,
        headers=["Query", "Language", "Expected Target", "Retrieved?", "Citation Verified?"],
        tablefmt="github"
    ))

    summary = report["summary"]
    print("\n--- EVALUATION SUMMARY METRICS ---")
    print(f"Queries tested: {summary['total_queries']}")
    print(f"Supported / unsupported: {summary['supported_queries']} / {summary['unsupported_queries']}")
    print(f"Recall@1: {summary['recall_1']:.1f}%")
    print(f"Recall@3: {summary['recall_3']:.1f}%")
    print(f"Recall@5: {summary['recall_5']:.1f}%")
    print(f"MRR (Mean Reciprocal Rank): {summary['mrr']:.3f}")
    print(f"Citation accuracy: {summary['citation_accuracy']:.1f}%")
    print(f"Unsupported refusal accuracy: {summary['refusal_accuracy']:.1f}%")
    print(f"{summary['explanation']}\n")


def run_serve(args):
    import uvicorn
    print(f"Starting IndicRAG Web Server on http://{args.host}:{args.port} ...")
    uvicorn.run("app:app", host=args.host, port=args.port, reload=False, app_dir=str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="IndicRAG — Multilingual Hindi Document Intelligence")
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # Ingest
    parser_ingest = subparsers.add_parser("ingest", help="Ingest documents into ChromaDB")
    parser_ingest.set_defaults(func=run_ingest)

    # Query
    parser_query = subparsers.add_parser("query", help="Run a query through RAG")
    parser_query.add_argument("query", nargs="?", default=None, help="The user question")
    parser_query.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve")
    parser_query.set_defaults(func=run_query)

    # Evaluate
    parser_eval = subparsers.add_parser("evaluate", help="Run benchmark evaluation")
    parser_eval.set_defaults(func=run_evaluate)

    # Serve
    parser_serve = subparsers.add_parser("serve", help="Launch web application")
    parser_serve.add_argument("--host", default="127.0.0.1", help="Host address")
    parser_serve.add_argument("--port", type=int, default=8000, help="Port number")
    parser_serve.set_defaults(func=run_serve)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
