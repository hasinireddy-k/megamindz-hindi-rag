"""Evidence-based evaluation helpers shared by the CLI and API."""

from typing import Any, Dict, List, Optional

from src.generator import GroundedGenerator
from src.normalizer import normalize_text


def _matches_expected(case: Dict[str, Any], item: Dict[str, Any]) -> bool:
    expected_pages = {str(page) for page in case.get("expected_pages", [])}
    metadata = item.get("metadata", {})
    page = str(metadata.get("page", item.get("page", "")))
    if not expected_pages or page not in expected_pages:
        return False

    phrases = case.get("key_phrases", [])
    if not phrases:
        return True

    text = normalize_text(item.get("text", ""), preserve_newlines=False).casefold()
    return any(
        normalize_text(phrase, preserve_newlines=False).casefold() in text
        for phrase in phrases
    )


def _citation_matches_trace(citation: Dict[str, Any], trace: List[Dict[str, Any]]) -> bool:
    for chunk in trace:
        metadata = chunk.get("metadata", {})
        if chunk.get("chunk_id") != citation.get("chunk_id"):
            continue
        return (
            str(metadata.get("page", chunk.get("page", ""))) == str(citation.get("page", ""))
            and metadata.get("section", chunk.get("section", "General")) == citation.get("section")
            and chunk.get("text", "") == citation.get("text", "")
            and chunk.get("source", "") == citation.get("source", "")
        )
    return False


def evaluate_cases(pipeline: Any, cases: List[Dict[str, Any]], top_k: int = 5) -> Dict[str, Any]:
    """Evaluate retrieval, refusal, and citation provenance on supplied cases."""
    supported_cases = 0
    unsupported_cases = 0
    hits_at = {1: 0, 3: 0, 5: 0}
    reciprocal_ranks: List[float] = []
    correct_refusals = 0
    correct_citations = 0
    details = []

    for case in cases:
        result = pipeline.ask(case["query"], top_k=top_k)
        trace = result.get("retrieval_trace", [])
        citations = result.get("citations", [])
        language = "Hindi" if case.get("language") == "hi" else "English"
        expected_pages = case.get("expected_pages", [])

        if not expected_pages:
            unsupported_cases += 1
            is_refusal = (
                not result.get("is_grounded", False)
                and not citations
                and result.get("answer") in {
                    GroundedGenerator.OUT_OF_DOMAIN_RESPONSE,
                    GroundedGenerator.OUT_OF_DOMAIN_RESPONSE_HI,
                }
            )
            correct_refusals += int(is_refusal)
            details.append({
                "id": case["id"],
                "query": case["query"],
                "language": language,
                "expected_source": "None (unsupported)",
                "retrieved_status": "Refusal (Correct)" if is_refusal else "Not refused",
                "citation_correct": "N/A (unsupported)",
                "answer": result.get("answer", ""),
                "citations": citations,
                "trace": trace,
            })
            continue

        supported_cases += 1
        first_relevant_rank: Optional[int] = next(
            (rank for rank, chunk in enumerate(trace, start=1) if _matches_expected(case, chunk)),
            None,
        )
        for cutoff in hits_at:
            if first_relevant_rank is not None and first_relevant_rank <= cutoff:
                hits_at[cutoff] += 1
        reciprocal_ranks.append(1.0 / first_relevant_rank if first_relevant_rank else 0.0)

        traceable = bool(citations) and all(_citation_matches_trace(citation, trace) for citation in citations)
        citation_supports_expected_text = any(
            _matches_expected(case, citation) for citation in citations
        )
        inline_citation = any(
            f"[Page {citation.get('page')}, Section: {citation.get('section')}, "
            f"Chunk: {citation.get('chunk_id')}]" in result.get("answer", "")
            for citation in citations
        )
        citation_correct = traceable and citation_supports_expected_text and inline_citation
        correct_citations += int(citation_correct)
        page_display = ", ".join(str(page) for page in expected_pages)
        details.append({
            "id": case["id"],
            "query": case["query"],
            "language": language,
            "expected_source": f"PDF page {page_display}",
            "retrieved_status": f"Rank {first_relevant_rank}" if first_relevant_rank else "Miss",
            "citation_correct": "Yes" if citation_correct else "No",
            "answer": result.get("answer", ""),
            "citations": citations,
            "trace": trace,
        })

    def percentage(value: int, denominator: int) -> float:
        return round(value * 100 / denominator, 1) if denominator else 0.0

    return {
        "summary": {
            "total_queries": len(cases),
            "supported_queries": supported_cases,
            "unsupported_queries": unsupported_cases,
            "recall_1": percentage(hits_at[1], supported_cases),
            "recall_3": percentage(hits_at[3], supported_cases),
            "recall_5": percentage(hits_at[5], supported_cases),
            "mrr": round(sum(reciprocal_ranks) / supported_cases, 3) if supported_cases else 0.0,
            "citation_accuracy": percentage(correct_citations, supported_cases),
            "refusal_accuracy": percentage(correct_refusals, unsupported_cases),
            "explanation": (
                "Retrieval metrics use supported queries only; a hit requires both an expected PDF page "
                "and a listed evidence phrase. Citation accuracy also checks the inline citation and "
                "exact retrieved chunk text. Refusal accuracy is reported separately."
            ),
        },
        "details": details,
    }