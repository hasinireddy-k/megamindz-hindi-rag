"""
Grounded Answer Generator with Strict Citation Attribution.

Supports:
1. Context-constrained grounded generation.
2. Out-of-domain rejection when similarity is below confidence threshold.
3. Accurate source citation extraction (Page, Section, Chunk ID, Score).
4. Dual-mode synthesis: Optional Google Gemini API if configured, with a robust
   deterministic extractive synthesizer as default for offline reproducibility.
"""

import os
import re
from typing import Dict, Any, List


class GroundedGenerator:
    """
    Produces strictly grounded answers with verifiable source citations.
    """

    OUT_OF_DOMAIN_RESPONSE = (
        "I could not find sufficient evidence in the supplied document to answer this question."
    )
    OUT_OF_DOMAIN_RESPONSE_HI = "दिए गए दस्तावेज़ में इस प्रश्न का उत्तर देने के लिए पर्याप्त प्रमाण नहीं मिला।"

    def __init__(self, api_key: str = None, model_name: str = "gemini-1.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", None)
        self.model_name = model_name
        self._llm_client = None

        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._llm_client = genai.GenerativeModel(self.model_name)
            except Exception:
                self._llm_client = None

    def generate(
        self,
        query: str,
        retrieval_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generates an answer strictly grounded in the retrieved context.
        """
        retrieved_chunks = retrieval_result.get("retrieved_chunks", [])
        is_confident = retrieval_result.get("is_confident", False)

        # Refusal on insufficient evidence or out-of-domain
        if not retrieved_chunks or not is_confident:
            return self._refusal_result(query)

        # Attempt LLM generation if available
        if self._llm_client:
            try:
                return self._generate_with_llm(query, retrieved_chunks)
            except Exception:
                pass

        # Fallback to deterministic extractive synthesis
        return self._generate_extractive(query, retrieved_chunks)

    def _refusal_result(self, query: str) -> Dict[str, Any]:
        is_hindi = any("\u0900" <= character <= "\u097f" for character in query)
        return {
            "answer": self.OUT_OF_DOMAIN_RESPONSE_HI if is_hindi else self.OUT_OF_DOMAIN_RESPONSE,
            "is_grounded": False,
            "citations": [],
            "mode": "refusal"
        }

    def _generate_with_llm(self, query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        context_blocks = []
        citations = []
        for c in chunks:
            meta = c.get("metadata", {})
            page = meta.get("page", "Unknown")
            section = meta.get("section", "Unknown")
            cid = c.get("chunk_id", "Unknown")
            score = c.get("similarity_score", 0.0)

            citations.append({
                "page": page,
                "section": section,
                "chunk_id": cid,
                "score": score,
                "source": c.get("source", ""),
                "text": c.get("text", "")
            })
            context_blocks.append(
                f"[Page {page}, Section: {section}, Chunk: {cid}]\n{c.get('text', '')}"
            )

        context_str = "\n\n".join(context_blocks)
        prompt = (
            "You are an expert AI document research assistant. Answer the user question strictly using only "
            "the provided context below. Every factual statement must cite its source using [Page X, Section: Y, Chunk: Z]. "
            "If the context does not contain the answer, say exactly: "
            f"'{self.OUT_OF_DOMAIN_RESPONSE}'\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {query}\n\nAnswer:"
        )

        response = self._llm_client.generate_content(prompt)
        answer_text = response.text.strip()

        if answer_text in {self.OUT_OF_DOMAIN_RESPONSE, self.OUT_OF_DOMAIN_RESPONSE_HI}:
            return self._refusal_result(query)

        citation_pattern = re.compile(
            r"\[Page ([^,\]]+), Section: ([^,\]]+), Chunk: ([^\]]+)\]"
        )
        available_citations = {
            (str(citation["page"]), str(citation["section"]), str(citation["chunk_id"])): citation
            for citation in citations
        }
        referenced_citations = citation_pattern.findall(answer_text)
        if not referenced_citations or any(
            reference not in available_citations for reference in referenced_citations
        ):
            return self._generate_extractive(query, chunks)

        answer_without_citations = citation_pattern.sub("", answer_text)
        answer_sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?।])\s+|\n+", answer_without_citations)
            if sentence.strip()
        ]
        if not answer_sentences or len(referenced_citations) < len(answer_sentences):
            return self._generate_extractive(query, chunks)

        used_citations = [
            available_citations[reference]
            for reference in dict.fromkeys(referenced_citations)
        ]

        return {
            "answer": answer_text,
            "is_grounded": True,
            "citations": used_citations,
            "mode": "llm"
        }

    HINDI_STOPWORDS = {
        "में", "ने", "को", "से", "का", "के", "की", "था", "थी", "थे", "है", "हैं",
        "पर", "और", "यह", "वह", "कब", "कहाँ", "कैसे", "किस", "किसे", "क्या", "तक",
        "ही", "भी", "द्वारा", "गया", "गई", "गए", "कर", "किया", "हुए", "हुआ", "हुई"
    }
    ENGLISH_STOPWORDS = {
        "the", "a", "an", "in", "on", "at", "by", "for", "with", "about", "against",
        "between", "into", "through", "during", "before", "after", "above", "below",
        "to", "from", "up", "down", "of", "and", "or", "is", "was", "are", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did", "which",
        "what", "where", "when", "who", "whom", "how", "why", "did"
    }

    def _generate_extractive(self, query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Grounded synthesis that extracts the most pertinent evidence from retrieved context
        and attaches verified citations. Honest refusal when evidence is absent.
        """
        if not chunks:
            return self._refusal_result(query)

        # Extract content keywords from query
        raw_words = re.findall(r'[a-z0-9]+(?:-[a-z0-9]+)*|[\u0900-\u097f]+', query.casefold())
        q_terms = [w for w in raw_words if w not in self.HINDI_STOPWORDS and w not in self.ENGLISH_STOPWORDS and len(w) > 1]

        citations_by_chunk = {}
        key_sentences = []

        for c in chunks:
            meta = c.get("metadata", {})
            page = meta.get("page", c.get("page", "Unknown"))
            section = meta.get("section", c.get("section", "General"))
            cid = c.get("chunk_id", "Unknown")
            score = c.get("similarity_score", 0.0)
            text = c.get("text", "")

            citations_by_chunk[cid] = {
                "page": str(page),
                "section": section,
                "chunk_id": cid,
                "score": score,
                "source": c.get("source", ""),
                "text": text
            }

            # Split text into sentences using Danda and newline/period boundaries
            raw_sentences = re.split(r'[\u0964\n\.]+', text)
            for s in raw_sentences:
                s_clean = s.strip()
                if len(s_clean) >= 20:
                    sentence_terms = set(re.findall(r'[a-z0-9]+(?:-[a-z0-9]+)*|[\u0900-\u097f]+', s_clean.casefold()))
                    overlap = sum(1 for term in q_terms if term in sentence_terms)
                    key_sentences.append((overlap, score, s_clean, str(page), section, cid))

        # Out-of-domain detection:
        is_hindi_query = any('\u0900' <= ch <= '\u097f' for ch in query)
        total_overlap = sum(k[0] for k in key_sentences)
        top_score = chunks[0].get("similarity_score", 0.0)

        # Refusal cases:
        # 1. Hindi query with zero content keyword overlap in retrieved context
        if is_hindi_query and total_overlap == 0:
            return self._refusal_result(query)
        # 2. English cross-lingual query where top similarity is below confidence threshold (0.55)
        if not is_hindi_query and top_score < 0.55:
            return self._refusal_result(query)

        # Select best sentence:
        # If term overlap exists, sort by (overlap, score). Otherwise, select the most informative sentence from top chunk.
        if total_overlap > 0:
            key_sentences.sort(key=lambda x: (x[0], x[1]), reverse=True)
            best = key_sentences[0]
            answer = f"{best[2]}। [Page {best[3]}, Section: {best[4]}, Chunk: {best[5]}]"

            # Complementary sentence if available
            if len(key_sentences) > 1 and key_sentences[1][0] > 0 and key_sentences[1][2] != best[2]:
                second = key_sentences[1]
                answer += f"\n\nइसके अतिरिक्त: {second[2]}। [Page {second[3]}, Section: {second[4]}, Chunk: {second[5]}]"
            used_chunk_ids = [best[5]]
            if len(key_sentences) > 1 and key_sentences[1][0] > 0 and key_sentences[1][2] != best[2]:
                used_chunk_ids.append(key_sentences[1][5])
        else:
            # Cross-lingual semantic match from top chunk
            top = next(iter(citations_by_chunk.values()))
            top_sentences = [s.strip() for s in re.split(r'[\u0964\n\.]+', top['text']) if len(s.strip()) >= 25]
            if top_sentences:
                # Pick the most substantial informative sentence
                best_s = top_sentences[1] if len(top_sentences) > 1 and not top_sentences[0].startswith(('1', '2', '3', '4', '5', '6', '7', '8', '9')) else top_sentences[0]
                answer = f"{best_s}। [Page {top['page']}, Section: {top['section']}, Chunk: {top['chunk_id']}]"
            else:
                answer = f"{top['text'][:250]}... [Page {top['page']}, Section: {top['section']}, Chunk: {top['chunk_id']}]"
            used_chunk_ids = [top["chunk_id"]]

        return {
            "answer": answer,
            "is_grounded": True,
            "citations": [citations_by_chunk[cid] for cid in dict.fromkeys(used_chunk_ids)],
            "mode": "extractive_grounded"
        }
