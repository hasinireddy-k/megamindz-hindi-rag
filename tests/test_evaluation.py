import unittest

from src.evaluation import evaluate_cases
from src.generator import GroundedGenerator


class FakePipeline:
    def __init__(self, results):
        self.results = results
        self.generator = GroundedGenerator()

    def ask(self, query, top_k=5):
        return self.results[query]


class EvaluationTests(unittest.TestCase):
    def test_generic_pdf_source_does_not_count_as_relevant_evidence(self):
        query = "किस संस्थान में प्रवेश लिया?"
        chunk = {
            "chunk_id": "source_p09_001",
            "text": "मिसाइल कार्यक्रम का विवरण।",
            "source": "Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf",
            "metadata": {"page": "9", "section": "मिसाइल कार्यक्रम"},
        }
        citation = {
            "page": "9",
            "section": "मिसाइल कार्यक्रम",
            "chunk_id": chunk["chunk_id"],
            "source": chunk["source"],
            "text": chunk["text"],
        }
        result = {
            "answer": "मिसाइल कार्यक्रम का विवरण। [Page 9, Section: मिसाइल कार्यक्रम, Chunk: source_p09_001]",
            "is_grounded": True,
            "citations": [citation],
            "retrieval_trace": [chunk],
        }
        report = evaluate_cases(FakePipeline({query: result}), [{
            "id": "wrong-page",
            "query": query,
            "language": "hi",
            "expected_pages": ["6"],
            "key_phrases": ["मद्रास"],
        }])

        self.assertEqual(report["summary"]["recall_5"], 0.0)
        self.assertEqual(report["details"][0]["retrieved_status"], "Miss")
        self.assertEqual(report["details"][0]["citation_correct"], "No")

    def test_supported_citation_and_unsupported_refusal_are_scored_separately(self):
        supported_query = "Which institute did Kalam attend?"
        unsupported_query = "चंद्रयान-3 कहाँ उतरा?"
        chunk = {
            "chunk_id": "source_p06_001",
            "text": "कलाम ने एरोनॉटिकल इंजीनियरिंग के लिए मद्रास इंस्टीट्यूट ऑफ टेक्नोलॉजी में प्रवेश लिया।",
            "source": "Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf",
            "metadata": {"page": "6", "section": "कॉलेज के वर्ष"},
        }
        citation = {
            "page": "6",
            "section": "कॉलेज के वर्ष",
            "chunk_id": chunk["chunk_id"],
            "source": chunk["source"],
            "text": chunk["text"],
        }
        results = {
            supported_query: {
                "answer": f"{chunk['text']} [Page 6, Section: कॉलेज के वर्ष, Chunk: {chunk['chunk_id']}]",
                "is_grounded": True,
                "citations": [citation],
                "retrieval_trace": [chunk],
            },
            unsupported_query: {
                "answer": GroundedGenerator.OUT_OF_DOMAIN_RESPONSE_HI,
                "is_grounded": False,
                "citations": [],
                "retrieval_trace": [chunk],
            },
        }
        report = evaluate_cases(FakePipeline(results), [
            {
                "id": "supported",
                "query": supported_query,
                "language": "en",
                "expected_pages": ["6"],
                "key_phrases": ["मद्रास इंस्टीट्यूट ऑफ टेक्नोलॉजी"],
            },
            {
                "id": "unsupported",
                "query": unsupported_query,
                "language": "hi",
                "expected_pages": [],
                "key_phrases": [],
            },
        ])

        self.assertEqual(report["summary"]["supported_queries"], 1)
        self.assertEqual(report["summary"]["unsupported_queries"], 1)
        self.assertEqual(report["summary"]["recall_1"], 100.0)
        self.assertEqual(report["summary"]["citation_accuracy"], 100.0)
        self.assertEqual(report["summary"]["refusal_accuracy"], 100.0)


if __name__ == "__main__":
    unittest.main()