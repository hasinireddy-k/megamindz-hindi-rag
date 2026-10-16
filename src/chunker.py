"""
Language-Aware Recursive Chunker.

Supports both Latin and Devanagari punctuation (Purna Viram '?', Deergh Viram '?',
periods, paragraph breaks) to prevent mid-clause breaks in multilingual text.
"""

from typing import List, Dict, Any


class MultilingualChunker:
    """
    Recursive character text splitter tailored for Hindi and English documents.
    Preserves semantic boundaries like paragraphs, Devanagari dandas, and periods.
    """

    DEFAULT_SEPARATORS = [
        "\n\n",       # Paragraph break
        "\u0964\n",   # Hindi Purna Viram (।) + newline
        "\u0965\n",   # Hindi Deergh Viram (॥) + newline
        ".\n",       # English period + newline
        "\u0964 ",    # Hindi Purna Viram (।) + space
        "\u0965 ",    # Hindi Deergh Viram (॥) + space
        ". ",        # English period + space
        "?\n",       # Question mark + newline
        "? ",        # Question mark + space
        "\n",         # Single newline
        " ",          # Space
        ""            # Fallback character split
    ]

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        separators: List[str] = None
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursively splits text using the highest priority separator that yields pieces."""
        final_chunks: List[str] = []
        if not separators:
            return [text]

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == "":
            return list(text)

        if separator in text:
            splits = text.split(separator)
        else:
            return self._split_text(text, remaining_separators)

        # Merge splits back respecting chunk_size and chunk_overlap
        current_chunk = ""
        for piece in splits:
            piece_with_sep = piece + (separator if separator != "" else "")
            if not current_chunk:
                current_chunk = piece_with_sep
            elif len(current_chunk) + len(piece_with_sep) <= self.chunk_size:
                current_chunk += piece_with_sep
            else:
                if len(current_chunk.strip()) > 0:
                    final_chunks.append(current_chunk.strip())
                # Handle overlap
                if self.chunk_overlap > 0 and len(current_chunk) > self.chunk_overlap:
                    overlap_seed = current_chunk[-self.chunk_overlap:]
                    current_chunk = overlap_seed + piece_with_sep
                else:
                    current_chunk = piece_with_sep

        if current_chunk.strip():
            final_chunks.append(current_chunk.strip())

        # If any resulting chunk is still larger than chunk_size, recursively split it with remaining separators
        refined_chunks: List[str] = []
        for ch in final_chunks:
            if len(ch) > self.chunk_size and remaining_separators:
                refined_chunks.extend(self._split_text(ch, remaining_separators))
            else:
                refined_chunks.append(ch)

        return refined_chunks

    def chunk_document(
        self,
        text: str,
        source: str,
        doc_metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Splits a normalized document into annotated chunk objects.
        """
        base_meta = doc_metadata or {}
        raw_chunks = self._split_text(text, self.separators)

        structured_chunks: List[Dict[str, Any]] = []
        current_offset = 0

        for idx, chunk_text in enumerate(raw_chunks):
            # Locate approximate character offsets
            start_char = text.find(chunk_text[:30], current_offset) if len(chunk_text) >= 30 else current_offset
            if start_char == -1:
                start_char = current_offset
            end_char = start_char + len(chunk_text)
            current_offset = max(current_offset, start_char + max(1, len(chunk_text) - self.chunk_overlap))

            chunk_id = f"{source}_chunk_{idx + 1:03d}"
            chunk_record = {
                "chunk_id": chunk_id,
                "text": chunk_text,
                "source": source,
                "chunk_index": idx + 1,
                "start_char": start_char,
                "end_char": end_char,
                "char_length": len(chunk_text),
                "token_estimate": len(chunk_text.split()),
                **base_meta
            }
            structured_chunks.append(chunk_record)

        return structured_chunks
