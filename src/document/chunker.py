"""Semantic text chunking with paragraph-boundary awareness.

Splits documents into chunks while respecting:
1. Paragraph boundaries (preferred split points)
2. Sentence boundaries (fallback for oversized paragraphs)
3. Configurable chunk_size and chunk_overlap
4. Metadata propagation to child chunks
"""

import re
import logging
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class SemanticChunker:
    """Split documents into semantically coherent chunks.

    Strategy:
    1. First, split on paragraph boundaries (\n\n).
    2. If a paragraph exceeds chunk_size, fall back to sentence splitting.
    3. Merge adjacent chunks that are too small.
    4. Apply overlap between consecutive chunks for continuity.

    Usage:
        chunker = SemanticChunker(chunk_size=512, chunk_overlap=64, max_chunk_size=1024)
        chunks = chunker.split(documents)
    """

    # Sentence boundary patterns for Chinese and English
    _SENTENCE_END = re.compile(r"[。！？.!?\n]")

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        max_chunk_size: int = 1024,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_chunk_size = max_chunk_size

        # Fallback splitter for oversized paragraphs
        self._fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
            keep_separator=True,
        )

    def split(self, documents: List[Document]) -> List[Document]:
        """Split a list of documents into chunks."""
        all_chunks: List[Document] = []
        for doc in documents:
            chunks = self._split_one(doc)
            all_chunks.extend(chunks)

        logger.info(
            "Split %d documents into %d chunks (size=%d, overlap=%d)",
            len(documents), len(all_chunks), self.chunk_size, self.chunk_overlap,
        )
        return all_chunks

    # ── Internal logic ───────────────────────────────────────────────

    def _split_one(self, doc: Document) -> List[Document]:
        """Split a single document into chunks."""
        text = doc.page_content
        if not text.strip():
            return []

        # Step 1: Split by paragraphs
        paragraphs = self._split_paragraphs(text)

        # Step 2: Handle oversized paragraphs
        segments = self._handle_oversized(paragraphs)

        # Step 3: Merge segments into chunks with overlap
        chunks = self._merge_segments(segments, doc.metadata)

        # Step 4: Add chunk index to metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)

        return chunks

    @staticmethod
    def _split_paragraphs(text: str) -> List[str]:
        """Split text into paragraphs, filtering out empty ones."""
        paragraphs = text.split("\n\n")
        return [p.strip() for p in paragraphs if p.strip()]

    def _handle_oversized(self, paragraphs: List[str]) -> List[str]:
        """Split any paragraph exceeding max_chunk_size using the fallback splitter."""
        result: List[str] = []
        for para in paragraphs:
            if len(para) <= self.max_chunk_size:
                result.append(para)
            else:
                # Use recursive splitter on oversized paragraphs
                sub_docs = self._fallback_splitter.create_documents(
                    [para], metadatas=[{}]
                )
                result.extend([d.page_content for d in sub_docs])
        return result

    def _merge_segments(
        self, segments: List[str], base_metadata: dict
    ) -> List[Document]:
        """Merge segments into chunks respecting chunk_size with overlap.

        Uses a simple greedy approach: add segments to the current chunk
        until adding the next one would exceed chunk_size; then start a
        new chunk with overlap from the previous chunk's tail.
        """
        if not segments:
            return []

        chunks: List[Document] = []
        current_parts: List[str] = []
        current_len = 0

        for seg in segments:
            seg_len = len(seg)
            # If adding this segment exceeds chunk_size, finalize current chunk
            if current_parts and current_len + seg_len > self.chunk_size:
                chunk_text = "\n\n".join(current_parts)
                chunks.append(self._make_doc(chunk_text, base_metadata))
                # Start new chunk with overlap
                current_parts, current_len = self._create_overlap_parts(current_parts)

            current_parts.append(seg)
            current_len += seg_len

        # Don't forget the last chunk
        if current_parts:
            chunk_text = "\n\n".join(current_parts)
            chunks.append(self._make_doc(chunk_text, base_metadata))

        return chunks

    def _create_overlap_parts(self, prev_parts: List[str]) -> Tuple[List[str], int]:
        """Create overlap prefix for the next chunk from the previous chunk's tail."""
        if self.chunk_overlap <= 0:
            return [], 0

        # Take characters from the end of the last chunk's text
        prev_text = "\n\n".join(prev_parts)
        overlap_text = prev_text[-self.chunk_overlap:]

        # Try to start at a clean boundary
        boundary = self._find_nearest_sentence_boundary(overlap_text)
        if boundary > 0:
            overlap_text = overlap_text[boundary:]

        return [overlap_text] if overlap_text else [], len(overlap_text)

    def _find_nearest_sentence_boundary(self, text: str) -> int:
        """Find the nearest sentence boundary from the start of text."""
        match = self._SENTENCE_END.search(text)
        if match:
            return match.end()
        return 0

    @staticmethod
    def _make_doc(text: str, metadata: dict) -> Document:
        """Create a Document with a copy of the source metadata."""
        return Document(
            page_content=text,
            metadata={**metadata, "chunk_char_count": len(text)},
        )
