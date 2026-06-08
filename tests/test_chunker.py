"""Tests for the SemanticChunker module."""

import pytest
from langchain_core.documents import Document

from src.document.chunker import SemanticChunker


def make_doc(text: str, **meta) -> Document:
    return Document(page_content=text, metadata=meta)


class TestSemanticChunker:
    """Test the semantic chunking strategy."""

    def test_empty_input(self):
        chunker = SemanticChunker(chunk_size=512)
        result = chunker.split([])
        assert result == []

    def test_single_short_doc(self):
        chunker = SemanticChunker(chunk_size=512, chunk_overlap=64)
        doc = make_doc("这是一段简短的测试文本。", source="test.md")
        result = chunker.split([doc])
        assert len(result) == 1
        assert result[0].page_content == "这是一段简短的测试文本。"

    def test_paragraph_boundary_split(self):
        chunker = SemanticChunker(chunk_size=60, chunk_overlap=10)
        doc = make_doc(
            "这是第一段内容。\n\n这是第二段内容，与第一段不同。\n\n这是第三段。",
            source="test.md"
        )
        result = chunker.split([doc])
        # Should split at paragraph boundaries
        assert len(result) >= 1

    def test_metadata_propagation(self):
        chunker = SemanticChunker(chunk_size=512)
        doc = make_doc(
            "段落A\n\n段落B",
            source="/path/to/doc.pdf",
            file_name="doc.pdf",
            page=1,
        )
        result = chunker.split([doc])
        for i, chunk in enumerate(result):
            assert chunk.metadata["source"] == "/path/to/doc.pdf"
            assert chunk.metadata["file_name"] == "doc.pdf"
            assert "chunk_index" in chunk.metadata
            assert "total_chunks" in chunk.metadata

    def test_chunk_indices(self):
        chunker = SemanticChunker(chunk_size=50, chunk_overlap=10, max_chunk_size=60)
        # Create paragraphs so the chunker has boundaries to split on
        long_text = "\n\n".join(["段落A内容。" for _ in range(20)])
        doc = make_doc(long_text, source="long.md")
        result = chunker.split([doc])
        # With 20 paragraphs and chunk_size=50, we should get multiple chunks
        assert len(result) > 1
        for i, chunk in enumerate(result):
            assert chunk.metadata["chunk_index"] == i
            assert chunk.metadata["total_chunks"] == len(result)

    def test_oversized_paragraph_fallback(self):
        chunker = SemanticChunker(
            chunk_size=100,
            chunk_overlap=20,
            max_chunk_size=200,
        )
        very_long_para = "这是一个很长的段落。" * 50  # ~500 chars
        doc = make_doc(very_long_para, source="long.md")
        result = chunker.split([doc])
        assert len(result) > 1
        for chunk in result:
            assert len(chunk.page_content) <= 200

    def test_multiple_documents(self):
        chunker = SemanticChunker(chunk_size=200, chunk_overlap=30)
        docs = [
            make_doc("文档一内容。" * 5, source="doc1.md"),
            make_doc("文档二内容。" * 5, source="doc2.md"),
        ]
        result = chunker.split(docs)
        assert len(result) >= 2
        sources = set(c.metadata["source"] for c in result)
        assert "doc1.md" in sources
        assert "doc2.md" in sources
