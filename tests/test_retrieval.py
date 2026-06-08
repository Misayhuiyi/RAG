"""Tests for the retrieval modules (BM25, hybrid fusion)."""

import pytest
from langchain_core.documents import Document

from src.retrieval.bm25 import BM25Retriever
from src.retrieval.vector import SearchResult


class TestBM25Retriever:
    """Test the BM25 keyword retriever."""

    @pytest.fixture
    def sample_docs(self):
        return [
            Document(page_content="公司年假政策：员工每年享有5天带薪年假。", metadata={"file_name": "handbook.md", "chunk_index": 0}),
            Document(page_content="设备借用流程：填写申请表，经导师签字后提交。", metadata={"file_name": "equipment.md", "chunk_index": 0}),
            Document(page_content="数据安全培训每季度进行一次。", metadata={"file_name": "security.md", "chunk_index": 0}),
            Document(page_content="年终奖在每年1月发放，金额为1-3个月工资。", metadata={"file_name": "handbook.md", "chunk_index": 1}),
        ]

    def test_build_index(self, sample_docs):
        bm25 = BM25Retriever()
        bm25.build_index(sample_docs)
        assert bm25.is_built
        assert bm25.doc_count == 4

    def test_search_returns_results(self, sample_docs):
        bm25 = BM25Retriever()
        bm25.build_index(sample_docs)
        results = bm25.search("年假政策", top_k=3)
        assert len(results) > 0
        assert isinstance(results[0], SearchResult)
        # The first doc should be about annual leave
        assert results[0].score > 0

    def test_search_empty_index(self):
        bm25 = BM25Retriever()
        results = bm25.search("测试")
        assert results == []

    def test_search_top_k_limit(self, sample_docs):
        bm25 = BM25Retriever()
        bm25.build_index(sample_docs)
        results = bm25.search("政策", top_k=2)
        assert len(results) <= 2

    def test_search_result_metadata(self, sample_docs):
        bm25 = BM25Retriever()
        bm25.build_index(sample_docs)
        results = bm25.search("设备", top_k=1)
        assert len(results) == 1
        assert results[0].document_name == "equipment.md"
        assert "设备" in results[0].text

    def test_rebuild_index(self, sample_docs):
        bm25 = BM25Retriever()
        bm25.build_index(sample_docs)
        assert bm25.doc_count == 4
        new_docs = [Document(page_content="新文档", metadata={"file_name": "new.md"})]
        bm25.rebuild_index(new_docs)
        assert bm25.doc_count == 1


class TestHybridFusion:
    """Test the hybrid score fusion logic."""

    @pytest.fixture
    def bm25_results(self):
        return [
            SearchResult(chunk_id="1", text="年假政策内容", score=2.5, metadata={"file_name": "h.md"}),
            SearchResult(chunk_id="2", text="设备借用", score=1.2, metadata={"file_name": "e.md"}),
        ]

    @pytest.fixture
    def vector_results(self):
        return [
            SearchResult(chunk_id="2", text="设备借用", score=0.95, metadata={"file_name": "e.md"}),
            SearchResult(chunk_id="3", text="安全培训", score=0.80, metadata={"file_name": "s.md"}),
        ]

    def test_fusion_dedup(self, bm25_results, vector_results):
        """Test that fusion deduplicates by chunk_id."""
        from src.retrieval.hybrid import HybridRetriever
        from src.retrieval.bm25 import BM25Retriever
        from src.retrieval.vector import VectorRetriever

        # We're testing the internal fusion method directly
        hybrid = HybridRetriever(
            bm25=BM25Retriever(),
            vector=VectorRetriever(),
            bm25_weight=0.3,
            vector_weight=0.7,
        )

        fused = hybrid._fuse_and_dedup(bm25_results, vector_results)
        # Should dedup chunk "2" — only 2 unique chunks remain
        assert len(fused) == 3  # chunk_id 1, 2, 3
        chunk_ids = [r.chunk_id for r in fused]
        assert "1" in chunk_ids
        assert "2" in chunk_ids
        assert "3" in chunk_ids

    def test_min_max_normalize_zero_range(self):
        from src.retrieval.hybrid import HybridRetriever
        from src.retrieval.vector import VectorRetriever
        hybrid = HybridRetriever(
            bm25=BM25Retriever(),
            vector=VectorRetriever(),
        )
        result = hybrid._min_max_normalize([5.0, 5.0, 5.0])
        assert result == [0.0, 0.0, 0.0]

    def test_min_max_normalize(self):
        from src.retrieval.hybrid import HybridRetriever
        from src.retrieval.vector import VectorRetriever
        hybrid = HybridRetriever(
            bm25=BM25Retriever(),
            vector=VectorRetriever(),
        )
        result = hybrid._min_max_normalize([0.0, 50.0, 100.0])
        assert result == [0.0, 0.5, 1.0]
