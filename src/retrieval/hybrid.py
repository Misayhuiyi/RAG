"""Hybrid retrieval orchestrator.

Combines BM25 keyword search and vector semantic search with
weighted score fusion and optional cross-encoder re-ranking.
"""

import logging
from typing import Dict, List, Optional

import numpy as np

from .bm25 import BM25Retriever
from .reranker import Reranker
from .vector import SearchResult, VectorRetriever

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Orchestrate hybrid BM25 + vector retrieval with re-ranking.

    Pipeline:
    1. Parallel BM25 + vector search (top_k1 candidates each)
    2. Min-max score normalization
    3. Weighted fusion: score = α * bm25 + (1-α) * vector
    4. De-duplicate by chunk_id, keep best score
    5. Take top_k2 candidates
    6. (Optional) Cross-encoder re-rank
    7. Return top_k3 final results

    Usage:
        hybrid = HybridRetriever(bm25, vector, reranker, config)
        results = hybrid.retrieve("如何配置网络？")
    """

    def __init__(
        self,
        bm25: BM25Retriever,
        vector: VectorRetriever,
        reranker: Optional[Reranker] = None,
        bm25_weight: float = 0.3,
        vector_weight: float = 0.7,
        top_k_retrieval: int = 20,
        top_k_fusion: int = 10,
        top_k_final: int = 5,
        rerank_enabled: bool = True,
    ):
        self.bm25 = bm25
        self.vector = vector
        self.reranker = reranker

        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.top_k_retrieval = top_k_retrieval
        self.top_k_fusion = top_k_fusion
        self.top_k_final = top_k_final
        self.rerank_enabled = rerank_enabled and reranker is not None

    def retrieve(self, query: str) -> List[SearchResult]:
        """Execute the full hybrid retrieval pipeline.

        Args:
            query: The user's natural language question.

        Returns:
            List of SearchResult objects, sorted by descending relevance.
        """
        # Step 1: Parallel retrieval (can be made truly parallel with threads)
        logger.debug("Hybrid retrieve: query='%s'", query[:100])

        bm25_results = self.bm25.search(query, self.top_k_retrieval)
        logger.debug("BM25 returned %d results", len(bm25_results))

        vector_results: List[SearchResult] = []
        if self.vector.collection_exists() and self.vector.size > 0:
            # Need embedding engine from outside; caller must pre-compute or
            # we accept results from an external embedder.
            # The embedding is injected via retrieve_with_embedding().
            pass
        logger.debug("Vector returned %d results", len(vector_results))

        # Step 2-5: Fusion, dedup, and top-k
        fused = self._fuse_and_dedup(bm25_results, vector_results)
        fused = fused[:self.top_k_fusion]

        # Step 6: Re-rank
        if self.rerank_enabled and self.reranker is not None and fused:
            fused = self.reranker.rerank(query, fused, self.top_k_final)
        else:
            fused = fused[:self.top_k_final]

        logger.debug("Hybrid final: %d results", len(fused))
        return fused

    def retrieve_with_embedding(
        self,
        query: str,
        query_embedding: List[float],
    ) -> List[SearchResult]:
        """Full hybrid retrieval with pre-computed query embedding.

        This is the primary entry point when using an external EmbeddingEngine.
        """
        # Step 1: Parallel retrieval
        bm25_results = self.bm25.search(query, self.top_k_retrieval)
        logger.debug("BM25 returned %d results", len(bm25_results))

        vector_results: List[SearchResult] = []
        if self.vector.collection_exists() and self.vector.size > 0:
            vector_results = self.vector.search(query_embedding, self.top_k_retrieval)
        logger.debug("Vector returned %d results", len(vector_results))

        # Step 2-5: Fusion, dedup, and top-k
        fused = self._fuse_and_dedup(bm25_results, vector_results)
        fused = fused[:self.top_k_fusion]

        # Step 6: Re-rank
        if self.rerank_enabled and self.reranker is not None and fused:
            fused = self.reranker.rerank(query, fused, self.top_k_final)
        else:
            fused = fused[:self.top_k_final]

        logger.debug("Hybrid final: %d results", len(fused))
        return fused

    # ── Score fusion ─────────────────────────────────────────────────

    def _fuse_and_dedup(
        self,
        bm25_results: List[SearchResult],
        vector_results: List[SearchResult],
    ) -> List[SearchResult]:
        """Fuse BM25 and vector results with weighted score normalization.

        Steps:
        1. Min-max normalize each result set's scores
        2. Apply weights: score = α * bm25_norm + (1-α) * vector_norm
        3. De-duplicate by chunk_id, keeping the highest fused score
        4. Sort descending by fused score
        """
        # Normalize scores
        bm25_norm = self._min_max_normalize([r.score for r in bm25_results])
        vector_norm = self._min_max_normalize([r.score for r in vector_results])

        # Build a dict: chunk_id -> (fused_score, result)
        fused_map: Dict[str, SearchResult] = {}

        # Process BM25 results
        for score_norm, result in zip(bm25_norm, bm25_results):
            cid = result.chunk_id
            fused_score = self.bm25_weight * score_norm
            if cid not in fused_map or fused_score > fused_map[cid].score:
                result.score = fused_score
                fused_map[cid] = result

        # Process vector results
        for score_norm, result in zip(vector_norm, vector_results):
            cid = result.chunk_id
            fused_score = self.vector_weight * score_norm
            if cid not in fused_map or fused_score > fused_map[cid].score:
                result.score = fused_score
                fused_map[cid] = result

        # Sort by fused score descending
        sorted_results = sorted(
            fused_map.values(), key=lambda r: r.score, reverse=True
        )
        return sorted_results

    # ── Score normalization ──────────────────────────────────────────

    @staticmethod
    def _min_max_normalize(scores: List[float]) -> List[float]:
        """Min-max normalize a list of scores to [0, 1].

        Returns all zeros if all scores are identical.
        """
        if not scores:
            return []

        scores_arr = np.array(scores, dtype=np.float64)
        min_val = scores_arr.min()
        max_val = scores_arr.max()

        if max_val == min_val:
            return [0.0] * len(scores)

        normalized = (scores_arr - min_val) / (max_val - min_val)
        return normalized.tolist()
