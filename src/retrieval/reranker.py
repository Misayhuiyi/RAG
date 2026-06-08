"""Cross-encoder reranker for refining retrieval results.

Uses transformers AutoModelForSequenceClassification natively to avoid
sentence-transformers CrossEncoder compatibility issues with BGE models.
"""

import logging
import os
from typing import List, Optional, Tuple

import torch

# Use local cache only — model must be pre-downloaded
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from .vector import SearchResult

logger = logging.getLogger(__name__)


class Reranker:
    """Re-rank candidate documents using a cross-encoder model.

    The cross-encoder takes (query, document) pairs and outputs a fine-grained
    relevance score. This is more accurate than bi-encoder (embedding) scores
    but slower — so we only re-rank the top fusion candidates.

    Uses transformers AutoModelForSequenceClassification directly for maximum
    compatibility with all reranker model variants.

    Usage:
        reranker = Reranker("BAAI/bge-reranker-large")
        reranked = reranker.rerank(query, candidates)
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-large",
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.device = device
        self._model: Optional[object] = None
        self._tokenizer: Optional[object] = None

    @property
    def model(self):
        """Lazy-load the model."""
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def tokenizer(self):
        """Lazy-load the tokenizer."""
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer

    def _load_model(self) -> None:
        """Load the cross-encoder model from HuggingFace using AutoModel."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logger.info("Loading reranker model: %s on %s ...", self.model_name, self.device)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            local_files_only=True,
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            local_files_only=True,
        )
        self._model.to(self.device)
        self._model.eval()

        logger.info("Reranker model loaded.")

    def rerank(
        self,
        query: str,
        candidates: List[SearchResult],
        top_k: Optional[int] = None,
    ) -> List[SearchResult]:
        """Re-rank candidate search results using cross-encoder scoring.

        Args:
            query: The search query.
            candidates: List of candidate search results to re-rank.
            top_k: Number of top results to return after re-ranking.
                   If None, returns all candidates in new order.

        Returns:
            Re-ordered list of SearchResult objects with updated scores.
        """
        if not candidates:
            return []

        if self._model is None:
            self._load_model()

        # Build (query, doc) pairs
        pairs: List[Tuple[str, str]] = [(query, c.text) for c in candidates]

        # Tokenize all pairs
        all_scores: List[float] = []
        batch_size = 16

        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                # Logits shape: (batch, 1) — take the positive class score
                logits = outputs.logits.squeeze(-1)
                batch_scores = logits.cpu().tolist()

            # Handle single-item batch
            if isinstance(batch_scores, float):
                batch_scores = [batch_scores]

            all_scores.extend(batch_scores)

        # Attach new scores and sort
        scored: List[Tuple[float, SearchResult]] = []
        for score, candidate in zip(all_scores, candidates):
            candidate.score = float(score)
            scored.append((float(score), candidate))

        # Sort descending by cross-encoder score
        scored.sort(key=lambda x: x[0], reverse=True)

        result = [sr for _, sr in scored]

        if top_k is not None:
            result = result[:top_k]

        # Normalize reranker scores to [0, 1] using sigmoid
        for r in result:
            r.score = self._sigmoid(r.score)

        return result

    def rerank_with_threshold(
        self,
        query: str,
        candidates: List[SearchResult],
        threshold: float = 0.5,
        top_k: Optional[int] = None,
    ) -> List[SearchResult]:
        """Re-rank and filter by a minimum relevance threshold.

        Args:
            query: The search query.
            candidates: List of candidate search results.
            threshold: Minimum sigmoid-normalized score to keep (0.0-1.0).
            top_k: Maximum number of results to return.

        Returns:
            Filtered, re-ranked results above the threshold.
        """
        reranked = self.rerank(query, candidates)
        filtered = [r for r in reranked if r.score >= threshold]
        if top_k:
            filtered = filtered[:top_k]
        return filtered

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Sigmoid function to normalize scores to (0, 1)."""
        import numpy as np
        return float(1.0 / (1.0 + np.exp(-x)))
