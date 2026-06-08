"""HuggingFace embedding engine using sentence-transformers.

Wraps BGE and other HF embedding models behind a unified interface.
Supports batch embedding with progress tracking and GPU/CPU modes.
"""

import logging
import os
from typing import List, Optional

import numpy as np

# Use local cache only — model must be pre-downloaded
os.environ.setdefault("HF_HUB_OFFLINE", "1")

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """Generate embeddings using HuggingFace sentence-transformers models.

    Usage:
        engine = EmbeddingEngine("BAAI/bge-large-zh-v1.5", device="cpu")
        embeddings = engine.embed_texts(["文档内容1", "文档内容2"])
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-zh-v1.5",
        device: str = "cpu",
        batch_size: int = 32,
        normalize: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self._model: Optional[object] = None
        self._dimension: Optional[int] = None

    @property
    def model(self):
        """Lazy-load the sentence-transformers model."""
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def dimension(self) -> int:
        """Return the embedding vector dimension."""
        if self._dimension is None:
            self._load_model()
        assert self._dimension is not None
        return self._dimension

    def _load_model(self) -> None:
        """Load the sentence-transformers model from HuggingFace."""
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model: %s on %s ...", self.model_name, self.device)
        self._model = SentenceTransformer(
            self.model_name,
            device=self.device,
            local_files_only=True,
        )
        # Determine dimension
        test_vec = self._model.encode(["test"], show_progress_bar=False)
        self._dimension = test_vec.shape[1]
        logger.info("Embedding model loaded. Dimension: %d", self._dimension)

    def embed_texts(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> List[List[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.
            show_progress: Whether to show a tqdm progress bar.

        Returns:
            List of embedding vectors as float lists.
        """
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )

        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query string.

        For BGE models, prepends the instruction prefix for asymmetric tasks.
        """
        # BGE models benefit from query instruction prefix
        if "bge" in self.model_name.lower():
            query = f"为这个句子生成表示以用于检索相关文章：{query}"

        embeddings = self.model.encode(
            [query],
            batch_size=1,
            show_progress_bar=False,
            normalize_embeddings=self.normalize,
        )
        return embeddings[0].tolist()

    def embed_queries(self, queries: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple queries."""
        if not queries:
            return []

        # Apply BGE query prefix
        if "bge" in self.model_name.lower():
            queries = [f"为这个句子生成表示以用于检索相关文章：{q}" for q in queries]

        embeddings = self.model.encode(
            queries,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.normalize,
        )
        return embeddings.tolist()

    def compute_similarity(
        self,
        query_embedding: List[float],
        doc_embeddings: List[List[float]],
    ) -> List[float]:
        """Compute cosine similarities between a query and documents."""
        q = np.array(query_embedding)
        d = np.array(doc_embeddings)
        # Cosine similarity for normalized vectors is just dot product
        if self.normalize:
            return (d @ q).tolist()
        # Otherwise compute full cosine
        q_norm = q / np.linalg.norm(q)
        d_norm = d / np.linalg.norm(d, axis=1, keepdims=True)
        return (d_norm @ q_norm).tolist()
