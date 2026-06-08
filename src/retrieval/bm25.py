"""BM25 keyword-based document retriever.

Uses the rank_bm25 library with jieba tokenization for Chinese text.
Maintains an in-memory index with persistence via pickle serialization.
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from .vector import SearchResult

logger = logging.getLogger(__name__)


class BM25Retriever:
    """Keyword-based retrieval using BM25 algorithm.

    Tokenizes Chinese text with jieba for effective keyword matching.
    The index lives in memory and can be persisted to disk.

    Usage:
        bm25 = BM25Retriever()
        bm25.build_index(documents)
        results = bm25.search("如何配置网络？", top_k=10)
    """

    def __init__(self):
        self._index: Optional[BM25Okapi] = None
        self._documents: List[Document] = []
        self._tokenized_corpus: List[List[str]] = []
        self._is_built = False

        # Ensure jieba is initialized
        self._init_jieba()

    @staticmethod
    def _init_jieba() -> None:
        """Pre-load jieba to avoid first-call latency."""
        try:
            import jieba
            jieba.initialize()
        except Exception:
            pass

    @property
    def is_built(self) -> bool:
        return self._is_built

    @property
    def doc_count(self) -> int:
        return len(self._documents)

    # ── Index building ───────────────────────────────────────────────

    def build_index(self, documents: List[Document]) -> None:
        """Build the BM25 index from a list of documents.

        Args:
            documents: List of LangChain Documents (post-chunking).
        """
        if not documents:
            logger.warning("No documents provided for BM25 indexing.")
            return

        logger.info("Building BM25 index for %d documents ...", len(documents))

        self._documents = list(documents)
        self._tokenized_corpus = self._tokenize_batch(
            [d.page_content for d in self._documents]
        )
        self._index = BM25Okapi(self._tokenized_corpus)
        self._is_built = True

        logger.info("BM25 index built. Corpus size: %d docs.", len(self._documents))

    def rebuild_index(self, documents: List[Document]) -> None:
        """Alias for build_index — replaces the current index."""
        self._documents = []
        self._tokenized_corpus = []
        self._index = None
        self._is_built = False
        self.build_index(documents)

    # ── Search ───────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Search the BM25 index for documents matching the query.

        Args:
            query: Natural language query string.
            top_k: Number of results to return.

        Returns:
            List of SearchResult objects sorted by descending BM25 score.
        """
        if not self._is_built or self._index is None:
            logger.warning("BM25 index not built. Returning empty results.")
            return []

        tokenized_query = self._tokenize(query)
        scores = self._index.get_scores(tokenized_query)

        # Get indices of top-K highest scores
        if len(scores) == 0:
            return []

        top_k = min(top_k, len(scores))
        # argsort descending
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results: List[SearchResult] = []
        for idx in sorted_indices[:top_k]:
            doc = self._documents[idx]
            results.append(SearchResult(
                chunk_id=doc.metadata.get("chunk_id", f"bm25_{idx}"),
                text=doc.page_content,
                score=float(scores[idx]),
                metadata=dict(doc.metadata),
            ))

        return results

    # ── Tokenization ─────────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenize text using jieba for Chinese, split on whitespace for others."""
        import jieba

        # Use jieba for Chinese text
        tokens = jieba.lcut(text)
        # Remove whitespace-only tokens and single-char tokens
        return [t.strip() for t in tokens if t.strip() and len(t.strip()) > 1]

    def _tokenize_batch(self, texts: List[str]) -> List[List[str]]:
        """Tokenize a batch of texts."""
        return [self._tokenize(t) for t in texts]

    # ── Persistence ──────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Serialize the BM25 index and documents to disk via pickle.

        Note: This saves the tokenized corpus and documents, not the
        BM25Okapi object itself (which may not pickle cleanly).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data: Dict[str, Any] = {
            "documents": self._documents,
            "tokenized_corpus": self._tokenized_corpus,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info("BM25 index saved to %s (%d docs)", path, len(self._documents))

    def load(self, path: str | Path) -> None:
        """Load a previously saved BM25 index from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"BM25 index file not found: {path}")

        with open(path, "rb") as f:
            data = pickle.load(f)

        self._documents = data["documents"]
        self._tokenized_corpus = data["tokenized_corpus"]
        self._index = BM25Okapi(self._tokenized_corpus)
        self._is_built = True

        logger.info("BM25 index loaded from %s (%d docs)", path, len(self._documents))

    # ── Utility ──────────────────────────────────────────────────────

    def get_document(self, idx: int) -> Optional[Document]:
        """Return the document at the given corpus index."""
        if 0 <= idx < len(self._documents):
            return self._documents[idx]
        return None
