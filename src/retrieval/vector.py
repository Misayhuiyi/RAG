"""Milvus Lite vector store retriever.

Manages vector storage and ANN search using Milvus Lite (embedded mode).
Uses the MilvusClient API for compatibility with milvus-lite 3.x.
"""

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pymilvus import DataType, MilvusClient

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result with score and metadata."""

    chunk_id: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def document_name(self) -> str:
        return self.metadata.get("file_name", "unknown")

    @property
    def chunk_index(self) -> int:
        return self.metadata.get("chunk_index", -1)


class VectorRetriever:
    """Milvus Lite vector store for semantic document retrieval.

    Usage:
        vr = VectorRetriever(uri="./data/milvus.db", collection_name="docs")
        vr.create_collection(dim=1024)
        vr.insert(embeddings, chunks)
        results = vr.search(query_embedding, top_k=10)
    """

    def __init__(
        self,
        uri: str = "./data/vector_store/milvus.db",
        collection_name: str = "enterprise_docs",
        metric_type: str = "COSINE",
        index_type: str = "IVF_FLAT",
        nlist: int = 128,
    ):
        self.uri = uri
        self.collection_name = collection_name
        self.metric_type = metric_type
        self.index_type = index_type
        self.nlist = nlist

        self._client: Optional[MilvusClient] = None

    @property
    def client(self) -> MilvusClient:
        """Lazy-initialize the MilvusClient."""
        if self._client is None:
            self._client = MilvusClient(uri=self.uri)
            logger.info("MilvusClient connected to %s", self.uri)
        return self._client

    def flush(self) -> None:
        """Flush pending writes (no-op with MilvusClient — auto-flushed)."""
        pass

    def load_collection(self) -> None:
        """Ensure the collection is loaded into memory for search."""
        if self.client.has_collection(self.collection_name):
            self.client.load_collection(self.collection_name)
            logger.debug("Collection '%s' loaded.", self.collection_name)

    def close(self) -> None:
        """Close the Milvus client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    # ── Collection lifecycle ─────────────────────────────────────────

    def collection_exists(self) -> bool:
        """Check if the collection already exists."""
        return self.client.has_collection(self.collection_name)

    def create_collection(self, dim: int, drop_if_exists: bool = False) -> None:
        """Create a new Milvus collection with the given embedding dimension."""
        if self.client.has_collection(self.collection_name):
            if drop_if_exists:
                logger.info("Dropping existing collection '%s'", self.collection_name)
                self.client.drop_collection(self.collection_name)
            else:
                logger.info("Collection '%s' already exists.", self.collection_name)
                return

        logger.info("Creating collection '%s' with dim=%d", self.collection_name, dim)

        schema = self.client.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
        )
        schema.add_field(
            field_name="id",
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=64,
        )
        schema.add_field(
            field_name="embedding",
            datatype=DataType.FLOAT_VECTOR,
            dim=dim,
        )
        schema.add_field(
            field_name="text",
            datatype=DataType.VARCHAR,
            max_length=65535,
        )
        schema.add_field(
            field_name="file_name",
            datatype=DataType.VARCHAR,
            max_length=512,
        )
        schema.add_field(
            field_name="chunk_index",
            datatype=DataType.INT64,
        )
        schema.add_field(
            field_name="source",
            datatype=DataType.VARCHAR,
            max_length=1024,
        )

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type=self.index_type,
            metric_type=self.metric_type,
            params={"nlist": self.nlist},
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )

        logger.info("Collection '%s' created and ready.", self.collection_name)
        self.load_collection()

    def drop_collection(self) -> None:
        """Drop the collection if it exists."""
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
            logger.info("Collection '%s' dropped.", self.collection_name)

    @property
    def size(self) -> int:
        """Return the number of entities in the collection."""
        try:
            stats = self.client.get_collection_stats(self.collection_name)
            return stats.get("row_count", 0)
        except Exception:
            return 0

    # ── Data operations ──────────────────────────────────────────────

    def insert(
        self,
        embeddings: List[List[float]],
        texts: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> List[str]:
        """Insert chunk embeddings and metadata into Milvus.

        Args:
            embeddings: List of embedding vectors.
            texts: List of chunk text contents.
            metadatas: List of metadata dicts (must contain 'file_name').

        Returns:
            List of inserted chunk IDs.
        """
        if not embeddings:
            return []

        chunk_ids = [str(uuid.uuid4()) for _ in range(len(embeddings))]
        data: List[Dict[str, Any]] = []

        for i, (cid, emb, text, meta) in enumerate(
            zip(chunk_ids, embeddings, texts, metadatas)
        ):
            data.append({
                "id": cid,
                "embedding": emb,
                "text": text,
                "file_name": meta.get("file_name", f"unknown_{i}"),
                "chunk_index": meta.get("chunk_index", i),
                "source": meta.get("source", ""),
            })

        result = self.client.insert(
            collection_name=self.collection_name,
            data=data,
        )
        logger.info("Inserted %d chunks into Milvus (insert_count=%d)", len(data), result.get("insert_count", 0))
        return chunk_ids

    # ── Search ───────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
    ) -> List[SearchResult]:
        """Perform ANN vector search.

        Args:
            query_embedding: The query embedding vector.
            top_k: Number of nearest neighbors to return.

        Returns:
            List of SearchResult objects sorted by descending similarity.
        """
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            anns_field="embedding",
            search_params={"metric_type": self.metric_type, "params": {"nprobe": 16}},
            limit=top_k,
            output_fields=["text", "file_name", "chunk_index", "source"],
        )

        # results is List[List[dict]]; take the first (only) query
        hits = results[0] if results else []

        return [
            SearchResult(
                chunk_id=hit.get("id", ""),
                text=hit.get("entity", {}).get("text", ""),
                score=hit.get("distance", 0.0),
                metadata={
                    "file_name": hit.get("entity", {}).get("file_name", ""),
                    "chunk_index": hit.get("entity", {}).get("chunk_index", -1),
                    "source": hit.get("entity", {}).get("source", ""),
                },
            )
            for hit in hits
        ]

    def delete_by_file(self, file_name: str) -> int:
        """Delete all chunks belonging to a specific document file."""
        expr = f'file_name == "{file_name}"'
        result = self.client.delete(
            collection_name=self.collection_name,
            filter=expr,
        )
        count = len(result) if isinstance(result, list) else result.get("delete_count", 0)
        logger.info("Deleted %d chunks for file '%s'", count, file_name)
        return count
