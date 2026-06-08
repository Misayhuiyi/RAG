"""Document ingestion pipeline.

Orchestrates the full ingestion flow:
  documents → clean → chunk → embed → store (Milvus + BM25)
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from ..config.loader import ConfigLoader
from ..document.chunker import SemanticChunker
from ..document.cleaner import TextCleaner
from ..document.loader import DocumentLoader
from ..embedding.embedder import EmbeddingEngine
from ..retrieval.bm25 import BM25Retriever
from ..retrieval.vector import VectorRetriever

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    """Result of an ingestion run."""

    documents_loaded: int = 0
    chunks_created: int = 0
    vectors_stored: int = 0
    elapsed_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0


class IngestionPipeline:
    """Full document ingestion pipeline.

    Usage:
        config = ConfigLoader("config.yaml")
        pipeline = IngestionPipeline(config)
        result = pipeline.run("./data/documents")
    """

    def __init__(self, config: ConfigLoader):
        self.config = config

        self.loader = DocumentLoader(config.supported_formats)
        self.cleaner = TextCleaner(
            remove_headers_footers=True,
            normalize_punct=True,
        )
        self.chunker = SemanticChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            max_chunk_size=config.max_chunk_size,
        )
        self.embedder = EmbeddingEngine(
            model_name=config.embedding_model_name,
            device=config.embedding_device,
            batch_size=config.embedding_batch_size,
            normalize=config.embedding_normalize,
        )
        self.vector_store = VectorRetriever(
            uri=config.milvus_uri,
            collection_name=config.milvus_collection_name,
            metric_type=config.milvus_metric_type,
            index_type=config.milvus_index_type,
            nlist=config.milvus_nlist,
        )
        self.bm25 = BM25Retriever()

    def run(
        self,
        doc_dir: Optional[str] = None,
        drop_existing: bool = False,
    ) -> IngestResult:
        """Run the full ingestion pipeline.

        Args:
            doc_dir: Directory containing documents. Defaults to config value.
            drop_existing: If True, drop and recreate the Milvus collection.

        Returns:
            IngestResult with statistics.
        """
        result = IngestResult()
        start = time.time()

        doc_dir = doc_dir or self.config.source_dir
        doc_path = Path(doc_dir)

        if not doc_path.exists():
            msg = f"Document directory not found: {doc_dir}"
            logger.error(msg)
            result.errors.append(msg)
            return result

        try:
            # Step 1: Load documents
            logger.info("=" * 50)
            logger.info("Step 1/5: Loading documents from %s ...", doc_dir)
            raw_docs = self.loader.load_directory(doc_dir)
            result.documents_loaded = len(raw_docs)
            logger.info("Loaded %d raw documents.", len(raw_docs))
            if not raw_docs:
                msg = "No documents found to ingest."
                logger.warning(msg)
                result.errors.append(msg)
                return result

            # Step 2: Clean text
            logger.info("Step 2/5: Cleaning text ...")
            for doc in raw_docs:
                doc.page_content = self.cleaner.clean(doc.page_content)
            # Filter out empty docs after cleaning
            raw_docs = [d for d in raw_docs if d.page_content.strip()]
            logger.info("Cleaned %d documents.", len(raw_docs))

            # Step 3: Chunk
            logger.info("Step 3/5: Chunking documents ...")
            chunks = self.chunker.split(raw_docs)
            result.chunks_created = len(chunks)
            logger.info("Created %d chunks.", len(chunks))

            # Step 4: Embed
            logger.info("Step 4/5: Generating embeddings ...")
            texts = [c.page_content for c in chunks]
            embeddings = self.embedder.embed_texts(texts)
            logger.info("Generated %d embeddings.", len(embeddings))

            # Step 5: Store in Milvus
            logger.info("Step 5/5: Storing in vector database ...")
            self.vector_store.create_collection(
                dim=self.embedder.dimension,
                drop_if_exists=drop_existing,
            )

            metadatas = [c.metadata for c in chunks]
            self.vector_store.insert(embeddings, texts, metadatas)
            self.vector_store.flush()
            result.vectors_stored = len(embeddings)

            # Also build BM25 index
            self.bm25.build_index(chunks)
            # Persist BM25 index for recovery on next startup
            bm25_path = Path(self.config.milvus_uri).parent / "bm25_index.pkl"
            self.bm25.save(bm25_path)
            logger.info("BM25 index saved to %s", bm25_path)
            logger.info("BM25 index built with %d chunks.", len(chunks))

        except Exception as e:
            msg = f"Ingestion error: {e}"
            logger.exception(msg)
            result.errors.append(msg)

        result.elapsed_seconds = round(time.time() - start, 2)
        logger.info(
            "Ingestion complete in %.2fs: %d docs → %d chunks → %d vectors.",
            result.elapsed_seconds,
            result.documents_loaded,
            result.chunks_created,
            result.vectors_stored,
        )
        return result

    def reingest_document(self, file_path: str | Path) -> IngestResult:
        """Re-ingest a single document (delete old chunks, insert new)."""
        file_path = Path(file_path)
        result = IngestResult()
        start = time.time()

        try:
            # Delete old chunks for this file
            self.vector_store.delete_by_file(file_path.name)

            # Load and process the single file
            raw_docs = self.loader.load_file(file_path)
            result.documents_loaded = len(raw_docs)

            for doc in raw_docs:
                doc.page_content = self.cleaner.clean(doc.page_content)
            raw_docs = [d for d in raw_docs if d.page_content.strip()]

            chunks = self.chunker.split(raw_docs)
            result.chunks_created = len(chunks)

            texts = [c.page_content for c in chunks]
            embeddings = self.embedder.embed_texts(texts)

            metadatas = [c.metadata for c in chunks]
            self.vector_store.insert(embeddings, texts, metadatas)
            self.vector_store.flush()
            result.vectors_stored = len(embeddings)

        except Exception as e:
            result.errors.append(str(e))

        result.elapsed_seconds = round(time.time() - start, 2)
        return result

    def get_stats(self) -> Dict[str, Any]:
        """Return current ingestion statistics."""
        return {
            "milvus_collection": self.config.milvus_collection_name,
            "milvus_size": self.vector_store.size,
            "bm25_doc_count": self.bm25.doc_count,
            "embedding_dim": self.embedder.dimension,
        }
