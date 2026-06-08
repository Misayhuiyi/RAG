"""Query pipeline.

Orchestrates the full Q&A flow:
  question → hybrid retrieve → rerank → build prompt → LLM generate → answer + sources
"""

import logging
import time
from typing import Any, Dict, List, Optional

from ..config.loader import ConfigLoader
from ..embedding.embedder import EmbeddingEngine
from ..llm.base import BaseLLM, LLMResponse, Message
from ..llm.ollama_llm import OllamaLLM
from ..llm.openai_llm import OpenAILLM
from ..memory.conversation import ConversationMemory
from ..retrieval.bm25 import BM25Retriever
from ..retrieval.hybrid import HybridRetriever
from ..retrieval.reranker import Reranker
from ..retrieval.vector import SearchResult, VectorRetriever

logger = logging.getLogger(__name__)


class QueryPipeline:
    """Full Q&A query pipeline.

    Usage:
        config = ConfigLoader("config.yaml")
        pipeline = QueryPipeline(config, bm25, vector_store, memory)
        response = await pipeline.answer("如何配置网络？", session_id="user123")
    """

    def __init__(
        self,
        config: ConfigLoader,
        bm25: BM25Retriever,
        vector_store: VectorRetriever,
        memory: ConversationMemory,
    ):
        self.config = config

        # Embedder
        self.embedder = EmbeddingEngine(
            model_name=config.embedding_model_name,
            device=config.embedding_device,
            batch_size=config.embedding_batch_size,
            normalize=config.embedding_normalize,
        )

        # Reranker (optional)
        self.reranker: Optional[Reranker] = None
        if config.retrieval_rerank_enabled:
            self.reranker = Reranker(
                model_name=config.retrieval_rerank_model,
                device=config.embedding_device,
            )

        # Hybrid retriever
        self.hybrid = HybridRetriever(
            bm25=bm25,
            vector=vector_store,
            reranker=self.reranker,
            bm25_weight=config.retrieval_bm25_weight,
            vector_weight=config.retrieval_vector_weight,
            top_k_retrieval=config.retrieval_top_k_retrieval,
            top_k_fusion=config.retrieval_top_k_fusion,
            top_k_final=config.retrieval_top_k_final,
            rerank_enabled=config.retrieval_rerank_enabled,
        )

        # LLM
        self.llm = self._create_llm()

        # Memory
        self.memory = memory

    def _create_llm(self) -> BaseLLM:
        """Create the LLM provider based on configuration."""
        provider = self.config.llm_provider
        cfg = self.config.get_llm_provider_config()

        if provider == "ollama":
            return OllamaLLM(
                model=cfg.get("model", "qwen2.5:7b"),
                base_url=cfg.get("base_url", "http://localhost:11434"),
                temperature=cfg.get("temperature", 0.1),
                max_tokens=cfg.get("max_tokens", 2048),
            )
        else:  # openai (default)
            return OpenAILLM(
                model=cfg.get("model", "gpt-4o"),
                api_key=cfg.get("api_key", ""),
                base_url=cfg.get("base_url", None),
                temperature=cfg.get("temperature", 0.1),
                max_tokens=cfg.get("max_tokens", 2048),
            )

    # ── Main API ─────────────────────────────────────────────────────

    async def answer(
        self,
        question: str,
        session_id: str = "default",
        top_k: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Answer a question using the full RAG pipeline.

        Args:
            question: The user's question.
            session_id: Conversation session ID for multi-turn memory.
            top_k: Override the configured top_k_final.
            temperature: Override LLM temperature.

        Returns:
            Dict with keys: answer, sources, session_id, processing_time_ms,
                            context_docs, model, usage.
        """
        start = time.time()

        # Step 1: Embed the query
        query_embedding = self.embedder.embed_query(question)

        # Step 2: Hybrid retrieval
        retrieved: List[SearchResult] = self.hybrid.retrieve_with_embedding(
            query=question,
            query_embedding=query_embedding,
        )

        if top_k:
            retrieved = retrieved[:top_k]

        # Step 3: Get conversation history
        history = self.memory.get_context(session_id)

        # Step 4: Generate answer via LLM
        llm_response: LLMResponse = await self.llm.generate(
            prompt=question,
            context=retrieved,
            history=history,
            system_prompt=self.config.prompt_system,
            user_template=self.config.prompt_user_template,
        )

        # Step 5: Build sources for traceability
        sources = self._build_sources(retrieved)

        # Step 6: Update conversation memory
        self.memory.add_turn(
            session_id=session_id,
            question=question,
            answer=llm_response.answer,
            sources=sources,
        )

        elapsed_ms = round((time.time() - start) * 1000)

        return {
            "answer": llm_response.answer,
            "sources": sources,
            "session_id": session_id,
            "processing_time_ms": elapsed_ms,
            "context_docs": len(retrieved),
            "model": llm_response.model,
            "usage": llm_response.usage,
        }

    async def search_only(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Perform retrieval only, without LLM generation.

        Useful for debugging retrieval quality.
        """
        query_embedding = self.embedder.embed_query(query)
        results = self.hybrid.retrieve_with_embedding(query, query_embedding)

        if top_k:
            results = results[:top_k]

        return self._build_sources(results)

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _build_sources(results: List[SearchResult]) -> List[Dict[str, Any]]:
        """Build the source documents list for the API response."""
        return [
            {
                "chunk_id": r.chunk_id,
                "document_name": r.document_name,
                "content": r.text[:500],  # Truncate for response
                "full_content": r.text,
                "score": round(r.score, 4),
                "chunk_index": r.chunk_index,
                "source_path": r.metadata.get("source", ""),
            }
            for r in results
        ]

    def get_prompt_preview(
        self,
        question: str,
        context_docs: List[SearchResult],
        session_id: str = "default",
    ) -> str:
        """Build a preview of the full prompt sent to the LLM (for debugging)."""
        history = self.memory.get_context(session_id)
        context_text = self.llm._build_context_text(context_docs)
        history_text = self.llm._build_history_text(history)
        return self.llm._format_prompt(
            question=question,
            context_text=context_text,
            history_text=history_text,
            user_template=self.config.prompt_user_template,
        )
