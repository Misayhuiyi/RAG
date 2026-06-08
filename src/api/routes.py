"""API route definitions for the RAG Q&A system."""

import logging
import time
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Request

from ..pipeline.ingest import IngestionPipeline
from ..pipeline.query import QueryPipeline
from .schemas import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    HealthResponse,
    IngestRequest,
    SearchRequest,
    SearchResponse,
    SourceDocument,
    StatsResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["RAG Q&A"])


# ── Dependency helpers ───────────────────────────────────────────────

def _get_app(request: Request):
    return request.app


def _get_pipelines(request: Request):
    app = request.app
    return app.state.query_pipeline, app.state.ingest_pipeline


# ── Chat ─────────────────────────────────────────────────────────────

@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Ask a question",
    description="Submit a question and get an AI-generated answer with source citations. "
                "Supports multi-turn conversation via session_id.",
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def chat(request: ChatRequest, req: Request):
    """Main Q&A endpoint with full RAG pipeline."""
    query_pipeline: QueryPipeline = req.app.state.query_pipeline

    session_id = request.session_id or str(uuid.uuid4())

    try:
        result = await query_pipeline.answer(
            question=request.question,
            session_id=session_id,
            top_k=request.top_k,
            temperature=request.temperature,
        )

        sources = [SourceDocument(**s) for s in result["sources"]]

        return ChatResponse(
            answer=result["answer"],
            sources=sources,
            session_id=result["session_id"],
            processing_time_ms=result["processing_time_ms"],
            context_docs=result["context_docs"],
            model=result.get("model", ""),
            usage=result.get("usage", {}),
        )

    except Exception as e:
        logger.exception("Chat endpoint error")
        raise HTTPException(status_code=500, detail=str(e))


# ── Search Only ──────────────────────────────────────────────────────

@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Retrieve documents without LLM generation",
    description="Search for relevant document chunks using hybrid retrieval. "
                "No LLM generation is performed — useful for debugging retrieval.",
)
async def search(request: SearchRequest, req: Request):
    """Retrieval-only endpoint (no LLM generation)."""
    query_pipeline: QueryPipeline = req.app.state.query_pipeline

    start = time.time()

    try:
        results = await query_pipeline.search_only(
            query=request.query,
            top_k=request.top_k,
        )

        elapsed_ms = round((time.time() - start) * 1000)

        return SearchResponse(
            query=request.query,
            results=[SourceDocument(**r) for r in results],
            total=len(results),
            processing_time_ms=elapsed_ms,
        )

    except Exception as e:
        logger.exception("Search endpoint error")
        raise HTTPException(status_code=500, detail=str(e))


# ── Documents ────────────────────────────────────────────────────────

@router.get(
    "/documents",
    summary="List indexed document statistics",
)
async def list_documents(req: Request):
    """Return statistics about currently indexed documents."""
    ingest_pipeline: IngestionPipeline = req.app.state.ingest_pipeline
    return ingest_pipeline.get_stats()


@router.post(
    "/documents/ingest",
    summary="Trigger document re-ingestion",
    description="Re-ingest documents from the configured or specified directory.",
)
async def ingest_documents(request: IngestRequest, req: Request):
    """Trigger document ingestion."""
    ingest_pipeline: IngestionPipeline = req.app.state.ingest_pipeline

    try:
        result = ingest_pipeline.run(
            doc_dir=request.doc_dir,
            drop_existing=request.drop_existing,
        )

        return {
            "success": result.success,
            "documents_loaded": result.documents_loaded,
            "chunks_created": result.chunks_created,
            "vectors_stored": result.vectors_stored,
            "elapsed_seconds": result.elapsed_seconds,
            "errors": result.errors,
        }

    except Exception as e:
        logger.exception("Ingest endpoint error")
        raise HTTPException(status_code=500, detail=str(e))


# ── Health & Stats ───────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
)
async def health_check(req: Request):
    """Return service health and component status."""
    import src
    ingest_pipeline: IngestionPipeline = req.app.state.ingest_pipeline
    stats = ingest_pipeline.get_stats()

    return HealthResponse(
        status="ok",
        version=src.__version__,
        milvus_connected=stats["milvus_size"] >= 0,
        documents_indexed=stats["milvus_size"],
        bm25_ready=stats["bm25_doc_count"] > 0,
    )


@router.get(
    "/stats",
    response_model=StatsResponse,
    summary="System statistics",
)
async def get_stats(req: Request):
    """Return detailed system statistics."""
    query_pipeline: QueryPipeline = req.app.state.query_pipeline
    ingest_pipeline: IngestionPipeline = req.app.state.ingest_pipeline

    ingest_stats = ingest_pipeline.get_stats()
    mem_stats = query_pipeline.memory.get_stats()

    return StatsResponse(
        milvus_collection=ingest_stats["milvus_collection"],
        milvus_size=ingest_stats["milvus_size"],
        bm25_doc_count=ingest_stats["bm25_doc_count"],
        embedding_dim=ingest_stats["embedding_dim"],
        active_sessions=mem_stats["active_sessions"],
        total_conversation_turns=mem_stats["total_turns"],
        llm_provider=query_pipeline.config.llm_provider,
    )
