"""Pydantic request/response schemas for the RAG Q&A API."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Request Models ───────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Request for the Q&A chat endpoint."""

    question: str = Field(
        ...,
        description="The user's question",
        min_length=1,
        max_length=5000,
        examples=["公司年假政策是什么？"],
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Conversation session ID. A new session is created if not provided.",
        examples=["sess_abc123"],
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Override the number of documents to retrieve.",
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Override LLM temperature for this request.",
    )


class SearchRequest(BaseModel):
    """Request for the retrieval-only search endpoint."""

    query: str = Field(
        ...,
        description="The search query",
        min_length=1,
        max_length=5000,
        examples=["实验室安全规范"],
    )
    top_k: Optional[int] = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of results to return.",
    )


class IngestRequest(BaseModel):
    """Request to trigger document re-ingestion."""

    doc_dir: Optional[str] = Field(
        default=None,
        description="Directory containing documents. Uses config default if omitted.",
    )
    drop_existing: bool = Field(
        default=False,
        description="If True, drop and recreate all vector data before ingestion.",
    )


# ── Response Models ──────────────────────────────────────────────────

class SourceDocument(BaseModel):
    """A source document chunk referenced in the answer."""

    chunk_id: str = Field(..., description="Unique chunk identifier")
    document_name: str = Field(..., description="Source document filename")
    content: str = Field(..., description="Truncated chunk content (500 chars)")
    full_content: str = Field(..., description="Full chunk content")
    score: float = Field(..., description="Relevance score (0.0-1.0)")
    chunk_index: int = Field(..., description="Chunk index within the source document")
    source_path: str = Field(default="", description="Absolute file path of the source")


class ChatResponse(BaseModel):
    """Response from the Q&A chat endpoint."""

    answer: str = Field(..., description="The LLM-generated answer")
    sources: List[SourceDocument] = Field(
        default_factory=list,
        description="Retrieved document chunks used to generate the answer",
    )
    session_id: str = Field(..., description="Conversation session ID")
    processing_time_ms: int = Field(..., description="Total processing time in milliseconds")
    context_docs: int = Field(..., description="Number of context documents used")
    model: str = Field(default="", description="LLM model used for generation")
    usage: Dict[str, int] = Field(
        default_factory=dict,
        description="Token usage statistics (prompt_tokens, completion_tokens, total_tokens)",
    )


class SearchResponse(BaseModel):
    """Response from the search-only endpoint."""

    query: str = Field(..., description="The original search query")
    results: List[SourceDocument] = Field(
        default_factory=list,
        description="Search results sorted by relevance",
    )
    total: int = Field(..., description="Total number of results")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(default="ok", description="Service health status")
    version: str = Field(..., description="Application version")
    milvus_connected: bool = Field(..., description="Whether Milvus is connected")
    documents_indexed: int = Field(..., description="Number of documents indexed")
    bm25_ready: bool = Field(..., description="Whether BM25 index is ready")


class StatsResponse(BaseModel):
    """System statistics response."""

    milvus_collection: str = Field(..., description="Milvus collection name")
    milvus_size: int = Field(..., description="Number of vectors in Milvus")
    bm25_doc_count: int = Field(..., description="Number of documents in BM25 index")
    embedding_dim: int = Field(..., description="Embedding vector dimension")
    active_sessions: int = Field(..., description="Number of active conversation sessions")
    total_conversation_turns: int = Field(..., description="Total conversation turns across all sessions")
    llm_provider: str = Field(..., description="Current LLM provider")


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error information")
