"""FastAPI application server with lifespan management.

Creates and manages the lifecycle of all pipeline components.
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config.loader import ConfigLoader, get_config
from src.memory.conversation import ConversationMemory
from src.pipeline.ingest import IngestionPipeline
from src.pipeline.query import QueryPipeline

# ── Logging ──────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan ─────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize on startup, cleanup on shutdown."""
    logger.info("=" * 60)
    logger.info("Starting RAG Q&A System...")
    logger.info("=" * 60)

    # Load configuration
    config_path = app.state.config_path if hasattr(app.state, "config_path") else "config.yaml"
    config = get_config(config_path)
    logger.info("Configuration loaded from %s", config_path)

    # Initialize conversation memory
    memory = ConversationMemory(
        max_turns=config.memory_max_turns,
        max_tokens=config.memory_max_tokens,
    )
    logger.info("Conversation memory initialized (max_turns=%d)", config.memory_max_turns)

    # Initialize ingestion pipeline
    ingest_pipeline = IngestionPipeline(config)
    logger.info("Ingestion pipeline initialized")

    # Check if we need to do initial ingestion
    if ingest_pipeline.vector_store.size == 0:
        logger.info("Vector store is empty. Running initial ingestion ...")
        result = ingest_pipeline.run()
        if result.success:
            logger.info(
                "Initial ingestion complete: %d documents → %d chunks",
                result.documents_loaded, result.chunks_created,
            )
        else:
            logger.warning(
                "Initial ingestion had errors: %s. You can trigger re-ingestion via API.",
                result.errors,
            )
    else:
        # Load existing collection into memory for search
        ingest_pipeline.vector_store.load_collection()
        # Try to load persisted BM25 index
        bm25_path = Path(config.milvus_uri).parent / "bm25_index.pkl"
        if bm25_path.exists():
            try:
                ingest_pipeline.bm25.load(bm25_path)
                logger.info("BM25 index loaded from %s (%d docs)", bm25_path, ingest_pipeline.bm25.doc_count)
            except Exception as e:
                logger.warning("Failed to load BM25 index: %s", e)

    # Initialize query pipeline
    query_pipeline = QueryPipeline(
        config=config,
        bm25=ingest_pipeline.bm25,
        vector_store=ingest_pipeline.vector_store,
        memory=memory,
    )
    logger.info("Query pipeline initialized (LLM: %s)", config.llm_provider)

    # Store in app state
    app.state.config = config
    app.state.memory = memory
    app.state.ingest_pipeline = ingest_pipeline
    app.state.query_pipeline = query_pipeline

    logger.info("=" * 60)
    logger.info("RAG Q&A System is ready!")
    logger.info("API docs: http://%s:%d/docs", config.server_host, config.server_port)
    logger.info("=" * 60)

    yield

    # Shutdown
    logger.info("Shutting down RAG Q&A System ...")
    ingest_pipeline.vector_store.close()
    if hasattr(query_pipeline.llm, 'close'):
        await query_pipeline.llm.close()
    logger.info("Shutdown complete.")


# ── App Factory ──────────────────────────────────────────────────────

def create_app(config_path: str = "config.yaml") -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title="RAG Enterprise Document Q&A System",
        description="""
        ## 企业文档智能问答系统

        基于 RAG (Retrieval-Augmented Generation) 架构的企业文档智能问答 API。

        ### 核心能力
        - **文档全链路处理**：支持 PDF、Word、Markdown 文档的自动加载、清洗与分块
        - **混合检索**：BM25 关键词检索 + 语义向量检索 + 重排序
        - **多轮对话**：带记忆的多轮问答，自动管理上下文
        - **答案溯源**：每个答案附带原始文档片段，有效抑制模型幻觉

        ### 使用流程
        1. 将文档放入 `data/documents/` 目录
        2. 调用 `POST /api/v1/documents/ingest` 触发文档摄入
        3. 使用 `POST /api/v1/chat` 进行问答
        """,
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Store config path for lifespan
    app.state.config_path = config_path

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    from src.api.routes import router
    app.include_router(router)

    return app
