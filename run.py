#!/usr/bin/env python
"""Entry point for the RAG Enterprise Document Q&A System.

Usage:
    python run.py                          # Start with default config
    python run.py --config config.yaml     # Specify config file
    python run.py --host 0.0.0.0 --port 8080
    python run.py --ingest-only            # Only run ingestion, don't start server
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run")


def parse_args():
    parser = argparse.ArgumentParser(
        description="RAG Enterprise Document Q&A System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                              Start server with default config
  python run.py --config my_config.yaml      Use custom config
  python run.py --host 127.0.0.1 --port 8080 Bind to specific host/port
  python run.py --ingest-only                Run ingestion only
        """,
    )
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Override server host (default: from config)",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=None,
        help="Override server port (default: from config)",
    )
    parser.add_argument(
        "--ingest-only",
        action="store_true",
        help="Run document ingestion only, then exit",
    )
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="Drop existing vector data before ingestion",
    )
    return parser.parse_args()


def run_ingestion(config_path: str, drop_existing: bool = False) -> None:
    """Run document ingestion and exit."""
    from src.config.loader import get_config
    from src.pipeline.ingest import IngestionPipeline

    config = get_config(config_path)
    pipeline = IngestionPipeline(config)

    logger.info("Starting document ingestion ...")
    result = pipeline.run(drop_existing=drop_existing)

    logger.info("=" * 40)
    logger.info("Ingestion Result:")
    logger.info("  Documents loaded: %d", result.documents_loaded)
    logger.info("  Chunks created:   %d", result.chunks_created)
    logger.info("  Vectors stored:   %d", result.vectors_stored)
    logger.info("  Time elapsed:     %.2fs", result.elapsed_seconds)
    if result.errors:
        logger.info("  Errors:           %s", result.errors)
    logger.info("=" * 40)


def run_server(config_path: str, host: str = None, port: int = None) -> None:
    """Start the FastAPI server."""
    import uvicorn

    from src.config.loader import get_config

    config = get_config(config_path)
    host = host or config.server_host
    port = port or config.server_port

    logger.info("Starting server at http://%s:%d", host, port)
    logger.info("API docs at http://%s:%d/docs", host, port)

    uvicorn.run(
        "src.api.server:create_app",
        factory=True,
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )


def main():
    args = parse_args()

    # Validate config exists
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path.absolute())
        logger.info("Tip: Copy and customize config.yaml from the project template.")
        sys.exit(1)

    if args.ingest_only:
        run_ingestion(str(config_path), drop_existing=args.drop_existing)
    else:
        run_server(str(config_path), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
