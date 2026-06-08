"""Configuration loader with YAML parsing and env var interpolation.

Reads config.yaml and provides typed access to each configuration section.
Supports ${ENV_VAR} interpolation for secrets like API keys.
"""

import os
import re
from pathlib import Path
from typing import Any, Optional

import yaml


class ConfigLoader:
    """Load and manage application configuration from YAML."""

    _ENV_VAR_RE = re.compile(r"\$\{([^}]+)\}")

    def __init__(self, config_path: str | Path = "config.yaml"):
        self._config_path = Path(config_path)
        self._data: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        """Load YAML config and resolve environment variables."""
        if not self._config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self._config_path}")

        with open(self._config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        self._data = self._resolve_env_vars(raw)

    def _resolve_env_vars(self, obj: Any) -> Any:
        """Recursively resolve ${ENV_VAR} placeholders in the config."""
        if isinstance(obj, str):
            def _replace(m: re.Match) -> str:
                var_name = m.group(1)
                return os.environ.get(var_name, m.group(0))
            return self._ENV_VAR_RE.sub(_replace, obj)
        elif isinstance(obj, dict):
            return {k: self._resolve_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._resolve_env_vars(item) for item in obj]
        return obj

    # ── Top-level accessors ──────────────────────────────────────────

    @property
    def raw(self) -> dict[str, Any]:
        """Return the full resolved config dict."""
        return self._data

    def get(self, key: str, default: Any = None) -> Any:
        """Get a top-level config value by key."""
        return self._data.get(key, default)

    # ── Document config ──────────────────────────────────────────────

    def get_document_config(self) -> dict[str, Any]:
        return self._data.get("documents", {})

    @property
    def source_dir(self) -> str:
        return self._data.get("documents", {}).get("source_dir", "./data/documents")

    @property
    def supported_formats(self) -> list[str]:
        return self._data.get("documents", {}).get("supported_formats", [".pdf", ".docx", ".md"])

    @property
    def chunk_size(self) -> int:
        return self._data.get("documents", {}).get("chunk_size", 512)

    @property
    def chunk_overlap(self) -> int:
        return self._data.get("documents", {}).get("chunk_overlap", 64)

    @property
    def max_chunk_size(self) -> int:
        return self._data.get("documents", {}).get("max_chunk_size", 1024)

    # ── Embedding config ─────────────────────────────────────────────

    def get_embedding_config(self) -> dict[str, Any]:
        return self._data.get("embedding", {})

    @property
    def embedding_model_name(self) -> str:
        return self._data.get("embedding", {}).get("model_name", "BAAI/bge-large-zh-v1.5")

    @property
    def embedding_device(self) -> str:
        return self._data.get("embedding", {}).get("device", "cpu")

    @property
    def embedding_batch_size(self) -> int:
        return self._data.get("embedding", {}).get("batch_size", 32)

    @property
    def embedding_normalize(self) -> bool:
        return self._data.get("embedding", {}).get("normalize", True)

    # ── Milvus config ────────────────────────────────────────────────

    def get_milvus_config(self) -> dict[str, Any]:
        return self._data.get("milvus", {})

    @property
    def milvus_uri(self) -> str:
        return self._data.get("milvus", {}).get("uri", "./data/vector_store/milvus.db")

    @property
    def milvus_collection_name(self) -> str:
        return self._data.get("milvus", {}).get("collection_name", "enterprise_docs")

    @property
    def milvus_index_type(self) -> str:
        return self._data.get("milvus", {}).get("index_type", "IVF_FLAT")

    @property
    def milvus_metric_type(self) -> str:
        return self._data.get("milvus", {}).get("metric_type", "COSINE")

    @property
    def milvus_nlist(self) -> int:
        return self._data.get("milvus", {}).get("nlist", 128)

    # ── Retrieval config ─────────────────────────────────────────────

    def get_retrieval_config(self) -> dict[str, Any]:
        return self._data.get("retrieval", {})

    @property
    def retrieval_bm25_weight(self) -> float:
        return self._data.get("retrieval", {}).get("bm25_weight", 0.3)

    @property
    def retrieval_vector_weight(self) -> float:
        return self._data.get("retrieval", {}).get("vector_weight", 0.7)

    @property
    def retrieval_top_k_retrieval(self) -> int:
        return self._data.get("retrieval", {}).get("top_k_retrieval", 20)

    @property
    def retrieval_top_k_fusion(self) -> int:
        return self._data.get("retrieval", {}).get("top_k_fusion", 10)

    @property
    def retrieval_top_k_final(self) -> int:
        return self._data.get("retrieval", {}).get("top_k_final", 5)

    @property
    def retrieval_rerank_enabled(self) -> bool:
        return self._data.get("retrieval", {}).get("rerank_enabled", True)

    @property
    def retrieval_rerank_model(self) -> str:
        return self._data.get("retrieval", {}).get("rerank_model", "BAAI/bge-reranker-large")

    # ── LLM config ───────────────────────────────────────────────────

    def get_llm_config(self) -> dict[str, Any]:
        return self._data.get("llm", {})

    @property
    def llm_provider(self) -> str:
        return self._data.get("llm", {}).get("provider", "openai")

    def get_llm_provider_config(self, provider: Optional[str] = None) -> dict[str, Any]:
        provider = provider or self.llm_provider
        return self._data.get("llm", {}).get(provider, {})

    # ── Memory config ────────────────────────────────────────────────

    def get_memory_config(self) -> dict[str, Any]:
        return self._data.get("memory", {})

    @property
    def memory_max_turns(self) -> int:
        return self._data.get("memory", {}).get("max_turns", 10)

    @property
    def memory_max_tokens(self) -> int:
        return self._data.get("memory", {}).get("max_tokens", 4000)

    # ── Prompt config ────────────────────────────────────────────────

    def get_prompt_config(self) -> dict[str, Any]:
        return self._data.get("prompt", {})

    @property
    def prompt_system(self) -> str:
        return self._data.get("prompt", {}).get(
            "system",
            "你是一个企业文档智能问答助手。请严格基于提供的文档内容回答问题。如果文档中没有相关信息，请明确说明。"
        )

    @property
    def prompt_user_template(self) -> str:
        return self._data.get("prompt", {}).get(
            "user_template",
            "参考文档：\n{context}\n\n对话历史：\n{history}\n\n用户问题：{question}\n\n"
            "请基于以上参考文档回答问题，并引用具体的文档来源。"
        )

    # ── Server config ────────────────────────────────────────────────

    def get_server_config(self) -> dict[str, Any]:
        return self._data.get("server", {})

    @property
    def server_host(self) -> str:
        return self._data.get("server", {}).get("host", "0.0.0.0")

    @property
    def server_port(self) -> int:
        return self._data.get("server", {}).get("port", 8000)

    @property
    def server_max_concurrency(self) -> int:
        return self._data.get("server", {}).get("max_concurrency", 20)

    # ── Evaluation config ────────────────────────────────────────────

    def get_evaluation_config(self) -> dict[str, Any]:
        return self._data.get("evaluation", {})

    @property
    def evaluation_badcase_dir(self) -> str:
        return self._data.get("evaluation", {}).get("badcase_dir", "./data/badcases")


# Global singleton
_config_instance: Optional[ConfigLoader] = None


def get_config(config_path: str | Path = "config.yaml") -> ConfigLoader:
    """Get or create the global ConfigLoader singleton."""
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigLoader(config_path)
    return _config_instance


def reload_config(config_path: str | Path = "config.yaml") -> ConfigLoader:
    """Force-reload the configuration."""
    global _config_instance
    _config_instance = ConfigLoader(config_path)
    return _config_instance
