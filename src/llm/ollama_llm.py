"""Ollama local LLM provider.

Connects to a local Ollama server for offline model inference.
"""

import json
import logging
from typing import Any, List, Optional

import httpx

from .base import BaseLLM, LLMResponse, Message

logger = logging.getLogger(__name__)


class OllamaLLM(BaseLLM):
    """LLM provider using a local Ollama server.

    Usage:
        llm = OllamaLLM(
            model="qwen2.5:7b",
            base_url="http://localhost:11434",
        )
        response = await llm.generate(
            prompt="问题",
            context=retrieved_docs,
            history=[],
            system_prompt="你是一个问答助手。",
            user_template="...",
        )
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ):
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)
        self.base_url = base_url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def client(self) -> httpx.AsyncClient:
        """Lazy-initialize the async HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(120.0))
        return self._client

    async def generate(
        self,
        prompt: str,
        context: List[Any],
        history: List[Message],
        system_prompt: str = "",
        user_template: str = "",
    ) -> LLMResponse:
        """Generate an answer using Ollama's chat API."""
        # Build context and history strings
        context_text = self._build_context_text(context)
        history_text = self._build_history_text(history)

        # Format the user message
        user_message = self._format_prompt(
            question=prompt,
            context_text=context_text,
            history_text=history_text,
            user_template=user_template,
        )

        # Build message list
        messages: list = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Add recent history
        for msg in history[-6:]:
            messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": user_message})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        logger.debug("Sending request to Ollama (model=%s)", self.model)

        try:
            response = await self.client.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            answer = data.get("message", {}).get("content", "")
            return LLMResponse(
                answer=answer,
                model=self.model,
                usage={
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                    "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
                },
                finish_reason=data.get("done_reason", ""),
            )

        except httpx.HTTPError as e:
            logger.error("Ollama API error: %s", e)
            raise
        except Exception as e:
            logger.error("Ollama unexpected error: %s", e)
            raise

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def health_check(self) -> bool:
        """Check if the Ollama server is reachable."""
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except Exception:
            return False
