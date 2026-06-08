"""OpenAI API LLM provider.

Supports both OpenAI and OpenAI-compatible APIs.
"""

import logging
from typing import Any, List, Optional

from openai import AsyncOpenAI

from .base import BaseLLM, LLMResponse, Message

logger = logging.getLogger(__name__)


class OpenAILLM(BaseLLM):
    """LLM provider using the OpenAI API.

    Usage:
        llm = OpenAILLM(
            model="gpt-4o",
            api_key="sk-...",
            base_url="https://api.openai.com/v1",   # optional
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
        model: str = "gpt-4o",
        api_key: str = "",
        base_url: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ):
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)
        self.api_key = api_key
        self.base_url = base_url
        self._client: Optional[AsyncOpenAI] = None

    @property
    def client(self) -> AsyncOpenAI:
        """Lazy-initialize the async OpenAI client."""
        if self._client is None:
            kwargs: dict = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = AsyncOpenAI(**kwargs)
        return self._client

    async def generate(
        self,
        prompt: str,
        context: List[Any],
        history: List[Message],
        system_prompt: str = "",
        user_template: str = "",
    ) -> LLMResponse:
        """Generate an answer using the OpenAI API."""
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

        # Add recent history as separate messages
        for msg in history[-6:]:  # Limit to recent 6 messages for API format
            messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": user_message})

        logger.debug("Sending %d messages to OpenAI (model=%s)", len(messages), self.model)

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            choice = response.choices[0]
            return LLMResponse(
                answer=choice.message.content or "",
                model=self.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
                finish_reason=choice.finish_reason or "",
            )

        except Exception as e:
            logger.error("OpenAI API error: %s", e)
            raise
