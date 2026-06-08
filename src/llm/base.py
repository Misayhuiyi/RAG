"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document


@dataclass
class Message:
    """A single message in a conversation."""

    role: str  # "user", "assistant", "system"
    content: str


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    answer: str
    model: str = ""
    usage: Dict[str, int] = field(default_factory=dict)  # prompt_tokens, completion_tokens
    finish_reason: str = ""


class BaseLLM(ABC):
    """Abstract interface for LLM providers.

    Subclasses implement the generate() method for specific providers
    (OpenAI, Ollama, etc.).

    Usage:
        llm = OpenAILLM(model="gpt-4o", api_key="...")
        response = await llm.generate(prompt="...", context=docs, history=messages)
    """

    def __init__(self, model: str, temperature: float = 0.1, max_tokens: int = 2048):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        context: List[Any],
        history: List[Message],
        system_prompt: str = "",
        user_template: str = "",
    ) -> LLMResponse:
        """Generate an answer from the LLM.

        Args:
            prompt: The user's question.
            context: Retrieved documents/chunks to ground the answer.
            history: Previous conversation messages.
            system_prompt: System-level instruction for the LLM.
            user_template: Template string for formatting context + history + prompt.

        Returns:
            LLMResponse with answer text and metadata.
        """
        ...

    def _build_context_text(self, context: List[Any]) -> str:
        """Convert retrieved chunks into a formatted context string."""
        parts: List[str] = []
        for i, doc in enumerate(context):
            # Handle both SearchResult and Document types
            if hasattr(doc, 'text'):
                content = doc.text
                source = getattr(doc, 'document_name', f'文档{i+1}')
            elif hasattr(doc, 'page_content'):
                content = doc.page_content
                source = doc.metadata.get('file_name', f'文档{i+1}')
            else:
                content = str(doc)
                source = f'文档{i+1}'

            parts.append(f"[来源{i+1}: {source}]\n{content}")

        return "\n\n---\n\n".join(parts)

    def _build_history_text(self, history: List[Message]) -> str:
        """Convert conversation history into a formatted string."""
        if not history:
            return "（无历史对话）"

        lines: List[str] = []
        for msg in history:
            role_label = "用户" if msg.role == "user" else "助手"
            lines.append(f"{role_label}: {msg.content}")
        return "\n".join(lines)

    def _format_prompt(
        self,
        question: str,
        context_text: str,
        history_text: str,
        user_template: str = "",
    ) -> str:
        """Format the user prompt using the configured template."""
        if user_template:
            try:
                return user_template.format(
                    context=context_text,
                    history=history_text,
                    question=question,
                )
            except KeyError:
                pass  # Fall back to default below

        # Default template
        return (
            f"参考文档：\n{context_text}\n\n"
            f"对话历史：\n{history_text}\n\n"
            f"用户问题：{question}\n\n"
            f"请基于以上参考文档回答问题，并引用具体的文档来源（如 [来源1]）。"
            f"如果文档中没有相关信息，请明确说明。"
        )
