"""Multi-turn conversation memory with token-aware trimming.

Manages per-session conversation history with automatic eviction
of older turns when the token budget is exceeded.
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, OrderedDict

from ..llm.base import Message

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""

    question: str
    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ConversationSession:
    """A conversation session with message history."""

    session_id: str
    turns: List[ConversationTurn] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)


class ConversationMemory:
    """In-memory conversation store with per-session history.

    Features:
    - LRU-like eviction of old sessions
    - Token-based trimming within sessions
    - Session TTL for automatic cleanup

    Usage:
        memory = ConversationMemory(max_turns=10, max_tokens=4000)
        memory.add_turn(session_id, "问题", "答案", [sources])
        history = memory.get_context(session_id)
    """

    def __init__(
        self,
        max_turns: int = 10,
        max_tokens: int = 4000,
        session_ttl_seconds: float = 3600.0,
        max_sessions: int = 1000,
    ):
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.session_ttl = session_ttl_seconds
        self.max_sessions = max_sessions

        self._sessions: OrderedDict[str, ConversationSession] = OrderedDict()

    # ── Public API ───────────────────────────────────────────────────

    def get_or_create_session(self, session_id: Optional[str] = None) -> ConversationSession:
        """Get an existing session or create a new one.

        Args:
            session_id: Existing session ID. If None, a new session is created.

        Returns:
            The ConversationSession object.
        """
        if session_id and session_id in self._sessions:
            session = self._sessions[session_id]
            session.last_active = time.time()
            # Move to end (most recently used)
            self._sessions.move_to_end(session_id)
            return session

        # Create new session
        new_id = session_id or str(uuid.uuid4())
        session = ConversationSession(session_id=new_id)
        self._add_session(new_id, session)
        return session

    def add_turn(
        self,
        session_id: str,
        question: str,
        answer: str,
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add a Q&A turn to a session's history."""
        session = self.get_or_create_session(session_id)
        turn = ConversationTurn(
            question=question,
            answer=answer,
            sources=sources or [],
        )
        session.turns.append(turn)

        # Trim if exceeding max_turns
        while len(session.turns) > self.max_turns:
            session.turns.pop(0)

        logger.debug("Added turn to session %s (total turns: %d)", session_id, len(session.turns))

    def get_context(self, session_id: str) -> List[Message]:
        """Get the conversation history as a list of Messages.

        Returns recent turns up to the token budget, trimmed from oldest.
        """
        session = self.get_or_create_session(session_id)

        messages: List[Message] = []
        # Build messages from turns, newest first, then reverse
        turns_to_include: List[ConversationTurn] = []
        token_count = 0

        for turn in reversed(session.turns):
            # Rough token estimation: ~1.5 chars per token for Chinese
            turn_tokens = (len(turn.question) + len(turn.answer)) / 1.5
            if token_count + turn_tokens > self.max_tokens:
                break
            turns_to_include.append(turn)
            token_count += turn_tokens

        # Reverse back to chronological order
        turns_to_include.reverse()

        for turn in turns_to_include:
            messages.append(Message(role="user", content=turn.question))
            messages.append(Message(role="assistant", content=turn.answer))

        return messages

    def get_history_text(self, session_id: str) -> str:
        """Get conversation history as a formatted string for prompts."""
        messages = self.get_context(session_id)
        if not messages:
            return "（无历史对话）"

        lines: List[str] = []
        for msg in messages:
            role_label = "用户" if msg.role == "user" else "助手"
            lines.append(f"{role_label}: {msg.content}")
        return "\n".join(lines)

    def get_turns(self, session_id: str) -> List[ConversationTurn]:
        """Get all turns for a session."""
        session = self.get_or_create_session(session_id)
        return list(session.turns)

    def clear_session(self, session_id: str) -> None:
        """Remove a session from memory."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.debug("Cleared session %s", session_id)

    def clear_all(self) -> None:
        """Remove all sessions."""
        self._sessions.clear()
        logger.info("Cleared all conversation sessions")

    # ── Session management ───────────────────────────────────────────

    def _add_session(self, session_id: str, session: ConversationSession) -> None:
        """Add a session, evicting old ones if limit reached."""
        # Evict expired sessions
        self._evict_expired()

        # Evict oldest if at capacity
        while len(self._sessions) >= self.max_sessions:
            oldest_key, _ = self._sessions.popitem(last=False)
            logger.debug("Evicted oldest session: %s", oldest_key)

        self._sessions[session_id] = session

    def _evict_expired(self) -> None:
        """Remove sessions that have exceeded the TTL."""
        now = time.time()
        expired = [
            sid for sid, s in self._sessions.items()
            if now - s.last_active > self.session_ttl
        ]
        for sid in expired:
            del self._sessions[sid]
            logger.debug("Evicted expired session: %s", sid)

    @property
    def session_count(self) -> int:
        self._evict_expired()
        return len(self._sessions)

    def get_stats(self) -> Dict[str, Any]:
        """Return memory statistics."""
        self._evict_expired()
        total_turns = sum(len(s.turns) for s in self._sessions.values())
        return {
            "active_sessions": len(self._sessions),
            "total_turns": total_turns,
            "max_sessions": self.max_sessions,
            "max_turns_per_session": self.max_turns,
        }
