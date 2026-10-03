from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import Profile, estimate_tokens, extract_profile_candidates
from model_provider import build_chat_model
from offline_responder import acknowledge, answer_from_facts

BASELINE_SYSTEM_PROMPT = (
    "Bạn là trợ lý tiếng Việt. Trả lời ngắn gọn. "
    "Bạn chỉ biết những gì có trong cuộc hội thoại hiện tại."
)


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


def message_text(content: Any) -> str:
    """LangChain content can be a string or a list of content blocks."""

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content or "")


def live_turn_usage(new_messages: list[Any], fallback_prompt: int) -> tuple[str, int, int]:
    """(final AI text, output tokens, input tokens) from the messages produced by one invoke."""

    answer, output_tokens, input_tokens = "", 0, 0
    for msg in new_messages:
        if getattr(msg, "type", "") != "ai":
            continue
        usage = getattr(msg, "usage_metadata", None) or {}
        text = message_text(msg.content)
        output_tokens += int(usage.get("output_tokens") or estimate_tokens(text))
        input_tokens += int(usage.get("input_tokens") or 0)
        if text.strip():
            answer = text
    if not input_tokens:
        input_tokens = fallback_prompt
    return answer, output_tokens or estimate_tokens(answer), input_tokens


class BaselineAgent:
    """Agent A: within-thread memory only.

    - keeps the raw message list per `thread_id`
    - no `User.md`, nothing survives into a new thread
    - every turn re-sends the whole thread, so prompt cost grows with thread length
    """

    name = "Baseline"

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = self._maybe_build_langchain_agent()

    @property
    def mode(self) -> str:
        return "live" if self.langchain_agent is not None else "offline"

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent is not None:
            return self._reply_live(user_id, thread_id, message)
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return session.token_usage if session else 0

    def prompt_token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return session.prompt_tokens_processed if session else 0

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def memory_file_size(self, user_id: str) -> int:
        # Baseline has no persistent memory file.
        return 0

    def _session(self, thread_id: str) -> SessionState:
        return self.sessions.setdefault(thread_id, SessionState())

    def _prompt_tokens(self, session: SessionState) -> int:
        """System prompt + the full thread history (including the new user message)."""

        return estimate_tokens(BASELINE_SYSTEM_PROMPT) + sum(estimate_tokens(m["content"]) for m in session.messages)

    def _thread_facts(self, session: SessionState) -> dict[str, str]:
        """What a model could read off the current thread's own messages. Nothing persisted."""

        profile = Profile()
        for msg in session.messages:
            if msg["role"] == "user":
                profile.apply(extract_profile_candidates(msg["content"]), threshold=self.config.profile_confidence_threshold)
        return profile.as_dict()

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self._session(thread_id)
        session.messages.append({"role": "user", "content": message})
        prompt_tokens = self._prompt_tokens(session)

        response = answer_from_facts(message, self._thread_facts(session), scope="trong phiên này") or acknowledge(message)

        session.messages.append({"role": "assistant", "content": response})
        agent_tokens = estimate_tokens(response)
        session.token_usage += agent_tokens
        session.prompt_tokens_processed += prompt_tokens
        return {
            "response": response,
            "thread_id": thread_id,
            "mode": "offline",
            "agent_tokens": agent_tokens,
            "prompt_tokens": prompt_tokens,
            "memory_updates": {},
        }

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        session = self._session(thread_id)
        session.messages.append({"role": "user", "content": message})
        result = self.langchain_agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": f"baseline:{thread_id}"}},
        )
        messages = result["messages"]
        last_human = max(i for i, m in enumerate(messages) if getattr(m, "type", "") == "human")
        response, agent_tokens, prompt_tokens = live_turn_usage(messages[last_human + 1 :], self._prompt_tokens(session))

        session.messages.append({"role": "assistant", "content": response})
        session.token_usage += agent_tokens
        session.prompt_tokens_processed += prompt_tokens
        return {
            "response": response,
            "thread_id": thread_id,
            "mode": "live",
            "agent_tokens": agent_tokens,
            "prompt_tokens": prompt_tokens,
            "memory_updates": {},
        }

    def _maybe_build_langchain_agent(self):
        """Live agent: `create_agent` + `InMemorySaver` (thread-scoped memory only).

        Returns None (offline mode) unless LAB_LIVE=1 and the provider is configured.
        """

        if self.force_offline or not self.config.live_mode or not self.config.model.is_live_ready:
            return None
        try:
            from langchain.agents import create_agent
            from langgraph.checkpoint.memory import InMemorySaver
        except ImportError:
            return None
        return create_agent(
            build_chat_model(self.config.model),
            tools=[],
            system_prompt=BASELINE_SYSTEM_PROMPT,
            checkpointer=InMemorySaver(),
        )
