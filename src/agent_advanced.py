from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent_baseline import live_turn_usage
from config import LabConfig, load_config
from memory_store import (
    FACT_ORDER,
    INTEREST_KEY,
    CompactMemoryManager,
    UserProfileStore,
    estimate_tokens,
)
from model_provider import build_chat_model
from offline_responder import acknowledge, answer_from_facts

try:  # Module-level so LangChain can resolve the (string) annotations of the live tools.
    from langchain.agents.middleware import ModelRequest
    from langchain.tools import ToolRuntime
except ImportError:  # offline-only environment
    ModelRequest = ToolRuntime = None

ADVANCED_SYSTEM_PROMPT = (
    "Bạn là trợ lý tiếng Việt có bộ nhớ dài hạn. "
    "Hồ sơ User.md bên dưới là nguồn sự thật về người dùng; nếu có đính chính, luôn ưu tiên fact mới nhất. "
    "Chỉ lưu fact ổn định (tên, nơi ở, nghề, sở thích, style trả lời), bỏ qua câu đùa hoặc giả định. "
    "Trả lời theo style người dùng thích."
)
WRITABLE_KEYS = FACT_ORDER + (INTEREST_KEY,)


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B: short-term thread memory + persistent `User.md` + compact memory.

    1. within-session memory: recent messages of the thread (CompactMemoryManager)
    2. persistent memory: `state/profiles/<user>/User.md`, survives new threads
    3. compact memory: older thread messages folded into a bounded summary
    """

    name = "Advanced"

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.langchain_agent = self._maybe_build_langchain_agent()

    @property
    def mode(self) -> str:
        return "live" if self.langchain_agent is not None else "offline"

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent is not None:
            return self._reply_live(user_id, thread_id, message)
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _remember(self, user_id: str, message: str) -> dict[str, str]:
        """Persist confident, stable facts from the user message into User.md."""

        return self.profile_store.apply_message(
            user_id,
            message,
            threshold=self.config.profile_confidence_threshold,
            max_interests=self.config.profile_max_interests,
        )

    def _record(self, thread_id: str, response: str, agent_tokens: int, prompt_tokens: int) -> None:
        self.compact_memory.append(thread_id, "assistant", response)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + agent_tokens
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        saved = self._remember(user_id, message)
        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)

        response = self._offline_response(user_id, thread_id, message, saved)
        agent_tokens = estimate_tokens(response)
        self._record(thread_id, response, agent_tokens, prompt_tokens)
        return {
            "response": response,
            "thread_id": thread_id,
            "mode": "offline",
            "agent_tokens": agent_tokens,
            "prompt_tokens": prompt_tokens,
            "memory_updates": saved,
            "compactions": self.compaction_count(thread_id),
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """System prompt + User.md digest + compact summary + recent kept messages."""

        ctx = self.compact_memory.context(thread_id)
        return (
            estimate_tokens(ADVANCED_SYSTEM_PROMPT)
            + estimate_tokens(self.profile_store.render_for_prompt(user_id))
            + estimate_tokens(ctx["summary"])
            + sum(estimate_tokens(m["content"]) for m in ctx["messages"])
        )

    def _offline_response(self, user_id: str, thread_id: str, message: str, saved: dict[str, str] | None = None) -> str:
        """Answer recall questions from User.md; otherwise acknowledge (and report what was saved)."""

        facts = self.profile_store.facts(user_id)
        return answer_from_facts(message, facts, scope="trong User.md") or acknowledge(message, saved)

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        # Deterministic guardrail: confident facts are saved even if the model forgets to call the tool.
        saved = self._remember(user_id, message)
        # Shadow compact memory keeps prompt/compaction accounting comparable with offline mode.
        self.compact_memory.append(thread_id, "user", message)
        result = self.langchain_agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": f"advanced:{thread_id}"}},
            context=AgentContext(user_id, str(self.profile_store.path_for(user_id))),
        )
        messages = result["messages"]
        last_human = max(i for i, m in enumerate(messages) if getattr(m, "type", "") == "human")
        fallback = self._estimate_prompt_context_tokens(user_id, thread_id)
        response, agent_tokens, prompt_tokens = live_turn_usage(messages[last_human + 1 :], fallback)
        self._record(thread_id, response, agent_tokens, prompt_tokens)
        return {
            "response": response,
            "thread_id": thread_id,
            "mode": "live",
            "agent_tokens": agent_tokens,
            "prompt_tokens": prompt_tokens,
            "memory_updates": saved,
            "compactions": self.compaction_count(thread_id),
        }

    def _maybe_build_langchain_agent(self):
        """Live agent: provider model + InMemorySaver + User.md tools + dynamic prompt + summarization.

        Returns None (offline mode) unless LAB_LIVE=1 and the provider is configured.
        """

        if self.force_offline or not self.config.live_mode or not self.config.model.is_live_ready:
            return None
        try:
            from langchain.agents import create_agent
            from langchain.agents.middleware import SummarizationMiddleware, dynamic_prompt
            from langchain.tools import tool
            from langgraph.checkpoint.memory import InMemorySaver
        except ImportError:
            return None

        store = self.profile_store

        @tool
        def read_user_memory(runtime: ToolRuntime[AgentContext]) -> str:
            """Đọc toàn bộ User.md (hồ sơ bền vững) của người dùng hiện tại."""

            return store.read_text(runtime.context.user_id)

        @tool
        def save_user_fact(key: str, value: str, runtime: ToolRuntime[AgentContext]) -> str:
            """Lưu hoặc cập nhật MỘT fact ổn định vào User.md.

            key: name | location | profession | favorite_drink | favorite_food | pet | response_style | interests.
            Không lưu câu đùa, giả định hay thông tin tạm thời.
            """

            if key not in WRITABLE_KEYS:
                return f"Key không hợp lệ: {key}. Chỉ dùng: {', '.join(WRITABLE_KEYS)}"
            store.upsert_fact(runtime.context.user_id, key, value, confidence=0.8)
            return f"Đã lưu {key} = {value}"

        @dynamic_prompt
        def inject_profile(request: ModelRequest) -> str:
            profile = store.render_for_prompt(request.runtime.context.user_id) or "(chưa có)"
            return f"{ADVANCED_SYSTEM_PROMPT}\n\n## User.md\n{profile}"

        model = build_chat_model(self.config.model)
        return create_agent(
            model,
            tools=[read_user_memory, save_user_fact],
            middleware=[
                inject_profile,
                SummarizationMiddleware(
                    model,
                    trigger=("tokens", self.config.compact_threshold_tokens),
                    keep=("messages", self.config.compact_keep_messages),
                ),
            ],
            context_schema=AgentContext,
            checkpointer=InMemorySaver(),
        )

