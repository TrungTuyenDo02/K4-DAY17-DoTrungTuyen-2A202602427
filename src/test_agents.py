from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from benchmark import load_conversations, recall_points, run_agent_benchmark
from config import LabConfig, load_config
from memory_store import (
    CompactMemoryManager,
    Profile,
    UserProfileStore,
    extract_profile_candidates,
    extract_profile_updates,
)
from model_provider import ProviderConfig, normalize_provider

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LONG_TURN = (
    "Mình kể thêm một đoạn rất dài về tin tức để làm phình ngữ cảnh: Artemis III, X-59, WMO và "
    "kế hoạch điện sạch của British Columbia đều cho thấy trade-off giữa scale và efficiency. "
) * 3


def make_config(tmp_path: Path, threshold: int = 300, keep: int = 4) -> LabConfig:
    """Isolated config: state in tmp_path, small compact threshold, always offline."""

    base = load_config()
    offline = ProviderConfig("openai", "offline-test", 0.0)
    return LabConfig(
        base_dir=base.base_dir,
        data_dir=base.data_dir,
        state_dir=tmp_path / "state",
        compact_threshold_tokens=threshold,
        compact_keep_messages=keep,
        model=offline,
        judge_model=offline,
    )


# --- User.md ---------------------------------------------------------------


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    user = "DũngCT/../x"

    path = store.path_for(user)
    assert path.name == "User.md" and (tmp_path / "profiles") in path.parents  # sanitised, no traversal
    assert "## Facts" in store.read_text(user)  # default template before any write
    assert store.file_size(user) == 0

    store.write_text(user, "# User.md\n\n## Facts\n- location: Đà Nẵng\n")
    assert store.facts(user)["location"] == "Đà Nẵng"
    size = store.file_size(user)
    assert size > 0

    assert store.edit_text(user, "Đà Nẵng", "Huế") is True
    assert store.facts(user)["location"] == "Huế"
    assert store.edit_text(user, "không tồn tại", "x") is False

    assert store.upsert_fact(user, "profession", "MLOps engineer") is True
    assert store.facts(user) == {"location": "Huế", "profession": "MLOps engineer"}


def test_advanced_writes_user_md(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    result = agent.reply("dungct", "t1", "Chào bạn, mình tên là DũngCT. Mình ở Đà Nẵng.")

    assert result["memory_updates"] == {"name": "DũngCT", "location": "Đà Nẵng"}
    text = agent.profile_store.read_text("dungct")
    assert "- name: DũngCT" in text and "- location: Đà Nẵng" in text
    assert agent.memory_file_size("dungct") > 0


# --- compact memory ----------------------------------------------------------


def test_compact_trigger(tmp_path: Path) -> None:
    manager = CompactMemoryManager(threshold_tokens=200, keep_messages=2)
    for i in range(3):
        manager.append("t", "user", f"tin ngắn {i}")
    assert manager.compaction_count("t") == 0  # short thread: no compaction

    for _ in range(4):
        manager.append("t", "user", LONG_TURN)
        manager.append("t", "assistant", "Đã ghi nhận.")

    ctx = manager.context("t")
    assert manager.compaction_count("t") >= 1
    assert len(ctx["messages"]) <= 2 + 1  # kept window (+ the message that triggered it)
    assert ctx["summary"].startswith("- ")
    assert len(ctx["summary"].splitlines()) <= manager.summary_max_items  # summary stays bounded

    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    for turn in load_conversations(DATA_DIR / "advanced_long_context.json")[0]["turns"]:
        agent.reply("stress", "long", turn)
    assert agent.compaction_count("long") >= 2


def test_compact_keeps_recent_follow_up(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path, threshold=200, keep=4), force_offline=True)
    for _ in range(5):
        agent.reply("u", "t", LONG_TURN)
    agent.reply("u", "t", "Câu cuối cùng cần giữ nguyên.")
    recent = [m["content"] for m in agent.compact_memory.context("t")["messages"]]
    assert "Câu cuối cùng cần giữ nguyên." in recent


# --- cross-session recall ---------------------------------------------------------


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    turns = [
        "Chào bạn, mình tên là DũngCT.",
        "Đồ uống yêu thích là cà phê sữa đá.",
        "Mình nuôi một bé corgi tên Bơ.",
    ]
    for turn in turns:
        baseline.reply("dungct", "session-1", turn)
        advanced.reply("dungct", "session-1", turn)

    question = "Mình tên gì và đồ uống yêu thích là gì?"
    expected = ["DũngCT", "cà phê sữa đá"]

    # Same thread: both remember.
    assert recall_points(baseline.reply("dungct", "session-1", question)["response"], expected) == 1.0

    # New thread: only the advanced agent remembers (via User.md).
    assert recall_points(baseline.reply("dungct", "session-2", question)["response"], expected) == 0.0
    assert recall_points(advanced.reply("dungct", "session-2", question)["response"], expected) == 1.0

    # A brand-new agent instance (process restart) still reads the persisted User.md.
    restarted = AdvancedAgent(config, force_offline=True)
    answer = restarted.reply("dungct", "session-3", "Mình nuôi con gì?")["response"]
    assert "corgi" in answer


def test_correction_replaces_old_fact(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    agent.reply("u", "s1", "Mình ở Đà Nẵng và đang làm backend engineer cho startup AI.")
    agent.reply("u", "s2", "À, mình đính chính: giờ mình đang ở Huế chứ không còn ở Đà Nẵng nữa.")
    agent.reply("u", "s3", "Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer.")

    answer = agent.reply("u", "s4", "Hiện tại mình làm nghề gì và đang ở đâu?")["response"]
    assert "MLOps engineer" in answer and "Huế" in answer
    assert "backend" not in answer and "Đà Nẵng" not in answer
    facts = agent.profile_store.facts("u")
    assert facts["location"] == "Huế" and facts["profession"] == "MLOps engineer"


def test_noise_and_questions_are_not_stored() -> None:
    joke = (
        "Có lúc mình đùa với đồng nghiệp rằng hay là chuyển sang product manager, nhưng đó chỉ là câu đùa. "
        "Hà Nội chỉ là nơi mình vừa bay ra họp hai ngày chứ không phải nơi ở hiện tại."
    )
    assert extract_profile_updates(joke) == {}
    low = [c for c in extract_profile_candidates(joke) if c.key == "profession"]
    assert low and all(c.confidence < 0.6 for c in low)  # seen, but rejected by the confidence threshold

    assert extract_profile_updates("Bạn có thể nhắc lại tên mình không?") == {}
    assert extract_profile_updates("Con corgi tên Bơ hay phá lúc mình họp.") == {}  # pet name is not the user's name


def test_interest_decay_prunes_stale_items() -> None:
    profile = Profile()
    profile.apply(extract_profile_candidates("Mình thích Python."), max_interests=2)
    for _ in range(3):
        profile.apply(extract_profile_candidates("Mình quan tâm đến RAG và evaluation."), max_interests=2)
    assert {f.value for f in profile.ranked_interests()} == {"RAG", "evaluation"}


# --- prompt load ------------------------------------------------------------------


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path, threshold=800, keep=4)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    conversation = load_conversations(DATA_DIR / "advanced_long_context.json")[0]

    per_turn_base, per_turn_adv = [], []
    for turn in conversation["turns"]:
        per_turn_base.append(baseline.reply(conversation["user_id"], "long", turn)["prompt_tokens"])
        per_turn_adv.append(advanced.reply(conversation["user_id"], "long", turn)["prompt_tokens"])

    assert advanced.compaction_count("long") > 0
    assert advanced.prompt_token_usage("long") < baseline.prompt_token_usage("long") * 0.7
    # Baseline prompt keeps growing; advanced stays bounded near the threshold.
    assert per_turn_base[-1] > per_turn_base[0] * 5
    assert max(per_turn_adv) < config.compact_threshold_tokens + 300


def test_short_conversation_advanced_costs_more(tmp_path: Path) -> None:
    """On short threads compact never fires, so User.md is pure overhead."""

    config = make_config(tmp_path, threshold=800)
    conversations = load_conversations(DATA_DIR / "conversations.json")[:3]
    base = run_agent_benchmark("Baseline", BaselineAgent(config, force_offline=True), conversations, config)
    adv = run_agent_benchmark("Advanced", AdvancedAgent(config, force_offline=True), conversations, config)

    assert adv.compactions == 0
    assert adv.prompt_tokens_processed > base.prompt_tokens_processed
    assert adv.recall_score > base.recall_score
    assert base.memory_growth_bytes == 0 and adv.memory_growth_bytes > 0


def test_provider_aliases() -> None:
    assert normalize_provider("anthorpic") == "anthropic"
    assert normalize_provider("Google") == "gemini"
    for name in ("openai", "custom", "gemini", "anthropic", "ollama", "openrouter"):
        assert normalize_provider(name) == name
