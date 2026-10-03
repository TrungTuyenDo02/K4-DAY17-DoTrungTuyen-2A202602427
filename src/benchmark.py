from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent, message_text
from config import load_config
from memory_store import estimate_tokens, nfc

COLUMNS = [
    "Agent",
    "Agent tokens only",
    "Prompt tokens processed",
    "Cross-session recall",
    "Response quality",
    "Memory growth (bytes)",
    "Compactions",
]


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int
    details: list[dict[str, Any]] = field(default_factory=list, repr=False)


def load_conversations(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, list) else [data]


def _norm(text: str) -> str:
    return nfc(text).casefold()


def _hits(answer: str, expected: list[str]) -> int:
    answer_n = _norm(answer)
    return sum(1 for item in expected if _norm(item) in answer_n)


def recall_points(answer: str, expected: list[str]) -> float:
    """1 if every expected fact appears, 0.5 if some do, 0 if none."""

    if not expected:
        return 1.0
    hits = _hits(answer, expected)
    if hits == len(expected):
        return 1.0
    return 0.5 if hits else 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Lightweight 0..1 quality score for offline mode.

    - 0.6 coverage: share of expected facts present
    - 0.2 concision: full marks up to 60 tokens, decays after
    - 0.1 structure: bullet list or at most two lines
    - 0.1 groundedness: no "chưa có thông tin" placeholders
    """

    if not answer.strip():
        return 0.0
    coverage = _hits(answer, expected) / len(expected) if expected else 1.0
    tokens = estimate_tokens(answer)
    concision = 1.0 if tokens <= 60 else max(0.0, 1 - (tokens - 60) / 120)
    lines = [line for line in answer.splitlines() if line.strip()]
    structured = 1.0 if len(lines) <= 2 or all(line.lstrip().startswith(("-", "*", "•")) for line in lines) else 0.5
    grounded = 0.0 if "chưa có thông tin" in _norm(answer) else 1.0
    return round(0.6 * coverage + 0.2 * concision + 0.1 * structured + 0.1 * grounded, 3)


def judge_quality(judge, question: str, answer: str, expected: list[str]) -> float | None:
    """Optional LLM-as-judge (live mode). Returns None on any failure so we fall back to heuristics."""

    prompt = (
        "Chấm chất lượng câu trả lời của trợ lý trên thang 0 đến 1.\n"
        f"Câu hỏi: {question}\nCác fact cần có: {', '.join(expected)}\nCâu trả lời: {answer}\n"
        "Tiêu chí: đúng fact hiện tại, không lẫn fact cũ, ngắn gọn. Chỉ in ra một số thập phân."
    )
    try:
        raw = message_text(judge.invoke(prompt).content)
        match = re.search(r"[01](?:[.,]\d+)?", raw)
        return max(0.0, min(1.0, float(match.group(0).replace(",", ".")))) if match else None
    except Exception:
        return None


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config, judge=None) -> BenchmarkRow:
    """Feed every conversation, then ask its recall questions in a fresh thread."""

    users = sorted({conv["user_id"] for conv in conversations})
    size_before = {user: agent.memory_file_size(user) for user in users}
    threads: list[str] = []
    recalls: list[float] = []
    qualities: list[float] = []
    details: list[dict[str, Any]] = []

    for conv in conversations:
        user_id = conv["user_id"]
        chat_thread = f"{conv['id']}:chat"
        threads.append(chat_thread)
        for turn in conv["turns"]:
            agent.reply(user_id, chat_thread, turn)

        for idx, item in enumerate(conv.get("recall_questions", []), start=1):
            recall_thread = f"{conv['id']}:recall-{idx}"  # new thread = new session
            threads.append(recall_thread)
            answer = agent.reply(user_id, recall_thread, item["question"])["response"]
            expected = item["expected_contains"]
            recall = recall_points(answer, expected)
            quality = judge_quality(judge, item["question"], answer, expected) if judge else None
            if quality is None:
                quality = heuristic_quality(answer, expected)
            recalls.append(recall)
            qualities.append(quality)
            details.append(
                {"conversation": conv["id"], "question": item["question"], "answer": answer, "recall": recall, "quality": quality}
            )

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=sum(agent.token_usage(t) for t in threads),
        prompt_tokens_processed=sum(agent.prompt_token_usage(t) for t in threads),
        recall_score=round(sum(recalls) / len(recalls), 3) if recalls else 0.0,
        response_quality=round(sum(qualities) / len(qualities), 3) if qualities else 0.0,
        memory_growth_bytes=sum(agent.memory_file_size(u) - size_before[u] for u in users),
        compactions=sum(agent.compaction_count(t) for t in threads),
        details=details,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    table = [
        [
            r.agent_name,
            r.agent_tokens_only,
            r.prompt_tokens_processed,
            f"{r.recall_score:.2f}",
            f"{r.response_quality:.2f}",
            r.memory_growth_bytes,
            r.compactions,
        ]
        for r in rows
    ]
    try:
        from tabulate import tabulate

        return tabulate(table, headers=COLUMNS, tablefmt="github")
    except ImportError:
        lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
        lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in table]
        return "\n".join(lines)


def _delta_line(baseline: BenchmarkRow, advanced: BenchmarkRow) -> str:
    def pct(new: int, old: int) -> str:
        return f"{(new - old) / old * 100:+.1f}%" if old else "n/a"

    return (
        f"Advanced vs Baseline: agent tokens {pct(advanced.agent_tokens_only, baseline.agent_tokens_only)}, "
        f"prompt tokens {pct(advanced.prompt_tokens_processed, baseline.prompt_tokens_processed)}, "
        f"recall {advanced.recall_score - baseline.recall_score:+.2f}"
    )


def run_suite(title: str, dataset: Path, config, judge=None, verbose: bool = False) -> tuple[str, list[BenchmarkRow]]:
    conversations = load_conversations(dataset)
    baseline = BaselineAgent(config, force_offline=not config.live_mode)
    advanced = AdvancedAgent(config, force_offline=not config.live_mode)
    rows = [
        run_agent_benchmark("Baseline", baseline, conversations, config, judge),
        run_agent_benchmark("Advanced", advanced, conversations, config, judge),
    ]
    out = [f"## {title}", f"Dataset: `data/{dataset.name}` | mode: {advanced.mode}", "", format_rows(rows), "", _delta_line(*rows)]
    if verbose:
        for row in rows:
            out += ["", f"### {row.agent_name}: recall answers"]
            for d in row.details:
                answer = d["answer"].replace("\n", " / ")
                out.append(f"- [{d['conversation']}] recall={d['recall']} quality={d['quality']:.2f} :: {answer}")
    return "\n".join(out), rows


def main() -> None:
    """Run the Standard benchmark and the Long-Context Stress benchmark, Baseline vs Advanced."""

    parser = argparse.ArgumentParser(description="Day 17 memory benchmark")
    parser.add_argument("--verbose", action="store_true", help="print every recall answer")
    parser.add_argument("--save", type=Path, help="also write the report to this markdown file")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    config = load_config(Path(__file__).resolve().parent.parent)
    # Fresh, isolated state so memory growth is measured from zero on every run.
    bench_state = config.state_dir / "benchmark"
    shutil.rmtree(bench_state, ignore_errors=True)
    config = replace(config, state_dir=bench_state)

    judge = None
    if config.live_mode and config.judge_model.is_live_ready:
        from model_provider import build_chat_model

        judge = build_chat_model(config.judge_model)

    header = (
        f"# Day 17 Memory Benchmark\n\ncompact_threshold_tokens={config.compact_threshold_tokens}, "
        f"compact_keep_messages={config.compact_keep_messages}, "
        f"profile_confidence_threshold={config.profile_confidence_threshold}"
    )
    standard, _ = run_suite("Standard Benchmark", config.data_dir / "conversations.json", config, judge, args.verbose)
    stress, _ = run_suite(
        "Long-Context Stress Benchmark", config.data_dir / "advanced_long_context.json", config, judge, args.verbose
    )
    report = "\n\n".join([header, standard, stress]) + "\n"
    print(report)
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
