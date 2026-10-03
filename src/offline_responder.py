"""Deterministic offline replies shared by both agents.

Both agents use the *same* answering logic; the only difference is which facts
they can see (baseline: facts derived from the current thread only, advanced:
User.md + compact summary + recent messages). That keeps the comparison fair.
"""

from __future__ import annotations

import re

from memory_store import FACT_LABELS, INTEREST_KEY, is_question_sentence, nfc, split_sentences

# (fact key, pattern in the question that asks for it)
_ASKS = (
    ("name", re.compile(r"\btên\b|là ai|tóm tắt", re.I)),
    ("profession", re.compile(r"nghề|công việc|làm gì|là ai|tóm tắt", re.I)),
    ("location", re.compile(r"ở đâu|nơi ở|sống ở|còn ở", re.I)),
    ("favorite_drink", re.compile(r"đồ uống|uống gì", re.I)),
    ("favorite_food", re.compile(r"món ăn|ăn gì", re.I)),
    ("pet", re.compile(r"\bnuôi\b|thú cưng", re.I)),
    ("response_style", re.compile(r"style|kiểu trả lời|cách trả lời|trả lời mình thích", re.I)),
    (INTEREST_KEY, re.compile(r"quan tâm|sở thích|là ai|tóm tắt", re.I)),
)


def requested_keys(message: str) -> list[str]:
    """Fact keys a recall question asks for; empty when the message is not a recall request."""

    sentences = split_sentences(message)
    asking = [s for s in sentences if is_question_sentence(s) or re.search(r"nhớ lại|nhắc lại", s, re.I)]
    if not asking:
        return []
    text = " ".join(asking)
    return [key for key, pattern in _ASKS if pattern.search(text)]


def answer_from_facts(message: str, facts: dict[str, str], scope: str) -> str | None:
    """Bullet answer for a recall question, or None if the message asks for nothing."""

    keys = requested_keys(message)
    if not keys:
        return None
    lines = []
    for key in keys:
        value = facts.get(key)
        lines.append(f"- {FACT_LABELS[key]}: {value}" if value else f"- {FACT_LABELS[key]}: chưa có thông tin {scope}")
    return "\n".join(lines)


def acknowledge(message: str, saved: dict[str, str] | None = None) -> str:
    """Short acknowledgement for a normal (non-recall) turn."""

    first = split_sentences(nfc(message))[0] if message.strip() else ""
    gist = " ".join(first.split()[:12])
    reply = f"Đã ghi nhận: {gist}"
    if saved:
        items = ", ".join(f"{FACT_LABELS.get(k, k)} = {v}" for k, v in saved.items())
        reply += f"\n(Đã lưu vào User.md: {items})"
    return reply
