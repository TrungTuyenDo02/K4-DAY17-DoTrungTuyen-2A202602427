from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Cheap, deterministic token estimate (~4 characters per token)."""

    text = (text or "").strip()
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


# ---------------------------------------------------------------------------
# Profile model (what lives inside User.md)
# ---------------------------------------------------------------------------

# Single-valued facts: a newer confident value replaces the old one (conflict handling).
SINGLE_FACT_KEYS = ("name", "location", "profession", "favorite_drink", "favorite_food", "pet")
# Multi-valued preference: new style items are merged into the set.
STYLE_KEY = "response_style"
INTEREST_KEY = "interests"
FACT_ORDER = SINGLE_FACT_KEYS + (STYLE_KEY,)

FACT_LABELS = {
    "name": "Tên",
    "location": "Nơi ở hiện tại",
    "profession": "Nghề nghiệp hiện tại",
    "favorite_drink": "Đồ uống yêu thích",
    "favorite_food": "Món ăn yêu thích",
    "pet": "Thú cưng",
    STYLE_KEY: "Style trả lời",
    INTEREST_KEY: "Mối quan tâm chính",
}

MAX_STYLE_ITEMS = 6
# Memory decay: an interest's weight halves every N profile turns without being mentioned again.
INTEREST_HALF_LIFE_TURNS = 40


@dataclass
class ProfileFact:
    key: str
    value: str
    confidence: float = 1.0
    seen: int = 1
    turn: int = 0
    previous: str | None = None

    def priority(self, now_turn: int, half_life: int = INTEREST_HALF_LIFE_TURNS) -> float:
        """Confidence x recency decay x (log) mention frequency."""

        age = max(0, now_turn - self.turn)
        decay = 0.5 ** (age / half_life)
        return self.confidence * decay * (1 + math.log2(max(1, self.seen)))


@dataclass(frozen=True)
class FactCandidate:
    key: str
    value: str
    confidence: float
    evidence: str = ""


@dataclass
class Profile:
    """In-memory representation of User.md. Pure logic, no I/O."""

    facts: dict[str, ProfileFact] = field(default_factory=dict)
    interests: dict[str, ProfileFact] = field(default_factory=dict)
    turn: int = 0

    # -- updates -----------------------------------------------------------
    def apply(
        self,
        candidates: list[FactCandidate],
        threshold: float = 0.6,
        max_interests: int = 8,
    ) -> dict[str, str]:
        """Apply extracted candidates. Returns {key: new_value} for what actually changed.

        - confidence threshold: candidates below `threshold` are dropped
        - conflict handling: single-valued facts are replaced, old value kept only as `prev`
        - style: merged as a small ordered set
        - interests: added/reinforced, then pruned by decayed priority
        """

        self.turn += 1
        changed: dict[str, str] = {}
        for cand in candidates:
            if cand.confidence < threshold:
                continue
            if cand.key in SINGLE_FACT_KEYS:
                if self._upsert_single(cand):
                    changed[cand.key] = cand.value
            elif cand.key == STYLE_KEY:
                if self._merge_style(cand):
                    changed[STYLE_KEY] = self.facts[STYLE_KEY].value
            elif cand.key == INTEREST_KEY:
                if self._touch_interest(cand):
                    changed.setdefault(INTEREST_KEY, cand.value)
        self._prune_interests(max_interests)
        return changed

    def _upsert_single(self, cand: FactCandidate) -> bool:
        current = self.facts.get(cand.key)
        if current and current.value.casefold() == cand.value.casefold():
            current.seen += 1
            current.turn = self.turn
            current.confidence = max(current.confidence, cand.confidence)
            return False
        self.facts[cand.key] = ProfileFact(
            cand.key,
            cand.value,
            confidence=cand.confidence,
            seen=1,
            turn=self.turn,
            previous=current.value if current else None,
        )
        return True

    def _merge_style(self, cand: FactCandidate) -> bool:
        current = self.facts.get(STYLE_KEY)
        items = split_list(current.value) if current else []
        before = list(items)
        # LRU merge: a re-mentioned item moves to the end, so the oldest unrepeated item is dropped first.
        for item in split_list(cand.value):
            items = [i for i in items if i.casefold() != item.casefold()] + [item]
        # "3 bullet" is more specific than plain "bullet".
        if any(re.fullmatch(r"\d+ bullet", i) for i in items):
            items = [i for i in items if i != "bullet"]
        items = items[-MAX_STYLE_ITEMS:]
        if current:
            current.seen += 1
            current.turn = self.turn
            current.value = ", ".join(items)
        else:
            self.facts[STYLE_KEY] = ProfileFact(STYLE_KEY, ", ".join(items), cand.confidence, 1, self.turn)
        return sorted(items) != sorted(before)

    def _touch_interest(self, cand: FactCandidate) -> bool:
        key = cand.value.casefold()
        current = self.interests.get(key)
        if current:
            current.seen += 1
            current.turn = self.turn
            current.confidence = max(current.confidence, cand.confidence)
            return False
        self.interests[key] = ProfileFact(INTEREST_KEY, cand.value, cand.confidence, 1, self.turn)
        return True

    def _prune_interests(self, max_interests: int) -> None:
        if len(self.interests) <= max_interests:
            return
        ranked = self.ranked_interests()
        keep = {fact.value.casefold() for fact in ranked[:max_interests]}
        self.interests = {k: v for k, v in self.interests.items() if k in keep}

    # -- reads -------------------------------------------------------------
    def ranked_interests(self) -> list[ProfileFact]:
        return sorted(self.interests.values(), key=lambda f: (-f.priority(self.turn), f.value))

    def as_dict(self, max_interests: int = 4) -> dict[str, str]:
        out = {key: self.facts[key].value for key in FACT_ORDER if key in self.facts}
        top = [f.value for f in self.ranked_interests()[:max_interests]]
        if top:
            out[INTEREST_KEY] = ", ".join(top)
        return out

    def render_for_prompt(self, max_interests: int = 4) -> str:
        """Clean digest injected into the prompt (no metadata comments)."""

        lines = [f"- {FACT_LABELS[k]}: {v}" for k, v in self.as_dict(max_interests).items()]
        return "\n".join(lines)


def split_list(value: str) -> list[str]:
    return [part.strip() for part in re.split(r",\s*", value or "") if part.strip()]


# ---------------------------------------------------------------------------
# User.md persistence
# ---------------------------------------------------------------------------

_FACT_LINE = re.compile(r"^- (?P<key>[a-z_]+): (?P<value>.*?)\s*(?:<!--(?P<meta>.*?)-->)?\s*$")
_ITEM_LINE = re.compile(r"^- (?P<value>.*?)\s*(?:<!--(?P<meta>.*?)-->)?\s*$")
_TURN_LINE = re.compile(r"<!--\s*profile-turn=(\d+)\s*-->")


def _format_meta(fact: ProfileFact) -> str:
    meta = f"conf={fact.confidence:.2f} seen={fact.seen} turn={fact.turn}"
    if fact.previous:
        meta += f" prev={fact.previous.replace(' ', '_')}"
    return meta


def _parse_meta(raw: str | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in (raw or "").split():
        if "=" in part:
            k, v = part.split("=", 1)
            out[k] = v
    return out


def _fact_from_line(key: str, value: str, raw_meta: str | None) -> ProfileFact:
    meta = _parse_meta(raw_meta)
    prev = meta.get("prev")
    return ProfileFact(
        key,
        value.strip(),
        confidence=float(meta.get("conf", 1.0)),
        seen=int(meta.get("seen", 1)),
        turn=int(meta.get("turn", 0)),
        previous=prev.replace("_", " ") if prev else None,
    )


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`: one markdown file per user id."""

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        folded = unicodedata.normalize("NFKD", nfc(user_id).replace("đ", "d").replace("Đ", "D"))
        ascii_id = folded.encode("ascii", "ignore").decode()
        slug = re.sub(r"[^A-Za-z0-9_-]+", "_", ascii_id).strip("_").lower() or "anonymous"
        return Path(self.root_dir) / slug / "User.md"

    def default_text(self, user_id: str) -> str:
        return f"# User.md: {user_id}\n\n## Facts\n\n## Interests\n\n<!-- profile-turn=0 -->\n"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return self.default_text(user_id)

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        content = self.read_text(user_id)
        if not search_text or search_text not in content:
            return False
        self.write_text(user_id, content.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    # -- structured helpers --------------------------------------------------
    def load_profile(self, user_id: str) -> Profile:
        profile = Profile()
        section = None
        for line in self.read_text(user_id).splitlines():
            stripped = line.strip()
            if stripped.startswith("## "):
                section = stripped[3:].strip().lower()
                continue
            turn_match = _TURN_LINE.search(stripped)
            if turn_match:
                profile.turn = int(turn_match.group(1))
                continue
            if section == "facts":
                match = _FACT_LINE.match(stripped)
                if match:
                    key = match.group("key")
                    profile.facts[key] = _fact_from_line(key, match.group("value"), match.group("meta"))
            elif section == "interests":
                match = _ITEM_LINE.match(stripped)
                if match and match.group("value"):
                    fact = _fact_from_line(INTEREST_KEY, match.group("value"), match.group("meta"))
                    profile.interests[fact.value.casefold()] = fact
        return profile

    def save_profile(self, user_id: str, profile: Profile) -> Path:
        lines = [f"# User.md: {user_id}", "", "## Facts"]
        ordered = [k for k in FACT_ORDER if k in profile.facts] + sorted(
            k for k in profile.facts if k not in FACT_ORDER
        )
        for key in ordered:
            fact = profile.facts[key]
            lines.append(f"- {key}: {fact.value} <!-- {_format_meta(fact)} -->")
        lines += ["", "## Interests"]
        for fact in profile.ranked_interests():
            lines.append(f"- {fact.value} <!-- {_format_meta(fact)} -->")
        lines += ["", f"<!-- profile-turn={profile.turn} -->", ""]
        return self.write_text(user_id, "\n".join(lines))

    def facts(self, user_id: str) -> dict[str, str]:
        return self.load_profile(user_id).as_dict()

    def upsert_fact(self, user_id: str, key: str, value: str, confidence: float = 1.0) -> bool:
        """Explicit write (used by tools / tests). Bypasses the confidence threshold."""

        profile = self.load_profile(user_id)
        values = split_list(nfc(value)) if key == INTEREST_KEY else [nfc(value).strip()]
        changed = profile.apply([FactCandidate(key, v, confidence) for v in values], threshold=0.0)
        self.save_profile(user_id, profile)
        return bool(changed)

    def apply_message(
        self,
        user_id: str,
        message: str,
        threshold: float = 0.6,
        max_interests: int = 8,
    ) -> dict[str, str]:
        """Extract candidates from one user message and persist the confident ones.

        The profile `turn` clock (used for decay) only ticks on messages that carry
        profile evidence, so chit-chat does not age stored facts.
        """

        candidates = extract_profile_candidates(message)
        if not candidates:
            return {}
        profile = self.load_profile(user_id)
        changed = profile.apply(candidates, threshold=threshold, max_interests=max_interests)
        self.save_profile(user_id, profile)
        return changed

    def render_for_prompt(self, user_id: str) -> str:
        return self.load_profile(user_id).render_for_prompt()


# ---------------------------------------------------------------------------
# Fact extraction (heuristic, Vietnamese)
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+|\n+")
_QUESTION_START = re.compile(
    r"^(bạn (có )?(biết|thể|nhớ)|bạn thử nhớ|nhắc lại|cho mình hỏi|tóm tắt|hãy nhắc)\b", re.IGNORECASE
)
# Sentence-level hedges lower confidence: jokes, hypotheticals, temporary context.
_HEDGES = (
    (re.compile(r"\bđùa\b", re.I), 0.6),
    (re.compile(r"^\s*nếu\b|giả sử", re.I), 0.4),
    (re.compile(r"\bhay là\b", re.I), 0.3),
    (re.compile(r"chỉ là nơi|không phải nơi ở", re.I), 0.5),
    (re.compile(r"có lẽ|đang cân nhắc|có thể sẽ", re.I), 0.3),
    (re.compile(r"tạm thời", re.I), 0.4),
)
_BOOSTS = re.compile(r"đính chính|thực ra|cập nhật|hiện tại|chuyển sang|nhớ là|vẫn", re.I)
# Short-lived time markers: fine for the thread, weak evidence for long-term interests.
_TEMPORAL = re.compile(r"tuần này|hôm nay|hôm qua|tối nay|chiều nay|sáng nay|tối qua|lát nữa", re.I)
# A match is ignored when this appears right before it ("không còn ở Đà Nẵng").
_NEGATION_BEFORE = re.compile(r"(không còn|không phải|chứ không|đừng|không làm|lúc đầu|trước đó|cũ)\s*(\S+\s*){0,3}$", re.I)

_STOP_WORDS = (" như ", " nhưng ", " và ", " vì ", " để ", " cho ", " mỗi ", " rồi ", " thì ")

# Captures are bounded to a few words so later matches in the same sentence stay visible.
_PHRASE = r"(\S+(?:\s+\S+){0,3})"
_NAME_PATTERNS = (
    re.compile(r"\b(?:mình|tôi|em)\s+tên\s+là\s+" + _PHRASE, re.I),
    re.compile(r"\btên\s+(?:của\s+)?(?:mình|tôi)\s+là\s+" + _PHRASE, re.I),
    re.compile(r"\bgọi\s+(?:mình|tôi)\s+là\s+" + _PHRASE, re.I),
)
_LOCATION_PATTERNS = (
    re.compile(r"(?:^|\s)(?:đang\s+|vẫn\s+|hiện\s+)?(?:sống\s+|làm việc\s+)?ở\s+(?:tại\s+)?" + _PHRASE, re.I),
    re.compile(r"nơi ở\s+(?:hiện tại\s+)?(?:vẫn\s+)?là\s+" + _PHRASE, re.I),
    re.compile(r"(?:chuyển|dọn)\s+(?:về|đến|tới|ra)\s+" + _PHRASE, re.I),
)
_ROLE = re.compile(
    r"\b([A-Za-z][A-Za-z0-9+/.-]*\s+(?:engineer|manager|developer|scientist|designer|analyst|researcher|architect))\b",
    re.I,
)
_ROLE_TRIGGER = re.compile(r"(làm|là|sang|nghề)\s*(\S+\s*){0,2}$", re.I)
_DRINK_PATTERNS = (
    (re.compile(r"đồ uống (?:yêu thích|ruột)(?:\s+của\s+mình)?\s+(?:vẫn\s+)?là\s+([^.,;!?]+)", re.I), 0.95),
    (re.compile(r"\b(?:mình|tôi)\s+(?:vẫn\s+|thường\s+|hay\s+)?uống\s+([^.,;!?]+)", re.I), 0.7),
)
_DRINK_WORDS = re.compile(r"(cà phê[^,.;!?]*?|trà sữa|trà [a-zà-ỹ]+|sinh tố [a-zà-ỹ]+)(?=,| và |\.|$)", re.I)
_FOOD_PATTERN = re.compile(r"món (?:ăn )?(?:yêu thích|ruột)(?:\s+của\s+mình)?\s+(?:vẫn\s+)?là\s+([^.,;!?]+)", re.I)
_PET_PATTERN = re.compile(r"\bnuôi\s+(?:một\s+)?(?:bé|con|chú|em)?\s*([a-zà-ỹ]+)(?:\s+tên\s+([^\s,.;!?]+))?", re.I)

_STYLE_CONTEXT = re.compile(r"trả lời|giải thích|style|trình bày|câu trả lời", re.I)
_STYLE_ITEMS = (
    (re.compile(r"ngắn gọn|ngắn và có cấu trúc|bullet ngắn", re.I), "ngắn gọn"),
    (re.compile(r"(\d+)\s*bullet", re.I), None),  # "3 bullet"
    (re.compile(r"bullet", re.I), "bullet"),
    (re.compile(r"rõ ý", re.I), "rõ ý"),
    (re.compile(r"có cấu trúc", re.I), "có cấu trúc"),
    (re.compile(r"ví dụ thực tế", re.I), "có ví dụ thực tế"),
    (re.compile(r"ví dụ thực chiến", re.I), "có ví dụ thực chiến"),
    (re.compile(r"trade-off", re.I), "nhấn trade-off"),
    (re.compile(r"lan man", re.I), "không lan man"),
)
_INTEREST_CONTEXT = (
    (re.compile(r"\bthích\b", re.I), 0.8),
    (re.compile(r"quan tâm", re.I), 0.85),
    (re.compile(r"đang học|học thêm|ôn lại|đọc về", re.I), 0.7),
)
_INTEREST_TERMS = (
    ("async Python", re.compile(r"async python", re.I)),
    ("Python", re.compile(r"\bpython\b", re.I)),
    ("AI ứng dụng", re.compile(r"\bAI ứng dụng\b", re.I)),
    ("AI agent", re.compile(r"\bAI agent\b", re.I)),
    ("MLOps", re.compile(r"\bMLOps\b(?!\s+engineer)", re.I)),
    ("RAG", re.compile(r"\bRAG\b")),
    ("evaluation", re.compile(r"\bevaluation\b", re.I)),
    ("memory architecture", re.compile(r"memory architecture", re.I)),
    ("benchmark memory", re.compile(r"benchmark memory", re.I)),
    ("LangChain", re.compile(r"\blangchain\b", re.I)),
    ("LangGraph", re.compile(r"\blanggraph\b", re.I)),
)


def split_sentences(message: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(nfc(message)) if s and s.strip()]


def is_question_sentence(sentence: str) -> bool:
    s = sentence.strip()
    return s.endswith("?") or bool(_QUESTION_START.match(s))


def _sentence_confidence(sentence: str, base: float = 0.9, instruction: bool = False) -> float:
    """`instruction=True`: "Nếu bạn giải thích, hãy..." is a directive, not a hypothetical."""

    conf = base
    for pattern, penalty in _HEDGES:
        if instruction and pattern is _HEDGES[1][0]:
            continue
        if pattern.search(sentence):
            conf -= penalty
    if _BOOSTS.search(sentence):
        conf += 0.05
    return round(max(0.0, min(1.0, conf)), 2)


def _negated(prefix: str) -> bool:
    return bool(_NEGATION_BEFORE.search(prefix[-30:]))


def _leading_proper_noun(text: str, max_words: int = 4) -> str:
    """Take consecutive capitalised words: 'Đà Nẵng và đang...' -> 'Đà Nẵng'."""

    words: list[str] = []
    for raw in text.split()[:max_words]:
        word = raw.strip("\"'“”()")
        clean = word.rstrip(",.;:!?…")
        if not clean or not clean[0].isupper():
            break
        words.append(clean)
        if clean != word:  # punctuation ends the noun phrase
            break
    return " ".join(words)


def _cut_phrase(text: str, max_words: int = 5) -> str:
    phrase = f" {text.strip()} "
    for stop in _STOP_WORDS:
        idx = phrase.find(stop)
        if idx > 0:
            phrase = phrase[:idx]
    return " ".join(phrase.split()[:max_words]).strip(" ,.")


def _last_valid_match(patterns, sentence: str, to_value) -> str | None:
    """Return the value of the last non-negated match across patterns (later = more current)."""

    best: tuple[int, str] | None = None
    for pattern in patterns:
        for match in pattern.finditer(sentence):
            if _negated(sentence[: match.start(1)]):
                continue
            value = to_value(match)
            if value and (best is None or match.start(1) >= best[0]):
                best = (match.start(1), value)
    return best[1] if best else None


def extract_profile_candidates(message: str) -> list[FactCandidate]:
    """Sentence-level extraction with a confidence score per candidate."""

    out: list[FactCandidate] = []
    for sentence in split_sentences(message):
        if is_question_sentence(sentence):
            continue
        conf = _sentence_confidence(sentence)

        def add(key: str, value: str | None, confidence: float = conf) -> None:
            if value:
                out.append(FactCandidate(key, value.strip(), round(confidence, 2), sentence))

        add("name", _last_valid_match(_NAME_PATTERNS, sentence, lambda m: _leading_proper_noun(m.group(1))))
        add("location", _last_valid_match(_LOCATION_PATTERNS, sentence, lambda m: _leading_proper_noun(m.group(1))))

        roles = [
            m.group(1)
            for m in _ROLE.finditer(sentence)
            if _ROLE_TRIGGER.search(sentence[: m.start(1)][-20:]) and not _negated(sentence[: m.start(1)])
        ]
        if roles:
            role = roles[-1]
            first, rest = role.split(None, 1)
            add("profession", f"{first} {rest.lower()}")

        for pattern, base in _DRINK_PATTERNS:
            match = pattern.search(sentence)
            if match:
                add("favorite_drink", _cut_phrase(match.group(1)), min(conf, base + 0.05))
                break
        else:
            if re.search(r"\bthích\b", sentence, re.I):
                drink = _DRINK_WORDS.search(sentence)
                if drink:
                    add("favorite_drink", _cut_phrase(drink.group(1)), min(conf, 0.7))

        food = _FOOD_PATTERN.search(sentence)
        if food:
            add("favorite_food", _cut_phrase(food.group(1)))

        pet = _PET_PATTERN.search(sentence)
        if pet:
            add("pet", f"{pet.group(1)} tên {pet.group(2)}" if pet.group(2) else pet.group(1))

        if _STYLE_CONTEXT.search(sentence):
            items: list[str] = []
            for pattern, label in _STYLE_ITEMS:
                match = pattern.search(sentence)
                if match:
                    item = label or f"{match.group(1)} bullet"
                    if item not in items:
                        items.append(item)
            if any(re.fullmatch(r"\d+ bullet", i) for i in items):
                items = [i for i in items if i != "bullet"]
            if items:
                add(STYLE_KEY, ", ".join(items), _sentence_confidence(sentence, instruction=True))

        interest_conf = max((c for p, c in _INTEREST_CONTEXT if p.search(sentence)), default=0.0)
        if interest_conf:
            interest_conf = min(interest_conf, conf)
            if _TEMPORAL.search(sentence):
                interest_conf -= 0.25
            for label, pattern in _INTEREST_TERMS:
                if pattern.search(sentence):
                    if label == "Python" and re.search(r"async python", sentence, re.I):
                        continue
                    add(INTEREST_KEY, label, interest_conf)
    return out


def extract_profile_updates(message: str, threshold: float = 0.6) -> dict[str, str]:
    """Stable profile facts confidently present in `message` ({key: value}).

    Question-only sentences are skipped; jokes / hypotheticals / negated mentions
    get low confidence and are filtered by `threshold`.
    """

    updates: dict[str, str] = {}
    for cand in extract_profile_candidates(message):
        if cand.confidence < threshold:
            continue
        if cand.key == INTEREST_KEY or cand.key == STYLE_KEY:
            existing = split_list(updates.get(cand.key, ""))
            for item in split_list(cand.value):
                if item not in existing:
                    existing.append(item)
            updates[cand.key] = ", ".join(existing)
        else:
            updates[cand.key] = cand.value
    return updates


# ---------------------------------------------------------------------------
# Compact memory
# ---------------------------------------------------------------------------


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6, previous: str = "") -> str:
    """Heuristic summary: one clipped key sentence per user message, newest kept.

    Assistant acknowledgements are dropped (low information). `previous` summary
    lines are merged so repeated compactions stay bounded at `max_items` bullets.
    """

    lines = [line for line in previous.splitlines() if line.startswith("- ")]
    for message in messages:
        if message.get("role") != "user":
            continue
        sentences = split_sentences(message.get("content", ""))
        if not sentences:
            continue
        # Prefer the sentence carrying the most extractable facts, else the first one.
        scored = sorted(
            enumerate(sentences),
            key=lambda pair: (-len(extract_profile_candidates(pair[1])), pair[0]),
        )
        lines.append(f"- {_clip(scored[0][1], 110)}")
    return "\n".join(lines[-max_items:])


@dataclass
class CompactMemoryManager:
    """Short-term thread memory that compacts itself when it grows too large.

    - recent `keep_messages` messages are kept verbatim
    - when summary + messages exceed `threshold_tokens`, older messages are folded
      into a bounded summary and `compactions` is incremented
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)
    summary_max_items: int = 6

    def _thread(self, thread_id: str) -> dict[str, object]:
        if thread_id not in self.state:
            self.state[thread_id] = {"messages": [], "summary": "", "compactions": 0, "compacted_messages": 0}
        return self.state[thread_id]

    def append(self, thread_id: str, role: str, content: str) -> None:
        thread = self._thread(thread_id)
        thread["messages"].append({"role": role, "content": content})
        if self.total_tokens(thread_id) > self.threshold_tokens and len(thread["messages"]) > self.keep_messages:
            self.compact(thread_id)

    def compact(self, thread_id: str) -> None:
        thread = self._thread(thread_id)
        messages: list[dict[str, str]] = thread["messages"]
        cut = len(messages) - self.keep_messages if self.keep_messages > 0 else len(messages)
        if cut <= 0:
            return
        old, recent = messages[:cut], messages[cut:]
        thread["summary"] = summarize_messages(old, self.summary_max_items, previous=thread["summary"])
        thread["messages"] = recent
        thread["compactions"] += 1
        thread["compacted_messages"] += len(old)

    def total_tokens(self, thread_id: str) -> int:
        thread = self._thread(thread_id)
        return estimate_tokens(thread["summary"]) + sum(estimate_tokens(m["content"]) for m in thread["messages"])

    def context(self, thread_id: str) -> dict[str, object]:
        return self._thread(thread_id)

    def compaction_count(self, thread_id: str) -> int:
        return int(self.state.get(thread_id, {}).get("compactions", 0))
