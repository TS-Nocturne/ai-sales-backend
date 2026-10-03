"""Text heuristics that patch over Gemini's quirks, one observed failure at a time.

Nothing here is clever: each marker list or regex exists because the model was
seen misbehaving in production chats (see README, "Why heuristics.py exists").
They are pure string functions with no I/O so they stay easy to unit test; the
graph nodes and tools import them and decide what to do with the answer.

Groups:
- Filler detection: the model *promises* a search but emits no tool call.
- Browse / recommend / budget intent: vague questions that must reach
  ``list_products`` instead of ending in "ไม่เข้าใจ".
- Query normalisation: vague phrases rewritten to concrete search keywords.
- Closing intent and confused replies: steer the turn after an offer.
"""

import re

from ai_sales.tools.order_total import looks_like_cod_intent, looks_like_transfer_intent

_PURE_CATALOG_BROWSE_RE = re.compile(
    r"มี(?:สินค้า)?อะไร(?:บ้าง|ขาย)|มีอะไรขาย|ขายอะไร|"
    r"สินค้ามีอะไร|ดูสินค้า|มีของอะไร",
    re.IGNORECASE,
)

_BROAD_BROWSE_QUERIES = frozenset(
    {
        "สินค้าแนะนำ",
        "สินค้าทั้งหมด",
        "สินค้ายอดนิยม",
        "อุปกรณ์เสริมมือถือ",
    }
)

_BROAD_BROWSE_RE = re.compile(
    r"มี(?:อะไร|สินค้า).*(?:ขาย|บ้าง)|มีอะไรขาย|ขายอะไร|"
    r"แนะนำ(?:สินค้า)?.*(?:หน่อย|บ้าง)|ซื้ออะไรได้|มีรุ่น(?:ไหน|ใหน)",
    re.IGNORECASE,
)


def _tokenize_query(query: str) -> list[str]:
    """Split a search query into meaningful tokens (supports Thai + Latin)."""
    query_lower = query.lower().strip()
    if not query_lower:
        return []
    tokens = [t for t in re.split(r"[\s,./\-_]+", query_lower) if len(t) >= 2]
    return tokens or [query_lower]


_VAGUE_QUERY_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"มีรุ่น.*(?:แนะนำ|บ้าง)|(?:แนะนำ|บ้าง).*รุ่น|มีรุ่นไหน|รุ่นไหนบ้าง",
            re.I,
        ),
        "สินค้าแนะนำ",
    ),
    (
        re.compile(
            r"แนะนำ.*(?:หน่อย|บ้าง)|มีอะไร(?:ขาย|บ้าง)|มีสินค้าอะไร|ขายอะไร",
            re.I,
        ),
        "สินค้าแนะนำ",
    ),
    (
        re.compile(r"ซื้ออะไรได้|งบ.*ซื้อ|ในงบ", re.I),
        "สินค้าแนะนำ",
    ),
)

_META_ONLY_WORDS = frozenset(
    {
        "รุ่น",
        "แนะนำ",
        "บ้าง",
        "หน่อย",
        "มี",
        "อะไร",
        "ขาย",
        "สินค้า",
        "ให้",
        "ครับ",
        "ค่ะ",
        "นะ",
        "คะ",
        "ได้",
        "ซื้อ",
        "งบ",
        "ไหน",
        "ใหน",
    }
)


def _normalize_search_query(query: str) -> str:
    """Rewrite vague/meta customer phrases into concrete vector-search keywords."""
    q = (query or "").strip()
    if not q:
        return q
    for pattern, replacement in _VAGUE_QUERY_PATTERNS:
        if pattern.search(q):
            return replacement
    tokens = _tokenize_query(q)
    if tokens and all(token in _META_ONLY_WORDS for token in tokens):
        return "สินค้าแนะนำ"
    if len(q) <= 12 and tokens == ["รุ่น"]:
        return "สินค้าแนะนำ"
    return q


def _is_pure_catalog_browse(text: str) -> bool:
    """True for 'what do you sell' — route to category overview, not product dump."""
    return bool(_PURE_CATALOG_BROWSE_RE.search((text or "").strip()))


def _is_broad_browse_query(query: str, *, raw_query: str | None = None) -> bool:
    """True when the query is a vague browse intent (not a specific product lookup)."""
    for candidate in (raw_query, query):
        if not candidate or not str(candidate).strip():
            continue
        text = str(candidate).strip()
        if _BROAD_BROWSE_RE.search(text):
            return True
        normalized = _normalize_search_query(text)
        compact = re.sub(r"\s+", "", normalized.lower())
        if any(
            re.sub(r"\s+", "", key.lower()) == compact for key in _BROAD_BROWSE_QUERIES
        ):
            return True
    return False


# Phrases that signal the model is *announcing* a search instead of doing it.
# When such a message arrives with NO tool call, the turn would end on filler,
# so we force a real tool call (see sales_agent_node safety net).
_SEARCH_FILLER_MARKERS = (
    "ขอค้นหา",
    "ขอเช็ก",
    "ขอเช็ค",
    "ขอตรวจสอบ",
    "ขอดูข้อมูล",
    "รอสักครู่",
    "สักครู่นะ",
    "เดี๋ยวเช็ก",
    "เดี๋ยวเช็ค",
    "กำลังค้นหา",
    "กำลังตรวจสอบ",
    "let me search",
    "let me check",
    "searching",
    "checking",
    "one moment",
)


def _looks_like_search_filler(text: str) -> bool:
    """True when the text merely promises to search rather than answering."""
    low = text.lower()
    return any(marker.lower() in low for marker in _SEARCH_FILLER_MARKERS)


_BROAD_CATALOG_MARKERS = (
    "มีสินค้าอะไร",
    "มีอะไรบ้าง",
    "มีอะไรขาย",
    "ขายอะไร",
    "สินค้ามีอะไร",
    "ดูสินค้า",
    "แนะนำสินค้า",
    "มีของอะไร",
    "มีอะไรแนะนำ",
    "แนะนำหน่อย",
)

_RECOMMEND_MARKERS = (
    "มีรุ่นไหน",
    "มีรุ่นใหน",
    "รุ่นไหนบ้าง",
    "รุ่นไหนแนะนำ",
    "แนะนำบ้าง",
    "มีอะไรแนะนำ",
)

_CLOSING_INTENT = re.compile(
    r"(สนใจ|เอาอันนี้|เอาเลย|ต้องทำยังไง|สั่งยังไง|จะซื้อ|รับเลย|สั่งเลย|ซื้อยังไง)",
    re.IGNORECASE,
)

_BUDGET_CEILING = re.compile(
    r"(?:งบ(?:ประมาณ)?|budget|ไม่เกิน|ภายใน)\s*([\d,.]+)",
    re.IGNORECASE,
)

_CONFUSED_REPLY_MARKERS = (
    "ไม่แน่ใจ",
    "ไม่เข้าใจ",
    "ไม่ทราบว่า",
)


def _looks_like_broad_catalog_query(text: str) -> bool:
    """True when the customer asks an open-ended what-do-you-sell question."""
    compact = re.sub(r"\s+", "", (text or "").lower())
    if not compact:
        return False
    return any(marker.replace(" ", "") in compact for marker in _BROAD_CATALOG_MARKERS)


def _looks_like_recommend_query(text: str) -> bool:
    """True when the customer asks which models/products to recommend."""
    compact = re.sub(r"\s+", "", (text or "").lower())
    if not compact:
        return False
    return any(marker.replace(" ", "") in compact for marker in _RECOMMEND_MARKERS)


def _looks_like_budget_browse_query(text: str) -> bool:
    """True for 'มีงบ X ซื้ออะไรได้บ้าง' style questions."""
    compact = re.sub(r"\s+", "", (text or "").lower())
    if not compact:
        return False
    has_budget = "งบ" in compact or "budget" in compact
    has_browse = any(w in compact for w in ("ซื้ออะไร", "ได้บ้าง", "แนะนำ", "อะไรได้"))
    return has_budget and has_browse


def _extract_budget_ceiling(text: str) -> float:
    match = _BUDGET_CEILING.search(text or "")
    if not match:
        return 0.0
    raw = match.group(1).replace(",", "").strip()
    try:
        value = float(raw)
    except ValueError:
        return 0.0
    return value if value > 0 else 0.0


def _should_auto_browse_catalog(text: str) -> bool:
    """Browse intents that must never end in 'ไม่เข้าใจ' — route to list_products."""
    if not text or looks_like_transfer_intent(text) or looks_like_cod_intent(text):
        return False
    if _CLOSING_INTENT.search(text):
        return False
    return (
        _looks_like_broad_catalog_query(text)
        or _looks_like_recommend_query(text)
        or _looks_like_budget_browse_query(text)
    )


def _looks_like_confused_reply(text: str) -> bool:
    low = (text or "").lower()
    return any(marker in low for marker in _CONFUSED_REPLY_MARKERS)
