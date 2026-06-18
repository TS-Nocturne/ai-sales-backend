"""Detect customer requests to speak with a human and standard handoff replies."""

from __future__ import annotations

import re

from langchain_core.messages import HumanMessage

HANDOFF_REPLY = "ระบบกำลังเรียกแอดมินให้ค่ะ รบกวนรอสักครู่นะคะ 👩‍💻"
HANDOFF_REASON_DEFAULT = "ลูกค้าขอคุยกับเจ้าหน้าที่"

_HANDOFF_KEYWORDS = (
    "พนักงาน",
    "แอดมิน",
    "admin",
    "ติดต่อคน",
    "คุยกับคน",
    "ขอคุยกับคน",
    "เจ้าหน้าที่",
    "เรียกคน",
    "คนจริง",
    "มนุษย์",
)

_HANDOFF_PATTERNS = (
    re.compile(r"เรียก\s*(?:พนักงาน|แอดมิน|เจ้าหน้าที่|คน)", re.IGNORECASE),
    re.compile(
        r"(?:ขอ|อยาก|ต้องการ).{0,16}(?:คุย|พูด|ติดต่อ).{0,16}(?:คน|พนักงาน|แอดมิน|เจ้าหน้าที่)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:แอดมิน|พนักงาน|เจ้าหน้าที่).{0,10}(?:อยู่|ว่าง|online|ออนไลน์|ไหม)",
        re.IGNORECASE,
    ),
)


def _normalize_message_text(content) -> str:
    if not content:
        return ""
    if isinstance(content, list):
        parts = [
            item.get("text", str(item))
            for item in content
            if isinstance(item, dict) and "text" in item
        ]
        return " ".join(parts)
    return str(content)


def last_customer_text(messages: list) -> str:
    """Return the most recent customer message as plain text."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) and msg.content:
            return _normalize_message_text(msg.content).strip()
    return ""


def looks_like_handoff_intent(text: str) -> bool:
    """True when the customer wants to talk to staff instead of the bot."""
    raw = (text or "").strip()
    if not raw:
        return False

    for pattern in _HANDOFF_PATTERNS:
        if pattern.search(raw):
            return True

    compact = re.sub(r"\s+", "", raw.lower())
    return any(keyword.replace(" ", "") in compact for keyword in _HANDOFF_KEYWORDS)


def handoff_reason_from_text(text: str) -> str:
    """Short reason for dashboard staff queue."""
    snippet = (text or "").strip()
    if not snippet:
        return HANDOFF_REASON_DEFAULT
    if len(snippet) > 120:
        snippet = snippet[:117] + "..."
    return snippet
