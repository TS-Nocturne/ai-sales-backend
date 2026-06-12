"""
Service layer that wraps the compiled LangGraph sales agent for HTTP access.

Responsibilities:
- Hold a single compiled graph (and its SQLite connection) for the process.
- Translate a stateless JSON request from Next.js into a graph invocation
  on the correct ``thread_id`` (one thread == one conversation).
- Extract the latest customer-facing reply and a snapshot of the agent state
  (lead score, pipeline stage, pending discount approval).
- Drive the Human-in-the-Loop (HITL) resume flow when a manager approves or
  rejects a pending discount.

This module is the only place that knows about LangGraph internals; the
FastAPI layer in ``server.py`` only deals with plain dictionaries.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from psycopg_pool import ConnectionPool

from langchain_core.messages import AIMessage, HumanMessage

from ai_sales.api.retry import invoke_with_retry, is_retryable_error
from ai_sales.channels import line_delivery
from ai_sales.consts import HUMAN_APPROVAL
from ai_sales.db_config import get_database_url, pool_max_size, warn_if_not_neon_pooler
from ai_sales.graph.builder import build_graph
from ai_sales.messages.approval_reply import format_discount_decision_reply
from ai_sales.prefetch import build_catalog_prefetch
from ai_sales.state import initial_state

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graph singleton (built lazily, reused across requests)
# ---------------------------------------------------------------------------
_graph = None
_pool = None
# PostgresSaver tolerates cross-thread access, but graph.invoke mutates the
# checkpoint; serialise invocations per thread to avoid interleaved writes.
_thread_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _thread_lock(thread_id: str) -> threading.Lock:
    """Return a lock scoped to one conversation thread."""
    with _locks_guard:
        lock = _thread_locks.get(thread_id)
        if lock is None:
            lock = threading.Lock()
            _thread_locks[thread_id] = lock
        return lock


def get_graph():
    """Return the process-wide compiled graph, building it on first use."""
    global _graph, _pool
    if _graph is None:
        db_url = get_database_url()
        warn_if_not_neon_pooler(db_url)
        max_size = pool_max_size()
        _pool = ConnectionPool(
            conninfo=db_url,
            min_size=1,
            max_size=max_size,
            kwargs={"autocommit": True},
        )
        logger.info("LangGraph Postgres pool max_size=%s (Neon pooler URL)", max_size)
        _graph = build_graph(_pool)
    return _graph


def close() -> None:
    """Close the underlying PostgreSQL connection pool (call on shutdown)."""
    global _graph, _pool
    if _pool is not None:
        _pool.close()
    _graph = None
    _pool = None


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------
_INTERNAL_PREFIXES = (
    "[Lead Scoring Complete]",
    "[การให้คะแนนลีดเสร็จสมบูรณ์]",
    "[APPROVED]",
    "[REJECTED]",
    "[อนุมัติ]",
    "[ปฏิเสธ]",
)


def _normalize_content(content) -> str:
    """Convert message content to a plain string (handles Gemini list parts)."""
    if not content:
        return ""
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            else:
                parts.append(str(item))
        return " ".join(parts)
    return str(content)


def _is_internal_message(msg: AIMessage) -> bool:
    """True for scoring/approval/tool-call messages that are not for the customer."""
    text = _normalize_content(msg.content).strip()
    if not text:
        return bool(getattr(msg, "tool_calls", None))
    return any(text.startswith(prefix) for prefix in _INTERNAL_PREFIXES)


def _strip_internal_block(text: str) -> str:
    """Cut off any internal bookkeeping block accidentally embedded in a reply.

    The lead-scoring/approval summaries live in their own messages, but if the
    model ever appends one inside a customer reply (e.g. "...ค่ะ[การให้คะแนน...]"),
    truncate at the first internal marker so the customer never sees it.
    """
    cut = len(text)
    for prefix in _INTERNAL_PREFIXES:
        idx = text.find(prefix)
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut].strip()


def _humanize_reply(text: str) -> str:
    """Turn markdown-ish model output into plain, human-looking chat text.

    Channels like LINE do not render markdown, so ``**bold**`` and ``*`` bullets
    leak through as raw symbols. Strip the markup and use friendly bullets so the
    reply reads like a person typed it.
    """
    out = text

    # **bold** / __bold__  ->  bold
    out = re.sub(r"\*\*(.+?)\*\*", r"\1", out, flags=re.DOTALL)
    out = re.sub(r"__(.+?)__", r"\1", out, flags=re.DOTALL)

    # Headings (#, ##, ...) at line start -> plain text
    out = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", out)

    cleaned_lines = []
    for line in out.split("\n"):
        stripped = line.strip()
        # Bullet markers (*, -, +) with optional indentation -> "• "
        m = re.match(r"^[*\-+]\s+(.*)$", stripped)
        if m:
            cleaned_lines.append(f"• {m.group(1).strip()}")
            continue
        # Leftover single * used for emphasis -> drop the symbol
        stripped = re.sub(r"(?<!\w)\*(?!\s)(.+?)(?<!\s)\*(?!\w)", r"\1", stripped)
        cleaned_lines.append(stripped)

    out = "\n".join(cleaned_lines)
    # Collapse 3+ blank lines into a single blank line.
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def _latest_reply(messages: list, since: int = 0) -> str:
    """Return the latest customer-facing agent reply added after ``since``."""
    new_messages = messages[since:]
    for msg in reversed(new_messages):
        if not isinstance(msg, AIMessage):
            continue
        if _is_internal_message(msg):
            continue
        text = _strip_internal_block(_normalize_content(msg.content).strip())
        if text:
            return _humanize_reply(text)
    return ""


def _latest_reply_after_last_human(messages: list) -> str:
    """Return the customer-facing reply for the current turn.

    Uses the index of the latest HumanMessage in the *post-invoke* message list.
    This stays correct even when ``context_summarizer_node`` removes older
    messages mid-turn (using a pre-invoke message count as ``since`` would
    slice past the end and yield an empty reply).
    """
    last_human_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            last_human_idx = i
            break
    return _latest_reply(messages, since=last_human_idx + 1)


def _messages_added_since(before: list, after: list) -> list:
    """Return messages present in ``after`` but not in ``before`` (by id)."""
    before_ids = {m.id for m in before if getattr(m, "id", None)}
    return [m for m in after if getattr(m, "id", None) not in before_ids]


# ---------------------------------------------------------------------------
# Thread / state helpers
# ---------------------------------------------------------------------------
def _thread(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _message_count(graph, thread: dict) -> int:
    state = graph.get_state(thread)
    return len(state.values.get("messages", []))


def _is_new_thread(graph, thread: dict) -> bool:
    state = graph.get_state(thread)
    return not state.values.get("messages")


def _snapshot(graph, thread: dict) -> dict:
    """Return a JSON-serialisable view of the agent state for this thread."""
    state = graph.get_state(thread)
    values = state.values
    pending = values.get("pending_discount_approval") or {}
    requires_approval = bool(state.next and HUMAN_APPROVAL in state.next)
    payment_qr = values.get("payment_qr") or {}
    return {
        "lead_score": values.get("lead_score", 0),
        "pipeline_stage": values.get("pipeline_stage", "new"),
        "requires_approval": requires_approval,
        "pending_discount_approval": pending or None,
        "next_nodes": list(state.next) if state.next else [],
        "shipping_info": values.get("shipping_info") or None,
        "order_ready": bool(values.get("order_ready", False)),
        "payment_qr": payment_qr
        if payment_qr.get("image_base64")
        or payment_qr.get("use_static")
        or payment_qr.get("static_url")
        else None,
        "pending_overpay": values.get("pending_overpay") or None,
        "overpay_resolution": values.get("overpay_resolution") or None,
        "overpay_credit_amount": values.get("overpay_credit_amount") or 0,
        "awaiting_refund_approval": bool(values.get("awaiting_refund_approval")),
    }


# ---------------------------------------------------------------------------
# Public service functions
# ---------------------------------------------------------------------------
def send_message(
    thread_id: str, message: str, payment_context: dict | None = None
) -> dict:
    """Send a customer message into the graph and return the agent's reply.

    Args:
        thread_id: Conversation identifier (one thread per conversation).
        message: The customer's message text.

    Returns:
        A dict with ``thread_id``, ``reply`` and the state snapshot fields.
    """
    graph = get_graph()
    thread = _thread(thread_id)

    payload: dict = {
        "messages": [HumanMessage(content=message)],
        "tool_iterations": 0,
        # order_ready is a one-shot signal: reset it each turn so it is only
        # True on the exact turn save_shipping_info ran. This prevents the
        # Next.js side from re-creating an order on every subsequent message.
        "order_ready": False,
        # payment_qr is one-shot: reset each turn so LINE only gets a fresh QR once.
        "payment_qr": {},
    }
    if payment_context:
        payload["pending_overpay"] = payment_context

    with _thread_lock(thread_id):
        summary = ""
        if _is_new_thread(graph, thread):
            payload.update(initial_state())
        else:
            existing = graph.get_state(thread).values
            summary = (existing.get("conversation_summary") or "").strip()
        payload["catalog_prefetch"] = build_catalog_prefetch(message, summary)

        def _run() -> None:
            graph.invoke(payload, thread)

        invoke_with_retry(_run)
        after_messages = graph.get_state(thread).values.get("messages", [])
        reply = _latest_reply_after_last_human(after_messages)
        if not reply:
            # Agent finished the turn without a customer-facing AIMessage (e.g. only
            # tool calls + lead-scoring bookkeeping). Give a helpful prompt instead
            # of leaving the channel layer to show a generic error.
            reply = (
                "ขออภัยค่ะ ไม่แน่ใจว่าหมายถึงสินค้าหรือเรื่องใด "
                "รบกวนพิมพ์ใหม่อีกครั้งได้ไหมคะ 🙏 "
                "เช่น แจ้งชื่อสินค้า รุ่นมือถือ หรืองบประมาณที่ต้องการ "
                "เดี๋ยวทางร้านช่วยแนะนำให้ค่ะ"
            )
        snapshot = _snapshot(graph, thread)

    payment_qr = snapshot.get("payment_qr")
    reply = line_delivery.embed_line_qr_tag(reply, payment_qr)

    return {"thread_id": thread_id, "reply": reply, **snapshot}


def resume_with_approval(thread_id: str, approved: bool) -> dict:
    """Resume an interrupted graph after a manager approves/rejects a discount.

    Args:
        thread_id: Conversation identifier that is paused at human approval.
        approved: True to approve the pending discount, False to reject.

    Returns:
        A dict with ``thread_id``, ``reply``, ``resumed`` flag and snapshot.
        ``resumed`` is False when the thread was not actually awaiting approval.
    """
    graph = get_graph()
    thread = _thread(thread_id)

    with _thread_lock(thread_id):
        state = graph.get_state(thread)
        awaiting = bool(state.next and HUMAN_APPROVAL in state.next)
        if not awaiting:
            return {
                "thread_id": thread_id,
                "reply": "",
                "resumed": False,
                **_snapshot(graph, thread),
            }

        before_messages = list(state.values.get("messages", []))
        pending_discount = dict(state.values.get("pending_discount_approval") or {})
        graph.update_state(thread, {"discount_approved": approved})

        def _run() -> None:
            graph.invoke(None, thread)

        invoke_with_retry(_run)
        after_messages = graph.get_state(thread).values.get("messages", [])
        new_messages = _messages_added_since(before_messages, after_messages)
        reply = _latest_reply(new_messages)
        if not reply and pending_discount:
            reply = format_discount_decision_reply(pending_discount, approved)
        elif not reply:
            reply = format_discount_decision_reply({}, approved)
        snapshot = _snapshot(graph, thread)

    return {"thread_id": thread_id, "reply": reply, "resumed": True, **snapshot}


def get_state(thread_id: str) -> dict:
    """Return the current state snapshot for a conversation thread."""
    graph = get_graph()
    thread = _thread(thread_id)
    return {"thread_id": thread_id, **_snapshot(graph, thread)}
