"""
Node functions for the AI Sales Agent graph.

Follows AGENTS.md guidelines:
- Every node function returns a dict updating only the specific state keys it modified.
- Use ToolNode(tools) from langgraph.prebuilt to handle tool execution automatically.
"""

import json
import logging
import re
import uuid

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.prebuilt import ToolNode

from ai_sales.config.llm import (
    get_llm,
    get_llm_with_tools,
    get_llm_with_tools_forced,
)
from ai_sales.config.prompts import (
    LEAD_SCORING_PROMPT,
    SALES_AGENT_SYSTEM_PROMPT,
    SUMMARY_PROMPT,
)
from ai_sales.consts import (
    MANAGER_APPROVAL_DISCOUNT_THRESHOLD,
    MAX_DISCOUNT_PERCENT,
    STAGE_CLOSED_WON,
    STAGE_NEGOTIATION,
    SUMMARY_KEEP_RECENT,
    SUMMARY_TRIGGER_COUNT,
)
from ai_sales.messages.approval_reply import format_discount_decision_reply
from ai_sales.state import SalesState
from ai_sales.tools.catalog import find_product_price
from ai_sales.tools.order_total import (
    looks_like_cod_intent,
    looks_like_transfer_intent,
    should_auto_generate_qr,
)
from ai_sales.tools.sales_tools import all_tools, _is_pure_catalog_browse, transfer_payment_qr_update

logger = logging.getLogger(__name__)

_MAX_TOOL_SUMMARY_CHARS = 900

# ToolNode for automatic tool execution (AGENTS.md: Use ToolNode(tools))
_base_tool_executor = ToolNode(all_tools)


def tool_executor_node(state: SalesState) -> dict | list:
    """Execute tools and increment the ReAct iteration counter."""
    result = _base_tool_executor.invoke(state)
    next_iterations = state.get("tool_iterations", 0) + 1

    # Command-returning tools (generate_promptpay_qr, save_shipping_info) make
    # ToolNode return a list of Command/message patches — not a dict.
    if isinstance(result, list):
        return [*result, {"tool_iterations": next_iterations}]

    return {
        **result,
        "tool_iterations": next_iterations,
    }


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
    has_browse = any(
        w in compact for w in ("ซื้ออะไร", "ได้บ้าง", "แนะนำ", "อะไรได้")
    )
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


def _browse_already_satisfied(messages: list) -> bool:
    """True when catalog data was already fetched for the latest customer turn."""
    last_human_idx = None
    for idx in range(len(messages) - 1, -1, -1):
        if isinstance(messages[idx], HumanMessage):
            last_human_idx = idx
            break
    if last_human_idx is None:
        return False
    for msg in messages[last_human_idx:]:
        if isinstance(msg, ToolMessage):
            text = _normalize_message_text(msg.content)
            if "[Catalog]" in text or "[Vector Search]" in text or "[Catalog Fallback]" in text or "[หมวดหมู่สินค้า]" in text:
                return True
    return False


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


def _browse_display_mode(customer_text: str) -> str:
    """Category overview for pure 'what do you sell'; featured samples otherwise."""
    if _is_pure_catalog_browse(customer_text):
        return "categories"
    return "featured"


def _browse_catalog_tool_call(customer_text: str) -> AIMessage:
    """Emit a list_products tool call for vague browse / recommend / budget turns."""
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "list_products",
                "args": {
                    "category": "",
                    "keyword": "",
                    "min_price": 0,
                    "max_price": _extract_budget_ceiling(customer_text),
                    "sort_by_price": "",
                    "limit": 3,
                    "display_mode": _browse_display_mode(customer_text),
                },
                "id": f"call_{uuid.uuid4().hex[:12]}",
                "type": "tool_call",
            }
        ],
    )


def _closing_context_hint(messages: list) -> str:
    """Inject a one-turn instruction when the customer signals purchase interest."""
    last = _last_customer_text(messages)
    if not last or not _CLOSING_INTENT.search(last):
        return ""
    for msg in reversed(messages[:-1] if len(messages) > 1 else messages):
        if not isinstance(msg, AIMessage) or getattr(msg, "tool_calls", None):
            continue
        text = _normalize_message_text(msg.content)
        if not text or any(text.startswith(p) for p in _INTERNAL_MESSAGE_PREFIXES):
            continue
        if "บาท" in text or any(
            kw in text for kw in ("เคส", "ฟิล์ม", "สายชาร์จ", "หูฟัง", "iPhone", "iPad")
        ):
            return (
                "คำสั่งเฉพาะเทิร์นนี้: ลูกค้าแสดงความสนใจหลังคุณเสนอสินค้าแล้ว "
                "ห้ามถามรุ่นมือถือใหม่หรือเริ่มคุยใหม่ — ให้ถามว่าต้องการรับสินค้าชิ้นไหน "
                "จากตัวเลือกที่เพิ่งเสนอ (ระบุชื่อและราคา) พร้อมแจ้งวิธีชำระเงิน "
                "(โอน PromptPay / เก็บปลายทาง COD)"
            )
    return ""


def _looks_like_confused_reply(text: str) -> bool:
    low = (text or "").lower()
    return any(marker in low for marker in _CONFUSED_REPLY_MARKERS)


# Internal bookkeeping messages (lead scoring / approval) that are written into
# the conversation for staff/manager visibility but must NEVER be shown to the
# customer — nor fed back into the sales agent (or it learns to imitate them).
_INTERNAL_MESSAGE_PREFIXES = (
    "[Lead Scoring Complete]",
    "[การให้คะแนนลีดเสร็จสมบูรณ์]",
    "[APPROVED]",
    "[REJECTED]",
    "[อนุมัติ]",
    "[ปฏิเสธ]",
)


def _strip_internal_messages(messages: list) -> list:
    """Remove internal scoring/approval AIMessages from the agent's context.

    Keeps everything needed for the ReAct loop (human messages, tool calls and
    tool results) but drops the bookkeeping summaries so the LLM does not copy
    their format into a customer-facing reply.
    """
    cleaned = []
    for msg in messages:
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            text = _normalize_message_text(msg.content).strip()
            if any(text.startswith(prefix) for prefix in _INTERNAL_MESSAGE_PREFIXES):
                continue
        cleaned.append(msg)
    return cleaned


def _is_turn_boundary(msg) -> bool:
    """A safe place to cut history: the start of a customer turn.

    Cutting on a HumanMessage guarantees we never split an AIMessage(tool_calls)
    from its ToolMessage result (which would break Gemini's tool protocol).
    """
    return isinstance(msg, HumanMessage)


def _render_for_summary(messages: list) -> str:
    """Flatten messages into a plain transcript for the summarizer LLM."""
    lines = []
    for m in messages:
        text = _normalize_message_text(getattr(m, "content", "")).strip()
        if not text:
            continue
        if isinstance(m, HumanMessage):
            lines.append(f"ลูกค้า: {text}")
        elif isinstance(m, AIMessage) and not getattr(m, "tool_calls", None):
            lines.append(f"ร้าน: {text}")
        elif isinstance(m, ToolMessage):
            if len(text) > _MAX_TOOL_SUMMARY_CHARS:
                text = text[:_MAX_TOOL_SUMMARY_CHARS] + "…"
            lines.append(f"ข้อมูลจากระบบ: {text}")
    return "\n".join(lines)


def _memory_context_for_agent(state: SalesState) -> str:
    """Inject rolling summary + durable scalar state so the agent keeps context."""
    blocks: list[str] = []
    summary = (state.get("conversation_summary") or "").strip()
    if summary:
        blocks.append(
            "สรุปบทสนทนาก่อนหน้า (ใช้ต่อเนื่อง ห้ามทวนซ้ำให้ลูกค้าฟัง):\n" + summary
        )

    shipping = state.get("shipping_info") or {}
    if isinstance(shipping, dict) and any(shipping.values()):
        try:
            shipping_text = json.dumps(shipping, ensure_ascii=False)
        except TypeError:
            shipping_text = str(shipping)
        blocks.append(f"ข้อมูลจัดส่งที่บันทึกในระบบแล้ว: {shipping_text}")

    pending = state.get("pending_discount_approval") or {}
    if isinstance(pending, dict) and pending:
        blocks.append(f"ส่วนลดที่เสนอ/รออนุมัติ: {json.dumps(pending, ensure_ascii=False)}")

    score = state.get("lead_score")
    stage = state.get("pipeline_stage")
    if score or stage:
        blocks.append(f"สถานะลีดปัจจุบัน: คะแนน {score}, ขั้น {stage}")

    return "\n\n".join(blocks)


def context_summarizer_node(state: SalesState) -> dict:
    """Compact long histories into a rolling summary to save tokens/memory.

    Runs at the start of every turn. When the transcript grows beyond
    SUMMARY_TRIGGER_COUNT messages, everything older than the most recent
    SUMMARY_KEEP_RECENT messages (snapped to a turn boundary) is summarized by a
    small model and removed from `messages`; the summary is stored in
    `conversation_summary` and injected into the agent's system prompt.
    """
    messages = state["messages"]
    if len(messages) <= SUMMARY_TRIGGER_COUNT:
        return {}

    # Choose a cut index near the end, then snap left to a turn boundary so we
    # keep complete turns and never orphan a tool call / tool result pair.
    cut = max(0, len(messages) - SUMMARY_KEEP_RECENT)
    while cut > 0 and not _is_turn_boundary(messages[cut]):
        cut -= 1

    older = messages[:cut]
    if not older:
        return {}

    transcript = _render_for_summary(older)
    if not transcript.strip():
        # Tool-only turns with no summarizable text — keep messages rather than
        # deleting context silently (a common cause of "AI forgot" reports).
        logger.debug(
            "context_summarizer: skip trim — nothing summarizable in %s messages",
            len(older),
        )
        return {}

    previous_summary = (state.get("conversation_summary") or "").strip()
    human_payload = transcript
    if previous_summary:
        human_payload = (
            f"สรุปเดิม:\n{previous_summary}\n\n"
            f"บทสนทนาเพิ่มเติมที่ต้องรวมเข้าไป:\n{transcript}"
        )

    llm = get_llm(temperature=0.1)
    try:
        result = llm.invoke(
            [
                SystemMessage(content=SUMMARY_PROMPT),
                HumanMessage(content=human_payload),
            ]
        )
        new_summary = _normalize_message_text(result.content).strip()
    except Exception as exc:
        # If summarization fails, do not lose context: keep messages as-is.
        logger.warning("context_summarizer LLM failed: %s", exc)
        return {}

    if not new_summary:
        return {}

    removals = [RemoveMessage(id=m.id) for m in older if getattr(m, "id", None)]
    return {
        "conversation_summary": new_summary,
        "messages": removals,
    }


def sales_agent_node(state: SalesState) -> dict:
    """Main sales agent node. Invokes the LLM with tools to handle customer interaction.

    Returns only the updated messages key (AGENTS.md guideline).
    """
    last_customer = _last_customer_text(state["messages"])

    # ลูกค้าเลือกโอนเงิน + มียอดในบทสนทนาแล้ว → สร้าง QR ทันที (ไม่พึ่ง LLM)
    if should_auto_generate_qr(state["messages"]):
        return transfer_payment_qr_update(state)

    # คำถามกว้างๆ / แนะนำ / งบ+ซื้ออะไรได้ → list_products ทันที (ไม่รอ LLM)
    if _should_auto_browse_catalog(last_customer) and not _browse_already_satisfied(
        state["messages"]
    ):
        return {"messages": [_browse_catalog_tool_call(last_customer)]}

    # Strip internal bookkeeping (scoring/approval) so the model never sees —
    # and therefore never imitates — those blocks in its reply.
    messages = _strip_internal_messages(state["messages"])

    # Prepend system prompt — augmented with durable memory (summary + state)
    # so the agent keeps context even when older messages were trimmed.
    system_text = SALES_AGENT_SYSTEM_PROMPT
    memory_block = _memory_context_for_agent(state)
    if memory_block:
        system_text = f"{SALES_AGENT_SYSTEM_PROMPT}\n\n{memory_block}"
    closing_hint = _closing_context_hint(state["messages"])
    if closing_hint:
        system_text = f"{system_text}\n\n{closing_hint}"
    prefetch = (state.get("catalog_prefetch") or "").strip()
    if prefetch:
        system_text = f"{system_text}\n\n{prefetch}"
    system_msg = SystemMessage(content=system_text)
    messages_with_system = [system_msg] + messages

    # Invoke LLM with tools bound (AGENTS.md: Run llm.bind_tools before invoking)
    llm_with_tools = get_llm_with_tools(all_tools)
    response = llm_with_tools.invoke(messages_with_system)

    # Safety net: Gemini sometimes *announces* a search ("ขอค้นหาสักครู่นะคะ")
    # without emitting a tool call. That message has no tool_calls, so routing
    # would send it straight to lead scoring → END, and the customer only ever
    # sees the filler. If we detect that, re-invoke once forcing a tool call so a
    # real search/calculation runs and the graph completes the turn properly.
    has_tool_calls = bool(getattr(response, "tool_calls", None))
    if not has_tool_calls:
        text = _normalize_message_text(response.content)
        last_customer = _last_customer_text(state["messages"])
        if last_customer and looks_like_transfer_intent(last_customer):
            if should_auto_generate_qr(state["messages"]):
                return transfer_payment_qr_update(state)
        needs_forced_tool = (
            (text and _looks_like_search_filler(text))
            or (
                text
                and _looks_like_confused_reply(text)
                and _should_auto_browse_catalog(last_customer)
                and not _browse_already_satisfied(state["messages"])
            )
            or (
                _should_auto_browse_catalog(last_customer)
                and not _browse_already_satisfied(state["messages"])
            )
        )
        if needs_forced_tool:
            if _should_auto_browse_catalog(last_customer):
                response = _browse_catalog_tool_call(last_customer)
            else:
                forced_llm = get_llm_with_tools_forced(all_tools)
                response = forced_llm.invoke(messages_with_system)
            if not getattr(response, "tool_calls", None):
                if last_customer and looks_like_transfer_intent(last_customer):
                    if should_auto_generate_qr(state["messages"]):
                        return transfer_payment_qr_update(state)
                if _should_auto_browse_catalog(last_customer):
                    response = _browse_catalog_tool_call(last_customer)
        elif not text.strip():
            if _should_auto_browse_catalog(last_customer) and not _browse_already_satisfied(
                state["messages"]
            ):
                response = _browse_catalog_tool_call(last_customer)
            else:
                # Never end the turn silently — an empty reply makes the channel
                # show a generic error. Ask the customer to clarify instead.
                response = AIMessage(
                    content=(
                        "ขออภัยค่ะ ไม่แน่ใจว่าหมายถึงสินค้าหรือเรื่องใด "
                        "รบกวนพิมพ์ใหม่อีกครั้งได้ไหมคะ 🙏 "
                        "เช่น แจ้งชื่อสินค้า รุ่นมือถือ หรืองบประมาณที่ต้องการ "
                        "เดี๋ยวทางร้านช่วยแนะนำให้ค่ะ"
                    )
                )

    return {"messages": [response]}


def lead_scorer_node(state: SalesState) -> dict:
    """Evaluates the conversation to compute a lead score, update pipeline stage,
    and determine if a special discount approval is needed.

    Returns lead_score, pipeline_stage, pending_discount_approval, and a summary message.
    """
    messages = state["messages"]

    # Build a scoring request with the full conversation context
    conversation_lines = []
    for m in messages:
        if isinstance(m, HumanMessage) and m.content:
            conversation_lines.append(f"Customer: {m.content}")
        elif isinstance(m, AIMessage) and m.content:
            # --- ป้องกัน Gemini ส่งค่าเป็น List เข้ามาในลอจิกวิเคราะห์ ---
            raw_content = m.content
            if isinstance(raw_content, list):
                text_parts = [
                    item["text"]
                    for item in raw_content
                    if isinstance(item, dict) and "text" in item
                ]
                text_content = " ".join(text_parts)
            else:
                text_content = str(raw_content)
            conversation_lines.append(f"Agent: {text_content}")

    scoring_messages = [
        SystemMessage(content=LEAD_SCORING_PROMPT),
        HumanMessage(
            content=(
                "Here is the full sales conversation to analyze:\n\n"
                + "\n".join(conversation_lines)
            )
        ),
    ]

    # Use the base LLM (no tools) for scoring
    llm = get_llm()
    response = llm.invoke(scoring_messages)

    # --- ป้องกัน Gemini คืนค่า JSON ออกมาเป็น List ---
    raw_resp = response.content
    if isinstance(raw_resp, list):
        resp_parts = [
            item["text"]
            for item in raw_resp
            if isinstance(item, dict) and "text" in item
        ]
        resp_str = " ".join(resp_parts)
    else:
        resp_str = str(raw_resp)

    # Parse the JSON response (robust extraction)
    scoring_data = _parse_scoring_response(resp_str)

    # Build the state update
    lead_score = scoring_data.get("lead_score", 50)
    pipeline_stage = scoring_data.get("pipeline_stage", "qualified")
    summary = scoring_data.get("summary", "N/A")

    result = {
        "lead_score": lead_score,
        "pipeline_stage": pipeline_stage,
        "messages": [
            AIMessage(
                content=(
                    f"[การให้คะแนนลีดเสร็จสมบูรณ์]\n"
                    f"  คะแนน: {lead_score}/100\n"
                    f"  สถานะ: {pipeline_stage}\n"
                    f"  การวิเคราะห์: {summary}"
                )
            )
        ],
    }

    # Enforce HITL in code — never trust needs_discount_approval from LLM alone
    pending = _resolve_pending_discount(messages, scoring_data)
    result["pending_discount_approval"] = pending or {}

    return result


def human_approval_node(state: SalesState) -> dict:
    """Human-in-the-loop approval node.

    This node is interrupted BEFORE execution (interrupt_before).
    A human manager reviews the pending discount and updates the state
    via graph.update_state() with discount_approved = True/False.

    When the graph resumes, this node reads the decision and generates
    a confirmation message.
    """
    discount_info = state.get("pending_discount_approval", {})
    approved = state.get("discount_approved", False)

    product = discount_info.get("product", "N/A")
    discount_pct = discount_info.get("discount_pct", 0)

    if approved:
        msg = (
            f"[อนุมัติ] ผู้จัดการอนุมัติส่วนลดแล้ว!\n"
            f"  สินค้า: {product}\n"
            f"  ส่วนลด: {discount_pct}%\n"
            f"  กำลังดำเนินการกับข้อเสนอส่วนลด..."
        )
    else:
        msg = (
            f"[ปฏิเสธ] ผู้จัดการปฏิเสธคำขอส่วนลด\n"
            f"  สินค้า: {product}\n"
            f"  ร้องขอ: {discount_pct}%\n"
            f"  จะคงราคามาตรฐานไว้"
        )

    # IMPORTANT: do NOT clear pending_discount_approval here. The next node
    # (post_approval_node) still needs product/price/discount to craft the
    # customer message. Clearing it now makes the reply fall back to defaults
    # ("the product", $0.00, 0%). post_approval_node clears it after use.
    return {
        "messages": [AIMessage(content=msg)],
    }


def _normalize_message_text(content) -> str:
    """Convert message content to string (handles Gemini list responses)."""
    if not content:
        return ""
    if isinstance(content, list):
        text_parts = [
            item["text"]
            for item in content
            if isinstance(item, dict) and "text" in item
        ]
        return " ".join(text_parts)
    return str(content)


def _last_customer_text(messages: list) -> str:
    """Return the most recent customer message as plain text."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) and msg.content:
            return _normalize_message_text(msg.content).strip()
    return ""


def post_approval_node(state: SalesState) -> dict:
    """Generates the final customer-facing response after the manager's decision.

    Uses a fixed template so approve/reject replies stay warm, clear, and
    consistent on every channel (LINE + dashboard chat).
    """
    approved = state.get("discount_approved", False)
    discount_info = state.get("pending_discount_approval", {}) or {}
    reply_text = format_discount_decision_reply(discount_info, approved)

    if approved:
        return {
            "messages": [AIMessage(content=reply_text)],
            "pipeline_stage": STAGE_CLOSED_WON,
            "pending_discount_approval": {},
        }

    return {
        "messages": [AIMessage(content=reply_text)],
        "pipeline_stage": STAGE_NEGOTIATION,
        "pending_discount_approval": {},
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_DISCOUNT_PATTERNS = (
    re.compile(
        r"(?:ส่วนลด|ลดราคา|ลด|discount|off)\s*(\d+(?:\.\d+)?)\s*%",
        re.IGNORECASE,
    ),
    re.compile(r"Discount:\s*(\d+(?:\.\d+)?)\s*%", re.IGNORECASE),
    re.compile(r"(\d+(?:\.\d+)?)\s*%\s*(?:off|ส่วนลด)", re.IGNORECASE),
)


def _extract_max_discount_from_conversation(messages: list) -> float:
    """Scan all messages for the highest discount % mentioned (code-level guard)."""
    max_pct = 0.0

    for msg in messages:
        if not hasattr(msg, "content") or not msg.content:
            continue
        text = _normalize_message_text(msg.content)
        if not text:
            continue

        if isinstance(msg, AIMessage) and any(
            text.startswith(prefix)
            for prefix in (
                "[Lead Scoring Complete]",
                "[การให้คะแนนลีดเสร็จสมบูรณ์]",
                "[APPROVED]",
                "[REJECTED]",
                "[อนุมัติ]",
                "[ปฏิเสธ]",
            )
        ):
            continue

        for pattern in _DISCOUNT_PATTERNS:
            for match in pattern.finditer(text):
                max_pct = max(max_pct, float(match.group(1)))

    return min(max_pct, MAX_DISCOUNT_PERCENT)


def _resolve_pending_discount(messages: list, scoring_data: dict) -> dict | None:
    """Determine if manager approval is required based on discount % in code.

    Uses the higher of scorer-reported % and conversation-extracted % so
    prompt-injection cannot bypass HITL by lying in needs_discount_approval.
    """
    scorer_pct = float(scoring_data.get("discount_percent", 0) or 0)
    conversation_pct = _extract_max_discount_from_conversation(messages)
    discount_pct = min(
        max(scorer_pct, conversation_pct),
        MAX_DISCOUNT_PERCENT,
    )

    if discount_pct <= MANAGER_APPROVAL_DISCOUNT_THRESHOLD:
        return None

    discount_product = (scoring_data.get("discount_product") or "").strip() or "Unknown"
    reason = (scoring_data.get("discount_reason") or "").strip()
    if not reason:
        reason = (
            f"ตรวจพบส่วนลด {discount_pct}% ในบทสนทนา "
            f"(เกินเกณฑ์ {MANAGER_APPROVAL_DISCOUNT_THRESHOLD}% ที่อนุมัติอัตโนมัติได้)"
        )

    return {
        "product": discount_product,
        "discount_pct": discount_pct,
        "reason": reason,
        "original_price": find_product_price(discount_product),
    }


def _parse_scoring_response(content: str) -> dict:
    """Robustly parse the lead scoring JSON from the LLM response.

    Handles cases where the LLM wraps the JSON in markdown code fences
    or includes extra text around it.
    """
    content = content.strip()

    # Strategy 1: Try direct parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Strip markdown code fences (```json ... ``` or ``` ... ```)
    fenced = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", content, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Strategy 3: Find first { ... } block
    brace_match = re.search(r"\{.*\}", content, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    # Fallback: return safe defaults
    return {
        "lead_score": 50,
        "pipeline_stage": "qualified",
        "needs_discount_approval": False,
        "discount_product": "",
        "discount_percent": 0,
        "discount_reason": "",
        "summary": "Unable to parse scoring response. Using default values.",
    }
