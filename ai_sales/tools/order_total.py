"""Derive payable order total from conversation history."""

from __future__ import annotations

import re

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

_DISCOUNT_BLOCK = re.compile(
    r"\[Discount Calculation\]\s*(.+?):\s*.*?Final Price:\s*([\d.]+)\s*THB",
    re.DOTALL | re.IGNORECASE,
)
_TOOL_PRICE_THB = re.compile(r"Price:\s*([\d.]+)\s*THB", re.IGNORECASE)
_TOOL_PRICE_BAHT = re.compile(r"\|\s*ราคา\s*([\d,.]+)\s*บาท")
_AI_FINAL_BAHT = re.compile(
    r"(?:เหลือ|ลด[^.\n]{0,40}?เหลือ|ราคา(?:พิเศษ)?(?:หลัง(?:หัก)?ส่วนลด|สุทธิ|รวม)?(?:เป็น)?|ยอด(?:รวม|ชำระ|โอน)?(?:เป็น)?)\s*([\d,.]+)\s*บาท",
    re.IGNORECASE,
)
_APPROVAL_FINAL_BAHT = re.compile(
    r"ราคาพิเศษหลังหักส่วนลด:\s*([\d,.]+)\s*บาท",
    re.IGNORECASE,
)
_AI_ANY_BAHT = re.compile(r"([\d,.]+)\s*บาท", re.IGNORECASE)
_TRANSFER_INTENT = re.compile(
    r"(โอนเงิน|โอนมา|โอนให้|จะโอน|ชำระ.*โอน|transfer)",
    re.IGNORECASE,
)
_TRANSFER_ALREADY_DONE = re.compile(
    r"(โอน(?:เงิน)?(?:แล้ว|ให้แล้ว|ไปแล้ว)|ส่งสลิป|แนบสลิป|สลิปแล้ว)",
    re.IGNORECASE,
)
_COD_INTENT = re.compile(
    r"(เก็บปลายทาง|ปลายทาง|cod\b)",
    re.IGNORECASE,
)
_BUY_INTENT = re.compile(
    r"(สนใจรับ|รับเลย|เอาเลย|สั่งเลย|ซื้อเลย|ตกลงซื้อ|เอาครับ|เอาค่ะ)",
    re.IGNORECASE,
)


def _message_text(content) -> str:
    if not content:
        return ""
    if isinstance(content, list):
        parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("text")
        ]
        return " ".join(parts)
    return str(content)


def looks_like_transfer_intent(text: str) -> bool:
    """True when the customer indicates bank transfer payment."""
    return bool(_TRANSFER_INTENT.search(text or ""))


def looks_like_cod_intent(text: str) -> bool:
    """True when the customer chooses cash-on-delivery."""
    return bool(_COD_INTENT.search(text or ""))


def looks_like_transfer_already_done(text: str) -> bool:
    """True when the customer says they already transferred / sent a slip."""
    return bool(_TRANSFER_ALREADY_DONE.search(text or ""))


def should_auto_generate_qr(messages: list) -> bool:
    """True when the latest customer message is choosing bank transfer with a known total."""
    if not messages:
        return False
    last = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) and msg.content:
            last = _message_text(msg.content).strip()
            break
    if not last or looks_like_cod_intent(last):
        return False
    if not looks_like_transfer_intent(last) or looks_like_transfer_already_done(last):
        return False
    return resolve_order_from_messages(messages) is not None


def _parse_amount(raw: str) -> float | None:
    cleaned = (raw or "").replace(",", "").strip()
    if not cleaned:
        return None
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return value if value > 0 else None


def _latest_discount(messages: list) -> dict | None:
    for msg in reversed(messages):
        if not isinstance(msg, ToolMessage):
            continue
        text = _message_text(msg.content)
        match = _DISCOUNT_BLOCK.search(text)
        if not match:
            continue
        amount = _parse_amount(match.group(2))
        if amount is None:
            continue
        product = match.group(1).strip()
        return {
            "amount": amount,
            "items": product,
            "source": "discount_calculation",
        }
    return None


def _latest_catalog_price(messages: list) -> dict | None:
    for msg in reversed(messages):
        if not isinstance(msg, ToolMessage):
            continue
        text = _message_text(msg.content)
        if "[Catalog]" not in text and "[Vector Search]" not in text and "[Keyword Fallback]" not in text:
            continue

        for line in text.splitlines():
            baht = _TOOL_PRICE_BAHT.search(line)
            if baht:
                amount = _parse_amount(baht.group(1))
                if amount is None:
                    continue
                name = line.split("|", 1)[0].strip().lstrip("- ").strip()
                return {
                    "amount": amount,
                    "items": name or "สินค้าที่สั่ง",
                    "source": "catalog_tool",
                }

            thb = _TOOL_PRICE_THB.search(line)
            if thb:
                amount = _parse_amount(thb.group(1))
                if amount is None:
                    continue
                name = "สินค้าที่สั่ง"
                if "[Product]" in line:
                    name = line.split("[Product]", 1)[-1].split("(", 1)[0].strip()
                return {
                    "amount": amount,
                    "items": name,
                    "source": "catalog_tool",
                }
    return None


def _latest_ai_quoted_price(messages: list) -> dict | None:
    """Fallback: last agent message with an explicit baht amount before transfer."""
    transfer_idx = None
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if isinstance(msg, HumanMessage) and looks_like_transfer_intent(
            _message_text(msg.content)
        ):
            transfer_idx = idx
            break

    scan_until = transfer_idx if transfer_idx is not None else len(messages)
    for msg in reversed(messages[:scan_until]):
        if not isinstance(msg, AIMessage):
            continue
        if getattr(msg, "tool_calls", None):
            continue
        text = _message_text(msg.content)
        matches = list(_AI_FINAL_BAHT.finditer(text))
        if matches:
            amount = _parse_amount(matches[-1].group(1))
            if amount is not None:
                return {
                    "amount": amount,
                    "items": "สินค้าที่สั่ง",
                    "source": "agent_quote",
                }
        # Fallback: last baht amount mentioned by the agent before transfer.
        any_matches = list(_AI_ANY_BAHT.finditer(text))
        if any_matches:
            amount = _parse_amount(any_matches[-1].group(1))
            if amount is not None:
                return {
                    "amount": amount,
                    "items": "สินค้าที่สั่ง",
                    "source": "agent_quote",
                }
    return None


def _latest_approved_discount_price(messages: list) -> dict | None:
    """Use the manager-approved final price from the structured approval reply."""
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        if getattr(msg, "tool_calls", None):
            continue
        text = _message_text(msg.content)
        match = _APPROVAL_FINAL_BAHT.search(text)
        if not match:
            continue
        amount = _parse_amount(match.group(1))
        if amount is None:
            continue
        product = "สินค้าที่สั่ง"
        product_match = re.search(r"•\s*สินค้า:\s*(.+)", text)
        if product_match:
            product = product_match.group(1).strip()
        return {
            "amount": amount,
            "items": product,
            "source": "approved_discount",
        }
    return None


def resolve_order_from_messages(messages: list) -> dict | None:
    """Return {amount, items, source} inferred from the chat, or None if unknown."""
    if not messages:
        return None

    for resolver in (
        _latest_approved_discount_price,
        _latest_discount,
        _latest_catalog_price,
        _latest_ai_quoted_price,
    ):
        resolved = resolver(messages)
        if resolved:
            return resolved

    return None


def conversation_has_buy_intent(messages: list) -> bool:
    """True if the customer already agreed to purchase in this thread."""
    for msg in messages:
        if isinstance(msg, HumanMessage) and _BUY_INTENT.search(
            _message_text(msg.content)
        ):
            return True
    return False
