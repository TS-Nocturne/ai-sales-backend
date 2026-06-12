"""LINE payload interception — strip QR tags from text and build text+image messages."""

from __future__ import annotations

import os
import re

LINE_QR_TAG_RE = re.compile(r"\[\[LINE_QR_IMAGE:([^\]]+)\]\]")
STATIC_STORE_QR_PATH = "/payment-qr/store-promptpay.png"


def format_line_qr_tag(image_url: str) -> str:
    """Machine-readable breadcrumb for the LINE delivery layer (hidden from customers)."""
    return f"[[LINE_QR_IMAGE:{image_url.strip()}]]"


def resolve_qr_image_url(
    payment_qr: dict | None, callback_qr_url: str | None = None
) -> str | None:
    """Resolve a public HTTPS URL for a PromptPay QR image."""
    if callback_qr_url:
        return callback_qr_url.strip() or None
    if not payment_qr:
        return None
    static_url = (payment_qr.get("static_url") or "").strip()
    if static_url:
        return static_url
    if payment_qr.get("use_static"):
        base = (os.getenv("PUBLIC_APP_URL") or "").strip().rstrip("/")
        if base.startswith("https://"):
            return f"{base}{STATIC_STORE_QR_PATH}"
    return None


def qr_tag_for_payment(payment_qr: dict | None) -> str | None:
    """Build a QR tag when we already know the public image URL."""
    url = resolve_qr_image_url(payment_qr)
    return format_line_qr_tag(url) if url else None


def embed_line_qr_tag(text: str, payment_qr: dict | None) -> str:
    """Append a QR tag to assistant text when payment metadata is present."""
    body = (text or "").strip()
    tag = qr_tag_for_payment(payment_qr)
    if not tag or tag in body:
        return body
    return f"{body}\n{tag}" if body else tag


def strip_line_qr_tags(text: str) -> str:
    """Remove QR tags before showing text to the customer on LINE."""
    cleaned = LINE_QR_TAG_RE.sub("", text or "")
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def extract_line_qr_urls(text: str) -> list[str]:
    return [m.group(1).strip() for m in LINE_QR_TAG_RE.finditer(text or "") if m.group(1).strip()]


def build_line_messages(
    reply: str,
    payment_qr: dict | None = None,
    qr_image_url: str | None = None,
) -> list[dict]:
    """Build LINE push payload: customer text + optional QR image message."""
    tagged_urls = extract_line_qr_urls(reply)
    text = strip_line_qr_tags(reply)
    if not text:
        text = "รับทราบค่ะ กรุณาสแกน QR ด้านล่างเพื่อชำระเงินนะคะ 🙏"

    messages: list[dict] = [{"type": "text", "text": text[:5000]}]

    img_url = qr_image_url or (tagged_urls[0] if tagged_urls else None)
    if not img_url:
        img_url = resolve_qr_image_url(payment_qr)

    if img_url:
        messages.append(
            {
                "type": "image",
                "originalContentUrl": img_url,
                "previewImageUrl": img_url,
            }
        )
    elif payment_qr and (payment_qr.get("use_static") or payment_qr.get("image_base64")):
        print(
            "[LINE] มี payment_qr แต่ resolve URL ไม่ได้ — "
            "ตั้ง PUBLIC_APP_URL เป็น HTTPS (เช่น ngrok) แล้วลองใหม่"
        )

    return messages
