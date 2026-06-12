"""LINE Messaging API push helpers (used after async LangGraph completes)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from ai_sales.channels import line_delivery

LINE_PUSH_URL = "https://api.line.me/v2/bot/message/push"


def line_target_from_thread_id(thread_id: str) -> str | None:
    """Extract LINE push ``to`` from our thread id (``line:user:U…`` etc.)."""
    for prefix in ("line:user:", "line:group:", "line:room:"):
        if thread_id.startswith(prefix):
            return thread_id[len(prefix) :]
    return None


def resolve_qr_image_url(
    payment_qr: dict | None, callback_qr_url: str | None = None
) -> str | None:
    return line_delivery.resolve_qr_image_url(payment_qr, callback_qr_url)


def push_messages(to: str, messages: list[dict]) -> bool:
    """Push up to 5 messages to a LINE user/group/room."""
    token = (os.getenv("LINE_CHANNEL_ACCESS_TOKEN") or "").strip()
    if not token or not to or not messages:
        return False

    payload = json.dumps({"to": to, "messages": messages[:5]}).encode("utf-8")
    request = urllib.request.Request(
        LINE_PUSH_URL,
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as resp:
            return 200 <= resp.status < 300
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print(f"[LINE] push failed ({exc.code}): {detail}")
        return False
    except urllib.error.URLError as exc:
        print(f"[LINE] push connection error: {exc.reason}")
        return False


def push_assistant_reply(
    to: str,
    text: str,
    payment_qr: dict | None = None,
    qr_image_url: str | None = None,
) -> bool:
    """Push the assistant text plus optional PromptPay QR image."""
    messages = line_delivery.build_line_messages(text, payment_qr, qr_image_url)
    return push_messages(to, messages)
