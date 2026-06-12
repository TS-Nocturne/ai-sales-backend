"""Background jobs: run LangGraph then notify dashboard + LINE."""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request

from ai_sales.api import service
from ai_sales.channels import line as line_channel

logger = logging.getLogger(__name__)


def _empty_error_result(thread_id: str, reply: str) -> dict:
    return {
        "thread_id": thread_id,
        "reply": reply,
        "lead_score": 0,
        "pipeline_stage": "new",
        "requires_approval": False,
        "pending_discount_approval": None,
        "next_nodes": [],
        "shipping_info": None,
        "order_ready": False,
        "payment_qr": None,
        "pending_overpay": None,
        "overpay_resolution": None,
        "overpay_credit_amount": 0,
        "awaiting_refund_approval": False,
    }


def _post_dashboard_callback(result: dict, display_name: str | None) -> str | None:
    """Persist brain output in PostgreSQL via the Next.js internal callback."""
    base = (os.getenv("DASHBOARD_URL") or "http://127.0.0.1:3000").rstrip("/")
    url = f"{base}/api/internal/brain-callback"
    body = json.dumps(
        {
            "thread_id": result["thread_id"],
            "result": result,
            "display_name": display_name,
        }
    ).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    api_key = (os.getenv("INTERNAL_API_KEY") or "").strip()
    if not api_key:
        if os.getenv("ENV", "").strip().lower() in ("production", "prod"):
            logger.error("INTERNAL_API_KEY is required in production for brain-callback")
            return None
    else:
        headers["x-internal-key"] = api_key

    last_exc: Exception | None = None
    for attempt in range(3):
        request = urllib.request.Request(url, data=body, method="POST", headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as resp:
                payload = json.loads(resp.read().decode("utf-8") or "{}")
                qr_url = payload.get("qr_image_url")
                return qr_url if isinstance(qr_url, str) and qr_url.strip() else None
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "Dashboard brain-callback attempt %s/3 failed: %s",
                attempt + 1,
                exc,
            )
            if attempt < 2:
                time.sleep(1 * (2**attempt))

    logger.exception("Dashboard brain-callback failed after retries: %s", last_exc)
    return None


def process_chat_async(
    thread_id: str,
    message: str,
    payment_context: dict | None,
    line_push_target: str | None,
    display_name: str | None,
    attach_payment_qr: dict | None = None,
) -> None:
    """Run the sales agent, sync DB via Next.js, then push the reply to LINE."""
    try:
        result = service.send_message(thread_id, message, payment_context)
    except Exception as exc:
        logger.exception("Async LangGraph failed for thread %s", thread_id)
        result = _empty_error_result(
            thread_id,
            "ขออภัยค่ะ ระบบ AI ไม่พร้อมใช้งานชั่วคราว กรุณาลองใหม่อีกครั้ง",
        )
        result["_error"] = str(exc)

    if attach_payment_qr:
        existing = result.get("payment_qr") or {}
        if not existing.get("image_base64") and not existing.get("use_static"):
            result["payment_qr"] = attach_payment_qr

    qr_image_url = _post_dashboard_callback(result, display_name)

    # Default: dashboard brain-callback pushes to LINE (token lives in Next.js).
    # Set LINE_PUSH_FROM_BRAIN=true to push from Python instead (needs LINE token in .env).
    if os.getenv("LINE_PUSH_FROM_BRAIN", "").lower() in ("1", "true", "yes"):
        push_target = line_push_target or line_channel.line_target_from_thread_id(thread_id)
        if push_target:
            pushed = line_channel.push_assistant_reply(
                push_target,
                result.get("reply") or "",
                result.get("payment_qr"),
                qr_image_url,
            )
            if not pushed:
                logger.warning(
                    "LINE push from brain failed for thread %s (check LINE_CHANNEL_ACCESS_TOKEN)",
                    thread_id,
                )
