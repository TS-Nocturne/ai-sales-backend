"""PromptPay QR helpers: Slip2Go payload → PNG image."""

from __future__ import annotations

import base64
import io
import os

import qrcode

from ai_sales.payments import slip2go


def _promptpay_settings() -> tuple[str, str, str]:
    code = slip2go.normalize_promptpay_code(os.getenv("PROMPTPAY_CODE"))
    ptype = slip2go.normalize_promptpay_type(os.getenv("PROMPTPAY_TYPE"))
    name = (os.getenv("PROMPTPAY_ACCOUNT_NAME") or "ร้านค้า").strip()
    if not code:
        raise slip2go.Slip2GoError(
            "ยังไม่ได้ตั้งค่า PROMPTPAY_CODE ใน .env (เบอร์โทร/เลขบัตร PromptPay)"
        )
    return code, ptype, name


def emv_payload_to_png_base64(emv_payload: str) -> str:
    """Render a PromptPay EMV payload string as a PNG data URL."""
    qr = qrcode.QRCode(version=None, box_size=8, border=2)
    qr.add_data(emv_payload)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def create_static_store_qr(amount: float, items: str = "") -> dict:
    """Return payment metadata pointing at the fixed store PromptPay QR image.

    The branded store QR is served by Next.js from
    ``/payment-qr/store-promptpay.png`` — LINE loads it via ``use_static``.
    The customer enters the transfer amount in their banking app; the bot
    states the amount in the accompanying text message.
    """
    if amount <= 0:
        raise slip2go.Slip2GoError("ยอดชำระเงินต้องมากกว่า 0 บาท")

    code, _, account_name = _promptpay_settings()
    return {
        "amount": float(amount),
        "account_name": account_name,
        "prompt_pay_code": code,
        "use_static": True,
        "is_partial": False,
        "items": (items or "").strip(),
    }


def create_promptpay_qr(amount: float, items: str = "", *, partial: bool = False) -> dict:
    """Generate payment QR data — static store image (full) or dynamic (partial top-up)."""
    if amount <= 0:
        raise slip2go.Slip2GoError("ยอดชำระเงินต้องมากกว่า 0 บาท")

    if not partial:
        return create_static_store_qr(amount, items)

    code, ptype, account_name = _promptpay_settings()
    resp = slip2go.generate_promptpay_qr(
        prompt_pay_code=code,
        prompt_pay_type=ptype,
        account_name=account_name,
        amount=amount,
    )
    data = resp.get("data") if isinstance(resp, dict) else None
    if not isinstance(data, dict):
        raise slip2go.Slip2GoError("Slip2Go ไม่ส่งข้อมูล QR กลับมา")

    emv = (data.get("qrCode") or "").strip()
    if not emv:
        raise slip2go.Slip2GoError("Slip2Go ไม่ส่ง qrCode กลับมา")

    return {
        "amount": float(amount),
        "account_name": data.get("accountName") or account_name,
        "prompt_pay_code": code,
        "emv_payload": emv,
        "image_base64": emv_payload_to_png_base64(emv),
        "is_partial": True,
        "items": (items or "").strip(),
    }
