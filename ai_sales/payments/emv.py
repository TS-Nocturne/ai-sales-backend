"""Local PromptPay EMV QR payload generation (no Slip2Go quota required).

Used for the fixed store QR image (no embedded amount). Dynamic partial-payment
QRs still go through Slip2Go in ``qr.py``.
"""

from __future__ import annotations

import re

from ai_sales.payments import slip2go

PROMPTPAY_AID = "A000000677010111"


def _field(tag: str, value: str) -> str:
    return f"{tag}{len(value):02d}{value}"


def _crc16_ccitt(data: str) -> str:
    crc = 0xFFFF
    for byte in data.encode("utf-8"):
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return f"{crc:04X}"


def _merchant_account(code: str, prompt_pay_type: str) -> str:
    digits = slip2go.normalize_promptpay_code(code)
    if not digits:
        raise slip2go.Slip2GoError("ต้องระบุ PROMPTPAY_CODE")

    ptype = slip2go.normalize_promptpay_type(prompt_pay_type)
    if ptype == "phone_number":
        if not re.fullmatch(r"0\d{9}", digits):
            raise slip2go.Slip2GoError(
                "PROMPTPAY_CODE สำหรับเบอร์โทรต้องเป็น 10 หลัก (เช่น 0812345678)"
            )
        target = f"0066{digits[1:]}"
        sub_tag = "01"
    elif ptype == "citizen_id":
        if len(digits) != 13:
            raise slip2go.Slip2GoError(
                "PROMPTPAY_CODE สำหรับบัตรประชาชนต้องเป็น 13 หลัก"
            )
        target = digits
        sub_tag = "02"
    elif ptype == "e_wallet":
        target = digits
        sub_tag = "03"
    else:
        raise slip2go.Slip2GoError(f"ไม่รองรับ promptPayType: {ptype}")

    merchant = _field("00", PROMPTPAY_AID) + _field(sub_tag, target)
    return _field("29", merchant)


def build_promptpay_emv(
    prompt_pay_code: str,
    prompt_pay_type: str = "phone_number",
    *,
    amount: float | None = None,
) -> str:
    """Build a Thai PromptPay EMV QR string (static when amount is None)."""
    payload = _field("00", "01")
    payload += _field("01", "12" if amount is not None else "11")
    payload += _merchant_account(prompt_pay_code, prompt_pay_type)
    payload += _field("58", "TH")
    payload += _field("53", "764")
    if amount is not None:
        if amount <= 0:
            raise slip2go.Slip2GoError("ยอดชำระเงินต้องมากกว่า 0 บาท")
        payload += _field("54", f"{amount:.2f}")
    payload += "6304"
    return payload + _crc16_ccitt(payload)
