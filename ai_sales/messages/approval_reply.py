"""Structured customer-facing replies after manager discount decisions."""

from __future__ import annotations


def _format_thb(amount: float) -> str:
    if amount <= 0:
        return ""
    text = f"{amount:,.2f}".rstrip("0").rstrip(".")
    return f"{text} บาท"


def format_discount_decision_reply(discount_info: dict, approved: bool) -> str:
    """Build a warm, structured Thai message for approve/reject outcomes."""
    product = (discount_info.get("product") or "").strip() or "สินค้าที่สนใจ"
    discount_pct = float(discount_info.get("discount_pct") or 0)
    original_price = float(discount_info.get("original_price") or 0)
    final_price = (
        original_price * (1 - discount_pct / 100) if original_price else 0.0
    )

    if approved:
        lines = [
            "สวัสดีค่ะ 🙏",
            "",
            "ข่าวดีค่ะ ผู้จัดการอนุมัติส่วนลดพิเศษให้แล้ว",
            "",
            "รายละเอียด",
            f"• สินค้า: {product}",
        ]
        if original_price > 0:
            lines.append(f"• ราคาปกติ: {_format_thb(original_price)}")
        if discount_pct > 0:
            lines.append(f"• ส่วนลด: {discount_pct:g}%")
        if final_price > 0:
            lines.append(f"• ราคาพิเศษหลังหักส่วนลด: {_format_thb(final_price)}")
        lines.extend(
            [
                "",
                "ขั้นตอนถัดไป",
                "หากพร้อมสั่งซื้อ แจ้งได้เลยนะคะ",
                'หรือพิมพ์ว่า "โอนเงิน" เพื่อรับ QR PromptPay สำหรับชำระเงินค่ะ',
            ]
        )
        return "\n".join(lines)

    lines = [
        "ขอบคุณที่รอค่ะ 🙏",
        "",
        "ผู้จัดการพิจารณาแล้ว ขออภัยที่ยังไม่สามารถให้ส่วนลด "
        f"{discount_pct:g}% สำหรับ {product} ได้ในขณะนี้ค่ะ",
        "",
        "รายละเอียด",
        f"• สินค้า: {product}",
    ]
    if original_price > 0:
        lines.append(f"• ราคามาตรฐาน: {_format_thb(original_price)}")
    lines.extend(
        [
            "",
            "ทางเลือกอื่น",
            "หากสนใจสั่งในราคามาตรฐาน หรืออยากดูสินค้าตัวอื่นในงบใกล้เคียง",
            "แจ้งได้เลยนะคะ ยินดีช่วยเหลือค่ะ",
        ]
    )
    return "\n".join(lines)
