"""Tests for structured approve/reject customer replies."""

from ai_sales.messages.approval_reply import format_discount_decision_reply


def test_approved_reply_has_clear_sections():
    reply = format_discount_decision_reply(
        {
            "product": "เคส iPhone 15 Pro Max",
            "discount_pct": 20,
            "original_price": 490,
        },
        approved=True,
    )
    assert "ข่าวดี" in reply
    assert "รายละเอียด" in reply
    assert "ขั้นตอนถัดไป" in reply
    assert "เคส iPhone 15 Pro Max" in reply
    assert "490 บาท" in reply
    assert "392 บาท" in reply
    assert "โอนเงิน" in reply


def test_rejected_reply_has_clear_sections():
    reply = format_discount_decision_reply(
        {
            "product": "เคส iPhone 15 Pro Max",
            "discount_pct": 25,
            "original_price": 490,
        },
        approved=False,
    )
    assert "ขอบคุณที่รอ" in reply
    assert "รายละเอียด" in reply
    assert "ทางเลือกอื่น" in reply
    assert "25%" in reply
    assert "490 บาท" in reply
    assert "หลายแบบ" not in reply
