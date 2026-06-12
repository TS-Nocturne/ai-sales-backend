from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ai_sales.tools.order_total import (
    looks_like_transfer_intent,
    resolve_order_from_messages,
    should_auto_generate_qr,
)


def test_transfer_intent_thai():
    assert looks_like_transfer_intent("โอนเงินครับ")
    assert looks_like_transfer_intent("จะโอนให้นะครับ")
    assert not looks_like_transfer_intent("สนใจรับเลยครับ")


def test_resolve_from_discount_tool():
    messages = [
        HumanMessage(content="สนใจรับเลยครับ"),
        ToolMessage(
            content=(
                "[Discount Calculation] เคส iPhone 15 Pro Max:\n"
                "  Original Price:  490.00 THB\n"
                "  Discount:        15.0% (-73.50 THB)\n"
                "  Final Price:     416.50 THB\n"
                "  Customer Saves:  73.50 THB"
            ),
            tool_call_id="t1",
        ),
        HumanMessage(content="โอนเงินครับ"),
    ]
    resolved = resolve_order_from_messages(messages)
    assert resolved is not None
    assert resolved["amount"] == 416.5
    assert "iPhone" in resolved["items"]


def test_resolve_from_catalog_tool():
    messages = [
        ToolMessage(
            content=(
                "[Catalog] พบ 1 รายการ:\n"
                "- เคส iPhone 15 Pro Max | ราคา 490 บาท | หมวด: เคส | มีสินค้า"
            ),
            tool_call_id="t2",
        ),
        HumanMessage(content="โอนเงินครับ"),
    ]
    resolved = resolve_order_from_messages(messages)
    assert resolved is not None
    assert resolved["amount"] == 490.0


def test_resolve_from_agent_quote():
    messages = [
        AIMessage(content="เคสรุ่นนี้ราคา 490 บาท ลด 15% เหลือ 416.50 บาทค่ะ"),
        HumanMessage(content="โอนเงินครับ"),
    ]
    resolved = resolve_order_from_messages(messages)
    assert resolved is not None
    assert resolved["amount"] == 416.5


def test_resolve_from_discounted_price_phrase():
    messages = [
        AIMessage(content="ราคาหลังส่วนลดเป็น 392.00 บาท"),
        HumanMessage(content="โอนเงินครับ"),
    ]
    resolved = resolve_order_from_messages(messages)
    assert resolved is not None
    assert resolved["amount"] == 392.0


def test_resolve_from_manager_approved_discount_reply():
    messages = [
        AIMessage(
            content=(
                "สวัสดีค่ะ 🙏\n\n"
                "ข่าวดีค่ะ ผู้จัดการอนุมัติส่วนลดพิเศษให้แล้ว\n\n"
                "รายละเอียด\n"
                "• สินค้า: เคส iPhone 15 Pro Max\n"
                "• ราคาปกติ: 490 บาท\n"
                "• ส่วนลด: 20%\n"
                "• ราคาพิเศษหลังหักส่วนลด: 392 บาท"
            )
        ),
        HumanMessage(content="โอนเงินครับ"),
    ]
    resolved = resolve_order_from_messages(messages)
    assert resolved is not None
    assert resolved["amount"] == 392.0
    assert resolved["source"] == "approved_discount"
    assert "iPhone" in resolved["items"]


def test_approved_discount_beats_catalog_price():
    messages = [
        ToolMessage(
            content=(
                "[Catalog] พบ 1 รายการ:\n"
                "- เคส iPhone 15 Pro Max | ราคา 490 บาท | หมวด: เคส | มีสินค้า"
            ),
            tool_call_id="t2",
        ),
        AIMessage(
            content=(
                "ข่าวดีค่ะ ผู้จัดการอนุมัติส่วนลดพิเศษให้แล้ว\n"
                "• สินค้า: เคส iPhone 15 Pro Max\n"
                "• ราคาพิเศษหลังหักส่วนลด: 392 บาท"
            )
        ),
        HumanMessage(content="โอนเงินครับ"),
    ]
    resolved = resolve_order_from_messages(messages)
    assert resolved is not None
    assert resolved["amount"] == 392.0
    assert resolved["source"] == "approved_discount"


def test_should_auto_generate_qr_when_total_known():
    messages = [
        AIMessage(content="ราคาหลังส่วนลดเป็น 392.00 บาท"),
        HumanMessage(content="โอนเงินครับ"),
    ]
    assert should_auto_generate_qr(messages)


def test_should_not_auto_qr_for_cod():
    messages = [
        AIMessage(content="ราคา 490 บาท"),
        HumanMessage(content="เก็บปลายทางครับ"),
    ]
    assert not should_auto_generate_qr(messages)
