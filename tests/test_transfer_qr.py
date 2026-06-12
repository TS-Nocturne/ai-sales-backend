from langchain_core.messages import AIMessage, HumanMessage

from ai_sales.tools.sales_tools import transfer_payment_qr_update


def test_transfer_payment_qr_update_uses_conversation_total(monkeypatch):
    monkeypatch.setattr(
        "ai_sales.tools.sales_tools.payment_qr.create_promptpay_qr",
        lambda amount, items, partial=False: {
            "amount": amount,
            "account_name": "ร้านทดสอบ",
            "use_static": True,
        },
    )

    state = {
        "messages": [
            AIMessage(content="ราคาหลังส่วนลดเป็น 392.00 บาท"),
            HumanMessage(content="โอนเงินครับ"),
        ]
    }
    result = transfer_payment_qr_update(state)

    reply = result["messages"][0].content
    assert "392" in reply
    assert "QR PromptPay" in reply
    assert result["payment_qr"]["amount"] == 392.0
