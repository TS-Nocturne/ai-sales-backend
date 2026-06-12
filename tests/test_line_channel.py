from ai_sales.channels import line_delivery
from ai_sales.channels.line import line_target_from_thread_id


def test_line_target_from_thread_id():
    assert line_target_from_thread_id("line:user:Uabc123") == "Uabc123"
    assert line_target_from_thread_id("line:group:Cgroup1") == "Cgroup1"
    assert line_target_from_thread_id("dashboard-conv-1") is None


def test_resolve_static_qr_url(monkeypatch):
    monkeypatch.setenv("PUBLIC_APP_URL", "https://example.ngrok.app")
    url = line_delivery.resolve_qr_image_url({"use_static": True})
    assert url == "https://example.ngrok.app/payment-qr/store-promptpay.png"


def test_build_line_messages_strips_tag_and_adds_image(monkeypatch):
    monkeypatch.setenv("PUBLIC_APP_URL", "https://example.ngrok.app")
    reply = (
        "รับทราบค่ะ ยอดโอน 392 บาท\n"
        "[[LINE_QR_IMAGE:https://example.ngrok.app/payment-qr/store-promptpay.png]]"
    )
    messages = line_delivery.build_line_messages(
        reply,
        payment_qr={"use_static": True, "amount": 392},
    )
    assert messages[0]["type"] == "text"
    assert "[[LINE_QR_IMAGE" not in messages[0]["text"]
    assert messages[1]["type"] == "image"
    assert messages[1]["originalContentUrl"].endswith("store-promptpay.png")
