"""Tests for customer reply extraction in the HTTP service layer."""

from langchain_core.messages import AIMessage, HumanMessage

from ai_sales.api.service import _latest_reply, _latest_reply_after_last_human


def test_latest_reply_respects_since_index():
    messages = [
        HumanMessage(content="old"),
        AIMessage(content="old reply"),
        HumanMessage(content="new"),
        AIMessage(content="new reply"),
    ]
    assert _latest_reply(messages, since=2) == "new reply"


def test_latest_reply_after_last_human_survives_synthetic_shrink():
    """Simulates context summarizer shrinking the list mid-turn."""
    messages = [
        HumanMessage(content="current question"),
        AIMessage(content="สวัสดีค่ะ มีเคส iPhone 15 ให้เลือกค่ะ"),
    ]
    # Old bug: since=12 on a 2-message list returned empty.
    assert _latest_reply(messages, since=12) == ""
    assert _latest_reply_after_last_human(messages) == "สวัสดีค่ะ มีเคส iPhone 15 ให้เลือกค่ะ"


def test_latest_reply_after_last_human_skips_internal_scoring():
    messages = [
        HumanMessage(content="สอบถามราคา"),
        AIMessage(content="[การให้คะแนนลีดเสร็จสมบูรณ์]\nคะแนน: 50"),
        AIMessage(content="ราคา 490 บาทค่ะ"),
    ]
    # Internal scoring message is after human but real reply is earlier AIMessage
    # when reversed we should find "ราคา 490 บาทค่ะ" - wait order is:
    # human, internal, customer reply - reversed: customer reply first after skip internal
    # Actually order: human idx 0, internal idx 1, reply idx 2
    # since=1: [internal, reply] reversed -> skip internal, get reply
    assert _latest_reply_after_last_human(messages) == "ราคา 490 บาทค่ะ"
