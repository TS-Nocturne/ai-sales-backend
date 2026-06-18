"""Tests for human handoff detection and routing."""

from langchain_core.messages import HumanMessage

from ai_sales.consts import HANDOFF, SALES_AGENT
from ai_sales.handoff import HANDOFF_REPLY, looks_like_handoff_intent
from ai_sales.nodes.agent_nodes import handoff_node, sales_agent_node
from ai_sales.nodes.routing import route_after_summarizer


def test_handoff_intent_detects_staff_request():
    assert looks_like_handoff_intent("เรียกพนักงานให้หน่อย")
    assert looks_like_handoff_intent("ขอคุยกับแอดมินครับ")
    assert looks_like_handoff_intent("แอดมินอยู่ไหมคะ")
    assert looks_like_handoff_intent("ติดต่อเจ้าหน้าที่หน่อย")


def test_handoff_intent_ignores_normal_sales_chat():
    assert not looks_like_handoff_intent("มีเคส iPad ไหม")
    assert not looks_like_handoff_intent("โอนเงินครับ")
    assert not looks_like_handoff_intent("สนใจรับเลยค่ะ")
    assert not looks_like_handoff_intent("")


def test_route_after_summarizer_escalates_before_sales_agent():
    state = {
        "messages": [HumanMessage(content="เรียกพนักงานให้หน่อย")],
    }
    assert route_after_summarizer(state) == HANDOFF


def test_route_after_summarizer_continues_to_sales_agent():
    state = {
        "messages": [HumanMessage(content="มีเคส iPhone 15 ไหม")],
    }
    assert route_after_summarizer(state) == SALES_AGENT


def test_handoff_node_returns_fixed_reply_and_flags():
    state = {
        "messages": [HumanMessage(content="เรียกพนักงานให้หน่อย")],
    }
    result = handoff_node(state)
    assert result["handoff_requested"] is True
    assert result["handoff_reason"] == "เรียกพนักงานให้หน่อย"
    assert result["messages"][0].content == HANDOFF_REPLY


def test_sales_agent_node_short_circuits_handoff_without_llm(monkeypatch):
    def fail_llm(*_args, **_kwargs):
        raise AssertionError("LLM must not run on handoff turns")

    monkeypatch.setattr("ai_sales.nodes.agent_nodes.get_llm_with_tools", fail_llm)

    state = {
        "messages": [HumanMessage(content="ขอคุยกับคน")],
    }
    result = sales_agent_node(state)
    assert result["handoff_requested"] is True
    assert result["messages"][0].content == HANDOFF_REPLY
