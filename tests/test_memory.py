"""Tests for conversation memory (summarizer transcript + durable state injection)."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ai_sales.nodes.agent_nodes import (
    _memory_context_for_agent,
    _render_for_summary,
    context_summarizer_node,
)


def test_render_for_summary_includes_tool_results():
    messages = [
        HumanMessage(content="มีเคส iPhone 15 ไหม"),
        AIMessage(content="", tool_calls=[{"name": "search_products", "args": {}, "id": "1"}]),
        ToolMessage(
            content='[{"name": "เคส iPhone 15", "price": 490}]',
            tool_call_id="1",
        ),
        AIMessage(content="มีเคส iPhone 15 ราคา 490 บาทค่ะ"),
    ]
    transcript = _render_for_summary(messages)
    assert "iPhone 15" in transcript
    assert "490" in transcript
    assert "ข้อมูลจากระบบ" in transcript


def test_render_for_summary_skips_tool_call_only_ai_messages():
    messages = [
        AIMessage(content="", tool_calls=[{"name": "search_products", "args": {}, "id": "1"}]),
        ToolMessage(content="no results", tool_call_id="1"),
    ]
    transcript = _render_for_summary(messages)
    assert "no results" in transcript
    assert transcript.count("ร้าน:") == 0


def test_memory_context_includes_shipping_and_summary():
    state = {
        "conversation_summary": "- ลูกค้าใช้ iPhone 15",
        "shipping_info": {"customer_name": "สมชาย", "address": "กรุงเทพ"},
        "lead_score": 60,
        "pipeline_stage": "negotiation",
        "pending_discount_approval": {},
    }
    block = _memory_context_for_agent(state)
    assert "iPhone 15" in block
    assert "สมชาย" in block
    assert "negotiation" in block


def test_context_summarizer_does_not_delete_when_transcript_empty():
    """Tool-only older slice must not be removed without a summary."""
    messages = []
    for _ in range(20):
        messages.append(
            AIMessage(content="", tool_calls=[{"name": "x", "args": {}, "id": "t"}])
        )
        messages.append(ToolMessage(content="{}", tool_call_id="t"))
    state = {"messages": messages, "conversation_summary": ""}
    result = context_summarizer_node(state)
    assert result == {}
