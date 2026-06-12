"""Tests for graph routing functions."""

from langchain_core.messages import AIMessage, HumanMessage

from ai_sales.consts import HUMAN_APPROVAL, LEAD_SCORER, ROUTE_END, TOOL_EXECUTOR
from ai_sales.nodes.routing import route_after_agent, route_after_scoring


def _state(**kwargs) -> dict:
    defaults = {
        "messages": [HumanMessage(content="Hello")],
        "lead_score": 0,
        "pipeline_stage": "new",
        "pending_discount_approval": {},
        "discount_approved": False,
        "tool_iterations": 0,
    }
    defaults.update(kwargs)
    return defaults


def test_route_after_agent_with_tool_calls():
    msg = AIMessage(content="", tool_calls=[{"name": "search_inventory", "args": {}, "id": "1"}])
    state = _state(messages=[msg])
    assert route_after_agent(state) == TOOL_EXECUTOR


def test_route_after_agent_without_tool_calls():
    msg = AIMessage(content="Here are our plans.")
    state = _state(messages=[msg])
    assert route_after_agent(state) == LEAD_SCORER


def test_route_after_agent_max_iterations_forces_scoring():
    msg = AIMessage(content="", tool_calls=[{"name": "search_inventory", "args": {}, "id": "1"}])
    state = _state(messages=[msg], tool_iterations=10)
    assert route_after_agent(state) == LEAD_SCORER


def test_route_after_scoring_with_pending_discount():
    state = _state(
        pending_discount_approval={"product": "Pro Plan", "discount_pct": 20}
    )
    assert route_after_scoring(state) == HUMAN_APPROVAL


def test_route_after_scoring_no_pending():
    state = _state(pending_discount_approval={})
    assert route_after_scoring(state) == ROUTE_END
