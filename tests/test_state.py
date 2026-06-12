"""Tests for state helpers."""

from ai_sales.state import initial_state


def test_initial_state_defaults():
    state = initial_state()
    assert state["lead_score"] == 0
    assert state["pipeline_stage"] == "new"
    assert state["pending_discount_approval"] == {}
    assert state["discount_approved"] is False
    assert state["tool_iterations"] == 0


def test_initial_state_overrides():
    state = initial_state(lead_score=80, pipeline_stage="qualified")
    assert state["lead_score"] == 80
    assert state["pipeline_stage"] == "qualified"
