"""Tests for lead scoring JSON parsing."""

from ai_sales.nodes.agent_nodes import _parse_scoring_response


def test_parse_direct_json():
    content = '{"lead_score": 75, "pipeline_stage": "qualified", "needs_discount_approval": false, "summary": "Good lead"}'
    result = _parse_scoring_response(content)
    assert result["lead_score"] == 75
    assert result["pipeline_stage"] == "qualified"
    assert result["needs_discount_approval"] is False


def test_parse_json_in_markdown_fence():
    content = '```json\n{"lead_score": 80, "pipeline_stage": "negotiation", "needs_discount_approval": true, "discount_percent": 20}\n```'
    result = _parse_scoring_response(content)
    assert result["lead_score"] == 80
    assert result["needs_discount_approval"] is True


def test_parse_json_with_surrounding_text():
    content = 'Analysis complete.\n{"lead_score": 60, "pipeline_stage": "new", "needs_discount_approval": false}\nDone.'
    result = _parse_scoring_response(content)
    assert result["lead_score"] == 60


def test_parse_invalid_json_returns_defaults():
    result = _parse_scoring_response("not valid json at all")
    assert result["lead_score"] == 50
    assert result["pipeline_stage"] == "qualified"
    assert result["needs_discount_approval"] is False
