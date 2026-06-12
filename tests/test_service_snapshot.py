"""Regression tests for service-layer state snapshots."""

from unittest.mock import MagicMock

from ai_sales.api.service import _snapshot
from ai_sales.consts import HUMAN_APPROVAL


def test_snapshot_detects_awaiting_human_approval():
    """Ensure HUMAN_APPROVAL is imported and used in _snapshot."""
    graph = MagicMock()
    graph.get_state.return_value = MagicMock(
        values={
            "lead_score": 72,
            "pipeline_stage": "negotiation",
            "pending_discount_approval": {"product": "เคส", "discount_pct": 20},
            "payment_qr": {},
        },
        next=(HUMAN_APPROVAL,),
    )

    snap = _snapshot(graph, {"configurable": {"thread_id": "test-thread"}})

    assert snap["requires_approval"] is True
    assert snap["lead_score"] == 72
    assert HUMAN_APPROVAL in snap["next_nodes"]
