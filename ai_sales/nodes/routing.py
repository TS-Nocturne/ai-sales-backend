"""
Routing functions for conditional edges in the sales agent graph.

Follows AGENTS.md guidelines:
- Check state["messages"][-1].tool_calls in conditional routing
  to decide if the graph should transition to an action node or END.
"""

from ai_sales.consts import (
    HUMAN_APPROVAL,
    LEAD_SCORER,
    MAX_TOOL_ITERATIONS,
    ROUTE_END,
    TOOL_EXECUTOR,
)
from ai_sales.state import SalesState


def route_after_agent(state: SalesState) -> str:
    """Conditional routing after the sales agent node.

    AGENTS.md: Check state["messages"][-1].tool_calls to decide
    if the graph should transition to the tool executor or lead scorer.
    """
    last_message = state["messages"][-1]
    iterations = state.get("tool_iterations", 0)

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        if iterations >= MAX_TOOL_ITERATIONS:
            return LEAD_SCORER
        return TOOL_EXECUTOR

    return LEAD_SCORER


def route_after_scoring(state: SalesState) -> str:
    """Conditional routing after the lead scorer node.

    If a discount approval is pending, route to the human approval node.
    Otherwise, end the graph.
    """
    pending = state.get("pending_discount_approval", {})

    if pending:
        return HUMAN_APPROVAL

    return ROUTE_END
