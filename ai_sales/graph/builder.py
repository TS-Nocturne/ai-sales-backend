"""
LangGraph StateGraph construction for the AI Sales Agent.

Follows AGENTS.md guidelines:
- Use StateGraph with SalesState schema.
- Set entry point with workflow.set_entry_point().
- Enable memory with PostgresSaver (Neon pooled DATABASE_URL).
- Pause for human feedback with interrupt_before=["human_approval_node"].
- Always include a thread_id in configuration.
"""

from dotenv import load_dotenv
import logging
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, StateGraph

from ai_sales.consts import (
    CONTEXT_SUMMARIZER,
    HANDOFF,
    HUMAN_APPROVAL,
    LEAD_SCORER,
    POST_APPROVAL,
    ROUTE_END,
    SALES_AGENT,
    TOOL_EXECUTOR,
)
from ai_sales.nodes.agent_nodes import (
    context_summarizer_node,
    handoff_node,
    human_approval_node,
    lead_scorer_node,
    post_approval_node,
    sales_agent_node,
    tool_executor_node,
)
from ai_sales.nodes.routing import route_after_agent, route_after_scoring, route_after_summarizer
from ai_sales.state import SalesState

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


def build_graph(pool):
    """Build and compile the Sales Agent StateGraph.

    Returns:
        compiled_graph
    """
    # -----------------------------------------------------------------------
    # Persistence: PostgresSaver
    # -----------------------------------------------------------------------
    memory = PostgresSaver(pool)
    memory.setup()

    # -----------------------------------------------------------------------
    # Build the StateGraph
    # -----------------------------------------------------------------------
    workflow = StateGraph(SalesState)

    workflow.add_node(CONTEXT_SUMMARIZER, context_summarizer_node)
    workflow.add_node(HANDOFF, handoff_node)
    workflow.add_node(SALES_AGENT, sales_agent_node)
    workflow.add_node(TOOL_EXECUTOR, tool_executor_node)
    workflow.add_node(LEAD_SCORER, lead_scorer_node)
    workflow.add_node(HUMAN_APPROVAL, human_approval_node)
    workflow.add_node(POST_APPROVAL, post_approval_node)

    # Compact long histories before the agent runs (token/memory saver), then
    # hand off to the sales agent. Tool loops return to the agent directly.
    workflow.set_entry_point(CONTEXT_SUMMARIZER)
    workflow.add_conditional_edges(
        CONTEXT_SUMMARIZER,
        route_after_summarizer,
        {
            HANDOFF: HANDOFF,
            SALES_AGENT: SALES_AGENT,
        },
    )
    workflow.add_edge(HANDOFF, END)

    workflow.add_conditional_edges(
        SALES_AGENT,
        route_after_agent,
        {
            TOOL_EXECUTOR: TOOL_EXECUTOR,
            LEAD_SCORER: LEAD_SCORER,
        },
    )

    workflow.add_edge(TOOL_EXECUTOR, SALES_AGENT)

    workflow.add_conditional_edges(
        LEAD_SCORER,
        route_after_scoring,
        {
            HUMAN_APPROVAL: HUMAN_APPROVAL,
            ROUTE_END: END,
        },
    )

    workflow.add_edge(HUMAN_APPROVAL, POST_APPROVAL)
    workflow.add_edge(POST_APPROVAL, END)

    graph = workflow.compile(
        checkpointer=memory,
        interrupt_before=[HUMAN_APPROVAL],
    )

    return graph
