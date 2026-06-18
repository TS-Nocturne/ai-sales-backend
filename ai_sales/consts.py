"""Named constants for the AI Sales Agent graph."""

import os
from pathlib import Path

# Node names
SALES_AGENT = "sales_agent_node"
TOOL_EXECUTOR = "tool_executor"
LEAD_SCORER = "lead_scorer"
HUMAN_APPROVAL = "human_approval_node"
POST_APPROVAL = "post_approval_node"
CONTEXT_SUMMARIZER = "context_summarizer_node"
HANDOFF = "handoff_node"

# Routing targets
ROUTE_END = "__end__"

# Safety limits
MAX_TOOL_ITERATIONS = 10
MANAGER_APPROVAL_DISCOUNT_THRESHOLD = 15
MAX_DISCOUNT_PERCENT = 50

# Retrieval / context limits (token efficiency — avoid context bleeding)
VECTOR_TOP_K = 5
PRODUCT_DISPLAY_LIMIT = 3
PREFETCH_DISPLAY_LIMIT = 3
LIST_PRODUCTS_DEFAULT_LIMIT = 3
LIST_PRODUCTS_MAX_LIMIT = 10

# Conversation memory: when the running transcript exceeds SUMMARY_TRIGGER_COUNT
# messages, fold everything older than SUMMARY_KEEP_RECENT messages (snapped to a
# turn boundary) into a rolling summary. Keep enough recent messages for multi-step
# tool turns (search → calculate → reply).
SUMMARY_TRIGGER_COUNT = 16
SUMMARY_KEEP_RECENT = 8



# Pipeline stages
STAGE_NEW = "new"
STAGE_QUALIFIED = "qualified"
STAGE_NEGOTIATION = "negotiation"
STAGE_CLOSED_WON = "closed_won"
STAGE_CLOSED_LOST = "closed_lost"
