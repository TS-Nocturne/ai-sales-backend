"""
State schema for the AI Sales & Lead Scoring Agent.

Follows AGENTS.md guidelines:
- Uses MessagesState as the base (provides messages: Annotated[list, add_messages]).
- Scalar fields use overwrite semantics (no Annotated wrapper needed).
"""

from langgraph.graph import MessagesState


class SalesState(MessagesState):
    """
    The shared state for the sales agent graph.

    Attributes:
        messages: Chat history (inherited from MessagesState, uses add_messages reducer).
        lead_score: Integer from 1-100 representing the customer's interest level.
        pipeline_stage: Current stage in the sales pipeline.
            One of: "new", "qualified", "negotiation", "closed_won", "closed_lost".
        pending_discount_approval: Dictionary with discount details awaiting manager approval.
            Empty dict {} means no pending approval.
            Example: {"product": "Pro Plan", "original_price": 299, "discount_pct": 20, "reason": "High-value lead"}
        discount_approved: Whether the manager approved the pending discount.
        tool_iterations: Count of tool execution rounds in the current turn (ReAct loop guard).
    """

    lead_score: int
    pipeline_stage: str
    pending_discount_approval: dict
    discount_approved: bool
    tool_iterations: int
    # Order / fulfillment
    shipping_info: dict  # {customer_name, phone, address, postal_code, payment_method, ...}
    order_ready: bool  # True once save_shipping_info has captured a complete address
    # One-shot PromptPay QR generated this turn (for LINE image push)
    payment_qr: dict  # {amount, account_name, image_base64, use_static, ...} or {}
    # Overpayment handling
    pending_overpay: dict  # {overpaid_amount, total_amount, paid_amount}
    overpay_resolution: str  # "", "KEPT_AS_CREDIT", "PENDING_REFUND", "REFUNDED"
    overpay_credit_amount: float
    awaiting_refund_approval: bool
    # Long-conversation memory
    conversation_summary: str  # rolling summary of older messages (token saver)
    # One-shot live catalog snapshot injected before the agent runs this turn
    catalog_prefetch: str


def initial_state(**overrides) -> dict:
    """Return default scalar state for a new conversation thread."""
    defaults = {
        "lead_score": 0,
        "pipeline_stage": "new",
        "pending_discount_approval": {},
        "discount_approved": False,
        "tool_iterations": 0,
        "shipping_info": {},
        "order_ready": False,
        "payment_qr": {},
        "pending_overpay": {},
        "overpay_resolution": "",
        "overpay_credit_amount": 0.0,
        "awaiting_refund_approval": False,
        "conversation_summary": "",
        "catalog_prefetch": "",
    }
    defaults.update(overrides)
    return defaults
