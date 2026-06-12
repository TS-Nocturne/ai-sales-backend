"""
Interactive CLI for the AI Sales Agent.

Supports multi-turn conversations with HITL discount approval.
"""

import sys
import uuid

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()

from ai_sales.consts import HUMAN_APPROVAL
from ai_sales.runtime import graph_runtime
from ai_sales.state import initial_state


def _normalize_content(content) -> str:
    """Convert message content to string (handles Gemini list responses)."""
    if not content:
        return ""
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                text_parts.append(item["text"])
            else:
                text_parts.append(str(item))
        return " ".join(text_parts)
    return str(content)


_INTERNAL_PREFIXES = (
    "[Lead Scoring Complete]",
    "[การให้คะแนนลีดเสร็จสมบูรณ์]",
    "[APPROVED]",
    "[REJECTED]",
    "[อนุมัติ]",
    "[ปฏิเสธ]",
)


def _is_internal_agent_message(msg: AIMessage) -> bool:
    """Return True for system/scoring/approval messages, not customer-facing."""
    text = _normalize_content(msg.content).strip()
    if not text:
        return bool(getattr(msg, "tool_calls", None))
    return any(text.startswith(prefix) for prefix in _INTERNAL_PREFIXES)


def _message_count(graph, thread: dict) -> int:
    """Return current message count in thread state."""
    state = graph.get_state(thread)
    return len(state.values.get("messages", []))


def _print_agent_reply(messages: list, since: int = 0) -> None:
    """Print the latest customer-facing agent response from this turn only.

    Only inspects messages added after `since` to avoid replaying old
    analysis when post_approval_node adds the final reply.
    """
    new_messages = messages[since:]

    for msg in reversed(new_messages):
        if not isinstance(msg, AIMessage):
            continue
        if _is_internal_agent_message(msg):
            continue
        text = _normalize_content(msg.content).strip()
        if not text:
            continue
        print(f"\n  เอเจนต์: {text}\n")
        return


def _print_status(graph, thread: dict) -> None:
    state = graph.get_state(thread)
    values = state.values
    print("\n  --- สถานะ ---")
    print(f"  คะแนนลีด:       {values.get('lead_score', 'N/A')}")
    print(f"  สถานะไปป์ไลน์:   {values.get('pipeline_stage', 'N/A')}")
    pending = values.get("pending_discount_approval", {})
    if pending:
        print(f"  รอการอนุมัติส่วนลด: {pending.get('discount_pct', 0)}% สำหรับ {pending.get('product', 'N/A')}")
    print(f"  โหนดถัดไป:       {state.next}")
    print()


def run_chat(thread_id: str | None = None) -> None:
    """Run an interactive sales chat session."""
    tid = thread_id or f"chat-{uuid.uuid4().hex[:8]}"
    thread = {"configurable": {"thread_id": tid}}

    print("=" * 60)
    print("  ตัวแทนฝ่ายขาย AI -- สนทนาโต้ตอบ")
    print(f"  รหัสการสนทนา (Thread ID): {tid}")
    print("  คำสั่ง: quit | status | approve | reject")
    print("=" * 60)

    first_turn = True

    with graph_runtime() as graph:
        while True:
            state = graph.get_state(thread)

            if state.next and HUMAN_APPROVAL in state.next:
                pending = state.values.get("pending_discount_approval", {})
                print("\n  [!] ต้องได้รับการอนุมัติจากผู้จัดการ!")
                print(f"      สินค้า:  {pending.get('product', 'N/A')}")
                print(f"      ส่วนลด: {pending.get('discount_pct', 0)}%")
                print(f"      เหตุผล:   {pending.get('reason', 'N/A')}")
                print("      พิมพ์ 'approve' หรือ 'reject' เพื่อทำต่อ\n")

            try:
                user_input = input("  คุณ: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  ลาก่อน!")
                break

            if not user_input:
                continue

            lower = user_input.lower()

            if lower in ("quit", "exit"):
                print("  ลาก่อน!")
                break

            if lower == "status":
                _print_status(graph, thread)
                continue

            if lower in ("approve", "reject") and state.next and HUMAN_APPROVAL in state.next:
                approved = lower == "approve"
                msg_count_before = _message_count(graph, thread)
                graph.update_state(thread, {"discount_approved": approved})
                result = graph.invoke(None, thread)
                _print_agent_reply(result.get("messages", []), since=msg_count_before)
                _print_status(graph, thread)
                continue

            payload: dict = {
                "messages": [HumanMessage(content=user_input)],
                "tool_iterations": 0,
            }
            if first_turn:
                payload.update(initial_state())
                first_turn = False

            msg_count_before = _message_count(graph, thread)
            result = graph.invoke(payload, thread)
            _print_agent_reply(result.get("messages", []), since=msg_count_before)

            state = graph.get_state(thread)
            if state.next and HUMAN_APPROVAL in state.next:
                pending = state.values.get("pending_discount_approval", {})
                print("\n  [!] กำลังรอการอนุมัติจากผู้จัดการ")
                print(f"      พิมพ์ 'approve' หรือ 'reject' (ส่วนลด: {pending.get('discount_pct', 0)}%)")


if __name__ == "__main__":
    run_chat()
