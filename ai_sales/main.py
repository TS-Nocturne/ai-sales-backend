"""
Demo script for the AI Sales & Lead Scoring Agent.

Demonstrates the full Human-in-the-Loop (HITL) workflow:
1. Invoke the graph with an initial customer message
2. Hit the interrupt (pause at human_approval_node)
3. Update the state as a human manager (using graph.update_state)
4. Resume the graph (graph.invoke(None, thread))

Follows AGENTS.md guidelines:
- Always include a thread_id when invoking a persisted graph.
- To resume from an interrupt, inject via graph.update_state() then graph.invoke(None, thread).
"""

import sys

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# AGENTS.md: Always run load_dotenv() at the top
load_dotenv()

from ai_sales.consts import HUMAN_APPROVAL
from ai_sales.runtime import graph_runtime
from ai_sales.state import initial_state


# ---------------------------------------------------------------------------
# Display Helpers (Windows cp874 encoding safe -- no emoji)
# ---------------------------------------------------------------------------
def print_separator(title: str = "") -> None:
    """Print a visual separator for readability."""
    print("\n" + "=" * 70)
    if title:
        print(f"  {title}")
        print("=" * 70)


def print_messages(messages: list) -> None:
    """Pretty-print message objects."""
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "[ลูกค้า]"
        elif isinstance(msg, AIMessage):
            role = "[เอเจนต์]"
        elif isinstance(msg, ToolMessage):
            role = f"[เครื่องมือ: {getattr(msg, 'name', 'unknown')}]"
        else:
            role = f"[{type(msg).__name__}]"

        # ---------------------------------------------------------
        # [ส่วนที่แก้ไข] จัดการ content ให้เป็น String เสมอก่อนนำไปใช้งานต่อ
        # ---------------------------------------------------------
        raw_content = msg.content
        if not raw_content:
            content = "[กำลังรอเรียกใช้เครื่องมือ]"
        elif isinstance(raw_content, list):
            # ดักจับกรณีที่ Gemini ส่งมาเป็น List ของ Dictionary
            text_parts = []
            for item in raw_content:
                if isinstance(item, dict) and "text" in item:
                    text_parts.append(item["text"])
                else:
                    text_parts.append(str(item))
            content = " ".join(text_parts)
        else:
            content = str(raw_content)
        # ---------------------------------------------------------

        # Truncate very long messages
        if len(content) > 500:
            content = content[:500] + "..."
            
        print(f"\n  {role}:")
        for line in content.split("\n"):
            print(f"    {line}")


def print_state_snapshot(graph, thread: dict) -> None:
    """Print the current state of the graph."""
    state = graph.get_state(thread)
    values = state.values

    print(f"\n  คะแนนลีด:          {values.get('lead_score', 'N/A')}")
    print(f"  สถานะไปป์ไลน์:      {values.get('pipeline_stage', 'N/A')}")
    print(f"  รอการอนุมัติส่วนลด:    {values.get('pending_discount_approval', {})}")
    print(f"  ส่วนลดที่อนุมัติ:   {values.get('discount_approved', 'N/A')}")
    print(f"  โหนดถัดไป:          {state.next}")


# ---------------------------------------------------------------------------
# Demo Runner
# ---------------------------------------------------------------------------
def run_demo():
    """Run the full demo of the AI Sales Agent with HITL."""

    print_separator("ตัวแทนฝ่ายขาย AI และการให้คะแนนลีด -- สาธิต")
    print("  กำลังสร้างกราฟพร้อมการบันทึกข้อมูล PostgresSaver (Neon)...")

    with graph_runtime() as graph:
        # AGENTS.md: Always include a thread_id when invoking a persisted graph
        thread = {"configurable": {"thread_id": "demo-customer-001"}}
        # ==================================================================
        # STEP 1: Customer asks about products
        # ==================================================================
        print_separator("ขั้นตอนที่ 1: ลูกค้าสอบถามข้อมูล")
        print("  กำลังส่ง: ลูกค้าถามเกี่ยวกับอุปกรณ์มือถือ...")

        result = graph.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "สวัสดีครับ มีเคส iPhone 15 Pro Max แบบใสๆ กันกระแทกไหมครับ? "
                            "แล้วก็อยากได้สายชาร์จถักด้วย งบประมาณไม่เกิน 1,000 บาทครับ"
                        )
                    )
                ],
                **initial_state(),
            },
            thread,
        )

        print("\n  --- ข้อความหลังจากขั้นตอนที่ 1 ---")
        print_messages(result.get("messages", [])[-4:])
        print_state_snapshot(graph, thread)

        # Check if we hit the interrupt
        state = graph.get_state(thread)
        if state.next:
            print(f"\n  >> กราฟหยุดพักที่: {state.next}")
        else:
            print("\n  >> ขั้นตอนที่ 1 เสร็จสมบูรณ์โดยไม่มีการหยุดพัก")

        # ==================================================================
        # STEP 2: Customer negotiates price (separate thread for clean demo)
        # ==================================================================
        thread_negotiate = {"configurable": {"thread_id": "demo-customer-negotiate"}}
        print_separator("ขั้นตอนที่ 2: ลูกค้าต่อรองราคา (Thread ใหม่)")
        print("  กำลังส่ง: ลูกค้าขอส่วนลด 20% สำหรับเคส iPad...")

        result = graph.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "สนใจเคส iPad Air 5 ราคา 790 บาทครับ แต่เห็นในร้านอื่นขายแค่ 600 บาทเอง "
                            "ลดให้ผม 20% ได้ไหมครับ จะโอนเงินให้ตอนนี้เลย"
                        )
                    )
                ],
                **initial_state(),
            },
            thread_negotiate,
        )

        print("\n  --- ข้อความหลังจากขั้นตอนที่ 2 ---")
        print_messages(result.get("messages", [])[-4:])
        print_state_snapshot(graph, thread_negotiate)

        state = graph.get_state(thread_negotiate)

        # ==================================================================
        # STEP 3: Handle the interrupt (HITL)
        # ==================================================================
        if state.next and HUMAN_APPROVAL in state.next:
            print_separator("ขั้นตอนที่ 3: หยุดพัก -- ต้องได้รับการอนุมัติจากผู้จัดการ")

            pending = state.values.get("pending_discount_approval", {})
            print(f"\n  [!] มีคำขออนุมัติส่วนลด!")
            print(f"      สินค้า:  {pending.get('product', 'N/A')}")
            print(f"      ส่วนลด: {pending.get('discount_pct', 0)}%")
            print(f"      เหตุผล:   {pending.get('reason', 'N/A')}")
            print(f"\n  ผู้จัดการกำลังพิจารณา...")

            # Manager injects decision; human_approval_node runs on resume
            print("\n  >> ผู้จัดการอนุมัติส่วนลด!")

            graph.update_state(thread_negotiate, {"discount_approved": True})

            print_separator("ขั้นตอนที่ 4: ทำงานต่อหลังจากได้รับการอนุมัติ")
            print("  กำลังเรียกใช้ graph.invoke(None, thread) เพื่อทำงานต่อ...")

            result = graph.invoke(None, thread_negotiate)

            print("\n  --- ข้อความหลังจากกลับมาทำงานต่อ ---")
            print_messages(result.get("messages", [])[-4:])
            print_state_snapshot(graph, thread_negotiate)
        else:
            print("\n  >> ไม่มีการขออนุมัติส่วนลดในรอบนี้")

        # ==================================================================
        # DEMO: Rejection Flow (New Thread)
        # ==================================================================
        print_separator("โบนัส: ขั้นตอนการปฏิเสธ (Thread ใหม่)")
        thread_reject = {"configurable": {"thread_id": "demo-customer-002"}}

        result = graph.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "ผมต้องการแพ็กเกจ Enterprise แต่ผมขอส่วนลด 40% "
                            "จะรับหรือไม่รับข้อเสนอนี้? คู่แข่งของคุณเสนอราคามาถูกกว่าคุณครึ่งนึง"
                        )
                    )
                ],
                **initial_state(),
            },
            thread_reject,
        )

        state_reject = graph.get_state(thread_reject)
        if state_reject.next and HUMAN_APPROVAL in state_reject.next:
            pending = state_reject.values.get("pending_discount_approval", {})
            print(f"\n  [!] มีคำขออนุมัติส่วนลด: {pending}")
            print("  >> ผู้จัดการปฏิเสธส่วนลด!")

            graph.update_state(thread_reject, {"discount_approved": False})

            result = graph.invoke(None, thread_reject)
            print("\n  --- ข้อความหลังจากถูกปฏิเสธ ---")
            print_messages(result.get("messages", [])[-3:])
            print_state_snapshot(graph, thread_reject)
        else:
            print("\n  >> ไม่มีการขออนุมัติส่วนลดในขั้นตอนการปฏิเสธ")

        print_separator("การสาธิตเสร็จสมบูรณ์")
        print("  แสดงทุกสถานการณ์สำเร็จแล้ว!")
        print("  ข้อมูล checkpoint ถูกบันทึกใน Neon PostgreSQL (PostgresSaver)")

        print("\n  ปิดการเชื่อมต่อฐานข้อมูลแล้ว")


if __name__ == "__main__":
    run_demo()