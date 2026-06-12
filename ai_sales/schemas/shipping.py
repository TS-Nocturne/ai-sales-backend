"""
Structured-output schema for shipping/order capture.

The sales agent extracts these fields from the conversation (LangChain
tool-calling validates them against this Pydantic model — i.e. structured
output), then they are persisted as an Order by the Next.js layer.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ShippingInfo(BaseModel):
    """ข้อมูลจัดส่งของลูกค้า (สกัดจากบทสนทนาแบบมีโครงสร้าง)."""

    customer_name: str = Field(
        ..., description="ชื่อ-นามสกุลผู้รับสินค้า (ไม่ต้องมีคำนำหน้า)"
    )
    phone: str = Field(
        ..., description="เบอร์โทรศัพท์ผู้รับ (ตัวเลขเท่านั้น เช่น 0812345678)"
    )
    address: str = Field(
        ...,
        description=(
            "ที่อยู่จัดส่งแบบเต็ม (บ้านเลขที่ ซอย ถนน ตำบล/แขวง อำเภอ/เขต จังหวัด)"
        ),
    )
    postal_code: str = Field(
        default="", description="รหัสไปรษณีย์ 5 หลัก (ถ้าทราบ)"
    )
