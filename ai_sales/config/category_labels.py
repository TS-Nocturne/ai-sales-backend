"""
Fixed Thai labels for catalog category names.

Dashboard / CSV often store categories in English. Add new entries here when
inventory sync introduces a category — tools always show these Thai labels to
customers (no LLM translation step).
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# English (or mixed) category name → fixed Thai label for customer-facing output.
CATEGORY_TH_MAP: dict[str, str] = {
    # Sample catalog + common mobile-accessory shop categories
    "Accessories": "อุปกรณ์เสริม",
    "Adapter": "อะแดปเตอร์/ฮับ",
    "Audio": "หูฟัง",
    "Cable": "สายชาร์จ",
    "Car Accessories": "อุปกรณ์รถยนต์",
    "Case": "เคส",
    "Charger": "หัวชาร์จ",
    "Cleaning": "อุปกรณ์ทำความสะอาด",
    "Earphone": "หูฟัง",
    "Gaming": "อุปกรณ์เกมมิ่ง",
    "Headphone": "หูฟัง",
    "Holder": "ที่วาง/ขาตั้ง",
    "Hub": "ฮับ USB",
    "Keyboard": "คีย์บอร์ด",
    "Mount": "ที่ยึด/ขาตั้ง",
    "Power Bank": "แบตสำรอง",
    "Screen Protector": "ฟิล์มกันรอย",
    "Smartwatch": "สมาร์ทวอทช์",
    "Speaker": "ลำโพง",
    "Stylus": "สไตลัส",
    "Tablet": "แท็บเล็ต",
    "Watch": "นาฬิกา",
    "Watch Band": "สายนาฬิกา",
    # Thai keys pass through as-is
    "อื่นๆ": "อื่นๆ",
    "เคส": "เคส",
    "สายชาร์จ": "สายชาร์จ",
    "ฟิล์มกันรอย": "ฟิล์มกันรอย",
    "หูฟัง": "หูฟัง",
    "แบตสำรอง": "แบตสำรอง",
}

# Fallback when a new English category is not yet in the map (still Thai, fixed).
DEFAULT_CATEGORY_TH = "อุปกรณ์เสริม"

# Case-insensitive lookup built once from CATEGORY_TH_MAP.
_CATEGORY_TH_LOOKUP: dict[str, str] = {
    key.lower(): value for key, value in CATEGORY_TH_MAP.items()
}


def category_label(raw: str) -> str:
    """Return the fixed Thai label for a catalog category name."""
    key = (raw or "").strip() or "อื่นๆ"
    if key in CATEGORY_TH_MAP:
        return CATEGORY_TH_MAP[key]
    lowered = key.lower()
    if lowered in _CATEGORY_TH_LOOKUP:
        return _CATEGORY_TH_LOOKUP[lowered]
    if re.search(r"[\u0e00-\u0e7f]", key):
        return key
    logger.warning(
        "Unmapped category %r — add to CATEGORY_TH_MAP in category_labels.py; "
        "using fallback %r",
        key,
        DEFAULT_CATEGORY_TH,
    )
    return DEFAULT_CATEGORY_TH
