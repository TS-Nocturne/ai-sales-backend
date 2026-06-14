"""
Pre-fetch live catalog context before the agent runs.

Runs synchronously in ``send_message`` so the model sees real inventory in the
system prompt on the first token — before it can hallucinate "หลายแบบให้เลือก".

Token budget: never inject the full catalog. Broad browse → categories only;
specific intents → at most PREFETCH_DISPLAY_LIMIT compact product rows.
"""

from __future__ import annotations

import re

from ai_sales.consts import PREFETCH_DISPLAY_LIMIT
from ai_sales.tools.catalog import get_product_catalog
from ai_sales.tools.sales_tools import (
    _catalog_browse_fallback,
    _catalog_categories,
    _format_category_overview,
    _format_product_compact,
    _is_broad_browse_query,
    _is_pure_catalog_browse,
    _search_in_memory,
)

# Customer turns that likely need product facts (Thai + common English).
_PRODUCT_INTENT = re.compile(
    r"(?:"
    r"เคส|case|ฟิล์|film|"
    r"สายชาร|charger|หูฟัง|earphone|headphone|"
    r"แบต|power\s?bank|"
    r"ซื้อ|สนใจ|มีไหม|มี.*บ้าง|มี.*อะไร.*บ้าง|มีอะไรขาย|ขายอะไร|"
    r"มีรุ่น|รุ่นไหน|ซื้ออะไรได้|"
    r"ราคา|งบ|แนะนำ|เลือก|"
    r"iphone|ไอโฟน|samsung|ซัมซung|"
    r"pro\s?max|promax"
    r")",
    re.IGNORECASE,
)

_CATEGORY_HINTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"เคส|case", re.I), "เคส"),
    (re.compile(r"ฟิล์|film|กระจก", re.I), "ฟilm"),
    (re.compile(r"สายชาร|charger|ชาร์จ", re.I), "สาย"),
    (re.compile(r"หูฟัง|earphone|headphone", re.I), "หูฟัง"),
    (re.compile(r"แบต|power\s?bank", re.I), "แบต"),
)


def _looks_like_product_turn(message: str, conversation_summary: str = "") -> bool:
    combined = f"{conversation_summary}\n{message}".strip()
    return bool(combined and _PRODUCT_INTENT.search(combined))


def _category_hint(text: str) -> str:
    for pattern, label in _CATEGORY_HINTS:
        if pattern.search(text):
            return label
    return ""


def _filter_by_category(products: list[dict], category_hint: str) -> list[dict]:
    if not category_hint:
        return products
    hint = category_hint.lower()
    filtered = []
    for product in products:
        searchable = (
            f"{product.get('name', '')} {product.get('category', '')} "
            f"{product.get('description', '')}"
        ).lower()
        if hint in searchable or str(product.get("category", "")).lower() == hint:
            filtered.append(product)
    return filtered


def _iphone_model_hint(text: str) -> str:
    match = re.search(
        r"iphone\s*\d+\s*(?:pro\s*max|pro|plus|max)?",
        text,
        re.IGNORECASE,
    )
    if not match:
        return ""
    return re.sub(r"\s+", " ", match.group(0).lower()).strip()


def _normalize_name(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").lower()).strip()


def _filter_by_model(products: list[dict], text: str) -> list[dict]:
    model = _iphone_model_hint(text)
    if not model:
        return products
    compact_model = model.replace(" ", "")
    filtered = []
    for product in products:
        name = _normalize_name(str(product.get("name", "")))
        if compact_model in name.replace(" ", ""):
            filtered.append(product)
            continue
        tokens = [t for t in model.split() if t != "iphone"]
        if tokens and all(token in name for token in tokens) and "iphone" in name:
            filtered.append(product)
    return filtered


def _resolve_prefetch_results(message: str, summary: str) -> list[dict]:
    query = message
    if summary:
        query = f"{message} {summary[-300:]}"

    combined = f"{message} {summary}".strip()
    category = _category_hint(combined)

    results = _search_in_memory(query, limit=PREFETCH_DISPLAY_LIMIT)
    if category:
        if not results:
            catalog = get_product_catalog()
            results = _filter_by_category(catalog, category)[:PREFETCH_DISPLAY_LIMIT]
    elif not results and _is_broad_browse_query(message):
        if _is_pure_catalog_browse(message):
            return []
        results = [
            {k: v for k, v in p.items() if k not in ("source_type", "score")}
            for p in _catalog_browse_fallback(limit=PREFETCH_DISPLAY_LIMIT)
        ]

    model_hint = _iphone_model_hint(combined)
    if model_hint:
        model_matches = _filter_by_model(results, combined)
        if model_matches:
            results = model_matches[:PREFETCH_DISPLAY_LIMIT]
        elif category:
            return []

    return results[:PREFETCH_DISPLAY_LIMIT]


def _format_prefetch(results: list[dict], query_label: str) -> str:
    lines = [
        "[ข้อมูลสินค้าจากระบบ ณ เวลานี้ — อ้างอิงเท่านี้เท่านั้น ห้ามเดา]",
        f"ค้นหาจาก: {query_label}",
        f"พบ {len(results)} รายการ:",
    ]
    for product in results:
        lines.append(_format_product_compact(product))

    if len(results) == 1:
        lines.append(
            "คำสั่ง: มีแค่ 1 รายการ — แนะนำรายการนี้ตรงๆ "
            "ห้ามบอกว่ามีหลายแบบ/หลายรุ่นให้เลือก และห้ามถามว่าต้องการแบบไหน"
        )
    else:
        lines.append(
            f"คำสั่ง: มี {len(results)} รายการ — แจกแจงตามรายการด้านบนเท่านั้น "
            "ห้ามอ้างว่ามีมากกว่าที่ระบุ ตอบกระชับเหมาะกับ LINE"
        )
    return "\n".join(lines)


def build_catalog_prefetch(message: str, conversation_summary: str = "") -> str:
    """Return a catalog snapshot for the system prompt, or empty if not needed."""
    message = (message or "").strip()
    summary = (conversation_summary or "").strip()
    if not _looks_like_product_turn(message, summary):
        return ""

    if _is_pure_catalog_browse(message):
        categories = _catalog_categories()
        if categories:
            return _format_category_overview(categories)

    results = _resolve_prefetch_results(message, summary)
    catalog_size = len(get_product_catalog())
    if not results and catalog_size > 0:
        if _is_broad_browse_query(message):
            results = [
                {k: v for k, v in p.items() if k not in ("source_type", "score")}
                for p in _catalog_browse_fallback(limit=PREFETCH_DISPLAY_LIMIT)
            ]
    if not results:
        return (
            "[ข้อมูลสินค้าจากระบบ ณ เวลานี้ — อ้างอิงเท่านี้เท่านั้น ห้ามเดา]\n"
            f"ค้นหาจาก: {message}\n"
            f"ไม่พบสินค้าที่ตรงในร้าน ({catalog_size} รายการทั้งหมดในระบบ)\n"
            "คำสั่ง: ห้ามบอกว่ามีหลายแบบให้เลือก — แจ้งตรงๆ ว่ายังไม่มีของที่ตรง "
            "แล้วถามรุ่นมือถือ/ความต้องการเพิ่มถ้าจำเป็น"
        )

    return _format_prefetch(results, message)
