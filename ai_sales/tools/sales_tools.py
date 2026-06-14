"""
Tool definitions for the AI Sales Agent.

Follows AGENTS.md guidelines:
- Decorate custom tools with @tool.
- Always write a descriptive docstring so the LLM understands its exact purpose.
"""

import logging
from typing import Annotated, Optional

import re

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field

from ai_sales.consts import MAX_DISCOUNT_PERCENT
from ai_sales.channels import line_delivery
from ai_sales.payments import qr as payment_qr
from ai_sales.payments import slip2go
from ai_sales.tools.catalog import get_product_catalog
from ai_sales.tools.order_total import resolve_order_from_messages

logger = logging.getLogger(__name__)

_CATALOG_FALLBACK_LIMIT = 3

_BROAD_BROWSE_QUERIES = frozenset(
    {
        "สินค้าแนะนำ",
        "สินค้าทั้งหมด",
        "สินค้ายอดนิยม",
        "อุปกรณ์เสริมมือถือ",
    }
)

_BROAD_BROWSE_RE = re.compile(
    r"มี(?:อะไร|สินค้า).*(?:ขาย|บ้าง)|มีอะไรขาย|ขายอะไร|"
    r"แนะนำ(?:สินค้า)?.*(?:หน่อย|บ้าง)|ซื้ออะไรได้|มีรุ่น(?:ไหน|ใหน)",
    re.IGNORECASE,
)


class SearchKnowledgeBaseInput(BaseModel):
    """Structured args so the LLM separates product keywords from budget ceiling."""

    query: str = Field(
        description=(
            "ชื่อ ประเภท รุ่น หรือหัวข้อที่ต้องการค้นหา "
            "(เช่น 'สายชาร์จ', 'เคส iPhone 15', 'นโยบายคืนเงิน') "
            "ห้ามใส่ตัวเลขราคา/งบประมาณในช่องนี้"
        )
    )
    max_price: Optional[float] = Field(
        default=None,
        description=(
            "งบประมาณสูงสุดของลูกค้า (ราคาไม่เกิน) เป็นบาท — "
            "ใส่เมื่อลูกค้าบอกงบ เช่น 30000 หรือ 'ในงบ 3000' "
            "ไม่ใช่ราคาเป๊ะๆ สายชาร์จ 350 บาทอยู่ในงบ 30000 ได้"
        ),
    )


class ListProductsInput(BaseModel):
    """Structured args for exact catalog filtering — budget goes in max_price only."""

    category: str = Field(
        default="",
        description=(
            "หมวดสินค้า เช่น 'เคส', 'สายชาร์จ', 'หูฟัง' — "
            "ว่างถ้าต้องการดูทั้งร้าน"
        ),
    )
    keyword: str = Field(
        default="",
        description="คีย์เวิร์ดเพิ่มในชื่อ/รายละเอียด เช่น 'iPhone 15' — ว่างถ้าไม่กรอง",
    )
    min_price: float = Field(
        default=0,
        description=(
            "ราคาขั้นต่ำ (บาท) — ใช้เฉพาะเมื่อลูกค้าขอของราคาตั้งแต่ X ขึ้นไป "
            "ห้ามใส่งบประมาณตรงนี้"
        ),
    )
    max_price: float = Field(
        default=0,
        description=(
            "งบประมาณสูงสุด (ราคาไม่เกิน) เป็นบาท — "
            "ใส่เมื่อลูกค้าบอกงบ เช่น 3000 หรือ 30000"
        ),
    )
    sort_by_price: str = Field(
        default="",
        description="'desc' = แพงสุดก่อน, 'asc' = ถูกสุดก่อน, ว่าง = ลำดับในคลัง",
    )


class CalculateDiscountInput(BaseModel):
    product_name: str = Field(description="ชื่อสินค้าที่กำลังให้ส่วนลด")
    original_price: float = Field(description="ราคาปกติก่อนหักส่วนลด (บาท)")
    discount_percent: float = Field(
        description="เปอร์เซ็นต์ส่วนลด (0–100) ส่วนลดเกิน 15% ต้องรอผู้จัดการ"
    )


def transfer_payment_qr_update(state: dict) -> dict:
    """Create PromptPay QR + a customer reply without relying on the LLM.

    Used when the customer chooses bank transfer (e.g. "โอนเงินครับ") after a
    price was already agreed in the conversation.
    """
    messages = state.get("messages", [])
    resolved = resolve_order_from_messages(messages)
    if not resolved:
        return {
            "messages": [
                AIMessage(
                    content=(
                        "รับทราบค่ะที่ต้องการโอนเงิน 🙏 "
                        "รบกวนแจ้งชื่อสินค้าหรือรุ่นมือถือที่สนใจอีกครั้ง "
                        "เดี๋ยวสรุปยอดและส่ง QR ให้ทันทีค่ะ"
                    )
                )
            ]
        }

    amount = resolved["amount"]
    items = resolved.get("items", "สินค้าที่สั่ง")
    try:
        qr_data = payment_qr.create_promptpay_qr(amount, items, partial=False)
    except slip2go.Slip2GoError as exc:
        return {
            "messages": [
                AIMessage(
                    content=(
                        f"ขออภัยค่ะ ระบบสร้าง QR ไม่สำเร็จ ({exc}) "
                        "รบกวนลองใหม่หรือแจ้งเจ้าหน้าที่ช่วยดูให้นะคะ 🙏"
                    )
                )
            ]
        }

    amount_text = f"{qr_data['amount']:,.2f}".rstrip("0").rstrip(".")
    item_note = ""
    if items and items not in ("สินค้าที่สั่ง", ""):
        item_note = f" ({items})"
    reply = (
        f"รับทราบค่ะ ยอดโอน {amount_text} บาท{item_note}\n"
        "ส่ง QR PromptPay ให้แล้ว กรุณาสแกนโอนและส่งสลิปกลับมาในแชทนะคะ 🙏"
    )
    reply = line_delivery.embed_line_qr_tag(reply, qr_data)
    return {"payment_qr": qr_data, "messages": [AIMessage(content=reply)]}


def _search_pinecone(query: str, top_k: int = 5) -> list[dict] | None:
    """Perform semantic search on Pinecone. Returns list of metadata dicts or None on failure."""
    try:
        from ai_sales.config.vectorstore import get_embeddings, get_pinecone_index

        embeddings = get_embeddings()
        index = get_pinecone_index()

        query_vector = embeddings.embed_query(query)
        results = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
        )

        if not results.get("matches"):
            return None

        return [
            {
                "id": match["id"],
                "score": match["score"],
                **match.get("metadata", {}),
            }
            for match in results["matches"]
            if match["score"] > 0.3  # minimum relevance threshold
        ]
    except Exception:
        return None


def _tokenize_query(query: str) -> list[str]:
    """Split a search query into meaningful tokens (supports Thai + Latin)."""
    query_lower = query.lower().strip()
    if not query_lower:
        return []
    tokens = [t for t in re.split(r"[\s,./\-_]+", query_lower) if len(t) >= 2]
    return tokens or [query_lower]


_VAGUE_QUERY_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"มีรุ่น.*(?:แนะนำ|บ้าง)|(?:แนะนำ|บ้าง).*รุ่น|มีรุ่นไหน|รุ่นไหนบ้าง",
            re.I,
        ),
        "สินค้าแนะนำ",
    ),
    (
        re.compile(
            r"แนะนำ.*(?:หน่อย|บ้าง)|มีอะไร(?:ขาย|บ้าง)|มีสินค้าอะไร|ขายอะไร",
            re.I,
        ),
        "สินค้าแนะนำ",
    ),
    (
        re.compile(r"ซื้ออะไรได้|งบ.*ซื้อ|ในงบ", re.I),
        "สินค้าแนะนำ",
    ),
)

_META_ONLY_WORDS = frozenset(
    {
        "รุ่น",
        "แนะนำ",
        "บ้าง",
        "หน่อย",
        "มี",
        "อะไร",
        "ขาย",
        "สินค้า",
        "ให้",
        "ครับ",
        "ค่ะ",
        "นะ",
        "คะ",
        "ได้",
        "ซื้อ",
        "งบ",
        "ไหน",
        "ใหน",
    }
)


def _normalize_search_query(query: str) -> str:
    """Rewrite vague/meta customer phrases into concrete vector-search keywords."""
    q = (query or "").strip()
    if not q:
        return q
    for pattern, replacement in _VAGUE_QUERY_PATTERNS:
        if pattern.search(q):
            return replacement
    tokens = _tokenize_query(q)
    if tokens and all(token in _META_ONLY_WORDS for token in tokens):
        return "สินค้าแนะนำ"
    if len(q) <= 12 and tokens == ["รุ่น"]:
        return "สินค้าแนะนำ"
    return q


def _is_broad_browse_query(query: str, *, raw_query: str | None = None) -> bool:
    """True when the query is a vague browse intent (not a specific product lookup)."""
    for candidate in (raw_query, query):
        if not candidate or not str(candidate).strip():
            continue
        text = str(candidate).strip()
        if _BROAD_BROWSE_RE.search(text):
            return True
        normalized = _normalize_search_query(text)
        compact = re.sub(r"\s+", "", normalized.lower())
        if any(
            re.sub(r"\s+", "", key.lower()) == compact for key in _BROAD_BROWSE_QUERIES
        ):
            return True
    return False


def _product_hits(results: list[dict]) -> list[dict]:
    """Keep only product rows from a mixed vector/FAQ result set."""
    return [r for r in results if r.get("source_type") == "product"]


def _format_catalog_fallback_response(
    picks: list[dict],
    query: str,
    budget_note: str = "",
    *,
    tag: str = "Catalog Fallback",
) -> str:
    header = (
        f"[{tag}] Showing {len(picks)} in-stock item(s) "
        f"for broad browse '{query}'{budget_note} "
        "(vector/keyword had no product match — using live catalog):\n\n"
    )
    return header + _format_search_results(picks)


def _catalog_browse_fallback(
    max_price: float | None = None, limit: int = _CATALOG_FALLBACK_LIMIT
) -> list[dict]:
    """Return in-stock catalog items when vector/keyword search has nothing for browse queries."""
    catalog = get_product_catalog()
    picks: list[dict] = []

    def _append(product: dict) -> None:
        if len(picks) >= limit:
            return
        price = float(product.get("price", 0) or 0)
        if max_price and max_price > 0 and price > max_price:
            return
        meta = product.copy()
        meta["source_type"] = "product"
        meta["score"] = 1.0
        picks.append(meta)

    for product in catalog:
        if int(product.get("stock", 0) or 0) > 0:
            _append(product)

    if not picks:
        for product in catalog:
            _append(product)

    return picks


def _filter_results_by_max_price(
    results: list[dict], max_price: float | None
) -> list[dict]:
    """Keep products at or below max_price; non-product hits (FAQ) pass through."""
    if not max_price or max_price <= 0:
        return results
    filtered: list[dict] = []
    for item in results:
        if item.get("source_type") == "product":
            price = float(item.get("price", 0) or 0)
            if price > max_price:
                continue
        filtered.append(item)
    return filtered


def _search_in_memory(query: str) -> list[dict]:
    """Fallback: keyword search against the in-memory product catalog only."""
    query_lower = query.lower().strip()
    tokens = _tokenize_query(query)
    results = []
    catalog = get_product_catalog()
    for product in catalog:
        searchable = (
            f"{product['name']} {product['category']} "
            f"{product['description']}"
        ).lower()
        matched = query_lower in searchable or any(
            token in searchable for token in tokens
        )
        if matched:
            product_meta = product.copy()
            product_meta["source_type"] = "product"
            results.append(product_meta)
    return results


def _format_search_results(results: list[dict]) -> str:
    """Format search results based on source_type (product or faq)."""
    lines = []
    for r in results:
        source_type = r.get("source_type", "unknown")
        score_pct = r.get("score", 0) * 100
        
        if source_type == "product":
            in_stock = "Yes" if int(r.get("stock", 0)) > 0 else "No"
            lines.append(
                f"[Product] {r.get('name', 'N/A')} (ID: {r.get('id', 'N/A')})\n"
                f"  Relevance: {score_pct:.0f}%\n"
                f"  Price: {r.get('price', 0)} THB | "
                f"Category: {r.get('category', 'N/A')}\n"
                f"  Description: {r.get('description', 'N/A')}\n"
                f"  Warranty: {r.get('warranty', 'N/A')}\n"
                f"  In Stock: {in_stock}"
            )
        elif source_type == "faq":
            lines.append(
                f"[FAQ] Source: {r.get('source', 'FAQ Document')} (Page {r.get('page', 'N/A')})\n"
                f"  Relevance: {score_pct:.0f}%\n"
                f"  Content: {r.get('text', 'N/A')}"
            )
        elif source_type == "knowledge":
            lines.append(
                f"[Knowledge] {r.get('title', 'เอกสารฐานความรู้')}\n"
                f"  Relevance: {score_pct:.0f}%\n"
                f"  Content: {r.get('text', 'N/A')}"
            )
        else:
            lines.append(f"[Unknown] {r.get('text', 'N/A')}")
            
    return "\n\n".join(lines)


@tool(args_schema=SearchKnowledgeBaseInput)
def search_knowledge_base(query: str, max_price: float | None = None) -> str:
    """Search the knowledge base for products, pricing, features, or FAQ answers.

    Use this tool when the customer asks about available products, pricing,
    policies, shipping, warranties, returns, or any general questions.
    Pass product/category keywords in `query` and budget (if any) in `max_price`
    — never combine them into one string like "สายชาร์จ 30000".

    CRITICAL — extract keywords before calling:
    Rewrite the customer's message into search keywords before passing `query`.
    Strip polite fillers and meta-requests (แนะนำ, ช่วยหา, มีไหม, ให้หน่อย,
    บ้างครับ, etc.) and keep only product names, phone models, categories, or
    policy topics. Put budget figures only in `max_price`.

    Examples (customer message → tool args):
        "งบ 30000 อยากได้สายชาร์จ"  → query="สายชาร์จ", max_price=30000
        "แนะนำสินค้าให้หน่อย"         → query="สินค้าแนะนำ"
        "มีรุ่นไหนแนะนำบ้าง"          → query="สินค้าแนะนำ" (ห้ามส่งแค่ "รุ่น")
        "เคส iPhone 15 มีไหม"         → query="เคส iPhone 15"
        "นโยบายการคืนเงิน"           → query="นโยบายคืนเงิน"

    Wrong — never do this:
        query="สายชาร์จ 30000" หรือ query="รุ่น" หรือ query="แนะนำหน่อย"

    For open-ended browse with no category, use query="สินค้าแนะนำ" or
    call list_products() with empty filters. If vector search finds nothing for
    a broad browse query, the tool automatically shows up to 3 in-stock items
    from the live catalog (Catalog Fallback).

    Returns:
        A formatted string listing relevant products and/or FAQ excerpts.
    """
    raw_query = query
    query = _normalize_search_query(query)
    budget_note = ""
    budget_cap: float | None = None
    if max_price and max_price > 0:
        budget_note = f" (งบไม่เกิน {max_price:,.0f} บาท)"
        budget_cap = max_price

    broad = _is_broad_browse_query(query, raw_query=raw_query)
    logger.info(
        "search_knowledge_base start raw=%r query=%r max_price=%s broad=%s",
        raw_query,
        query,
        max_price,
        broad,
    )

    pinecone_results = _search_pinecone(query)

    if pinecone_results:
        pinecone_results = _filter_results_by_max_price(pinecone_results, budget_cap)
        products = _product_hits(pinecone_results)
        if products:
            logger.info("search_knowledge_base hit vector products=%s", len(products))
            header = (
                f"[Vector Search] Found {len(products)} result(s) "
                f"matching '{query}'{budget_note}:\n\n"
            )
            return header + _format_search_results(products)
        if not broad and pinecone_results:
            logger.info(
                "search_knowledge_base hit vector non-product count=%s",
                len(pinecone_results),
            )
            header = (
                f"[Vector Search] Found {len(pinecone_results)} result(s) "
                f"matching '{query}'{budget_note}:\n\n"
            )
            return header + _format_search_results(pinecone_results)

    memory_results = _search_in_memory(query)
    memory_results = _filter_results_by_max_price(memory_results, budget_cap)

    if memory_results:
        logger.info(
            "search_knowledge_base hit keyword count=%s", len(memory_results)
        )
        header = (
            f"[Keyword Fallback] Found {len(memory_results)} product(s) "
            f"matching '{query}'{budget_note} (FAQ not available offline):\n\n"
        )
        return header + _format_search_results(memory_results)

    if broad:
        fallback = _catalog_browse_fallback(budget_cap)
        if fallback:
            logger.info(
                "search_knowledge_base catalog fallback count=%s query=%r",
                len(fallback),
                query,
            )
            return _format_catalog_fallback_response(fallback, query, budget_note)

    logger.info("search_knowledge_base miss query=%r max_price=%s", query, max_price)
    if max_price and max_price > 0:
        return (
            f"No relevant information found for '{query}' within budget "
            f"{max_price:,.0f} THB. Try a broader category or higher budget."
        )
    return (
        f"No relevant information found for '{query}'. "
        "Try searching with broader terms or different keywords."
    )


@tool(args_schema=CalculateDiscountInput)
def calculate_discount(
    product_name: str, original_price: float, discount_percent: float
) -> str:
    """Calculate the final price after applying a discount percentage.

    Use this tool when the customer is negotiating price or when you need
    to propose a discount to close the deal. This tool computes the exact
    savings and final price.

    Args:
        product_name: The name of the product being discounted.
        original_price: The original price in THB before discount.
        discount_percent: The discount percentage to apply (0-100).

    Returns:
        A formatted string showing the original price, discount amount,
        and final price after discount.
    """
    if discount_percent < 0 or discount_percent > MAX_DISCOUNT_PERCENT:
        return (
            f"Error: discount must be between 0 and "
            f"{MAX_DISCOUNT_PERCENT}%. Discounts above 15% require manager approval."
        )

    if original_price <= 0:
        return "Error: original price must be a positive number."

    discount_amount = original_price * (discount_percent / 100)
    final_price = original_price - discount_amount

    return (
        f"[Discount Calculation] {product_name}:\n"
        f"  Original Price:  {original_price:.2f} THB\n"
        f"  Discount:        {discount_percent:.1f}% (-{discount_amount:.2f} THB)\n"
        f"  Final Price:     {final_price:.2f} THB\n"
        f"  Customer Saves:  {discount_amount:.2f} THB"
    )


def _normalize_sort(value: str) -> str:
    v = (value or "").strip().lower()
    if v in ("desc", "high", "expensive", "max", "แพง", "แพงสุด"):
        return "desc"
    if v in ("asc", "low", "cheap", "min", "ถูก", "ถูกสุด"):
        return "asc"
    return ""


@tool(args_schema=ListProductsInput)
def list_products(
    category: str = "",
    keyword: str = "",
    min_price: float = 0,
    max_price: float = 0,
    sort_by_price: str = "",
) -> str:
    """List products from the shop catalog with exact filtering and price sorting.

    Use this tool (NOT search_knowledge_base) whenever the customer asks about
    price ranges, rankings, or budgets — e.g. "หูฟังแพงที่สุด", "ของถูกสุดในร้าน",
    "มีอะไรในงบ 3000 บาท", "มีเคสรุ่นไหนบ้าง", "หูฟังมีกี่รุ่น".
    Put the product/category in `category` or `keyword`; put budget only in
    `max_price` — never in both min and max for the same budget figure.

    Returns:
        A formatted list of matching products (name, price, category, stock,
        description), or a clear message that no product matches the filters.
    """
    catalog = get_product_catalog()
    cat = category.strip().lower()
    kw = keyword.strip().lower()

    # LLM often sets min_price == max_price for "งบ X" — budget is a ceiling only.
    if min_price > 0 and max_price > 0 and min_price == max_price:
        min_price = 0

    matches = []
    for product in catalog:
        name_l = str(product.get("name", "")).lower()
        cat_l = str(product.get("category", "")).lower()
        desc_l = str(product.get("description", "")).lower()
        price = float(product.get("price", 0))

        if cat and cat not in cat_l and cat not in name_l:
            continue
        if kw and kw not in name_l and kw not in desc_l:
            continue
        if min_price and price < min_price:
            continue
        if max_price and price > max_price:
            continue
        matches.append(product)

    if not matches:
        if not cat and not kw:
            fallback = _catalog_browse_fallback(
                max_price if max_price > 0 else None
            )
            if fallback:
                logger.info(
                    "list_products catalog fallback count=%s cat=%r kw=%r",
                    len(fallback),
                    category,
                    keyword,
                )
                lines = [
                    "[Catalog Fallback] แสดงสินค้าแนะนำจากคลังสด "
                    f"({len(fallback)} รายการ — vector/keyword ไม่ตรง):"
                ]
                for p in fallback:
                    in_stock = (
                        "มีสินค้า" if int(p.get("stock", 0)) > 0 else "สินค้าหมด"
                    )
                    lines.append(
                        f"- {p.get('name', 'N/A')} | ราคา {float(p.get('price', 0)):.0f} บาท "
                        f"| หมวด: {p.get('category', 'N/A')} | {in_stock}\n"
                        f"  รายละเอียด: {p.get('description', 'N/A')}"
                    )
                return "\n".join(lines)
        return (
            "[Catalog] ไม่พบสินค้าที่ตรงกับเงื่อนไขนี้ในร้าน "
            f"(category='{category}', keyword='{keyword}', "
            f"min_price={min_price}, max_price={max_price}). "
            "อย่าแต่งสินค้าขึ้นมาเอง — ให้แจ้งลูกค้าตามจริงว่าไม่มี "
            "และเสนอหมวดสินค้าที่ร้านมีแทน"
        )

    order = _normalize_sort(sort_by_price)
    if order == "desc":
        matches.sort(key=lambda p: float(p.get("price", 0)), reverse=True)
    elif order == "asc":
        matches.sort(key=lambda p: float(p.get("price", 0)))

    lines = [f"[Catalog] พบ {len(matches)} รายการ:"]
    for p in matches:
        in_stock = "มีสินค้า" if int(p.get("stock", 0)) > 0 else "สินค้าหมด"
        lines.append(
            f"- {p.get('name', 'N/A')} | ราคา {float(p.get('price', 0)):.0f} บาท "
            f"| หมวด: {p.get('category', 'N/A')} | {in_stock}\n"
            f"  รายละเอียด: {p.get('description', 'N/A')}"
        )
    return "\n".join(lines)


_VALID_PAYMENT_METHODS = ("TRANSFER", "COD")


@tool
def save_shipping_info(
    customer_name: str,
    phone: str,
    address: str,
    postal_code: str,
    payment_method: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    amount: float = 0,
    items: str = "",
) -> Command:
    """บันทึกข้อมูลจัดส่งของลูกค้าเพื่อเปิดออเดอร์ให้พนักงานแพ็คของ.

    เรียกใช้เครื่องมือนี้ "หลังจาก" ที่ลูกค้าชำระเงินเรียบร้อย (โอนเงิน/สลิปผ่าน
    การตรวจสอบ หรือ เลือกเก็บเงินปลายทาง COD) และคุณเก็บข้อมูลจัดส่งครบแล้ว
    เท่านั้น ห้ามเรียกก่อนปิดการขายหรือก่อนได้ที่อยู่ครบ

    Args:
        customer_name: ชื่อ-นามสกุลผู้รับ (ไม่ต้องมีคำนำหน้า).
        phone: เบอร์โทรผู้รับ (ตัวเลขเท่านั้น).
        address: ที่อยู่จัดส่งแบบเต็ม.
        postal_code: รหัสไปรษณีย์ 5 หลัก (ถ้าทราบ).
        payment_method: วิธีชำระเงิน "TRANSFER" (โอนเงิน) หรือ "COD" (เก็บปลายทาง).
        amount: ยอดเงินรวมของออเดอร์ (บาท) ถ้าทราบ.
        items: สรุปรายการสินค้าที่ลูกค้าสั่ง (ถ้าทราบ).

    Returns:
        อัปเดต state ให้พร้อมเปิดออเดอร์ (order_ready=True).
    """
    method = (payment_method or "TRANSFER").strip().upper()
    if method not in _VALID_PAYMENT_METHODS:
        method = "TRANSFER"

    shipping = {
        "customer_name": customer_name.strip(),
        "phone": phone.strip(),
        "address": address.strip(),
        "postal_code": (postal_code or "").strip(),
        "payment_method": method,
        "amount": amount or 0,
        "items": (items or "").strip(),
    }

    confirm = (
        "[บันทึกข้อมูลจัดส่งแล้ว]\n"
        f"  ผู้รับ: {shipping['customer_name']}\n"
        f"  โทร: {shipping['phone']}\n"
        f"  ที่อยู่: {shipping['address']} {shipping['postal_code']}\n"
        f"  ชำระเงิน: {method}"
    )

    return Command(
        update={
            "shipping_info": shipping,
            "order_ready": True,
            "messages": [ToolMessage(content=confirm, tool_call_id=tool_call_id)],
        }
    )


@tool
def generate_promptpay_qr(
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
    amount: float = 0,
    items: str = "",
    partial: bool = False,
) -> Command:
    """สร้าง QR PromptPay สำหรับให้ลูกค้าโอนเงิน.

    เรียกเมื่อลูกค้าแจ้งว่าจะโอนเงิน (เช่น "โอนเงินครับ") และมีสินค้า+ราคาที่
    ตกลงแล้วในบทสนทนา — ไม่ต้องถามยอดซ้ำในกรณีนั้น
    ถ้า amount เป็น 0 หรือไม่ระบุ ระบบจะสรุปยอดจากบทสนทนาก่อนหน้าอัตโนมัติ
    (ราคาหลังส่วนลด / ราคาจากเครื่องมือค้นหา / ราคาที่เคยแจ้งลูกค้าแล้ว)

    ห้ามเรียกถ้า:
    - ลูกค้าเลือก COD (เก็บปลายทาง)
    - ยังไม่เคยคุยสินค้า/ราคาที่ตกลงซื้อเลย
    - มีแค่ตัวเลข "งบประมาณ" (เช่น งบ 30,000) โดยยังไม่ได้เลือกสินค้า —
      งบประมาณไม่ใช่ยอดโอน ต้องค้นหาสินค้าและแจ้งราคาก่อน

    ระบบจะสร้าง QR ผ่าน Slip2Go แล้วส่งรูป QR ให้ลูกค้าทาง LINE อัตโนมัติ
    (แจ้งลูกค้าว่าส่ง QR แล้ว และขอให้ส่งสลิปหลังโอน)

    Args:
        amount: ยอดเงินรวม (บาท). ใส่ 0 เพื่อให้ระบบสรุปจากบทสนทนา.
        items: สรุปรายการสินค้า (ไม่บังคับ) เช่น "เคส iPhone 15 Pro Max x1".

    Returns:
        อัปเดต state ด้วย payment_qr พร้อมรูป QR สำหรับช่องทาง LINE.
    """
    resolved = None
    if amount <= 0:
        resolved = resolve_order_from_messages(state.get("messages", []))
        if not resolved:
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=(
                                "[สรุปยอดไม่ได้] ยังไม่พบสินค้า/ราคาที่ตกลงในบทสนทนา "
                                "ให้ค้นหาสินค้าและแจ้งราคาให้ลูกค้าก่อน แล้วค่อยสร้าง QR"
                            ),
                            tool_call_id=tool_call_id,
                        )
                    ]
                }
            )
        amount = resolved["amount"]
        if not items.strip():
            items = resolved.get("items", "")

    try:
        qr_data = payment_qr.create_promptpay_qr(amount, items, partial=partial)
    except slip2go.Slip2GoError as exc:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"[สร้าง QR ไม่สำเร็จ] {exc}",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    amount_text = f"{qr_data['amount']:,.2f}".rstrip("0").rstrip(".")
    confirm = (
        "[สร้าง QR PromptPay แล้ว]\n"
        f"  ยอดโอน: {amount_text} บาท\n"
        f"  ชื่อบัญชี: {qr_data.get('account_name', 'ร้านค้า')}\n"
        f"  รูป QR จะถูกส่งให้ลูกค้าทาง LINE อัตโนมัติ"
    )
    if items.strip():
        confirm += f"\n  รายการ: {items.strip()}"
    if resolved:
        confirm += f"\n  สรุปจาก: {resolved.get('source', 'conversation')}"
    confirm = line_delivery.embed_line_qr_tag(confirm, qr_data)

    return Command(
        update={
            "payment_qr": qr_data,
            "messages": [ToolMessage(content=confirm, tool_call_id=tool_call_id)],
        }
    )


@tool
def apply_overpay_as_store_credit(
    amount: float,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """บันทึกยอดโอนเกินเป็นเครดิตร้านเมื่อลูกค้าเลือกเก็บไว้ใช้ครั้งหน้า.

    เรียกเมื่อลูกค้าตอบว่าต้องการเก็บเป็นเครดิตหลังระบบแจ้งว่าโอนเกิน

    Args:
        amount: ยอดเงินที่โอนเกิน (บาท).
    """
    if amount <= 0:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="[เครดิตไม่สำเร็จ] ยอดเครดิตต้องมากกว่า 0",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    confirm = f"[บันทึกเครดิตร้านแล้ว] ยอด {amount:,.2f} บาท จะถูกเก็บเป็นเครดิตสำหรับซื้อครั้งหน้า"
    return Command(
        update={
            "overpay_resolution": "KEPT_AS_CREDIT",
            "overpay_credit_amount": float(amount),
            "messages": [ToolMessage(content=confirm, tool_call_id=tool_call_id)],
        }
    )


@tool
def request_overpay_refund(
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
) -> Command:
    """ส่งเรื่องขอคืนเงินส่วนเกินให้ผู้จัดการเมื่อลูกค้าเลือกโอนคืน.

    เรียกเมื่อลูกค้าตอบว่าต้องการให้โอนเงินส่วนเกินคืน ระบบจะหยุดบอทชั่วคราว
    รอผู้จัดการดำเนินการคืนเงิน
    """
    pending = state.get("pending_overpay") or {}
    amount = float(pending.get("overpaid_amount") or 0)
    confirm = (
        "[ส่งเรื่องคืนเงินให้ผู้จัดการแล้ว]\n"
        f"  ยอดที่ต้องคืน: {amount:,.2f} บาท\n"
        "  รอผู้จัดการโอนคืนและยืนยันออเดอร์"
    )
    return Command(
        update={
            "overpay_resolution": "PENDING_REFUND",
            "awaiting_refund_approval": True,
            "messages": [ToolMessage(content=confirm, tool_call_id=tool_call_id)],
        }
    )


# List of all tools for binding to the LLM
all_tools = [
    search_knowledge_base,
    calculate_discount,
    list_products,
    generate_promptpay_qr,
    save_shipping_info,
    apply_overpay_as_store_credit,
    request_overpay_refund,
]
