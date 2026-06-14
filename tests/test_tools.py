"""Tests for sales tools."""

from ai_sales.tools.sales_tools import calculate_discount, list_products, search_knowledge_base


def test_search_knowledge_base_finds_iphone_case():
    result = search_knowledge_base.invoke({"query": "iPhone 15"})
    assert "iPhone 15" in result or "P001" in result


def test_search_knowledge_base_no_match(monkeypatch):
    import ai_sales.tools.sales_tools as sales_tools

    monkeypatch.setattr(sales_tools, "_search_pinecone", lambda query, top_k=5: None)
    result = search_knowledge_base.invoke({"query": "nonexistent-xyz-12345"})
    assert "No relevant information found" in result


def test_normalize_search_query_rewrites_vague_phrases():
    import ai_sales.tools.sales_tools as sales_tools

    assert sales_tools._normalize_search_query("มีรุ่นใหนแนะนำบ้าง") == "สินค้าแนะนำ"
    assert sales_tools._normalize_search_query("รุ่น") == "สินค้าแนะนำ"
    assert sales_tools._normalize_search_query("เคส iPhone 15") == "เคส iPhone 15"


def test_list_products_budget_ceiling_not_exact_price():
    """งบ 30,000 must not be treated as exact price (min=max bug)."""
    result = list_products.invoke(
        {
            "category": "สาย",
            "keyword": "",
            "min_price": 30000,
            "max_price": 30000,
            "sort_by_price": "",
        }
    )
    assert "ไม่พบสินค้า" not in result
    assert "บาท" in result


def test_calculate_discount_valid():
    result = calculate_discount.invoke(
        {
            "product_name": "เคส iPhone 15",
            "original_price": 490.0,
            "discount_percent": 20.0,
        }
    )
    assert "392.00 THB" in result
    assert "20.0%" in result


def test_calculate_discount_exceeds_max():
    result = calculate_discount.invoke(
        {
            "product_name": "เคส iPhone 15",
            "original_price": 490.0,
            "discount_percent": 60.0,
        }
    )
    assert "Error" in result


def test_calculate_discount_negative_price():
    result = calculate_discount.invoke(
        {
            "product_name": "เคส iPhone 15",
            "original_price": -10.0,
            "discount_percent": 10.0,
        }
    )
    assert "Error" in result
