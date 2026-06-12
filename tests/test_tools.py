"""Tests for sales tools."""

from ai_sales.tools.sales_tools import calculate_discount, search_knowledge_base


def test_search_knowledge_base_finds_iphone_case():
    result = search_knowledge_base.invoke({"query": "iPhone 15"})
    assert "iPhone 15" in result or "P001" in result


def test_search_knowledge_base_no_match(monkeypatch):
    import ai_sales.tools.sales_tools as sales_tools

    monkeypatch.setattr(sales_tools, "_search_pinecone", lambda query, top_k=5: None)
    result = search_knowledge_base.invoke({"query": "nonexistent-xyz-12345"})
    assert "No relevant information found" in result


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
