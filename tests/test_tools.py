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


def test_filter_results_by_max_price():
    import ai_sales.tools.sales_tools as sales_tools

    rows = [
        {"source_type": "product", "name": "Cable", "price": 350},
        {"source_type": "product", "name": "Hub", "price": 990},
        {"source_type": "faq", "text": "policy"},
    ]
    filtered = sales_tools._filter_results_by_max_price(rows, 400)
    assert len(filtered) == 2
    assert filtered[0]["name"] == "Cable"
    assert filtered[1]["text"] == "policy"


def test_search_knowledge_base_max_price_filter(monkeypatch):
    import ai_sales.tools.sales_tools as sales_tools

    monkeypatch.setattr(sales_tools, "_search_pinecone", lambda query, top_k=5: None)
    result = search_knowledge_base.invoke(
        {"query": "สายชาร์จ", "max_price": 400}
    )
    assert "350" in result or "P004" in result
    assert "1290" not in result
    assert "งบไม่เกิน" in result


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


def test_search_knowledge_base_broad_browse_catalog_fallback(monkeypatch):
    import ai_sales.tools.sales_tools as sales_tools

    monkeypatch.setattr(sales_tools, "_search_pinecone", lambda query, top_k=5: None)
    monkeypatch.setattr(sales_tools, "_search_in_memory", lambda query: [])
    result = search_knowledge_base.invoke({"query": "สินค้าแนะนำ"})
    assert "[Catalog Fallback]" in result
    assert "No relevant information found" not in result
    assert "Product]" in result or "P001" in result


def test_catalog_browse_fallback_respects_max_price():
    import ai_sales.tools.sales_tools as sales_tools

    picks = sales_tools._catalog_browse_fallback(max_price=400, limit=10)
    assert picks
    assert all(float(p.get("price", 0)) <= 400 for p in picks)


def test_is_broad_browse_query():
    import ai_sales.tools.sales_tools as sales_tools

    assert sales_tools._is_broad_browse_query("สินค้าแนะนำ")
    assert sales_tools._is_broad_browse_query("แนะนำสินค้าให้หน่อย")
    assert sales_tools._is_broad_browse_query("มีอะไรขายบ้างครับ")
    assert not sales_tools._is_broad_browse_query("เคส iPhone 15")


def test_list_products_open_browse_catalog_fallback(monkeypatch):
    import ai_sales.tools.sales_tools as sales_tools

    monkeypatch.setattr(sales_tools, "get_product_catalog", lambda: [])
    monkeypatch.setattr(
        sales_tools,
        "_catalog_browse_fallback",
        lambda max_price=None, limit=3: [
            {
                "name": "สายชาร์จทดสอบ",
                "price": 350,
                "category": "Cable",
                "stock": 10,
                "description": "ทดสอบ",
            }
        ],
    )
    result = list_products.invoke(
        {
            "category": "",
            "keyword": "",
            "min_price": 0,
            "max_price": 0,
            "sort_by_price": "",
            "display_mode": "featured",
        }
    )
    assert "[Catalog]" in result
    assert "สายชาร์จทดสอบ" in result


def test_list_products_pure_browse_returns_categories():
    result = list_products.invoke(
        {
            "category": "",
            "keyword": "",
            "min_price": 0,
            "max_price": 0,
            "sort_by_price": "",
            "display_mode": "categories",
        }
    )
    assert "[หมวดหมู่สินค้า]" in result
    assert "รายการ)" in result


def test_list_products_respects_limit():
    result = list_products.invoke(
        {
            "category": "",
            "keyword": "",
            "min_price": 0,
            "max_price": 0,
            "sort_by_price": "",
            "limit": 2,
            "display_mode": "featured",
        }
    )
    assert "แสดง 2" in result or "(2 รายการ)" in result


def test_is_pure_catalog_browse():
    import ai_sales.tools.sales_tools as sales_tools

    assert sales_tools._is_pure_catalog_browse("มีอะไรขายบ้าง")
    assert sales_tools._is_pure_catalog_browse("ขายอะไรบ้าง")
    assert not sales_tools._is_pure_catalog_browse("แนะนำสินค้าให้หน่อย")


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
