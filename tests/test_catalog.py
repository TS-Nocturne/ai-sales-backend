"""Tests for product catalog helpers."""

from ai_sales.tools.catalog import find_product_price, get_product_catalog


def test_find_product_price_exact_match():
    assert find_product_price("เคสใสกันกระแทก Crystal Clear สำหรับ iPhone 15 Pro Max") == 490.0


def test_find_product_price_partial_match():
    assert find_product_price("iPhone 15 Pro Max") == 490.0


def test_find_product_price_not_found():
    assert find_product_price("Nonexistent Product") == 0.0


def test_find_product_price_empty():
    assert find_product_price("") == 0.0


def test_catalog_falls_back_to_csv_when_api_empty(monkeypatch):
    """When the dashboard returns an empty list, use bundled CSV."""
    monkeypatch.setenv("CATALOG_API_URL", "http://127.0.0.1:9999/catalog")
    monkeypatch.setenv("INTERNAL_API_KEY", "test-key")

    def fake_load_from_api():
        return []

    import ai_sales.tools.catalog as catalog_module

    monkeypatch.setattr(catalog_module, "_load_from_api", fake_load_from_api)
    catalog_module._cache = {"data": None, "fetched_at": 0.0}

    catalog = get_product_catalog(force_refresh=True)
    assert len(catalog) >= 10
    assert any("iPhone 15" in p.get("name", "") for p in catalog)
