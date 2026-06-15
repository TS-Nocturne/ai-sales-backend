"""Tests for fixed category Thai labels."""

from ai_sales.config.category_labels import (
    CATEGORY_TH_MAP,
    DEFAULT_CATEGORY_TH,
    category_label,
)


def test_category_label_known_english():
    assert category_label("Case") == "เคส"
    assert category_label("Cable") == "สายชาร์จ"
    assert category_label("case") == "เคส"


def test_category_label_smartwatch_fixed():
    assert category_label("Smartwatch") == "สมาร์ทวอทช์"


def test_category_label_thai_passthrough():
    assert category_label("เคสมือถือ") == "เคสมือถือ"


def test_category_label_unknown_uses_fixed_fallback():
    assert category_label("BrandNewGadget") == DEFAULT_CATEGORY_TH


def test_sample_catalog_categories_all_mapped():
    sample_categories = {
        "Case",
        "Screen Protector",
        "Charger",
        "Cable",
        "Power Bank",
        "Audio",
        "Car Accessories",
        "Adapter",
        "Accessories",
    }
    for name in sample_categories:
        assert category_label(name) in CATEGORY_TH_MAP.values()
