"""Tests for pre-fetch catalog context injection."""

from ai_sales.prefetch import build_catalog_prefetch


def test_prefetch_skips_non_product_greeting():
    assert build_catalog_prefetch("สวัสดีครับ") == ""


def test_prefetch_broad_catalog_question():
    ctx = build_catalog_prefetch("มีสินค้าอะไรบ้างครับ")
    assert ctx
    assert "พบ" in ctx


def test_prefetch_single_iphone_case():
    ctx = build_catalog_prefetch("จะซื้อเคส iPhone 15 Pro Max")
    assert ctx
    assert "พบ 1 รายการ" in ctx
    assert "หลายแบบ" in ctx
    assert "490" in ctx


def test_prefetch_buy_case_intent():
    ctx = build_catalog_prefetch("จะซื้อเคส")
    assert ctx
    assert "พบ" in ctx
    assert "ห้ามบอกว่ามีหลายแบบ" in ctx or "ห้ามอ้างว่ามีมากกว่า" in ctx


def test_prefetch_uses_summary_for_context():
    summary = "ลูกค้าใช้ iPhone 15 Pro Max สนใจเคส"
    ctx = build_catalog_prefetch("จะซื้อเลยครับ", summary)
    assert ctx
    assert "490" in ctx
