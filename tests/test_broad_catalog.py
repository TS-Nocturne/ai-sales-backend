"""Tests for broad catalog intent detection."""

from ai_sales.nodes.agent_nodes import _looks_like_broad_catalog_query


def test_broad_catalog_thai_phrases():
    assert _looks_like_broad_catalog_query("มีสินค้าอะไรบ้างครับ")
    assert _looks_like_broad_catalog_query("มีอะไรขายบ้าง")
    assert _looks_like_broad_catalog_query("แนะนำสินค้าให้หน่อย")


def test_broad_catalog_skips_specific_product():
    assert not _looks_like_broad_catalog_query("เคส iPhone 15 มีไหม")
