"""Tests for broad catalog intent detection."""

from ai_sales.nodes.agent_nodes import (
    _looks_like_broad_catalog_query,
    _looks_like_budget_browse_query,
    _looks_like_recommend_query,
    _should_auto_browse_catalog,
)


def test_broad_catalog_thai_phrases():
    assert _looks_like_broad_catalog_query("มีสินค้าอะไรบ้างครับ")
    assert _looks_like_broad_catalog_query("มีอะไรขายบ้าง")
    assert _looks_like_broad_catalog_query("แนะนำสินค้าให้หน่อย")


def test_recommend_query_phrases():
    assert _looks_like_recommend_query("มีรุ่นใหนแนะนำบ้าง")
    assert _looks_like_recommend_query("มีรุ่นไหนบ้างครับ")


def test_budget_browse_query():
    assert _looks_like_budget_browse_query("มีงบ 3000 ซื้ออะไรได้บ้าง")


def test_should_auto_browse_catalog():
    assert _should_auto_browse_catalog("มีอะไรขายบ้าง")
    assert _should_auto_browse_catalog("มีรุ่นใหนแนะนำบ้าง")
    assert _should_auto_browse_catalog("มีงบ 3000 ซื้ออะไรได้บ้าง")
    assert not _should_auto_browse_catalog("เคส iPhone 15 มีไหม")
    assert not _should_auto_browse_catalog("สนใจต้องทำยังไง")


def test_broad_catalog_skips_specific_product():
    assert not _looks_like_broad_catalog_query("เคส iPhone 15 มีไหม")
