"""Tests for built-in technical reference (IP, Bluetooth)."""

from ai_sales.knowledge.technical_reference import (
    format_technical_reference,
    looks_like_technical_query,
    match_technical_reference,
)


def test_ipx4_query_matches():
    assert looks_like_technical_query("IPX4 นี่มันประมาณไหนครับ")
    hit = match_technical_reference("IPX4")
    assert hit is not None
    assert "IPX4" in hit["title"]


def test_ip_general_waterproof_query():
    assert looks_like_technical_query("กันน้ำระดับไหนดี")
    hit = match_technical_reference("มาตรฐานกันน้ำ")
    assert hit is not None


def test_bluetooth_query_matches():
    assert looks_like_technical_query("สเปก Bluetooth 5.3 ต่างจาก 5.0 ยังไง")
    hit = match_technical_reference("Bluetooth codec AAC")
    assert hit is not None


def test_non_technical_query_ignored():
    assert not looks_like_technical_query("มีเคส iPhone 15 ไหม")
    assert match_technical_reference("มีเคส iPhone 15 ไหม") is None


def test_format_technical_reference_includes_instructions():
    text = format_technical_reference("IPX4")
    assert text is not None
    assert "[Technical Reference]" in text
    assert "analogy" in text or "อธิบาย" in text
