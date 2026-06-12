"""Tests for PromptPay / Slip2Go configuration normalization."""

import pytest

from ai_sales.payments import slip2go
from ai_sales.payments.slip2go import (
    normalize_promptpay_code,
    normalize_promptpay_type,
)


def test_normalize_promptpay_type_natid_alias():
    assert normalize_promptpay_type("NATID") == "citizen_id"
    assert normalize_promptpay_type("citizen_id") == "citizen_id"
    assert normalize_promptpay_type("phone") == "phone_number"


def test_build_check_condition_uses_type_number_schema():
    cond = slip2go.build_check_condition("1739902268848", "NATID")
    receivers = cond["checkReceiver"]
    assert isinstance(receivers, list)
    assert receivers[0]["type"] == "citizen_id"
    assert receivers[0]["number"] == "1739902268848"
    assert "accountType" not in receivers[0]


def test_verify_with_receiver_schema_accepted_by_api():
    """Regression: accountType/accountNumber caused HTTP 400 from Slip2Go."""
    tiny = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )
    cond = slip2go.build_check_condition("1739902268848", "NATID")
    try:
        resp = slip2go.verify_slip_base64(tiny, cond)
    except slip2go.Slip2GoError as exc:
        pytest.fail(f"Slip2Go rejected request schema: {exc}")
    assert resp.get("code") != "400400"
