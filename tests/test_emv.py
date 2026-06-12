from ai_sales.payments.emv import build_promptpay_emv


def test_static_citizen_id_emv_format():
    emv = build_promptpay_emv("1739902268848", "NATID", amount=None)
    assert emv.startswith("000201")
    assert "010211" in emv  # static (no fixed amount)
    assert "5802TH" in emv
    assert "5303764" in emv
    assert emv.endswith(emv[-4:])  # CRC present
    assert len(emv) > 40


def test_dynamic_phone_emv_includes_amount():
    emv = build_promptpay_emv("0812345678", "phone_number", amount=1234.5)
    assert "010212" in emv  # dynamic
    assert "54071234.50" in emv
