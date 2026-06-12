#!/usr/bin/env python3
"""Generate the static store PromptPay QR PNG for the dashboard.

The image is served at /payment-qr/store-promptpay.png when the brain sets
payment_qr.use_static = true (full payment — customer enters amount in bank app).

Reads PROMPTPAY_* from AI-Sales/.env (or environment). Does not call Slip2Go.

Usage (from repo root):
  python AI-Sales/scripts/generate_store_qr.py
  python AI-Sales/scripts/generate_store_qr.py --output ai-sale-dashboard/public/payment-qr/store-promptpay.png
"""

from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_sales.payments import slip2go  # noqa: E402
from ai_sales.payments.emv import build_promptpay_emv  # noqa: E402
from ai_sales.payments.qr import _promptpay_settings, emv_payload_to_png_base64  # noqa: E402

DEFAULT_OUTPUT = (
    ROOT.parent / "ai-sale-dashboard" / "public" / "payment-qr" / "store-promptpay.png"
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate static store PromptPay QR PNG")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output PNG path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()
    out: Path = args.output

    code, ptype, name = _promptpay_settings()
    emv = build_promptpay_emv(code, ptype, amount=None)
    data_url = emv_payload_to_png_base64(emv)
    prefix = "data:image/png;base64,"
    if not data_url.startswith(prefix):
        raise ValueError("Unexpected PNG data URL format")
    png = base64.b64decode(data_url[len(prefix) :])

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(png)

    print(f"Saved static store QR -> {out}")
    print(f"  PromptPay: {code} ({ptype})")
    print(f"  Account:   {name}")
    print("  Amount:    (customer enters in banking app)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except slip2go.Slip2GoError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
