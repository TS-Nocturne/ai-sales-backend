"""
Slip2Go integration: Thai bank-slip verification + PromptPay QR generation.

Uses the official Slip2Go "API Connect" service. Authentication is a Bearer
token (your account's API Secret), supplied via the ``SLIP_VERIFY_SECRET``
environment variable. The base URL can be overridden with ``SLIP2GO_API_URL``
(defaults to the production host).

Endpoints used (confirmed against the Slip2Go guide):
- Verify slip by Base64 image : POST /api/verify-slip/qr-base64/info
- Generate PromptPay QR code   : POST /api/qr-payment/generate-qr-code

Only the standard library is used (``urllib``) so no extra dependency is
required for the brain service.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request

try:  # python-dotenv is normally present; degrade gracefully if not.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - relies on env vars already being set
    pass

DEFAULT_BASE_URL = "https://connect.slip2go.com"

VERIFY_BASE64_PATH = "/api/verify-slip/qr-base64/info"
GENERATE_QR_CODE_PATH = "/api/qr-payment/generate-qr-code"
GET_SLIP_PATH = "/api/verify-slip/{reference_id}"

# Valid PromptPay account types accepted by Slip2Go.
PROMPTPAY_TYPES = ("phone_number", "citizen_id", "e_wallet")

_TYPE_ALIASES = {
    "natid": "citizen_id",
    "national_id": "citizen_id",
    "citizen": "citizen_id",
    "citizen_id": "citizen_id",
    "id_card": "citizen_id",
    "phone": "phone_number",
    "mobile": "phone_number",
    "tel": "phone_number",
    "phone_number": "phone_number",
    "e_wallet": "e_wallet",
    "ewallet": "e_wallet",
    "wallet": "e_wallet",
}


def normalize_promptpay_type(value: str | None) -> str:
    """Map .env aliases (e.g. NATID) to Slip2Go account types."""
    key = (value or "phone_number").strip().lower()
    normalized = _TYPE_ALIASES.get(key, key)
    if normalized in PROMPTPAY_TYPES:
        return normalized
    return "phone_number"


def normalize_promptpay_code(value: str | None) -> str:
    """Strip non-digits so masked IDs still match."""
    return re.sub(r"\D", "", value or "")


class Slip2GoError(Exception):
    """Raised when the Slip2Go API call fails or is misconfigured."""

    def __init__(self, message: str, status: int | None = None, payload=None):
        super().__init__(message)
        self.status = status
        self.payload = payload


def _base_url() -> str:
    return os.getenv("SLIP2GO_API_URL", DEFAULT_BASE_URL).rstrip("/")


def _secret() -> str:
    secret = (os.getenv("SLIP_VERIFY_SECRET") or "").strip()
    if not secret:
        raise Slip2GoError("ยังไม่ได้ตั้งค่า SLIP_VERIFY_SECRET ใน .env")
    return secret


def _request(method: str, path: str, body: dict | None = None, timeout: int = 30) -> dict:
    """Call Slip2Go and return the parsed JSON response."""
    url = f"{_base_url()}{path}"
    headers = {"Authorization": f"Bearer {_secret()}"}
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url, data=data, method=method, headers=headers)

    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(detail)
            message = payload.get("message") or f"Slip2Go API error {exc.code}"
        except json.JSONDecodeError:
            payload = {"raw": detail}
            message = f"Slip2Go API error {exc.code}"
        raise Slip2GoError(message, status=exc.code, payload=payload) from exc
    except urllib.error.URLError as exc:
        raise Slip2GoError(
            f"เชื่อมต่อ Slip2Go ไม่ได้: {exc.reason}", status=503
        ) from exc


def _post(path: str, body: dict, timeout: int = 30) -> dict:
    return _request("POST", path, body, timeout)


def _get(path: str, timeout: int = 30) -> dict:
    return _request("GET", path, None, timeout)


def build_check_condition(
    prompt_pay_code: str | None = None,
    prompt_pay_type: str | None = None,
    *,
    check_duplicate: bool = True,
) -> dict:
    """Build Slip2Go checkCondition using the documented {type, number} schema."""
    condition: dict = {}
    if check_duplicate:
        condition["checkDuplicate"] = True
    code = normalize_promptpay_code(
        prompt_pay_code or os.getenv("PROMPTPAY_CODE")
    )
    ptype = normalize_promptpay_type(
        prompt_pay_type or os.getenv("PROMPTPAY_TYPE")
    )
    if code:
        condition["checkReceiver"] = [{"type": ptype, "number": code}]
    return condition


def verify_slip_base64(
    image_base64: str, check_condition: dict | None = None
) -> dict:
    """Verify a bank-transfer slip from a Base64-encoded image.

    Args:
        image_base64: The slip image as a Base64 string. May include the
            ``data:image/...;base64,`` prefix.
        check_condition: Optional dict of extra checks (checkDuplicate,
            checkReceiver, checkAmount, checkDate) following the Slip2Go schema.

    Returns:
        The raw Slip2Go JSON response (slip data + verification result).
    """
    if not image_base64 or not image_base64.strip():
        raise Slip2GoError("ต้องระบุรูปสลิปแบบ Base64")

    payload: dict = {"imageBase64": image_base64}
    if check_condition:
        payload["checkCondition"] = check_condition

    return _post(VERIFY_BASE64_PATH, {"payload": payload})


def generate_promptpay_qr(
    prompt_pay_code: str,
    prompt_pay_type: str = "phone_number",
    account_name: str = "",
    amount: float | str | None = None,
) -> dict:
    """Generate a PromptPay QR-code payload for receiving payment.

    Args:
        prompt_pay_code: PromptPay id (phone number / citizen id / e-wallet id).
        prompt_pay_type: One of "phone_number", "citizen_id", "e_wallet".
        account_name: Receiver display name (optional).
        amount: Amount in THB (optional). Sent as a string per the API spec.

    Returns:
        The raw Slip2Go JSON response containing the QR code data.
    """
    if prompt_pay_type not in PROMPTPAY_TYPES:
        prompt_pay_type = normalize_promptpay_type(prompt_pay_type)
    if prompt_pay_type not in PROMPTPAY_TYPES:
        raise Slip2GoError(
            f"promptPayType ไม่ถูกต้อง ต้องเป็นหนึ่งใน {', '.join(PROMPTPAY_TYPES)}"
        )

    code = normalize_promptpay_code(prompt_pay_code)
    if not code:
        raise Slip2GoError("ต้องระบุ promptPayCode")

    body: dict = {
        "promptPayCode": code,
        "promptPayType": prompt_pay_type,
    }
    if account_name:
        body["accountName"] = account_name
    if amount is not None and str(amount).strip() != "":
        # API spec: amount is a string, no commas.
        body["amount"] = str(amount)

    return _post(GENERATE_QR_CODE_PATH, body)


def get_slip_by_reference(reference_id: str) -> dict:
    """Fetch a previously verified slip by its Slip2Go referenceId.

    Use this to re-read an already-checked slip WITHOUT spending verification
    quota — e.g. to detect customers re-submitting an old slip: keep the
    referenceId of every accepted slip, and on a new submission look it up here
    (or pass checkCondition={"checkDuplicate": true} to verify_slip_base64).

    Args:
        reference_id: The referenceId returned by a prior verification.

    Returns:
        The raw Slip2Go JSON response with the stored slip data.
    """
    if not reference_id or not reference_id.strip():
        raise Slip2GoError("ต้องระบุ referenceId ของสลิป")

    quoted = urllib.parse.quote(reference_id.strip(), safe="")
    return _get(GET_SLIP_PATH.format(reference_id=quoted))
