"""
Product catalog for the AI Sales Agent.

Source of truth is PostgreSQL, owned by the Next.js dashboard (the Inventory
Sync writes products there). This brain has no DB driver by design, so it reads
the *live* catalog over HTTP from the dashboard's internal endpoint
(``CATALOG_API_URL``) and caches it briefly. If the dashboard is unreachable or
not configured, it falls back to the bundled ``documents/sample_product.csv`` so
local development and tests keep working offline.
"""

import csv
import os
import time

import urllib.error
import urllib.request
import json

# Get path relative to the project root
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CSV_PATH = os.path.join(_BASE_DIR, "documents", "sample_product.csv")

# Cache the resolved catalog for a short window so repeated tool calls within a
# conversation do not hammer the dashboard. Set to 0 to disable caching.
_CACHE_TTL_SECONDS = 30
_cache: dict = {"data": None, "fetched_at": 0.0}


def _coerce_product(row: dict) -> dict:
    """Normalise types for a single product row (price float, stock int)."""
    product = dict(row)
    try:
        product["price"] = float(product.get("price", 0) or 0)
    except (TypeError, ValueError):
        product["price"] = 0.0
    try:
        product["stock"] = int(float(product.get("stock", 0) or 0))
    except (TypeError, ValueError):
        product["stock"] = 0
    return product


def _load_from_api() -> list[dict] | None:
    """Fetch the live catalog from the dashboard's internal endpoint.

    Returns a list of products, or None if not configured / unreachable.
    """
    url = os.getenv("CATALOG_API_URL")
    if not url:
        return None

    request = urllib.request.Request(url, method="GET")
    api_key = os.getenv("INTERNAL_API_KEY")
    if api_key:
        request.add_header("x-internal-key", api_key)

    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
        print(f"Warning: live catalog fetch failed ({exc}); using CSV fallback.")
        return None

    products = payload.get("products")
    if not isinstance(products, list):
        return None
    return [_coerce_product(p) for p in products]


def _load_from_csv() -> list[dict]:
    """Load the bundled sample catalog from CSV (offline fallback)."""
    catalog: list[dict] = []
    try:
        with open(_CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                catalog.append(_coerce_product(row))
    except Exception as exc:
        print(f"Warning: Failed to load product catalog CSV: {exc}")
    return catalog


def get_product_catalog(force_refresh: bool = False) -> list[dict]:
    """Return the product catalog (live from the dashboard, else CSV).

    Args:
        force_refresh: Bypass the in-memory cache and re-resolve immediately.
    """
    now = time.monotonic()
    if (
        not force_refresh
        and _cache["data"] is not None
        and (now - _cache["fetched_at"]) < _CACHE_TTL_SECONDS
    ):
        return _cache["data"]

    catalog = _load_from_api()
    if not catalog:
        catalog = _load_from_csv()

    _cache["data"] = catalog
    _cache["fetched_at"] = now
    return catalog


def find_product_price(product_name: str) -> float:
    """Look up a product's price from the catalog by name.

    Prefers exact match, then longest partial match to avoid ambiguity.

    Args:
        product_name: Full or partial product name to search for.

    Returns:
        The product price, or 0.0 if not found.
    """
    if not product_name:
        return 0.0

    catalog = get_product_catalog()
    product_name_lower = product_name.lower().strip()

    for product in catalog:
        if product["name"].lower() == product_name_lower:
            return product["price"]

    best_match = None
    best_len = 0
    for product in catalog:
        name_lower = product["name"].lower()
        if product_name_lower in name_lower or name_lower in product_name_lower:
            if len(name_lower) > best_len:
                best_match = product
                best_len = len(name_lower)

    return best_match["price"] if best_match else 0.0
