"""Retry helpers for transient Gemini / network failures."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

_RETRYABLE_MARKERS = (
    "503",
    "429",
    "UNAVAILABLE",
    "RESOURCEEXHAUSTED",
    "HIGH DEMAND",
    "OVERLOADED",
    "RATE LIMIT",
    "DEADLINE EXCEEDED",
    "INTERNAL ERROR",
    "TRY AGAIN",
)


def is_retryable_error(exc: BaseException) -> bool:
    """True when the error is likely transient (Gemini overload, rate limit, etc.)."""
    msg = str(exc).upper()
    return any(marker in msg for marker in _RETRYABLE_MARKERS)


def invoke_with_retry(
    fn: Callable[[], T],
    *,
    max_attempts: int = 3,
    base_delay: float = 1.0,
) -> T:
    """Run ``fn`` with exponential backoff on retryable failures."""
    last: BaseException | None = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as exc:
            last = exc
            if not is_retryable_error(exc) or attempt >= max_attempts - 1:
                raise
            time.sleep(base_delay * (2**attempt))
    assert last is not None
    raise last
