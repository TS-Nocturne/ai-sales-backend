"""Neon PostgreSQL connection settings for the brain (LangGraph checkpoints).

Use the **pooled** Neon URL at runtime (hostname contains ``-pooler``).
Keep ``DATABASE_URL_UNPOOLED`` for one-off migrations only (dashboard entrypoint).
"""

from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger(__name__)

_POOLER_HOST = re.compile(r"-pooler\.", re.IGNORECASE)


def get_database_url() -> str:
    """Return DATABASE_URL or raise if missing."""
    url = (os.getenv("DATABASE_URL") or "").strip()
    if not url:
        raise ValueError(
            "DATABASE_URL environment variable is required "
            "(use Neon pooled URL with -pooler in the hostname)"
        )
    return url


def warn_if_not_neon_pooler(url: str) -> None:
    """Log a warning when the brain might open too many direct Neon connections."""
    if _POOLER_HOST.search(url):
        return
    logger.warning(
        "DATABASE_URL does not look like a Neon pooler URL (-pooler in host). "
        "Both Dashboard and Brain should use the pooled endpoint at runtime to "
        "avoid hitting Neon connection limits."
    )


def pool_max_size() -> int:
    """Client-side pool cap — keep small when Neon pooler multiplexes connections."""
    raw = (os.getenv("DATABASE_POOL_MAX_SIZE") or "5").strip()
    try:
        size = int(raw)
    except ValueError:
        size = 5
    return max(1, min(size, 20))
