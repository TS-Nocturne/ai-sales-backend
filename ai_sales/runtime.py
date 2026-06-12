"""Shared LangGraph runtime for CLI, demo, and local scripts."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from psycopg_pool import ConnectionPool

from ai_sales.db_config import get_database_url, pool_max_size, warn_if_not_neon_pooler
from ai_sales.graph.builder import build_graph


@contextmanager
def graph_runtime() -> Iterator:
    """Open a Postgres pool, build the graph, and close the pool on exit."""
    db_url = get_database_url()
    warn_if_not_neon_pooler(db_url)
    pool = ConnectionPool(
        conninfo=db_url,
        min_size=1,
        max_size=pool_max_size(),
        kwargs={"autocommit": True},
    )
    try:
        yield build_graph(pool)
    finally:
        pool.close()
