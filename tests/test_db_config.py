from ai_sales.db_config import get_database_url, pool_max_size, warn_if_not_neon_pooler


def test_pool_max_size_default():
    assert pool_max_size() >= 1


def test_get_database_url_requires_env(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    try:
        get_database_url()
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "pooler" in str(exc).lower() or "DATABASE_URL" in str(exc)


def test_warn_if_not_neon_pooler(caplog):
    import logging

    caplog.set_level(logging.WARNING)
    warn_if_not_neon_pooler("postgresql://user@ep-direct.region.aws.neon.tech/db")
    assert "pooler" in caplog.text.lower()
    caplog.clear()
    warn_if_not_neon_pooler(
        "postgresql://user@ep-little-pooler.region.aws.neon.tech/db"
    )
    assert caplog.text == ""
