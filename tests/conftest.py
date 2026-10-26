import psycopg
import pytest

from datapilot import db


def _db_up() -> bool:
    try:
        with db.connect("admin") as c:
            c.execute("SELECT 1 FROM bookings LIMIT 1")
        return True
    except psycopg.Error:
        return False


def pytest_collection_modifyitems(config, items):
    if _db_up():
        return
    skip = pytest.mark.skip(
        reason="Postgres demo database not available (docker compose up + demo load)"
    )
    for item in items:
        if "db" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _skip_cli_role_guard(request, monkeypatch):
    """CLI commands check database roles at start; unit tests with stubs have no database."""
    if request.node.get_closest_marker("real_guard"):
        return
    from datapilot import cli

    monkeypatch.setattr(cli, "_require_safe_roles", lambda *kinds: None)
