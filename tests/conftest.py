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
