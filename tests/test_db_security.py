"""Security boundary tests: these exercise Postgres roles directly, no LLM involved."""

import psycopg
import pytest

from datapilot import db
from datapilot.hints import DEMO_HINTS_PATH, Hints, TableHints, load_hints

pytestmark = pytest.mark.db


@pytest.fixture
def ro():
    with db.connect("ro") as c:
        yield c


@pytest.fixture
def etl():
    with db.connect("etl") as c:
        yield c


def test_ro_can_select_visible_columns(ro):
    row = ro.execute("SELECT count(*) AS n FROM bookings").fetchone()
    assert row["n"] > 0


@pytest.mark.parametrize("col", ["phone", "email"])
def test_ro_cannot_read_hidden_columns(ro, col):
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        ro.execute(f"SELECT {col} FROM customers LIMIT 1")


def test_ro_cannot_select_star_from_table_with_hidden_columns(ro):
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        ro.execute("SELECT * FROM customers LIMIT 1")


@pytest.mark.parametrize(
    "stmt",
    [
        "UPDATE bookings SET fare_amount = 0",
        "DELETE FROM bookings",
        "INSERT INTO cities (id, name, state) VALUES (999, 'x', 'y')",
        "DROP TABLE bookings",
        "CREATE TABLE public.evil (id int)",
        "CREATE TABLE staging.evil (id int)",
    ],
)
def test_ro_cannot_write_or_ddl(ro, stmt):
    with pytest.raises(psycopg.Error):
        ro.execute(stmt)


def test_ro_transaction_is_read_only(ro):
    assert (
        ro.execute("SHOW default_transaction_read_only").fetchone()["default_transaction_read_only"]
        == "on"
    )


def test_ro_statement_timeout_enforced():
    with db.connect("ro") as c:
        assert c.execute("SHOW statement_timeout").fetchone()["statement_timeout"] == "10s"
        with pytest.raises(psycopg.errors.QueryCanceled):
            c.execute("SET statement_timeout = '200ms'")
            c.execute("SELECT pg_sleep(2)")


def test_etl_cannot_read_source_tables(etl):
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        etl.execute("SELECT 1 FROM public.bookings LIMIT 1")


def test_etl_cannot_write_source_tables(etl):
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        etl.execute("DELETE FROM public.bookings")


def test_etl_can_write_staging_and_ro_can_read_it(etl):
    etl.execute("DROP TABLE IF EXISTS staging.t_roundtrip")
    etl.execute("CREATE TABLE staging.t_roundtrip (v int)")
    etl.execute("INSERT INTO staging.t_roundtrip VALUES (1), (2)")
    etl.commit()
    try:
        with db.connect("ro") as ro:
            assert ro.execute("SELECT count(*) AS n FROM staging.t_roundtrip").fetchone()["n"] == 2
    finally:
        etl.execute("DROP TABLE IF EXISTS staging.t_roundtrip")
        etl.commit()


def test_introspection_hides_hidden_columns_and_finds_keys(ro):
    hints = load_hints(DEMO_HINTS_PATH)
    tables = {t.name: t for t in db.introspect(ro, hints)}
    assert {"bookings", "cities", "customers", "drivers", "payments"} <= set(tables)
    assert "phone" not in {c.name for c in tables["customers"].columns}
    assert "email" not in {c.name for c in tables["customers"].columns}
    cols = {c.name: c for c in tables["bookings"].columns}
    assert cols["id"].is_pk
    assert cols["city_id"].fk == "cities.id"
    assert cols["status"].samples == ["cancelled", "completed", "no_show"]


def test_demo_patterns_hold():
    """The eval set depends on these injected patterns."""
    with db.connect("admin") as c:
        top = c.execute(
            """
            SELECT ci.name FROM bookings b JOIN cities ci ON ci.id = b.city_id
            WHERE b.status = 'cancelled' AND extract(isodow FROM b.created_at) >= 6
            GROUP BY ci.name ORDER BY count(*) DESC LIMIT 1
            """
        ).fetchone()
        assert top["name"] == "Bengaluru"
        fare = c.execute(
            """
            SELECT ci.name FROM bookings b JOIN cities ci ON ci.id = b.city_id
            WHERE b.status = 'completed' GROUP BY ci.name ORDER BY avg(b.fare_amount) DESC LIMIT 1
            """
        ).fetchone()
        assert fare["name"] == "Mumbai"
        reason = c.execute(
            "SELECT cancel_reason FROM bookings WHERE status = 'cancelled' "
            "GROUP BY 1 ORDER BY count(*) DESC LIMIT 1"
        ).fetchone()
        assert reason["cancel_reason"] == "driver_unavailable"


def test_check_hints_flags_stale_entries():
    hints = Hints(
        tables={
            "nope": TableHints(),
            "bookings": TableHints(columns={"ghost": "x", "status": "ok"}),
        },
        hidden_columns=["customers.ssn", "customers.email"],
    )
    with db.connect("admin") as c:
        warnings = db.check_hints(c, hints)
    assert any("table 'nope'" in w for w in warnings)
    assert any("bookings.ghost" in w for w in warnings)
    assert any("customers.ssn" in w and "nothing is hidden" in w for w in warnings)
    assert not any("customers.email" in w or "bookings.status" in w for w in warnings)


def test_demo_hints_match_demo_schema():
    with db.connect("admin") as c:
        assert db.check_hints(c, load_hints(DEMO_HINTS_PATH)) == []
