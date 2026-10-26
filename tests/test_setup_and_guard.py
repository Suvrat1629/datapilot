"""Phase 6: role guard, `db setup` on a scratch database, demo guard, multi-schema."""

import secrets

import psycopg
import pytest
from psycopg import sql
from typer.testing import CliRunner

from datapilot import cli, db, guard
from datapilot.config import Settings, get_settings
from datapilot.demo.generate import NotDemoDatabase, check_demo_target
from datapilot.hints import Hints, TableHints
from datapilot.setup import connection_url, setup_database, verify

pytestmark = pytest.mark.db
runner = CliRunner()


@pytest.fixture
def scratch():
    """A throwaway database plus role names; everything is dropped afterwards."""
    tag = secrets.token_hex(3)
    name, ro, etl = f"dp_scratch_{tag}", f"dp_ro_{tag}", f"dp_etl_{tag}"
    base = get_settings().admin_database_url
    with psycopg.connect(base, autocommit=True) as c:
        c.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    url = base.rsplit("/", 1)[0] + "/" + name
    admin = psycopg.connect(url, row_factory=psycopg.rows.dict_row)
    admin.execute("CREATE SCHEMA sales")
    admin.execute("CREATE TABLE orders (id int PRIMARY KEY, item text, secret text)")
    admin.execute("CREATE TABLE sales.invoices (id int PRIMARY KEY, total numeric, ssn text)")
    admin.execute("INSERT INTO orders VALUES (1, 'a', 's')")
    admin.execute("INSERT INTO sales.invoices VALUES (1, 9.5, '123')")
    admin.commit()
    yield admin, url, ro, etl
    admin.close()
    with psycopg.connect(base, autocommit=True) as c:
        c.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name)))
        for role in (ro, etl):
            c.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(role)))


def settings_for(url, ro, etl, **kw):
    base = dict(
        admin_database_url=url,
        database_url=url,
        etl_database_url=url,
        ro_role=ro,
        etl_role=etl,
        schemas="public,sales",
        etl_readable_columns="orders.id,sales.invoices.id",
    )
    return Settings(**{**base, **kw})


HIDE = Hints(hidden_columns=["orders.secret", "sales.invoices.ssn"])


def run_setup(scratch, **kw):
    admin, url, ro, etl = scratch
    s = settings_for(url, ro, etl)
    return s, setup_database(admin, s, HIDE, ro_role=ro, etl_role=etl, **kw)


def as_role(url, role, password):
    return psycopg.connect(connection_url(url, role, password), row_factory=psycopg.rows.dict_row)


# ---- db setup ----------------------------------------------------------------------------


def test_setup_creates_limited_roles_and_verifies(scratch):
    admin, url, ro, etl = scratch
    s, report = run_setup(scratch)
    assert report.created == [ro, etl] and report.ok, report.problems
    assert set(report.granted) == {"orders", "sales.invoices"}
    assert report.granted["orders"] == ["id", "item"]  # secret hidden
    assert report.granted["sales.invoices"] == ["id", "total"]  # ssn hidden (qualified entry)
    assert report.etl_granted == {"orders": ["id"], "sales.invoices": ["id"]}

    with as_role(url, ro, report.passwords[ro]) as c:
        assert c.execute("SELECT item FROM orders").fetchone()["item"] == "a"
        assert c.execute("SELECT total FROM sales.invoices").fetchone()["total"] == 9.5
        for stmt in (
            "SELECT secret FROM orders",
            "SELECT ssn FROM sales.invoices",
            "INSERT INTO orders VALUES (2, 'b', 'x')",
            "CREATE TABLE public.t (x int)",
            "CREATE TABLE sales.t (x int)",
        ):
            with pytest.raises(psycopg.Error):
                c.execute(stmt)
            c.rollback()
    with as_role(url, etl, report.passwords[etl]) as c:
        c.execute("CREATE TABLE staging.w (x int)")
        c.execute("INSERT INTO staging.w VALUES (1)")
        c.commit()
        assert c.execute("SELECT id FROM public.orders").fetchone()["id"] == 1
        assert c.execute("SELECT id FROM sales.invoices").fetchone()["id"] == 1
        for stmt in (
            "SELECT item FROM public.orders",
            "SELECT total FROM sales.invoices",
            "CREATE TABLE public.t (x int)",
            "UPDATE public.orders SET id = 5",
        ):
            with pytest.raises(psycopg.Error):
                c.execute(stmt)
            c.rollback()
    with as_role(url, ro, report.passwords[ro]) as c:  # ETL-created tables are readable
        assert c.execute("SELECT x FROM staging.w").fetchone()["x"] == 1


def test_setup_is_idempotent_and_keeps_existing_passwords(scratch):
    s, first = run_setup(scratch)
    _, second = run_setup(scratch)
    assert second.created == [] and second.passwords == {} and second.ok
    assert set(second.existing) == set(first.created)
    _, third = run_setup(scratch, reset_passwords=True)
    assert set(third.passwords) == set(first.created) and third.ok


@pytest.mark.parametrize("name", ["", "a b", "x;drop", 'q"q', "a" * 64])
def test_setup_rejects_bad_role_names(scratch, name):
    admin, url, ro, etl = scratch
    with pytest.raises(ValueError):
        setup_database(admin, settings_for(url, ro, etl), HIDE, ro_role=name, etl_role=etl)


def test_setup_rejects_same_role_for_both(scratch):
    admin, url, ro, _ = scratch
    with pytest.raises(ValueError, match="different"):
        setup_database(admin, settings_for(url, ro, ro), HIDE, ro_role=ro, etl_role=ro)


def test_setup_refuses_hidden_etl_column_and_bad_entries(scratch):
    admin, url, ro, etl = scratch
    for cols in ("orders.secret", "orders.nope", "orders", "nosuch.id", "a.b.c.d"):
        s = settings_for(url, ro, etl, etl_readable_columns=cols)
        with pytest.raises(ValueError):
            setup_database(admin, s, HIDE, ro_role=ro, etl_role=etl)
        admin.rollback()


def test_verify_catches_public_grants_the_tool_cannot_fix(scratch):
    """A PUBLIC grant on a hidden column or table write is surfaced, not hidden."""
    admin, url, ro, etl = scratch
    s, report = run_setup(scratch)
    assert report.ok
    admin.execute("GRANT SELECT ON orders TO PUBLIC")
    admin.execute("GRANT INSERT ON orders TO PUBLIC")
    admin.commit()
    problems = verify(admin, s, HIDE, ro_role=ro, etl_role=etl)
    assert any("hidden column orders.secret" in p for p in problems[ro])
    assert any("INSERT" in p and "orders" in p for p in problems[ro])
    assert any("outside ETL_READABLE_COLUMNS" in p for p in problems[etl])


# ---- guard -------------------------------------------------------------------------------


def test_guard_flags_superuser_and_passes_the_real_roles():
    settings = get_settings()
    from datapilot.hints import load_hints

    hints = load_hints(settings.hints_path)
    with db.connect("admin") as c:
        problems = guard.check_role(c, "ro", hints, settings)
    text = " ".join(problems)
    assert "SUPERUSER" in text and "hidden column customers.email" in text
    for kind in ("ro", "etl"):
        with db.connect(kind) as c:
            assert guard.check_role(c, kind, hints, settings) == [], kind


def test_guard_flags_etl_role_with_wide_reads(scratch):
    admin, url, ro, etl = scratch
    s, report = run_setup(scratch)
    admin.execute(sql.SQL("GRANT SELECT ON orders TO {}").format(sql.Identifier(etl)))
    admin.commit()
    with as_role(url, etl, report.passwords[etl]) as c:
        problems = guard.check_role(c, "etl", HIDE, s)
    assert any("outside ETL_READABLE_COLUMNS" in p and "orders.item" in p for p in problems)


def test_ensure_safe_raises_and_remembers(monkeypatch):
    guard._checked.clear()
    s = get_settings()
    monkeypatch.setattr(s, "database_url", s.admin_database_url)  # superuser as the "ro" role
    with pytest.raises(guard.UnsafeRole) as e:
        guard.ensure_safe(("ro",), s)
    assert "SUPERUSER" in str(e.value)
    guard._checked.clear()
    monkeypatch.undo()
    s2 = get_settings()
    guard.ensure_safe(("ro", "etl"), s2)
    assert ("ro", s2.database_url) in guard._checked


@pytest.mark.real_guard
def test_cli_refuses_to_run_with_an_unsafe_role(monkeypatch):
    guard._checked.clear()
    monkeypatch.setattr(get_settings(), "database_url", get_settings().admin_database_url)
    res = runner.invoke(cli.app, ["ask", "how many bookings"])
    assert res.exit_code == 2 and "can do too much" in res.output
    assert "datapilot db setup" in res.output
    guard._checked.clear()


@pytest.mark.real_guard
def test_cli_reports_unreachable_database_cleanly(monkeypatch):
    guard._checked.clear()
    monkeypatch.setattr(get_settings(), "etl_database_url", "postgresql://x:y@127.0.0.1:1/none")
    res = runner.invoke(cli.app, ["etl", "load something"])
    assert res.exit_code == 2 and "Cannot connect" in res.output and "Traceback" not in res.output
    guard._checked.clear()


def test_db_check_command_passes_on_the_real_roles():
    res = runner.invoke(cli.app, ["db", "check"])
    assert res.exit_code == 0 and "limited as expected" in res.output


# ---- demo guard --------------------------------------------------------------------------


def test_demo_guard_allows_empty_marked_and_forced(scratch):
    admin, *_ = scratch
    with pytest.raises(NotDemoDatabase):  # has `orders` and friends, no marker
        check_demo_target(admin)
    check_demo_target(admin, force=True)
    admin.execute("CREATE TABLE _datapilot_demo (x int)")
    check_demo_target(admin)  # marked as a demo database
    admin.execute("DROP TABLE _datapilot_demo, orders")
    admin.commit()
    admin.execute("DROP TABLE sales.invoices")
    admin.commit()
    check_demo_target(admin)  # empty public schema


def test_demo_guard_blocks_real_tables_with_demo_names(scratch):
    admin, *_ = scratch
    admin.execute("DROP TABLE orders")
    admin.execute("CREATE TABLE bookings (id int, customer text)")  # someone's real table
    admin.commit()
    with pytest.raises(NotDemoDatabase, match="DROP"):
        check_demo_target(admin)


def test_demo_load_cli_refuses_on_non_demo_database(monkeypatch, scratch):
    admin, url, *_ = scratch
    monkeypatch.setattr(get_settings(), "admin_database_url", url)
    res = runner.invoke(cli.app, ["demo", "load"])
    assert res.exit_code == 1 and "Refusing" in res.output
    assert admin.execute("SELECT count(*) AS n FROM orders").fetchone()["n"] == 1


# ---- multi-schema introspection and hints -------------------------------------------------


def test_introspection_and_context_cover_extra_schemas(scratch):
    admin, url, ro, etl = scratch
    s, report = run_setup(scratch)
    hints = Hints(
        tables={"sales.invoices": TableHints(description="Billing documents")},
        hidden_columns=HIDE.hidden_columns,
    )
    with as_role(url, ro, report.passwords[ro]) as c:
        tables = db.introspect(c, hints, ("public", "sales", "staging"))
    by = {t.qualified: t for t in tables}
    assert set(by) == {"orders", "sales.invoices"}
    assert [c.name for c in by["sales.invoices"].columns] == ["id", "total"]
    ctx = db.render_schema_context(tables, hints)
    assert "TABLE sales.invoices -- Billing documents" in ctx and "ssn" not in ctx


def test_hint_resolution_rules():
    h = Hints(hidden_columns=["t.a", "sales.t.b", "public.t.c"])
    assert h.is_hidden("t", "a") and h.is_hidden("t", "c")
    assert h.is_hidden("t", "b", "sales") and not h.is_hidden("t", "a", "sales")
    assert not h.is_hidden("t", "b")  # same table name in public is a different table
    h2 = Hints(tables={"x": TableHints(description="pub"), "s.x": TableHints(description="s")})
    assert h2.for_table("x").description == "pub" and h2.for_table("x", "s").description == "s"
    assert h2.for_table("y") is None


def test_check_hints_understands_qualified_entries(scratch):
    admin, *_ = scratch
    schemas = ["public", "sales"]
    good = Hints(hidden_columns=["orders.secret", "sales.invoices.ssn"])
    bad = Hints(hidden_columns=["sales.invoices.nope", "sales.nothere.x", "a.b.c.d"])
    assert db.check_hints(admin, good, schemas) == []
    assert len(db.check_hints(admin, bad, schemas)) == 3
    only_public = db.check_hints(admin, good, ["public"])
    assert only_public == [
        "hints: hidden column 'sales.invoices.ssn' does not exist, nothing is hidden by it"
    ]
