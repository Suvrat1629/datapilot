"""Adversarial SQL: every attack must be stopped by the static layer or by the database role.

The second half matters more: the static layer is best-effort, so each attack is also run
against Postgres as the read-only role to prove the privileges alone hold the line.
"""

import psycopg
import pytest

from datapilot import db
from datapilot.config import Settings
from datapilot.hints import DEMO_HINTS_PATH, load_hints
from datapilot.sql_analyst.nodes import Deps, SqlAnalystNodes
from datapilot.sql_analyst.safety import check_sql
from datapilot.sql_analyst.state import SqlState

pytestmark = pytest.mark.db

# Hidden-data attempts: whole-row references, JSON conversion, aliases, quoting, case tricks,
# subqueries, set operations, side channels through predicates, XML/ts helpers.
LEAKS = [
    "SELECT c FROM customers c",
    "SELECT row_to_json(c) FROM customers c",
    "SELECT to_jsonb(customers) FROM customers",
    'SELECT "email" FROM customers',
    "SELECT EMAIL FROM customers",
    "SELECT customers.EMail FROM customers",
    "SELECT (SELECT email FROM customers LIMIT 1)",
    "SELECT * FROM (SELECT * FROM customers) t",
    "SELECT t.* FROM customers t",
    "SELECT * FROM customers",
    "SELECT * FROM ONLY customers",
    "TABLE customers",
    "SELECT email FROM customers UNION SELECT name FROM cities",
    "SELECT CASE WHEN (SELECT count(*) FROM customers WHERE email LIKE 'a%') > 0 THEN 1 END",
    "SELECT count(*) FROM customers WHERE phone LIKE '9%'",
    "SELECT ts_stat('select email from customers')",
    "SELECT query_to_xml('select * from customers', true, false, '')",
    "SELECT table_to_xml('customers', true, false, '')",
    "SELECT database_to_xml(true, false, '')",
]

# Writes, session changes, files, other schemas, resource abuse.
ABUSE = [
    "/* x */ DROP TABLE bookings",
    "DO $$ BEGIN PERFORM 1; END $$",
    "LISTEN x",
    "NOTIFY x",
    "VACUUM bookings",
    "CALL foo()",
    "PREPARE p AS SELECT 1",
    "EXECUTE p",
    "DISCARD ALL",
    "RESET ALL",
    "COPY (SELECT 1) TO STDOUT",
    "SET search_path = pg_catalog",
    "SET statement_timeout = 0",
    "SELECT set_config('statement_timeout', '0', false)",
    "SELECT pg_ls_dir('.')",
    "SELECT pg_read_file('/etc/passwd')",
    "SELECT lo_get(1)",
    'SELECT * FROM "pg_catalog"."pg_shadow"',
    "SELECT * FROM PG_SHADOW",
    "SELECT * FROM pg_user",
    "SELECT * FROM information_schema . columns",
    'SELECT * FROM "Public".cities',
    "SELECT * FROM _datapilot_demo",
    "SELECT * FROM bookings b, LATERAL (SELECT pg_sleep(10)) s",
    "SELECT * FROM generate_series(1, 100000000000)",
]

# Allowed through, and harmless: comments are not statements, and these reveal nothing sensitive.
HARMLESS = [
    "SELECT 1 -- ; DROP TABLE bookings",
    "SELECT 1 /*; DROP TABLE bookings*/",
    "SELECT * FROM bookings TABLESAMPLE SYSTEM (100)",
    "SELECT current_user",
]


@pytest.fixture(scope="module")
def hints():
    return load_hints(DEMO_HINTS_PATH)


@pytest.fixture
def ro():
    with db.connect("ro") as c:
        c.execute("SET statement_timeout = 1500")
        yield c


def outcome(ro, q, hints):
    """'static', 'db', or the rows if the query ran."""
    r = check_sql(q, hints, max_rows=5)
    if not r.ok:
        return "static"
    try:
        return ro.execute(r.sql).fetchall()
    except psycopg.Error:
        return "db"
    finally:
        ro.rollback()


@pytest.mark.parametrize("q", LEAKS + ABUSE)
def test_attack_is_stopped(q, ro, hints):
    assert outcome(ro, q, hints) in ("static", "db")


@pytest.mark.parametrize("q", LEAKS)
def test_the_database_alone_stops_every_leak(q, ro):
    """Skip the static layer entirely: grants must hold even if the checker is bypassed."""
    try:
        rows = ro.execute(q).fetchall()
    except psycopg.Error:
        return
    finally:
        ro.rollback()
    text = str(rows).lower()
    assert "@" not in text, "an email address leaked"


@pytest.mark.parametrize("q", HARMLESS)
def test_harmless_queries_expose_nothing_sensitive(q, ro, hints):
    rows = outcome(ro, q, hints)
    assert isinstance(rows, list)
    assert "@" not in str(rows)


def test_no_write_survives_even_when_the_checker_is_skipped(ro):
    for stmt in (
        "DELETE FROM bookings",
        "UPDATE cities SET name = 'x'",
        "INSERT INTO cities VALUES (999, 'x', 'y')",
        "TRUNCATE payments",
        "CREATE TABLE public.pwn (x int)",
        "CREATE TABLE staging.pwn (x int)",
        "DROP TABLE bookings",
        "ALTER TABLE cities RENAME TO c2",
    ):
        with pytest.raises(psycopg.Error):
            ro.execute(stmt)
        ro.rollback()
    with db.connect("admin") as a:
        assert a.execute("SELECT count(*) AS n FROM cities").fetchone()["n"] == 12


def test_runaway_query_is_cancelled_by_the_timeout():
    deps = Deps(settings=Settings(statement_timeout_ms=400), hints=load_hints(DEMO_HINTS_PATH))
    state = SqlState(user_question="q", sql="SELECT count(*) FROM generate_series(1, 10000000000)")
    out = SqlAnalystNodes(deps).run_query(state)
    assert out.get("rows") is None and "timeout" in out["feedback"]


def test_row_cap_stops_a_huge_result():
    deps = Deps(settings=Settings(max_rows=7), hints=load_hints(DEMO_HINTS_PATH))
    state = SqlState(user_question="q", sql="SELECT * FROM generate_series(1, 1000000)")
    out = SqlAnalystNodes(deps).run_query(state)
    assert len(out["rows"]) == 7 and out["truncated"] is True
