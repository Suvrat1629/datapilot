"""Refuse to run with database roles that are more powerful than the agent needs.

The agent's safety rests on database privileges, so a misconfigured DATABASE_URL (for example a
superuser pointed at a real database) must stop the program, not rely on prompts or SQL checks.
Every check asks Postgres what the *connected* role may do, so it needs no admin access.
"""

from __future__ import annotations

from typing import Literal

import psycopg

from datapilot import db
from datapilot.config import Settings, get_settings, readable_schemas, split_csv
from datapilot.hints import Hints, load_hints

Kind = Literal["ro", "etl"]

SYSTEM_SCHEMAS = ("pg_catalog", "information_schema")
WRITE_PRIVILEGES = ("INSERT", "UPDATE", "DELETE", "TRUNCATE")
ATTRIBUTE_NAMES = {
    "rolsuper": "SUPERUSER",
    "rolcreaterole": "CREATEROLE",
    "rolcreatedb": "CREATEDB",
    "rolreplication": "REPLICATION",
    "rolbypassrls": "BYPASSRLS",
}
FIX = "Run `datapilot db setup` to create correctly limited roles."


class UnsafeRole(Exception):
    def __init__(self, role: str, kind: str, problems: list[str]):
        self.role, self.kind, self.problems = role, kind, problems
        super().__init__(f"role {role!r} ({kind}) is not safe: " + "; ".join(problems))


def _user_schemas(conn: psycopg.Connection) -> list[dict]:
    return conn.execute(
        """
        SELECT nspname, has_schema_privilege(current_user, oid, 'CREATE') AS can_create
        FROM pg_namespace
        WHERE nspname <> ALL(%s) AND nspname NOT LIKE 'pg\\_%%'
        """,
        (list(SYSTEM_SCHEMAS),),
    ).fetchall()


def check_role(
    conn: psycopg.Connection, kind: Kind, hints: Hints, settings: Settings | None = None
) -> list[str]:
    """Problems with the role `conn` is connected as. Empty means safe to run with."""
    settings = settings or get_settings()
    problems: list[str] = []

    attrs = conn.execute(
        "SELECT rolsuper, rolcreaterole, rolcreatedb, rolreplication, rolbypassrls "
        "FROM pg_roles WHERE rolname = current_user"
    ).fetchone()
    for name, value in (attrs or {}).items():
        if value:
            problems.append(f"the role has the {ATTRIBUTE_NAMES[name]} attribute")
    if conn.execute(
        "SELECT has_database_privilege(current_user, current_database(), 'CREATE') AS c"
    ).fetchone()["c"]:
        problems.append("the role can create schemas in this database")

    # `staging` is the one place the ETL role may write; the SQL role may write nowhere.
    writable_ok = {"staging"} if kind == "etl" else set()
    for s in _user_schemas(conn):
        if s["can_create"] and s["nspname"] not in writable_ok:
            problems.append(f"the role can create objects in schema {s['nspname']!r}")

    tables = conn.execute(
        """
        SELECT n.nspname AS schema, c.relname AS name,
               ARRAY(SELECT p FROM unnest(%s::text[]) AS p
                     WHERE has_table_privilege(current_user, c.oid, p)) AS writes
        FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE c.relkind IN ('r', 'p', 'v', 'm', 'f')
          AND n.nspname <> ALL(%s) AND n.nspname NOT LIKE 'pg\\_%%'
        """,
        (list(WRITE_PRIVILEGES), list(SYSTEM_SCHEMAS)),
    ).fetchall()
    for t in tables:
        if t["writes"] and t["schema"] not in writable_ok:
            problems.append(f"the role can {'/'.join(t['writes'])} on {t['schema']}.{t['name']}")

    problems += _hidden_column_problems(conn, hints)
    if kind == "etl":
        problems += _etl_read_problems(conn, settings)
    return problems


def _hidden_column_problems(conn: psycopg.Connection, hints: Hints) -> list[str]:
    out = []
    for ref in hints.hidden_columns:
        parsed = db._split_ref(ref)
        if parsed is None:
            continue
        schema, table, col = parsed
        row = conn.execute(
            """
            SELECT CASE WHEN to_regclass(format('%%I.%%I', %s::text, %s::text)) IS NULL
                        THEN false
                        ELSE has_column_privilege(
                            current_user, format('%%I.%%I', %s::text, %s::text)::regclass,
                            %s::text, 'SELECT') END AS readable
            """,
            (schema, table, schema, table, col),
        ).fetchone()
        if row["readable"]:
            out.append(f"the role can read hidden column {ref}")
    return out


def _etl_read_problems(conn: psycopg.Connection, settings: Settings) -> list[str]:
    """The ETL role may read only ETL_READABLE_COLUMNS outside staging."""
    allowed = set()
    for ref in split_csv(settings.etl_readable_columns):
        if parsed := db._split_ref(ref):
            allowed.add(parsed)
    rows = conn.execute(
        """
        SELECT n.nspname AS schema, c.relname AS table, a.attname AS col
        FROM pg_attribute a
        JOIN pg_class c ON c.oid = a.attrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE a.attnum > 0 AND NOT a.attisdropped
          AND c.relkind IN ('r', 'p', 'v', 'm', 'f')
          AND n.nspname <> ALL(%s) AND n.nspname NOT LIKE 'pg\\_%%' AND n.nspname <> 'staging'
          AND has_column_privilege(current_user, c.oid, a.attnum, 'SELECT')
        """,
        (list(SYSTEM_SCHEMAS),),
    ).fetchall()
    extra = sorted(
        f"{r['schema']}.{r['table']}.{r['col']}"
        for r in rows
        if (r["schema"], r["table"], r["col"]) not in allowed
    )
    if extra:
        shown = ", ".join(extra[:5]) + (f" and {len(extra) - 5} more" if len(extra) > 5 else "")
        return [f"the role can read columns outside ETL_READABLE_COLUMNS: {shown}"]
    return []


_checked: set[tuple[str, str]] = set()


def ensure_safe(kinds: tuple[Kind, ...], settings: Settings | None = None) -> None:
    """Raise UnsafeRole if a role the program is about to use is too powerful.

    Connection problems propagate as psycopg errors for the caller to report. Results are
    remembered per process, so a chat session checks once.
    """
    settings = settings or get_settings()
    hints = load_hints(settings.hints_path)
    for kind in kinds:
        url = settings.database_url if kind == "ro" else settings.etl_database_url
        if (kind, url) in _checked:
            continue
        with db.connect(kind) as conn:
            role = conn.execute("SELECT current_user AS u").fetchone()["u"]
            problems = check_role(conn, kind, hints, settings)
        if problems:
            raise UnsafeRole(role, kind, problems)
        _checked.add((kind, url))


__all__ = ["FIX", "UnsafeRole", "check_role", "ensure_safe", "readable_schemas"]
