"""`datapilot db setup`: create the limited roles, `staging` schema and grants on any database.

Everything runs from an admin connection and never touches existing data. It deliberately does
not revoke privileges from PUBLIC, because that could break other applications sharing the
database; instead `verify` reports anything that still lets the agent roles write or read too
much, so the problem is surfaced rather than silently papered over.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from urllib.parse import quote

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict

from datapilot import db, guard
from datapilot.config import Settings, split_csv
from datapilot.hints import Hints

ROLE_NAME_MAX = 63


@dataclass
class SetupReport:
    database: str
    created: list[str] = field(default_factory=list)
    existing: list[str] = field(default_factory=list)
    passwords: dict[str, str] = field(default_factory=dict)  # only roles created now
    granted: dict[str, list[str]] = field(default_factory=dict)
    etl_granted: dict[str, list[str]] = field(default_factory=dict)
    problems: dict[str, list[str]] = field(default_factory=dict)  # role -> problems

    @property
    def ok(self) -> bool:
        return not any(self.problems.values())


def _check_name(name: str) -> str:
    if not name or len(name) > ROLE_NAME_MAX or not name.replace("_", "").isalnum():
        raise ValueError(f"invalid role name {name!r}: use letters, digits and underscores")
    return name


def connection_url(admin_url: str, role: str, password: str | None) -> str:
    """Connection URL for `role` on the same server and database as the admin URL."""
    info = conninfo_to_dict(admin_url)
    host = info.get("host", "localhost")
    port = info.get("port", "5432")
    auth = quote(role, safe="") + (":" + quote(password, safe="") if password else "")
    return f"postgresql://{auth}@{host}:{port}/{info.get('dbname', '')}"


def setup_database(
    admin: psycopg.Connection,
    settings: Settings,
    hints: Hints,
    *,
    ro_role: str,
    etl_role: str,
    ro_password: str | None = None,
    etl_password: str | None = None,
    reset_passwords: bool = False,
) -> SetupReport:
    ro_role, etl_role = _check_name(ro_role), _check_name(etl_role)
    if ro_role == etl_role:
        raise ValueError("the read-only and ETL roles must be different")
    schemas = split_csv(settings.schemas) or ["public"]
    ident = sql.Identifier
    dbname = admin.execute("SELECT current_database() AS d").fetchone()["d"]
    report = SetupReport(database=dbname)

    for role, given in ((ro_role, ro_password), (etl_role, etl_password)):
        exists = admin.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (role,)).fetchone()
        if exists and not reset_passwords:
            report.existing.append(role)
            continue
        password = given or secrets.token_urlsafe(18)
        verb = (
            "ALTER ROLE {} WITH LOGIN PASSWORD {}"
            if exists
            else (
                "CREATE ROLE {} LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION "
                "NOBYPASSRLS PASSWORD {}"
            )
        )
        admin.execute(sql.SQL(verb).format(ident(role), sql.Literal(password)))
        (report.existing if exists else report.created).append(role)
        report.passwords[role] = password

    for stmt, args in (
        ("ALTER ROLE {} SET default_transaction_read_only = on", (ident(ro_role),)),
        ("ALTER ROLE {} SET statement_timeout = {}", (ident(ro_role), sql.Literal("10s"))),
        ("ALTER ROLE {} SET search_path = {}", (ident(ro_role), ident(schemas[0]))),
        ("ALTER ROLE {} SET statement_timeout = {}", (ident(etl_role), sql.Literal("60s"))),
        ("ALTER ROLE {} SET search_path = {}", (ident(etl_role), ident("staging"))),
        (
            "GRANT CONNECT ON DATABASE {} TO {}, {}",
            (ident(dbname), ident(ro_role), ident(etl_role)),
        ),
        ("CREATE SCHEMA IF NOT EXISTS {}", (ident("staging"),)),
        ("GRANT USAGE, CREATE ON SCHEMA staging TO {}", (ident(etl_role),)),
        ("GRANT USAGE ON SCHEMA staging TO {}", (ident(ro_role),)),
        (
            "ALTER DEFAULT PRIVILEGES FOR ROLE {} IN SCHEMA staging GRANT SELECT ON TABLES TO {}",
            (ident(etl_role), ident(ro_role)),
        ),
        (
            "ALTER DEFAULT PRIVILEGES IN SCHEMA staging GRANT SELECT ON TABLES TO {}",
            (ident(ro_role),),
        ),
    ):
        admin.execute(sql.SQL(stmt).format(*args))
    for schema in schemas:
        admin.execute(
            sql.SQL("REVOKE CREATE ON SCHEMA {} FROM {}, {}").format(
                ident(schema), ident(ro_role), ident(etl_role)
            )
        )
    admin.execute(
        sql.SQL("REVOKE CREATE ON DATABASE {} FROM {}, {}").format(
            ident(dbname), ident(ro_role), ident(etl_role)
        )
    )
    admin.execute(
        """
        CREATE TABLE IF NOT EXISTS staging._loads (
            id bigserial PRIMARY KEY, table_name text NOT NULL, source text NOT NULL,
            row_count integer, status text NOT NULL, loaded_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    admin.execute(sql.SQL("ALTER TABLE staging._loads OWNER TO {}").format(ident(etl_role)))
    admin.commit()

    report.granted = db.apply_grants(admin, hints, schemas, ro_role)
    report.etl_granted = db.apply_etl_grants(
        admin, hints, split_csv(settings.etl_readable_columns), schemas, etl_role
    )
    report.problems = verify(admin, settings, hints, ro_role=ro_role, etl_role=etl_role)
    return report


def _probe_create(admin: psycopg.Connection, schema: str) -> bool:
    """Whether the current (SET ROLE) role can create a table in `schema`."""
    admin.execute("SAVEPOINT probe")
    try:
        admin.execute(
            sql.SQL("CREATE TABLE {}.dp_setup_probe (x int)").format(sql.Identifier(schema))
        )
        return True
    except psycopg.Error:
        return False
    finally:
        admin.execute("ROLLBACK TO SAVEPOINT probe")


def verify(
    admin: psycopg.Connection,
    settings: Settings,
    hints: Hints,
    *,
    ro_role: str | None = None,
    etl_role: str | None = None,
) -> dict[str, list[str]]:
    """Check both roles by assuming them (no passwords needed). role -> problems."""
    ro_role = ro_role or settings.ro_role
    etl_role = etl_role or settings.etl_role
    schemas = split_csv(settings.schemas) or ["public"]
    result: dict[str, list[str]] = {}
    for role, kind in ((ro_role, "ro"), (etl_role, "etl")):
        admin.execute(sql.SQL("SET ROLE {}").format(sql.Identifier(role)))
        try:
            problems = guard.check_role(admin, kind, hints, settings)  # type: ignore[arg-type]
            # Behavioural probes, in addition to the privilege checks above.
            for schema in [*schemas, "staging"]:
                allowed = kind == "etl" and schema == "staging"
                if _probe_create(admin, schema) and not allowed:
                    problems.append(f"a CREATE TABLE in schema {schema!r} succeeded")
            if kind == "etl" and not _probe_create(admin, "staging"):
                problems.append("the ETL role cannot create tables in staging")
        finally:
            admin.execute("RESET ROLE")
            admin.rollback()
        result[role] = problems
    return result
