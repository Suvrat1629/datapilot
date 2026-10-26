from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

from datapilot.config import get_settings, readable_schemas, split_csv
from datapilot.hints import Hints

Role = Literal["admin", "ro", "etl"]

SAMPLE_MAX_DISTINCT = 12


def connect(role: Role = "ro") -> psycopg.Connection:
    s = get_settings()
    url = {"admin": s.admin_database_url, "ro": s.database_url, "etl": s.etl_database_url}[role]
    return psycopg.connect(url, row_factory=dict_row)


@dataclass
class Column:
    name: str
    type: str
    nullable: bool
    is_pk: bool = False
    fk: str | None = None  # "other_table.col"
    samples: list[str] = field(default_factory=list)


@dataclass
class Table:
    name: str
    columns: list[Column]
    schema: str = "public"

    @property
    def qualified(self) -> str:
        """Name to use in SQL and prompts: staging tables must be schema-qualified."""
        return self.name if self.schema == "public" else f"{self.schema}.{self.name}"


def introspect(
    conn: psycopg.Connection, hints: Hints, schemas: tuple[str, ...] | None = None
) -> list[Table]:
    """Introspect the configured schemas plus staging (where the ETL agent loads tables)."""
    tables: list[Table] = []
    for schema in schemas or readable_schemas(get_settings()):
        tables.extend(_introspect_schema(conn, hints, schema))
    return tables


def _introspect_schema(conn: psycopg.Connection, hints: Hints, schema: str) -> list[Table]:
    """Read the schema as the connected role sees it.

    information_schema only lists columns the role has privileges on, so with the
    read-only role's column-level grants, hidden columns never appear here. The hints
    filter is a second guard for the case where grants have not been applied yet.
    """
    cols = conn.execute(
        """
        SELECT table_name, column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema = %s
        ORDER BY table_name, ordinal_position
        """,
        (schema,),
    ).fetchall()

    pks = {
        (r["table_name"], r["column_name"])
        for r in conn.execute(
            """
            SELECT cl.relname AS table_name, a.attname AS column_name
            FROM pg_constraint c
            JOIN pg_class cl ON cl.oid = c.conrelid
            JOIN pg_namespace n ON n.oid = cl.relnamespace
            JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = ANY (c.conkey)
            WHERE c.contype = 'p' AND n.nspname = %s
            """,
            (schema,),
        )
    }
    fks = {
        (r["table_name"], r["column_name"]): f"{r['ref_table']}.{r['ref_column']}"
        for r in conn.execute(
            """
            SELECT cl.relname AS table_name, a.attname AS column_name,
                   cf.relname AS ref_table, af.attname AS ref_column
            FROM pg_constraint c
            JOIN pg_class cl ON cl.oid = c.conrelid
            JOIN pg_class cf ON cf.oid = c.confrelid
            JOIN pg_namespace n ON n.oid = cl.relnamespace
            JOIN LATERAL unnest(c.conkey, c.confkey) AS k(attnum, fattnum) ON true
            JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = k.attnum
            JOIN pg_attribute af ON af.attrelid = c.confrelid AND af.attnum = k.fattnum
            WHERE c.contype = 'f' AND n.nspname = %s
            """,
            (schema,),
        )
    }

    tables: dict[str, Table] = {}
    for r in cols:
        t, c = r["table_name"], r["column_name"]
        if t.startswith("_"):  # internal bookkeeping such as staging._loads
            continue
        if hints.is_hidden(t, c, schema):
            continue
        col = Column(
            name=c,
            type=r["data_type"],
            nullable=r["is_nullable"] == "YES",
            is_pk=(t, c) in pks,
            fk=fks.get((t, c)),
        )
        tables.setdefault(t, Table(t, [], schema)).columns.append(col)

    for table in tables.values():
        for col in table.columns:
            if col.type in ("text", "character varying") and not col.is_pk and not col.fk:
                col.samples = _low_cardinality_values(conn, schema, table.name, col.name)
    return list(tables.values())


def _low_cardinality_values(conn, schema: str, table: str, column: str) -> list[str]:
    q = sql.SQL("SELECT DISTINCT {c} AS v FROM {s}.{t} WHERE {c} IS NOT NULL LIMIT %s").format(
        c=sql.Identifier(column), s=sql.Identifier(schema), t=sql.Identifier(table)
    )
    rows = conn.execute(q, (SAMPLE_MAX_DISTINCT + 1,)).fetchall()
    if len(rows) > SAMPLE_MAX_DISTINCT:
        return []
    return sorted(str(r["v"]) for r in rows)


def render_schema_context(tables: list[Table], hints: Hints) -> str:
    """Plain-text schema description for the LLM prompt."""
    lines: list[str] = []
    for t in tables:
        th = hints.for_table(t.name, t.schema)
        header = f"TABLE {t.qualified}"
        if th and th.description:
            header += f" -- {th.description}"
        lines.append(header)
        for c in t.columns:
            parts = [f"  {c.name} {c.type}"]
            if c.is_pk:
                parts.append("PK")
            if c.fk:
                parts.append(f"FK -> {c.fk}")
            if not c.nullable and not c.is_pk:
                parts.append("NOT NULL")
            note = th.columns.get(c.name) if th else None
            if note:
                parts.append(f"-- {note}")
            if c.samples:
                parts.append(f"values: {', '.join(c.samples)}")
            lines.append(" ".join(parts))
        lines.append("")
    return "\n".join(lines).rstrip()


def _columns_by_schema(conn: psycopg.Connection, schemas: list[str]) -> set[tuple[str, str, str]]:
    rows = conn.execute(
        "SELECT table_schema, table_name, column_name FROM information_schema.columns "
        "WHERE table_schema = ANY(%s)",
        (schemas,),
    ).fetchall()
    return {(r["table_schema"], r["table_name"], r["column_name"]) for r in rows}


def _split_ref(ref: str) -> tuple[str, str, str] | None:
    """`table.col` (public) or `schema.table.col` -> (schema, table, col); None if malformed."""
    parts = ref.split(".")
    if len(parts) == 2 and all(parts):
        return "public", parts[0], parts[1]
    if len(parts) == 3 and all(parts):
        return parts[0], parts[1], parts[2]
    return None


def check_hints(
    conn: psycopg.Connection, hints: Hints, schemas: list[str] | None = None
) -> list[str]:
    """Warnings for hints that do not match the database (a stale hints file is dangerous:
    an unmatched hidden column means the intended protection is not applied)."""
    schemas = schemas or split_csv(get_settings().schemas) or ["public"]
    existing = _columns_by_schema(conn, schemas)
    tables = {(s, t) for s, t, _ in existing}
    warnings: list[str] = []
    for name, th in hints.tables.items():
        s, _, t = name.rpartition(".")
        s = s or "public"
        if (s, t) not in tables:
            warnings.append(f"hints: table '{name}' does not exist")
            continue
        for col in th.columns:
            if (s, t, col) not in existing:
                warnings.append(f"hints: column '{name}.{col}' does not exist")
    for ref in hints.hidden_columns:
        parsed = _split_ref(ref)
        if parsed is None or parsed not in existing:
            warnings.append(f"hints: hidden column '{ref}' does not exist, nothing is hidden by it")
    return warnings


def _qualified(schema: str, table: str) -> str:
    return table if schema == "public" else f"{schema}.{table}"


def apply_grants(
    admin: psycopg.Connection,
    hints: Hints,
    schemas: list[str] | None = None,
    role: str | None = None,
) -> dict[str, list[str]]:
    """Grant the read-only role SELECT on every non-hidden column of the configured schemas.

    Returns qualified table -> columns. Tables starting with `_` are internal and skipped.
    """
    settings = get_settings()
    schemas = schemas or split_csv(settings.schemas) or ["public"]
    role = role or settings.ro_role
    granted: dict[str, list[str]] = {}
    for schema in schemas:
        admin.execute(
            sql.SQL("REVOKE ALL ON ALL TABLES IN SCHEMA {} FROM {}").format(
                sql.Identifier(schema), sql.Identifier(role)
            )
        )
        admin.execute(
            sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(
                sql.Identifier(schema), sql.Identifier(role)
            )
        )
        rows = admin.execute(
            """
            SELECT table_name, column_name
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name NOT LIKE '\\_%%'
            ORDER BY table_name, ordinal_position
            """,
            (schema,),
        ).fetchall()
        by_table: dict[str, list[str]] = {}
        for r in rows:
            if not hints.is_hidden(r["table_name"], r["column_name"], schema):
                by_table.setdefault(r["table_name"], []).append(r["column_name"])
        for table, columns in by_table.items():
            admin.execute(
                sql.SQL("GRANT SELECT ({cols}) ON {s}.{t} TO {r}").format(
                    cols=sql.SQL(", ").join(sql.Identifier(c) for c in columns),
                    s=sql.Identifier(schema),
                    t=sql.Identifier(table),
                    r=sql.Identifier(role),
                )
            )
            granted[_qualified(schema, table)] = columns
    admin.commit()
    return granted


def apply_etl_grants(
    admin: psycopg.Connection,
    hints: Hints,
    columns: list[str],
    schemas: list[str] | None = None,
    role: str | None = None,
) -> dict[str, list[str]]:
    """Let the ETL role read only the listed `table.column` / `schema.table.column` entries.

    Revokes everything first, so re-running is idempotent and removing an entry removes the
    grant. Refuses hidden columns and entries that do not exist, instead of skipping them.
    """
    settings = get_settings()
    schemas = schemas or split_csv(settings.schemas) or ["public"]
    role = role or settings.etl_role
    wanted: dict[tuple[str, str], list[str]] = {}
    for ref in columns:
        parsed = _split_ref(ref)
        if parsed is None:
            raise ValueError(
                f"ETL_READABLE_COLUMNS entry {ref!r} must look like table.column "
                "or schema.table.column"
            )
        schema, table, col = parsed
        if hints.is_hidden(table, col, schema):
            raise ValueError(f"{ref} is a hidden column and cannot be made readable to ETL")
        if schema not in schemas:
            raise ValueError(f"{ref}: schema {schema!r} is not in SCHEMAS ({', '.join(schemas)})")
        wanted.setdefault((schema, table), []).append(col)

    existing = _columns_by_schema(admin, schemas)
    missing = [ref for ref in columns if _split_ref(ref) not in existing]
    if missing:
        raise ValueError(f"ETL_READABLE_COLUMNS names columns that do not exist: {missing}")

    for schema in schemas:
        admin.execute(
            sql.SQL("REVOKE ALL ON ALL TABLES IN SCHEMA {} FROM {}").format(
                sql.Identifier(schema), sql.Identifier(role)
            )
        )
        used = any(s == schema for s, _ in wanted)
        verb = "GRANT USAGE ON SCHEMA {} TO {}" if used else "REVOKE USAGE ON SCHEMA {} FROM {}"
        admin.execute(sql.SQL(verb).format(sql.Identifier(schema), sql.Identifier(role)))
    granted: dict[str, list[str]] = {}
    for (schema, table), cols in wanted.items():
        admin.execute(
            sql.SQL("GRANT SELECT ({cols}) ON {s}.{t} TO {r}").format(
                cols=sql.SQL(", ").join(sql.Identifier(c) for c in cols),
                s=sql.Identifier(schema),
                t=sql.Identifier(table),
                r=sql.Identifier(role),
            )
        )
        granted[_qualified(schema, table)] = cols
    admin.commit()
    return granted
