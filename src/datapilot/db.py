from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

from datapilot.config import get_settings
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


def introspect(conn: psycopg.Connection, hints: Hints, schema: str = "public") -> list[Table]:
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
        if hints.is_hidden(t, c):
            continue
        col = Column(
            name=c,
            type=r["data_type"],
            nullable=r["is_nullable"] == "YES",
            is_pk=(t, c) in pks,
            fk=fks.get((t, c)),
        )
        tables.setdefault(t, Table(t, [])).columns.append(col)

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
        th = hints.tables.get(t.name)
        header = f"TABLE {t.name}"
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


def check_hints(conn: psycopg.Connection, hints: Hints, schema: str = "public") -> list[str]:
    """Warnings for hints that do not match the database (a stale hints file is dangerous:
    an unmatched hidden column means the intended protection is not applied)."""
    rows = conn.execute(
        "SELECT table_name, column_name FROM information_schema.columns WHERE table_schema = %s",
        (schema,),
    ).fetchall()
    existing = {(r["table_name"], r["column_name"]) for r in rows}
    tables = {t for t, _ in existing}
    warnings: list[str] = []
    for name, th in hints.tables.items():
        if name not in tables:
            warnings.append(f"hints: table '{name}' does not exist")
            continue
        for col in th.columns:
            if (name, col) not in existing:
                warnings.append(f"hints: column '{name}.{col}' does not exist")
    for ref in hints.hidden_columns:
        t, _, c = ref.partition(".")
        if (t, c) not in existing:
            warnings.append(f"hints: hidden column '{ref}' does not exist, nothing is hidden by it")
    return warnings


def apply_grants(
    admin: psycopg.Connection, hints: Hints, schema: str = "public"
) -> dict[str, list[str]]:
    """Grant the read-only role SELECT on every non-hidden column. Returns table -> columns."""
    granted: dict[str, list[str]] = {}
    admin.execute(
        sql.SQL("REVOKE ALL ON ALL TABLES IN SCHEMA {} FROM datapilot_ro").format(
            sql.Identifier(schema)
        )
    )
    rows = admin.execute(
        """
        SELECT table_name, column_name
        FROM information_schema.columns
        WHERE table_schema = %s
        ORDER BY table_name, ordinal_position
        """,
        (schema,),
    ).fetchall()
    by_table: dict[str, list[str]] = {}
    for r in rows:
        if not hints.is_hidden(r["table_name"], r["column_name"]):
            by_table.setdefault(r["table_name"], []).append(r["column_name"])
    for table, columns in by_table.items():
        admin.execute(
            sql.SQL("GRANT SELECT ({cols}) ON {s}.{t} TO datapilot_ro").format(
                cols=sql.SQL(", ").join(sql.Identifier(c) for c in columns),
                s=sql.Identifier(schema),
                t=sql.Identifier(table),
            )
        )
        granted[table] = columns
    admin.commit()
    return granted
