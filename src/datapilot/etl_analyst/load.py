"""Atomic, idempotent loads into the staging schema."""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Callable

import pandas as pd
import psycopg
from psycopg import sql

from datapilot import db
from datapilot.config import Settings
from datapilot.etl_analyst.store import EtlToolError, LoadRecord

SCHEMA = "staging"
IDENT = re.compile(r"^[a-z][a-z0-9_]{0,62}$")
COLUMN_IDENT = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


def validate_table_name(name: str) -> str:
    if not IDENT.match(name) or name.startswith("_") or name.endswith("__new"):
        raise EtlToolError(
            f"invalid table name {name!r}: use lowercase letters, digits and underscores, "
            "starting with a letter (names starting with _ are reserved)"
        )
    return name


def _pg_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_integer_dtype(series):
        return "bigint"
    if pd.api.types.is_float_dtype(series):
        return "double precision"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "timestamptz"
    values = series.dropna()
    if len(values) and all(
        isinstance(v, dt.date) and not isinstance(v, dt.datetime) for v in values
    ):
        return "date"
    return "text"


def _python_rows(df: pd.DataFrame, types: list[str]) -> list[tuple]:
    clean = df.astype(object).where(df.notna(), None)
    rows = []
    for rec in clean.itertuples(index=False, name=None):
        row = []
        for value, typ in zip(rec, types, strict=True):
            if value is None:
                row.append(None)
            elif typ == "text":
                row.append(str(value))
            elif typ == "timestamptz":
                row.append(pd.Timestamp(value).to_pydatetime())
            elif typ in ("bigint",):
                row.append(int(value))
            elif typ == "double precision":
                row.append(float(value))
            else:
                row.append(value)
        rows.append(tuple(row))
    return rows


def _record_load(
    connect: Callable[[], psycopg.Connection],
    table: str,
    source: str,
    rows: int | None,
    status: str,
) -> None:
    with connect() as conn:
        conn.execute(
            f"INSERT INTO {SCHEMA}._loads (table_name, source, row_count, status) "
            "VALUES (%s, %s, %s, %s)",
            (table, source[:500], rows, status),
        )
        conn.commit()


def load_staging(
    df: pd.DataFrame,
    table: str,
    source: str,
    settings: Settings,
    connect: Callable[[], psycopg.Connection] | None = None,
) -> LoadRecord:
    """Replace staging.<table> with df atomically.

    The new data is written to staging.<table>__new and swapped in inside one transaction
    (Postgres DDL is transactional), so a failure at any point leaves the previous table
    untouched. Re-running the same load produces the same table, never duplicates.
    """
    connect = connect or (lambda: db.connect("etl"))
    validate_table_name(table)
    if df.empty:
        raise EtlToolError("nothing to load: the dataset has no rows")
    if len(df) > settings.etl_max_rows:
        raise EtlToolError(f"too many rows ({len(df):,}); the limit is {settings.etl_max_rows:,}")
    bad = [c for c in df.columns if not COLUMN_IDENT.match(str(c))]
    if bad:
        raise EtlToolError(
            f"column names must be lowercase snake_case identifiers; rename {bad} "
            "with a rename step"
        )
    if len(set(df.columns)) != len(df.columns):
        raise EtlToolError("duplicate column names; rename or drop one")

    types = [_pg_type(df[c]) for c in df.columns]
    rows = _python_rows(df, types)
    tmp = f"{table}__new"
    ident = sql.Identifier
    cols_ddl = sql.SQL(", ").join(
        sql.SQL("{} {}").format(ident(c), sql.SQL(t))
        for c, t in zip(df.columns, types, strict=True)
    )

    try:
        with connect() as conn:
            conn.execute(sql.SQL("DROP TABLE IF EXISTS {}.{}").format(ident(SCHEMA), ident(tmp)))
            conn.execute(
                sql.SQL("CREATE TABLE {}.{} ({})").format(ident(SCHEMA), ident(tmp), cols_ddl)
            )
            with conn.cursor() as cur:
                copy_sql = sql.SQL("COPY {}.{} ({}) FROM STDIN").format(
                    ident(SCHEMA),
                    ident(tmp),
                    sql.SQL(", ").join(ident(c) for c in df.columns),
                )
                with cur.copy(copy_sql) as cp:
                    for row in rows:
                        cp.write_row(row)
            conn.execute(sql.SQL("DROP TABLE IF EXISTS {}.{}").format(ident(SCHEMA), ident(table)))
            conn.execute(
                sql.SQL("ALTER TABLE {}.{} RENAME TO {}").format(
                    ident(SCHEMA), ident(tmp), ident(table)
                )
            )
            conn.execute(
                f"INSERT INTO {SCHEMA}._loads (table_name, source, row_count, status) "
                "VALUES (%s, %s, %s, 'ok')",
                (table, source[:500], len(df)),
            )
            conn.commit()
    except EtlToolError:
        raise
    except Exception as e:
        # The transaction rolled back, taking any in-transaction log row with it, so the
        # failure is recorded separately afterwards (best effort).
        try:
            _record_load(connect, table, source, None, "failed")
        except Exception:
            pass
        raise EtlToolError(
            f"load failed and was rolled back: {type(e).__name__}: {str(e)[:200]}"
        ) from None

    return LoadRecord(table=table, rows=len(df), columns=list(df.columns), source=source)
