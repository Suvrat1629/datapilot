"""Tools the ETL agent can call. Each returns a short string; failures are returned, not raised,
so the model can read the message and adjust."""

from __future__ import annotations

import functools
import json
import re
from collections.abc import Callable
from typing import Any

import httpx
import pandas as pd
import psycopg
from langchain_core.tools import BaseTool, StructuredTool
from psycopg import sql

from datapilot import db
from datapilot.config import Settings
from datapilot.etl_analyst import http, load, transforms
from datapilot.etl_analyst.store import DataStore, EtlToolError

PLACEHOLDER = re.compile(r"\{([a-z_][a-z0-9_]*)\}")
PREVIEW_ROWS = 5
IDENT = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


def _describe(handle: str, df: pd.DataFrame) -> str:
    cols = ", ".join(f"{c}:{t}" for c, t in zip(df.columns, df.dtypes.astype(str), strict=True))
    cols = cols if len(cols) < 600 else cols[:600] + " ..."
    preview = df.head(PREVIEW_ROWS).astype(object).where(df.head(PREVIEW_ROWS).notna(), None)
    return (
        f"{handle}: {len(df)} rows. Columns: {cols}\n"
        f"First rows: {json.dumps(preview.to_dict('records'), default=str)[:900]}"
    )


def _walk(data: Any, path: str | None) -> Any:
    for part in [p for p in (path or "").split(".") if p]:
        if isinstance(data, dict) and part in data:
            data = data[part]
        elif isinstance(data, list) and part.isdigit() and int(part) < len(data):
            data = data[int(part)]
        else:
            keys = list(data) if isinstance(data, dict) else type(data).__name__
            raise EtlToolError(f"data_path {path!r}: no {part!r} here. Available: {keys}")
    return data


def to_frame(data: Any) -> pd.DataFrame:
    """Normalise JSON into rows: list of objects, or an object of parallel arrays."""
    if isinstance(data, list):
        if not all(isinstance(x, dict) for x in data):
            raise EtlToolError("expected a list of objects at data_path")
        return pd.json_normalize(data, sep="_")
    if isinstance(data, dict):
        lists = {k: v for k, v in data.items() if isinstance(v, list)}
        if lists and len({len(v) for v in lists.values()}) == 1:
            return pd.DataFrame(lists)
        return pd.json_normalize(data, sep="_")
    raise EtlToolError("data_path must point to an object or a list of objects")


def _fill(params: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    def sub(v: Any) -> Any:
        if not isinstance(v, str):
            return v

        def repl(m: re.Match) -> str:
            if m.group(1) not in row:
                raise EtlToolError(
                    f"placeholder {{{m.group(1)}}} is not a column of the for_each dataset "
                    f"(columns: {list(row)})"
                )
            return str(row[m.group(1)])

        return PLACEHOLDER.sub(repl, v)

    return {k: sub(v) for k, v in params.items()}


def build_tools(
    store: DataStore,
    settings: Settings,
    *,
    connect: Callable[[], psycopg.Connection] | None = None,
    client: httpx.Client | None = None,
) -> list[BaseTool]:
    connect = connect or (lambda: db.connect("etl"))

    def guarded(fn: Callable[..., str]) -> Callable[..., str]:
        @functools.wraps(fn)
        def run(*a: Any, **kw: Any) -> str:
            try:
                return fn(*a, **kw)
            except EtlToolError as e:
                return f"ERROR: {e}"
            except psycopg.errors.InsufficientPrivilege:
                return "ERROR: permission denied; that table or column is not readable for ETL"
            except psycopg.Error as e:
                return f"ERROR: database error: {(str(e).splitlines() or [''])[0][:200]}"

        return run

    def list_tables() -> str:
        """List tables the ETL role can read (reference) and the staging tables that exist."""
        with connect() as conn:
            rows = conn.execute(
                "SELECT table_schema, table_name, column_name FROM information_schema.columns "
                "WHERE table_schema IN ('public', 'staging') AND table_name NOT LIKE '\\_%' "
                "ORDER BY 1, 2, ordinal_position"
            ).fetchall()
        if not rows:
            return "No readable tables."
        grouped: dict[str, list[str]] = {}
        for r in rows:
            grouped.setdefault(f"{r['table_schema']}.{r['table_name']}", []).append(
                r["column_name"]
            )
        return "\n".join(f"{t}({', '.join(c)})" for t, c in grouped.items())

    def read_reference(table: str, columns: list[str], limit: int = 100) -> str:
        """Read columns of a reference table (e.g. public.cities) into a dataset handle."""
        schema, _, name = table.rpartition(".")
        schema = schema or "public"
        if not (IDENT.match(schema) and IDENT.match(name) and all(IDENT.match(c) for c in columns)):
            raise EtlToolError("invalid table or column name")
        if not columns:
            raise EtlToolError("list the columns you need")
        limit = max(1, min(int(limit), 1000))
        query = sql.SQL("SELECT {} FROM {}.{} LIMIT {}").format(
            sql.SQL(", ").join(sql.Identifier(c) for c in columns),
            sql.Identifier(schema),
            sql.Identifier(name),
            sql.Literal(limit),
        )
        with connect() as conn:
            rows = conn.execute(query).fetchall()
        df = pd.DataFrame(rows, columns=columns)
        handle = store.put(df, f"{schema}.{name}")
        return _describe(handle, df)

    def fetch_api(
        url: str,
        params: dict[str, Any] | None = None,
        data_path: str | None = None,
        for_each: str | None = None,
        tag_columns: list[str] | None = None,
    ) -> str:
        """GET a JSON API (https, allow-listed hosts only) into a dataset handle.

        With for_each=<handle>, the request repeats once per row of that dataset; use {column}
        placeholders in param values. tag_columns copies those columns onto the fetched rows.
        data_path is a dotted path to the rows, e.g. "daily" or "results".
        """
        if for_each is None:
            store.count_fetch()
            df = to_frame(_walk(http.fetch_json(url, params, settings, client), data_path))
            handle = store.put(df, url.split("?")[0])
            return _describe(handle, df)

        base = store.get(for_each)
        tags = tag_columns or []
        missing = [c for c in tags if c not in base.columns]
        if missing:
            raise EtlToolError(f"tag_columns {missing} not in {for_each}: {list(base.columns)}")
        store.count_fetch(len(base))
        frames, failed = [], []
        for rec in base.to_dict("records"):
            try:
                data = http.fetch_json(url, _fill(params or {}, rec), settings, client)
                part = to_frame(_walk(data, data_path))
            except EtlToolError as e:
                failed.append(f"{ {c: rec[c] for c in tags} or rec }: {e}")
                continue
            for c in tags:
                part[c] = rec[c]
            frames.append(part)
        if not frames:
            raise EtlToolError(
                "every request failed. First error: " + (failed[0] if failed else "")
            )
        df = pd.concat(frames, ignore_index=True)
        handle = store.put(df, url.split("?")[0])
        note = f"\n{len(failed)} of {len(base)} requests failed, e.g. {failed[0]}" if failed else ""
        return _describe(handle, df) + note

    def inspect_data(handle: str) -> str:
        """Show shape, columns, types and first rows of a dataset handle."""
        return _describe(handle, store.get(handle))

    def transform(handle: str, steps: list[dict[str, Any]]) -> str:
        """Apply transform steps to a dataset, producing a new handle."""
        out = transforms.run_transform(store.get(handle), steps)
        new = store.put(out, f"{store.sources[handle]} (transformed)")
        return _describe(new, out)

    def load_staging(handle: str, table: str) -> str:
        """Atomically replace staging.<table> with the dataset. Re-running gives the same table."""
        rec = load.load_staging(store.get(handle), table, store.sources[handle], settings, connect)
        store.loads.append(rec)
        return f"Loaded {rec.rows} rows into staging.{rec.table} ({', '.join(rec.columns)})."

    def refuse(reason: str) -> str:
        """Decline a request that is not a safe data-loading task, with a short reason."""
        store.refused = reason
        return "Request declined. Reply with a one-sentence explanation."

    fns = [
        list_tables,
        read_reference,
        fetch_api,
        inspect_data,
        transform,
        load_staging,
        refuse,
    ]
    return [
        StructuredTool.from_function(
            func=guarded(f),
            name=f.__name__,
            description=(f.__doc__ or "").strip()
            + (f" Steps: {transforms.OPS_HELP}" if f is transform else ""),
        )
        for f in fns
    ]
