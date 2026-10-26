"""Pick a simple chart for a query result, or none. Pure and deterministic.

A later visualisation agent can replace `suggest_chart` without touching the API or the UI.
"""

from __future__ import annotations

import re
from typing import Any

MAX_BAR_ROWS = 50
MAX_LINE_ROWS = 400
MAX_SERIES = 3
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}([T ].*)?$")


def _kind(values: list[Any]) -> str:
    """'number', 'date', 'text' or 'empty' for the non-null values of a column."""
    present = [v for v in values if v is not None]
    if not present:
        return "empty"
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in present):
        return "number"
    if all(isinstance(v, str) and DATE_RE.match(v) for v in present):
        return "date"
    if all(isinstance(v, str) for v in present):
        return "text"
    return "other"


def _is_id(name: str) -> bool:
    return name.lower() == "id" or name.lower().endswith("_id")


def suggest_chart(columns: list[str], rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """{"type": "bar" | "line", "x": column, "y": [columns]} when the shape fits, else None."""
    if len(columns) < 2 or len(rows) < 2:
        return None
    kinds = {c: _kind([r.get(c) for r in rows]) for c in columns}
    numeric = [c for c in columns if kinds[c] == "number" and not _is_id(c)]
    labels = [c for c in columns if kinds[c] in ("text", "date")]
    # Exactly one label column, and the rest numeric measures; anything else is a table.
    if len(labels) != 1 or not 1 <= len(numeric) <= MAX_SERIES:
        return None
    if any(kinds[c] not in ("number", "text", "date") for c in columns):
        return None
    x = labels[0]
    if kinds[x] == "date":
        if len(rows) > MAX_LINE_ROWS:
            return None
        return {"type": "line", "x": x, "y": numeric}
    if len(rows) > MAX_BAR_ROWS or len({r.get(x) for r in rows}) != len(rows):
        return None
    return {"type": "bar", "x": x, "y": numeric}
