"""One JSON shape for a finished chat turn, used by `chat --json` and the web API."""

from __future__ import annotations

from typing import Any

from datapilot.parent.charts import suggest_chart
from datapilot.parent.graph import TurnResult

MAX_PAYLOAD_ROWS = 200


def turn_payload(turn: TurnResult, request_id: str, latency_s: float) -> dict[str, Any]:
    """Compact, UI-ready summary. Never includes the schema context or chat history."""
    sql = turn.sql
    ok_rows = sql is not None and turn.status == "ok" and sql.rows is not None
    rows = (sql.rows or [])[:MAX_PAYLOAD_ROWS] if ok_rows else []
    columns = list(sql.columns) if ok_rows else []
    return {
        "request_id": request_id,
        "message": turn.message,
        "route": turn.route,
        "status": turn.status,
        "text": turn.text,
        "sql": sql.sql if sql is not None else None,
        "assumptions": list(sql.assumptions) if sql is not None else [],
        "columns": columns,
        "rows": rows,
        "row_count": len(sql.rows or []) if ok_rows else 0,
        "truncated": bool(ok_rows and (sql.truncated or len(sql.rows or []) > len(rows))),
        "chart": suggest_chart(columns, rows) if ok_rows else None,
        "loads": [
            {"table": ld.table, "rows": ld.rows, "columns": ld.columns}
            for ld in (turn.etl.loads if turn.etl is not None else [])
        ],
        "llm_calls": turn.llm_calls,
        "latency_s": round(latency_s, 2),
    }
