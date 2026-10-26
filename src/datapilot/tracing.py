"""Per-request observability: a JSONL log (always) and Langfuse callbacks (when configured).

Observability must never break a request, so every failure here is swallowed with a warning.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from datapilot.config import get_settings
from datapilot.sql_analyst.state import SqlState

log = logging.getLogger("datapilot.tracing")

DEFAULT_LOG_PATH = Path("logs/traces.jsonl")


def new_request_id() -> str:
    return uuid.uuid4().hex[:8]


def trace_record(state: SqlState, request_id: str, latency_s: float) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "question": state.user_question,
        "curated_question": state.curated_question,
        "sql": state.sql,
        "safety": state.safety.model_dump() if state.safety else None,
        "attempts": state.attempts,
        "attempt_log": state.attempt_log,
        "status": state.status,
        "reason": state.reason,
        "row_count": len(state.rows) if state.rows is not None else None,
        "truncated": state.truncated,
        "llm_calls": state.llm_calls,
        "tokens_in": state.tokens_in,
        "tokens_out": state.tokens_out,
        "latency_ms": round(latency_s * 1000),
    }


def log_request(
    state: SqlState, request_id: str, latency_s: float, path: Path = DEFAULT_LOG_PATH
) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(trace_record(state, request_id, latency_s)) + "\n")
    except OSError as e:
        log.warning("could not write trace log: %s", e)


def langfuse_config(request_id: str) -> dict[str, Any]:
    """LangChain run config carrying a Langfuse handler, or {} when Langfuse is not set up."""
    s = get_settings()
    if not (s.langfuse_host and s.langfuse_public_key and s.langfuse_secret_key):
        return {}
    try:
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler

        Langfuse(
            public_key=s.langfuse_public_key, secret_key=s.langfuse_secret_key, host=s.langfuse_host
        )
        return {"callbacks": [CallbackHandler()], "metadata": {"request_id": request_id}}
    except Exception as e:  # SDK/server version mismatch, network, bad keys
        log.warning("Langfuse tracing disabled: %s", e)
        return {}


def flush_langfuse() -> None:
    """Send buffered traces. The SDK exports in the background, so a short-lived process
    must flush before exiting or the trace is lost."""
    s = get_settings()
    if not (s.langfuse_host and s.langfuse_public_key and s.langfuse_secret_key):
        return
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception as e:
        log.warning("Langfuse flush failed: %s", e)
