"""Per-request observability: a JSONL log (always) and Langfuse callbacks (when configured).

Observability must never break a request, so every failure here is swallowed with a warning.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from functools import lru_cache
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


def langfuse_configured() -> bool:
    s = get_settings()
    return bool(s.langfuse_host and s.langfuse_public_key and s.langfuse_secret_key)


@lru_cache
def langfuse_client():
    """One Langfuse client built from our settings. The SDK itself only reads environment
    variables, but our keys live in .env, so every caller must go through here."""
    from langfuse import Langfuse

    s = get_settings()
    return Langfuse(
        public_key=s.langfuse_public_key, secret_key=s.langfuse_secret_key, host=s.langfuse_host
    )


def langfuse_config(
    request_id: str, *, session_id: str | None = None, tags: list[str] | None = None
) -> dict[str, Any]:
    """LangChain run config carrying a Langfuse handler, or {} when Langfuse is not set up."""
    if not langfuse_configured():
        return {}
    try:
        from langfuse.langchain import CallbackHandler

        langfuse_client()
        metadata: dict[str, Any] = {"request_id": request_id}
        if session_id:
            metadata["langfuse_session_id"] = session_id
        if tags:
            metadata["langfuse_tags"] = tags
        return {"callbacks": [CallbackHandler()], "metadata": metadata}
    except Exception as e:  # SDK/server version mismatch, network, bad keys
        log.warning("Langfuse tracing disabled: %s", e)
        return {}


def flush_langfuse() -> None:
    """Send buffered traces. The SDK exports in the background, so a short-lived process
    must flush before exiting or the trace is lost."""
    if not langfuse_configured():
        return
    try:
        langfuse_client().flush()
    except Exception as e:
        log.warning("Langfuse flush failed: %s", e)
