"""LLM calls with structured output, parse repair, and rate-limit backoff.

Everything above this module calls `call_llm` through `Deps`, so tests inject a scripted fake
and nodes never know which provider is behind a tier.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, ValidationError

from datapilot.config import Tier, get_llm

BACKOFF_ATTEMPTS = 4
_TOOL_FAILURE_MARKERS = ("tool_use_failed", "Tool choice is required")
BACKOFF_BASE_SECONDS = 2.0


class LLMError(Exception):
    """The model could not produce a usable reply (provider outage, quota, bad output)."""


@dataclass
class LLMResult:
    parsed: Any  # a schema instance, or str when no schema was requested
    tokens_in: int = 0
    tokens_out: int = 0


def _is_rate_limit(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(s in text for s in ("429", "rate limit", "ratelimit", "resource_exhausted", "quota"))


def _is_daily_quota(exc: Exception) -> bool:
    """A per-day quota cannot recover within a request, so retrying only wastes time."""
    text = str(exc).lower()
    return "perday" in text or "per day" in text or "tokens per day" in text


def _with_backoff(fn):
    for attempt in range(BACKOFF_ATTEMPTS):
        try:
            return fn()
        except Exception as e:  # provider SDKs raise many unrelated types
            if _is_rate_limit(e) and _is_daily_quota(e):
                retry = re.search(r"retry in ([0-9hms.]+)", str(e))
                when = f" (resets in about {retry.group(1).split('.')[0]})" if retry else ""
                raise LLMError(
                    f"daily free-tier quota exhausted for this model{when}. "
                    "Switch LLM_LOW / LLM_HIGH in .env to another model, or try again later."
                ) from e
            if _is_rate_limit(e) and attempt < BACKOFF_ATTEMPTS - 1:
                time.sleep(BACKOFF_BASE_SECONDS * 2**attempt)
                continue
            raise LLMError(f"model call failed: {type(e).__name__}: {str(e)[:200]}") from e
    raise LLMError("model call failed")  # pragma: no cover


def _usage(msg: Any) -> tuple[int, int]:
    meta = getattr(msg, "usage_metadata", None) or {}
    return int(meta.get("input_tokens", 0)), int(meta.get("output_tokens", 0))


def _extract_json(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        return fenced.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end > start else text


def call_llm(tier: Tier, schema: type[BaseModel] | None, messages: list[BaseMessage]) -> LLMResult:
    try:
        llm = get_llm(tier)
    except Exception as e:  # missing API key, unsupported provider, bad model spec
        raise LLMError(f"model for tier '{tier}' is not configured: {str(e)[:160]}") from e

    if schema is None:
        msg = _with_backoff(lambda: llm.invoke(messages))
        tin, tout = _usage(msg)
        text = msg.content if isinstance(msg.content, str) else str(msg.content)
        return LLMResult(text.strip(), tin, tout)

    runnable = llm.with_structured_output(schema, include_raw=True)
    try:
        out = _with_backoff(lambda: runnable.invoke(messages))
    except LLMError as e:
        # Some providers reject the reply server-side when the model skips the tool call
        # (Groq: 400 tool_use_failed). Treat it like a parse failure and use the repair path.
        if not any(m in str(e) for m in _TOOL_FAILURE_MARKERS):
            raise
        out = {
            "raw": None,
            "parsed": None,
            "parsing_error": "model did not return structured output",
        }
    tin, tout = _usage(out.get("raw"))
    if out.get("parsed") is not None:
        return LLMResult(out["parsed"], tin, tout)

    # Smaller models sometimes break structured output. One repair attempt with plain JSON.
    err = out.get("parsing_error")
    repair = messages + [
        HumanMessage(
            f"Your previous reply could not be parsed ({err}). Reply with ONLY a JSON object "
            f"matching this schema, no prose:\n{json.dumps(schema.model_json_schema())}"
        )
    ]
    msg = _with_backoff(lambda: llm.invoke(repair))
    rin, rout = _usage(msg)
    text = msg.content if isinstance(msg.content, str) else str(msg.content)
    try:
        parsed = schema.model_validate_json(_extract_json(text))
    except (ValidationError, ValueError) as e:
        raise LLMError(f"model returned unparseable output after repair: {str(e)[:200]}") from e
    return LLMResult(parsed, tin + rin, tout + rout)
