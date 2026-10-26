"""Optional Langfuse integration for evals: a dataset of the cases and a score per trace.

Every call is guarded: an eval must never fail because the dashboard is down.
"""

from __future__ import annotations

import logging

from datapilot.config import get_settings
from datapilot.eval.cases import AnswerCase, SafetyCase
from datapilot.eval.runner import CaseResult
from datapilot.tracing import langfuse_client, langfuse_configured

log = logging.getLogger("datapilot.eval")
DATASET_NAME = "datapilot-eval"


def enabled() -> bool:
    return langfuse_configured()


def _client():
    return langfuse_client()


def sync_dataset(cases: list[AnswerCase | SafetyCase]) -> bool:
    if not enabled():
        return False
    try:
        client = _client()
        try:
            client.create_dataset(
                name=DATASET_NAME, description="DataPilot SQL Analyst eval questions"
            )
        except Exception:  # already exists
            pass
        for c in cases:
            expected = (
                {"reference_sql": c.reference_sql}
                if isinstance(c, AnswerCase)
                else {"must_refuse": c.must_refuse}
            )
            client.create_dataset_item(
                dataset_name=DATASET_NAME,
                id=c.id,
                input=c.question,
                expected_output=expected,
                metadata={"kind": c.kind},
            )
        return True
    except Exception as e:
        log.warning("Langfuse dataset sync failed: %s", e)
        return False


def _post_score(host: str, auth: tuple[str, str], payload: dict) -> None:
    import httpx

    httpx.post(
        f"{host.rstrip('/')}/api/public/scores", auth=auth, json=payload, timeout=10
    ).raise_for_status()


def score_result(res: CaseResult) -> None:
    """Attach correctness and effort scores to the case's trace.

    Uses Langfuse's REST API directly: the Python SDK's score call returns HTTP 400 against
    the v3 server we run, while the REST endpoint accepts the same data.
    """
    if not (enabled() and res.trace_id):
        return
    s = get_settings()
    auth = (s.langfuse_public_key, s.langfuse_secret_key)
    try:
        _post_score(
            s.langfuse_host,
            auth,
            {
                "traceId": res.trace_id,
                "name": "correct",
                "value": 1.0 if res.passed else 0.0,
                "dataType": "NUMERIC",
                **({"comment": res.detail} if res.detail else {}),
            },
        )
        _post_score(
            s.langfuse_host,
            auth,
            {
                "traceId": res.trace_id,
                "name": "attempts",
                "value": float(res.attempts),
                "dataType": "NUMERIC",
            },
        )
    except Exception as e:
        log.warning("Langfuse scoring failed: %s", e)


def flush() -> None:
    if not enabled():
        return
    try:
        _client().flush()
    except Exception as e:
        log.warning("Langfuse flush failed: %s", e)
