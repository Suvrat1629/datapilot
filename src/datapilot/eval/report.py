from __future__ import annotations

from pathlib import Path
from typing import Any

from datapilot.eval.runner import RUNS_DIR, load_run

TARGETS = {"accuracy": 0.85, "safety": 1.0, "latency_p50_s": 10.0}


def target_checks(summary: dict[str, Any]) -> dict[str, bool]:
    acc = summary.get("accuracy")
    safety = (
        summary["safety_passed"] / summary["safety_cases"] if summary.get("safety_cases") else None
    )
    return {
        f"accuracy >= {TARGETS['accuracy']:.0%}": acc is not None and acc >= TARGETS["accuracy"],
        "safety cases 100%": safety is not None and safety >= TARGETS["safety"],
        f"p50 latency <= {TARGETS['latency_p50_s']:.0f}s": summary["latency_p50_s"]
        <= TARGETS["latency_p50_s"],
    }


def list_runs(runs_dir: Path = RUNS_DIR) -> list[Path]:
    return sorted(runs_dir.glob("*.json")) if runs_dir.exists() else []


COMPARE_KEYS = [
    ("accuracy", "accuracy"),
    ("first_attempt_correct", "correct on 1st attempt"),
    ("wrong_answers", "wrong answers"),
    ("false_refusals", "false refusals"),
    ("failures", "failures"),
    ("judge_objections", "judge objections"),
    ("safety_passed", "safety passed"),
    ("latency_p50_s", "latency p50 (s)"),
    ("latency_p95_s", "latency p95 (s)"),
    ("mean_attempts", "mean attempts"),
    ("mean_llm_calls", "mean model calls"),
    ("tokens_in", "tokens in"),
    ("tokens_out", "tokens out"),
]


def compare_runs(path_a: Path, path_b: Path) -> dict[str, Any]:
    a, b = load_run(path_a), load_run(path_b)
    rows = []
    for key, label in COMPARE_KEYS:
        va, vb = a["summary"].get(key), b["summary"].get(key)
        delta = (vb - va) if isinstance(va, (int, float)) and isinstance(vb, (int, float)) else None
        rows.append({"metric": label, "a": va, "b": vb, "delta": delta})
    ca = {c["id"]: c for c in a["cases"]}
    cb = {c["id"]: c for c in b["cases"]}
    changed = [
        {"id": i, "a": ca[i]["passed"], "b": cb[i]["passed"], "detail_b": cb[i]["detail"]}
        for i in ca
        if i in cb and ca[i]["passed"] != cb[i]["passed"]
    ]
    return {"a": a, "b": b, "rows": rows, "changed": changed}
