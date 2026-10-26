from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table as RichTable

from datapilot import db
from datapilot.config import get_settings
from datapilot.hints import DEMO_HINTS_PATH, load_hints
from datapilot.sql_analyst.graph import run_sql_analyst
from datapilot.tracing import flush_langfuse, langfuse_config, log_request, new_request_id

app = typer.Typer(
    help="DataPilot: natural-language SQL and ETL agents over PostgreSQL.", no_args_is_help=True
)
demo_app = typer.Typer(help="Demo dataset commands.", no_args_is_help=True)
db_app = typer.Typer(help="Database commands.", no_args_is_help=True)
eval_app = typer.Typer(help="Evaluate the SQL Analyst.", no_args_is_help=True)
app.add_typer(demo_app, name="demo")
app.add_typer(eval_app, name="eval")
app.add_typer(db_app, name="db")

console = Console()


@demo_app.command("load")
def demo_load() -> None:
    """Create the demo schema, generate synthetic data, and apply grants."""
    from datapilot.demo.generate import generate

    hints = load_hints(DEMO_HINTS_PATH)
    with db.connect("admin") as admin:
        counts = generate(admin)
        granted = db.apply_grants(admin, hints)

    t = RichTable(title="Demo data loaded")
    t.add_column("what")
    t.add_column("rows", justify="right")
    for k, v in counts.items():
        t.add_row(k, f"{v:,}")
    console.print(t)
    console.print(f"Grants applied to {len(granted)} tables for datapilot_ro.")


@db_app.command("apply-grants")
def db_apply_grants() -> None:
    """Apply column-level SELECT grants to the read-only role, honouring hidden_columns."""
    hints = load_hints(get_settings().hints_path)
    with db.connect("admin") as admin:
        for w in db.check_hints(admin, hints):
            console.print(f"[yellow]warning:[/yellow] {w}")
        granted = db.apply_grants(admin, hints)
    for table, cols in granted.items():
        console.print(f"[bold]{table}[/bold]: {', '.join(cols)}")
    if hints.hidden_columns:
        console.print(f"Hidden: {', '.join(hints.hidden_columns)}")


@db_app.command("schema")
def db_schema() -> None:
    """Print the schema context exactly as the agent sees it (read-only role)."""
    hints = load_hints(get_settings().hints_path)
    with db.connect("ro") as conn:
        tables = db.introspect(conn, hints)
    console.print(
        db.render_schema_context(tables, hints), markup=False, highlight=False, soft_wrap=True
    )


MAX_TABLE_ROWS = 20


def _render_result(state, show_sql: bool, request_id: str, latency: float) -> None:
    if state.status == "ok":
        console.print(Panel(Markdown(state.answer or ""), title="DataPilot", title_align="left"))
        if state.rows:
            t = RichTable()
            for c in state.columns:
                t.add_column(c)
            for r in state.rows[:MAX_TABLE_ROWS]:
                t.add_row(*(str(r[c]) for c in state.columns))
            console.print(t)
            if len(state.rows) > MAX_TABLE_ROWS:
                console.print(f"[dim]showing {MAX_TABLE_ROWS} of {len(state.rows)} rows[/dim]")
        if state.truncated:
            console.print("[yellow]Results were truncated by the row cap.[/yellow]")
    else:
        border = "yellow" if state.status == "refused" else "red"
        console.print(
            Panel(state.reason or "Something went wrong.", title="DataPilot", border_style=border)
        )
    if show_sql and state.sql:
        console.print("[bold]SQL[/bold]")
        console.print(state.sql, markup=False, highlight=False, soft_wrap=True)
    mark = {"ok": "[green]\u2714 ok[/green]", "refused": "[yellow]\u2716 refused[/yellow]"}.get(
        state.status or "", "[red]\u2716 failed[/red]"
    )
    console.print(
        f"{mark}   {state.attempts} attempt(s)   {latency:.1f}s   "
        f"{state.llm_calls} model calls   trace {request_id}",
        highlight=False,
    )


@app.command()
def ask(
    question: str = typer.Argument(..., help="A question about the data, in plain English."),
    show_sql: bool = typer.Option(False, "--show-sql", help="Print the SQL that was run."),
    as_json: bool = typer.Option(False, "--json", help="Print the full result as JSON."),
) -> None:
    """Ask one question; the SQL Analyst answers it from the database."""
    request_id = new_request_id()
    start = time.monotonic()
    state = run_sql_analyst(question, config=langfuse_config(request_id))
    latency = time.monotonic() - start
    log_request(state, request_id, latency)
    flush_langfuse()

    if as_json:
        payload = state.model_dump(mode="json", exclude={"schema_context", "history"})
        payload.update(request_id=request_id, latency_s=round(latency, 2))
        typer.echo(json.dumps(payload, default=str))
    else:
        _render_result(state, show_sql, request_id, latency)
    if state.status != "ok":
        raise typer.Exit(1)


# ---- eval ---------------------------------------------------------------------------------


def _split(csv: str | None) -> list[str]:
    return [x.strip() for x in (csv or "").split(",") if x.strip()]


@eval_app.command("run")
def eval_run(
    label: str = typer.Option("baseline", help="Name for this run (used in the saved file)."),
    judge: bool = typer.Option(True, "--judge/--no-judge", help="Use the LLM judge."),
    low: str | None = typer.Option(None, help="Override LLM_LOW (provider:model)."),
    high: str | None = typer.Option(None, help="Override LLM_HIGH (provider:model)."),
    only: str | None = typer.Option(None, help="Comma-separated case ids to run."),
    answers_only: bool = typer.Option(False, "--answers-only", help="Skip the safety cases."),
    no_langfuse: bool = typer.Option(
        False, "--no-langfuse", help="Do not send anything to Langfuse."
    ),
) -> None:
    """Run the eval set against the live models and score it against the database."""
    from datapilot.config import get_settings
    from datapilot.eval import langfuse_sink
    from datapilot.eval.cases import ALL_CASES, ANSWER_CASES
    from datapilot.eval.report import target_checks
    from datapilot.eval.runner import integrity_check, run_eval, save_run, summarize
    from datapilot.sql_analyst.nodes import Deps

    if low:
        os.environ["LLM_LOW"] = low
    if high:
        os.environ["LLM_HIGH"] = high
    os.environ["JUDGE_ENABLED"] = "true" if judge else "false"
    if no_langfuse:
        os.environ["LANGFUSE_PUBLIC_KEY"] = ""
    get_settings.cache_clear()
    settings = get_settings()

    cases = list(ANSWER_CASES if answers_only else ALL_CASES)
    wanted = set(_split(only))
    if wanted:
        cases = [c for c in cases if c.id in wanted]
        if not cases:
            console.print(f"[red]No cases match {sorted(wanted)}[/red]")
            raise typer.Exit(2)

    session = f"eval-{label}-{time.strftime('%Y%m%d-%H%M%S')}"
    if langfuse_sink.sync_dataset(cases):
        console.print(f"Langfuse session: {session}")

    def config_for(case_id: str) -> dict:
        return langfuse_config(
            f"{label}-{case_id}", session_id=session, tags=["eval", label, case_id]
        )

    def show(res) -> None:
        mark = "[green]\u2714[/green]" if res.passed else "[red]\u2716[/red]"
        extra = f"   [dim]{res.detail[:90]}[/dim]" if not res.passed else ""
        console.print(
            f" {mark} {res.id:32} {res.status or '-':8} att={res.attempts} "
            f"calls={res.llm_calls:<2} {res.latency_s:5.1f}s{extra}",
            highlight=False,
        )

    console.print(
        f"[bold]Eval '{label}'[/bold]: {len(cases)} cases | low={settings.llm_low} "
        f"high={settings.llm_high} judge={'on' if judge else 'off'}"
    )
    results, aborted = run_eval(
        cases, Deps(), config_factory=config_for, on_result=show, scorer=langfuse_sink.score_result
    )
    flush_langfuse()
    langfuse_sink.flush()

    integrity = integrity_check()
    summary = summarize(results)
    config = {
        "llm_low": settings.llm_low,
        "llm_high": settings.llm_high,
        "judge_enabled": judge,
        "max_sql_retries": settings.max_sql_retries,
        "session": session,
    }
    path = save_run(label, config, results, aborted, integrity)

    t = RichTable(title="Summary")
    t.add_column("metric")
    t.add_column("value", justify="right")
    t.add_row("accuracy", f"{summary['answer_correct']}/{summary['answer_cases']}")
    t.add_row("correct on first attempt", str(summary["first_attempt_correct"]))
    t.add_row(
        "wrong answers / false refusals / failures",
        f"{summary['wrong_answers']} / {summary['false_refusals']} / {summary['failures']}",
    )
    t.add_row("retried -> recovered", f"{summary['retry_recovered']}/{summary['retried_cases']}")
    t.add_row("judge objections", str(summary["judge_objections"]))
    t.add_row("safety cases passed", f"{summary['safety_passed']}/{summary['safety_cases']}")
    t.add_row("latency p50 / p95", f"{summary['latency_p50_s']}s / {summary['latency_p95_s']}s")
    t.add_row(
        "mean attempts / model calls", f"{summary['mean_attempts']} / {summary['mean_llm_calls']}"
    )
    t.add_row("tokens in / out", f"{summary['tokens_in']:,} / {summary['tokens_out']:,}")
    t.add_row(
        "data integrity", "[green]ok[/green]" if integrity[0] else f"[red]{integrity[1]}[/red]"
    )
    console.print(t)
    checks = target_checks(summary)
    for name, ok in checks.items():
        console.print(f" {'[green]PASS[/green]' if ok else '[red]FAIL[/red]'}  target: {name}")
    if aborted:
        console.print(
            "[red]Run aborted early: provider quota exhausted. Results are partial.[/red]"
        )
    console.print(f"Saved: {path}")
    if aborted or not integrity[0] or not all(checks.values()):
        raise typer.Exit(1)


@eval_app.command("list")
def eval_list() -> None:
    """List saved eval runs."""
    from datapilot.eval.report import list_runs
    from datapilot.eval.runner import load_run

    runs = list_runs()
    if not runs:
        console.print("No saved runs yet. Try `datapilot eval run`.")
        return
    t = RichTable()
    for col in ("file", "accuracy", "safety", "p50", "models", "judge"):
        t.add_column(col)
    for p in runs:
        r = load_run(p)
        s, c = r["summary"], r["config"]
        t.add_row(
            p.name,
            f"{s['answer_correct']}/{s['answer_cases']}",
            f"{s['safety_passed']}/{s['safety_cases']}",
            f"{s['latency_p50_s']}s",
            f"{c['llm_low'].split(':')[-1]} + {c['llm_high'].split(':')[-1]}",
            "on" if c["judge_enabled"] else "off",
        )
    console.print(t)


@eval_app.command("compare")
def eval_compare(
    a: Annotated[Path, typer.Argument()], b: Annotated[Path, typer.Argument()]
) -> None:
    """Compare two saved runs: metric deltas and cases whose result changed."""
    from datapilot.eval.report import compare_runs

    out = compare_runs(a, b)
    t = RichTable(title=f"{out['a']['label']}  ->  {out['b']['label']}")
    for col in ("metric", "A", "B", "change"):
        t.add_column(col, justify="right" if col != "metric" else "left")
    for r in out["rows"]:
        d = "" if r["delta"] is None else f"{r['delta']:+g}"
        t.add_row(r["metric"], str(r["a"]), str(r["b"]), d)
    console.print(t)
    for c in out["changed"]:
        arrow = "[green]now passes[/green]" if c["b"] else "[red]now fails[/red]"
        console.print(f" {c['id']}: {arrow} {c['detail_b'][:100]}")
    if not out["changed"]:
        console.print("No case changed between the two runs.")
