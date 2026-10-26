from __future__ import annotations

import json
import time

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
app.add_typer(demo_app, name="demo")
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
