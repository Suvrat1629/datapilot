from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table as RichTable

from datapilot import db
from datapilot.config import get_settings
from datapilot.hints import DEMO_HINTS_PATH, load_hints

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
