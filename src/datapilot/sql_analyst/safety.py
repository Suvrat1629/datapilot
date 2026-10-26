"""Static SQL safety check (layer 1 of 3).

Defense in depth only: the real boundary is the read-only Postgres role. This layer gives a
clear refusal reason before anything executes and blocks constructs a parser can see.
It is deterministic and needs no LLM.
"""

from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from datapilot.hints import Hints

ALLOWED_SCHEMAS = frozenset({"public", "staging"})

# Anything that can write, change state, or reach outside the query.
FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.TruncateTable,
    exp.Copy,
    exp.Command,
    exp.Set,
    exp.Transaction,
    exp.Commit,
    exp.Rollback,
    exp.Grant,
    exp.Lock,  # SELECT ... FOR UPDATE / FOR SHARE
    exp.Into,  # SELECT ... INTO new_table
)

DENIED_FUNCTIONS = frozenset(
    {
        "pg_sleep",
        "pg_sleep_for",
        "pg_sleep_until",
        "set_config",
        "pg_terminate_backend",
        "pg_cancel_backend",
        "pg_reload_conf",
        "pg_rotate_logfile",
        "pg_read_file",
        "pg_read_binary_file",
        "pg_stat_file",
        "pg_ls_dir",
        "query_to_xml",
        "query_to_xml_and_xmlschema",
        "query_to_xmlschema",
        "database_to_xml",
        "schema_to_xml",
        "table_to_xml",
        "current_setting",
        "txid_current",
    }
)
DENIED_FUNCTION_PREFIXES = (
    "lo_",
    "dblink",
    "pg_ls_",
    "pg_file_",
    "pg_logical_",
    "pg_replication_",
    "pg_advisory_",
    "pg_notify",
)


@dataclass(frozen=True)
class SafetyResult:
    ok: bool
    reason: str = ""
    sql: str = ""

    @classmethod
    def refuse(cls, reason: str) -> SafetyResult:
        return cls(ok=False, reason=reason)


def _function_name(node: exp.Func) -> str:
    name = node.name if isinstance(node, exp.Anonymous) else node.sql_name()
    return name.lower()


def check_sql(
    raw: str,
    hints: Hints | None = None,
    *,
    max_rows: int = 1000,
    allowed_schemas: frozenset[str] = ALLOWED_SCHEMAS,
) -> SafetyResult:
    """Validate one generated SQL string. Returns the normalised SQL if it is allowed."""
    if not raw or not raw.strip():
        return SafetyResult.refuse("Empty query.")

    try:
        statements = [s for s in sqlglot.parse(raw, dialect="postgres") if s is not None]
    except SqlglotError as e:
        return SafetyResult.refuse(f"Could not parse the SQL: {str(e).splitlines()[0]}")

    if len(statements) == 0:
        return SafetyResult.refuse("Empty query.")
    if len(statements) > 1:
        return SafetyResult.refuse("Only a single statement is allowed.")

    stmt = statements[0]

    # Root must be a plain query. Unsupported syntax falls back to Command and lands here.
    if not isinstance(stmt, exp.Query):
        return SafetyResult.refuse("Only SELECT queries are allowed.")

    # Walk the whole tree: data-modifying CTEs, SELECT INTO and FOR UPDATE all hide
    # inside a root that parses as an ordinary SELECT.
    for node in stmt.walk():
        if isinstance(node, FORBIDDEN_NODES):
            kind = type(node).__name__.upper()
            return SafetyResult.refuse(f"{kind} is not allowed; the agent is read-only.")

    for node in stmt.find_all(exp.Func):
        name = _function_name(node)
        if name in DENIED_FUNCTIONS or name.startswith(DENIED_FUNCTION_PREFIXES):
            return SafetyResult.refuse(f"Function '{name}' is not allowed.")

    # Only allow-listed schemas. This also blocks pg_catalog and information_schema.
    cte_names = {c.alias_or_name.lower() for c in stmt.find_all(exp.CTE)}
    for table in stmt.find_all(exp.Table):
        schema = table.db.lower() if table.db else ""
        if table.catalog:
            return SafetyResult.refuse("Cross-database references are not allowed.")
        if schema and schema not in allowed_schemas:
            return SafetyResult.refuse(f"Schema '{schema}' is not allowed.")
        if (
            not schema
            and table.name.lower().startswith("pg_")
            and table.name.lower() not in cte_names
        ):
            return SafetyResult.refuse(f"Table '{table.name}' is not allowed.")

    # Best-effort early refusal for hidden columns (Postgres column grants are the real
    # enforcement). Matches by column name, so it may over-block a same-named column elsewhere.
    if hints and hints.hidden_columns:
        hidden_names = {ref.partition(".")[2].lower() for ref in hints.hidden_columns}
        for col in stmt.find_all(exp.Column):
            if col.name.lower() in hidden_names:
                return SafetyResult.refuse(f"Column '{col.name}' is hidden and cannot be queried.")

    # Optimisation only: run_query's fetch cap is authoritative.
    if not stmt.args.get("limit"):
        stmt = stmt.limit(max_rows)

    return SafetyResult(ok=True, sql=stmt.sql(dialect="postgres"))
