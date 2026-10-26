"""SQL Analyst graph nodes. Each returns a partial state update."""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import psycopg
from langchain_core.messages import HumanMessage, SystemMessage

from datapilot import db
from datapilot.config import Settings, get_settings, readable_schemas
from datapilot.hints import Hints, load_hints
from datapilot.llm import LLMError, LLMResult, call_llm
from datapilot.sql_analyst.context import build_context
from datapilot.sql_analyst.safety import check_sql
from datapilot.sql_analyst.state import (
    CuratedQuestion,
    JudgeVerdict,
    SafetyVerdict,
    SqlDraft,
    SqlState,
)

REPRESENT_MAX_ROWS = 50


@dataclass
class Deps:
    """Everything the nodes touch from outside, so tests can substitute each piece."""

    settings: Settings = field(default_factory=get_settings)
    hints: Hints | None = None
    call_llm: Callable[..., LLMResult] = call_llm
    connect: Callable[[], psycopg.Connection] = lambda: db.connect("ro")
    today: Callable[[], dt.date] = lambda: dt.datetime.now(dt.UTC).date()

    def __post_init__(self) -> None:
        if self.hints is None:
            self.hints = load_hints(self.settings.hints_path)


class BudgetExceeded(Exception):
    pass


CURATE_SYSTEM = """You prepare questions for a SQL analyst working on a PostgreSQL database.
Rewrite the user's latest message as ONE standalone, precise analytics question, using the chat
history to resolve references like "that" or "those". Keep every detail (filters, time ranges).
If the message is not a read-only question about data (small talk, a request to change, delete or
create data, or something unrelated), set answerable=false and give a short reason."""

GENERATE_SYSTEM = """You are an expert PostgreSQL analyst. Write ONE read-only SELECT statement that
answers the question, using only the tables and columns in the schema below.

Rules:
- A single SELECT (CTEs allowed). No DDL, no writes, no semicolon-separated extras.
- Use only listed tables/columns. Respect the column notes and the allowed values shown.
- Use explicit JOINs on the foreign keys shown. Qualify columns when joining.
- Today's date is {today}. Interpret relative dates ("last week", "this month") against it and
  state the interpretation in `assumptions`. Prefer date expressions (CURRENT_DATE, date_trunc,
  intervals) over hard-coded literal dates, so the query stays correct when re-run later.
- If the schema does not contain the data the question needs, do NOT invent columns or
  substitute other data: set `cannot_answer_reason` to what is missing and leave `sql` empty.
- Prefer clear column aliases. Add ORDER BY and LIMIT when the question asks for a top/bottom N.

SCHEMA
{schema}"""

JUDGE_SYSTEM = """You review a PostgreSQL query. Decide whether it correctly answers the question.
Check the joins, filters (including status values and date ranges), grouping and aggregation.
Only reject for a real correctness problem, and say precisely what to fix. Style is irrelevant.
Today's date is {today}. Literal dates are NOT an error when they are correct for today's date;
do not reject a query merely for being less reusable.

SCHEMA
{schema}"""

REPRESENT_SYSTEM = """You write the final answer to a business user from query results.
Be concise: 1-3 sentences. State the key numbers exactly as given. If the result is empty, say no
matching data was found. If results were truncated, say only the first rows are shown.
Mention any assumption that affects the meaning (for example how "last week" was interpreted).
Round decimals sensibly (2 places for money and rates) and use thousands separators.
Plain text only, no markdown."""


def _jsonable(v: Any) -> Any:
    if isinstance(v, Decimal):
        return float(v) if v % 1 else int(v)
    if isinstance(v, (dt.datetime, dt.date)):
        return v.isoformat()
    return v


class SqlAnalystNodes:
    def __init__(self, deps: Deps):
        self.d = deps

    # -- helpers ---------------------------------------------------------------------------

    def _call(self, state: SqlState, tier, schema, messages) -> tuple[Any, dict[str, int]]:
        if state.llm_calls >= self.d.settings.max_llm_calls_per_request:
            raise BudgetExceeded
        res = self.d.call_llm(tier, schema, messages)
        usage = {
            "llm_calls": state.llm_calls + 1,
            "tokens_in": state.tokens_in + res.tokens_in,
            "tokens_out": state.tokens_out + res.tokens_out,
        }
        return res.parsed, usage

    @staticmethod
    def _fail(reason: str, **extra) -> dict[str, Any]:
        return {"status": "failed", "reason": reason, **extra}

    @staticmethod
    def _mark_attempt(state: SqlState, **fields: Any) -> list[dict[str, Any]]:
        """Copy of the attempt log with `fields` merged into the latest attempt."""
        log = [dict(e) for e in state.attempt_log]
        if log:
            log[-1].update(fields)
        return log

    def _history_text(self, state: SqlState) -> str:
        return "\n".join(f"{m['role']}: {m['content']}" for m in state.history[-6:])

    # -- nodes -----------------------------------------------------------------------------

    def curate_question(self, state: SqlState) -> dict[str, Any]:
        human = f"Chat history:\n{self._history_text(state) or '(none)'}\n\n"
        human += f"Latest message: {state.user_question}"
        messages = [SystemMessage(CURATE_SYSTEM), HumanMessage(human)]
        try:
            out, usage = self._call(state, "low", CuratedQuestion, messages)
            if not out.answerable or not out.question.strip():
                # The small model sometimes refuses plain data questions. Refusing is costly for
                # the user, so confirm with the stronger model before giving up.
                second, usage = self._call(
                    state.model_copy(update=usage), "high", CuratedQuestion, messages
                )
                out = second
        except BudgetExceeded:
            return self._fail("Too many model calls for one request.")
        except LLMError as e:
            return self._fail(str(e))
        if not out.answerable or not out.question.strip():
            reason = out.reason or "That is not a question I can answer from the data."
            return {"status": "refused", "reason": reason, **usage}
        return {"curated_question": out.question.strip(), **usage}

    def add_context(self, state: SqlState) -> dict[str, Any]:
        try:
            with self.d.connect() as conn:
                tables = db.introspect(conn, self.d.hints, readable_schemas(self.d.settings))
        except psycopg.Error as e:
            return self._fail(f"Could not read the database schema: {str(e)[:150]}")
        if not tables:
            return self._fail("The agent can see no tables. Run `datapilot db apply-grants`.")
        ctx = build_context(
            tables, self.d.hints, state.curated_question or "", self.d.settings.max_context_chars
        )
        return {"schema_context": ctx}

    def generate_sql(self, state: SqlState) -> dict[str, Any]:
        system = GENERATE_SYSTEM.format(
            today=self.d.today().isoformat(), schema=state.schema_context
        )
        human = f"Question: {state.curated_question}"
        if state.feedback and state.sql:
            human += (
                f"\n\nYour previous query:\n{state.sql}\n\nIt was rejected or failed:\n"
                f"{state.feedback}\n\nWrite a corrected query."
            )
        try:
            out, usage = self._call(
                state, "high", SqlDraft, [SystemMessage(system), HumanMessage(human)]
            )
        except BudgetExceeded:
            return self._fail("Too many model calls for one request.")
        except LLMError as e:
            return self._fail(str(e))
        if out.cannot_answer_reason.strip() or not out.sql.strip():
            reason = out.cannot_answer_reason.strip() or "No query could be written."
            entry = {
                "attempt": state.attempts + 1,
                "sql": None,
                "problem": f"cannot answer: {reason}",
            }
            return {
                "status": "refused",
                "reason": f"I can't answer that from the available data: {reason}",
                "attempts": state.attempts + 1,
                "attempt_log": [*state.attempt_log, entry],
                **usage,
            }
        return {
            "sql": out.sql.strip(),
            "assumptions": out.assumptions,
            "attempts": state.attempts + 1,
            "feedback": None,
            "attempt_log": [
                *state.attempt_log,
                {"attempt": state.attempts + 1, "sql": out.sql.strip(), "problem": None},
            ],
            **usage,
        }

    def safety_check(self, state: SqlState) -> dict[str, Any]:
        # Inject cap + 1 so run_query can tell "exactly cap rows" from "more than cap rows".
        result = check_sql(
            state.sql or "",
            self.d.hints,
            max_rows=self.d.settings.max_rows + 1,
            allowed_schemas=frozenset(readable_schemas(self.d.settings)),
        )
        if not result.ok:
            return {
                "safety": SafetyVerdict(ok=False, reason=result.reason),
                "status": "refused",
                "reason": result.reason,
                "attempt_log": self._mark_attempt(state, problem=f"unsafe: {result.reason}"),
            }

        update: dict[str, Any] = {"sql": result.sql, "safety": SafetyVerdict(ok=True)}
        if not self.d.settings.judge_enabled:
            return update

        human = f"Question: {state.curated_question}\n\nSQL:\n{result.sql}"
        try:
            verdict, usage = self._call(
                state,
                "low",
                JudgeVerdict,
                [
                    SystemMessage(
                        JUDGE_SYSTEM.format(
                            today=self.d.today().isoformat(), schema=state.schema_context
                        )
                    ),
                    HumanMessage(human),
                ],
            )
        except BudgetExceeded:
            return self._fail("Too many model calls for one request.")
        except LLMError:
            # The judge checks correctness, not safety (layers 1 and 2 already passed), so
            # a judge outage degrades the check instead of failing the request.
            update["llm_calls"] = state.llm_calls + 1
            return update
        update.update(usage)
        if not verdict.answers_question:
            # A wrong query is not a malicious one: send the objection back for another try.
            update["safety"] = SafetyVerdict(ok=False, reason=verdict.reason)
            update["feedback"] = f"Reviewer objection: {verdict.reason}"
            update["attempt_log"] = self._mark_attempt(
                state, problem=f"judge objected: {verdict.reason}"
            )
        return update

    def run_query(self, state: SqlState) -> dict[str, Any]:
        cap = self.d.settings.max_rows
        try:
            with self.d.connect() as conn:
                conn.execute(f"SET statement_timeout = {int(self.d.settings.statement_timeout_ms)}")
                with conn.cursor() as cur:
                    cur.execute(state.sql)
                    columns = [c.name for c in cur.description or []]
                    fetched = cur.fetchmany(cap + 1)
                conn.rollback()
        except psycopg.errors.InsufficientPrivilege:
            return {
                "status": "refused",
                "reason": "That query touches data the agent is not permitted to read.",
                "attempt_log": self._mark_attempt(state, problem="permission denied"),
            }
        except psycopg.Error as e:
            msg = (str(e).strip().splitlines() or [type(e).__name__])[0]
            return {
                "feedback": f"Database error: {msg}",
                "rows": None,
                "attempt_log": self._mark_attempt(state, problem=f"database error: {msg}"),
            }

        truncated = len(fetched) > cap
        rows = [{k: _jsonable(v) for k, v in r.items()} for r in fetched[:cap]]
        return {"columns": columns, "rows": rows, "truncated": truncated, "feedback": None}

    def represent(self, state: SqlState) -> dict[str, Any]:
        rows = state.rows or []
        shown = rows[:REPRESENT_MAX_ROWS]
        cap_note = ", truncated by the row cap" if state.truncated else ""
        human = (
            f"Question: {state.curated_question}\n"
            f"Assumptions: {'; '.join(state.assumptions) or 'none'}\n"
            f"Columns: {state.columns}\n"
            f"Rows ({len(rows)} total{cap_note}; first {len(shown)} shown):\n"
            f"{shown}\n\n"
            f"Schema notes (use for units and meanings, e.g. currency):\n{state.schema_context}"
        )
        try:
            text, usage = self._call(
                state, "low", None, [SystemMessage(REPRESENT_SYSTEM), HumanMessage(human)]
            )
        except (BudgetExceeded, LLMError):
            # The query already succeeded; never lose the result over a summary failure.
            n = f"{len(rows)}{'+' if state.truncated else ''}"
            return {"answer": f"The query returned {n} row(s).", "status": "ok"}
        return {"answer": str(text), "status": "ok", **usage}
