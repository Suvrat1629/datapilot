"""Parent graph: route a chat message to the SQL Analyst, the ETL Analyst, or a direct reply."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from datapilot.config import Settings, get_settings, readable_schemas
from datapilot.etl_analyst import http
from datapilot.etl_analyst.agent import EtlResult, run_etl_analyst
from datapilot.llm import LLMError, LLMResult, call_llm
from datapilot.parent.router import RouteDecision, router_messages
from datapilot.sql_analyst.graph import run_sql_analyst
from datapilot.sql_analyst.nodes import Deps as SqlDeps
from datapilot.sql_analyst.state import SqlState

HISTORY_LIMIT = 10  # messages kept in a chat session


@dataclass
class ParentDeps:
    """External pieces of the parent, substitutable in tests."""

    settings: Settings = field(default_factory=get_settings)
    call_llm: Callable[..., LLMResult] = call_llm
    run_sql: Callable[..., SqlState] = run_sql_analyst
    run_etl: Callable[..., EtlResult] = run_etl_analyst
    sql_deps: SqlDeps | None = None
    list_tables: Callable[[], list[str]] | None = None  # defaults to introspecting the database


class ChatState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    message: str
    history: list[dict[str, str]] = Field(default_factory=list)

    decision: RouteDecision | None = None
    sql: SqlState | None = None
    etl: EtlResult | None = None

    text: str = ""
    status: Literal["ok", "refused", "failed"] | None = None
    reason: str = ""
    llm_calls: int = 0


def _table_names(deps: ParentDeps) -> list[str]:
    """Names of the tables the SQL Analyst can see, so the router knows what data exists.

    Best effort: if the database is unreachable the router still works, and the SQL Analyst
    reports the real problem when it runs.
    """
    try:
        if deps.list_tables:
            return deps.list_tables()
        from datapilot import db
        from datapilot.hints import load_hints

        with db.connect("ro") as conn:
            tables = db.introspect(
                conn, load_hints(deps.settings.hints_path), readable_schemas(deps.settings)
            )
        return [t.qualified for t in tables]
    except Exception:
        return []


def build_parent_graph(deps: ParentDeps | None = None):
    deps = deps or ParentDeps()

    def route(state: ChatState) -> dict[str, Any]:
        hosts = sorted(http.allowed_hosts(deps.settings))
        messages = router_messages(state.message, state.history, hosts, _table_names(deps))
        calls = 0
        decision: RouteDecision | None = None
        problem = None
        # Small models sometimes leave a required field empty; one corrective retry is cheap.
        for _ in range(2):
            try:
                res = deps.call_llm("high", RouteDecision, messages)
            except LLMError as e:
                return {"status": "failed", "reason": str(e), "llm_calls": calls}
            calls += 1
            decision = res.parsed
            problem = decision.problem()
            if not problem:
                break
            messages = [
                *messages,
                HumanMessage(f"Your answer was incomplete: {problem}. Answer again, complete."),
            ]
        if problem:
            # Better to stop than to guess which agent the user meant.
            return {
                "status": "failed",
                "reason": f"I could not decide how to handle that ({problem}). Please rephrase.",
                "llm_calls": calls,
            }
        return {"decision": decision, "llm_calls": calls}

    def after_route(state: ChatState) -> str:
        if state.status:
            return END
        return {"sql": "run_sql", "etl": "run_etl", "both": "run_etl"}.get(
            state.decision.route, "direct"
        )

    def direct(state: ChatState) -> dict[str, Any]:
        d = state.decision
        # A clarifying question is a normal outcome, not a failure.
        return {"status": "ok", "text": d.reply.strip()}

    def run_etl(state: ChatState, config: RunnableConfig) -> dict[str, Any]:
        res = deps.run_etl(state.decision.etl_request, settings=deps.settings, config=config)
        update: dict[str, Any] = {"etl": res, "llm_calls": state.llm_calls + res.llm_calls}
        if res.status == "ok":
            update.update(status=None, text=res.summary)
        else:
            update.update(status=res.status, reason=res.reason or res.summary)
        return update

    def after_etl(state: ChatState) -> str:
        if state.status:  # the load did not happen, so analysing it would be misleading
            return END
        return "run_sql" if state.decision.route == "both" else "finish_etl"

    def finish_etl(state: ChatState) -> dict[str, Any]:
        return {"status": "ok"}

    def run_sql(state: ChatState, config: RunnableConfig) -> dict[str, Any]:
        res = deps.run_sql(state.decision.sql_question, state.history, deps.sql_deps, config=config)
        update: dict[str, Any] = {"sql": res, "llm_calls": state.llm_calls + res.llm_calls}
        if res.status == "ok":
            answer = res.answer or ""
            update.update(status="ok", text=f"{state.text}\n\n{answer}".strip())
        else:
            prefix = f"{state.text}\n\n" if state.text else ""
            update.update(status=res.status or "failed", reason=prefix + res.reason)
        return update

    g = StateGraph(ChatState)
    g.add_node("route", route)
    g.add_node("direct", direct)
    g.add_node("run_etl", run_etl)
    g.add_node("finish_etl", finish_etl)
    g.add_node("run_sql", run_sql)
    g.add_edge(START, "route")
    g.add_conditional_edges("route", after_route, ["run_sql", "run_etl", "direct", END])
    g.add_conditional_edges("run_etl", after_etl, ["run_sql", "finish_etl", END])
    g.add_edge("direct", END)
    g.add_edge("finish_etl", END)
    g.add_edge("run_sql", END)
    return g.compile()


@dataclass
class TurnResult:
    message: str
    route: str
    status: str
    text: str  # what to show the user: the answer, or the reason it failed
    sql: SqlState | None = None
    etl: EtlResult | None = None
    llm_calls: int = 0


class ChatSession:
    """Holds the conversation history across turns."""

    def __init__(self, deps: ParentDeps | None = None):
        self.deps = deps or ParentDeps()
        self.history: list[dict[str, str]] = []
        self._graph = build_parent_graph(self.deps)

    def clear(self) -> None:
        self.history = []

    def turn(self, message: str, config: dict[str, Any] | None = None) -> TurnResult:
        out = ChatState.model_validate(
            self._graph.invoke(
                ChatState(message=message, history=list(self.history)), config=config
            )
        )
        status = out.status or "failed"
        shown = out.text if status == "ok" else (out.reason or "Something went wrong.")
        route = out.decision.route if out.decision else "none"
        # Only successful turns enter history; failures would teach later turns nothing true.
        # Small talk is skipped too: a chat reply that misstates what the assistant can do would
        # otherwise steer later routing.
        if status == "ok" and shown and route != "chat":
            self.history += [
                {"role": "user", "content": message},
                {"role": "assistant", "content": shown},
            ]
            self.history = self.history[-HISTORY_LIMIT:]
        return TurnResult(message, route, status, shown, out.sql, out.etl, out.llm_calls)
