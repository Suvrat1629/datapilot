from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from datapilot.sql_analyst.nodes import Deps, SqlAnalystNodes
from datapilot.sql_analyst.state import SqlState


def build_graph(deps: Deps | None = None):
    deps = deps or Deps()
    n = SqlAnalystNodes(deps)
    max_attempts = 1 + deps.settings.max_sql_retries

    def done_or(next_node: str):
        """Route to END as soon as any node has set a final status."""

        def route(state: SqlState) -> str:
            return END if state.status else next_node

        return route

    def after_safety(state: SqlState) -> str:
        if state.status:
            return END
        if state.feedback:  # judge objection
            return "generate_sql" if state.attempts < max_attempts else "give_up"
        return "run_query"

    def after_run(state: SqlState) -> str:
        if state.status:
            return END
        if state.feedback:  # database error
            return "generate_sql" if state.attempts < max_attempts else "give_up"
        return "represent"

    def give_up(state: SqlState) -> dict[str, Any]:
        return {
            "status": "failed",
            "reason": f"Could not produce a working query after {state.attempts} attempts. "
            f"Last problem: {state.feedback}",
        }

    g = StateGraph(SqlState)
    g.add_node("curate_question", n.curate_question)
    g.add_node("add_context", n.add_context)
    g.add_node("generate_sql", n.generate_sql)
    g.add_node("safety_check", n.safety_check)
    g.add_node("run_query", n.run_query)
    g.add_node("represent", n.represent)
    g.add_node("give_up", give_up)

    g.add_edge(START, "curate_question")
    g.add_conditional_edges("curate_question", done_or("add_context"), ["add_context", END])
    g.add_conditional_edges("add_context", done_or("generate_sql"), ["generate_sql", END])
    g.add_conditional_edges("generate_sql", done_or("safety_check"), ["safety_check", END])
    g.add_conditional_edges(
        "safety_check", after_safety, ["generate_sql", "run_query", "give_up", END]
    )
    g.add_conditional_edges("run_query", after_run, ["generate_sql", "represent", "give_up", END])
    g.add_edge("represent", END)
    g.add_edge("give_up", END)
    return g.compile()


def run_sql_analyst(
    question: str,
    history: list[dict[str, str]] | None = None,
    deps: Deps | None = None,
    config: dict[str, Any] | None = None,
) -> SqlState:
    graph = build_graph(deps)
    out = graph.invoke(SqlState(user_question=question, history=history or []), config=config)
    return SqlState.model_validate(out)
