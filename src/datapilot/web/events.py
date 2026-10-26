"""Turn LangGraph/LangChain callbacks into short, human-readable progress steps."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit

from langchain_core.callbacks import BaseCallbackHandler

NODE_LABELS = {
    "route": "Understanding your request",
    "curate_question": "Rephrasing the question",
    "add_context": "Reading the database schema",
    "generate_sql": "Writing SQL",
    "safety_check": "Checking the query is safe",
    "run_query": "Running the query",
    "represent": "Writing the answer",
    "run_etl": "Starting the data load",
}

TOOL_LABELS = {
    "list_tables": "Looking at the available tables",
    "read_reference": "Reading reference data",
    "inspect_data": "Inspecting the data",
    "transform": "Reshaping the data",
    "refuse": "Declining the request",
}


def _host(url: Any) -> str:
    try:
        return urlsplit(str(url)).hostname or "the API"
    except ValueError:
        return "the API"


class StepHandler(BaseCallbackHandler):
    """Calls `emit({"type": "step", ...})` as the graphs progress.

    Only known node and tool names produce steps; the many internal runnables LangGraph
    reports are ignored. Labels are fixed text plus a few values (host, table), never raw
    model or database output.
    """

    raise_error = False  # progress reporting must never break a turn

    def __init__(self, emit: Callable[[dict[str, Any]], None]):
        self.emit = emit
        self._sql_attempts = 0
        self._tool_names: dict[Any, str] = {}

    def _step(self, label: str, node: str) -> None:
        self.emit({"type": "step", "node": node, "label": label})

    def on_chain_start(self, serialized, inputs, *, run_id, metadata=None, name=None, **kw):
        node = (metadata or {}).get("langgraph_node")
        if not node or name != node or node not in NODE_LABELS:
            return
        label = NODE_LABELS[node]
        if node == "generate_sql":
            self._sql_attempts += 1
            if self._sql_attempts > 1:
                label = f"Rewriting the SQL (attempt {self._sql_attempts})"
        self._step(label, node)

    def on_tool_start(self, serialized, input_str, *, run_id, inputs=None, **kw):
        name = (serialized or {}).get("name") or kw.get("name") or ""
        self._tool_names[run_id] = name
        args = inputs if isinstance(inputs, dict) else {}
        if name == "fetch_api":
            where = _host(args.get("url"))
            many = " (one request per row)" if args.get("for_each") else ""
            label = f"Fetching data from {where}{many}"
        elif name == "load_staging":
            label = f"Loading into staging.{str(args.get('table', ''))[:63]}"
        else:
            label = TOOL_LABELS.get(name)
        if label:
            self._step(label, f"tool:{name}")

    def on_tool_end(self, output, *, run_id, **kw):
        name = self._tool_names.pop(run_id, "")
        text = getattr(output, "content", output)
        if isinstance(text, str) and text.startswith("ERROR"):
            self.emit({"type": "step_error", "node": f"tool:{name}", "detail": text[:200]})
