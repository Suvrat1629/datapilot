"""Schema context for the prompt, trimmed to the tables relevant to the question."""

from __future__ import annotations

import re

from datapilot.db import Table, render_schema_context
from datapilot.hints import Hints


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2}


def _score(table: Table, question_words: set[str], hints: Hints) -> int:
    th = hints.for_table(table.name, table.schema)
    haystack = _words(table.name.replace("_", " "))
    for c in table.columns:
        haystack |= _words(c.name.replace("_", " "))
    if th:
        haystack |= _words(th.description)
        for note in th.columns.values():
            haystack |= _words(note)
    singular = {w.rstrip("s") for w in question_words}
    return len((question_words | singular) & (haystack | {w.rstrip("s") for w in haystack}))


def build_context(tables: list[Table], hints: Hints, question: str, max_chars: int) -> str:
    """Render the full schema if it fits, else the best-matching tables plus FK neighbours."""
    full = render_schema_context(tables, hints)
    if len(full) <= max_chars:
        return full

    by_name = {t.qualified: t for t in tables}
    qw = _words(question)
    ranked = sorted(tables, key=lambda t: (-_score(t, qw, hints), t.name))

    chosen: list[Table] = []

    def add(t: Table) -> bool:
        if t in chosen:
            return True
        trial = render_schema_context(chosen + [t], hints)
        if len(trial) > max_chars:
            return False
        chosen.append(t)
        return True

    for t in ranked:
        if _score(t, qw, hints) == 0 and chosen:
            break
        if not add(t):
            continue
        for c in t.columns:  # pull in tables this one joins to
            ref = c.fk.split(".")[0] if c.fk else None
            if ref and t.schema != "public":  # foreign keys stay within the table's schema
                ref = f"{t.schema}.{ref}"
            if ref in by_name:
                add(by_name[ref])
    if not chosen:  # nothing matched or nothing fit: fall back to the first tables that fit
        for t in tables:
            if not add(t):
                break
    return render_schema_context(chosen, hints)
