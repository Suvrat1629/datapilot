import pytest

from datapilot.db import Column, Table, render_schema_context
from datapilot.hints import Hints, TableHints, load_hints


def test_load_hints_empty_path_means_no_hints():
    h = load_hints("")
    assert h.tables == {} and h.hidden_columns == []


def test_load_hints_missing_file_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_hints(tmp_path / "nope.yaml")


def test_load_hints_parses_yaml(tmp_path):
    p = tmp_path / "h.yaml"
    p.write_text(
        "tables:\n  t:\n    description: d\n    columns: {c: note}\nhidden_columns: [t.secret]\n"
    )
    h = load_hints(p)
    assert h.tables["t"].columns["c"] == "note"
    assert h.is_hidden("t", "secret") and not h.is_hidden("t", "c")


def test_render_schema_context_includes_hints_and_samples():
    hints = Hints(tables={"t": TableHints(description="a table", columns={"status": "state"})})
    table = Table(
        "t",
        [
            Column("id", "integer", False, is_pk=True),
            Column("owner_id", "integer", False, fk="owners.id"),
            Column("status", "text", False, samples=["a", "b"]),
        ],
    )
    out = render_schema_context([table], hints)
    assert "TABLE t -- a table" in out
    assert "id integer PK" in out
    assert "FK -> owners.id" in out
    assert "-- state values: a, b" in out
