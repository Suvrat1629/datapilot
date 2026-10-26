from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class TableHints(BaseModel):
    description: str = ""
    columns: dict[str, str] = Field(default_factory=dict)


class Hints(BaseModel):
    tables: dict[str, TableHints] = Field(default_factory=dict)
    hidden_columns: list[str] = Field(default_factory=list)  # "table.column"

    def is_hidden(self, table: str, column: str) -> bool:
        return f"{table}.{column}" in self.hidden_columns


DEMO_HINTS_PATH = Path(__file__).parent / "demo" / "hints.yaml"


def load_hints(path: str | Path) -> Hints:
    """Load hints from `path`. An empty path means no hints; a path that is set but missing
    is an error, so a typo cannot silently drop hidden_columns."""
    if not str(path):
        return Hints()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"hints file not found: {p}")
    return Hints.model_validate(yaml.safe_load(p.read_text()) or {})
