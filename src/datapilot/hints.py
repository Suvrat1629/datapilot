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

    def is_hidden(self, table: str, column: str, schema: str = "public") -> bool:
        """Entries are `table.column` for public and `schema.table.column` for other schemas
        (`public.table.column` is accepted too)."""
        if f"{schema}.{table}.{column}" in self.hidden_columns:
            return True
        return schema == "public" and f"{table}.{column}" in self.hidden_columns

    def for_table(self, table: str, schema: str = "public") -> TableHints | None:
        found = self.tables.get(f"{schema}.{table}")
        if found is None and schema == "public":
            found = self.tables.get(table)
        return found


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
