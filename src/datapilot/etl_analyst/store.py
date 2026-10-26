"""Per-request working state for the ETL agent: dataframes behind short handles.

The model never sees full datasets, only handles and previews, so a large API response cannot
flood its context and tool outputs stay small.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


class EtlToolError(Exception):
    """A tool-level failure with a message written for the model to act on."""


@dataclass
class LoadRecord:
    table: str
    rows: int
    columns: list[str]
    source: str


@dataclass
class DataStore:
    max_fetches: int = 50
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)
    loads: list[LoadRecord] = field(default_factory=list)
    refused: str | None = None
    fetches: int = 0
    _n: int = 0

    def put(self, df: pd.DataFrame, source: str) -> str:
        self._n += 1
        handle = f"d{self._n}"
        self.frames[handle] = df
        self.sources[handle] = source
        return handle

    def get(self, handle: str) -> pd.DataFrame:
        try:
            return self.frames[handle]
        except KeyError:
            known = ", ".join(self.frames) or "none yet"
            raise EtlToolError(f"unknown handle {handle!r}; available handles: {known}") from None

    def count_fetch(self, n: int = 1) -> None:
        if self.fetches + n > self.max_fetches:
            raise EtlToolError(
                f"request limit reached: {self.fetches} used, {n} more needed, "
                f"limit is {self.max_fetches} HTTP requests per task"
            )
        self.fetches += n
