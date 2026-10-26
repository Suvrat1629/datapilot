"""Typed, declarative transforms. The model describes steps; this code interprets them.

There is no eval, no expression strings and no code execution: every operation is a fixed,
validated pandas call, so prompt injection from API data cannot run code here.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

import pandas as pd
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from datapilot.etl_analyst.store import EtlToolError


class Rename(BaseModel):
    op: Literal["rename"]
    mapping: dict[str, str]


class Select(BaseModel):
    op: Literal["select"]
    columns: list[str]


class Drop(BaseModel):
    op: Literal["drop"]
    columns: list[str]


class Cast(BaseModel):
    op: Literal["cast"]
    column: str
    dtype: Literal["int", "float", "str", "bool", "date", "datetime"]


class Filter(BaseModel):
    op: Literal["filter"]
    column: str
    cmp: Literal["eq", "ne", "gt", "ge", "lt", "le", "in", "notnull", "isnull"]
    value: Any = None


class AddColumn(BaseModel):
    op: Literal["add_column"]
    name: str
    value: str | int | float | bool


class Aggregate(BaseModel):
    op: Literal["aggregate"]
    group_by: list[str]
    aggs: dict[str, Literal["sum", "mean", "min", "max", "count"]]


class Sort(BaseModel):
    op: Literal["sort"]
    by: list[str]
    descending: bool = False


class Dedupe(BaseModel):
    op: Literal["dedupe"]
    columns: list[str] | None = None


class Limit(BaseModel):
    op: Literal["limit"]
    n: int = Field(ge=1)


Step = Annotated[
    Rename | Select | Drop | Cast | Filter | AddColumn | Aggregate | Sort | Dedupe | Limit,
    Field(discriminator="op"),
]
_STEPS = TypeAdapter(list[Step])

OPS_HELP = (
    "rename{mapping}, select{columns}, drop{columns}, cast{column,dtype:int|float|str|bool|date|"
    "datetime}, filter{column,cmp:eq|ne|gt|ge|lt|le|in|notnull|isnull,value}, "
    "add_column{name,value}, aggregate{group_by,aggs:{col:sum|mean|min|max|count}}, "
    "sort{by,descending}, dedupe{columns}, limit{n}"
)


OP_NAMES = {"rename", "select", "drop", "cast", "filter", "add_column", "aggregate", "sort",
            "dedupe", "limit"}  # fmt: skip


def _normalise(step: Any) -> Any:
    """Accept the common shorthand {"cast": {...}} as {"op": "cast", ...}.

    Models often wrap a step in its own name; the meaning is unambiguous, so accept it.
    """
    if isinstance(step, dict) and "op" not in step and len(step) == 1:
        ((name, body),) = step.items()
        if name in OP_NAMES and isinstance(body, dict):
            if name == "rename" and "mapping" not in body:
                body = {"mapping": body}
            return {"op": name, **body}
    return step


def parse_steps(raw: list[dict[str, Any]]) -> list[BaseModel]:
    try:
        return _STEPS.validate_python([_normalise(s) for s in raw])
    except ValidationError as e:
        first = e.errors()[0]
        loc = ".".join(str(x) for x in first["loc"])
        raise EtlToolError(
            f"invalid transform step at {loc}: {first['msg']}. Each step is an object with an "
            f'"op" field, e.g. {{"op": "rename", "mapping": {{"a": "b"}}}}. Valid steps: {OPS_HELP}'
        ) from None


def _need(df: pd.DataFrame, columns: list[str], what: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise EtlToolError(
            f"{what}: unknown column(s) {missing}. Available columns: {list(df.columns)}"
        )


def _cast(series: pd.Series, dtype: str) -> pd.Series:
    try:
        if dtype == "int":
            return pd.to_numeric(series).astype("Int64")
        if dtype == "float":
            return pd.to_numeric(series).astype("float64")
        if dtype == "str":
            return series.astype("string").astype(object).where(series.notna(), None)
        if dtype == "bool":
            return series.astype("boolean")
        parsed = pd.to_datetime(series, utc=True)
        return parsed.dt.date if dtype == "date" else parsed
    except (ValueError, TypeError) as e:
        raise EtlToolError(f"cast to {dtype} failed: {str(e)[:150]}") from None


def _mask(col: pd.Series, step: Filter) -> pd.Series:
    v = step.value
    match step.cmp:
        case "eq":
            return col == v
        case "ne":
            return col != v
        case "gt":
            return col > v
        case "ge":
            return col >= v
        case "lt":
            return col < v
        case "le":
            return col <= v
        case "in":
            return col.isin(v if isinstance(v, list) else [v])
        case "notnull":
            return col.notna()
        case _:
            return col.isna()


def apply_steps(df: pd.DataFrame, steps: list[BaseModel]) -> pd.DataFrame:
    out = df.copy()
    for i, step in enumerate(steps, start=1):
        label = f"step {i} ({step.op})"  # type: ignore[attr-defined]
        if isinstance(step, Rename):
            _need(out, list(step.mapping), label)
            out = out.rename(columns=step.mapping)
        elif isinstance(step, Select):
            _need(out, step.columns, label)
            out = out[step.columns]
        elif isinstance(step, Drop):
            _need(out, step.columns, label)
            out = out.drop(columns=step.columns)
        elif isinstance(step, Cast):
            _need(out, [step.column], label)
            out[step.column] = _cast(out[step.column], step.dtype)
        elif isinstance(step, Filter):
            _need(out, [step.column], label)
            try:
                out = out[_mask(out[step.column], step)]
            except TypeError as e:
                raise EtlToolError(
                    f"{label}: cannot compare ({str(e)[:100]}); cast first"
                ) from None
        elif isinstance(step, AddColumn):
            out[step.name] = step.value
        elif isinstance(step, Aggregate):
            _need(out, step.group_by + list(step.aggs), label)
            out = out.groupby(step.group_by, dropna=False).agg(step.aggs).reset_index()
        elif isinstance(step, Sort):
            _need(out, step.by, label)
            out = out.sort_values(step.by, ascending=not step.descending)
        elif isinstance(step, Dedupe):
            if step.columns:
                _need(out, step.columns, label)
            out = out.drop_duplicates(subset=step.columns)
        elif isinstance(step, Limit):
            out = out.head(step.n)
        out = out.reset_index(drop=True)
    return out


def run_transform(df: pd.DataFrame, raw_steps: list[dict[str, Any]]) -> pd.DataFrame:
    return apply_steps(df, parse_steps(raw_steps))
