import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from datapilot import llm as llm_mod
from datapilot.db import Column, Table, render_schema_context
from datapilot.hints import Hints
from datapilot.sql_analyst.context import build_context


def _t(name, cols, fk=None):
    columns = [Column("id", "integer", False, is_pk=True)]
    columns += [Column(c, "text", True) for c in cols]
    if fk:
        columns.append(Column(f"{fk}_id", "integer", False, fk=f"{fk}.id"))
    return Table(name, columns)


def test_small_schema_is_rendered_in_full():
    tables = [_t("a", ["x"]), _t("b", ["y"])]
    assert build_context(tables, Hints(), "anything", 10_000) == render_schema_context(
        tables, Hints()
    )


def test_large_schema_is_trimmed_to_relevant_tables_and_fk_neighbours():
    tables = [_t(f"noise_{i}", ["junk_col"] * 3) for i in range(40)]
    tables += [_t("invoices", ["amount"], fk="customers"), _t("customers", ["name"])]
    ctx = build_context(tables, Hints(), "total invoice amount per customer", 1500)
    assert "TABLE invoices" in ctx and "TABLE customers" in ctx
    assert "noise_" not in ctx and len(ctx) <= 1500


def test_unmatched_question_still_returns_something():
    tables = [_t(f"t{i}", ["c"]) for i in range(40)]
    assert build_context(tables, Hints(), "zzzz qqqq", 600).startswith("TABLE")


class Out(BaseModel):
    n: int


class FakeChat:
    """Mimics a LangChain chat model for the repair and backoff paths."""

    def __init__(self, structured, plain):
        self.structured, self.plain = structured, plain

    def with_structured_output(self, schema, include_raw=False):
        outer = self

        class R:
            def invoke(self, messages):
                r = outer.structured.pop(0)
                if isinstance(r, Exception):
                    raise r
                return r

        return R()

    def invoke(self, messages):
        r = self.plain.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def _patch(monkeypatch, chat):
    monkeypatch.setattr(llm_mod, "get_llm", lambda tier: chat)
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: None)


def test_structured_output_happy_path(monkeypatch):
    raw = AIMessage("x", usage_metadata={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10})
    _patch(monkeypatch, FakeChat([{"raw": raw, "parsed": Out(n=1), "parsing_error": None}], []))
    r = llm_mod.call_llm("low", Out, [HumanMessage("hi")])
    assert r.parsed == Out(n=1) and (r.tokens_in, r.tokens_out) == (7, 3)


def test_parse_failure_is_repaired_with_plain_json(monkeypatch):
    bad = {"raw": AIMessage("junk"), "parsed": None, "parsing_error": "bad"}
    _patch(monkeypatch, FakeChat([bad], [AIMessage('Sure!\n```json\n{"n": 5}\n```')]))
    assert llm_mod.call_llm("low", Out, [HumanMessage("hi")]).parsed == Out(n=5)


def test_unrepairable_output_raises(monkeypatch):
    bad = {"raw": AIMessage("junk"), "parsed": None, "parsing_error": "bad"}
    _patch(monkeypatch, FakeChat([bad], [AIMessage("still not json")]))
    with pytest.raises(llm_mod.LLMError, match="unparseable"):
        llm_mod.call_llm("low", Out, [HumanMessage("hi")])


def test_rate_limit_is_retried_with_backoff(monkeypatch):
    ok = {"raw": AIMessage("x"), "parsed": Out(n=2), "parsing_error": None}
    sleeps = []
    chat = FakeChat([Exception("Error 429: rate limit exceeded"), Exception("429"), ok], [])
    monkeypatch.setattr(llm_mod, "get_llm", lambda tier: chat)
    monkeypatch.setattr(llm_mod.time, "sleep", sleeps.append)
    assert llm_mod.call_llm("low", Out, [HumanMessage("hi")]).parsed == Out(n=2)
    assert sleeps == [2.0, 4.0]


def test_non_rate_limit_error_is_not_retried(monkeypatch):
    _patch(monkeypatch, FakeChat([Exception("invalid api key")], []))
    with pytest.raises(llm_mod.LLMError, match="invalid api key"):
        llm_mod.call_llm("low", Out, [HumanMessage("hi")])


def test_persistent_rate_limit_gives_up(monkeypatch):
    _patch(monkeypatch, FakeChat([Exception("429")] * 4, []))
    with pytest.raises(llm_mod.LLMError):
        llm_mod.call_llm("low", Out, [HumanMessage("hi")])


def test_plain_text_call(monkeypatch):
    _patch(monkeypatch, FakeChat([], [AIMessage("  hello  ")]))
    assert llm_mod.call_llm("low", None, [HumanMessage("hi")]).parsed == "hello"


def test_unconfigured_model_is_a_clean_error(monkeypatch):
    def boom(tier):
        raise RuntimeError("api_key must be set")

    monkeypatch.setattr(llm_mod, "get_llm", boom)
    with pytest.raises(llm_mod.LLMError, match="not configured"):
        llm_mod.call_llm("high", Out, [HumanMessage("hi")])


def test_provider_tool_use_failure_falls_back_to_repair(monkeypatch):
    err = Exception(
        "Error code: 400 - {'code': 'tool_use_failed', 'message': 'Tool choice is required'}"
    )
    _patch(monkeypatch, FakeChat([err], [AIMessage('{"n": 9}')]))
    assert llm_mod.call_llm("low", Out, [HumanMessage("hi")]).parsed == Out(n=9)


def test_daily_quota_fails_fast_without_retrying(monkeypatch):
    msg = (
        "429 RESOURCE_EXHAUSTED: Quota exceeded ... GenerateRequestsPerDayPerProjectPerModel-FreeTier"
        "\nPlease retry in 20h53m17.03s."
    )
    sleeps = []
    monkeypatch.setattr(llm_mod, "get_llm", lambda tier: FakeChat([Exception(msg)] * 4, []))
    monkeypatch.setattr(llm_mod.time, "sleep", sleeps.append)
    with pytest.raises(llm_mod.LLMError, match=r"daily free-tier quota.*20h53m17"):
        llm_mod.call_llm("high", Out, [HumanMessage("hi")])
    assert sleeps == []
