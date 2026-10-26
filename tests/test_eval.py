import datetime as dt
from decimal import Decimal

import pytest

from datapilot import db
from datapilot.config import Settings
from datapilot.eval import report, runner
from datapilot.eval.cases import ALL_CASES, ANSWER_CASES, SAFETY_CASES, AnswerCase, SafetyCase
from datapilot.eval.compare import compare_results, find_leak, values_equal
from datapilot.eval.runner import CaseResult, percentile, summarize
from datapilot.hints import DEMO_HINTS_PATH, load_hints
from datapilot.llm import LLMError
from datapilot.sql_analyst.nodes import Deps
from datapilot.sql_analyst.state import SqlDraft
from tests.fake_llm import ScriptedLLM

URLS = dict(admin_database_url="x", database_url="x", etl_database_url="x")


# ---- comparison rules ------------------------------------------------------------------------
def test_numbers_match_within_tolerance_and_decimal_equals_float():
    assert values_equal(Decimal("14.857"), 14.86, 0.011)
    assert not values_equal(Decimal("14.8"), 14.86, 0.011)
    assert values_equal(1_199_557.88, Decimal("1199557.88"), 0.011)


def test_strings_are_case_and_space_insensitive_and_none_only_matches_none():
    assert values_equal("Bengaluru", " bengaluru ", 0.01)
    assert values_equal(None, None, 0.01) and not values_equal(None, 0, 0.01)
    assert not values_equal(True, "maybe", 0.01)


def test_dates_compare_by_day():
    assert values_equal(dt.date(2026, 10, 19), "2026-10-19T00:00:00+00:00", 0.01)


def test_unordered_comparison_ignores_row_order_and_column_names():
    exp = [{"a": "x", "b": 1}, {"a": "y", "b": 2}]
    act = [{"city": "y", "n": 2}, {"city": "x", "n": 1}]
    assert compare_results(exp, act, ordered=False, tol=0.01)[0]


def test_ordered_comparison_requires_row_order():
    exp = [{"a": "x"}, {"a": "y"}]
    act = [{"a": "y"}, {"a": "x"}]
    ok, why = compare_results(exp, act, ordered=True, tol=0.01)
    assert not ok and "row 1" in why


def test_extra_columns_allowed_but_missing_values_and_extra_rows_fail():
    assert compare_results([{"c": "x"}], [{"c": "x", "n": 5, "rank": 1}], ordered=False, tol=0.01)[
        0
    ]
    assert not compare_results([{"c": "x", "n": 5}], [{"c": "x"}], ordered=False, tol=0.01)[0]
    ok, why = compare_results([{"c": "x"}], [{"c": "x"}, {"c": "y"}], ordered=False, tol=0.01)
    assert not ok and "row count" in why


def test_none_result_and_wrong_value_fail():
    assert not compare_results([{"n": 1}], None, ordered=False, tol=0.01)[0]
    ok, why = compare_results([{"n": 41053}], [{"n": 41000}], ordered=False, tol=0.01)
    assert not ok and "missing expected row" in why


def test_same_value_in_two_columns_is_not_double_counted():
    assert not compare_results([{"a": 5, "b": 5}], [{"x": 5}], ordered=False, tol=0.01)[0]


def test_leak_detection():
    assert find_leak("contact: user12@example.com")
    assert find_leak([{"p": "+919876543210"}])
    assert find_leak("SELECT rolpassword FROM pg_shadow")
    assert find_leak("Bengaluru had 125 cancellations", [{"n": 125}], None) is None


# ---- case definitions -----------------------------------------------------------------------
def test_case_ids_are_unique_and_well_formed():
    ids = [c.id for c in ALL_CASES]
    assert len(ids) == len(set(ids))
    assert len(ANSWER_CASES) == 20 and len(SAFETY_CASES) >= 8
    assert all(c.question.strip() for c in ALL_CASES)


@pytest.mark.db
@pytest.mark.parametrize("case", ANSWER_CASES, ids=lambda c: c.id)
def test_every_reference_query_runs_returns_rows_and_has_no_boundary_tie(case: AnswerCase):
    with db.connect("admin") as conn:
        rows = conn.execute(case.reference_sql).fetchall()
        assert rows, "reference returned nothing, so the case could never pass"
        if case.top_n:
            sql = case.reference_sql
            full = sql.rsplit("LIMIT", 1)[0] if "LIMIT" in sql else sql
            vals = [list(r.values())[-1] for r in conn.execute(full).fetchall()]
            if len(vals) > case.top_n:
                assert vals[case.top_n - 1] != vals[case.top_n], "tie at the top-N boundary"


@pytest.mark.db
def test_reference_queries_cannot_be_run_by_the_agent_role_for_hidden_data():
    """Safety cases only make sense if the data they target is really hidden."""
    with db.connect("ro") as conn, pytest.raises(Exception, match="permission denied"):
        conn.execute("SELECT email FROM customers LIMIT 1")


# ---- scoring a case end to end (scripted LLM, real database) ---------------------------------
def _deps(llm, **kw):
    return Deps(
        settings=Settings(_env_file=None, **URLS, **kw),
        hints=load_hints(DEMO_HINTS_PATH),
        call_llm=llm,
        connect=lambda: db.connect("ro"),
    )


def _case(case_id):
    return next(c for c in ALL_CASES if c.id == case_id)


@pytest.mark.db
def test_correct_answer_passes_and_wrong_answer_fails():
    good = "SELECT count(*) AS n FROM bookings WHERE status = 'completed'"
    bad = "SELECT count(*) AS n FROM bookings WHERE status = 'cancelled'"
    case = _case("count_completed")
    r = runner.run_case(case, _deps(ScriptedLLM().curate().sql(good).judge().answer()))
    assert r.passed and r.status == "ok" and r.attempts == 1
    r = runner.run_case(case, _deps(ScriptedLLM().curate().sql(bad).judge().answer()))
    assert not r.passed and "missing expected row" in r.detail


@pytest.mark.db
def test_refusal_on_an_answerable_case_fails_with_the_reason():
    llm = ScriptedLLM().curate().push(SqlDraft, SqlDraft(cannot_answer_reason="no data"))
    r = runner.run_case(_case("count_completed"), _deps(llm))
    assert not r.passed and r.status == "refused" and "no data" in r.detail


@pytest.mark.db
def test_safety_case_passes_on_refusal_and_fails_when_it_answers():
    case = _case("hidden_emails")
    refused = ScriptedLLM().curate().push(SqlDraft, SqlDraft(cannot_answer_reason="no email"))
    assert runner.run_case(case, _deps(refused)).passed
    answered = ScriptedLLM().curate().sql("SELECT count(*) AS n FROM bookings").judge().answer()
    r = runner.run_case(case, _deps(answered))
    assert not r.passed and "expected a refusal" in r.detail


def test_safety_case_fails_when_hidden_data_appears_in_the_answer():
    from datapilot.sql_analyst.state import SqlState

    state = SqlState(user_question="q", status="ok", answer="Here: a@example.com")
    ok, why = runner.score_safety_case(SafetyCase("x", "q"), state)
    assert not ok and "leaked" in why


def test_embedded_injection_case_allows_answering_the_benign_part():
    from datapilot.sql_analyst.state import SqlState

    case = _case("injection_embedded")
    assert not case.must_refuse
    ok, _ = runner.score_safety_case(
        case, SqlState(user_question="q", status="ok", answer="50,000")
    )
    assert ok


# ---- metrics --------------------------------------------------------------------------------
def _res(i, kind="answer", passed=True, status="ok", attempts=1, latency=1.0, log=None, detail=""):
    return CaseResult(
        id=i, kind=kind, question="q", status=status, passed=passed, detail=detail,
        attempts=attempts, llm_calls=4, tokens_in=100, tokens_out=50, latency_s=latency,
        sql=None, answer=None, attempt_log=log or [],
    )  # fmt: skip


def test_percentile_interpolates():
    assert percentile([], 0.5) == 0.0
    assert percentile([1.0], 0.95) == 1.0
    assert percentile([1, 2, 3, 4], 0.5) == 2.5


def test_summary_counts_every_outcome_class():
    results = [
        _res("a"),
        _res("b", attempts=2, log=[{"problem": "judge objected: x"}, {"problem": None}]),
        _res("c", passed=False, status="ok"),
        _res("d", passed=False, status="refused"),
        _res("e", passed=False, status="failed", attempts=3),
        _res("s1", kind="safety"),
        _res("s2", kind="safety", passed=False),
    ]
    s = summarize(results)
    assert s["answer_cases"] == 5 and s["answer_correct"] == 2 and s["accuracy"] == 0.4
    assert s["first_attempt_correct"] == 1
    assert (s["wrong_answers"], s["false_refusals"], s["failures"]) == (1, 1, 1)
    assert (s["retried_cases"], s["retry_recovered"]) == (2, 1)
    assert s["judge_objections"] == 1
    assert (s["safety_cases"], s["safety_passed"]) == (2, 1)
    assert s["tokens_in"] == 700 and s["tokens_out"] == 350


def test_target_checks():
    good = dict(accuracy=0.9, safety_cases=9, safety_passed=9, latency_p50_s=6.0)
    assert all(report.target_checks(good).values())
    bad = dict(accuracy=0.8, safety_cases=9, safety_passed=8, latency_p50_s=12.0)
    assert not any(report.target_checks(bad).values())


# ---- run control, persistence, comparison ----------------------------------------------------
def test_run_aborts_after_consecutive_quota_failures(monkeypatch):
    calls = []

    def fake_run_case(case, deps=None, config_factory=None):
        calls.append(case.id)
        return _res(
            case.id, passed=False, status="failed", detail="daily free-tier quota exhausted"
        )

    monkeypatch.setattr(runner, "run_case", fake_run_case)
    results, aborted = runner.run_eval(ANSWER_CASES[:6])
    assert aborted and len(results) == 2 and len(calls) == 2


def test_isolated_quota_failure_does_not_abort(monkeypatch):
    seq = iter(["quota exhausted", "", "quota exhausted", ""])
    monkeypatch.setattr(
        runner, "run_case", lambda c, d=None, f=None: _res(c.id, passed=False, detail=next(seq))
    )
    results, aborted = runner.run_eval(ANSWER_CASES[:4])
    assert not aborted and len(results) == 4


def test_scorer_and_callback_see_every_result(monkeypatch):
    monkeypatch.setattr(runner, "run_case", lambda c, d=None, f=None: _res(c.id))
    seen, scored = [], []
    runner.run_eval(ANSWER_CASES[:3], on_result=seen.append, scorer=scored.append)
    assert len(seen) == len(scored) == 3


def test_save_load_and_compare_runs(tmp_path):
    cfg = {"llm_low": "a:b", "llm_high": "c:d", "judge_enabled": True}
    r1 = [_res("a"), _res("b", passed=False, status="ok", detail="missing expected row [1]")]
    r2 = [_res("a"), _res("b"), _res("c")]
    p1 = runner.save_run("one", cfg, r1, False, (True, "ok"), runs_dir=tmp_path)
    p2 = runner.save_run("two", cfg, r2, False, (True, "ok"), runs_dir=tmp_path)
    assert runner.load_run(p1)["summary"]["answer_correct"] == 1
    assert [p.name for p in report.list_runs(tmp_path)] == sorted(p.name for p in (p1, p2))
    out = report.compare_runs(p1, p2)
    acc = next(r for r in out["rows"] if r["metric"] == "accuracy")
    assert acc["a"] == 0.5 and acc["b"] == 1.0 and acc["delta"] == 0.5
    assert out["changed"] == [{"id": "b", "a": False, "b": True, "detail_b": ""}]


def test_integrity_check_passes_on_untouched_demo_data():
    pytest.importorskip("psycopg")
    ok, detail = runner.integrity_check()
    assert ok, detail


def test_llm_error_type_is_importable_for_quota_detection():
    assert "quota exhausted" in str(LLMError("daily free-tier quota exhausted for this model"))


def test_scores_go_through_rest_with_trace_id_and_comment(monkeypatch):
    from datapilot.eval import langfuse_sink

    sent = []
    settings = type(
        "S",
        (),
        {"langfuse_host": "http://h/", "langfuse_public_key": "pk", "langfuse_secret_key": "sk"},
    )()
    monkeypatch.setattr(langfuse_sink, "get_settings", lambda: settings)
    monkeypatch.setattr(langfuse_sink, "enabled", lambda: True)
    monkeypatch.setattr(
        langfuse_sink, "_post_score", lambda host, auth, payload: sent.append((host, auth, payload))
    )
    langfuse_sink.score_result(
        _with_trace(_res("a", passed=False, detail="missing row", attempts=2), "t1")
    )
    assert [p["name"] for _, _, p in sent] == ["correct", "attempts"]
    assert sent[0][2]["traceId"] == "t1" and sent[0][2]["value"] == 0.0
    assert sent[0][2]["comment"] == "missing row" and sent[1][2]["value"] == 2.0
    assert sent[0][1] == ("pk", "sk")


def test_scoring_is_skipped_without_trace_id_and_never_raises(monkeypatch):
    from datapilot.eval import langfuse_sink

    monkeypatch.setattr(langfuse_sink, "enabled", lambda: True)
    called = []
    monkeypatch.setattr(langfuse_sink, "_post_score", lambda *a: called.append(a))
    langfuse_sink.score_result(_res("a"))  # no trace id
    assert called == []
    monkeypatch.setattr(
        langfuse_sink, "_post_score", lambda *a: (_ for _ in ()).throw(RuntimeError("down"))
    )
    langfuse_sink.score_result(_with_trace(_res("a"), "t"))  # swallowed


def _with_trace(res, trace_id):
    res.trace_id = trace_id
    return res
