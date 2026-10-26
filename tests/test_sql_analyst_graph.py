import pytest

from datapilot.config import Settings
from datapilot.hints import DEMO_HINTS_PATH, load_hints
from datapilot.llm import LLMError
from datapilot.sql_analyst.graph import run_sql_analyst
from datapilot.sql_analyst.nodes import Deps
from datapilot.sql_analyst.state import CuratedQuestion, JudgeVerdict, SqlDraft
from tests.fake_llm import ScriptedLLM

pytestmark = pytest.mark.db

URLS = dict(admin_database_url="x", database_url="x", etl_database_url="x")
GOOD_SQL = (
    "SELECT c.name, count(*) AS cancellations FROM bookings b JOIN cities c ON c.id = b.city_id "
    "WHERE b.status = 'cancelled' GROUP BY c.name ORDER BY cancellations DESC LIMIT 1"
)


def make(llm: ScriptedLLM, **settings) -> Deps:
    from datapilot import db

    s = Settings(_env_file=None, **URLS, **settings)
    return Deps(
        settings=s,
        hints=load_hints(DEMO_HINTS_PATH),
        call_llm=llm,
        connect=lambda: db.connect("ro"),
    )


def test_happy_path():
    llm = ScriptedLLM().curate("Which city has the most cancellations?").sql(GOOD_SQL)
    llm.judge(True).answer("Bengaluru.")
    st = run_sql_analyst("which city cancels most", deps=make(llm))
    assert st.status == "ok" and st.answer == "Bengaluru."
    assert st.attempts == 1 and st.llm_calls == 4
    assert st.columns == ["name", "cancellations"] and len(st.rows) == 1
    assert st.rows[0]["name"] == "Bengaluru"
    assert st.tokens_in == 40 and st.tokens_out == 20
    assert st.safety.ok


def test_models_are_tiered():
    llm = ScriptedLLM().curate().sql(GOOD_SQL).judge().answer()
    run_sql_analyst("q", deps=make(llm))
    tiers = {s: t for t, s, _ in llm.calls}
    assert tiers[SqlDraft] == "high"
    assert tiers[CuratedQuestion] == "low" and tiers[JudgeVerdict] == "low" and tiers[None] == "low"


def test_not_a_question_is_refused_without_more_calls():
    llm = ScriptedLLM().curate(ok=False, reason="That asks to delete data.")
    st = run_sql_analyst("delete all rides", deps=make(llm))
    assert st.status == "refused" and "delete" in st.reason
    assert len(llm.calls) == 1


@pytest.mark.parametrize(
    "bad_sql",
    ["DROP TABLE bookings", "SELECT 1; DELETE FROM bookings", "SELECT email FROM customers"],
)
def test_unsafe_sql_refused_and_never_executed(bad_sql):
    llm = ScriptedLLM().curate().sql(bad_sql)
    st = run_sql_analyst("q", deps=make(llm))
    assert st.status == "refused" and st.reason
    assert not st.safety.ok
    assert st.rows is None
    assert llm.count(JudgeVerdict) == 0  # static refusal short-circuits the judge
    assert len(llm.calls) == 2


def test_retry_after_database_error_then_success():
    bad = "SELECT nope FROM bookings"
    llm = ScriptedLLM().curate().sql(bad).judge().sql(GOOD_SQL).judge().answer("ok")
    st = run_sql_analyst("q", deps=make(llm))
    assert st.status == "ok" and st.attempts == 2
    retry_prompt = [m for t, s, m in llm.calls if s is SqlDraft][1][-1].content
    assert bad in retry_prompt and 'column "nope" does not exist' in retry_prompt


def test_retries_exhausted_fails_with_reason():
    bad = "SELECT nope FROM bookings"
    llm = ScriptedLLM().curate().sql(bad, bad, bad).judge().judge().judge()
    st = run_sql_analyst("q", deps=make(llm, max_sql_retries=2))
    assert st.status == "failed" and st.attempts == 3
    assert "3 attempts" in st.reason and "nope" in st.reason


def test_zero_retries_means_one_attempt():
    llm = ScriptedLLM().curate().sql("SELECT nope FROM bookings").judge()
    st = run_sql_analyst("q", deps=make(llm, max_sql_retries=0))
    assert st.status == "failed" and st.attempts == 1


def test_judge_objection_triggers_retry():
    wrong = "SELECT c.name, count(*) FROM bookings b JOIN cities c ON c.id=b.city_id GROUP BY 1"
    llm = ScriptedLLM().curate().sql(wrong).judge(False, "missing the cancelled filter")
    llm.sql(GOOD_SQL).judge(True).answer("Bengaluru")
    st = run_sql_analyst("q", deps=make(llm))
    assert st.status == "ok" and st.attempts == 2
    assert "missing the cancelled filter" in llm.last_human(SqlDraft)
    assert st.rows[0]["name"] == "Bengaluru"


def test_judge_objection_exhausted_does_not_execute():
    llm = ScriptedLLM().curate().sql(GOOD_SQL, GOOD_SQL)
    llm.judge(False, "wrong").judge(False, "still wrong")
    st = run_sql_analyst("q", deps=make(llm, max_sql_retries=1))
    assert st.status == "failed" and st.rows is None and "still wrong" in st.reason


def test_judge_can_be_disabled():
    llm = ScriptedLLM().curate().sql(GOOD_SQL).answer("ok")
    st = run_sql_analyst("q", deps=make(llm, judge_enabled=False))
    assert st.status == "ok" and llm.count(JudgeVerdict) == 0


def test_database_permission_error_is_refused_not_retried():
    # `SELECT *` passes the static check but Postgres blocks the hidden columns.
    llm = ScriptedLLM().curate().sql("SELECT * FROM customers LIMIT 3").judge()
    st = run_sql_analyst("q", deps=make(llm))
    assert st.status == "refused" and "not permitted" in st.reason
    assert st.attempts == 1


def test_row_cap_marks_truncation():
    llm = ScriptedLLM().curate().sql("SELECT id FROM bookings").judge().answer("many")
    st = run_sql_analyst("q", deps=make(llm, max_rows=5))
    assert st.status == "ok" and st.truncated and len(st.rows) == 5


def test_llm_call_budget_enforced():
    llm = ScriptedLLM().curate().sql(GOOD_SQL).judge().answer()
    st = run_sql_analyst("q", deps=make(llm, max_llm_calls_per_request=2))
    assert st.status == "failed" and "Too many model calls" in st.reason
    assert len(llm.calls) == 2


def test_llm_outage_fails_cleanly():
    llm = ScriptedLLM().curate().push(SqlDraft, LLMError("provider down"))
    st = run_sql_analyst("q", deps=make(llm))
    assert st.status == "failed" and "provider down" in st.reason


def test_represent_failure_keeps_the_result():
    llm = ScriptedLLM().curate().sql(GOOD_SQL).judge().push(None, LLMError("quota"))
    st = run_sql_analyst("q", deps=make(llm))
    assert st.status == "ok" and "1 row" in st.answer and st.rows


def test_history_reaches_the_curate_prompt():
    llm = ScriptedLLM().curate().sql(GOOD_SQL).judge().answer()
    hist = [
        {"role": "user", "content": "top cities by revenue"},
        {"role": "assistant", "content": "Mumbai"},
    ]
    run_sql_analyst("and cancellations?", history=hist, deps=make(llm))
    assert "top cities by revenue" in llm.last_human(CuratedQuestion)


def test_injected_limit_is_cap_plus_one_so_truncation_is_detectable():
    llm = ScriptedLLM().curate().sql("SELECT id FROM bookings").judge().answer()
    st = run_sql_analyst("q", deps=make(llm, max_rows=7))
    assert st.sql.endswith("LIMIT 8")
    assert len(st.rows) == 7 and st.truncated


def test_exact_cap_rows_is_not_truncated():
    llm = ScriptedLLM().curate().sql("SELECT id FROM cities LIMIT 12").judge().answer()
    st = run_sql_analyst("q", deps=make(llm, max_rows=12))
    assert len(st.rows) == 12 and not st.truncated


def test_judge_outage_does_not_fail_the_request():
    llm = ScriptedLLM().curate().sql(GOOD_SQL).push(JudgeVerdict, LLMError("judge provider down"))
    llm.answer("Bengaluru.")
    st = run_sql_analyst("q", deps=make(llm))
    assert st.status == "ok" and st.rows[0]["name"] == "Bengaluru"


def test_judge_prompt_includes_todays_date_and_allows_literal_dates():
    import datetime as dt

    llm = ScriptedLLM().curate().sql(GOOD_SQL).judge().answer()
    deps = make(llm)
    deps.today = lambda: dt.date(2026, 10, 26)
    run_sql_analyst("q", deps=deps)
    judge_system = [m for t, s, m in llm.calls if s is JudgeVerdict][0][0].content
    assert "2026-10-26" in judge_system and "Literal dates are NOT an error" in judge_system
    gen_system = [m for t, s, m in llm.calls if s is SqlDraft][0][0].content
    assert "Prefer date expressions" in gen_system


def test_missing_data_is_refused_immediately_without_judge_or_retries():
    llm = ScriptedLLM().curate("List customer emails")
    llm.push(SqlDraft, SqlDraft(cannot_answer_reason="there is no email column"))
    st = run_sql_analyst("list customer emails", deps=make(llm))
    assert st.status == "refused" and "no email column" in st.reason
    assert st.rows is None and llm.count(JudgeVerdict) == 0 and len(llm.calls) == 2


def test_empty_sql_without_reason_is_refused_not_executed():
    llm = ScriptedLLM().curate().push(SqlDraft, SqlDraft(sql="  "))
    st = run_sql_analyst("q", deps=make(llm))
    assert st.status == "refused" and st.rows is None


def test_represent_prompt_carries_schema_notes_for_units():
    llm = ScriptedLLM().curate().sql(GOOD_SQL).judge().answer()
    run_sql_analyst("q", deps=make(llm))
    prompt = llm.last_human(None)
    assert "Schema notes" in prompt and "INR" in prompt


def test_attempt_log_records_each_attempt_and_why_it_failed():
    bad = "SELECT nope FROM bookings"
    llm = ScriptedLLM().curate().sql(bad).judge().sql(GOOD_SQL).judge(False, "wrong filter")
    llm.sql(GOOD_SQL).judge().answer("ok")
    st = run_sql_analyst("q", deps=make(llm))
    log = st.attempt_log
    assert [e["attempt"] for e in log] == [1, 2, 3]
    assert "nope" in log[0]["problem"] and log[0]["sql"] == bad
    assert log[1]["problem"] == "judge objected: wrong filter"
    assert log[2]["problem"] is None


def test_attempt_log_records_unsafe_and_unanswerable():
    st = run_sql_analyst("q", deps=make(ScriptedLLM().curate().sql("DROP TABLE bookings")))
    assert st.attempt_log[0]["problem"].startswith("unsafe:")
    llm = ScriptedLLM().curate().push(SqlDraft, SqlDraft(cannot_answer_reason="no such data"))
    st = run_sql_analyst("q", deps=make(llm))
    assert "cannot answer" in st.attempt_log[0]["problem"]
