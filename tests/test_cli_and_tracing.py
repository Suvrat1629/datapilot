import json

from typer.testing import CliRunner

from datapilot import cli, tracing
from datapilot.sql_analyst.state import SafetyVerdict, SqlState

runner = CliRunner()


def _state(**kw):
    base = dict(
        user_question="q",
        curated_question="q2",
        sql="SELECT 1",
        safety=SafetyVerdict(ok=True),
        attempts=1,
        llm_calls=4,
        tokens_in=40,
        tokens_out=20,
        status="ok",
        answer="Bengaluru had the most.",
        columns=["name", "n"],
        rows=[{"name": "Bengaluru", "n": 1342}],
    )
    return SqlState(**{**base, **kw})


def test_log_request_appends_json_lines(tmp_path):
    path = tmp_path / "logs" / "t.jsonl"
    tracing.log_request(_state(), "abc12345", 1.234, path)
    tracing.log_request(_state(status="refused", reason="no"), "def", 0.5, path)
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["request_id"] for r in lines] == ["abc12345", "def"]
    assert lines[0]["latency_ms"] == 1234 and lines[0]["row_count"] == 1
    assert lines[0]["sql"] == "SELECT 1" and lines[0]["tokens_in"] == 40
    assert "attempt_log" in lines[0]
    assert lines[1]["status"] == "refused"


def test_log_failure_never_raises(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    tracing.log_request(_state(), "id", 1.0, blocker / "sub" / "t.jsonl")  # parent is a file


def test_langfuse_off_without_config(monkeypatch):
    monkeypatch.setattr(
        tracing,
        "get_settings",
        lambda: type(
            "S", (), {"langfuse_host": "", "langfuse_public_key": "", "langfuse_secret_key": ""}
        )(),
    )
    assert tracing.langfuse_config("id") == {}


def _run(monkeypatch, state, args, tmp_path):
    monkeypatch.setattr(cli, "run_sql_analyst", lambda q, **kw: state)
    monkeypatch.setattr(cli, "langfuse_config", lambda rid: {})
    monkeypatch.setattr(tracing, "DEFAULT_LOG_PATH", tmp_path / "t.jsonl")
    monkeypatch.setattr(
        cli,
        "log_request",
        lambda s, rid, lat: tracing.log_request(s, rid, lat, tmp_path / "t.jsonl"),
    )
    return runner.invoke(cli.app, ["ask", "q", *args])


def test_ask_prints_answer_table_and_sql(monkeypatch, tmp_path):
    r = _run(monkeypatch, _state(), ["--show-sql"], tmp_path)
    assert r.exit_code == 0
    assert "Bengaluru had the most." in r.output and "1342" in r.output
    assert "SELECT 1" in r.output and "4 model calls" in r.output
    assert (tmp_path / "t.jsonl").exists()


def test_ask_refusal_exits_nonzero_with_reason(monkeypatch, tmp_path):
    r = _run(
        monkeypatch, _state(status="refused", reason="Hidden column.", answer=None), [], tmp_path
    )
    assert r.exit_code == 1 and "Hidden column." in r.output


def test_ask_json_output_is_parseable(monkeypatch, tmp_path):
    r = _run(monkeypatch, _state(), ["--json"], tmp_path)
    payload = json.loads(r.output)
    assert payload["status"] == "ok" and payload["rows"][0]["n"] == 1342
    assert "schema_context" not in payload and payload["request_id"]


def test_flush_is_a_noop_without_langfuse_config(monkeypatch):
    monkeypatch.setattr(
        tracing,
        "get_settings",
        lambda: type(
            "S", (), {"langfuse_host": "", "langfuse_public_key": "", "langfuse_secret_key": ""}
        )(),
    )
    tracing.flush_langfuse()  # must not raise or import the SDK


def test_langfuse_setup_failure_is_swallowed(monkeypatch):
    monkeypatch.setattr(
        tracing,
        "get_settings",
        lambda: type(
            "S",
            (),
            {"langfuse_host": "http://x", "langfuse_public_key": "pk", "langfuse_secret_key": "sk"},
        )(),
    )
    import langfuse

    monkeypatch.setattr(
        langfuse, "Langfuse", lambda **kw: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    tracing.langfuse_client.cache_clear()
    try:
        assert tracing.langfuse_config("id") == {}
    finally:
        tracing.langfuse_client.cache_clear()


def test_langfuse_client_is_built_from_settings_not_environment(monkeypatch):
    import langfuse

    seen = {}
    monkeypatch.setattr(langfuse, "Langfuse", lambda **kw: seen.update(kw) or object())
    monkeypatch.setattr(
        tracing,
        "get_settings",
        lambda: type(
            "S",
            (),
            {"langfuse_host": "http://h", "langfuse_public_key": "pk", "langfuse_secret_key": "sk"},
        )(),
    )
    tracing.langfuse_client.cache_clear()
    try:
        tracing.langfuse_client()
        tracing.langfuse_client()  # cached: constructed once
    finally:
        tracing.langfuse_client.cache_clear()
    assert seen == {"public_key": "pk", "secret_key": "sk", "host": "http://h"}
