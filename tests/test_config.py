import pytest

from datapilot import config

URLS = dict(admin_database_url="x", database_url="x", etl_database_url="x")


def test_get_llm_rejects_bad_spec(monkeypatch):
    monkeypatch.setattr(config, "get_settings", lambda: config.Settings(llm_low="nomodel", **URLS))
    with pytest.raises(ValueError, match="provider:model"):
        config.get_llm("low")


def test_get_llm_rejects_unknown_provider(monkeypatch):
    monkeypatch.setattr(config, "get_settings", lambda: config.Settings(llm_low="foo:bar", **URLS))
    with pytest.raises(ValueError, match="Unsupported"):
        config.get_llm("low")


def test_get_llm_builds_groq(monkeypatch):
    monkeypatch.setattr(
        config,
        "get_settings",
        lambda: config.Settings(llm_low="groq:openai/gpt-oss-20b", groq_api_key="test", **URLS),
    )
    assert type(config.get_llm("low")).__name__ == "ChatGroq"


def test_database_urls_are_required(monkeypatch):
    for k in ("ADMIN_DATABASE_URL", "DATABASE_URL", "ETL_DATABASE_URL"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(Exception, match="database_url"):
        config.Settings(_env_file=None)


def test_unsafe_defaults_are_closed():
    s = config.Settings(_env_file=None, **URLS)
    assert s.etl_allowed_hosts == "" and s.hints_path == ""
