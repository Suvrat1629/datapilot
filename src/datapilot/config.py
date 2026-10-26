from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

Tier = Literal["low", "high"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    admin_database_url: str
    database_url: str
    etl_database_url: str

    llm_low: str = "groq:openai/gpt-oss-20b"
    llm_high: str = "groq:openai/gpt-oss-120b"
    groq_api_key: str = ""
    google_api_key: str = ""

    max_sql_retries: int = 2
    max_rows: int = 1000
    statement_timeout_ms: int = 10_000
    max_llm_calls_per_request: int = 12
    max_context_chars: int = 12_000
    judge_enabled: bool = True
    etl_allowed_hosts: str = ""  # comma-separated exact hostnames; empty means deny all
    # "table.column" entries the ETL role may read from public (column-level grants applied by
    # `datapilot db apply-grants`). Empty means the ETL role reads nothing outside staging.
    etl_readable_columns: str = ""
    etl_max_steps: int = 20  # model calls per ETL request
    etl_max_fetches: int = 50  # HTTP requests per ETL request (including for_each fan-out)
    etl_max_rows: int = 100_000  # rows per staging load
    etl_max_response_bytes: int = 5_000_000
    etl_fetch_timeout_s: float = 15.0
    hints_path: str = ""
    chat_db_path: str = "data/chats.db"  # saved web conversations (SQLite)
    # Schemas the agent may read, comma-separated. `staging` (ETL output) is always included.
    schemas: str = "public"
    ro_role: str = "datapilot_ro"  # role behind DATABASE_URL
    etl_role: str = "datapilot_etl"  # role behind ETL_DATABASE_URL

    langfuse_host: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""


def split_csv(value: str | None) -> list[str]:
    return [x.strip() for x in (value or "").split(",") if x.strip()]


def readable_schemas(settings: Settings) -> tuple[str, ...]:
    """Configured schemas plus `staging`, in a stable order without duplicates."""
    out: list[str] = []
    for s in [*split_csv(settings.schemas), "staging"]:
        if s not in out:
            out.append(s)
    return tuple(out)


@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_llm(tier: Tier, temperature: float = 0.0, max_retries: int | None = None):
    """Build the chat model for a tier from its `provider:model` setting.

    Imports are lazy so the DB-only parts of the project work without LLM keys.
    """
    s = get_settings()
    spec = s.llm_low if tier == "low" else s.llm_high
    provider, _, model = spec.partition(":")
    if not model:
        raise ValueError(
            f"LLM spec for tier {tier!r} must look like 'provider:model', got {spec!r}"
        )

    if provider == "groq":
        from langchain_groq import ChatGroq

        extra = {} if max_retries is None else {"max_retries": max_retries}
        return ChatGroq(
            model=model, temperature=temperature, api_key=s.groq_api_key or None, **extra
        )
    if provider in ("google", "gemini"):
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model, temperature=temperature, google_api_key=s.google_api_key or None
        )
    raise ValueError(f"Unsupported LLM provider {provider!r} (supported: groq, google)")
