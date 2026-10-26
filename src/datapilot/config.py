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

    llm_low: str = "groq:llama-3.1-8b-instant"
    llm_high: str = "google:gemini-flash-latest"
    groq_api_key: str = ""
    google_api_key: str = ""

    max_sql_retries: int = 2
    max_rows: int = 1000
    statement_timeout_ms: int = 10_000
    max_llm_calls_per_request: int = 12
    etl_allowed_hosts: str = ""
    hints_path: str = ""

    langfuse_host: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_llm(tier: Tier, temperature: float = 0.0):
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

        return ChatGroq(model=model, temperature=temperature, api_key=s.groq_api_key or None)
    if provider in ("google", "gemini"):
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model, temperature=temperature, google_api_key=s.google_api_key or None
        )
    raise ValueError(f"Unsupported LLM provider {provider!r} (supported: groq, google)")
