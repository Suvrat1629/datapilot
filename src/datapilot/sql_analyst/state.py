from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Status = Literal["ok", "refused", "failed"]


class SafetyVerdict(BaseModel):
    ok: bool
    reason: str = ""


# Structured outputs requested from the LLM.
class CuratedQuestion(BaseModel):
    """A standalone analytics question, or a reason the message is not one."""

    answerable: bool = Field(description="False if this is not a read-only data question.")
    question: str = Field(description="Standalone, precise question. Empty if not answerable.")
    reason: str = Field(default="", description="Why it is not answerable, if so.")


class SqlDraft(BaseModel):
    sql: str = Field(
        default="",
        description="One PostgreSQL SELECT statement, no trailing explanation. Empty only when "
        "cannot_answer_reason is set.",
    )
    cannot_answer_reason: str = Field(
        default="",
        description="Set ONLY when the schema does not contain the data the question needs. "
        "Say what is missing. Never invent columns or substitute other data.",
    )
    assumptions: list[str] = Field(
        default_factory=list, description="Interpretations made, e.g. what 'last week' means."
    )


class JudgeVerdict(BaseModel):
    answers_question: bool
    reason: str = Field(default="", description="What is wrong, if it does not answer.")


class SqlState(BaseModel):
    user_question: str
    history: list[dict[str, str]] = Field(default_factory=list)  # [{"role", "content"}]

    curated_question: str | None = None
    schema_context: str | None = None
    sql: str | None = None
    assumptions: list[str] = Field(default_factory=list)
    safety: SafetyVerdict | None = None

    columns: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] | None = None
    truncated: bool = False

    feedback: str | None = None  # judge objection or DB error fed back into generate_sql
    attempt_log: list[dict[str, Any]] = Field(default_factory=list)  # one entry per SQL attempt
    attempts: int = 0  # SQL generations so far; capped at 1 + max_sql_retries
    llm_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0

    answer: str | None = None
    status: Status | None = None
    reason: str = ""
