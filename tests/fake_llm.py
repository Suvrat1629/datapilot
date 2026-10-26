"""Scripted stand-in for the LLM layer. Replies are queued per output schema."""

from __future__ import annotations

from collections import defaultdict, deque

from datapilot.llm import LLMError, LLMResult
from datapilot.sql_analyst.state import CuratedQuestion, JudgeVerdict, SqlDraft


class ScriptedLLM:
    def __init__(self):
        self.queues: dict[object, deque] = defaultdict(deque)
        self.calls: list[tuple[str, object, list]] = []

    def push(self, schema, *replies):
        self.queues[schema].extend(replies)
        return self

    # convenience builders
    def curate(self, question="q", ok=True, reason=""):
        return self.push(
            CuratedQuestion, CuratedQuestion(answerable=ok, question=question, reason=reason)
        )

    def sql(self, *sqls, assumptions=None):
        for sql in sqls:
            self.push(SqlDraft, SqlDraft(sql=sql, assumptions=assumptions or []))
        return self

    def judge(self, ok=True, reason=""):
        return self.push(JudgeVerdict, JudgeVerdict(answers_question=ok, reason=reason))

    def answer(self, text="the answer"):
        return self.push(None, text)

    def __call__(self, tier, schema, messages):
        self.calls.append((tier, schema, messages))
        q = self.queues[schema]
        if not q:
            raise AssertionError(f"unexpected LLM call for schema {schema}")
        reply = q.popleft()
        if isinstance(reply, Exception):
            raise reply
        return LLMResult(reply, tokens_in=10, tokens_out=5)

    def count(self, schema) -> int:
        return sum(1 for _, s, _ in self.calls if s is schema)

    def last_human(self, schema) -> str:
        for _, s, msgs in reversed(self.calls):
            if s is schema:
                return msgs[-1].content
        raise AssertionError("no call recorded")


__all__ = ["LLMError", "ScriptedLLM"]
