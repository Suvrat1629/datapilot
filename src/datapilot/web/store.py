"""Saved conversations in a local SQLite file."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS chats (
    id TEXT PRIMARY KEY, title TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    payload TEXT,
    in_history INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_chat ON messages (chat_id, id);
"""
TITLE_MAX = 60


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class ChatStore:
    def __init__(self, path: str | Path):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.executescript(SCHEMA)
        self._lock = threading.Lock()  # one connection shared by request and worker threads

    def create_chat(self) -> dict[str, Any]:
        chat = {"id": uuid.uuid4().hex[:12], "title": "New chat", "created_at": _now()}
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO chats VALUES (?, ?, ?, ?)",
                (chat["id"], chat["title"], chat["created_at"], chat["created_at"]),
            )
        return chat

    def list_chats(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT id, title, created_at, updated_at FROM chats ORDER BY updated_at DESC, id"
            ).fetchall()
        return [dict(r) for r in rows]

    def exists(self, chat_id: str) -> bool:
        with self._lock:
            return bool(self._db.execute("SELECT 1 FROM chats WHERE id = ?", (chat_id,)).fetchone())

    def delete_chat(self, chat_id: str) -> bool:
        with self._lock, self._db:
            return self._db.execute("DELETE FROM chats WHERE id = ?", (chat_id,)).rowcount > 0

    def rename_chat(self, chat_id: str, title: str) -> bool:
        title = title.strip()[:TITLE_MAX] or "New chat"
        with self._lock, self._db:
            n = self._db.execute(
                "UPDATE chats SET title = ? WHERE id = ?", (title, chat_id)
            ).rowcount
        return n > 0

    def add_message(
        self,
        chat_id: str,
        role: str,
        content: str,
        payload: dict[str, Any] | None = None,
        in_history: bool = False,
    ) -> int:
        with self._lock, self._db:
            cur = self._db.execute(
                "INSERT INTO messages (chat_id, role, content, payload, in_history, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    chat_id,
                    role,
                    content,
                    json.dumps(payload, default=str) if payload is not None else None,
                    int(in_history),
                    _now(),
                ),
            )
            self._db.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (_now(), chat_id))
            if role == "user":
                self._db.execute(
                    "UPDATE chats SET title = ? WHERE id = ? AND title = 'New chat'",
                    (content.strip().replace("\n", " ")[:TITLE_MAX], chat_id),
                )
            return int(cur.lastrowid)

    def mark_in_history(self, message_ids: list[int]) -> None:
        with self._lock, self._db:
            self._db.executemany(
                "UPDATE messages SET in_history = 1 WHERE id = ?", [(i,) for i in message_ids]
            )

    def messages(self, chat_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT id, role, content, payload, created_at FROM messages "
                "WHERE chat_id = ? ORDER BY id",
                (chat_id,),
            ).fetchall()
        return [
            {
                "id": r["id"],
                "role": r["role"],
                "content": r["content"],
                "payload": json.loads(r["payload"]) if r["payload"] else None,
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    def history(self, chat_id: str, limit: int) -> list[dict[str, str]]:
        """Messages the router and SQL Analyst may see (same rule as ChatSession)."""
        with self._lock:
            rows = self._db.execute(
                "SELECT role, content FROM messages WHERE chat_id = ? AND in_history = 1 "
                "ORDER BY id DESC LIMIT ?",
                (chat_id, limit),
            ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
