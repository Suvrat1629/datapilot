"""FastAPI app: saved chats plus a streamed chat turn.

The app is a thin layer over `ChatSession`; all agent behaviour, limits and safety checks are the
same as in the CLI. It has no authentication, so `datapilot serve` binds to localhost.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from datapilot.parent.graph import HISTORY_LIMIT, ChatSession, ParentDeps, counts_for_history
from datapilot.parent.serialize import turn_payload
from datapilot.tracing import (
    flush_langfuse,
    langfuse_config,
    log_turn,
    new_request_id,
)
from datapilot.web.events import StepHandler
from datapilot.web.store import ChatStore

DEV_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
KEEPALIVE_S = 10.0
MAX_MESSAGE_CHARS = 4000


class NewMessage(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)


class Rename(BaseModel):
    title: str = Field(min_length=1, max_length=200)


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, default=str)}\n\n"


def create_app(
    *,
    store: ChatStore,
    deps_factory: Callable[[], ParentDeps] = ParentDeps,
    guard: Callable[[], None] | None = None,
    static_dir: Path | None = None,
    log_path: Path | None = None,
) -> FastAPI:
    busy: set[str] = set()
    busy_lock = threading.Lock()
    state: dict[str, Any] = {"problems": []}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # A failed role check leaves the server up but unusable, with the reasons on /api/health.
        if guard is not None:
            try:
                guard()
            except Exception as e:
                problems = getattr(e, "problems", None) or [f"{type(e).__name__}: {str(e)[:200]}"]
                state["problems"] = list(problems)
        yield

    app = FastAPI(title="DataPilot", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=DEV_ORIGINS,
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["Content-Type"],
    )

    @app.middleware("http")
    async def same_origin_writes(request: Request, call_next):
        """Browsers send Origin on cross-site writes; refuse any that is not us or the dev UI."""
        origin = request.headers.get("origin")
        if request.method not in ("GET", "HEAD", "OPTIONS") and origin:
            own = f"{request.url.scheme}://{request.headers.get('host', '')}"
            if origin not in (own, *DEV_ORIGINS):
                return JSONResponse({"detail": "cross-origin request refused"}, status_code=403)
        return await call_next(request)

    def ready() -> None:
        if state["problems"]:
            raise HTTPException(
                503,
                detail={
                    "message": "The database roles are not safe to run with.",
                    "problems": state["problems"],
                },
            )

    def chat_or_404(chat_id: str) -> str:
        if not store.exists(chat_id):
            raise HTTPException(404, "chat not found")
        return chat_id

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": not state["problems"], "problems": state["problems"]}

    @app.get("/api/chats", dependencies=[Depends(ready)])
    def list_chats() -> list[dict[str, Any]]:
        return store.list_chats()

    @app.post("/api/chats", dependencies=[Depends(ready)], status_code=201)
    def create_chat() -> dict[str, Any]:
        return store.create_chat()

    @app.get("/api/chats/{chat_id}", dependencies=[Depends(ready)])
    def get_chat(chat_id: str) -> dict[str, Any]:
        chat_or_404(chat_id)
        chat = next(c for c in store.list_chats() if c["id"] == chat_id)
        return {**chat, "messages": store.messages(chat_id), "busy": chat_id in busy}

    @app.patch("/api/chats/{chat_id}", dependencies=[Depends(ready)])
    def rename_chat(chat_id: str, body: Rename) -> dict[str, bool]:
        chat_or_404(chat_id)
        return {"ok": store.rename_chat(chat_id, body.title)}

    @app.delete("/api/chats/{chat_id}", dependencies=[Depends(ready)])
    def delete_chat(chat_id: str) -> dict[str, bool]:
        chat_or_404(chat_id)
        if chat_id in busy:
            raise HTTPException(409, "a reply is still being written for this chat")
        return {"ok": store.delete_chat(chat_id)}

    def run_turn(chat_id: str, text: str, user_id: int, emit: Callable[[dict], None]) -> None:
        """Worker thread: run one turn, save it, and always finish with a `result` event."""
        request_id = new_request_id()
        steps: list[dict[str, Any]] = []

        def record(event: dict[str, Any]) -> None:
            if event["type"] == "step":
                steps.append({"label": event["label"], "node": event["node"]})
            elif event["type"] == "step_error" and steps:
                steps[-1]["error"] = event["detail"]
            emit(event)

        start = time.monotonic()
        try:
            session = ChatSession(deps_factory())
            session.history = store.history(chat_id, HISTORY_LIMIT)
            config = langfuse_config(request_id, session_id=chat_id, tags=["web"])
            config["callbacks"] = [*config.get("callbacks", []), StepHandler(record)]
            turn = session.turn(text, config=config)
            latency = time.monotonic() - start
            payload = turn_payload(turn, request_id, latency)
            keep = counts_for_history(turn.status, turn.route, turn.text)
            try:
                log_turn(turn, request_id, chat_id, latency, *([log_path] if log_path else []))
            except Exception:
                pass
        except Exception as e:  # a bug or provider outage must still end the stream cleanly
            latency = time.monotonic() - start
            payload = {
                "request_id": request_id,
                "message": text,
                "route": "none",
                "status": "failed",
                "text": f"Something went wrong while handling that ({type(e).__name__}).",
                "sql": None,
                "assumptions": [],
                "columns": [],
                "rows": [],
                "row_count": 0,
                "truncated": False,
                "chart": None,
                "loads": [],
                "llm_calls": 0,
                "latency_s": round(latency, 2),
            }
            keep = False
        payload["steps"] = steps
        try:
            assistant_id = store.add_message(chat_id, "assistant", payload["text"], payload, keep)
            if keep:
                store.mark_in_history([user_id])
            message = {"id": assistant_id, "role": "assistant", "content": payload["text"]}
            message["payload"] = payload
        except Exception as e:
            message = {
                "id": -1,
                "role": "assistant",
                "content": "The reply could not be saved.",
                "payload": {
                    **payload,
                    "status": "failed",
                    "text": f"Could not save ({type(e).__name__}).",
                },
            }
        finally:
            with busy_lock:
                busy.discard(chat_id)
            try:
                flush_langfuse()
            except Exception:
                pass
        emit({"type": "result", "message": message})

    @app.post("/api/chats/{chat_id}/messages", dependencies=[Depends(ready)])
    async def send_message(chat_id: str, body: NewMessage) -> StreamingResponse:
        chat_or_404(chat_id)
        text = body.text.strip()
        if not text:
            raise HTTPException(422, "message is empty")
        with busy_lock:
            if chat_id in busy:
                raise HTTPException(409, "a reply is still being written for this chat")
            busy.add(chat_id)
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        try:
            user_id = store.add_message(chat_id, "user", text)
        except Exception:
            with busy_lock:
                busy.discard(chat_id)
            raise

        def emit(event: dict[str, Any]) -> None:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, event)
            except RuntimeError:  # loop closed during shutdown
                pass

        threading.Thread(
            target=run_turn, args=(chat_id, text, user_id, emit), daemon=True, name="turn"
        ).start()

        async def stream():
            yield _sse({"type": "start", "user_message_id": user_id})
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), KEEPALIVE_S)
                except TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                yield _sse(event)
                if event["type"] == "result":
                    return

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    if static_dir is not None and (static_dir / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=static_dir / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            if path.startswith("api/"):
                raise HTTPException(404)
            candidate = (static_dir / path).resolve()
            if path and candidate.is_file() and static_dir.resolve() in candidate.parents:
                return FileResponse(candidate)
            return FileResponse(static_dir / "index.html")

    app.state.busy = busy
    return app
