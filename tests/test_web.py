"""Web API tests: streaming turns, saved chats, safety gate. No live models."""

import json
import threading

import pytest
from fastapi.testclient import TestClient

from datapilot.config import Settings
from datapilot.guard import UnsafeRole
from datapilot.parent.charts import suggest_chart
from datapilot.parent.graph import ParentDeps, TurnResult
from datapilot.parent.serialize import MAX_PAYLOAD_ROWS, turn_payload
from datapilot.sql_analyst.state import SqlState
from datapilot.web.app import create_app
from datapilot.web.events import StepHandler
from datapilot.web.store import ChatStore
from tests.fake_llm import ScriptedLLM
from tests.test_parent import Children, decide


def make_app(tmp_path, llm=None, children=None, guard=None, name="chats.db"):
    llm = llm or ScriptedLLM()
    children = children or Children()
    store = ChatStore(tmp_path / name)

    def deps():
        return ParentDeps(
            settings=Settings(etl_allowed_hosts="api.open-meteo.com"),
            call_llm=llm,
            run_sql=children.run_sql,
            run_etl=children.run_etl,
            list_tables=lambda: ["bookings"],
        )

    app = create_app(store=store, deps_factory=deps, guard=guard, log_path=tmp_path / "t.jsonl")
    return app, llm, children


def send(client, chat_id, text):
    """POST a message and return the list of SSE events."""
    events = []
    with client.stream("POST", f"/api/chats/{chat_id}/messages", json={"text": text}) as r:
        assert r.status_code == 200, r.read()
        for line in r.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    return events


def new_chat(client):
    return client.post("/api/chats").json()["id"]


# ---- a turn --------------------------------------------------------------------------------


def test_sql_turn_streams_steps_then_result_and_saves(tmp_path):
    sql = SqlState(
        user_question="q",
        status="ok",
        answer="Mumbai leads.",
        sql="SELECT city, n FROM x",
        columns=["city", "n"],
        rows=[{"city": "Mumbai", "n": 5}, {"city": "Delhi", "n": 3}],
        assumptions=["last week = previous calendar week"],
        llm_calls=4,
    )
    app, llm, ch = make_app(tmp_path, children=Children(sql=sql))
    decide(llm, route="sql", sql_question="who leads?")
    with TestClient(app) as c:
        chat = new_chat(c)
        events = send(c, chat, "who leads?")
        types = [e["type"] for e in events]
        assert types[0] == "start" and types[-1] == "result" and "step" in types
        assert events[1]["label"] == "Understanding your request"
        msg = events[-1]["message"]
        p = msg["payload"]
        assert p["status"] == "ok" and p["route"] == "sql" and p["text"] == "Mumbai leads."
        assert p["sql"] == "SELECT city, n FROM x" and p["columns"] == ["city", "n"]
        assert p["chart"] == {"type": "bar", "x": "city", "y": ["n"]}
        assert p["assumptions"] == ["last week = previous calendar week"]
        assert [s["label"] for s in p["steps"]][0] == "Understanding your request"

        saved = c.get(f"/api/chats/{chat}").json()
        assert [m["role"] for m in saved["messages"]] == ["user", "assistant"]
        assert saved["title"] == "who leads?" and saved["busy"] is False
        assert saved["messages"][1]["payload"]["rows"][0]["city"] == "Mumbai"
        assert [c_["id"] for c_ in c.get("/api/chats").json()] == [chat]


def test_history_survives_restart_and_skips_small_talk(tmp_path):
    app, llm, ch = make_app(tmp_path)
    decide(llm, route="sql", sql_question="How many bookings?")
    decide(llm, route="chat", reply="You're welcome")
    with TestClient(app) as c:
        chat = new_chat(c)
        send(c, chat, "how many bookings")
        send(c, chat, "thanks")

    app2, llm2, ch2 = make_app(tmp_path)  # a new process on the same file
    decide(llm2, route="sql", sql_question="and in Delhi?")
    with TestClient(app2) as c:
        assert [m["content"] for m in c.get(f"/api/chats/{chat}").json()["messages"]][:1] == [
            "how many bookings"
        ]
        send(c, chat, "and Delhi?")
    # Only the successful SQL turn is remembered; "thanks" (chat) is not.
    assert ch2.calls[0][2] == [
        {"role": "user", "content": "how many bookings"},
        {"role": "assistant", "content": "SQL answer"},
    ]


def test_etl_turn_reports_loads(tmp_path):
    app, llm, ch = make_app(tmp_path)
    decide(llm, route="etl", etl_request="load weather")
    with TestClient(app) as c:
        p = send(c, new_chat(c), "load weather")[-1]["message"]["payload"]
    assert p["status"] == "ok" and p["loads"][0]["table"] == "weather_daily"
    assert p["loads"][0]["rows"] == 360 and p["sql"] is None and p["chart"] is None


def test_failed_turn_still_ends_the_stream_and_frees_the_chat(tmp_path):
    class Boom(Children):
        def run_sql(self, *a, **k):
            raise RuntimeError("secret internals: db password hunter2")

    app, llm, ch = make_app(tmp_path, children=Boom())
    decide(llm, route="sql", sql_question="q")
    decide(llm, route="chat", reply="ok")
    with TestClient(app) as c:
        chat = new_chat(c)
        events = send(c, chat, "q")
        p = events[-1]["message"]["payload"]
        assert events[-1]["type"] == "result" and p["status"] == "failed"
        assert "hunter2" not in json.dumps(events)  # internals are not echoed to the browser
        assert send(c, chat, "hi")[-1]["message"]["payload"]["status"] == "ok"  # not stuck busy
        msgs = c.get(f"/api/chats/{chat}").json()["messages"]
        assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]


def test_second_turn_on_a_busy_chat_gets_409(tmp_path):
    release, started = threading.Event(), threading.Event()

    class Slow(Children):
        def run_sql(self, *a, **k):
            started.set()
            release.wait(10)
            return super().run_sql(*a, **k)

    app, llm, ch = make_app(tmp_path, children=Slow())
    decide(llm, route="sql", sql_question="q")
    with TestClient(app) as c:
        chat = new_chat(c)
        first = threading.Thread(target=lambda: send(c, chat, "q"))
        first.start()
        assert started.wait(10)
        r = c.post(f"/api/chats/{chat}/messages", json={"text": "again"})
        assert r.status_code == 409
        assert c.delete(f"/api/chats/{chat}").status_code == 409
        assert c.get(f"/api/chats/{chat}").json()["busy"] is True
        release.set()
        first.join(10)
        assert c.get(f"/api/chats/{chat}").json()["busy"] is False
        assert c.delete(f"/api/chats/{chat}").status_code == 200


# ---- gate and validation -------------------------------------------------------------------


def test_unsafe_roles_make_every_endpoint_503(tmp_path):
    def guard():
        raise UnsafeRole("postgres", "ro", ["the role has the SUPERUSER attribute"])

    app, *_ = make_app(tmp_path, guard=guard)
    with TestClient(app) as c:
        h = c.get("/api/health").json()
        assert h["ok"] is False and "SUPERUSER" in h["problems"][0]
        for method, url in (("get", "/api/chats"), ("post", "/api/chats")):
            r = getattr(c, method)(url)
            assert r.status_code == 503 and "SUPERUSER" in json.dumps(r.json())


def test_guard_that_crashes_also_blocks(tmp_path):
    def guard():
        raise ConnectionError("cannot reach database")

    app, *_ = make_app(tmp_path, guard=guard)
    with TestClient(app) as c:
        assert c.get("/api/chats").status_code == 503
        assert "ConnectionError" in c.get("/api/health").json()["problems"][0]


def test_healthy_guard_allows_use(tmp_path):
    app, *_ = make_app(tmp_path, guard=lambda: None)
    with TestClient(app) as c:
        assert c.get("/api/health").json() == {"ok": True, "problems": []}
        assert c.get("/api/chats").status_code == 200


@pytest.mark.parametrize("body", [{"text": ""}, {"text": "x" * 4001}, {}, {"text": 5}])
def test_bad_message_bodies_are_rejected(tmp_path, body):
    app, *_ = make_app(tmp_path)
    with TestClient(app) as c:
        chat = new_chat(c)
        assert c.post(f"/api/chats/{chat}/messages", json=body).status_code == 422
        assert c.get(f"/api/chats/{chat}").json()["messages"] == []


def test_whitespace_only_message_is_rejected(tmp_path):
    app, *_ = make_app(tmp_path)
    with TestClient(app) as c:
        chat = new_chat(c)
        assert c.post(f"/api/chats/{chat}/messages", json={"text": "   \n"}).status_code == 422


def test_unknown_chat_is_404_and_cross_origin_writes_are_refused(tmp_path):
    app, *_ = make_app(tmp_path)
    with TestClient(app) as c:
        assert c.get("/api/chats/nope").status_code == 404
        assert c.post("/api/chats/nope/messages", json={"text": "hi"}).status_code == 404
        r = c.post("/api/chats", headers={"Origin": "https://evil.example.com"})
        assert r.status_code == 403
        assert c.post("/api/chats", headers={"Origin": "http://localhost:5173"}).status_code == 201
        assert c.post("/api/chats", headers={"Origin": "http://testserver"}).status_code == 201


def test_rename_and_delete(tmp_path):
    app, *_ = make_app(tmp_path)
    with TestClient(app) as c:
        chat = new_chat(c)
        assert c.patch(f"/api/chats/{chat}", json={"title": "  Weather  "}).json() == {"ok": True}
        assert c.get(f"/api/chats/{chat}").json()["title"] == "Weather"
        assert c.delete(f"/api/chats/{chat}").json() == {"ok": True}
        assert c.get("/api/chats").json() == []


def test_markup_in_messages_is_stored_verbatim(tmp_path):
    """The server never interprets text; the UI must render it as text."""
    app, llm, _ = make_app(tmp_path)
    decide(llm, route="chat", reply="<img src=x onerror=alert(1)>")
    with TestClient(app) as c:
        chat = new_chat(c)
        p = send(c, chat, "<script>alert(1)</script>")[-1]["message"]["payload"]
        assert p["text"] == "<img src=x onerror=alert(1)>"
        assert c.get(f"/api/chats/{chat}").json()["messages"][0]["content"].startswith("<script>")


# ---- store, serializer, charts, steps ------------------------------------------------------


def test_store_history_rule_and_ordering(tmp_path):
    s = ChatStore(tmp_path / "s.db")
    chat = s.create_chat()["id"]
    ids = [s.add_message(chat, "user", f"u{i}") for i in range(3)]
    s.add_message(chat, "assistant", "a0", {"k": 1}, in_history=True)
    s.mark_in_history(ids[:1])
    assert [m["content"] for m in s.history(chat, 10)] == ["u0", "a0"]
    assert s.history(chat, 1) == [{"role": "assistant", "content": "a0"}]
    assert s.messages(chat)[3]["payload"] == {"k": 1}


def test_payload_caps_rows_and_hides_internals():
    rows = [{"d": f"2026-01-{i % 28 + 1:02d}", "n": i} for i in range(MAX_PAYLOAD_ROWS + 50)]
    sql = SqlState(
        user_question="q",
        status="ok",
        columns=["d", "n"],
        rows=rows,
        schema_context="SECRET SCHEMA",
    )
    turn = TurnResult("q", "sql", "ok", "answer", sql=sql, llm_calls=3)
    p = turn_payload(turn, "rid", 1.234)
    assert len(p["rows"]) == MAX_PAYLOAD_ROWS and p["row_count"] == len(rows)
    assert p["truncated"] is True and p["latency_s"] == 1.23
    assert "SECRET SCHEMA" not in json.dumps(p)


def test_payload_for_failed_turn_has_no_rows():
    sql = SqlState(user_question="q", status="failed", rows=[{"a": 1}], columns=["a"])
    p = turn_payload(TurnResult("q", "sql", "failed", "nope", sql=sql), "r", 0.1)
    assert p["rows"] == [] and p["columns"] == [] and p["chart"] is None


@pytest.mark.parametrize(
    "columns,rows,expected",
    [
        (
            ["city", "n"],
            [{"city": "A", "n": 1}, {"city": "B", "n": 2}],
            {"type": "bar", "x": "city", "y": ["n"]},
        ),
        (
            ["day", "n", "m"],
            [{"day": "2026-01-01", "n": 1, "m": 2.5}, {"day": "2026-01-02", "n": 2, "m": 1}],
            {"type": "line", "x": "day", "y": ["n", "m"]},
        ),
        (["city", "n"], [{"city": "A", "n": 1}], None),  # one row
        (["n"], [{"n": 1}, {"n": 2}], None),  # no label
        (["a", "b"], [{"a": "x", "b": "y"}, {"a": "z", "b": "w"}], None),  # no numbers
        (["city", "id"], [{"city": "A", "id": 1}, {"city": "B", "id": 2}], None),  # only an id
        (["city", "n"], [{"city": "A", "n": 1}, {"city": "A", "n": 2}], None),  # repeated labels
        (["a", "b", "n"], [{"a": "x", "b": "y", "n": 1}, {"a": "z", "b": "w", "n": 2}], None),
        (["city", "n"], [{"city": f"c{i}", "n": i} for i in range(51)], None),  # too many bars
        (["city", "n"], [{"city": "A", "n": True}, {"city": "B", "n": False}], None),  # bools
        (["city", "n"], [{"city": "A", "n": None}, {"city": "B", "n": None}], None),
        (
            ["city", "n"],
            [{"city": "A", "n": 1}, {"city": None, "n": 2}],
            {"type": "bar", "x": "city", "y": ["n"]},
        ),
        (["city", "n"], [], None),
    ],
)
def test_suggest_chart(columns, rows, expected):
    assert suggest_chart(columns, rows) == expected


def test_step_handler_labels_and_ignores_internals():
    events = []
    h = StepHandler(events.append)
    h.on_chain_start(
        {}, {}, run_id=1, metadata={"langgraph_node": "generate_sql"}, name="generate_sql"
    )
    h.on_chain_start({}, {}, run_id=2, metadata={"langgraph_node": "generate_sql"}, name="route")
    h.on_chain_start({}, {}, run_id=3, metadata={"langgraph_node": "tools"}, name="tools")
    h.on_chain_start({}, {}, run_id=4, metadata={}, name="x")
    h.on_chain_start(
        {}, {}, run_id=5, metadata={"langgraph_node": "generate_sql"}, name="generate_sql"
    )
    h.on_tool_start(
        {"name": "fetch_api"},
        "",
        run_id=6,
        inputs={"url": "https://api.open-meteo.com/v1/forecast", "for_each": "d1"},
    )
    h.on_tool_start({"name": "load_staging"}, "", run_id=7, inputs={"table": "wx"})
    h.on_tool_start({"name": "mystery"}, "", run_id=8, inputs={})
    h.on_tool_end("ERROR: host not allow-listed", run_id=6)
    labels = [e.get("label") or e["type"] for e in events]
    assert labels == [
        "Writing SQL",
        "Rewriting the SQL (attempt 2)",
        "Fetching data from api.open-meteo.com (one request per row)",
        "Loading into staging.wx",
        "step_error",
    ]


@pytest.mark.db
def test_real_sql_graph_emits_live_steps_in_order(tmp_path):
    from datapilot.hints import load_hints
    from datapilot.sql_analyst.graph import run_sql_analyst
    from datapilot.sql_analyst.nodes import Deps

    sql_llm = (
        ScriptedLLM()
        .curate("How many cities?")
        .sql("SELECT count(*) AS n FROM cities")
        .judge(True)
        .answer("There are some cities.")
    )
    sql_deps = Deps(settings=Settings(), hints=load_hints(""), call_llm=sql_llm)
    router = decide(ScriptedLLM(), route="sql", sql_question="How many cities?")
    store = ChatStore(tmp_path / "c.db")
    app = create_app(
        store=store,
        deps_factory=lambda: ParentDeps(
            settings=Settings(),
            call_llm=router,
            run_sql=run_sql_analyst,
            sql_deps=sql_deps,
            list_tables=lambda: ["cities"],
        ),
        log_path=tmp_path / "t.jsonl",
    )
    with TestClient(app) as c:
        events = send(c, new_chat(c), "how many cities?")
    labels = [e["label"] for e in events if e["type"] == "step"]
    assert labels == [
        "Understanding your request",
        "Rephrasing the question",
        "Reading the database schema",
        "Writing SQL",
        "Checking the query is safe",
        "Running the query",
        "Writing the answer",
    ]
    p = events[-1]["message"]["payload"]
    assert p["status"] == "ok" and p["rows"][0]["n"] > 0 and p["sql"].lower().startswith("select")
