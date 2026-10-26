import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, api, sendMessage } from "./api";
import AssistantMessage, { PendingMessage } from "./components/AssistantMessage";
import Composer from "./components/Composer";
import Sidebar from "./components/Sidebar";
import type { ChatDetail, ChatMessage, ChatSummary, Step, StreamEvent } from "./types";

const STARTERS = [
  "Which city had the most cancellations last week?",
  "Show the average fare by vehicle type",
  "Load the last 30 days of weather for every city",
];

const hashId = () => window.location.hash.replace(/^#\/chat\//, "") || null;

export default function App() {
  const [chats, setChats] = useState<ChatSummary[]>([]);
  const [activeId, setActiveId] = useState<string | null>(hashId());
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [pending, setPending] = useState<Step[] | null>(null);
  const [problems, setProblems] = useState<string[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sidebar, setSidebar] = useState(false);
  const bottom = useRef<HTMLDivElement>(null);
  const active = useRef<string | null>(activeId);
  active.current = activeId;

  const refreshChats = useCallback(async () => setChats(await api.listChats()), []);

  const open = useCallback(async (id: string) => {
    setActiveId(id);
    window.location.hash = `/chat/${id}`;
    setError(null);
    setSidebar(false);
    try {
      const detail: ChatDetail = await api.getChat(id);
      if (active.current !== id) return;
      setMessages(detail.messages);
      setPending(detail.busy ? [] : null);
    } catch (e) {
      if (e instanceof ApiError && e.status === 404) {
        setActiveId(null);
        setMessages([]);
        window.location.hash = "";
      } else {
        setError((e as Error).message);
      }
    }
  }, []);

  const newChat = useCallback(async () => {
    const chat = await api.createChat();
    await refreshChats();
    await open(chat.id);
  }, [open, refreshChats]);

  // First load: check the server is safe to use, then pick or create a chat.
  useEffect(() => {
    (async () => {
      try {
        const health = await api.health();
        if (!health.ok) return setProblems(health.problems);
        const list = await api.listChats();
        setChats(list);
        const wanted = hashId();
        if (wanted && list.some((c) => c.id === wanted)) await open(wanted);
        else if (list.length > 0) await open(list[0].id);
        else await newChat();
      } catch (e) {
        setError(`Cannot reach the DataPilot server: ${(e as Error).message}`);
      }
    })();
  }, [open, newChat]);

  // A reply that was still being written when the page loaded: poll until it is saved.
  useEffect(() => {
    if (!activeId || pending === null) return;
    const id = activeId;
    const timer = setInterval(async () => {
      try {
        const detail = await api.getChat(id);
        if (active.current === id && !detail.busy) {
          setMessages(detail.messages);
          setPending(null);
          refreshChats();
        }
      } catch {
        // keep polling
      }
    }, 2000);
    return () => clearInterval(timer);
  }, [activeId, pending, refreshChats]);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, pending]);

  const send = async (text: string) => {
    if (!activeId || pending !== null) return;
    const id = activeId;
    setError(null);
    setMessages((m) => [...m, { id: -Date.now(), role: "user", content: text, payload: null }]);
    setPending([]);
    const onEvent = (ev: StreamEvent) => {
      if (active.current !== id) return;
      if (ev.type === "step") setPending((s) => [...(s ?? []), { label: ev.label, node: ev.node }]);
      if (ev.type === "step_error")
        setPending((s) => (s && s.length ? [...s.slice(0, -1), { ...s[s.length - 1], error: ev.detail }] : s));
    };
    try {
      const result = await sendMessage(id, text, onEvent);
      if (active.current === id) {
        if (result) setMessages((m) => [...m, result]);
        else setMessages((await api.getChat(id)).messages); // connection dropped; server saved it
        setPending(null);
      }
    } catch (e) {
      if (active.current === id) {
        setPending(null);
        setError((e as Error).message);
        setMessages((m) => m.filter((x) => x.id >= 0));
      }
    } finally {
      refreshChats();
    }
  };

  const remove = async (id: string) => {
    try {
      await api.deleteChat(id);
    } catch (e) {
      return setError((e as Error).message);
    }
    const rest = chats.filter((c) => c.id !== id);
    setChats(rest);
    if (id === activeId) {
      if (rest.length) await open(rest[0].id);
      else await newChat();
    }
  };

  const rename = async (id: string, title: string) => {
    await api.renameChat(id, title);
    refreshChats();
  };

  if (problems) {
    return (
      <div className="blocked">
        <h1>DataPilot can’t start safely</h1>
        <p>The database roles it is configured with can do more than the agent should be allowed to:</p>
        <ul>
          {problems.slice(0, 8).map((p, i) => (
            <li key={i}>{p}</li>
          ))}
        </ul>
        <p>
          Run <code>datapilot db setup</code> to create correctly limited roles, then restart.
        </p>
      </div>
    );
  }

  const busy = pending !== null;
  return (
    <div className="app">
      <Sidebar
        chats={chats}
        activeId={activeId}
        open={sidebar}
        onSelect={open}
        onNew={newChat}
        onDelete={remove}
        onRename={rename}
        onClose={() => setSidebar(false)}
      />
      <main>
        <header>
          <button className="icon menu" aria-label="Conversations" onClick={() => setSidebar(true)}>
            ☰
          </button>
          <span className="title">{chats.find((c) => c.id === activeId)?.title ?? "DataPilot"}</span>
        </header>
        <div className="scroll">
          <div className="thread">
            {messages.length === 0 && !busy && (
              <div className="empty">
                <h1>What would you like to know?</h1>
                <p className="muted">Ask a question about your data, or ask for outside data to be loaded.</p>
                <div className="starters">
                  {STARTERS.map((s) => (
                    <button key={s} onClick={() => send(s)} disabled={!activeId}>
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            )}
            {messages.map((m) =>
              m.role === "user" ? (
                <div className="msg user" key={m.id}>
                  <div className="bubble">{m.content}</div>
                </div>
              ) : (
                <AssistantMessage key={m.id} message={m} />
              ),
            )}
            {pending !== null && <PendingMessage steps={pending} />}
            {error && (
              <div className="error" role="alert">
                {error}
              </div>
            )}
            <div ref={bottom} />
          </div>
        </div>
        <Composer disabled={busy || !activeId} onSend={send} focusKey={activeId ?? ""} />
      </main>
    </div>
  );
}
