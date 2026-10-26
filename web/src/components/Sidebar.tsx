import { useState } from "react";
import type { ChatSummary } from "../types";

interface Props {
  chats: ChatSummary[];
  activeId: string | null;
  open: boolean;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
  onRename: (id: string, title: string) => void;
  onClose: () => void;
}

export default function Sidebar({ chats, activeId, open, onSelect, onNew, onDelete, onRename, onClose }: Props) {
  const [editing, setEditing] = useState<string | null>(null);
  const [draft, setDraft] = useState("");

  const commit = () => {
    if (editing && draft.trim()) onRename(editing, draft.trim());
    setEditing(null);
  };

  return (
    <>
      {open && <div className="scrim" onClick={onClose} />}
      <aside className={`sidebar ${open ? "open" : ""}`}>
        <div className="brand">
          <span className="logo">D</span> DataPilot
        </div>
        <button className="new" onClick={onNew}>
          + New chat
        </button>
        <nav aria-label="Conversations">
          {chats.map((c) => (
            <div key={c.id} className={`chat-row ${c.id === activeId ? "active" : ""}`}>
              {editing === c.id ? (
                <input
                  autoFocus
                  value={draft}
                  maxLength={60}
                  onChange={(e) => setDraft(e.target.value)}
                  onBlur={commit}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") commit();
                    if (e.key === "Escape") setEditing(null);
                  }}
                />
              ) : (
                <button className="chat-title" onClick={() => onSelect(c.id)} onDoubleClick={() => { setEditing(c.id); setDraft(c.title); }} title={c.title}>
                  {c.title}
                </button>
              )}
              <button className="icon" aria-label={`Delete ${c.title}`} onClick={() => confirm("Delete this conversation?") && onDelete(c.id)}>
                ×
              </button>
            </div>
          ))}
          {chats.length === 0 && <div className="muted small pad">No conversations yet.</div>}
        </nav>
      </aside>
    </>
  );
}
