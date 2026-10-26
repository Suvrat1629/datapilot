import { SseParser } from "./stream";
import type { ChatDetail, ChatMessage, ChatSummary, StreamEvent } from "./types";

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public problems: string[] = [],
  ) {
    super(message);
  }
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    let message = `Request failed (${res.status})`;
    let problems: string[] = [];
    try {
      const body = await res.json();
      const detail = body?.detail;
      if (typeof detail === "string") message = detail;
      else if (detail?.message) {
        message = detail.message;
        problems = detail.problems ?? [];
      }
    } catch {
      // keep the generic message
    }
    throw new ApiError(message, res.status, problems);
  }
  return (await res.json()) as T;
}

const json = (method: string, body?: unknown): RequestInit => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: body === undefined ? undefined : JSON.stringify(body),
});

export const api = {
  health: () => request<{ ok: boolean; problems: string[] }>("/api/health"),
  listChats: () => request<ChatSummary[]>("/api/chats"),
  createChat: () => request<ChatSummary>("/api/chats", json("POST")),
  getChat: (id: string) => request<ChatDetail>(`/api/chats/${id}`),
  renameChat: (id: string, title: string) => request<{ ok: boolean }>(`/api/chats/${id}`, json("PATCH", { title })),
  deleteChat: (id: string) => request<{ ok: boolean }>(`/api/chats/${id}`, json("DELETE")),
};

/** Send a message and report stream events. Resolves with the final message, or null if the
 * connection ended before a result arrived (the server still finishes and saves the turn). */
export async function sendMessage(
  chatId: string,
  text: string,
  onEvent: (event: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<ChatMessage | null> {
  const res = await fetch(`/api/chats/${chatId}/messages`, { ...json("POST", { text }), signal });
  if (!res.ok || !res.body) {
    let message = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (typeof body?.detail === "string") message = body.detail;
      else if (body?.detail?.message) message = body.detail.message;
    } catch {
      // keep the generic message
    }
    throw new ApiError(message, res.status);
  }
  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  const parser = new SseParser();
  let result: ChatMessage | null = null;
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      for (const event of parser.push(value)) {
        onEvent(event);
        if (event.type === "result") result = event.message;
      }
    }
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    // network drop: fall through and let the caller re-load the saved chat
  }
  return result;
}
