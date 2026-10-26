export type Status = "ok" | "refused" | "failed";

export interface ChartSpec {
  type: "bar" | "line";
  x: string;
  y: string[];
}

export interface Step {
  label: string;
  node: string;
  error?: string;
}

export interface Payload {
  request_id: string;
  route: string;
  status: Status;
  text: string;
  sql: string | null;
  assumptions: string[];
  columns: string[];
  rows: Record<string, unknown>[];
  row_count: number;
  truncated: boolean;
  chart: ChartSpec | null;
  loads: { table: string; rows: number; columns: string[] }[];
  llm_calls: number;
  latency_s: number;
  steps: Step[];
}

export interface ChatMessage {
  id: number;
  role: "user" | "assistant";
  content: string;
  payload: Payload | null;
}

export interface ChatSummary {
  id: string;
  title: string;
  updated_at: string;
}

export interface ChatDetail extends ChatSummary {
  messages: ChatMessage[];
  busy: boolean;
}

export type StreamEvent =
  | { type: "start"; user_message_id: number }
  | { type: "step"; label: string; node: string }
  | { type: "step_error"; node: string; detail: string }
  | { type: "result"; message: ChatMessage };
