import { Suspense, lazy, useState } from "react";
import type { ChatMessage, Step } from "../types";
import ResultTable from "./ResultTable";
import { DoneSteps, LiveSteps } from "./Steps";

const ResultChart = lazy(() => import("./ResultChart")); // keeps the charting library out of the first load

const ROUTE_LABEL: Record<string, string> = {
  sql: "SQL analyst",
  etl: "ETL analyst",
  both: "ETL + SQL",
  chat: "Chat",
  clarify: "Question",
  none: "Error",
};

export function PendingMessage({ steps }: { steps: Step[] }) {
  return (
    <div className="msg assistant">
      <div className="avatar">D</div>
      <div className="body">
        <LiveSteps steps={steps} />
      </div>
    </div>
  );
}

export default function AssistantMessage({ message }: { message: ChatMessage }) {
  const p = message.payload;
  const [copied, setCopied] = useState(false);
  if (!p) {
    return (
      <div className="msg assistant">
        <div className="avatar">D</div>
        <div className="body">
          <div className="bubble">{message.content}</div>
        </div>
      </div>
    );
  }
  const copySql = async () => {
    try {
      await navigator.clipboard.writeText(p.sql ?? "");
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // clipboard unavailable (insecure context); ignore
    }
  };
  return (
    <div className="msg assistant">
      <div className="avatar">D</div>
      <div className="body">
        <div className="meta">
          <span className="tag">{ROUTE_LABEL[p.route] ?? p.route}</span>
          {p.status !== "ok" && <span className={`tag ${p.status}`}>{p.status}</span>}
          <span className="muted small">{p.latency_s.toFixed(1)}s</span>
        </div>
        <div className={`bubble ${p.status}`}>{p.text}</div>
        {p.loads.map((l) => (
          <div className="loaded" key={l.table}>
            ✓ Loaded <code>staging.{l.table}</code> · {l.rows.toLocaleString()} rows
          </div>
        ))}
        {p.assumptions.length > 0 && (
          <div className="muted small">Assumed: {p.assumptions.join("; ")}</div>
        )}
        {p.chart && (
          <Suspense fallback={<div className="card chart muted small pad">Loading chart…</div>}>
            <ResultChart spec={p.chart} rows={p.rows} />
          </Suspense>
        )}
        {p.columns.length > 0 && p.rows.length > 0 && (
          <ResultTable columns={p.columns} rows={p.rows} rowCount={p.row_count} truncated={p.truncated} />
        )}
        {p.sql && (
          <details className="details">
            <summary>SQL</summary>
            <div className="code-wrap">
              <button className="link copy" onClick={copySql}>
                {copied ? "Copied" : "Copy"}
              </button>
              <pre>{p.sql}</pre>
            </div>
          </details>
        )}
        <DoneSteps steps={p.steps} />
      </div>
    </div>
  );
}
