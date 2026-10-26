import type { Step } from "../types";

export function LiveSteps({ steps }: { steps: Step[] }) {
  return (
    <ol className="steps live" aria-live="polite">
      {steps.map((s, i) => {
        const active = i === steps.length - 1;
        return (
          <li key={i} className={active ? "active" : "done"}>
            <span className="dot">{active ? <span className="spinner" /> : "✓"}</span>
            <span>{s.label}</span>
            {s.error && <span className="step-error">{s.error}</span>}
          </li>
        );
      })}
      {steps.length === 0 && (
        <li className="active">
          <span className="dot">
            <span className="spinner" />
          </span>
          <span>Thinking…</span>
        </li>
      )}
    </ol>
  );
}

export function DoneSteps({ steps }: { steps: Step[] }) {
  if (steps.length === 0) return null;
  return (
    <details className="details">
      <summary>Steps ({steps.length})</summary>
      <ol className="steps">
        {steps.map((s, i) => (
          <li key={i} className="done">
            <span className="dot">{s.error ? "!" : "✓"}</span>
            <span>{s.label}</span>
            {s.error && <span className="step-error">{s.error}</span>}
          </li>
        ))}
      </ol>
    </details>
  );
}
