import type { ChartSpec } from "./types";

/** Recharts wants numbers as numbers; DB numerics can arrive as strings in odd cases. */
export function chartData(spec: ChartSpec, rows: Record<string, unknown>[]) {
  return rows.map((r) => {
    const point: Record<string, string | number | null> = { [spec.x]: String(r[spec.x] ?? "") };
    for (const y of spec.y) {
      const v = r[y];
      point[y] = typeof v === "number" ? v : v === null || v === undefined ? null : Number(v);
    }
    return point;
  });
}

export function shortLabel(value: string, max = 14): string {
  const v = value.length >= 10 && /^\d{4}-\d{2}-\d{2}/.test(value) ? value.slice(5, 10) : value;
  return v.length > max ? `${v.slice(0, max - 1)}…` : v;
}
