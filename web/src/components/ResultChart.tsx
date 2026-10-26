import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { chartData, shortLabel } from "../chart";
import type { ChartSpec } from "../types";

const COLORS = ["var(--c1)", "var(--c2)", "var(--c3)"];

export default function ResultChart({ spec, rows }: { spec: ChartSpec; rows: Record<string, unknown>[] }) {
  const data = chartData(spec, rows);
  const axis = { stroke: "var(--muted)", fontSize: 12 };
  const common = { data, margin: { top: 8, right: 12, bottom: 4, left: 0 } };
  const many = data.length > 8;
  return (
    <div className="card chart">
      <ResponsiveContainer width="100%" height={260}>
        {spec.type === "bar" ? (
          <BarChart {...common}>
            <CartesianGrid stroke="var(--border)" vertical={false} />
            <XAxis dataKey={spec.x} tickFormatter={(v) => shortLabel(String(v))} {...axis} interval={0} angle={many ? -35 : 0} textAnchor={many ? "end" : "middle"} height={many ? 70 : 30} />
            <YAxis {...axis} width={56} />
            <Tooltip contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8 }} />
            {spec.y.length > 1 && <Legend />}
            {spec.y.map((y, i) => (
              <Bar key={y} dataKey={y} fill={COLORS[i % COLORS.length]} radius={[4, 4, 0, 0]} />
            ))}
          </BarChart>
        ) : (
          <LineChart {...common}>
            <CartesianGrid stroke="var(--border)" vertical={false} />
            <XAxis dataKey={spec.x} tickFormatter={(v) => shortLabel(String(v))} {...axis} minTickGap={24} />
            <YAxis {...axis} width={56} domain={["auto", "auto"]} />
            <Tooltip contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8 }} />
            {spec.y.length > 1 && <Legend />}
            {spec.y.map((y, i) => (
              <Line key={y} type="monotone" dataKey={y} stroke={COLORS[i % COLORS.length]} strokeWidth={2} dot={data.length < 40} />
            ))}
          </LineChart>
        )}
      </ResponsiveContainer>
    </div>
  );
}
