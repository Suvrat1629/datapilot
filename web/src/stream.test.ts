import { describe, expect, it } from "vitest";
import { chartData, shortLabel } from "./chart";
import { toCsv } from "./csv";
import { SseParser } from "./stream";

describe("SseParser", () => {
  it("emits events split across chunks and ignores keepalives", () => {
    const p = new SseParser();
    const a = p.push('data: {"type":"start","user_message_id":1}\n\n: keepalive\n\ndata: {"type":"st');
    expect(a).toEqual([{ type: "start", user_message_id: 1 }]);
    const b = p.push('ep","label":"Writing SQL","node":"generate_sql"}\n\n');
    expect(b).toEqual([{ type: "step", label: "Writing SQL", node: "generate_sql" }]);
  });

  it("handles CRLF and survives a malformed block", () => {
    const p = new SseParser();
    const out = p.push('data: not json\r\n\r\ndata: {"type":"step_error","node":"t","detail":"x"}\r\n\r\n');
    expect(out).toEqual([{ type: "step_error", node: "t", detail: "x" }]);
  });

  it("keeps an incomplete event buffered", () => {
    const p = new SseParser();
    expect(p.push('data: {"type":"start"')).toEqual([]);
    expect(p.push(',"user_message_id":2}\n\n')).toEqual([{ type: "start", user_message_id: 2 }]);
  });
});

describe("toCsv", () => {
  it("quotes, escapes and neutralises formulas", () => {
    const csv = toCsv(["a", "b"], [{ a: 'x,"y"', b: "=HYPERLINK(1)" }, { a: null, b: -5 }, { a: "-cmd", b: "2" }]);
    expect(csv).toBe(`a,b\n"x,""y""",'=HYPERLINK(1)\n,-5\n'-cmd,2`);
  });
});

describe("chart helpers", () => {
  it("coerces numeric strings and nulls", () => {
    const data = chartData({ type: "bar", x: "c", y: ["n"] }, [{ c: "A", n: "3.5" }, { c: "B", n: null }, { c: "C", n: 2 }]);
    expect(data).toEqual([{ c: "A", n: 3.5 }, { c: "B", n: null }, { c: "C", n: 2 }]);
  });

  it("shortens dates and long labels", () => {
    expect(shortLabel("2026-10-05")).toBe("10-05");
    expect(shortLabel("a very long city name indeed")).toHaveLength(14);
  });
});
