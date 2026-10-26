import type { StreamEvent } from "./types";

/** Incremental parser for a text/event-stream body: feed chunks, get complete events. */
export class SseParser {
  private buffer = "";

  push(chunk: string): StreamEvent[] {
    this.buffer += chunk;
    const events: StreamEvent[] = [];
    let split: number;
    while ((split = this.buffer.search(/\r?\n\r?\n/)) !== -1) {
      const block = this.buffer.slice(0, split);
      this.buffer = this.buffer.slice(split).replace(/^\r?\n\r?\n/, "");
      const data = block
        .split(/\r?\n/)
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).replace(/^ /, ""))
        .join("\n");
      if (!data) continue; // keepalive comment
      try {
        events.push(JSON.parse(data) as StreamEvent);
      } catch {
        // ignore a malformed block rather than killing the stream
      }
    }
    return events;
  }
}
