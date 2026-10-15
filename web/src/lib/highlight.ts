// Split page text into plain and highlighted segments for evidence spans.

export interface Span {
  start: number;
  end: number;
  key: string; // identifies the field the span belongs to (e.g. "cs1.wb_sar_max_w_per_kg")
}

export interface Segment {
  text: string;
  start: number;
  keys: string[]; // empty = plain text
}

/**
 * Cut `text` at every span boundary; each segment lists the keys of all spans covering it.
 * Overlapping and nested spans are supported; out-of-range offsets are clamped.
 */
export function segmentText(text: string, spans: Span[]): Segment[] {
  const n = text.length;
  const valid = spans
    .map((s) => ({ ...s, start: Math.max(0, Math.min(n, s.start)), end: Math.max(0, Math.min(n, s.end)) }))
    .filter((s) => s.end > s.start);
  const cuts = new Set<number>([0, n]);
  for (const s of valid) {
    cuts.add(s.start);
    cuts.add(s.end);
  }
  const points = [...cuts].sort((a, b) => a - b);
  const out: Segment[] = [];
  for (let i = 0; i < points.length - 1; i++) {
    const a = points[i]!;
    const b = points[i + 1]!;
    if (b <= a) continue;
    const keys = valid.filter((s) => s.start <= a && s.end >= b).map((s) => s.key);
    const prev = out[out.length - 1];
    if (prev && prev.keys.join("|") === keys.join("|")) {
      prev.text += text.slice(a, b);
    } else {
      out.push({ text: text.slice(a, b), start: a, keys });
    }
  }
  return out;
}
