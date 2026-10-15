import { describe, expect, it } from "vitest";
import { segmentText } from "./highlight";

describe("evidence highlighting", () => {
  it("returns plain text without spans", () => {
    expect(segmentText("abc", [])).toEqual([{ text: "abc", start: 0, keys: [] }]);
  });

  it("splits around a span", () => {
    const segs = segmentText("Whole-Body SAR ≤ 2 W/kg here", [{ start: 0, end: 23, key: "cs1.wb" }]);
    expect(segs.map((s) => [s.text, s.keys])).toEqual([
      ["Whole-Body SAR ≤ 2 W/kg", ["cs1.wb"]],
      [" here", []],
    ]);
  });

  it("handles overlapping spans", () => {
    const segs = segmentText("0123456789", [
      { start: 2, end: 6, key: "a" },
      { start: 4, end: 8, key: "b" },
    ]);
    expect(segs.map((s) => [s.text, s.keys])).toEqual([
      ["01", []],
      ["23", ["a"]],
      ["45", ["a", "b"]],
      ["67", ["b"]],
      ["89", []],
    ]);
  });

  it("clamps out-of-range offsets and drops empty spans", () => {
    const segs = segmentText("abc", [
      { start: 1, end: 99, key: "a" },
      { start: 2, end: 2, key: "empty" },
    ]);
    expect(segs.map((s) => [s.text, s.keys])).toEqual([
      ["a", []],
      ["bc", ["a"]],
    ]);
  });
});
