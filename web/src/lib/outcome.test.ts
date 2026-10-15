import { describe, expect, it } from "vitest";
import type { ImplantResult } from "../types";
import {
  aggregateOutcome,
  byConcern,
  claimsSafety,
  OUTCOME_TEXT,
  outcomeTone,
  requiresSignOff,
  selectedChecks,
  worstStatus,
} from "./outcome";

describe("outcome aggregation", () => {
  it("takes the most conservative status", () => {
    expect(worstStatus(["PASS", "PASS"])).toBe("PASS");
    expect(worstStatus(["PASS", "INSUFFICIENT_INFO"])).toBe("INSUFFICIENT_INFO");
    expect(worstStatus(["INSUFFICIENT_INFO", "NEEDS_REVIEW"])).toBe("NEEDS_REVIEW");
    expect(worstStatus(["NEEDS_REVIEW", "FAIL", "PASS"])).toBe("FAIL");
  });

  it("requires every implant to pass", () => {
    expect(aggregateOutcome(["PASS", "PASS"])).toBe("NO_CONFLICT_FOUND");
    expect(aggregateOutcome(["PASS", "FAIL"])).toBe("CONFLICT_FOUND");
    expect(aggregateOutcome(["PASS", "NEEDS_REVIEW"])).toBe("NEEDS_REVIEW");
    expect(aggregateOutcome(["INSUFFICIENT_INFO", "FAIL"])).toBe("CONFLICT_FOUND");
  });

  it("never reports no conflict without implants", () => {
    expect(aggregateOutcome([])).toBe("NEEDS_REVIEW");
  });
});

describe("outcome display", () => {
  it("never claims that a scan is safe", () => {
    for (const text of Object.values(OUTCOME_TEXT)) expect(claimsSafety(text)).toBe(false);
    expect(OUTCOME_TEXT.NO_CONFLICT_FOUND).toContain("sign-off");
    expect(claimsSafety("The patient is safe to scan")).toBe(true);
  });

  it("maps outcomes to tones and sign-off", () => {
    expect(outcomeTone("CONFLICT_FOUND")).toBe("conflict");
    expect(outcomeTone("NEEDS_REVIEW")).toBe("review");
    expect(outcomeTone("NO_CONFLICT_FOUND")).toBe("neutral");
    expect(requiresSignOff("NO_CONFLICT_FOUND")).toBe(true);
    expect(requiresSignOff("CONFLICT_FOUND")).toBe(false);
  });

  it("orders conflicts first", () => {
    const sorted = byConcern([{ status: "PASS" }, { status: "FAIL" }, { status: "NEEDS_REVIEW" }] as const);
    expect(sorted.map((s) => s.status)).toEqual(["FAIL", "NEEDS_REVIEW", "PASS"]);
  });

  it("collects implant-level and selected-set checks", () => {
    const check = (field: string, status: "PASS" | "FAIL" | "NEEDS_REVIEW") => ({
      field,
      label: field,
      status,
      reason: "",
      label_value: null,
      entered_value: null,
      evidence: [],
    });
    const implant: ImplantResult = {
      implant_id: "x",
      device_name: null,
      mr_status: "MR Conditional",
      verification: "draft-unverified",
      status: "FAIL",
      selected_set_id: "cs2",
      implant_checks: [check("verification", "NEEDS_REVIEW")],
      set_results: [
        { set_id: "cs1", scope: null, applicable: false, status: "FAIL", checks: [check("b0_tesla", "FAIL")], notes: [] },
        {
          set_id: "cs2",
          scope: null,
          applicable: true,
          status: "FAIL",
          checks: [check("b0_tesla", "PASS"), check("wb_sar_max_w_per_kg", "FAIL")],
          notes: [],
        },
      ],
    };
    expect(selectedChecks(implant).map((c) => c.field)).toEqual(["wb_sar_max_w_per_kg", "verification", "b0_tesla"]);
  });
});
