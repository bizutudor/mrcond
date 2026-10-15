// Outcome aggregation and display logic. Mirrors src/mrcond/checker.py.
// The UI never states that a scan is safe: the best possible outcome is
// "No conflict found with entered conditions — requires MR Safety Officer/Expert sign-off".
import type { CheckStatus, ImplantResult, OutcomeCode } from "../types";

export const OUTCOME_TEXT: Record<OutcomeCode, string> = {
  NO_CONFLICT_FOUND: "No conflict found with entered conditions — requires MR Safety Officer/Expert sign-off",
  CONFLICT_FOUND: "Conflict found",
  NEEDS_REVIEW: "Insufficient information / needs review",
};

export type Tone = "neutral" | "conflict" | "review";

const RANK: Record<CheckStatus, number> = { PASS: 0, INSUFFICIENT_INFO: 1, NEEDS_REVIEW: 1, FAIL: 2 };

/** Most conservative status of a list (FAIL > NEEDS_REVIEW > INSUFFICIENT_INFO > PASS). */
export function worstStatus(statuses: CheckStatus[]): CheckStatus {
  if (statuses.includes("FAIL")) return "FAIL";
  if (statuses.includes("NEEDS_REVIEW")) return "NEEDS_REVIEW";
  if (statuses.includes("INSUFFICIENT_INFO")) return "INSUFFICIENT_INFO";
  return "PASS";
}

/** Overall outcome over several implants: every implant must pass; the most conservative wins. */
export function aggregateOutcome(implantStatuses: CheckStatus[]): OutcomeCode {
  if (implantStatuses.length === 0) return "NEEDS_REVIEW";
  const worst = worstStatus(implantStatuses);
  if (worst === "FAIL") return "CONFLICT_FOUND";
  if (worst !== "PASS") return "NEEDS_REVIEW";
  return "NO_CONFLICT_FOUND";
}

export function outcomeTone(outcome: OutcomeCode): Tone {
  if (outcome === "CONFLICT_FOUND") return "conflict";
  if (outcome === "NEEDS_REVIEW") return "review";
  return "neutral";
}

/** Whether the sign-off block must be shown (always for the "no conflict" outcome). */
export function requiresSignOff(outcome: OutcomeCode): boolean {
  return outcome === "NO_CONFLICT_FOUND";
}

export const STATUS_LABEL: Record<CheckStatus, string> = {
  PASS: "No conflict",
  FAIL: "Conflict",
  INSUFFICIENT_INFO: "Missing input",
  NEEDS_REVIEW: "Needs review",
};

/** Sort checks so that conflicts come first, then review items, then passes. */
export function byConcern<T extends { status: CheckStatus }>(items: T[]): T[] {
  return [...items].sort((a, b) => RANK[b.status] - RANK[a.status]);
}

/** Defensive check used by tests and the UI: outcome texts never claim safety. */
export function claimsSafety(text: string): boolean {
  return /\b(safe to scan|is safe|are safe|scan is safe|cleared for scanning)\b/i.test(text);
}

export function selectedChecks(implant: ImplantResult) {
  const set = implant.set_results.find((s) => s.set_id === implant.selected_set_id);
  return byConcern([...implant.implant_checks, ...(set ? set.checks : [])]);
}
