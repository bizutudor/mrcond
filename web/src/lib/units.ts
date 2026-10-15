// Display helpers for MR Conditional quantities. All conversions mirror src/mrcond/units.py.
import type { FieldName, FieldValue, Normalized } from "../types";

/** 1 T/m = 100 G/cm */
export function teslaPerMeterToGaussPerCm(tpm: number): number {
  return roundSig(tpm * 100);
}

/**
 * 100 G/cm = 1 T/m. Exact division without rounding: this converts *user input* that is compared with a
 * labeled limit, and rounding could move a value that is just above the limit onto it.
 */
export function gaussPerCmToTeslaPerMeter(gcm: number): number {
  return gcm / 100;
}

/** mT/m to T/m (gradient units). */
export function milliTeslaPerMeterToTeslaPerMeter(mtpm: number): number {
  return roundSig(mtpm / 1000);
}

export function secondsToMinutes(s: number): number {
  return roundSig(s / 60);
}

/** Round to `sig` significant digits to remove floating-point noise (e.g. 0.1 * 3). */
export function roundSig(x: number, sig = 6): number {
  if (x === 0 || !Number.isFinite(x)) return x;
  const d = Math.ceil(Math.log10(Math.abs(x)));
  const power = sig - d;
  const magnitude = Math.pow(10, power);
  return Math.round(x * magnitude) / magnitude;
}

/** Compact number formatting without trailing zeros: 3.0 -> "3", 1.70 -> "1.7", 3000 -> "3,000". */
export function fmtNumber(x: number): string {
  const r = roundSig(x);
  if (Math.abs(r) >= 1000 && Number.isInteger(r)) return r.toLocaleString("en-US");
  return String(r);
}

const UNIT_DISPLAY: Record<string, string> = {
  uT: "µT",
  "T/m": "T/m",
  "T/m/s": "T/m/s",
  "W/kg": "W/kg",
  min: "min",
  mm: "mm",
  T: "T",
};

export function displayUnit(unit: string | null | undefined): string {
  if (!unit) return "";
  return UNIT_DISPLAY[unit] ?? unit;
}

const COMPARATOR_SYMBOL: Record<string, string> = { le: "≤ ", ge: "≥ ", eq: "", in: "" };

function asNumbers(v: Normalized): number[] {
  if (v === null || v === undefined) return [];
  if (Array.isArray(v)) return v.filter((x): x is number => typeof x === "number");
  return typeof v === "number" ? [v] : [];
}

/** Field strength display: [1.5, 3] -> "1.5 T or 3 T"; comparator le -> "≤ 3 T". */
export function formatB0(fv: Pick<FieldValue, "normalized_value" | "comparator">): string {
  const vals = asNumbers(fv.normalized_value);
  if (vals.length === 0) return "—";
  if (fv.comparator === "le") return `≤ ${fmtNumber(Math.max(...vals))} T`;
  return vals.map((v) => `${fmtNumber(v)} T`).join(" or ");
}

const TOKEN_LABELS: Record<string, string> = {
  any: "any",
  body: "body coil",
  head: "head coil",
  extremity: "extremity coil",
  local: "local coil",
  CP: "circularly polarized (CP)",
  "MC-2": "multichannel (MC-2)",
  normal: "Normal Operating Mode",
  first_level: "up to First Level Controlled Operating Mode",
  cylindrical: "cylindrical (closed bore)",
  horizontal: "horizontal",
  vertical: "vertical",
  unrestricted: "no restriction (any landmark)",
};

export function tokenLabel(token: string): string {
  return TOKEN_LABELS[token] ?? token;
}

/**
 * Human-readable value of a labeled condition, e.g. "≤ 25 T/m (2,500 G/cm)", "≤ 1.7 µT",
 * "1.5 T or 3 T", "body coil", "no limit stated".
 */
export function formatFieldValue(field: FieldName | "mr_status", fv: FieldValue): string {
  if (fv.status === "NOT_STATED" && fv.value === null && fv.normalized_value === null) return "not stated";
  if (field === "mr_status") return String(fv.normalized_value ?? fv.value ?? "—");
  if (field === "b0_tesla") return formatB0(fv);
  if (fv.comparator === "any") return "no limit (explicitly unrestricted)";
  const nv = fv.normalized_value;
  if (typeof nv === "number") {
    const sym = COMPARATOR_SYMBOL[fv.comparator ?? "eq"] ?? "";
    const base = `${sym}${fmtNumber(nv)} ${displayUnit(fv.normalized_unit)}`.trim();
    if (field === "max_sfg_t_per_m") return `${base} (${fmtNumber(teslaPerMeterToGaussPerCm(nv))} G/cm)`;
    return base;
  }
  if (typeof nv === "boolean") {
    return field === "sar_is_b1rms_fallback" ? (nv ? "yes (SAR limit applies only without B1+rms)" : "no") : String(nv);
  }
  if (Array.isArray(nv)) return nv.map((t) => (typeof t === "number" ? fmtNumber(t) : tokenLabel(t))).join(", ");
  if (typeof nv === "string") return tokenLabel(nv);
  if (fv.value !== null && fv.value !== undefined) return String(fv.value);
  return "—";
}

/**
 * Parse a user-entered decimal number; empty or invalid input becomes null. Accepts "2.8", "1,5" (decimal
 * comma) and "3,000" (thousands separator); rejects hexadecimal, exponents and anything else.
 */
export function parseNumberInput(raw: string): number | null {
  const s = raw.trim();
  if (!s) return null;
  let normalized: string;
  if (/^-?\d{1,3}(,\d{3})+(\.\d+)?$/.test(s)) normalized = s.replace(/,/g, "");
  else if (/^-?\d+,\d+$/.test(s)) normalized = s.replace(",", ".");
  else if (/^-?(\d+(\.\d*)?|\.\d+)$/.test(s)) normalized = s;
  else return null;
  const n = Number(normalized);
  return Number.isFinite(n) ? n : null;
}

export interface NumberRule {
  required?: boolean;
  integer?: boolean;
  /** Smallest allowed value (default 0: quantities entered for the checker are never negative). */
  min?: number;
  minExclusive?: boolean;
  max?: number;
}

/** Why a user-entered number cannot be used, or null if it can. Empty input means "not entered". */
export function numberInputError(raw: string, rule: NumberRule = {}): string | null {
  const s = raw.trim();
  if (!s) return rule.required ? "Required" : null;
  const n = parseNumberInput(s);
  if (n === null) return "Not a number";
  const min = rule.min ?? 0;
  if (rule.minExclusive ? n <= min : n < min) return rule.minExclusive ? `Must be above ${min}` : `Must be at least ${min}`;
  if (rule.max !== undefined && n > rule.max) return `Must be at most ${fmtNumber(rule.max)}`;
  if (rule.integer && !Number.isInteger(n)) return "Must be a whole number";
  return null;
}
