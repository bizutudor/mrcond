// TypeScript mirror of the MRCond pydantic models (src/mrcond/schema.py).

export type FieldStatus = "EXTRACTED" | "NOT_STATED" | "NEEDS_REVIEW";
export type Comparator = "le" | "ge" | "eq" | "in" | "any";
export type Normalized = boolean | number | number[] | string[] | string | null;

export interface Evidence {
  doc_id: string;
  page: number;
  char_start: number;
  char_end: number;
  quote: string;
  verified?: boolean | null;
}

export interface FieldValue {
  status: FieldStatus;
  value: Normalized;
  unit: string | null;
  normalized_value: Normalized;
  normalized_unit: string | null;
  comparator: Comparator | null;
  evidence: Evidence[];
  review_reasons: string[];
}

export const FIELD_NAMES = [
  "b0_tesla",
  "nucleus",
  "scanner_type",
  "b0_orientation",
  "max_sfg_t_per_m",
  "max_slew_rate_t_per_m_per_s",
  "rf_excitation",
  "transmit_coil",
  "receive_coil",
  "operating_mode",
  "b1rms_max_uT",
  "wb_sar_max_w_per_kg",
  "head_sar_max_w_per_kg",
  "sar_is_b1rms_fallback",
  "max_active_scan_time_min",
  "wait_time_min",
  "scan_region_restrictions",
  "device_configuration",
  "artifact_extent_mm",
  "other_conditions",
] as const;

export type FieldName = (typeof FIELD_NAMES)[number];

export type ConditionSet = { set_id: string; scope: string | null } & Record<FieldName, FieldValue>;

export type Verification = "machine-extracted" | "machine-verified" | "draft-unverified" | "human-verified";

export interface Provenance {
  extractor: string;
  verification: Verification;
  annotator: string | null;
  reviewer: string | null;
  reviewed_at: string | null;
  created_at: string | null;
  text_extractor: string | null;
  notes: string[];
}

export interface LabelDocument {
  schema_version: string;
  doc_id: string;
  device_name: string | null;
  manufacturer: string | null;
  device_type: string | null;
  mr_status: FieldValue;
  condition_sets: ConditionSet[];
  provenance: Provenance;
}

export interface FieldSpec {
  name: FieldName;
  label: string;
  kind: "b0" | "numeric" | "categorical" | "text" | "bool";
  canonical_unit: string | null;
  direction: "upper" | "lower" | "info" | null;
  evaluated: boolean;
  vocabulary: string[];
}

export interface DocumentSummary {
  doc_id: string;
  title: string;
  manufacturer: string | null;
  device_name: string | null;
  device_type: string | null;
  source: "track_r" | "synthetic";
  url: string | null;
  retrieved: string | null;
  text_available: boolean;
  redistributable: boolean;
  annotation_status: string;
  field_counts: { EXTRACTED: number; NEEDS_REVIEW: number };
}

export interface Page {
  number: number;
  text: string;
}

export interface DocumentPages {
  doc_id: string;
  text_extractor: string;
  pages: Page[];
}

export type DecisionAction = "accept" | "edit" | "reject";

export interface FieldDecision {
  set_id: string;
  field: FieldName | "mr_status";
  action: DecisionAction;
  value?: Normalized;
  unit?: string | null;
  normalized_value?: Normalized;
  comparator?: Comparator | null;
  quote?: string | null;
  page?: number | null;
  comment?: string | null;
}

export interface ScannerProfile {
  profile_id: string;
  name: string;
  b0_tesla: number;
  scanner_type: string | null;
  b0_orientation: string | null;
  max_slew_rate_t_per_m_per_s: number | null;
  transmit_coils: string[];
  rf_excitation_modes: string[];
  notes: string | null;
}

export interface SequencePlan {
  name: string;
  b1rms_uT: number | null;
  wb_sar_w_per_kg: number | null;
  head_sar_w_per_kg: number | null;
}

export interface ProtocolPlan {
  operating_mode: "normal" | "first_level" | null;
  transmit_coil: string | null;
  receive_coil: string | null;
  rf_excitation: string | null;
  nucleus: string | null;
  sequences: SequencePlan[];
  total_active_scan_time_min: number | null;
  sessions: number;
  wait_between_sessions_min: number | null;
  scan_region: string | null;
  slew_rate_limit_t_per_m_per_s: number | null;
  sfg_at_implant_t_per_m: number | null;
  device_configuration_confirmed: boolean | null;
}

export type CheckStatus = "PASS" | "FAIL" | "INSUFFICIENT_INFO" | "NEEDS_REVIEW";
export type OutcomeCode = "NO_CONFLICT_FOUND" | "CONFLICT_FOUND" | "NEEDS_REVIEW";

export interface ConditionCheck {
  field: string;
  label: string;
  status: CheckStatus;
  reason: string;
  label_value: string | null;
  entered_value: string | null;
  evidence: Evidence[];
}

export interface ConditionSetResult {
  set_id: string;
  scope: string | null;
  applicable: boolean;
  status: CheckStatus;
  checks: ConditionCheck[];
  notes: string[];
}

export interface ImplantResult {
  implant_id: string;
  device_name: string | null;
  mr_status: string;
  verification: string;
  status: CheckStatus;
  selected_set_id: string | null;
  implant_checks: ConditionCheck[];
  set_results: ConditionSetResult[];
}

export interface CheckResult {
  outcome: OutcomeCode;
  outcome_text: string;
  implants: ImplantResult[];
  disclaimer: string;
  generated_at: string | null;
}

export interface ImplantSummary {
  implant_id: string;
  device_name: string | null;
  manufacturer: string | null;
  mr_status: string;
  verification: string;
  source: string;
  n_condition_sets: number;
}
