import type { CheckStatus, FieldStatus } from "../types";
import { STATUS_LABEL } from "../lib/outcome";

type Kind = "ok" | "review" | "fail" | "muted";

const FIELD_STATUS: Record<FieldStatus, { kind: Kind; label: string }> = {
  EXTRACTED: { kind: "ok", label: "Extracted" },
  NEEDS_REVIEW: { kind: "review", label: "Needs review" },
  NOT_STATED: { kind: "muted", label: "Not stated" },
};

const CHECK_KIND: Record<CheckStatus, Kind> = {
  PASS: "ok",
  FAIL: "fail",
  INSUFFICIENT_INFO: "review",
  NEEDS_REVIEW: "review",
};

function Icon({ kind }: { kind: Kind }) {
  // Distinct shapes, so status never relies on colour alone.
  switch (kind) {
    case "ok":
      return <path d="M3.5 8.5 6.5 11.5 12.5 4.5" fill="none" stroke="currentColor" strokeWidth="2" />;
    case "review":
      return (
        <>
          <path d="M8 1.8 14.6 13.8H1.4Z" fill="none" stroke="currentColor" strokeWidth="1.6" />
          <rect x="7.2" y="5.6" width="1.6" height="4.2" fill="currentColor" />
          <rect x="7.2" y="10.8" width="1.6" height="1.6" fill="currentColor" />
        </>
      );
    case "fail":
      return <path d="M3.5 3.5 12.5 12.5M12.5 3.5 3.5 12.5" stroke="currentColor" strokeWidth="2" />;
    default:
      return <path d="M3.5 8h9" stroke="currentColor" strokeWidth="2" />;
  }
}

export function Badge({ kind, label }: { kind: Kind; label: string }) {
  return (
    <span className={`badge badge-${kind}`}>
      <svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true">
        <Icon kind={kind} />
      </svg>
      {label}
    </span>
  );
}

export function FieldStatusBadge({ status }: { status: FieldStatus }) {
  const s = FIELD_STATUS[status];
  return <Badge kind={s.kind} label={s.label} />;
}

export function CheckStatusBadge({ status }: { status: CheckStatus }) {
  return <Badge kind={CHECK_KIND[status]} label={STATUS_LABEL[status]} />;
}
