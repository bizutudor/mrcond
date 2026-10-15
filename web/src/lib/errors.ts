// Readable messages for API errors (FastAPI returns a string or a list of pydantic validation errors).

interface ValidationIssue {
  loc?: unknown[];
  msg?: unknown;
}

/** "protocol.sequences.0.b1rms_uT: Input should be greater than or equal to 0" instead of raw JSON. */
export function formatErrorDetail(detail: unknown, fallback: string): string {
  if (typeof detail === "string" && detail) return detail;
  if (Array.isArray(detail) && detail.length > 0) {
    return detail
      .map((item: unknown) => {
        if (item && typeof item === "object" && "msg" in item) {
          const issue = item as ValidationIssue;
          const loc = Array.isArray(issue.loc) ? issue.loc.filter((x) => x !== "body").join(".") : "";
          return loc ? `${loc}: ${String(issue.msg)}` : String(issue.msg);
        }
        return JSON.stringify(item);
      })
      .join("; ");
  }
  if (detail) return JSON.stringify(detail);
  return fallback;
}
