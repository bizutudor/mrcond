import { useState } from "react";
import type { Comparator, FieldDecision, FieldName, FieldSpec, FieldValue, Normalized, Page } from "../types";
import { parseNumberInput, tokenLabel } from "../lib/units";

const SINGLE_CHOICE = new Set<FieldName>(["operating_mode", "scanner_type", "b0_orientation"]);
const MR_STATUSES = ["MR Conditional", "MR Safe", "MR Unsafe", "Not stated"];

interface Props {
  setId: string;
  field: FieldName | "mr_status";
  spec: FieldSpec | null;
  current: FieldValue;
  pages: Page[];
  onApply: (d: FieldDecision) => void;
  onCancel: () => void;
}

function initialTokens(v: Normalized): string[] {
  if (Array.isArray(v)) return v.map(String);
  if (typeof v === "string") return [v];
  return [];
}

export function FieldEditor({ setId, field, spec, current, pages, onApply, onCancel }: Props) {
  const kind = field === "mr_status" ? "status" : spec?.kind ?? "text";
  const nv = current.normalized_value;
  const [valueText, setValueText] = useState<string>(() => {
    if (kind === "b0") return Array.isArray(nv) ? nv.join(", ") : nv === null ? "" : String(nv);
    if (kind === "numeric") return current.value === null || current.value === undefined ? "" : String(current.value);
    if (kind === "text") return typeof nv === "string" && nv !== "unrestricted" ? nv : String(current.value ?? "");
    return "";
  });
  const [unit, setUnit] = useState<string>(current.unit ?? spec?.canonical_unit ?? "");
  const [comparator, setComparator] = useState<Comparator>(current.comparator ?? (kind === "b0" ? "in" : "le"));
  const [tokens, setTokens] = useState<string[]>(() => initialTokens(nv));
  const [flag, setFlag] = useState<boolean>(kind === "bool" ? Boolean(nv) : nv === "unrestricted");
  const [status, setStatus] = useState<string>(typeof nv === "string" ? nv : "MR Conditional");
  const [quote, setQuote] = useState<string>(current.evidence[0]?.quote ?? "");
  const [page, setPage] = useState<number | null>(current.evidence[0]?.page ?? pages[0]?.number ?? null);
  const [comment, setComment] = useState("");
  const [error, setError] = useState<string | null>(null);

  const vocabulary = spec?.vocabulary ?? [];

  function toggle(tok: string) {
    if (field !== "mr_status" && SINGLE_CHOICE.has(field)) {
      setTokens([tok]);
      return;
    }
    setTokens((t) => (t.includes(tok) ? t.filter((x) => x !== tok) : [...t, tok]));
  }

  function takeSelection() {
    const sel = window.getSelection()?.toString().replace(/\s+/g, " ").trim();
    if (sel) setQuote(sel);
  }

  function apply() {
    setError(null);
    if (!quote.trim()) {
      setError("Paste or select the exact sentence from the document that states this value.");
      return;
    }
    if (quote.trim().split(/\s+/).length > 25) {
      setError("Keep the quote to 25 words or fewer.");
      return;
    }
    const d: FieldDecision = { set_id: setId, field, action: "edit", quote: quote.trim(), page, comment: comment || null };
    if (kind === "status") {
      d.value = status;
      d.normalized_value = status;
    } else if (kind === "numeric") {
      if (comparator === "any") {
        d.value = "no limit";
        d.comparator = "any";
      } else {
        const n = parseNumberInput(valueText);
        if (n === null) {
          setError("Enter the number exactly as stated in the labeling.");
          return;
        }
        d.value = n;
        d.unit = unit || spec?.canonical_unit || null;
        d.comparator = comparator;
      }
    } else if (kind === "b0") {
      const vals = valueText
        .split(/[,;/ ]+/)
        .map(parseNumberInput)
        .filter((x): x is number => x !== null);
      if (vals.length === 0) {
        setError("Enter the permitted field strength(s) in tesla, e.g. 1.5, 3.");
        return;
      }
      d.value = vals;
      d.normalized_value = comparator === "le" ? [Math.max(...vals)] : vals;
      d.unit = "T";
      d.comparator = comparator === "le" ? "le" : "in";
    } else if (kind === "categorical") {
      if (tokens.length === 0) {
        setError("Choose at least one option.");
        return;
      }
      const single = field !== "mr_status" && SINGLE_CHOICE.has(field);
      d.normalized_value = single ? tokens[0]! : tokens;
      d.value = tokens.map(tokenLabel).join(", ");
      d.comparator = field === "operating_mode" ? "le" : single ? "eq" : "in";
    } else if (kind === "bool") {
      d.value = flag;
      d.normalized_value = flag;
    } else {
      if (!flag && !valueText.trim()) {
        setError("Describe the condition in a few words.");
        return;
      }
      d.value = flag ? "unrestricted" : valueText.trim();
      d.normalized_value = flag ? "unrestricted" : valueText.trim();
      d.comparator = "eq";
    }
    onApply(d);
  }

  return (
    <div className="editor" role="group" aria-label="Edit value">
      {kind === "status" && (
        <label>
          MR status
          <select value={status} onChange={(e) => setStatus(e.target.value)}>
            {MR_STATUSES.map((s) => (
              <option key={s}>{s}</option>
            ))}
          </select>
        </label>
      )}
      {kind === "numeric" && (
        <div className="editor-row">
          <label>
            Limit
            <select value={comparator} onChange={(e) => setComparator(e.target.value as Comparator)}>
              <option value="le">at most (≤)</option>
              <option value="ge">at least (≥)</option>
              <option value="eq">equals</option>
              <option value="any">no limit stated</option>
            </select>
          </label>
          {comparator !== "any" && (
            <>
              <label>
                Value as stated
                <input value={valueText} onChange={(e) => setValueText(e.target.value)} inputMode="decimal" />
              </label>
              <label>
                Unit as stated
                <input value={unit} onChange={(e) => setUnit(e.target.value)} placeholder={spec?.canonical_unit ?? ""} />
              </label>
            </>
          )}
        </div>
      )}
      {kind === "b0" && (
        <div className="editor-row">
          <label>
            Field strengths (T)
            <input value={valueText} onChange={(e) => setValueText(e.target.value)} placeholder="1.5, 3" />
          </label>
          <label className="check">
            <input
              type="checkbox"
              checked={comparator === "le"}
              onChange={(e) => setComparator(e.target.checked ? "le" : "in")}
            />
            stated as “or less”
          </label>
        </div>
      )}
      {kind === "categorical" && (
        <div className="chips" role="group" aria-label="Options">
          {vocabulary.map((tok) => (
            <button
              type="button"
              key={tok}
              className={tokens.includes(tok) ? "chip chip-on" : "chip"}
              aria-pressed={tokens.includes(tok)}
              onClick={() => toggle(tok)}
            >
              {tokenLabel(tok)}
            </button>
          ))}
        </div>
      )}
      {kind === "bool" && (
        <label className="check">
          <input type="checkbox" checked={flag} onChange={(e) => setFlag(e.target.checked)} />
          the SAR limit applies only when B1+rms is not available
        </label>
      )}
      {kind === "text" && (
        <>
          {field === "scan_region_restrictions" && (
            <label className="check">
              <input type="checkbox" checked={flag} onChange={(e) => setFlag(e.target.checked)} />
              labeling states no scan-region restriction
            </label>
          )}
          {!flag && (
            <label>
              Condition
              <input value={valueText} onChange={(e) => setValueText(e.target.value)} />
            </label>
          )}
        </>
      )}
      <label>
        Evidence quote (verbatim, at most 25 words)
        <textarea rows={2} value={quote} onChange={(e) => setQuote(e.target.value)} />
      </label>
      <div className="editor-row">
        <button type="button" className="btn-quiet" onClick={takeSelection}>
          Use text selected in the document
        </button>
        <label>
          Page
          <select value={page ?? ""} onChange={(e) => setPage(e.target.value ? Number(e.target.value) : null)}>
            {pages.map((p) => (
              <option key={p.number} value={p.number}>
                {p.number}
              </option>
            ))}
          </select>
        </label>
      </div>
      <label>
        Comment (optional)
        <input value={comment} onChange={(e) => setComment(e.target.value)} />
      </label>
      {error && (
        <p className="form-error" role="alert">
          {error}
        </p>
      )}
      <div className="editor-row">
        <button type="button" className="btn" onClick={apply}>
          Use this value
        </button>
        <button type="button" className="btn-quiet" onClick={onCancel}>
          Cancel
        </button>
      </div>
    </div>
  );
}
