import { useCallback, useEffect, useMemo, useState } from "react";
import { api, ApiError } from "../api";
import type {
  DocumentPages,
  DocumentSummary,
  FieldDecision,
  FieldName,
  FieldSpec,
  FieldValue,
  LabelDocument,
} from "../types";
import { FIELD_NAMES } from "../types";
import type { Span } from "../lib/highlight";
import { formatFieldValue } from "../lib/units";
import { FieldStatusBadge } from "./StatusBadge";
import { SourceText } from "./SourceText";
import { FieldEditor } from "./FieldEditor";

type Base = "annotation" | "extraction:rule+verifier" | "extraction:llm+verifier";

const BASE_LABEL: Record<Base, string> = {
  annotation: "Draft annotation",
  "extraction:rule+verifier": "Rule extractor + verifier",
  "extraction:llm+verifier": "Local LLM + verifier",
};

const DOC_KEY = "_doc";

function statusChip(d: DocumentSummary): { label: string; kind: string } {
  if (!d.text_available) return { label: "Text not fetched", kind: "muted" };
  switch (d.annotation_status) {
    case "human-verified":
      return { label: "Verified", kind: "ok" };
    case "draft-unverified":
      return { label: "Draft", kind: "review" };
    case "synthetic-gold":
      return { label: "Synthetic", kind: "muted" };
    default:
      return { label: "Not annotated", kind: "muted" };
  }
}

function readReviewer(): string {
  try {
    return window.localStorage.getItem("mrcond.reviewer") ?? "";
  } catch {
    return "";
  }
}

function storeReviewer(name: string) {
  try {
    window.localStorage.setItem("mrcond.reviewer", name);
  } catch {
    // storage unavailable (private mode); the name is still sent with the request
  }
}

export function ReviewPage({ initialDocId }: { initialDocId: string | null }) {
  const [docs, setDocs] = useState<DocumentSummary[]>([]);
  const [specs, setSpecs] = useState<Map<FieldName, FieldSpec>>(new Map());
  const [docId, setDocId] = useState<string | null>(initialDocId);
  const [pages, setPages] = useState<DocumentPages | null>(null);
  const [pagesError, setPagesError] = useState<string | null>(null);
  const [base, setBase] = useState<Base>("annotation");
  const [label, setLabel] = useState<LabelDocument | null>(null);
  const [labelNote, setLabelNote] = useState<string | null>(null);
  const [setIdx, setSetIdx] = useState(0);
  const [activeKey, setActiveKey] = useState<string | null>(null);
  const [editing, setEditing] = useState<string | null>(null);
  const [decisions, setDecisions] = useState<Record<string, FieldDecision>>({});
  const [showNotStated, setShowNotStated] = useState(false);
  const [reviewer, setReviewer] = useState(readReviewer);
  const [saveMsg, setSaveMsg] = useState<{ kind: "ok" | "error"; text: string } | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    api.documents().then((d) => {
      setDocs(d);
      setDocId((cur) => cur ?? d.find((x) => x.text_available && x.annotation_status !== "pending")?.doc_id ?? null);
    });
    api.fields().then((f) => setSpecs(new Map(f.map((s) => [s.name, s]))));
  }, []);

  useEffect(() => {
    if (initialDocId) setDocId(initialDocId);
  }, [initialDocId]);

  useEffect(() => {
    if (!docId) return;
    setPages(null);
    setPagesError(null);
    setDecisions({});
    setEditing(null);
    setActiveKey(null);
    setSaveMsg(null);
    api
      .pages(docId)
      .then(setPages)
      .catch((e: unknown) => setPagesError(e instanceof Error ? e.message : String(e)));
  }, [docId]);

  useEffect(() => {
    if (!docId) return;
    setLabel(null);
    setLabelNote(null);
    setSetIdx(0);
    setDecisions({});
    const load =
      base === "annotation" ? api.annotation(docId) : api.extraction(docId, base.replace("extraction:", ""));
    load.then(setLabel).catch((e: unknown) => {
      if (e instanceof ApiError && base === "annotation" && e.status === 404) {
        setLabelNote("This document has no annotation yet; showing the rule extractor + verifier output.");
        setBase("extraction:rule+verifier");
        return;
      }
      setLabelNote(e instanceof Error ? e.message : String(e));
    });
  }, [docId, base]);

  const doc = docs.find((d) => d.doc_id === docId) ?? null;
  const cs = label?.condition_sets[setIdx] ?? null;

  const rows = useMemo(() => {
    if (!cs) return [] as { key: string; field: FieldName; fv: FieldValue }[];
    return FIELD_NAMES.map((f) => ({ key: `${cs.set_id}.${f}`, field: f, fv: cs[f] })).filter(
      (r) => showNotStated || r.fv.status !== "NOT_STATED",
    );
  }, [cs, showNotStated]);

  const spansByPage = useMemo(() => {
    const m = new Map<number, Span[]>();
    const add = (key: string, fv: FieldValue) => {
      for (const ev of fv.evidence) {
        const arr = m.get(ev.page) ?? [];
        arr.push({ start: ev.char_start, end: ev.char_end, key });
        m.set(ev.page, arr);
      }
    };
    if (label) add(`${DOC_KEY}.mr_status`, label.mr_status);
    if (cs) for (const f of FIELD_NAMES) if (cs[f].status !== "NOT_STATED") add(`${cs.set_id}.${f}`, cs[f]);
    return m;
  }, [label, cs]);

  const keyLabel = useCallback(
    (key: string) => {
      const field = key.split(".").slice(1).join(".");
      if (field === "mr_status") return "MR status";
      return specs.get(field as FieldName)?.label ?? field;
    },
    [specs],
  );

  const focusKey = useCallback((key: string) => {
    setActiveKey(key);
    requestAnimationFrame(() => {
      const el = Array.from(document.querySelectorAll<HTMLElement>("mark[data-anchor]")).find((m) =>
        (m.dataset.anchor ?? "").split(" ").includes(key),
      );
      el?.scrollIntoView({ block: "center", behavior: "smooth" });
    });
  }, []);

  const flagged = useMemo(() => {
    if (!label) return [] as string[];
    const out: string[] = [];
    if (label.mr_status.status === "NEEDS_REVIEW") out.push(`${DOC_KEY}.mr_status`);
    for (const s of label.condition_sets)
      for (const f of FIELD_NAMES) if (s[f].status === "NEEDS_REVIEW") out.push(`${s.set_id}.${f}`);
    return out;
  }, [label]);
  const undecided = flagged.filter((k) => !decisions[k]);

  function decide(key: string, d: FieldDecision | null) {
    setDecisions((cur) => {
      const next = { ...cur };
      if (d) next[key] = d;
      else delete next[key];
      return next;
    });
    setSaveMsg(null);
  }

  async function save() {
    if (!docId || !label) return;
    if (reviewer.trim().length < 2) {
      setSaveMsg({ kind: "error", text: "Enter your name as reviewer before saving." });
      return;
    }
    storeReviewer(reviewer.trim());
    setSaving(true);
    try {
      const saved = await api.verify(docId, { reviewer: reviewer.trim(), base, decisions: Object.values(decisions) });
      setSaveMsg({
        kind: "ok",
        text: `Saved as human-verified by ${saved.provenance.reviewer ?? reviewer} (${Object.keys(decisions).length} decisions).`,
      });
      setDocs(await api.documents());
      setBase("annotation");
      setLabel(saved);
      setDecisions({});
    } catch (e) {
      setSaveMsg({ kind: "error", text: e instanceof Error ? e.message : String(e) });
    } finally {
      setSaving(false);
    }
  }

  function renderRow(key: string, field: FieldName | "mr_status", fv: FieldValue, setId: string) {
    const decision = decisions[key];
    const spec = field === "mr_status" ? null : specs.get(field) ?? null;
    const shown: FieldValue =
      decision?.action === "edit"
        ? {
            ...fv,
            status: "EXTRACTED",
            value: decision.value ?? null,
            unit: decision.unit ?? null,
            normalized_value: decision.normalized_value ?? decision.value ?? null,
            comparator: decision.comparator ?? fv.comparator,
            evidence: decision.quote
              ? [{ doc_id: docId ?? "", page: decision.page ?? 1, char_start: 0, char_end: 0, quote: decision.quote }]
              : fv.evidence,
          }
        : fv;
    const ev = shown.evidence[0];
    const isActive = activeKey === key;
    return (
      <li key={key} className={`field${isActive ? " field-active" : ""}${fv.status === "NEEDS_REVIEW" ? " field-flag" : ""}`}>
        <button type="button" className="field-head" onClick={() => focusKey(key)} aria-pressed={isActive}>
          <span className="field-label">{field === "mr_status" ? "MR status" : spec?.label ?? field}</span>
          <span className="field-value">
            {decision?.action === "reject" ? <s>{formatFieldValue(field, fv)}</s> : formatFieldValue(field, shown)}
          </span>
          <FieldStatusBadge status={fv.status} />
        </button>
        <div className="field-body">
          {ev && ev.quote && decision?.action !== "reject" ? (
            <blockquote className="quote">
              “{ev.quote.replace(/\s+/g, " ")}”<span className="quote-ref"> p. {ev.page}</span>
            </blockquote>
          ) : (
            <span className="quote-none">{decision?.action === "reject" ? "Rejected by reviewer" : "No evidence"}</span>
          )}
          <div className="decide" role="group" aria-label="Decision">
            {fv.status !== "NOT_STATED" && (
              <button
                type="button"
                className={decision?.action === "accept" ? "seg seg-on" : "seg"}
                aria-pressed={decision?.action === "accept"}
                onClick={() => decide(key, decision?.action === "accept" ? null : { set_id: setId, field, action: "accept" })}
              >
                Accept
              </button>
            )}
            <button
              type="button"
              className={decision?.action === "edit" ? "seg seg-on" : "seg"}
              aria-pressed={decision?.action === "edit"}
              onClick={() => setEditing(editing === key ? null : key)}
            >
              {fv.status === "NOT_STATED" ? "Add value" : "Edit"}
            </button>
            {fv.status !== "NOT_STATED" && (
              <button
                type="button"
                className={decision?.action === "reject" ? "seg seg-on seg-reject" : "seg"}
                aria-pressed={decision?.action === "reject"}
                onClick={() => decide(key, decision?.action === "reject" ? null : { set_id: setId, field, action: "reject" })}
              >
                Reject
              </button>
            )}
          </div>
        </div>
        {fv.review_reasons.length > 0 && fv.status === "NEEDS_REVIEW" && (
          <p className="reasons">Flagged: {fv.review_reasons.join("; ")}</p>
        )}
        {editing === key && pages && (
          <FieldEditor
            setId={setId}
            field={field}
            spec={spec}
            current={fv}
            pages={pages.pages}
            onApply={(d) => {
              decide(key, d);
              setEditing(null);
            }}
            onCancel={() => setEditing(null)}
          />
        )}
      </li>
    );
  }

  return (
    <div className="review">
      <aside className="doclist" aria-label="Documents">
        <h2 className="pane-title">Labeling documents</h2>
        <label className="doclist-select">
          Document
          <select value={docId ?? ""} onChange={(e) => setDocId(e.target.value)}>
            {docs.map((d) => (
              <option key={d.doc_id} value={d.doc_id}>
                {d.device_name ?? d.doc_id}
              </option>
            ))}
          </select>
        </label>
        <ul>
          {docs.map((d) => {
            const chip = statusChip(d);
            return (
              <li key={d.doc_id}>
                <button
                  type="button"
                  className={d.doc_id === docId ? "doc doc-on" : "doc"}
                  aria-current={d.doc_id === docId ? "true" : undefined}
                  onClick={() => setDocId(d.doc_id)}
                >
                  <span className="doc-name">{d.device_name ?? d.doc_id}</span>
                  <span className="doc-maker">{d.manufacturer ?? (d.source === "synthetic" ? "Synthetic example" : "")}</span>
                  <span className={`chip-status chip-${chip.kind}`}>{chip.label}</span>
                </button>
              </li>
            );
          })}
        </ul>
      </aside>

      <section className="sourcepane" aria-label="Source labeling text">
        <div className="pane-head">
          <h2 className="pane-title">{doc?.title ?? "Select a document"}</h2>
          {doc?.url && (
            <p className="pane-sub">
              Source: <a href={doc.url}>{new URL(doc.url).hostname}</a>
              {doc.retrieved ? `, retrieved ${doc.retrieved}` : ""}
              {pages ? `, text by ${pages.text_extractor}` : ""}
            </p>
          )}
        </div>
        {pagesError && (
          <p className="empty">
            The text of this document is not available locally. Run <code>mrcond fetch</code> to download it from the
            manufacturer (MRCond does not redistribute labeling).
          </p>
        )}
        {pages && (
          <SourceText
            pages={pages.pages}
            spansByPage={spansByPage}
            activeKey={activeKey}
            onSelect={focusKey}
            keyLabel={keyLabel}
          />
        )}
      </section>

      <section className="fieldspane" aria-label="Extracted conditions">
        <div className="pane-head">
          <h2 className="pane-title">Conditions</h2>
          <label className="inline">
            Shown values
            <select value={base} onChange={(e) => setBase(e.target.value as Base)}>
              {(Object.keys(BASE_LABEL) as Base[]).map((b) => (
                <option key={b} value={b}>
                  {BASE_LABEL[b]}
                </option>
              ))}
            </select>
          </label>
        </div>
        {labelNote && <p className="note">{labelNote}</p>}
        {label && (
          <>
            <p className="provenance">
              {label.provenance.verification === "human-verified"
                ? `Human-verified by ${label.provenance.reviewer ?? "unknown"}`
                : label.provenance.verification === "draft-unverified"
                  ? "Draft annotation, not yet verified by a person"
                  : `Machine output (${label.provenance.extractor}), not verified`}
            </p>
            <ul className="fields">{renderRow(`${DOC_KEY}.mr_status`, "mr_status", label.mr_status, DOC_KEY)}</ul>
            {label.condition_sets.length > 1 && (
              <div className="settabs" role="tablist" aria-label="Condition sets">
                {label.condition_sets.map((s, i) => (
                  <button
                    key={s.set_id}
                    role="tab"
                    aria-selected={i === setIdx}
                    className={i === setIdx ? "settab settab-on" : "settab"}
                    onClick={() => setSetIdx(i)}
                  >
                    {s.scope ?? s.set_id}
                  </button>
                ))}
              </div>
            )}
            {label.condition_sets.length === 0 && <p className="empty">No condition sets recorded.</p>}
            <ul className="fields">{rows.map((r) => renderRow(r.key, r.field, r.fv, cs?.set_id ?? "cs1"))}</ul>
            <label className="check toggle">
              <input type="checkbox" checked={showNotStated} onChange={(e) => setShowNotStated(e.target.checked)} />
              Show conditions not stated in this document
            </label>
            <div className="savebar">
              <label>
                Reviewer
                <input value={reviewer} onChange={(e) => setReviewer(e.target.value)} placeholder="Your name" />
              </label>
              <button type="button" className="btn" onClick={save} disabled={saving}>
                {saving ? "Saving…" : "Save verified annotation"}
              </button>
              <p className="savehint">
                {Object.keys(decisions).length} decisions.{" "}
                {undecided.length > 0
                  ? `${undecided.length} flagged ${undecided.length === 1 ? "value needs" : "values need"} a decision before saving.`
                  : "Values you do not touch are kept as shown."}
              </p>
              {saveMsg && (
                <p className={saveMsg.kind === "ok" ? "save-ok" : "form-error"} role="status">
                  {saveMsg.text}
                </p>
              )}
            </div>
          </>
        )}
      </section>
    </div>
  );
}
