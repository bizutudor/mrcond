import { useCallback, useEffect, useState } from "react";
import { api } from "../api";
import type {
  CheckResult,
  ConditionCheck,
  ImplantResult,
  ImplantSummary,
  ProtocolPlan,
  ScannerProfile,
  SequencePlan,
} from "../types";
import { byConcern, outcomeTone, requiresSignOff, selectedChecks } from "../lib/outcome";
import {
  fmtNumber,
  gaussPerCmToTeslaPerMeter,
  numberInputError,
  parseNumberInput,
  teslaPerMeterToGaussPerCm,
  type NumberRule,
} from "../lib/units";
import { Badge, CheckStatusBadge } from "./StatusBadge";

const EMPTY_PROTOCOL: ProtocolPlan = {
  operating_mode: "normal",
  transmit_coil: "body",
  receive_coil: "local",
  rf_excitation: "CP",
  nucleus: "1H",
  sequences: [
    { name: "T2 TSE", b1rms_uT: null, wb_sar_w_per_kg: 1.4, head_sar_w_per_kg: 1.1 },
    { name: "T1 GRE", b1rms_uT: null, wb_sar_w_per_kg: 0.6, head_sar_w_per_kg: 0.5 },
  ],
  total_active_scan_time_min: 25,
  sessions: 1,
  wait_between_sessions_min: null,
  scan_region: "lumbar spine",
  slew_rate_limit_t_per_m_per_s: null,
  sfg_at_implant_t_per_m: null,
  device_configuration_confirmed: null,
};

function numField(v: number | null): string {
  return v === null || Number.isNaN(v) ? "" : String(v);
}

/** Equal numbers; null and NaN both mean "no usable value". */
function sameNumber(a: number | null, b: number | null): boolean {
  const none = (x: number | null) => x === null || Number.isNaN(x);
  return none(a) ? none(b) : a === b;
}

type Validity = (key: string, error: string | null) => void;

/**
 * Number field that keeps what the user typed. Text that is not a usable number is reported to the form
 * (which then refuses to compare) instead of being silently replaced by an earlier or default value.
 */
function NumberInput({
  id,
  label,
  value,
  onChange,
  onValidity,
  rule,
  unit,
  hint,
}: {
  id: string;
  label: string;
  value: number | null;
  onChange: (v: number | null) => void;
  onValidity: Validity;
  rule?: NumberRule;
  unit?: string;
  hint?: string;
}) {
  const [text, setText] = useState(numField(value));
  // follow external changes (e.g. a new scanner preset) but keep what the user typed, including partial or
  // invalid input such as "0," or "abc" (shown with its error instead of being wiped)
  useEffect(() => {
    setText((t) => (sameNumber(numberInputError(t, rule) ? null : parseNumberInput(t), value) ? t : numField(value)));
  }, [value]);
  const error = numberInputError(text, rule);
  useEffect(() => onValidity(id, error), [id, error, onValidity]);
  useEffect(() => () => onValidity(id, null), [id, onValidity]);
  return (
    <label>
      {label}
      <span className="with-unit">
        <input
          value={text}
          inputMode="decimal"
          aria-invalid={error ? true : undefined}
          onChange={(e) => {
            setText(e.target.value);
            onChange(numberInputError(e.target.value, rule) ? null : parseNumberInput(e.target.value));
          }}
        />
        {unit && <span className="unit">{unit}</span>}
      </span>
      {error && <span className="field-error">{error}</span>}
      {hint && <span className="hint">{hint}</span>}
    </label>
  );
}

/** One RF value of one sequence (B1+rms or SAR), validated like NumberInput. */
function SeqCell({
  id,
  ariaLabel,
  value,
  onChange,
  onValidity,
}: {
  id: string;
  ariaLabel: string;
  value: number | null;
  onChange: (v: number | null) => void;
  onValidity: Validity;
}) {
  const [text, setText] = useState(numField(value));
  const error = numberInputError(text);
  useEffect(() => onValidity(id, error), [id, error, onValidity]);
  useEffect(() => () => onValidity(id, null), [id, onValidity]);
  return (
    <>
      <input
        aria-label={ariaLabel}
        aria-invalid={error ? true : undefined}
        inputMode="decimal"
        value={text}
        onChange={(e) => {
          setText(e.target.value);
          onChange(numberInputError(e.target.value) ? null : parseNumberInput(e.target.value));
        }}
      />
      {error && <span className="field-error">{error}</span>}
    </>
  );
}

function verificationLabel(v: string, source: string): { kind: "ok" | "review" | "muted"; text: string } {
  if (source.startsWith("synthetic")) return { kind: "muted", text: "Synthetic demo" };
  if (v === "human-verified") return { kind: "ok", text: "Human-verified" };
  return { kind: "review", text: "Draft, not verified" };
}

function CheckRow({ c }: { c: ConditionCheck }) {
  const ev = c.evidence[0];
  return (
    <tr className={`check-${c.status.toLowerCase()}`}>
      <td>
        <CheckStatusBadge status={c.status} />
      </td>
      <th scope="row">{c.label}</th>
      <td>{c.label_value ?? "—"}</td>
      <td>{c.entered_value ?? "—"}</td>
      <td>
        {c.reason}
        {ev && ev.quote && (
          <span className="check-ev">
            “{ev.quote.replace(/\s+/g, " ")}”{" "}
            {ev.doc_id.startsWith("IMP-") ? null : (
              <a href={`#/review?doc=${encodeURIComponent(ev.doc_id)}`}>source p. {ev.page}</a>
            )}
          </span>
        )}
      </td>
    </tr>
  );
}

function ChecksTable({ checks, collapsePasses = false }: { checks: ConditionCheck[]; collapsePasses?: boolean }) {
  const concerns = checks.filter((c) => c.status !== "PASS");
  const passes = checks.filter((c) => c.status === "PASS");
  if (collapsePasses && concerns.length > 0 && passes.length > 0) {
    return (
      <>
        <ChecksTable checks={concerns} />
        <details className="passes">
          <summary>
            {passes.length} {passes.length === 1 ? "condition" : "conditions"} without conflict
          </summary>
          <ChecksTable checks={passes} />
        </details>
      </>
    );
  }
  return (
    <div className="table-scroll">
      <table className="checks">
        <thead>
          <tr>
            <th scope="col">Result</th>
            <th scope="col">Condition</th>
            <th scope="col">Labeling</th>
            <th scope="col">Entered</th>
            <th scope="col">Reason and evidence</th>
          </tr>
        </thead>
        <tbody>
          {checks.map((c, i) => (
            <CheckRow key={`${c.field}-${i}`} c={c} />
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ImplantBlock({ r }: { r: ImplantResult }) {
  const selected = r.set_results.find((s) => s.set_id === r.selected_set_id) ?? null;
  const others = r.set_results.filter((s) => s.set_id !== r.selected_set_id);
  return (
    <section className="implant-result">
      <header className="implant-head">
        <h3>{r.device_name ?? r.implant_id}</h3>
        <CheckStatusBadge status={r.status} />
      </header>
      <p className="implant-meta">
        {r.mr_status} labeling, record {r.verification}.{" "}
        {selected
          ? `Condition set used: ${selected.scope ?? selected.set_id}.`
          : r.set_results.length
            ? "No condition set applies to this scanner."
            : ""}
      </p>
      <ChecksTable checks={selectedChecks(r)} collapsePasses />
      {selected?.notes.map((n) => (
        <p className="implant-note" key={n}>
          {n}
        </p>
      ))}
      {others.length > 0 && (
        <details className="other-sets">
          <summary>Other condition sets in this labeling ({others.length})</summary>
          {others.map((s) => (
            <div key={s.set_id}>
              <h4>
                {s.scope ?? s.set_id} <CheckStatusBadge status={s.status} />
              </h4>
              <ChecksTable checks={byConcern(s.checks)} collapsePasses />
            </div>
          ))}
        </details>
      )}
    </section>
  );
}

export function CheckerPage() {
  const [implants, setImplants] = useState<ImplantSummary[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [profiles, setProfiles] = useState<ScannerProfile[]>([]);
  const [scanner, setScanner] = useState<ScannerProfile | null>(null);
  const [protocol, setProtocol] = useState<ProtocolPlan>(EMPTY_PROTOCOL);
  const [sfgUnit, setSfgUnit] = useState<"T/m" | "G/cm">("T/m");
  const [sfgText, setSfgText] = useState("");
  const [result, setResult] = useState<CheckResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [invalid, setInvalid] = useState<Record<string, string>>({});
  // stable row keys, so removing a sequence never shows one sequence's values next to another's name
  const [seqKeys, setSeqKeys] = useState<number[]>(() => EMPTY_PROTOCOL.sequences.map((_, i) => i));
  const [nextSeqKey, setNextSeqKey] = useState(EMPTY_PROTOCOL.sequences.length);

  const onValidity = useCallback<Validity>((key, message) => {
    setInvalid((cur) => {
      if ((cur[key] ?? null) === message) return cur;
      const next = { ...cur };
      if (message) next[key] = message;
      else delete next[key];
      return next;
    });
  }, []);

  useEffect(() => {
    if (!result) return;
    // show what changed: bring the outcome into view after each comparison
    document.querySelector(".outcome")?.scrollIntoView({ block: "start" });
  }, [result]);

  useEffect(() => {
    api.implants().then(setImplants);
    api.scannerProfiles().then((p) => {
      setProfiles(p);
      setScanner((cur) => cur ?? (p[1] ?? p[0] ?? null));
    });
  }, []);

  // any change to the inputs makes a shown result stale: remove it rather than show it next to new inputs
  const set = <K extends keyof ProtocolPlan>(k: K, v: ProtocolPlan[K]) => {
    setProtocol((p) => ({ ...p, [k]: v }));
    setResult(null);
  };
  const setSeq = (i: number, patch: Partial<SequencePlan>) => {
    setProtocol((p) => ({ ...p, sequences: p.sequences.map((s, j) => (j === i ? { ...s, ...patch } : s)) }));
    setResult(null);
  };
  const updateScanner = (s: ScannerProfile) => {
    setScanner(s);
    setResult(null);
  };

  async function run() {
    if (!scanner) return;
    setError(null);
    const problems = Object.keys(invalid);
    if (problems.length > 0) {
      setResult(null);
      setError(`Correct the highlighted ${problems.length === 1 ? "entry" : "entries"} before comparing.`);
      return;
    }
    setRunning(true);
    try {
      setResult(await api.check({ implant_ids: selected, scanner, protocol }));
    } catch (e) {
      setResult(null);
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  }

  const sfgError = numberInputError(sfgText);
  useEffect(() => onValidity("sfg", sfgError), [sfgError, onValidity]);
  const sfgFromText = (text: string, unit: "T/m" | "G/cm") => {
    const v = numberInputError(text) ? null : parseNumberInput(text);
    return v === null ? null : unit === "T/m" ? v : gaussPerCmToTeslaPerMeter(v);
  };

  return (
    <div className="checker">
      <form
        className="checker-form"
        onSubmit={(e) => {
          e.preventDefault();
          void run();
        }}
      >
        <fieldset>
          <legend>Implants</legend>
          <p className="hint">Every selected implant must meet its labeled conditions.</p>
          <ul className="implant-list">
            {implants.map((im) => {
              const v = verificationLabel(im.verification, im.source);
              const on = selected.includes(im.implant_id);
              return (
                <li key={im.implant_id}>
                  <label className="implant-pick">
                    <input
                      type="checkbox"
                      checked={on}
                      onChange={() => {
                        setSelected((s) => (on ? s.filter((x) => x !== im.implant_id) : [...s, im.implant_id]));
                        setResult(null);
                      }}
                    />
                    <span className="implant-name">{im.device_name ?? im.implant_id}</span>
                    <span className="implant-sub">
                      {im.mr_status}
                      {im.manufacturer ? `, ${im.manufacturer}` : ""}
                    </span>
                    <Badge kind={v.kind} label={v.text} />
                  </label>
                </li>
              );
            })}
          </ul>
        </fieldset>

        <fieldset>
          <legend>Scanner</legend>
          <label>
            Preset
            <select
              value={scanner?.profile_id ?? ""}
              onChange={(e) => {
                const p = profiles.find((x) => x.profile_id === e.target.value);
                if (p) updateScanner({ ...p });
              }}
            >
              {profiles.map((p) => (
                <option key={p.profile_id} value={p.profile_id}>
                  {p.name}
                </option>
              ))}
            </select>
            <span className="hint">Generic presets. Replace every value with your scanner's specifications.</span>
          </label>
          {scanner && (
            <div className="grid2">
              <NumberInput
                id="b0"
                label="Static field B0"
                unit="T"
                value={scanner.b0_tesla}
                rule={{ required: true, minExclusive: true, max: 25 }}
                onValidity={onValidity}
                onChange={(v) => updateScanner({ ...scanner, b0_tesla: v ?? Number.NaN })}
              />
              <NumberInput
                id="scanner-slew"
                label="Max slew rate per axis"
                unit="T/m/s"
                value={scanner.max_slew_rate_t_per_m_per_s}
                rule={{ minExclusive: true, max: 1000 }}
                onValidity={onValidity}
                onChange={(v) => updateScanner({ ...scanner, max_slew_rate_t_per_m_per_s: v })}
                hint="Per axis, not the effective (vector) slew rate"
              />
              <label>
                Scanner type
                <select
                  value={scanner.scanner_type ?? ""}
                  onChange={(e) => updateScanner({ ...scanner, scanner_type: e.target.value || null })}
                >
                  <option value="cylindrical">Cylindrical (closed bore)</option>
                  <option value="open">Open</option>
                  <option value="">Unknown</option>
                </select>
              </label>
              <label>
                B0 orientation
                <select
                  value={scanner.b0_orientation ?? ""}
                  onChange={(e) => updateScanner({ ...scanner, b0_orientation: e.target.value || null })}
                >
                  <option value="horizontal">Horizontal</option>
                  <option value="vertical">Vertical</option>
                  <option value="">Unknown</option>
                </select>
              </label>
            </div>
          )}
        </fieldset>

        <fieldset>
          <legend>Planned exam</legend>
          <div className="grid2">
            <label>
              Operating mode
              <select
                value={protocol.operating_mode ?? ""}
                onChange={(e) => set("operating_mode", (e.target.value || null) as ProtocolPlan["operating_mode"])}
              >
                <option value="normal">Normal Operating Mode</option>
                <option value="first_level">First Level Controlled</option>
                <option value="">Not decided</option>
              </select>
            </label>
            <label>
              RF excitation
              <select value={protocol.rf_excitation ?? ""} onChange={(e) => set("rf_excitation", e.target.value || null)}>
                <option value="CP">Circularly polarized (CP)</option>
                <option value="MC-2">Multichannel (MC-2)</option>
                <option value="">Unknown</option>
              </select>
            </label>
            <label>
              Transmit coil
              <select value={protocol.transmit_coil ?? ""} onChange={(e) => set("transmit_coil", e.target.value || null)}>
                <option value="body">Body coil</option>
                <option value="head">Head transmit/receive coil</option>
                <option value="extremity">Extremity transmit/receive coil</option>
                <option value="local">Other local transmit coil</option>
              </select>
            </label>
            <label>
              Receive coil
              <select value={protocol.receive_coil ?? ""} onChange={(e) => set("receive_coil", e.target.value || null)}>
                <option value="local">Local receive-only coil</option>
                <option value="body">Body coil</option>
                <option value="head">Head coil</option>
                <option value="extremity">Extremity coil</option>
              </select>
            </label>
            <label>
              Imaged nucleus
              <select value={protocol.nucleus ?? ""} onChange={(e) => set("nucleus", e.target.value || null)}>
                <option value="1H">1H (proton)</option>
                <option value="31P">31P</option>
                <option value="23Na">23Na</option>
              </select>
            </label>
            <label>
              Scan region or landmark
              <input value={protocol.scan_region ?? ""} onChange={(e) => set("scan_region", e.target.value || null)} />
            </label>
          </div>

          <table className="seq-table">
            <caption>RF exposure per sequence, as displayed by the scanner</caption>
            <thead>
              <tr>
                <th scope="col">Sequence</th>
                <th scope="col">B1+rms (µT)</th>
                <th scope="col">Whole-body SAR (W/kg)</th>
                <th scope="col">Head SAR (W/kg)</th>
                <th scope="col">
                  <span className="sr-only">Remove</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {protocol.sequences.map((s, i) => (
                <tr key={seqKeys[i] ?? `row-${i}`}>
                  <td>
                    <input aria-label="Sequence name" value={s.name} onChange={(e) => setSeq(i, { name: e.target.value })} />
                  </td>
                  {(["b1rms_uT", "wb_sar_w_per_kg", "head_sar_w_per_kg"] as const).map((k) => (
                    <td key={k}>
                      <SeqCell
                        id={`seq-${seqKeys[i] ?? i}-${k}`}
                        ariaLabel={`${s.name} ${k}`}
                        value={s[k]}
                        onValidity={onValidity}
                        onChange={(v) => setSeq(i, { [k]: v })}
                      />
                    </td>
                  ))}
                  <td>
                    <button
                      type="button"
                      className="btn-quiet"
                      aria-label={`Remove ${s.name}`}
                      onClick={() => {
                        set("sequences", protocol.sequences.filter((_, j) => j !== i));
                        setSeqKeys((keys) => keys.filter((_, j) => j !== i));
                      }}
                    >
                      Remove
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <button
            type="button"
            className="btn-quiet"
            onClick={() => {
              set("sequences", [
                ...protocol.sequences,
                { name: `Sequence ${protocol.sequences.length + 1}`, b1rms_uT: null, wb_sar_w_per_kg: null, head_sar_w_per_kg: null },
              ]);
              setSeqKeys((keys) => [...keys, nextSeqKey]);
              setNextSeqKey((k) => k + 1);
            }}
          >
            Add sequence
          </button>

          <div className="grid2">
            <NumberInput
              id="scan-time"
              label="Total active scan time"
              unit="min"
              value={protocol.total_active_scan_time_min}
              rule={{ max: 1440 }}
              onValidity={onValidity}
              onChange={(v) => set("total_active_scan_time_min", v)}
            />
            <NumberInput
              id="protocol-slew"
              label="Protocol slew-rate limit"
              unit="T/m/s"
              value={protocol.slew_rate_limit_t_per_m_per_s}
              rule={{ minExclusive: true, max: 1000 }}
              onValidity={onValidity}
              onChange={(v) => set("slew_rate_limit_t_per_m_per_s", v)}
              hint="Only if lower than the scanner maximum (per axis)"
            />
            <label>
              Spatial field gradient at the implant
              <span className="with-unit">
                <input
                  inputMode="decimal"
                  value={sfgText}
                  aria-invalid={sfgError ? true : undefined}
                  onChange={(e) => {
                    setSfgText(e.target.value);
                    set("sfg_at_implant_t_per_m", sfgFromText(e.target.value, sfgUnit));
                  }}
                />
                <select
                  aria-label="Gradient unit"
                  value={sfgUnit}
                  onChange={(e) => {
                    // the number keeps its meaning: convert the entered value to the new unit
                    const unit = e.target.value as "T/m" | "G/cm";
                    const tpm = protocol.sfg_at_implant_t_per_m;
                    if (tpm !== null && !sfgError)
                      setSfgText(fmtNumber(unit === "T/m" ? tpm : teslaPerMeterToGaussPerCm(tpm)));
                    setSfgUnit(unit);
                  }}
                >
                  <option value="T/m">T/m</option>
                  <option value="G/cm">G/cm</option>
                </select>
              </span>
              {sfgError && <span className="field-error">{sfgError}</span>}
              <span className="hint">From the scanner's spatial-field-gradient map. Leave empty if unknown.</span>
            </label>
            <NumberInput
              id="sessions"
              label="Imaging sessions"
              value={protocol.sessions}
              rule={{ required: true, integer: true, min: 1, max: 100 }}
              onValidity={onValidity}
              onChange={(v) => set("sessions", v ?? Number.NaN)}
            />
            {protocol.sessions > 1 && (
              <NumberInput
                id="wait"
                label="Wait between sessions"
                unit="min"
                value={protocol.wait_between_sessions_min}
                rule={{ max: 100000 }}
                onValidity={onValidity}
                onChange={(v) => set("wait_between_sessions_min", v)}
              />
            )}
          </div>
          <label className="check">
            <input
              type="checkbox"
              checked={protocol.device_configuration_confirmed === true}
              onChange={(e) => set("device_configuration_confirmed", e.target.checked ? true : null)}
            />
            Implant configured as its labeling requires (for example stimulation off, MRI mode on)
          </label>
        </fieldset>

        <button type="submit" className="btn btn-run" disabled={!scanner || running}>
          {running ? "Comparing…" : "Compare with labeling"}
        </button>
        {error && (
          <p className="form-error" role="alert">
            {error}
          </p>
        )}
      </form>

      <section className="results" aria-live="polite" aria-label="Results">
        {!result && (
          <div className="empty-results">
            <h2>Compare a planned exam with verified labeling</h2>
            <p>
              Select the patient's implants, the scanner and the planned sequences, then compare. MRCond lists every
              labeled condition with the evidence it comes from. It does not decide whether the exam can go ahead:
              that decision stays with the MR Safety Officer or Expert.
            </p>
          </div>
        )}
        {result && (
          <>
            <div className={`outcome outcome-${outcomeTone(result.outcome)}`}>
              <p className="outcome-text">{result.outcome_text}</p>
              <p className="outcome-disclaimer">{result.disclaimer}</p>
              {requiresSignOff(result.outcome) && (
                <div className="signoff" aria-label="Sign-off required">
                  <span>MR Safety Officer / Expert</span>
                  <span>Signature</span>
                  <span>Date and time</span>
                </div>
              )}
            </div>
            {result.implants.length === 0 && <p className="empty">No implant selected.</p>}
            {result.implants.map((r) => (
              <ImplantBlock key={r.implant_id} r={r} />
            ))}
          </>
        )}
      </section>
    </div>
  );
}
