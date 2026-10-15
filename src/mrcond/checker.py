"""Checker: compare a planned MR exam with verified MR Conditional labeling.

Decision support only. The checker never states that a scan is safe. Its
overall outcome is one of:

* ``CONFLICT_FOUND`` - "Conflict found"
* ``NEEDS_REVIEW`` - "Insufficient information / needs review"
* ``NO_CONFLICT_FOUND`` - "No conflict found with entered conditions — requires
  MR Safety Officer/Expert sign-off"

Per condition: PASS (no conflict with the entered value), FAIL, INSUFFICIENT_INFO
(a needed value was not entered) or NEEDS_REVIEW (the condition cannot be
evaluated automatically, e.g. scan-region restrictions, other conditions,
spatial field gradient without a value at the implant location, unverified
extraction).

Semantics:

* condition sets of one implant are *alternatives* (e.g. 1.5 T vs 3 T, or
  whole-body vs head-only); the applicable set with the best result is used
  and all sets are reported
* several implants: every implant must pass; the overall result is the most
  conservative implant result
* RF limits are checked per sequence. B1+rms is the primary limit; a SAR limit
  marked as fallback (``sar_is_b1rms_fallback``) is used for sequences without
  B1+rms. When a sequence reports both values, both are checked
  (conservative). A SAR limit not marked as fallback must always be met.
* malformed input never passes: a labeled limit that is not a finite number
  (and not an explicit "no limit") is NEEDS_REVIEW; an entered value that is
  not a finite, non-negative number is treated as not entered.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

from .merge import MODE_ORDER
from .schema import (
    FIELD_SPECS,
    OUTCOME_TEXT,
    CheckResult,
    CheckStatus,
    ConditionCheck,
    ConditionSet,
    ConditionSetResult,
    FieldStatus,
    FieldValue,
    ImplantResult,
    LabelDocument,
    MRStatus,
    ProtocolPlan,
    ScannerProfile,
)

__all__ = ["check", "RANK", "worst"]

RANK: dict[str, int] = {"PASS": 0, "INSUFFICIENT_INFO": 1, "NEEDS_REVIEW": 1, "FAIL": 2}
B0_TOL = 0.05
EPS = 1e-9


def worst(statuses: list[str]) -> CheckStatus:
    if not statuses:
        return "PASS"
    if "FAIL" in statuses:
        return "FAIL"
    if "NEEDS_REVIEW" in statuses:
        return "NEEDS_REVIEW"
    if "INSUFFICIENT_INFO" in statuses:
        return "INSUFFICIENT_INFO"
    return "PASS"


def _num(v) -> float | None:
    """A finite float, or None (None, bool, text, NaN and infinity are not readable limits)."""
    if isinstance(v, bool):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _entry(v) -> float | None:
    """A user-entered quantity: finite and non-negative, else None (= not entered)."""
    x = _num(v)
    return x if x is not None and x >= 0 else None


def _as_list(v) -> list:
    if v is None:
        return []
    return list(v) if isinstance(v, list) else [v]


def fmt_value(name: str, fv: FieldValue) -> str:
    """Human-readable label value (e.g. '≤ 2 W/kg', '1.5 T, 3 T', 'body')."""
    spec = FIELD_SPECS[name]
    if fv.status == FieldStatus.NOT_STATED:
        return "not stated"
    if fv.comparator == "any":
        return "no limit stated (explicitly unrestricted)"
    nv = fv.normalized_value
    if spec.kind == "b0":
        vals = [f"{float(v):g} T" for v in _as_list(nv) if _num(v) is not None]
        return ("≤ " + vals[0]) if fv.comparator == "le" and vals else ", ".join(vals) or "?"
    if spec.kind == "numeric":
        if _num(nv) is None:
            return str(fv.value)
        sym = {"upper": "≤ ", "lower": "≥ ", "info": ""}[spec.direction or "info"]
        extra = ""
        if name == "max_sfg_t_per_m":
            extra = f" ({float(nv) * 100:g} G/cm)"
        unit = "µT" if spec.canonical_unit == "uT" else spec.canonical_unit
        return f"{sym}{float(nv):g} {unit}{extra}"
    if spec.kind == "bool":
        return "yes" if nv else "no"
    if isinstance(nv, list):
        return ", ".join(map(str, nv))
    return str(nv if nv is not None else fv.value)


def _mk(
    name: str, status: CheckStatus, reason: str, fv: FieldValue | None, entered: str | None = None
) -> ConditionCheck:
    label = FIELD_SPECS[name].label if name in FIELD_SPECS else name
    return ConditionCheck(
        field=name,
        label=label,
        status=status,
        reason=reason,
        label_value=fmt_value(name, fv) if fv is not None and name in FIELD_SPECS else None,
        entered_value=entered,
        evidence=list(fv.evidence) if fv is not None else [],
    )


def _review(name: str, fv: FieldValue) -> ConditionCheck:
    reasons = "; ".join(fv.review_reasons) or "value flagged for review"
    return _mk(name, "NEEDS_REVIEW", f"Labeling value needs human review ({reasons}).", fv)


# --------------------------------------------------------------------------------------------
# individual conditions
# --------------------------------------------------------------------------------------------


def _check_b0(cs: ConditionSet, scanner: ScannerProfile) -> ConditionCheck:
    fv = cs.b0_tesla
    entered = f"{scanner.b0_tesla:g} T"
    if fv.status == FieldStatus.NEEDS_REVIEW:
        return _review("b0_tesla", fv)
    if fv.status == FieldStatus.NOT_STATED:
        return _mk(
            "b0_tesla", "NEEDS_REVIEW", "This condition set does not state a static field strength.", fv, entered
        )
    raw = _as_list(fv.normalized_value)
    vals = [x for x in (_num(v) for v in raw) if x is not None and x > 0]
    if not vals or len(vals) != len(raw):
        return _mk("b0_tesla", "NEEDS_REVIEW", "Field strength value could not be read.", fv, entered)
    b0 = _num(scanner.b0_tesla)
    if b0 is None or b0 <= 0:
        return _mk("b0_tesla", "NEEDS_REVIEW", "The scanner field strength is not a valid number.", fv, entered)
    if fv.comparator == "le":
        ok = scanner.b0_tesla <= max(vals) + EPS
        why = f"Scanner {scanner.b0_tesla:g} T {'≤' if ok else '>'} labeled maximum {max(vals):g} T."
    else:
        ok = any(abs(scanner.b0_tesla - v) <= B0_TOL for v in vals)
        why = (
            f"Scanner {scanner.b0_tesla:g} T is {'one of' if ok else 'not one of'} the labeled field strengths "
            f"({', '.join(f'{v:g} T' for v in vals)})."
        )
    return _mk("b0_tesla", "PASS" if ok else "FAIL", why, fv, entered)


def _check_upper(name: str, fv: FieldValue, entered: float | None, what: str, hint: str) -> ConditionCheck | None:
    if fv.status == FieldStatus.NOT_STATED:
        return None
    if fv.status == FieldStatus.NEEDS_REVIEW:
        return _review(name, fv)
    if fv.comparator == "any":
        return _mk(
            name,
            "PASS",
            "Labeling states no limit for this parameter.",
            fv,
            None if entered is None else f"{entered:g}",
        )
    limit = _num(fv.normalized_value)
    unit = FIELD_SPECS[name].canonical_unit
    if limit is None:
        return _mk(name, "NEEDS_REVIEW", "Labeled limit could not be read as a number.", fv)
    if entered is not None and _entry(entered) is None:
        return _mk(name, "NEEDS_REVIEW", f"The entered {what} is not a valid non-negative number.", fv, str(entered))
    if entered is None:
        return _mk(name, "INSUFFICIENT_INFO", f"Enter the {what} ({hint}).", fv)
    ok = entered <= limit + EPS
    return _mk(
        name,
        "PASS" if ok else "FAIL",
        f"Entered {what} {entered:g} {unit} {'≤' if ok else 'exceeds'} labeled limit {limit:g} {unit}.",
        fv,
        f"{entered:g} {unit}",
    )


def _norm_token(v: str | None) -> str | None:
    if v is None:
        return None
    s = v.strip().lower()
    table = {
        "body": "body",
        "body coil": "body",
        "whole-body": "body",
        "whole body": "body",
        "integrated body": "body",
        "head": "head",
        "head coil": "head",
        "head t/r": "head",
        "head transmit/receive": "head",
        "extremity": "extremity",
        "knee": "extremity",
        "local": "local",
        "surface": "local",
        "cp": "CP",
        "circularly polarized": "CP",
        "circular": "CP",
        "mc-2": "MC-2",
        "mc2": "MC-2",
        "multichannel": "MC-2",
        "multi-channel": "MC-2",
        "ptx": "MC-2",
        "1h": "1H",
        "proton": "1H",
        "hydrogen": "1H",
        "31p": "31P",
        "23na": "23Na",
        "13c": "13C",
        "19f": "19F",
        "any": "any",
    }
    return table.get(s, s)


def _check_allowed(
    name: str, fv: FieldValue, entered: str | None, what: str, implied: list[str] | None = None
) -> ConditionCheck | None:
    if fv.status == FieldStatus.NEEDS_REVIEW:
        return _review(name, fv)
    if fv.status == FieldStatus.NOT_STATED:
        if implied is None:
            return None
        allowed = implied
    else:
        allowed = [str(x) for x in _as_list(fv.normalized_value)]
    tok = _norm_token(entered)
    if "any" in allowed:
        return _mk(name, "PASS", f"Labeling does not restrict the {what}.", fv, entered)
    if tok is None:
        return _mk(name, "INSUFFICIENT_INFO", f"Enter the planned {what}.", fv)
    ok = tok in allowed
    src = "labeled" if fv.status == FieldStatus.EXTRACTED else "assumed (not listed in labeling)"
    return _mk(
        name,
        "PASS" if ok else "FAIL",
        f"Planned {what} '{entered}' is {'' if ok else 'not '}among the {src} options ({', '.join(allowed)}).",
        fv,
        entered,
    )


def _check_equal(name: str, fv: FieldValue, entered: str | None, what: str) -> ConditionCheck | None:
    if fv.status == FieldStatus.NOT_STATED:
        return None
    if fv.status == FieldStatus.NEEDS_REVIEW:
        return _review(name, fv)
    want = str(fv.normalized_value)
    if want == "any":
        return _mk(name, "PASS", f"Labeling does not restrict the {what}.", fv, entered)
    if entered is None:
        return _mk(name, "INSUFFICIENT_INFO", f"Scanner profile lacks the {what}.", fv)
    ok = entered.strip().lower() == want.lower()
    return _mk(
        name,
        "PASS" if ok else "FAIL",
        f"{what[0].upper()}{what[1:]} of the scanner ('{entered}') {'matches' if ok else 'does not match'} "
        f"the labeled '{want}'.",
        fv,
        entered,
    )


def _check_mode(cs: ConditionSet, protocol: ProtocolPlan) -> ConditionCheck | None:
    fv = cs.operating_mode
    if fv.status == FieldStatus.NOT_STATED:
        return None
    if fv.status == FieldStatus.NEEDS_REVIEW:
        return _review("operating_mode", fv)
    allowed = str(fv.normalized_value)
    if allowed not in MODE_ORDER:
        return _mk("operating_mode", "NEEDS_REVIEW", "Unrecognised labeled operating mode.", fv)
    if protocol.operating_mode is None:
        return _mk("operating_mode", "INSUFFICIENT_INFO", "Enter the planned operating mode.", fv)
    ok = MODE_ORDER[protocol.operating_mode] <= MODE_ORDER[allowed]
    nice = {"normal": "Normal Operating Mode", "first_level": "First Level Controlled Operating Mode"}
    return _mk(
        "operating_mode",
        "PASS" if ok else "FAIL",
        f"Planned {nice[protocol.operating_mode]} is {'within' if ok else 'above'} the labeled maximum "
        f"({nice[allowed]}).",
        fv,
        nice[protocol.operating_mode],
    )


def _rf_checks(cs: ConditionSet, protocol: ProtocolPlan) -> list[ConditionCheck]:
    out: list[ConditionCheck] = []
    b1, wb, head, fb = cs.b1rms_max_uT, cs.wb_sar_max_w_per_kg, cs.head_sar_max_w_per_kg, cs.sar_is_b1rms_fallback
    stated = {
        n: fv
        for n, fv in (("b1rms_max_uT", b1), ("wb_sar_max_w_per_kg", wb), ("head_sar_max_w_per_kg", head))
        if fv.status != FieldStatus.NOT_STATED
    }
    for n, fv in stated.items():
        if fv.status == FieldStatus.NEEDS_REVIEW:
            out.append(_review(n, fv))
    if fb.status == FieldStatus.NEEDS_REVIEW:
        out.append(_review("sar_is_b1rms_fallback", fb))
    if out:
        return out
    if not stated:
        if cs.operating_mode.status == FieldStatus.NOT_STATED:
            out.append(
                _mk(
                    "wb_sar_max_w_per_kg",
                    "NEEDS_REVIEW",
                    "This condition set states no RF exposure limit (B1+rms, SAR or operating mode); "
                    "verify the source labeling.",
                    None,
                )
            )
        return out
    # a stated limit must be a finite number (or an explicit "no limit"); anything else cannot be compared
    unreadable = [
        _mk(n, "NEEDS_REVIEW", "Labeled limit could not be read as a number.", fv)
        for n, fv in stated.items()
        if fv.comparator != "any" and _num(fv.normalized_value) is None
    ]
    if unreadable:
        return unreadable
    seqs = protocol.sequences
    if not seqs:
        for n, fv in stated.items():
            out.append(_mk(n, "INSUFFICIENT_INFO", "Enter per-sequence B1+rms and/or SAR values.", fv))
        return out
    fallback = fb.status == FieldStatus.EXTRACTED and bool(fb.normalized_value) and "b1rms_max_uT" in stated

    def lim(fv: FieldValue) -> float | None:
        return None if fv.comparator == "any" else _num(fv.normalized_value)

    b1_lim, wb_lim, head_lim = lim(b1), lim(wb), lim(head)
    # entered values that are not finite, non-negative numbers count as not entered
    seq_b1 = {id(s): _entry(s.b1rms_uT) for s in seqs}
    seq_wb = {id(s): _entry(s.wb_sar_w_per_kg) for s in seqs}
    seq_head = {id(s): _entry(s.head_sar_w_per_kg) for s in seqs}

    if "b1rms_max_uT" in stated:
        fails, missing, notes = [], [], []
        for s in seqs:
            sb1, swb = seq_b1[id(s)], seq_wb[id(s)]
            if sb1 is not None:
                if b1_lim is not None and sb1 > b1_lim + EPS:
                    fails.append(f"{s.name}: {sb1:g} µT")
            elif fallback and wb_lim is not None and swb is not None:
                notes.append(s.name)
                if swb > wb_lim + EPS:
                    fails.append(f"{s.name}: whole-body SAR {swb:g} W/kg (B1+rms not available)")
            else:
                missing.append(s.name)
        entered = ", ".join(f"{s.name}: {seq_b1[id(s)]:g} µT" for s in seqs if seq_b1[id(s)] is not None) or None
        if fails:
            out.append(_mk("b1rms_max_uT", "FAIL", f"Exceeds labeled limit: {'; '.join(fails)}.", b1, entered))
        elif missing:
            msg = "Enter B1+rms" + (" (or whole-body SAR, labeled fallback)" if fallback else "")
            out.append(_mk("b1rms_max_uT", "INSUFFICIENT_INFO", f"{msg} for: {', '.join(missing)}.", b1, entered))
        else:
            extra = f" Whole-body SAR fallback used for: {', '.join(notes)}." if notes else ""
            out.append(_mk("b1rms_max_uT", "PASS", f"All sequences within labeled B1+rms limit.{extra}", b1, entered))

    if "wb_sar_max_w_per_kg" in stated:
        fails, missing = [], []
        for s in seqs:
            swb = seq_wb[id(s)]
            if swb is None:
                if fallback and seq_b1[id(s)] is not None:
                    continue  # SAR is only a fallback; B1+rms checked above
                missing.append(s.name)
            elif wb_lim is not None and swb > wb_lim + EPS:
                fails.append(f"{s.name}: {swb:g} W/kg")
        entered = ", ".join(f"{s.name}: {seq_wb[id(s)]:g} W/kg" for s in seqs if seq_wb[id(s)] is not None) or None
        if fails:
            out.append(_mk("wb_sar_max_w_per_kg", "FAIL", f"Exceeds labeled limit: {'; '.join(fails)}.", wb, entered))
        elif missing:
            out.append(
                _mk(
                    "wb_sar_max_w_per_kg",
                    "INSUFFICIENT_INFO",
                    f"Enter whole-body SAR for: {', '.join(missing)}.",
                    wb,
                    entered,
                )
            )
        else:
            why = (
                "Labeled as fallback when B1+rms is unavailable; entered SAR values are within the limit."
                if fallback
                else "All sequences within the labeled whole-body SAR limit."
            )
            out.append(_mk("wb_sar_max_w_per_kg", "PASS", why, wb, entered))

    if "head_sar_max_w_per_kg" in stated:
        fails, missing = [], []
        for s in seqs:
            sh = seq_head[id(s)]
            if sh is None:
                missing.append(s.name)
            elif head_lim is not None and sh > head_lim + EPS:
                fails.append(f"{s.name}: {sh:g} W/kg")
        entered = ", ".join(f"{s.name}: {seq_head[id(s)]:g} W/kg" for s in seqs if seq_head[id(s)] is not None) or None
        if fails:
            out.append(
                _mk("head_sar_max_w_per_kg", "FAIL", f"Exceeds labeled limit: {'; '.join(fails)}.", head, entered)
            )
        elif missing:
            out.append(
                _mk(
                    "head_sar_max_w_per_kg",
                    "INSUFFICIENT_INFO",
                    f"Enter head SAR for: {', '.join(missing)}.",
                    head,
                    entered,
                )
            )
        else:
            out.append(
                _mk("head_sar_max_w_per_kg", "PASS", "All sequences within the labeled head SAR limit.", head, entered)
            )
    return out


def _check_set(cs: ConditionSet, scanner: ScannerProfile, protocol: ProtocolPlan) -> ConditionSetResult:
    checks: list[ConditionCheck] = []
    b0c = _check_b0(cs, scanner)
    checks.append(b0c)
    applicable = b0c.status != "FAIL"

    c = _check_allowed("nucleus", cs.nucleus, protocol.nucleus, "imaged nucleus", implied=["1H"])
    if c:
        checks.append(c)
    for name, entered, what in (
        ("scanner_type", scanner.scanner_type, "scanner type"),
        ("b0_orientation", scanner.b0_orientation, "B0 orientation"),
    ):
        c = _check_equal(name, cs.field(name), entered, what)
        if c:
            checks.append(c)

    # spatial field gradient: position dependent -> needs review unless the value at the implant is entered
    sfg = cs.max_sfg_t_per_m
    if sfg.status == FieldStatus.NEEDS_REVIEW:
        checks.append(_review("max_sfg_t_per_m", sfg))
    elif sfg.status == FieldStatus.EXTRACTED:
        if protocol.sfg_at_implant_t_per_m is None:
            checks.append(
                _mk(
                    "max_sfg_t_per_m",
                    "NEEDS_REVIEW",
                    "Spatial field gradient depends on where the implant passes through the field. Consult "
                    "the scanner's spatial-field-gradient map and enter the value at the implant location.",
                    sfg,
                )
            )
        else:
            c = _check_upper(
                "max_sfg_t_per_m", sfg, protocol.sfg_at_implant_t_per_m, "spatial field gradient at the implant", "T/m"
            )
            if c:
                checks.append(c)

    slew = protocol.slew_rate_limit_t_per_m_per_s or scanner.max_slew_rate_t_per_m_per_s
    c = _check_upper(
        "max_slew_rate_t_per_m_per_s",
        cs.max_slew_rate_t_per_m_per_s,
        slew,
        "maximum gradient slew rate per axis",
        "scanner maximum or protocol limit, T/m/s",
    )
    if c:
        if c.status == "FAIL" and protocol.slew_rate_limit_t_per_m_per_s is None:
            c.reason += " Use a slew-rate-limited gradient mode and enter the limited value if available."
        checks.append(c)

    c = _check_allowed("rf_excitation", cs.rf_excitation, protocol.rf_excitation, "RF excitation")
    if c:
        checks.append(c)
    c = _check_allowed("transmit_coil", cs.transmit_coil, protocol.transmit_coil, "transmit coil")
    if c:
        checks.append(c)
    c = _check_allowed("receive_coil", cs.receive_coil, protocol.receive_coil, "receive coil")
    if c:
        checks.append(c)
    c = _check_mode(cs, protocol)
    if c:
        checks.append(c)
    checks.extend(_rf_checks(cs, protocol))

    c = _check_upper(
        "max_active_scan_time_min",
        cs.max_active_scan_time_min,
        protocol.total_active_scan_time_min,
        "total active scan time",
        "minutes",
    )
    if c:
        checks.append(c)

    wt = cs.wait_time_min
    if wt.status == FieldStatus.NEEDS_REVIEW:
        checks.append(_review("wait_time_min", wt))
    elif wt.status == FieldStatus.EXTRACTED and wt.comparator != "any" and _num(wt.normalized_value) is None:
        checks.append(_mk("wait_time_min", "NEEDS_REVIEW", "Labeled wait time could not be read as a number.", wt))
    elif wt.status == FieldStatus.EXTRACTED and _num(wt.normalized_value) is not None:
        need = float(wt.normalized_value)  # type: ignore[arg-type]
        wait = _entry(protocol.wait_between_sessions_min)
        if not isinstance(protocol.sessions, int) or protocol.sessions < 1:
            checks.append(
                _mk(
                    "wait_time_min",
                    "NEEDS_REVIEW",
                    "The number of imaging sessions is not valid.",
                    wt,
                    str(protocol.sessions),
                )
            )
        elif protocol.sessions == 1:
            checks.append(
                _mk(
                    "wait_time_min",
                    "PASS",
                    f"Single session planned; observe a wait of at least {need:g} min before any further "
                    "imaging session.",
                    wt,
                    "1 session",
                )
            )
        elif wait is None:
            checks.append(_mk("wait_time_min", "INSUFFICIENT_INFO", "Enter the wait time between sessions.", wt))
        else:
            ok = wait + EPS >= need
            checks.append(
                _mk(
                    "wait_time_min",
                    "PASS" if ok else "FAIL",
                    f"Planned wait {protocol.wait_between_sessions_min:g} min "
                    f"{'≥' if ok else '<'} required {need:g} min.",
                    wt,
                    f"{protocol.wait_between_sessions_min:g} min",
                )
            )

    reg = cs.scan_region_restrictions
    if reg.status == FieldStatus.NEEDS_REVIEW:
        checks.append(_review("scan_region_restrictions", reg))
    elif reg.status == FieldStatus.EXTRACTED:
        if str(reg.normalized_value) == "unrestricted":
            checks.append(
                _mk(
                    "scan_region_restrictions",
                    "PASS",
                    "Labeling states no scan-region restriction.",
                    reg,
                    protocol.scan_region,
                )
            )
        else:
            checks.append(
                _mk(
                    "scan_region_restrictions",
                    "NEEDS_REVIEW",
                    f"Scan-region/landmark restriction must be checked by a person against the planned "
                    f'region ({protocol.scan_region or "not entered"}): "{reg.normalized_value}".',
                    reg,
                    protocol.scan_region,
                )
            )

    dev = cs.device_configuration
    if dev.status == FieldStatus.NEEDS_REVIEW:
        checks.append(_review("device_configuration", dev))
    elif dev.status == FieldStatus.EXTRACTED:
        if protocol.device_configuration_confirmed is True:
            checks.append(
                _mk(
                    "device_configuration",
                    "PASS",
                    f'User confirmed the device is configured as labeled: "{dev.normalized_value}".',
                    dev,
                    "confirmed",
                )
            )
        elif protocol.device_configuration_confirmed is False:
            checks.append(
                _mk(
                    "device_configuration",
                    "FAIL",
                    f'User indicated the device is not configured as the labeling requires: "{dev.normalized_value}".',
                    dev,
                    "not configured",
                )
            )
        else:
            checks.append(
                _mk(
                    "device_configuration",
                    "INSUFFICIENT_INFO",
                    f'Confirm the device configuration required by the labeling: "{dev.normalized_value}".',
                    dev,
                )
            )

    oth = cs.other_conditions
    if oth.status in (FieldStatus.EXTRACTED, FieldStatus.NEEDS_REVIEW):
        checks.append(
            _mk(
                "other_conditions",
                "NEEDS_REVIEW",
                f'Additional labeled conditions must be reviewed by a person: "{oth.normalized_value or oth.value}".',
                oth,
            )
        )

    notes = []
    art = cs.artifact_extent_mm
    if art.status == FieldStatus.EXTRACTED and _num(art.normalized_value) is not None:
        notes.append(f"Image artifact may extend about {float(art.normalized_value):g} mm from the device.")  # type: ignore[arg-type]

    status = worst([c.status for c in checks]) if applicable else "FAIL"
    return ConditionSetResult(
        set_id=cs.set_id, scope=cs.scope, applicable=applicable, status=status, checks=checks, notes=notes
    )


def _check_implant(doc: LabelDocument, scanner: ScannerProfile, protocol: ProtocolPlan) -> ImplantResult:
    implant_checks: list[ConditionCheck] = []
    verification = doc.provenance.verification
    if verification != "human-verified":
        implant_checks.append(
            ConditionCheck(
                field="verification",
                label="Human verification",
                status="NEEDS_REVIEW",
                reason=f"Labeling record is '{verification}', not human-verified; verify every value against the source "
                "labeling first.",
            )
        )
    status = doc.status_value()
    st_fv = doc.mr_status
    set_results: list[ConditionSetResult] = []
    selected: str | None = None
    selected_status: CheckStatus = "NEEDS_REVIEW"
    if st_fv.status == FieldStatus.NEEDS_REVIEW:
        implant_checks.append(
            ConditionCheck(
                field="mr_status",
                label="MR status",
                status="NEEDS_REVIEW",
                reason="MR status needs human review.",
                evidence=st_fv.evidence,
            )
        )
    elif status == MRStatus.MR_UNSAFE:
        implant_checks.append(
            ConditionCheck(
                field="mr_status",
                label="MR status",
                status="FAIL",
                reason="Device is labeled MR Unsafe.",
                label_value="MR Unsafe",
                evidence=st_fv.evidence,
            )
        )
    elif status == MRStatus.NOT_STATED:
        implant_checks.append(
            ConditionCheck(
                field="mr_status",
                label="MR status",
                status="NEEDS_REVIEW",
                reason="No MR safety status found in the labeling.",
                label_value="Not stated",
                evidence=st_fv.evidence,
            )
        )
    elif status == MRStatus.MR_SAFE:
        if doc.condition_sets:
            # contradictory record: MR Safe items have no conditions; a person must decide which applies
            implant_checks.append(
                ConditionCheck(
                    field="mr_status",
                    label="MR status",
                    status="NEEDS_REVIEW",
                    reason="Labeled MR Safe but MR conditions are recorded; review the labeling.",
                    label_value="MR Safe",
                    evidence=st_fv.evidence,
                )
            )
        else:
            implant_checks.append(
                ConditionCheck(
                    field="mr_status",
                    label="MR status",
                    status="PASS",
                    reason="Device is labeled MR Safe (no MR conditions).",
                    label_value="MR Safe",
                    evidence=st_fv.evidence,
                )
            )
    else:  # MR Conditional
        if not doc.condition_sets:
            implant_checks.append(
                ConditionCheck(
                    field="mr_status",
                    label="MR status",
                    status="NEEDS_REVIEW",
                    reason="Labeled MR Conditional but no conditions are recorded.",
                    label_value="MR Conditional",
                    evidence=st_fv.evidence,
                )
            )
        set_results = [_check_set(cs, scanner, protocol) for cs in doc.condition_sets]
        applicable = [r for r in set_results if r.applicable]
        if doc.condition_sets and not applicable:
            implant_checks.append(
                ConditionCheck(
                    field="b0_tesla",
                    label=FIELD_SPECS["b0_tesla"].label,
                    status="FAIL",
                    reason=f"Scanner field strength {scanner.b0_tesla:g} T is not permitted by any condition set.",
                    entered_value=f"{scanner.b0_tesla:g} T",
                )
            )
        elif applicable:
            best = min(applicable, key=lambda r: (RANK[r.status], sum(c.status != "PASS" for c in r.checks)))
            selected = best.set_id
            selected_status = best.status
    statuses = [c.status for c in implant_checks]
    if selected is not None:
        statuses.append(selected_status)
    return ImplantResult(
        implant_id=doc.doc_id,
        device_name=doc.device_name,
        mr_status=status.value,
        verification=verification,
        status=worst(statuses),
        selected_set_id=selected,
        implant_checks=implant_checks,
        set_results=set_results,
    )


def check(implants: list[LabelDocument], scanner: ScannerProfile, protocol: ProtocolPlan) -> CheckResult:
    """Evaluate every implant against the scanner profile and planned protocol."""
    results = [_check_implant(doc, scanner, protocol) for doc in implants]
    statuses = [r.status for r in results]
    if not results:
        outcome = "NEEDS_REVIEW"
    elif "FAIL" in statuses:
        outcome = "CONFLICT_FOUND"
    elif any(s != "PASS" for s in statuses):
        outcome = "NEEDS_REVIEW"
    else:
        outcome = "NO_CONFLICT_FOUND"
    return CheckResult(
        outcome=outcome, outcome_text=OUTCOME_TEXT[outcome], implants=results, generated_at=datetime.now(UTC)
    )
