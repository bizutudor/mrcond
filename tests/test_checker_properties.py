"""Randomised safety test of the checker against an independent oracle.

The oracle below re-implements the documented labeling semantics directly from the inputs (README "Safety
design rules"; FDA 2023 guidance: a parameter that is not listed carries no condition, an unlisted nucleus
means 1H only). It does not call or import checker internals. For thousands of seeded random combinations of
labels (including flagged, missing and malformed values, several condition sets and several implants) and
exams (values exactly at, just within float noise of, and clearly beyond each limit, or not entered):

* safety property: the checker never returns NO_CONFLICT_FOUND when the oracle expects CONFLICT_FOUND or
  NEEDS_REVIEW;
* exactness: checker and oracle agree on the overall outcome (so over-conservative drift is caught too);
* wording: only the three permitted outcome texts, and no reason claims that something is safe, cleared
  or approved.
"""

from __future__ import annotations

import math
import random
import re

import pytest

from mrcond.checker import check
from mrcond.schema import (
    ConditionSet,
    Evidence,
    FieldStatus,
    FieldValue,
    LabelDocument,
    ProtocolPlan,
    Provenance,
    ScannerProfile,
)

PASS, REVIEW, FAIL = 0, 1, 2  # oracle levels (INSUFFICIENT_INFO counts as REVIEW)
OUTCOME = {PASS: "NO_CONFLICT_FOUND", REVIEW: "NEEDS_REVIEW", FAIL: "CONFLICT_FOUND"}
ALLOWED_TEXTS = {
    "No conflict found with entered conditions — requires MR Safety Officer/Expert sign-off",
    "Conflict found",
    "Insufficient information / needs review",
}
SAFETY_WORDS = re.compile(r"\bsafe(ly)?\b|\bcleared\b|\bapproved\b", re.IGNORECASE)

NUMERIC_LIMITS = {
    "max_sfg_t_per_m": [3.0, 7.2, 25.0, 30.0],
    "max_slew_rate_t_per_m_per_s": [100.0, 150.0, 200.0],
    "b1rms_max_uT": [0.3, 1.0, 1.7, 2.0, 2.8],
    "wb_sar_max_w_per_kg": [0.1, 0.4, 1.2, 2.0, 4.0],
    "head_sar_max_w_per_kg": [0.4, 1.0, 3.2],
    "max_active_scan_time_min": [10.0, 15.0, 30.0, 60.0],
    "wait_time_min": [5.0, 30.0, 60.0],
}
UPPER = [f for f in NUMERIC_LIMITS if f != "wait_time_min"]
MODE_RANK = {"normal": 0, "first_level": 1}


# --------------------------------------------------------------------------------------------------------
# oracle
# --------------------------------------------------------------------------------------------------------


def _number(v) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return None
    return float(v)


def _at_most(entered: float, limit: float) -> bool:
    return entered < limit or math.isclose(entered, limit, rel_tol=0.0, abs_tol=1e-9)


def _tokens(v) -> set[str]:
    return {str(x) for x in (v if isinstance(v, list) else [v])}


def _usable(x) -> float | None:
    """An entered quantity counts only if it is a finite, non-negative number."""
    n = _number(x)
    return n if n is not None and n >= 0 else None


def _upper(fv: FieldValue, entered) -> int | None:
    """Upper-limit condition. None: no condition stated."""
    if fv.status == FieldStatus.NOT_STATED:
        return None
    if fv.status == FieldStatus.NEEDS_REVIEW:
        return REVIEW
    if fv.comparator == "any":
        return PASS
    limit = _number(fv.normalized_value)
    if limit is None:
        return REVIEW
    e = _usable(entered)
    if e is None:
        return REVIEW
    return PASS if _at_most(e, limit) else FAIL


def _allowed(fv: FieldValue, entered: str | None, implied: set[str] | None = None) -> int | None:
    if fv.status == FieldStatus.NEEDS_REVIEW:
        return REVIEW
    if fv.status == FieldStatus.NOT_STATED:
        if implied is None:
            return None
        allowed = implied
    else:
        allowed = _tokens(fv.normalized_value)
    if "any" in allowed:
        return PASS
    if entered is None:
        return REVIEW
    return PASS if entered in allowed else FAIL


def _equal(fv: FieldValue, entered: str | None) -> int | None:
    if fv.status == FieldStatus.NOT_STATED:
        return None
    if fv.status == FieldStatus.NEEDS_REVIEW:
        return REVIEW
    want = str(fv.normalized_value)
    if want == "any":
        return PASS
    if entered is None:
        return REVIEW
    return PASS if entered.lower() == want.lower() else FAIL


def _rf(cs: ConditionSet, p: ProtocolPlan) -> list[int]:
    b1, wb, head, fb = cs.b1rms_max_uT, cs.wb_sar_max_w_per_kg, cs.head_sar_max_w_per_kg, cs.sar_is_b1rms_fallback
    stated = {k: fv for k, fv in (("b1", b1), ("wb", wb), ("head", head)) if fv.status != FieldStatus.NOT_STATED}
    if any(fv.status == FieldStatus.NEEDS_REVIEW for fv in stated.values()) or fb.status == FieldStatus.NEEDS_REVIEW:
        return [REVIEW]
    if not stated:
        # no RF limit at all is acceptable only when an operating mode limits RF exposure
        return [REVIEW] if cs.operating_mode.status == FieldStatus.NOT_STATED else []
    if any(fv.comparator != "any" and _number(fv.normalized_value) is None for fv in stated.values()):
        return [REVIEW]
    if not p.sequences:
        return [REVIEW]
    fallback = fb.status == FieldStatus.EXTRACTED and fb.normalized_value is True and "b1" in stated

    def limit(fv):
        return None if fv.comparator == "any" else _number(fv.normalized_value)

    out = []
    for s in p.sequences:
        sb1, swb, shead = _usable(s.b1rms_uT), _usable(s.wb_sar_w_per_kg), _usable(s.head_sar_w_per_kg)
        if "b1" in stated:
            if sb1 is not None:
                lim = limit(b1)
                out.append(PASS if lim is None or _at_most(sb1, lim) else FAIL)
            elif fallback and "wb" in stated and limit(wb) is not None and swb is not None:
                out.append(PASS if _at_most(swb, limit(wb)) else FAIL)
            else:
                out.append(REVIEW)
        if "wb" in stated:
            if swb is None:
                out.append(PASS if fallback and sb1 is not None else REVIEW)
            else:
                lim = limit(wb)
                out.append(PASS if lim is None or _at_most(swb, lim) else FAIL)
        if "head" in stated:
            if shead is None:
                out.append(REVIEW)
            else:
                lim = limit(head)
                out.append(PASS if lim is None or _at_most(shead, lim) else FAIL)
    return out


def oracle_set(cs: ConditionSet, sc: ScannerProfile, p: ProtocolPlan) -> tuple[bool, int]:
    """(applicable, level) of one alternative condition set."""
    b0 = cs.b0_tesla
    if b0.status != FieldStatus.EXTRACTED:
        b0_level = REVIEW
    else:
        raw = b0.normalized_value if isinstance(b0.normalized_value, list) else [b0.normalized_value]
        vals = [_number(v) for v in raw]
        if not vals or any(v is None or v <= 0 for v in vals):
            b0_level = REVIEW
        elif b0.comparator == "le":
            b0_level = PASS if _at_most(sc.b0_tesla, max(vals)) else FAIL
        else:  # nominal field strengths; generated scanners are never near the +-0.05 T edge
            b0_level = PASS if any(abs(sc.b0_tesla - v) <= 0.05 for v in vals) else FAIL
    if b0_level == FAIL:
        return False, FAIL
    levels = [b0_level]

    def add(x):
        if x is not None:
            levels.append(x)

    add(_allowed(cs.nucleus, p.nucleus, implied={"1H"}))
    add(_equal(cs.scanner_type, sc.scanner_type))
    add(_equal(cs.b0_orientation, sc.b0_orientation))
    sfg = cs.max_sfg_t_per_m
    if sfg.status == FieldStatus.EXTRACTED and p.sfg_at_implant_t_per_m is None:
        add(REVIEW)  # position-dependent: never assumed
    else:
        add(_upper(sfg, p.sfg_at_implant_t_per_m))
    add(_upper(cs.max_slew_rate_t_per_m_per_s, p.slew_rate_limit_t_per_m_per_s or sc.max_slew_rate_t_per_m_per_s))
    add(_allowed(cs.rf_excitation, p.rf_excitation))
    add(_allowed(cs.transmit_coil, p.transmit_coil))
    add(_allowed(cs.receive_coil, p.receive_coil))
    mode = cs.operating_mode
    if mode.status == FieldStatus.NEEDS_REVIEW:
        add(REVIEW)
    elif mode.status == FieldStatus.EXTRACTED:
        if str(mode.normalized_value) not in MODE_RANK:
            add(REVIEW)
        elif p.operating_mode is None:
            add(REVIEW)
        else:
            add(PASS if MODE_RANK[p.operating_mode] <= MODE_RANK[str(mode.normalized_value)] else FAIL)
    levels += _rf(cs, p)
    add(_upper(cs.max_active_scan_time_min, p.total_active_scan_time_min))
    wt = cs.wait_time_min
    if wt.status == FieldStatus.NEEDS_REVIEW:
        add(REVIEW)
    elif wt.status == FieldStatus.EXTRACTED and wt.comparator != "any":
        need = _number(wt.normalized_value)
        if need is None:
            add(REVIEW)
        elif p.sessions == 1:
            add(PASS)
        elif _usable(p.wait_between_sessions_min) is None:
            add(REVIEW)
        else:
            w = _usable(p.wait_between_sessions_min)
            add(PASS if w > need or math.isclose(w, need, abs_tol=1e-9) else FAIL)
    reg = cs.scan_region_restrictions
    if reg.status == FieldStatus.NEEDS_REVIEW:
        add(REVIEW)
    elif reg.status == FieldStatus.EXTRACTED:
        add(PASS if reg.normalized_value == "unrestricted" else REVIEW)
    dev = cs.device_configuration
    if dev.status == FieldStatus.NEEDS_REVIEW:
        add(REVIEW)
    elif dev.status == FieldStatus.EXTRACTED:
        add({True: PASS, False: FAIL, None: REVIEW}[p.device_configuration_confirmed])
    if cs.other_conditions.status != FieldStatus.NOT_STATED:
        add(REVIEW)
    return True, max(levels)


def oracle_implant(doc: LabelDocument, sc: ScannerProfile, p: ProtocolPlan) -> int:
    levels = [PASS if doc.provenance.verification == "human-verified" else REVIEW]
    st = doc.mr_status
    status = st.normalized_value if st.status == FieldStatus.EXTRACTED else None
    if st.status == FieldStatus.NEEDS_REVIEW or status not in ("MR Conditional", "MR Safe", "MR Unsafe"):
        levels.append(REVIEW)
    elif status == "MR Unsafe":
        levels.append(FAIL)
    elif status == "MR Safe":
        levels.append(REVIEW if doc.condition_sets else PASS)
    elif not doc.condition_sets:
        levels.append(REVIEW)
    else:
        results = [oracle_set(cs, sc, p) for cs in doc.condition_sets]
        applicable = [lvl for ok, lvl in results if ok]
        levels.append(min(applicable) if applicable else FAIL)  # alternatives: the best applicable set
    return max(levels)


def oracle(implants: list[LabelDocument], sc: ScannerProfile, p: ProtocolPlan) -> str:
    if not implants:
        return OUTCOME[REVIEW]
    return OUTCOME[max(oracle_implant(d, sc, p) for d in implants)]


# --------------------------------------------------------------------------------------------------------
# random cases
# --------------------------------------------------------------------------------------------------------


def _ev(q: str) -> list[Evidence]:
    return [Evidence(doc_id="prop", page=1, char_start=0, char_end=len(q), quote=q)]


def _fv(norm, comparator: str | None = None, status: FieldStatus = FieldStatus.EXTRACTED) -> FieldValue:
    return FieldValue(
        status=status,
        value=norm,
        normalized_value=norm,
        comparator=comparator,
        evidence=_ev(str(norm)) if status != FieldStatus.NOT_STATED else [],
    )


def _flag() -> FieldValue:
    return FieldValue(status=FieldStatus.NEEDS_REVIEW, review_reasons=["random flag"], evidence=_ev("x"))


def _random_numeric(rng: random.Random, name: str) -> FieldValue:
    r = rng.random()
    if r < 0.45:
        return FieldValue()
    if r < 0.52:
        return _flag()
    if r < 0.56:  # malformed: stated limit that is not a number
        return _fv(rng.choice([None, "two", "2 W/kg"]), "le" if name != "wait_time_min" else "ge")
    if r < 0.61:
        return FieldValue(
            status=FieldStatus.EXTRACTED,
            value="no limit",
            normalized_value=None,
            comparator="any",
            evidence=_ev("no limit"),
        )
    return _fv(rng.choice(NUMERIC_LIMITS[name]), "ge" if name == "wait_time_min" else "le")


def _random_cat(rng: random.Random, options: list) -> FieldValue:
    r = rng.random()
    if r < 0.5:
        return FieldValue()
    if r < 0.57:
        return _flag()
    v = rng.choice(options)
    return _fv(v, "eq" if isinstance(v, str) else "in")


def random_set(rng: random.Random, i: int) -> ConditionSet:
    r = rng.random()
    if r < 0.06:
        b0 = FieldValue()
    elif r < 0.11:
        b0 = _flag()
    elif r < 0.25:
        b0 = _fv([rng.choice([1.5, 3.0])], "le")
    else:
        b0 = _fv(rng.choice([[1.5], [3.0], [1.5, 3.0]]), "in")
    cs = ConditionSet(set_id=f"cs{i}", b0_tesla=b0)
    for name in NUMERIC_LIMITS:
        cs.set_field(name, _random_numeric(rng, name))
    cs.nucleus = _random_cat(rng, [["1H"], ["1H", "31P"], ["any"]])
    cs.scanner_type = _random_cat(rng, ["cylindrical", "open", "any"])
    cs.b0_orientation = _random_cat(rng, ["horizontal", "vertical"])
    cs.rf_excitation = _random_cat(rng, [["CP"], ["CP", "MC-2"], ["any"]])
    cs.transmit_coil = _random_cat(rng, [["body"], ["head"], ["body", "head"], ["any"]])
    cs.receive_coil = _random_cat(rng, [["any"], ["head"], ["body", "local"]])
    m = rng.random()
    cs.operating_mode = (
        FieldValue()
        if m < 0.35
        else _flag()
        if m < 0.4
        else _fv("turbo", "le")
        if m < 0.43
        else _fv(rng.choice(["normal", "first_level"]), "le")
    )
    f = rng.random()
    cs.sar_is_b1rms_fallback = FieldValue() if f < 0.5 else _flag() if f < 0.54 else _fv(f < 0.85, "eq")
    g = rng.random()
    cs.scan_region_restrictions = (
        FieldValue()
        if g < 0.6
        else _flag()
        if g < 0.64
        else _fv("unrestricted", "eq")
        if g < 0.85
        else _fv("Head Only", "eq")
    )
    d = rng.random()
    cs.device_configuration = FieldValue() if d < 0.6 else _flag() if d < 0.64 else _fv("Stimulation OFF", "eq")
    o = rng.random()
    cs.other_conditions = FieldValue() if o < 0.85 else _fv("Patient temperature below 39 °C", "eq")
    if rng.random() < 0.3:
        cs.artifact_extent_mm = _fv(10.0, "eq")
    return cs


def random_label(rng: random.Random, k: int) -> LabelDocument:
    r = rng.random()
    if r < 0.8:
        status = _fv("MR Conditional", "eq")
    elif r < 0.85:
        status = _fv("MR Safe", "eq")
    elif r < 0.9:
        status = _fv("MR Unsafe", "eq")
    elif r < 0.95:
        status = FieldValue()
    else:
        status = _flag()
    n_sets = rng.choice([0, 1, 1, 1, 2, 2, 3]) if status.normalized_value != "MR Safe" or rng.random() < 0.3 else 0
    verification = "human-verified" if rng.random() < 0.85 else rng.choice(["draft-unverified", "machine-verified"])
    return LabelDocument(
        doc_id=f"imp{k}",
        mr_status=status,
        condition_sets=[random_set(rng, i) for i in range(n_sets)],
        provenance=Provenance(extractor="property-test", verification=verification),
    )


def _entered(rng: random.Random, limit: float | None, low_ok: bool = True) -> float | None:
    """A value relative to a labeled limit: at it (also with float noise), clearly within, clearly beyond, or
    not entered. Never within (1e-9, 1e-3] of a limit, so there is no ambiguous boundary."""
    if limit is None or rng.random() < 0.12:
        return None if rng.random() < 0.5 else rng.choice([0.0, 0.5, 1.0, 2.0, 8.0])
    r = rng.random()
    if r < 0.3:
        return limit
    if r < 0.4:
        return math.nextafter(limit, math.inf)  # float noise above the limit
    if r < 0.5:
        return (limit * 10 + 1) / 10 - 0.1  # arithmetic noise around the limit
    if r < 0.75:
        return round(limit * rng.choice([0.25, 0.5, 0.9]), 6) if low_ok else limit
    return round(limit * rng.choice([1.01, 1.5, 3.0]) + 0.01, 6)


def random_exam(rng: random.Random, labels: list[LabelDocument]) -> tuple[ScannerProfile, ProtocolPlan]:
    sets = [cs for lab in labels for cs in lab.condition_sets]
    ref = rng.choice(sets) if sets else ConditionSet()

    def lim(name):
        fv = ref.field(name)
        return _number(fv.normalized_value) if fv.status == FieldStatus.EXTRACTED else None

    scanner = ScannerProfile(
        profile_id="p",
        name="p",
        b0_tesla=rng.choice([1.5, 1.5, 3.0, 3.0, 1.494, 2.89, 0.55, 1.0, 7.0]),
        scanner_type=rng.choice(["cylindrical", "cylindrical", "open", None]),
        b0_orientation=rng.choice(["horizontal", "horizontal", "vertical", None]),
        max_slew_rate_t_per_m_per_s=_entered(rng, lim("max_slew_rate_t_per_m_per_s")) or None,
    )

    def tok(fv: FieldValue, pool: list[str]) -> str | None:
        if rng.random() < 0.1:
            return None
        allowed = sorted(_tokens(fv.normalized_value) - {"any"}) if fv.status == FieldStatus.EXTRACTED else []
        return rng.choice(allowed) if allowed and rng.random() < 0.75 else rng.choice(pool)

    seqs = []
    for i in range(rng.choice([0, 1, 1, 2, 3])):
        seqs.append(
            {
                "name": f"s{i}",
                "b1rms_uT": _entered(rng, lim("b1rms_max_uT")) if rng.random() < 0.7 else None,
                "wb_sar_w_per_kg": _entered(rng, lim("wb_sar_max_w_per_kg")) if rng.random() < 0.75 else None,
                "head_sar_w_per_kg": _entered(rng, lim("head_sar_max_w_per_kg")) if rng.random() < 0.6 else None,
            }
        )
    sessions = rng.choice([1, 1, 1, 2, 3])
    mode = rng.choice(["normal", "normal", "first_level", None])
    slew_limit = _entered(rng, lim("max_slew_rate_t_per_m_per_s")) if rng.random() < 0.2 else None
    protocol = ProtocolPlan(
        operating_mode=mode,
        transmit_coil=tok(ref.transmit_coil, ["body", "head", "local"]),
        receive_coil=tok(ref.receive_coil, ["body", "head", "local"]),
        rf_excitation=tok(ref.rf_excitation, ["CP", "MC-2"]),
        nucleus=tok(ref.nucleus, ["1H", "1H", "31P", "23Na"]),
        sequences=seqs,
        total_active_scan_time_min=_entered(rng, lim("max_active_scan_time_min")),
        sessions=sessions,
        wait_between_sessions_min=_entered(rng, lim("wait_time_min")) if sessions > 1 else None,
        slew_rate_limit_t_per_m_per_s=slew_limit or None,
        sfg_at_implant_t_per_m=_entered(rng, lim("max_sfg_t_per_m")) if rng.random() < 0.8 else None,
        device_configuration_confirmed=rng.choice([True, True, None, False]),
        scan_region=rng.choice([None, "head", "lumbar spine"]),
    )
    return scanner, protocol


def clean_set(rng: random.Random, i: int) -> ConditionSet:
    """A well-formed condition set without flags, so that compliant exams reach "No conflict"."""
    b0 = (
        _fv([3.0 if rng.random() < 0.5 else 1.5], "le")
        if rng.random() < 0.15
        else _fv(rng.choice([[1.5], [3.0], [1.5, 3.0]]), "in")
    )
    cs = ConditionSet(set_id=f"cs{i}", b0_tesla=b0)
    for name, vals in NUMERIC_LIMITS.items():
        r = rng.random()
        if r < 0.5:
            continue
        if r < 0.55:
            cs.set_field(
                name,
                FieldValue(
                    status=FieldStatus.EXTRACTED,
                    value="no limit",
                    normalized_value=None,
                    comparator="any",
                    evidence=_ev("no limit"),
                ),
            )
        else:
            cs.set_field(name, _fv(rng.choice(vals), "ge" if name == "wait_time_min" else "le"))
    for name, options in (
        ("nucleus", [["1H"], ["1H", "31P"], ["any"]]),
        ("scanner_type", ["cylindrical", "any"]),
        ("b0_orientation", ["horizontal"]),
        ("rf_excitation", [["CP"], ["CP", "MC-2"]]),
        ("transmit_coil", [["body"], ["head"], ["body", "head"], ["any"]]),
        ("receive_coil", [["any"], ["head", "body", "local"]]),
    ):
        if rng.random() < 0.4:
            v = rng.choice(options)
            cs.set_field(name, _fv(v, "eq" if isinstance(v, str) else "in"))
    if rng.random() < 0.75:
        cs.operating_mode = _fv(rng.choice(["normal", "first_level"]), "le")
    if rng.random() < 0.3:
        cs.sar_is_b1rms_fallback = _fv(True, "eq")
    if rng.random() < 0.2:
        cs.scan_region_restrictions = _fv("unrestricted", "eq")
    if rng.random() < 0.3:
        cs.device_configuration = _fv("Stimulation OFF", "eq")
    return cs


def compliant_case(rng: random.Random) -> tuple[list[LabelDocument], ScannerProfile, ProtocolPlan]:
    """Implants with clean labels and an exam built to meet one set of each, then (often) one perturbation."""
    labels = [
        LabelDocument(
            doc_id=f"imp{k}",
            mr_status=_fv("MR Conditional", "eq"),
            condition_sets=[clean_set(rng, i) for i in range(rng.choice([1, 1, 2]))],
            provenance=Provenance(extractor="property-test", verification="human-verified"),
        )
        for k in range(rng.choice([1, 1, 1, 2]))
    ]
    refs = [rng.choice(lab.condition_sets) for lab in labels]

    def limits(name):
        return [
            _number(cs.field(name).normalized_value)
            for cs in refs
            if cs.field(name).status == FieldStatus.EXTRACTED and cs.field(name).comparator != "any"
        ]

    def entered(name, default):
        lims = limits(name)
        if not lims:
            return default
        v = min(lims)
        return v if rng.random() < 0.5 else round(v * 0.5, 6)

    permitted = None
    for cs in refs:
        vals = cs.b0_tesla.normalized_value
        ok = {x for x in (1.0, 1.5, 3.0) if x <= max(vals) + 1e-9} if cs.b0_tesla.comparator == "le" else set(vals)
        permitted = ok if permitted is None else permitted & ok
    b0 = rng.choice(sorted(permitted)) if permitted else 3.0

    def token(name, default):
        allowed = None
        for cs in refs:
            fv = cs.field(name)
            toks = (
                _tokens(fv.normalized_value)
                if fv.status == FieldStatus.EXTRACTED
                else ({"1H"} if name == "nucleus" else {"any"})
            )
            if "any" in toks:
                continue
            allowed = toks if allowed is None else allowed & toks
        if allowed is None:
            return default
        return rng.choice(sorted(allowed)) if allowed else default

    modes = [
        str(cs.operating_mode.normalized_value) for cs in refs if cs.operating_mode.status == FieldStatus.EXTRACTED
    ]
    seqs = [
        {
            "name": f"s{i}",
            "b1rms_uT": entered("b1rms_max_uT", 1.0),
            "wb_sar_w_per_kg": entered("wb_sar_max_w_per_kg", 0.1),
            "head_sar_w_per_kg": entered("head_sar_max_w_per_kg", 0.4),
        }
        for i in range(rng.choice([1, 2]))
    ]
    sessions = rng.choice([1, 1, 2])
    waits = limits("wait_time_min")
    scanner = dict(
        profile_id="p",
        name="p",
        b0_tesla=b0,
        scanner_type="cylindrical",
        b0_orientation="horizontal",
        max_slew_rate_t_per_m_per_s=entered("max_slew_rate_t_per_m_per_s", 200.0),
    )
    protocol = dict(
        operating_mode="normal" if "normal" in modes or not modes or rng.random() < 0.5 else "first_level",
        transmit_coil=token("transmit_coil", "body"),
        receive_coil=token("receive_coil", "local"),
        rf_excitation=token("rf_excitation", "CP"),
        nucleus=token("nucleus", "1H"),
        sequences=seqs,
        total_active_scan_time_min=entered("max_active_scan_time_min", 10.0),
        sessions=sessions,
        wait_between_sessions_min=(max(waits) if waits else 5.0) if sessions > 1 else None,
        sfg_at_implant_t_per_m=entered("max_sfg_t_per_m", 1.0),
        device_configuration_confirmed=True,
    )
    r = rng.random()
    if r < 0.4:  # one perturbation
        kind = rng.choice(["over", "missing", "mode", "token", "b0", "config", "seq-missing", "noise"])
        if kind == "over":
            key = rng.choice(["total_active_scan_time_min", "sfg_at_implant_t_per_m"])
            protocol[key] = round((protocol[key] or 0) * 1.5 + 0.01, 6)
        elif kind == "missing":
            protocol[
                rng.choice(
                    [
                        "total_active_scan_time_min",
                        "sfg_at_implant_t_per_m",
                        "operating_mode",
                        "nucleus",
                        "transmit_coil",
                        "rf_excitation",
                    ]
                )
            ] = None
        elif kind == "mode":
            protocol["operating_mode"] = "first_level"
        elif kind == "token":
            protocol[rng.choice(["transmit_coil", "receive_coil", "rf_excitation", "nucleus"])] = rng.choice(
                ["extremity", "MC-2", "23Na", "head"]
            )
        elif kind == "b0":
            scanner["b0_tesla"] = rng.choice([0.55, 7.0, 2.89, 1.494])
        elif kind == "config":
            protocol["device_configuration_confirmed"] = rng.choice([None, False])
        elif kind == "seq-missing":
            protocol["sequences"][0][rng.choice(["b1rms_uT", "wb_sar_w_per_kg", "head_sar_w_per_kg"])] = None
        else:  # float noise exactly at the limits
            for sq in protocol["sequences"]:
                for k in ("b1rms_uT", "wb_sar_w_per_kg", "head_sar_w_per_kg"):
                    if sq[k]:
                        sq[k] = math.nextafter(sq[k], math.inf)
    elif r < 0.5:
        sc2 = (
            dict(scanner, scanner_type=None) if rng.random() < 0.5 else dict(scanner, max_slew_rate_t_per_m_per_s=None)
        )
        scanner = sc2
    return labels, ScannerProfile(**scanner), ProtocolPlan(**protocol)


def _check_wording(res) -> None:
    assert res.outcome_text in ALLOWED_TEXTS
    texts = [res.outcome_text]
    for imp in res.implants:
        for c in imp.implant_checks + [c for s in imp.set_results for c in s.checks]:
            texts.append(c.reason)
    for t in texts:
        cleaned = t.replace("MR Safety", "").replace("MR Safe", "")
        assert not SAFETY_WORDS.search(cleaned), t
    assert not SAFETY_WORDS.search(res.disclaimer.replace("MR Safety", "").replace("MRI safety", ""))


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_checker_never_more_permissive_than_oracle(seed):
    rng = random.Random(20261015 + seed)
    n_cases = 1500
    seen = {"NO_CONFLICT_FOUND": 0, "CONFLICT_FOUND": 0, "NEEDS_REVIEW": 0}
    for case in range(n_cases):
        if case % 2:
            labels, scanner, protocol = compliant_case(rng)
        else:
            n_imp = rng.choice([0, 1, 1, 1, 1, 2, 2, 3]) if case % 10 else 1
            labels = [random_label(rng, k) for k in range(n_imp)]
            scanner, protocol = random_exam(rng, labels)
        res = check(labels, scanner, protocol)
        want = oracle(labels, scanner, protocol)
        seen[want] += 1
        context = f"seed={seed} case={case} checker={res.outcome} oracle={want}"
        # the safety property
        assert not (res.outcome == "NO_CONFLICT_FOUND" and want != "NO_CONFLICT_FOUND"), context
        # exact agreement (catches over-conservative drift and oracle mistakes)
        assert res.outcome == want, context
        # multiple implants: every implant must pass
        if res.outcome == "NO_CONFLICT_FOUND":
            assert res.implants and all(i.status == "PASS" for i in res.implants), context
        _check_wording(res)
    # the generator must exercise all three outcomes substantially
    assert min(seen.values()) >= 100, seen
