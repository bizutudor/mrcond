"""Conservative merging of several statements that constrain the same field.

Rules (most conservative wins):

* numeric upper limits (SAR, B1+rms, spatial field gradient, slew rate, scan
  time): minimum value; an explicit "no limit" statement never relaxes a limit
* numeric lower limits (wait time): maximum value
* image artifact extent (informational): maximum value
* allowed sets (B0 list, nuclei, transmit/receive coils, RF excitation):
  intersection ("any" acts as the universal set); an empty intersection is a
  conflict -> NEEDS_REVIEW
* operating mode: Normal < First Level -> the lowest permitted mode
* scanner type / B0 orientation: must agree, otherwise NEEDS_REVIEW
* "SAR applies only as B1+rms fallback": True only if every statement says so
* free text: union of statements; for scan regions an "any landmark"
  statement never overrides a restriction
"""

from __future__ import annotations

import math

from .schema import FIELD_SPECS, FieldStatus, FieldValue

__all__ = ["merge_values", "MODE_ORDER", "values_equal"]

MODE_ORDER = {"normal": 0, "first_level": 1}
TOL_REL = 1e-3


def values_equal(a: float, b: float, rel: float = TOL_REL, abs_tol: float = 1e-9) -> bool:
    return abs(a - b) <= max(abs_tol, rel * max(abs(a), abs(b)))


def _as_list(v) -> list:
    if v is None:
        return []
    if isinstance(v, list):
        return list(v)
    return [v]


def _dedupe_evidence(fvs: list[FieldValue]):
    seen = set()
    out = []
    for fv in fvs:
        for ev in fv.evidence:
            k = (ev.page, ev.char_start, ev.char_end, ev.quote)
            if k in seen:
                continue
            seen.add(k)
            out.append(ev)
    return out


def _pick(fvs: list[FieldValue], best: FieldValue) -> FieldValue:
    """Copy of `best` whose evidence lists the binding statement first, then the others."""
    others = [fv for fv in fvs if fv is not best]
    return best.model_copy(update={"evidence": _dedupe_evidence([best, *others]), "review_reasons": []})


def merge_values(field: str, fvs: list[FieldValue]) -> FieldValue:
    """Merge EXTRACTED values for one field into the most conservative value."""
    fvs = [fv for fv in fvs if fv.status == FieldStatus.EXTRACTED]
    if not fvs:
        return FieldValue()
    if len(fvs) == 1:
        return fvs[0].model_copy(deep=True)
    spec = FIELD_SPECS[field]

    if spec.kind == "numeric":

        def readable(fv: FieldValue) -> bool:
            v = fv.normalized_value
            return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)

        if any(fv.comparator != "any" and not readable(fv) for fv in fvs):
            # a stated limit that is not a number must never be dropped in favour of another statement
            return FieldValue(
                status=FieldStatus.NEEDS_REVIEW,
                evidence=_dedupe_evidence(fvs),
                review_reasons=["a statement for this limit could not be read as a number"],
            )
        limited = [fv for fv in fvs if fv.comparator != "any"]
        if not limited:
            return _pick(fvs, fvs[0])
        if spec.direction == "lower":
            best = max(limited, key=lambda fv: float(fv.normalized_value))  # type: ignore[arg-type]
        elif spec.direction == "upper":
            best = min(limited, key=lambda fv: float(fv.normalized_value))  # type: ignore[arg-type]
        else:  # info (artifact): report the largest
            best = max(limited, key=lambda fv: float(fv.normalized_value))  # type: ignore[arg-type]
        return _pick(fvs, best)

    if spec.kind == "b0":

        def b0_ok(fv: FieldValue) -> bool:
            vals = _as_list(fv.normalized_value)
            return bool(vals) and all(
                isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in vals
            )

        if not all(b0_ok(fv) for fv in fvs):
            return FieldValue(
                status=FieldStatus.NEEDS_REVIEW,
                evidence=_dedupe_evidence(fvs),
                review_reasons=["a field-strength statement could not be read as a number"],
            )
        in_sets = [set(map(float, _as_list(fv.normalized_value))) for fv in fvs if fv.comparator != "le"]
        le_vals = [min(map(float, _as_list(fv.normalized_value))) for fv in fvs if fv.comparator == "le"]
        ev = _dedupe_evidence(fvs)
        if in_sets:
            allowed = set.intersection(*in_sets)
            if le_vals:
                allowed = {v for v in allowed if v <= min(le_vals) + 1e-9}
            if not allowed:
                return FieldValue(
                    status=FieldStatus.NEEDS_REVIEW,
                    evidence=ev,
                    value=None,
                    review_reasons=["conflicting field-strength statements"],
                )
            vals = sorted(allowed)
            return FieldValue(
                status=FieldStatus.EXTRACTED,
                value=vals,
                unit="T",
                normalized_value=vals,
                normalized_unit="T",
                comparator="in",
                evidence=ev,
            )
        m = min(le_vals)
        return FieldValue(
            status=FieldStatus.EXTRACTED,
            value=m,
            unit="T",
            normalized_value=[m],
            normalized_unit="T",
            comparator="le",
            evidence=ev,
        )

    if spec.kind == "bool":
        vals = [bool(fv.normalized_value) for fv in fvs]
        best = next((fv for fv in fvs if not bool(fv.normalized_value)), fvs[0]) if not all(vals) else fvs[0]
        return _pick(fvs, best)

    if spec.kind == "categorical":
        if field == "operating_mode":
            ranked = [fv for fv in fvs if str(fv.normalized_value) in MODE_ORDER]
            if len(ranked) != len(fvs):  # an unrecognised mode statement could be the most restrictive one
                return FieldValue(
                    status=FieldStatus.NEEDS_REVIEW,
                    evidence=_dedupe_evidence(fvs),
                    review_reasons=["unrecognised operating mode"],
                )
            best = min(ranked, key=lambda fv: MODE_ORDER[str(fv.normalized_value)])
            return _pick(fvs, best)
        if field in ("scanner_type", "b0_orientation"):
            specific = {str(fv.normalized_value) for fv in fvs if fv.normalized_value not in (None, "any")}
            if len(specific) > 1:
                return FieldValue(
                    status=FieldStatus.NEEDS_REVIEW,
                    evidence=_dedupe_evidence(fvs),
                    review_reasons=[f"conflicting statements: {sorted(specific)}"],
                )
            best = next((fv for fv in fvs if fv.normalized_value not in (None, "any")), fvs[0])
            return _pick(fvs, best)
        # allowed-set fields
        sets = []
        for fv in fvs:
            vals = set(map(str, _as_list(fv.normalized_value)))
            sets.append(None if "any" in vals else vals)  # None = universe
        concrete = [s for s in sets if s is not None]
        if not concrete:
            return _pick(fvs, fvs[0])
        inter = set.intersection(*concrete)
        if not inter:
            return FieldValue(
                status=FieldStatus.NEEDS_REVIEW,
                evidence=_dedupe_evidence(fvs),
                review_reasons=["statements permit disjoint options; review which applies"],
            )
        vals = sorted(inter)
        # keep the raw value of the most restrictive statement
        best = min((fv for fv, s in zip(fvs, sets) if s is not None), key=lambda fv: len(_as_list(fv.normalized_value)))
        out = _pick(fvs, best)
        out.normalized_value = vals
        return out

    # text
    texts: list[str] = []
    for fv in fvs:
        t = str(fv.normalized_value if fv.normalized_value is not None else fv.value or "").strip()
        if t and t not in texts:
            texts.append(t)
    if field == "scan_region_restrictions":
        restrictive = [t for t in texts if t != "unrestricted"]
        texts = restrictive or texts
    joined = " | ".join(texts)
    return FieldValue(
        status=FieldStatus.EXTRACTED,
        value=joined,
        unit=None,
        normalized_value=joined,
        normalized_unit=None,
        comparator="eq",
        evidence=_dedupe_evidence(fvs),
    )
