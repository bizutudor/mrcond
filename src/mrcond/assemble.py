"""Assemble extraction candidates into a LabelDocument.

* MR status: explicit statements only; MR Conditional > MR Unsafe > MR Safe
  when a document mentions several (e.g. an MR Conditional implant whose
  external processor is MR Unsafe).
* Permitted field strengths: union of the accepted field-strength statements
  and of the field strengths that scope other statements.
* One condition set per field strength when any statement is scoped to a field
  strength; otherwise a single set covering all permitted field strengths.
* Within a set, all applicable statements for a field are merged with
  :func:`mrcond.merge.merge_values` (most conservative value).
* Any candidate rejected by the verifier turns its field into NEEDS_REVIEW
  (the conservatively merged accepted value, if any, is kept as a suggestion).
"""

from __future__ import annotations

from datetime import UTC, datetime

from .extractors.base import Candidate, ExtractionOutput
from .merge import merge_values
from .schema import (
    FIELD_NAMES,
    ConditionSet,
    FieldStatus,
    FieldValue,
    LabelDocument,
    MRStatus,
    Provenance,
)

__all__ = ["assemble", "b0_key"]

_STATUS_PRIORITY = [MRStatus.MR_CONDITIONAL, MRStatus.MR_UNSAFE, MRStatus.MR_SAFE, MRStatus.NOT_STATED]


def b0_key(v: float) -> str:
    return f"{float(v):g}"


def _close(a: float, b: float) -> bool:
    return abs(a - b) < 0.05


def _merge_field(field: str, cands: list[Candidate]) -> FieldValue:
    accepted = [c for c in cands if c.accepted is not False]
    rejected = [c for c in cands if c.accepted is False]
    if not accepted and not rejected:
        return FieldValue()
    merged = merge_values(field, [c.fv for c in accepted]) if accepted else FieldValue()
    if rejected:
        reasons = list(dict.fromkeys(r for c in rejected for r in (c.reasons or ["rejected by verifier"])))
        merged = merged.model_copy(deep=True)
        merged.status = FieldStatus.NEEDS_REVIEW
        merged.review_reasons = list(dict.fromkeys([*merged.review_reasons, *reasons]))
        if not merged.evidence:
            merged.evidence = [ev for c in rejected for ev in c.fv.evidence][:3]
        if merged.value is None and len(rejected) == 1 and not accepted:
            # surface the rejected suggestion so the reviewer sees what was claimed
            merged.value = rejected[0].fv.value
            merged.unit = rejected[0].fv.unit
    return merged


def _status_field(cands: list[Candidate]) -> FieldValue:
    if not cands:
        return FieldValue()
    accepted = [c for c in cands if c.accepted is not False]
    if not accepted:
        fv = FieldValue(
            status=FieldStatus.NEEDS_REVIEW,
            review_reasons=list(dict.fromkeys(r for c in cands for r in c.reasons))
            or ["status statement rejected by verifier"],
            evidence=[ev for c in cands for ev in c.fv.evidence][:2],
        )
        return fv
    by_status: dict[str, list[Candidate]] = {}
    for c in accepted:
        by_status.setdefault(str(c.fv.normalized_value), []).append(c)
    for st in _STATUS_PRIORITY:
        if st.value in by_status:
            chosen = by_status[st.value]
            fv = chosen[0].fv.model_copy(deep=True)
            fv.evidence = [ev for c in chosen for ev in c.fv.evidence][:3]
            if st == MRStatus.NOT_STATED:
                fv.status = FieldStatus.NOT_STATED
            return fv
    return FieldValue(status=FieldStatus.NEEDS_REVIEW, review_reasons=["unrecognised MR status value"])


def assemble(
    out: ExtractionOutput,
    *,
    device_name: str | None = None,
    manufacturer: str | None = None,
    device_type: str | None = None,
    text_extractor: str | None = None,
) -> LabelDocument:
    status = _status_field(out.status_candidates)

    b0_cands = [c for c in out.candidates if c.field == "b0_tesla"]
    other = [c for c in out.candidates if c.field != "b0_tesla"]

    # permitted field strengths
    b0_global = _merge_field("b0_tesla", [c for c in b0_cands if c.scope_b0 is None]) if b0_cands else FieldValue()
    if b0_cands:
        # union across separate statements (each statement is evidence for its own values)
        acc = [c for c in b0_cands if c.accepted is not False]
        in_vals: list[float] = []
        le_vals: list[float] = []
        for c in acc:
            vals = c.fv.normalized_value if isinstance(c.fv.normalized_value, list) else [c.fv.normalized_value]
            for v in vals:
                if v is None:
                    continue
                (le_vals if c.fv.comparator == "le" else in_vals).append(float(v))
        if in_vals or le_vals:
            ev = [e for c in acc for e in c.fv.evidence]
            if in_vals:
                vals = sorted({round(v, 3) for v in in_vals})
                b0_union = FieldValue(
                    status=FieldStatus.EXTRACTED,
                    value=vals,
                    unit="T",
                    normalized_value=vals,
                    normalized_unit="T",
                    comparator="in",
                    evidence=ev,
                )
            else:
                m = min(le_vals)
                b0_union = FieldValue(
                    status=FieldStatus.EXTRACTED,
                    value=m,
                    unit="T",
                    normalized_value=[m],
                    normalized_unit="T",
                    comparator="le",
                    evidence=ev,
                )
            if b0_global.status == FieldStatus.NEEDS_REVIEW and any(c.accepted is False for c in b0_cands):
                b0_union.status = FieldStatus.NEEDS_REVIEW
                b0_union.review_reasons = b0_global.review_reasons
            b0_global = b0_union

    scoped_vals: list[float] = []
    for c in other:
        if c.scope_b0 and c.accepted is not False:
            for v in c.scope_b0:
                if not any(_close(v, s) for s in scoped_vals):
                    scoped_vals.append(float(v))

    sets: list[ConditionSet] = []
    if scoped_vals:
        keys: list[float] = []
        if b0_global.status == FieldStatus.EXTRACTED and b0_global.comparator == "in":
            keys = [float(v) for v in b0_global.normalized_value]  # type: ignore[union-attr]
        for v in scoped_vals:
            if not any(_close(v, k) for k in keys):
                keys.append(v)
        keys.sort()
        for i, k in enumerate(keys, start=1):
            cs = ConditionSet(set_id=f"cs{i}", scope=f"{b0_key(k)} T")
            ev = [e for e in b0_global.evidence if b0_key(k) in e.quote.replace(".0", "")] or b0_global.evidence
            if not ev:
                ev = [e for c in other if c.scope_b0 and any(_close(k, v) for v in c.scope_b0) for e in c.fv.evidence][
                    :1
                ]
            cs.b0_tesla = FieldValue(
                status=b0_global.status if b0_global.status != FieldStatus.NOT_STATED else FieldStatus.EXTRACTED,
                value=[k],
                unit="T",
                normalized_value=[k],
                normalized_unit="T",
                comparator="in",
                evidence=ev,
                review_reasons=list(b0_global.review_reasons),
            )
            for name in FIELD_NAMES:
                if name == "b0_tesla":
                    continue
                cands = [
                    c
                    for c in other
                    if c.field == name and (c.scope_b0 is None or any(_close(k, v) for v in c.scope_b0))
                ]
                cs.set_field(name, _merge_field(name, cands))
            sets.append(cs)
    else:
        cs = ConditionSet(set_id="cs1")
        cs.b0_tesla = b0_global
        for name in FIELD_NAMES:
            if name == "b0_tesla":
                continue
            cs.set_field(name, _merge_field(name, [c for c in other if c.field == name]))
        if any(fv.status != FieldStatus.NOT_STATED for _, fv in cs.iter_fields()):
            sets.append(cs)

    # A document with conditions but no explicit status statement keeps status NOT_STATED;
    # MR Unsafe / MR Safe documents keep no condition sets.
    if status.status == FieldStatus.EXTRACTED and status.normalized_value in (
        MRStatus.MR_UNSAFE.value,
        MRStatus.MR_SAFE.value,
    ):
        sets = []

    return LabelDocument(
        doc_id=out.doc_id,
        device_name=device_name,
        manufacturer=manufacturer,
        device_type=device_type,
        mr_status=status,
        condition_sets=sets,
        provenance=Provenance(
            extractor=out.extractor,
            verification="machine-verified" if out.verified else "machine-extracted",
            created_at=datetime.now(UTC),
            text_extractor=text_extractor,
        ),
    )
