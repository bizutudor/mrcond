"""Benchmark metrics.

Labels are compared as *slots*: (field-strength key, field). Condition sets are
flattened per permitted field strength (key "1.5", "3", or "<=3" for "3 T or
less"); when several sets share a key, numeric limits are merged with the most
conservative value and categorical permissions with their union.

Per slot the gold state is STATED (with value) or NOT_STATED; the predicted
state is EXTRACTED, NOT_STATED or NEEDS_REVIEW (abstention). Slots where both
are NOT_STATED are ignored.

* TP: predicted value matches gold (numeric: 1 % relative tolerance; B0: same
  key; sets: equal sets; text: token-F1 >= 0.5 or overlapping evidence span).
  Keys are aligned before comparison: a gold field strength Y is scored
  against a predicted "<=X" set (X >= Y), and a gold "<=X" set against the
  predicted set for the largest field strength <= X. A predicted field
  strength is an "extra field strength" only if the gold does not permit it.
* FP: predicted EXTRACTED and (gold NOT_STATED or value mismatch).
* FN: gold STATED and (prediction not EXTRACTED or mismatch).
* less-conservative error (safety-critical): the prediction is more permissive
  than the gold labeling: a higher upper limit (SAR, B1+rms, SFG, slew rate,
  scan time), a lower wait time, an extra field strength, a more permissive
  operating mode / coil / excitation / nucleus set, an "unrestricted" scan
  region where the labeling restricts it, a SAR limit wrongly marked as a
  B1+rms fallback, a stated constraint that was missed (NOT_STATED), or a more
  permissive MR status (MR Unsafe < not stated < MR Conditional < MR Safe, in
  the order the checker treats them). A predicted field strength inside a gold
  "X T or less" range is also checked against that range's limits. Image
  artifact extent is informational and excluded. Abstentions are never
  counted as less conservative.
* unsupported-value rate: EXTRACTED values none of whose evidence quotes occur
  verbatim (whitespace-normalised) in the source text.
* abstention rate: NEEDS_REVIEW slots / scored slots; selective accuracy:
  correct / answered (EXTRACTED or NOT_STATED) slots.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from ..ingest import DocumentText
from ..merge import merge_values, values_equal
from ..schema import FIELD_SPECS, FieldStatus, FieldValue, LabelDocument
from ..textnorm import find_all

__all__ = ["Slot", "flatten", "compare_docs", "DocScore", "aggregate", "unsupported_values", "consistency"]

EVAL_FIELDS = [f for f, s in FIELD_SPECS.items() if s.evaluated and f != "b0_tesla"]
REL_TOL = 0.01


@dataclass
class Slot:
    state: str  # STATED | NOT_STATED | NEEDS_REVIEW
    fv: FieldValue | None = None


def _keys(fv: FieldValue) -> list[str]:
    vals = fv.normalized_value if isinstance(fv.normalized_value, list) else [fv.normalized_value]
    vals = [float(v) for v in vals if isinstance(v, (int, float))]
    if not vals:
        return ["?"]
    if fv.comparator == "le":
        return [f"<={max(vals):g}"]
    return [f"{v:g}" for v in vals]


def _union_categorical(fvs: list[FieldValue]) -> FieldValue:
    vals: set[str] = set()
    for fv in fvs:
        nv = fv.normalized_value
        vals |= set(map(str, nv if isinstance(nv, list) else [nv]))
    out = fvs[0].model_copy(deep=True)
    out.normalized_value = ["any"] if "any" in vals else sorted(vals)
    out.evidence = [e for fv in fvs for e in fv.evidence]
    return out


def flatten(label: LabelDocument) -> dict[tuple[str, str], Slot]:
    """Map (b0 key, field) -> Slot. Includes (key, 'b0_tesla') for every permitted key."""
    groups: dict[tuple[str, str], list[FieldValue]] = defaultdict(list)
    b0_state: dict[str, str] = {}
    for cs in label.condition_sets:
        b0 = cs.b0_tesla
        keys = _keys(b0) if b0.status != FieldStatus.NOT_STATED else ["?"]
        for k in keys:
            st = (
                "NEEDS_REVIEW"
                if b0.status == FieldStatus.NEEDS_REVIEW
                else ("STATED" if b0.status == FieldStatus.EXTRACTED else "NOT_STATED")
            )
            if b0_state.get(k) != "STATED":
                b0_state[k] = st
            for name in EVAL_FIELDS:
                groups[(k, name)].append(cs.field(name))
            groups[(k, "b0_tesla")].append(b0)
    out: dict[tuple[str, str], Slot] = {}
    for k, st in b0_state.items():
        out[(k, "b0_tesla")] = Slot(st, groups[(k, "b0_tesla")][0])
    for (k, name), fvs in groups.items():
        if name == "b0_tesla":
            continue
        stated = [fv for fv in fvs if fv.status == FieldStatus.EXTRACTED]
        review = [fv for fv in fvs if fv.status == FieldStatus.NEEDS_REVIEW]
        if review:
            out[(k, name)] = Slot("NEEDS_REVIEW", review[0])
        elif stated:
            spec = FIELD_SPECS[name]
            if (
                spec.kind == "categorical"
                and name not in ("operating_mode", "scanner_type", "b0_orientation")
                and len(stated) > 1
            ):
                fv = _union_categorical(stated)
            else:
                fv = merge_values(name, stated)
                if fv.status != FieldStatus.EXTRACTED:
                    out[(k, name)] = Slot("NEEDS_REVIEW", fv)
                    continue
            out[(k, name)] = Slot("STATED", fv)
        else:
            out[(k, name)] = Slot("NOT_STATED", None)
    return out


_TOKEN = re.compile(r"[a-z0-9]+")


def _token_f1(a: str, b: str) -> float:
    ta, tb = _TOKEN.findall(a.lower()), _TOKEN.findall(b.lower())
    if not ta or not tb:
        return 0.0
    common = Counter(ta) & Counter(tb)
    n = sum(common.values())
    if n == 0:
        return 0.0
    p, r = n / len(ta), n / len(tb)
    return 2 * p * r / (p + r)


def _overlap(a: FieldValue, b: FieldValue) -> bool:
    for x in a.evidence:
        for y in b.evidence:
            if x.page == y.page and x.char_start < y.char_end and y.char_start < x.char_end:
                return True
    return False


def _set(v) -> set[str]:
    return set(map(str, v if isinstance(v, list) else [v]))


def values_match(name: str, g: FieldValue, p: FieldValue) -> bool:
    spec = FIELD_SPECS[name]
    if spec.kind == "numeric":
        if g.comparator == "any" or p.comparator == "any":
            return g.comparator == p.comparator
        try:
            return values_equal(float(g.normalized_value), float(p.normalized_value), REL_TOL)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False
    if spec.kind == "bool":
        return bool(g.normalized_value) == bool(p.normalized_value)
    if spec.kind == "categorical":
        if name == "nucleus":
            return _set(g.normalized_value) | {"1H"} == _set(p.normalized_value) | {"1H"}
        return _set(g.normalized_value) == _set(p.normalized_value)
    # text
    gv, pv = str(g.normalized_value or ""), str(p.normalized_value or "")
    if gv == "unrestricted" or pv == "unrestricted":
        return gv == pv
    return _token_f1(gv, pv) >= 0.5 or _overlap(g, p)


def less_conservative(name: str, g: FieldValue | None, p: FieldValue | None) -> str | None:
    """Reason string if prediction `p` is more permissive than gold `g`, else None."""
    spec = FIELD_SPECS[name]
    if spec.direction == "info" or name in ("other_conditions",):
        return None
    if g is None or g.status != FieldStatus.EXTRACTED:
        # gold has no such constraint: only a wrongly asserted SAR fallback loosens anything
        if name == "sar_is_b1rms_fallback" and p is not None and bool(p.normalized_value):
            return "SAR wrongly marked as B1+rms fallback"
        return None
    if p is None or p.status == FieldStatus.NOT_STATED:
        # missing a constraint is permissive unless gold explicitly states 'no restriction'
        if g.comparator == "any" or (spec.kind == "categorical" and "any" in _set(g.normalized_value)):
            return None
        if name == "scan_region_restrictions" and g.normalized_value == "unrestricted":
            return None
        if name == "sar_is_b1rms_fallback":
            return None  # missing the fallback flag is conservative
        if name == "nucleus" and _set(g.normalized_value) <= {"1H"}:
            return None  # FDA 2023: if no nucleus is listed, 1H is assumed
        return "missed constraint"
    if p.status != FieldStatus.EXTRACTED:
        return None
    if spec.kind == "numeric":
        if p.comparator == "any" and g.comparator != "any":
            return "limit dropped (prediction says unlimited)"
        if g.comparator == "any":
            return None
        try:
            gv, pv = float(g.normalized_value), float(p.normalized_value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        if spec.direction == "upper" and pv > gv * (1 + REL_TOL) + 1e-9:
            return f"higher limit ({pv:g} > {gv:g})"
        if spec.direction == "lower" and pv < gv * (1 - REL_TOL) - 1e-9:
            return f"shorter wait ({pv:g} < {gv:g})"
        return None
    if spec.kind == "bool":
        if bool(p.normalized_value) and not bool(g.normalized_value):
            return "SAR wrongly marked as B1+rms fallback"
        return None
    if spec.kind == "categorical":
        gs, ps = _set(g.normalized_value), _set(p.normalized_value)
        if name == "operating_mode":
            order = {"normal": 0, "first_level": 1}
            if order.get(next(iter(ps)), 0) > order.get(next(iter(gs)), 0):
                return "more permissive operating mode"
            return None
        if name in ("scanner_type", "b0_orientation"):
            return "different scanner requirement" if gs != ps and "any" not in gs else None
        if name == "nucleus":
            gs, ps = gs | {"1H"}, ps | {"1H"}
        if "any" in gs:
            return None
        if "any" in ps or not ps <= gs:
            return f"permits {sorted(ps - gs) or ['any']}"
        return None
    if name == "scan_region_restrictions":
        if p.normalized_value == "unrestricted" and g.normalized_value != "unrestricted":
            return "restriction dropped"
    return None


@dataclass
class DocScore:
    doc_id: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    abstain: int = 0
    answered: int = 0
    correct: int = 0
    scored: int = 0
    gold_stated: int = 0
    lc: list[tuple[str, str, str]] = field(default_factory=list)  # (key, field, reason)
    per_field: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    status_correct: bool = True
    n_values: int = 0
    n_unsupported: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def all_correct(self) -> bool:
        return self.status_correct and self.fp == 0 and self.fn == 0 and self.abstain == 0


def unsupported_values(label: LabelDocument, doc: DocumentText) -> tuple[int, int]:
    """(n EXTRACTED values, n whose evidence quotes are not found verbatim in `doc`)."""
    n = bad = 0
    fvs = [label.mr_status] if label.mr_status.status == FieldStatus.EXTRACTED else []
    for cs in label.condition_sets:
        fvs += [fv for _, fv in cs.iter_fields() if fv.status == FieldStatus.EXTRACTED]
    for fv in fvs:
        n += 1
        ok = False
        for ev in fv.evidence:
            if ev.quote and any(find_all(p.text, ev.quote) for p in doc.pages):
                ok = True
                break
        if not ok:
            bad += 1
    return n, bad


def _parse_key(k: str) -> tuple[str, float] | None:
    if k == "?":
        return None
    return ("le", float(k[2:])) if k.startswith("<=") else ("in", float(k))


def _align_keys(gold_keys: set[str], pred_keys: set[str]) -> dict[str, str]:
    """Gold key -> the predicted key whose conditions apply to it.

    Exact keys first. A gold field strength Y is covered by a predicted "<=X" set (X >= Y). A gold "<=X" set
    is compared with the predicted set for the largest field strength <= X, else with a wider "<=Z" set."""
    out: dict[str, str] = {}
    pk = {k: v for k in pred_keys if (v := _parse_key(k)) is not None}
    for gk in gold_keys:
        if gk in pred_keys:
            out[gk] = gk
            continue
        gp = _parse_key(gk)
        if gp is None:
            continue
        kind, v = gp
        les = sorted((pv, k) for k, (pkind, pv) in pk.items() if pkind == "le" and pv >= v - 1e-9)
        ins = sorted((pv, k) for k, (pkind, pv) in pk.items() if pkind == "in" and pv <= v + 1e-9)
        if kind == "le" and ins:
            out[gk] = ins[-1][1]
        elif les:
            out[gk] = les[0][1]
    return out


def _permitted_by_gold(gold_keys: set[str], key: str) -> bool:
    """True if every field strength that predicted key `key` permits is also permitted by the gold labeling."""
    kp = _parse_key(key)
    if kp is None:
        return True
    kind, v = kp
    for gk in gold_keys:
        gp = _parse_key(gk)
        if gp is None:
            continue
        gkind, gv = gp
        if gkind == "le" and v <= gv + 1e-9:
            return True
        if gkind == "in" and kind == "in" and abs(gv - v) < 1e-9:
            return True
    return False


# How permissive an MR status makes the checker: MR Unsafe -> conflict; not stated -> needs review;
# MR Conditional -> conditions are compared; MR Safe -> no conditions at all.
_STATUS_PERMISSIVENESS = {"MR Unsafe": 0, "Not stated": 1, "MR Conditional": 2, "MR Safe": 3}


def status_less_conservative(gold: LabelDocument, pred: LabelDocument) -> str | None:
    """Reason if the predicted MR status is more permissive than the gold status (abstention excluded)."""
    if pred.mr_status.status == FieldStatus.NEEDS_REVIEW:
        return None
    gs, ps = gold.status_value().value, pred.status_value().value
    if _STATUS_PERMISSIVENESS[ps] > _STATUS_PERMISSIVENESS[gs]:
        return f"more permissive MR status ({ps} for {gs})"
    return None


def compare_docs(gold: LabelDocument, pred: LabelDocument, doc: DocumentText | None = None) -> DocScore:
    sc = DocScore(gold.doc_id)
    sc.status_correct = gold.status_value() == pred.status_value()
    status_lc = status_less_conservative(gold, pred)
    if status_lc:
        sc.lc.append(("-", "mr_status", status_lc))
    g, p = flatten(gold), flatten(pred)
    gold_keys = {k for (k, f) in g if f == "b0_tesla"}
    pred_keys = {k for (k, f) in p if f == "b0_tesla"}
    unknown_key = "?" in pred_keys  # a predicted set without a field strength: score its fields against every key
    align = _align_keys(gold_keys, pred_keys)  # "3 T or less" vs "3 T" etc.

    def pslot(k: str, name: str) -> Slot:
        if k not in gold_keys:  # predicted-only field strength
            return p.get((k, name), Slot("NOT_STATED"))
        pk = align.get(k)
        if pk is not None:
            if name == "b0_tesla" and pk != k:
                # a gold field strength inside a predicted "<=X" range is permitted by the prediction; a gold
                # "<=X" range predicted as single field strengths is a (conservative) miss
                return p[(pk, name)] if not k.startswith("<=") else Slot("NOT_STATED")
            if (pk, name) in p:
                return p[(pk, name)]
            return Slot("NOT_STATED")
        if unknown_key and name != "b0_tesla" and ("?", name) in p:
            return p[("?", name)]
        return Slot("NOT_STATED")

    for k in sorted(gold_keys | (pred_keys - {"?"})):
        fields = ["b0_tesla", *EVAL_FIELDS] if k in gold_keys else ["b0_tesla"]
        for name in fields:
            gs = g.get((k, name), Slot("NOT_STATED"))
            ps = pslot(k, name)
            if gs.state == "NEEDS_REVIEW":
                continue  # ambiguous in gold: not scored
            if gs.state == "NOT_STATED" and ps.state == "NOT_STATED":
                continue
            sc.scored += 1
            pf = sc.per_field[name]
            if gs.state == "STATED":
                sc.gold_stated += 1
                pf["gold"] += 1
            if ps.state == "STATED":
                pf["pred"] += 1
            if ps.state == "NEEDS_REVIEW":
                sc.abstain += 1
                pf["abstain"] += 1
                if gs.state == "STATED":
                    sc.fn += 1
                    pf["fn"] += 1
                continue
            sc.answered += 1
            if name == "b0_tesla":
                if gs.state == "STATED" and ps.state == "STATED":
                    sc.tp += 1
                    pf["tp"] += 1
                    sc.correct += 1
                elif ps.state == "STATED":
                    sc.fp += 1
                    pf["fp"] += 1
                    if not _permitted_by_gold(gold_keys, k):
                        sc.lc.append((k, name, "extra field strength"))
                else:
                    sc.fn += 1
                    pf["fn"] += 1
                continue
            if gs.state == "STATED" and ps.state == "STATED":
                if values_match(name, gs.fv, ps.fv):  # type: ignore[arg-type]
                    sc.tp += 1
                    pf["tp"] += 1
                    sc.correct += 1
                else:
                    sc.fp += 1
                    sc.fn += 1
                    pf["fp"] += 1
                    pf["fn"] += 1
                    reason = less_conservative(name, gs.fv, ps.fv)
                    if reason:
                        sc.lc.append((k, name, reason))
            elif gs.state == "STATED":
                sc.fn += 1
                pf["fn"] += 1
                if k in align or unknown_key:  # a missing field strength is conservative, not a missed constraint
                    reason = less_conservative(name, gs.fv, None)
                    if reason:
                        sc.lc.append((k, name, reason))
            else:  # gold not stated, predicted
                sc.fp += 1
                pf["fp"] += 1
                reason = less_conservative(name, None, ps.fv)
                if reason:
                    sc.lc.append((k, name, reason))
    # A predicted field strength inside a gold "<=X" range (e.g. a 1.5 T set for "3 T or less") is not aligned with
    # the gold set above, but its limits apply to that range too: count less-conservative errors (not P/R).
    aligned = set(align.values())
    for k in sorted(pred_keys - gold_keys - aligned - {"?"}):
        kp = _parse_key(k)
        if kp is None:
            continue
        ranges = [gk for gk in gold_keys if (gp := _parse_key(gk)) and gp[0] == "le" and kp[1] <= gp[1] + 1e-9]
        for gk in ranges[:1]:
            for name in EVAL_FIELDS:
                gs, ps = g.get((gk, name), Slot("NOT_STATED")), p.get((k, name), Slot("NOT_STATED"))
                if gs.state != "STATED" or ps.state == "NEEDS_REVIEW":
                    continue
                reason = less_conservative(name, gs.fv, ps.fv if ps.state == "STATED" else None)
                if reason:
                    sc.lc.append((k, name, reason))
    if doc is not None:
        sc.n_values, sc.n_unsupported = unsupported_values(pred, doc)
    return sc


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def aggregate(scores: list[DocScore]) -> dict:
    tp = sum(s.tp for s in scores)
    fp = sum(s.fp for s in scores)
    fn = sum(s.fn for s in scores)
    p, r, f = _prf(tp, fp, fn)
    scored = sum(s.scored for s in scores)
    abstain = sum(s.abstain for s in scores)
    answered = sum(s.answered for s in scores)
    correct = sum(s.correct for s in scores)
    gold_stated = sum(s.gold_stated for s in scores)
    lc = [x for s in scores for x in s.lc]
    n_values = sum(s.n_values for s in scores)
    n_unsup = sum(s.n_unsupported for s in scores)
    per_field = {}
    for name in ["b0_tesla", *EVAL_FIELDS]:
        c = Counter()
        for s in scores:
            c.update(s.per_field.get(name, Counter()))
        fp_, fr_, ff_ = _prf(c["tp"], c["fp"], c["fn"])
        per_field[name] = {
            "gold": c["gold"],
            "tp": c["tp"],
            "fp": c["fp"],
            "fn": c["fn"],
            "abstain": c["abstain"],
            "precision": round(fp_, 4),
            "recall": round(fr_, 4),
            "f1": round(ff_, 4),
        }
    lc_kinds = Counter(reason.split(" (")[0] for _, _, reason in lc)
    return {
        "n_docs": len(scores),
        "status_accuracy": round(sum(s.status_correct for s in scores) / len(scores), 4) if scores else 0.0,
        "docs_fully_correct": sum(s.all_correct for s in scores),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f, 4),
        "scored_slots": scored,
        "gold_stated_slots": gold_stated,
        "abstentions": abstain,
        "abstention_rate": round(abstain / scored, 4) if scored else 0.0,
        "answered": answered,
        "selective_accuracy": round(correct / answered, 4) if answered else 0.0,
        "less_conservative_errors": len(lc),
        "less_conservative_rate": round(len(lc) / gold_stated, 4) if gold_stated else 0.0,
        "less_conservative_by_kind": dict(lc_kinds),
        "less_conservative_examples": [
            f"{s.doc_id} [{k}] {name}: {reason}" for s in scores for k, name, reason in s.lc
        ][:25],
        "n_values": n_values,
        "unsupported_values": n_unsup,
        "unsupported_rate": round(n_unsup / n_values, 4) if n_values else 0.0,
        "per_field": per_field,
    }


def consistency(runs: list[LabelDocument]) -> float | None:
    """Fraction of scored slots on which all runs agree (state and value). None if < 2 runs."""
    if len(runs) < 2:
        return None
    flats = [flatten(r) for r in runs]
    keys = set().union(*[set(f) for f in flats])
    agree = total = 0
    for key in keys:
        slots = [f.get(key, Slot("NOT_STATED")) for f in flats]
        if all(s.state == "NOT_STATED" for s in slots):
            continue
        total += 1
        first = slots[0]
        same = True
        for s in slots[1:]:
            if s.state != first.state:
                same = False
                break
            if s.state == "STATED" and key[1] != "b0_tesla" and not values_match(key[1], first.fv, s.fv):  # type: ignore[arg-type]
                same = False
                break
        agree += same
    return agree / total if total else 1.0
