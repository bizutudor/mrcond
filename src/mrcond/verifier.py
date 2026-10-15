"""Quote-or-abstain verifier.

Every candidate value must be supported by evidence that passes all checks,
otherwise it is rejected and the field becomes NEEDS_REVIEW (abstention):

1. **Evidence present** - at least one quote.
2. **Verbatim quote** - the quote occurs in the source text after whitespace
   normalisation (see :mod:`mrcond.textnorm`). Offsets and page are re-anchored
   to the actual occurrence; quotes longer than 25 words are trimmed around
   the value (MRCond never stores long manufacturer text).
3. **Parse check** - the claimed value *and unit* can be parsed from the
   quote: e.g. a claimed whole-body SAR of 2 W/kg needs a "2 W/kg" quantity
   in the quote; 3,000 G/cm supports 30 T/m; a value whose stated unit has the
   wrong dimension (µT for a SAR limit) is rejected.
4. **Context check** - the quote mentions the quantity it is attributed to
   (e.g. "head" for head SAR, "B1" for B1+rms, "spatial"/"gradient" for the
   spatial field gradient).
5. **Plausibility** - normalised values must lie in physically plausible
   ranges (e.g. whole-body SAR 0.01-4 W/kg, B1+rms 0.1-10 µT).
6. **Scope** - a value quoted from a line scoped to one field strength
   ("For 1.5T MR Scanner: Whole-Body SAR ≤ 2 W/kg") must not be applied to
   other field strengths (added after inspecting LLM errors on Track R).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .extractors.base import Candidate, ExtractionOutput
from .ingest import DocumentText
from .merge import values_equal
from .schema import FIELD_SPECS, Evidence, FieldValue, MRStatus
from .textnorm import find_all, trim_quote, word_count
from .units import find_numbers, find_quantities, find_word_durations, normalize_unit, round_sig

__all__ = ["VerificationReport", "verify_candidate", "verify_output", "locate_quote", "MAX_QUOTE_WORDS"]

MAX_QUOTE_WORDS = 25

_TOKEN_SUPPORT: dict[str, dict[str, str]] = {
    "nucleus": {
        "1H": r"1H|¹H|[Hh]ydrogen|[Pp]roton|[Nn]o additional nucle|[Nn]ucle",
        "31P": r"31P|³¹P|[Pp]hosph",
        "23Na": r"23Na|²³Na|[Ss]odium",
        "13C": r"13C|¹³C|[Cc]arbon",
        "19F": r"19F|¹⁹F|[Ff]luorine",
        "129Xe": r"129Xe|[Xx]enon",
        "any": r"\b[Aa]ny\b|[Nn]o restriction",
    },
    "scanner_type": {
        "cylindrical": r"[Cc]ylindric|[Cc]losed",
        "open": r"[Oo]pen",
        "any": r"\b[Aa]ny\b|[Nn]o restriction",
    },
    "b0_orientation": {
        "horizontal": r"[Hh]orizontal",
        "vertical": r"[Vv]ertical",
        "any": r"\b[Aa]ny\b",
    },
    "rf_excitation": {
        "CP": r"[Cc]ircular|\bCP\b|[Qq]uadrature",
        "MC-2": r"MC-?2|[Mm]ulti-?\s?channel|[Pp]arallel|pTx|[Ee]lliptic",
        "any": r"\b[Aa]ny\b|[Nn]o restriction",
    },
    "transmit_coil": {
        "body": r"[Bb]ody",
        "head": r"[Hh]ead",
        "extremity": r"[Ee]xtremit|[Kk]nee|[Ww]rist|[Aa]nkle|[Ff]oot",
        "local": r"[Ll]ocal|[Ss]urface",
        "any": r"\b[Aa]ny\b|[Nn]o .{0,40}restrict|[Nn]ot restricted|[Uu]nrestricted|\b[Aa]ll\b",
    },
    "receive_coil": {
        "any": r"\b[Aa]ny\b|[Nn]o .{0,40}restrict|[Nn]ot restricted|[Uu]nrestricted|\b[Aa]ll\b",
        "body": r"[Bb]ody",
        "head": r"[Hh]ead",
        "extremity": r"[Ee]xtremit|[Kk]nee|[Ww]rist|[Aa]nkle|[Ff]oot",
        "local": r"[Ll]ocal|[Ss]urface|[Rr]eceive-only",
    },
    "operating_mode": {
        "normal": r"[Nn]ormal",
        "first_level": r"[Ff]irst[\s-]*[Ll]evel",
    },
}

_STATUS_SUPPORT = {
    MRStatus.MR_CONDITIONAL.value: r"MRI?\s*-?\s*[Cc]onditional|MR\s*-?\s*CONDITIONAL",
    MRStatus.MR_SAFE.value: r"MRI?\s*-?\s*[Ss]afe(?!ty)|MR\s*-?\s*SAFE",
    MRStatus.MR_UNSAFE.value: r"MRI?\s*-?\s*[Uu]nsafe|MR\s*-?\s*UNSAFE",
    MRStatus.NOT_STATED.value: r"not been (?:evaluated|tested)|[Nn]ot [Ee]valuated|unknown",
}

FALLBACK_RE = re.compile(
    r"(?:not|n't|unable to|cannot)\s+(?:\w+\s+){0,3}(?:report|display|provid|availab|show)\w*\s+(?:\w+\s+){0,3}B\s*1"
    r"|B\s*1\s*\+?\s*,?\s*[Rr][Mm][Ss]\S*\s+(?:\w+\s+){0,2}(?:is|are)\s+not\s+(?:available|reported|displayed|provided)"
    r"|(?:[Ii]f|[Ww]hen|[Ww]here)\s+B\s*1\S*(?:\s+\S+){0,4}\s+(?:not|unavailable)"
    r"|(?:without|lacking)\s+(?:\w+\s+){0,2}B\s*1",
)
# "any"/"all" only supports an unrestricted option if the quote does not also restrict it
# ("Use the body coil only; all other coils are prohibited", "Do not use any local transmit coil")
_RESTRICTIVE_RE = re.compile(
    r"(?<![-\w])[Oo]nly\b|\bONLY\b|[Pp]rohibit|\b[Dd]o not\b|\b[Mm]ust not\b|\b[Nn]ever\b|"
    r"\b[Nn]ot (?:be )?(?:used|permitted|allowed)|[Ee]xclusively"
)
NO_LIMIT_RE = re.compile(
    r"\b[Nn]o\b[^.;]{0,30}?\b(?:limit|limitation|restriction)s?\b|[Uu]nlimited|"
    r"\b[Nn]ot (?:limited|restricted)\b"
)
_STATUS_NEGATION_RE = re.compile(r"(?:\b(?:not|never)|n't)\s+(?:\w+\s+){0,2}$", re.IGNORECASE)
_EFFECTIVE_SLEW_RE = re.compile(r"[Ee]ffective|[Vv]ector|[Cc]ombined|all (?:three )?axes")
_UNRESTRICTED_REGION_RE = re.compile(
    r"[Aa]ny landmark|[Nn]o (?:exclusion|scan region|landmark|positioning)\w*\s*(?:zone|restriction)?|"
    r"[Ff]ull[\s-]*[Bb]ody|[Aa]nywhere|[Aa]ll (?:body )?regions|[Nn]o restrictions? on (?:the )?(?:scan|landmark|position)",
)
_UNIT_HINT = {
    "gradient": r"T/m|G/cm|[Gg]auss\s*/\s*cm|mT/m|T\s*/\s*m",
    "slew": r"T/m/s|mT/m/ms",
    "b1": r"µT|μT|uT|[Mm]icro-?[Tt]esla",
    "sar": r"W/kg|W/Kg",
    "time": r"min|hour|second|\bs\b",
    "length": r"mm|cm",
    "field": r"\bT\b|[Tt]esla|\dT\b",
}


@dataclass
class VerificationReport:
    n_candidates: int = 0
    n_accepted: int = 0
    n_rejected: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    def add(self, accepted: bool, reasons: list[str]) -> None:
        self.n_candidates += 1
        if accepted:
            self.n_accepted += 1
        else:
            self.n_rejected += 1
            for r in reasons:
                key = r.split(":")[0]
                self.reasons[key] = self.reasons.get(key, 0) + 1


def locate_quote(doc: DocumentText, ev: Evidence) -> Evidence | None:
    """Re-anchor `ev` to the occurrence of its quote in the source (cited page first)."""
    pages = [doc.page(ev.page)] if doc.page(ev.page) else []
    pages += [p for p in doc.pages if p.number != ev.page]
    for p in pages:
        if p is None:
            continue
        spans = find_all(p.text, ev.quote)
        if spans:
            hint = ev.char_start if p.number == ev.page else 0
            s, e = min(spans, key=lambda sp: abs(sp[0] - hint))
            return Evidence(
                doc_id=doc.doc_id, page=p.number, char_start=s, char_end=e, quote=p.text[s:e], verified=True
            )
    return None


def _num(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _numeric_support(spec, fv: FieldValue, quotes: list[str]) -> list[str]:
    reasons = []
    target = _num(fv.normalized_value)
    if target is None:
        if fv.comparator == "any":
            # "no limit" is a value too: the quote must say so ("There is no limit on scan duration")
            if any(NO_LIMIT_RE.search(q) for q in quotes):
                return []
            return ["parse: quote does not state that there is no limit"]
        return ["parse: normalised value is not a number"]
    # stated unit must belong to the field's dimension
    if fv.unit:
        info = normalize_unit(fv.unit)
        if info is None:
            reasons.append(f"unit: unrecognised unit '{fv.unit}'")
        elif info.dimension != spec.dimension:
            reasons.append(f"unit: '{fv.unit}' is not a {spec.dimension} unit")
        elif _num(fv.value) is not None:
            expect = round_sig(float(fv.value) * info.factor)  # type: ignore[arg-type]
            if not values_equal(expect, target):
                reasons.append(f"parse: stated {fv.value} {fv.unit} != normalised {target}")
    if reasons:
        return reasons
    found = False
    for q in quotes:
        qs = find_quantities(q, {spec.dimension})
        if spec.dimension == "time":
            qs += find_word_durations(q)
        if any(values_equal(x.canonical_value, target) for x in qs):
            found = True
            break
    if not found:
        # table cells: bare number with the unit stated elsewhere in the quote (e.g. "SAR (W/kg) 2.0"). Only
        # for quotes without an explicit quantity of this dimension, and only numbers that are not part of
        # another quantity: "For 3T: Whole-Body SAR <= 2 W/kg" must not support a SAR of 3 W/kg.
        for q in quotes:
            if find_quantities(q, {spec.dimension}) or not re.search(_UNIT_HINT.get(spec.dimension, "$^"), q):
                continue
            taken = [(x.start, x.end) for x in find_quantities(q)]
            info = normalize_unit(fv.unit) if fv.unit else None
            factor = info.factor if info else 1.0
            bare = [v for v, s, e in find_numbers(q) if not any(s < te and ts < e for ts, te in taken)]
            if any(values_equal(round_sig(v * factor), target) for v in bare):
                found = True
                break
    if not found:
        reasons.append(f"parse: no {target} {spec.canonical_unit} quantity in quote")
    return reasons


def _b0_support(fv: FieldValue, quotes: list[str]) -> list[str]:
    vals = fv.normalized_value if isinstance(fv.normalized_value, list) else [fv.normalized_value]
    qs = [x for q in quotes for x in find_quantities(q, {"field"}) if x.unit == "T"]
    missing = [v for v in vals if _num(v) is None or not any(abs(x.canonical_value - float(v)) < 1e-6 for x in qs)]
    return [f"parse: field strength {missing} T not found in quote"] if missing else []


def _context_ok(spec, quotes: list[str]) -> list[str]:
    text = " ".join(quotes)
    for pat in spec.context:
        if not re.search(pat, text):
            return [f"context: quote does not mention the {spec.label.lower()}"]
    if (
        spec.name == "wb_sar_max_w_per_kg"
        and re.search(r"[Hh]ead", text)
        and not re.search(r"[Ww]hole|[Bb]ody|WB", text)
    ):
        return ["context: quote refers to head SAR"]
    if (
        spec.name == "max_sfg_t_per_m"
        and re.search(r"T\s*²\s*/\s*m|T2/m|product", text)
        and not re.search(r"[Ss]patial", text)
    ):
        return ["context: quote refers to the force product, not the spatial field gradient"]
    if spec.name == "max_slew_rate_t_per_m_per_s" and _EFFECTIVE_SLEW_RE.search(text):
        return ["context: effective (vector) slew rate, not the per-axis slew rate"]
    return []


def _plausible(spec, fv: FieldValue) -> list[str]:
    if spec.plausible is None:
        return []
    lo, hi = spec.plausible
    vals = fv.normalized_value if isinstance(fv.normalized_value, list) else [fv.normalized_value]
    bad = [v for v in vals if _num(v) is not None and not (lo <= float(v) <= hi)]
    return [f"plausibility: {bad} outside {lo}-{hi} {spec.canonical_unit}"] if bad else []


def verify_candidate(c: Candidate, doc: DocumentText) -> Candidate:
    """Check one candidate in place; sets ``accepted`` and ``reasons``; re-anchors evidence."""
    reasons: list[str] = []
    fv = c.fv
    if not fv.evidence:
        c.accepted, c.reasons = False, ["evidence: no supporting quote"]
        return c
    verified: list[Evidence] = []
    for ev in fv.evidence:
        if not ev.quote or not ev.quote.strip():
            continue
        loc = locate_quote(doc, ev)
        if loc is None:
            continue
        if word_count(loc.quote) > MAX_QUOTE_WORDS:
            page_text = doc.page_text(loc.page)
            focus = (loc.char_start, loc.char_end)
            # focus on the claimed number, if present
            target = None if isinstance(fv.value, list) else fv.value
            if target is not None:
                for v, s, e in find_numbers(loc.quote):
                    if _num(target) is not None and values_equal(v, float(target)):  # type: ignore[arg-type]
                        focus = (loc.char_start + s, loc.char_start + e)
                        break
            s, e = trim_quote(page_text, loc.char_start, loc.char_end, focus[0], focus[1], MAX_QUOTE_WORDS)
            loc = Evidence(
                doc_id=doc.doc_id, page=loc.page, char_start=s, char_end=e, quote=page_text[s:e], verified=True
            )
        verified.append(loc)
    if not verified:
        c.accepted, c.reasons = False, ["evidence: quote not found verbatim in source"]
        fv.evidence = [ev.model_copy(update={"verified": False}) for ev in fv.evidence]
        return c
    quotes = [ev.quote for ev in verified]

    if c.field == "mr_status":
        pat = _STATUS_SUPPORT.get(str(fv.normalized_value))
        text = " ".join(quotes)
        hits = list(re.finditer(pat, text)) if pat else []
        if fv.normalized_value != MRStatus.NOT_STATED.value:  # "is not MR Safe" does not state MR Safe
            hits = [m for m in hits if not _STATUS_NEGATION_RE.search(text[max(0, m.start() - 30) : m.start()])]
        if not hits:
            reasons.append("parse: quote does not state this MR status")
    else:
        spec = FIELD_SPECS[c.field]
        if spec.kind == "numeric":
            reasons += _numeric_support(spec, fv, quotes)
            if not reasons:
                reasons += _context_ok(spec, quotes)
            reasons += _plausible(spec, fv)
        elif spec.kind == "b0":
            reasons += _b0_support(fv, quotes)
            reasons += _plausible(spec, fv)
        elif spec.kind == "categorical":
            vals = fv.normalized_value if isinstance(fv.normalized_value, list) else [fv.normalized_value]
            table = _TOKEN_SUPPORT.get(c.field, {})
            text = " ".join(quotes)
            for v in vals:
                pat = table.get(str(v))
                if pat is None:
                    reasons.append(f"parse: '{v}' is not a recognised {spec.label.lower()} value")
                elif not re.search(pat, text):
                    reasons.append(f"parse: quote does not support {spec.label.lower()} '{v}'")
                elif str(v) == "any" and _RESTRICTIVE_RE.search(text):
                    reasons.append(f"parse: quote restricts the {spec.label.lower()}; it does not permit any")
            if not reasons:  # the quote must be about this quantity ("Any transmit coil" is not an RF excitation)
                reasons += _context_ok(spec, quotes)
        elif spec.kind == "bool":
            if bool(fv.normalized_value) and not FALLBACK_RE.search(" ".join(quotes)):
                reasons.append("parse: quote does not state that SAR applies only without B1+rms")
        else:  # text
            if c.field == "scan_region_restrictions" and fv.normalized_value == "unrestricted":
                if not _UNRESTRICTED_REGION_RE.search(" ".join(quotes)):
                    reasons.append("parse: quote does not state that scan regions are unrestricted")
    fv.evidence = verified
    c.accepted = not reasons
    c.reasons = reasons
    return c


_SCOPE_CUE = re.compile(
    r"(?:\b[Ff]or|\b[Aa]t)\s+(?:an?\s+|the\s+)?(\d+(?:\.\d+)?)\s*-?\s*T(?:esla)?\b"
    r"(?:\s+(?:MRI?|MR|[Ss]canners?|[Ss]ystems?|[Ee]nvironments?))*\s*:"
)


def _quote_scope(doc: DocumentText, ev: Evidence, fv: FieldValue) -> float | None:
    """Field strength named by a 'For 1.5T MR Scanner:' cue on the quote's line, before the value."""
    page_text = doc.page_text(ev.page)
    line_start = page_text.rfind("\n", 0, ev.char_start) + 1
    end = ev.char_end
    target = _num(fv.value)
    if target is not None:
        for v, s, _e in find_numbers(ev.quote):
            if values_equal(v, target):
                end = ev.char_start + s
                break
    cues = list(_SCOPE_CUE.finditer(page_text[line_start:end]))
    return round_sig(float(cues[-1].group(1))) if cues else None


def _scope_check(c: Candidate, doc: DocumentText, doc_b0: set[float]) -> list[str]:
    """6. Scope check: a value quoted from a statement scoped to one field strength must not apply to others."""
    applicable = {round_sig(float(v)) for v in c.scope_b0} if c.scope_b0 else doc_b0
    if not applicable:
        return []
    for ev in c.fv.evidence:
        f = _quote_scope(doc, ev, c.fv)
        if f is not None and applicable - {f}:
            return [f"scope: quote is stated for {f:g} T, value applied to {sorted(applicable)} T"]
    return []


def verify_output(out: ExtractionOutput, doc: DocumentText) -> VerificationReport:
    """Verify all candidates of an extraction in place."""
    rep = VerificationReport()
    for c in out.all_candidates():
        verify_candidate(c, doc)
    doc_b0: set[float] = set()
    for c in out.candidates:
        if c.field == "b0_tesla" and c.accepted:
            vals = c.fv.normalized_value if isinstance(c.fv.normalized_value, list) else [c.fv.normalized_value]
            doc_b0 |= {round_sig(float(v)) for v in vals if _num(v) is not None}
    for c in out.all_candidates():
        if c.accepted and c.field not in ("mr_status", "b0_tesla"):
            reasons = _scope_check(c, doc, doc_b0)
            if reasons:
                c.accepted, c.reasons = False, reasons
        rep.add(bool(c.accepted), c.reasons)
    out.verified = True
    out.meta["verification"] = {
        "n_candidates": rep.n_candidates,
        "n_accepted": rep.n_accepted,
        "n_rejected": rep.n_rejected,
        "reasons": rep.reasons,
    }
    return rep
