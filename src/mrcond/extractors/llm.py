"""LLM extractor with JSON-schema-constrained output and mandatory verbatim quotes.

The model sees only MRI-relevant statements of the document (keyword
retrieval, page markers kept) and must return, for every condition, the value
and unit *as written* plus a verbatim quote. MRCond - not the model - does all
unit conversion and normalisation. Quotes are located in the source text to
build evidence (page + character offsets); quotes that cannot be located keep
``verified=False`` and are rejected by the verifier in ``llm+verifier`` mode.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..ingest import DocumentText, segment_page
from ..schema import FIELD_SPECS, Evidence, MRStatus
from ..textnorm import find_all
from ..units import find_quantities, normalize_unit, parse_number, round_sig
from .backends import GenerationRequest, LLMBackend, backend_from_env
from .base import Candidate, ExtractionOutput, extracted

__all__ = ["LLMExtractor", "OUTPUT_SCHEMA", "SYSTEM_PROMPT", "build_context", "PROMPT_VERSION"]

PROMPT_VERSION = "v2"

LLM_FIELDS = [f for f in FIELD_SPECS if f not in ("b0_tesla", "other_conditions")]

OUTPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "mr_status": {"type": "string", "enum": [s.value for s in MRStatus]},
        "mr_status_quote": {"type": "string"},
        "condition_sets": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "field_strengths_tesla": {"type": "array", "items": {"type": "number"}},
                    "field_strength_quote": {"type": "string"},
                    "conditions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "field": {"type": "string", "enum": LLM_FIELDS},
                                "value": {"type": "string"},
                                "unit": {"type": "string"},
                                "quote": {"type": "string"},
                            },
                            "required": ["field", "value", "unit", "quote"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["field_strengths_tesla", "field_strength_quote", "conditions"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["mr_status", "mr_status_quote", "condition_sets"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You extract MRI safety conditions from medical device labeling. Use ONLY the provided text and never "
    "guess. For every value, copy a short exact quote (5 to 20 words) from the text that contains the value. "
    "If a condition is not stated, leave it out. Answer with compact JSON only."
)

FIELD_GLOSSARY = """Allowed `field` names - meaning (typical unit):
- max_sfg_t_per_m - maximum spatial field gradient (T/m or gauss/cm)
- max_slew_rate_t_per_m_per_s - maximum gradient slew rate per axis (T/m/s)
- b1rms_max_uT - maximum B1+rms (uT)
- wb_sar_max_w_per_kg - maximum whole-body averaged SAR (W/kg)
- head_sar_max_w_per_kg - maximum head averaged SAR (W/kg)
- sar_is_b1rms_fallback - value "true" if the SAR limit only applies when B1+rms is not available
- max_active_scan_time_min - maximum active or continuous scan time (minutes)
- wait_time_min - required wait or cooling time between scans (minutes)
- operating_mode - highest permitted mode, e.g. Normal Operating Mode
- rf_excitation - RF excitation, e.g. Circularly Polarized (CP)
- transmit_coil - permitted RF transmit coil(s), e.g. body coil, head transmit/receive coil, any
- receive_coil - permitted RF receive coil(s)
- nucleus - nuclei that may be imaged
- scanner_type - e.g. cylindrical
- b0_orientation - e.g. horizontal
- scan_region_restrictions - allowed scan regions, landmarks or exclusion zones
- device_configuration - required device setting during MRI, e.g. Stimulation OFF
- artifact_extent_mm - extent of the image artifact (mm)"""

EXAMPLE = (
    'Example. TEXT: "Static magnetic field of 1.5 T only. Maximum whole-body SAR of 2 W/kg in Normal Operating '
    'Mode." OUTPUT: {"mr_status":"Not stated","mr_status_quote":"","condition_sets":[{"field_strengths_tesla":[1.5],'
    '"field_strength_quote":"Static magnetic field of 1.5 T only","conditions":[{"field":"wb_sar_max_w_per_kg",'
    '"value":"2","unit":"W/kg","quote":"Maximum whole-body SAR of 2 W/kg"},{"field":"operating_mode",'
    '"value":"Normal Operating Mode","unit":"","quote":"in Normal Operating Mode"}]}]}'
)

USER_TEMPLATE = """{glossary}

Rules:
- `value` is the number or words exactly as written in the TEXT (for example "2" or "Normal Operating Mode"),
  never the field description; `unit` is the unit as written; do not convert units.
- Use one condition set per group of static field strengths that share the same conditions. If limits differ by
  field strength, make one condition set per field strength.
- Do not report temperature rise, test methods or test results; only conditions for scanning patients.
- mr_status is the MRI safety status of the device ("Not stated" if absent).

{example}

TEXT:
{context}
"""

_RELEVANT = re.compile(
    r"MRI?\b|[Tt]esla|\d\s*T\b|SAR|W/kg|B\s*1|μT|µT|uT|T/m|G/cm|[Gg]auss|[Ss]lew|[Gg]radient|[Cc]oil|[Tt]ransmit|"
    r"[Rr]eceive|[Oo]perating [Mm]ode|[Ss]can|[Aa]rtifact|[Cc]onditional|[Uu]nsafe|[Nn]ucle|[Cc]ylindrical|"
    r"[Hh]orizontal|[Ll]andmark|[Ee]xclusion|[Ss]timulation|[Ww]ait|[Mm]inutes"
)


def build_context(doc: DocumentText, max_chars: int = 6000) -> str:
    """Relevant statements in document order, with ``[[page N]]`` markers, within `max_chars`."""
    total = sum(len(p.text) for p in doc.pages)
    if total <= max_chars:
        return "\n".join(f"[[page {p.number}]]\n{p.text.strip()}" for p in doc.pages)
    scored = []
    for p in doc.pages:
        for i, seg in enumerate(segment_page(p.text, p.number)):
            score = len(_RELEVANT.findall(seg.text))
            if score:
                scored.append((score, p.number, i, seg.text))
    keep = sorted(scored, key=lambda x: (-x[0], x[1], x[2]))
    chosen, used = [], 0
    for item in keep:
        if used + len(item[3]) + 1 > max_chars:
            continue
        chosen.append(item)
        used += len(item[3]) + 1
    chosen.sort(key=lambda x: (x[1], x[2]))
    lines, cur_page = [], None
    for _, pn, _, text in chosen:
        if pn != cur_page:
            lines.append(f"[[page {pn}]]")
            cur_page = pn
        lines.append(" ".join(text.split()))
    return "\n".join(lines)


def parse_json_lenient(raw: str) -> dict:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {}


# --------------------------------------------------------------------------------------------
# value normalisation (model output -> MRCond values)
# --------------------------------------------------------------------------------------------

_CAT_MAP: dict[str, list[tuple[str, str]]] = {
    "transmit_coil": [
        ("any", r"\bany\b|no .*restrict|unrestricted|\ball\b"),
        ("body", r"body"),
        ("head", r"head"),
        ("extremity", r"extremit|knee|wrist|ankle|foot"),
        ("local", r"local|surface"),
    ],
    "receive_coil": [
        ("any", r"\bany\b|no .*restrict|unrestricted|\ball\b"),
        ("body", r"body"),
        ("head", r"head"),
        ("extremity", r"extremit|knee|wrist"),
        ("local", r"local|surface|receive-only"),
    ],
    "rf_excitation": [
        ("CP", r"circular|\bcp\b|quadrature"),
        ("MC-2", r"mc-?2|multi-?\s?channel|parallel|ptx"),
        ("any", r"\bany\b|no restriction"),
    ],
    "nucleus": [
        ("31P", r"31p|phosph"),
        ("23Na", r"23na|sodium"),
        ("13C", r"13c|carbon"),
        ("19F", r"19f|fluor"),
        ("1H", r"1h|hydrogen|proton|no additional"),
    ],
    "scanner_type": [("cylindrical", r"cylind|closed"), ("open", r"open"), ("any", r"\bany\b")],
    "b0_orientation": [("horizontal", r"horizontal"), ("vertical", r"vertical"), ("any", r"\bany\b")],
}


def _norm_categorical(field: str, value: str):
    v = value.lower()
    if field == "operating_mode":
        if re.search(r"first", v):
            return "first_level"
        if re.search(r"normal", v):
            return "normal"
        return None
    toks = [tok for tok, pat in _CAT_MAP.get(field, []) if re.search(pat, v)]
    if "any" in toks and re.search(r"(?<![-\w])only\b|prohibit|\bdo not\b|\bmust not\b|\bnever\b", v):
        toks.remove("any")  # "body coil only; all other coils prohibited" does not permit any coil
    if field == "nucleus":
        toks = sorted(set(toks) | {"1H"}) if toks else []
        return toks or None
    if field in ("scanner_type", "b0_orientation"):
        return toks[0] if toks else None
    if "any" in toks:
        return ["any"]
    return sorted(set(toks)) or None


@dataclass
class _Located:
    evidence: Evidence
    found: bool


def _b0_comparator(quote: str, b0s: list[float]) -> str:
    """The output schema lists field strengths; the quote tells whether lower ones are permitted ("3 T or less")."""
    from .rule import B0_LE_AFTER, B0_LE_BEFORE

    top = max(b0s)
    for q in find_quantities(quote):
        if q.dimension != "field" or abs(q.canonical_value - top) > 1e-6:
            continue
        if B0_LE_BEFORE.search(quote[max(0, q.start - 60) : q.start]) or B0_LE_AFTER.search(quote[q.end : q.end + 30]):
            return "le"
    return "in"


def _locate(doc: DocumentText, quote: str) -> _Located:
    for p in doc.pages:
        spans = find_all(p.text, quote)
        if spans:
            s, e = spans[0]
            return _Located(
                Evidence(doc_id=doc.doc_id, page=p.number, char_start=s, char_end=e, quote=p.text[s:e]), True
            )
    first = doc.pages[0].number if doc.pages else 1
    return _Located(
        Evidence(doc_id=doc.doc_id, page=first, char_start=0, char_end=0, quote=quote, verified=False), False
    )


class LLMExtractor:
    """Schema-constrained LLM extraction with mandatory quotes."""

    def __init__(
        self,
        backend: LLMBackend,
        model_tag: str | None = None,
        *,
        max_context_chars: int = 6000,
        temperature: float = 0.0,
        seed: int | None = 0,
        run: int = 0,
        max_tokens: int = 1000,
    ) -> None:
        self.backend = backend
        self.model_tag = model_tag or backend.name
        self.max_context_chars = max_context_chars
        self.temperature, self.seed, self.run, self.max_tokens = temperature, seed, run, max_tokens
        self.name = f"llm:{self.model_tag}"

    @classmethod
    def from_config(cls, backend: LLMBackend | None = None, **kwargs) -> LLMExtractor:
        return cls(backend or backend_from_env(), **kwargs)

    def request(self, doc: DocumentText) -> GenerationRequest:
        context = build_context(doc, self.max_context_chars)
        user = USER_TEMPLATE.format(glossary=FIELD_GLOSSARY, example=EXAMPLE, context=context)
        return GenerationRequest(
            SYSTEM_PROMPT,
            user,
            OUTPUT_SCHEMA,
            doc.doc_id,
            self.run,
            self.temperature,
            self.seed,
            self.max_tokens,
            {"prompt_version": PROMPT_VERSION},
        )

    def extract(self, doc: DocumentText) -> ExtractionOutput:
        req = self.request(doc)
        raw = self.backend.generate(req)
        data = parse_json_lenient(raw)
        out = ExtractionOutput(
            doc_id=doc.doc_id,
            extractor=self.name,
            raw=data,
            meta={"parse_ok": bool(data), "prompt_version": PROMPT_VERSION},
        )
        self._convert(doc, data, out)
        out.meta["value_from_quote"] = sum(c.source == "llm:value-from-quote" for c in out.candidates)
        return out

    def _convert(self, doc: DocumentText, data: dict, out: ExtractionOutput) -> None:
        status = data.get("mr_status")
        if isinstance(status, str) and status in {s.value for s in MRStatus}:
            loc = _locate(doc, str(data.get("mr_status_quote", "")))
            out.status_candidates.append(
                Candidate("mr_status", extracted(status, None, status, None, "eq", [loc.evidence]), None, "llm:status")
            )
        sets = data.get("condition_sets") or []
        if not isinstance(sets, list):
            return
        for cs in sets:
            if not isinstance(cs, dict):
                continue
            b0s = []
            for v in cs.get("field_strengths_tesla") or []:
                try:
                    b0s.append(round_sig(float(v)))
                except (TypeError, ValueError):
                    continue
            b0s = sorted(set(b0s))
            scope = tuple(b0s) if b0s and len(sets) > 1 else None
            if b0s:
                quote = str(cs.get("field_strength_quote", ""))
                loc = _locate(doc, quote)
                out.candidates.append(
                    Candidate(
                        "b0_tesla",
                        extracted(b0s, "T", b0s, "T", _b0_comparator(quote, b0s), [loc.evidence]),
                        None,
                        "llm:b0",
                    )
                )
            for cond in cs.get("conditions") or []:
                if not isinstance(cond, dict):
                    continue
                c = self._condition(doc, cond, scope)
                if c is not None:
                    out.candidates.append(c)

    def _condition(self, doc: DocumentText, cond: dict, scope) -> Candidate | None:
        field = cond.get("field")
        if field not in FIELD_SPECS or field in ("b0_tesla",):
            return None
        spec = FIELD_SPECS[field]
        value = str(cond.get("value", "")).strip()
        unit = str(cond.get("unit", "")).strip() or None
        loc = _locate(doc, str(cond.get("quote", "")))
        ev = [loc.evidence]
        if not value:
            return None
        if spec.kind == "numeric":
            if re.search(r"no limit|unlimited|no restriction", value, re.IGNORECASE):
                return Candidate(field, extracted(value, None, None, spec.canonical_unit, "any", ev), scope, "llm")
            nums = re.findall(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:[.,]\d+)?", value)
            if not nums:
                # the model located the evidence but did not copy the number: parse it from its own quote
                qs = [q for q in find_quantities(str(cond.get("quote", ""))) if q.dimension == spec.dimension]
                vals = {q.canonical_value for q in qs}
                if len(vals) != 1:
                    return None
                q0 = qs[0]
                comparator = {"upper": "le", "lower": "ge", "info": "eq"}[spec.direction or "info"]
                return Candidate(
                    field,
                    extracted(q0.value, q0.raw_unit.strip(), q0.canonical_value, spec.canonical_unit, comparator, ev),
                    scope,
                    "llm:value-from-quote",
                )
            num = parse_number(nums[0])
            info = normalize_unit(unit) if unit else None
            if info is None:
                # unit embedded in the value ("2 W/kg")?
                qs = find_quantities(value)
                if qs:
                    info = normalize_unit(qs[0].unit)
                    unit = qs[0].raw_unit
            if info is not None and info.dimension == spec.dimension:
                normalized = round_sig(num * info.factor)
            else:
                normalized = num  # unit missing or of the wrong dimension: the verifier decides
            comparator = {"upper": "le", "lower": "ge", "info": "eq"}[spec.direction or "info"]
            return Candidate(field, extracted(num, unit, normalized, spec.canonical_unit, comparator, ev), scope, "llm")
        if spec.kind == "bool":
            truthy = value.lower() in ("true", "yes", "1")
            if not truthy:
                return None
            return Candidate(field, extracted(True, None, True, None, "eq", ev), scope, "llm")
        if spec.kind == "categorical":
            norm = _norm_categorical(field, value)
            if norm is None:
                return None
            comparator = "le" if field == "operating_mode" else ("eq" if isinstance(norm, str) else "in")
            return Candidate(field, extracted(value, None, norm, None, comparator, ev), scope, "llm")
        # text
        norm = value
        if field == "scan_region_restrictions" and re.search(
            r"any landmark|no restriction|full[\s-]*body|anywhere", value, re.IGNORECASE
        ):
            norm = "unrestricted"
        return Candidate(field, extracted(value, None, norm, None, "eq", ev), scope, "llm")
