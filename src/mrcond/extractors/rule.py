"""Rule-based extractor: regular expressions + units + context windows.

Pipeline per document:

1. segment pages into statements (table rows, list items, sentences)
2. detect MR status statements
3. classify each static-field mention as *permitted field strength*,
   *scope* of the statement ("For 3T MR Scanner: B1+rms <= 1.7 uT"), table
   *column header* ("Parameter | 1.5 T | 3 T"), *section heading*, or ignore
   it (negated: "do not scan at 7 T"; test description: "artifact ... at 3 T")
4. assign every other quantity (W/kg, µT, T/m, G/cm, T/m/s, minutes, mm) to a
   field using the nearest keyword in the same clause
5. detect categorical/text conditions (coils, RF excitation, operating mode,
   nucleus, scanner type, B0 orientation, device configuration, scan regions)

Every candidate carries a verbatim evidence span (<= 25 words) taken from the
page text; the verifier and the conservative merge run afterwards.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..ingest import DocumentText, Segment, segment_document
from ..schema import FIELD_SPECS, MRStatus
from ..units import Quantity, find_quantities, find_word_durations
from .base import Candidate, ExtractionOutput, evidence_from_segment, extracted

__all__ = ["RuleExtractor"]

# ---------------------------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------------------------

STATUS_PATTERNS: list[tuple[MRStatus, re.Pattern[str]]] = [
    (MRStatus.MR_CONDITIONAL, re.compile(r"\bMRI?\s*-?\s*(?:[Cc]onditional|CONDITIONAL)\b")),
    (MRStatus.MR_UNSAFE, re.compile(r"\bMRI?\s*-?\s*(?:[Uu]nsafe|UNSAFE)\b")),
    (MRStatus.MR_SAFE, re.compile(r"\bMRI?\s*-?\s*(?:[Ss]afe|SAFE)\b(?!\s*[Tt]y|ty)")),
    (
        MRStatus.NOT_STATED,
        re.compile(
            r"not been (?:evaluated|tested) for safety (?:in|and compatibility in) the MR|"
            r"[Ss]afety in (?:the )?MRI? (?:environment )?(?:has )?[Nn]ot (?:been )?[Ee]valuated"
        ),
    ),
]

NEG_RE = re.compile(
    r"\b(?:not|never|unsafe|untested|other than|except|excluding|cannot|can't|don't|do not|avoid|prohibited|"
    r"contraindicated|unknown)\b",
    re.IGNORECASE,
)
TEST_CONTEXT_RE = re.compile(
    r"[Aa]rtifact|[Aa]rtefact|[Tt]emperature|[Hh]eating|[Hh]eat\b|°C|ºC|˚C|\bdegrees? C|[Nn]on-?clinical|"
    r"[Tt]ested|[Tt]esting|[Pp]hantom|[Ii]n vitro|[Ww]orst[\s-]*case|[Dd]eflection|[Tt]orque|[Dd]isplacement"
)
STRONG_B0_CUE = re.compile(
    r"[Ss]tatic\s+(?:[Mm]agnetic\s+)?[Ff]ield|[Ff]ield\s+[Ss]trength|\bB\s*0\b|B₀|MRI?\s*-?\s*[Cc]onditional|"
    r"(?:can|may)\s+(?:be\s+)?(?:safely\s+)?(?:scanned|imaged|undergo)"
)
NEAR_ALLOWED_CUE = re.compile(
    r"(?:[Ss]tatic\s+(?:[Mm]agnetic\s+)?[Ff]ield(?:\s+[Ss]trength)?|[Ff]ield\s+[Ss]trength|\bB\s*0\b\)?|B₀\)?)"
    r"[^.;]{0,25}$"
)
B0_LE_BEFORE = re.compile(
    r"(?:≤|<=|[Uu]p to|[Mm]aximum|[Mm]ax\.?|not (?:to )?exceed\w*|no (?:more|greater) than|"
    r"less than or equal to)\s*(?:of\s*)?$"
)
B0_LE_AFTER = re.compile(r"^\s*(?:or (?:less|lower|below|under)|and (?:below|lower|less)|maximum|max)\b")
SCOPE_PREP = re.compile(r"(?:\b[Aa]t|\b[Ff]or|\b[Ii]n|\b[Oo]n|\b[Ww]ith|\b[Uu]sing)\s+(?:an?\s+|the\s+)?$")
MRCOND_NEAR = re.compile(r"MRI?\s*-?\s*[Cc]onditional[^.;:]{0,20}$")
SCOPE_AFTER = re.compile(r"^\s*(?:MRI?\s+)?(?:[Ss]canners?|[Ss]ystems?)?\s*:")

SAR_ANY = re.compile(r"SAR|[Ss]pecific [Aa]bsorption")
HEAD_KW = re.compile(r"[Hh]ead")
WB_KW = re.compile(r"[Ww]hole[\s-]*[Bb]ody|[Ww]hole[\s-]*-?\s*[Bb]ody|\bWB\b|[Bb]ody[\s-]*averaged|[Ww]hole[\s-]body")
PARTIAL_KW = re.compile(r"[Pp]artial[\s-]*[Bb]ody|[Ll]ocal(?:ized)?\s+SAR|[Ee]xtremit")
B1_KW = re.compile(r"B\s*1|B₁")
SFG_KW = re.compile(r"[Ss]patial|SFG|dB/d[rxyz]|[Gg]radient\s+(?:[Mm]agnetic\s+)?[Ff]ield|[Ff]ield\s+[Gg]radient")
GRAD_KW = re.compile(r"[Gg]radient")
PRODUCT_KW = re.compile(r"[Pp]roduct|T\s*²\s*/\s*m|T2/m")
WAIT_KW = re.compile(r"[Ww]ait|[Cc]ool(?:ing|-?down)?|[Bb]reak|[Pp]ause|between\s+(?:scans|sessions|sequences|exams)")
SCAN_KW = re.compile(r"[Ss]can|[Aa]ctive|[Cc]ontinuous|\bRF\b|[Ss]equence|[Ee]xam|[Ss]ession|[Dd]uration|[Ii]maging")
WINDOW_AFTER = re.compile(r"^\s*-?\s*(?:period|window|interval|time period)\b")
POST_IMPLANT = re.compile(
    r"after\s+(?:the\s+)?(?:implant|implantation|surgery|placement)|post[\s-]*(?:implant|operative)"
)
ARTIFACT_KW = re.compile(r"[Aa]rtifact|[Aa]rtefact|[Ss]ignal void")
DEVICE_DIM_KW = re.compile(r"diameter|length|long\b|size|wide|thick|[Ss]tent of")

NUCLEUS_CTX = re.compile(
    r"[Nn]ucle(?:i|us)|[Pp]roton imaging|[Hh]ydrogen (?:proton )?imaging|1H imaging|"
    r"imaging of (?:hydrogen|protons)"
)
NUCLEUS_TOKENS = [
    ("31P", r"31P|³¹P|[Pp]hosphorus"),
    ("23Na", r"23Na|²³Na|[Ss]odium"),
    ("13C", r"13C|¹³C"),
    ("19F", r"19F|¹⁹F"),
    ("129Xe", r"129Xe|[Xx]enon"),
]

SCANNER_CYL = re.compile(r"[Cc]ylindrical|[Cc]losed[\s-]*bore")
SCANNER_OPEN = re.compile(r"\b[Oo]pen[\s-]*(?:bore|MR|MRI|system|scanner)s?")
ORIENT_RE = re.compile(r"\b([Hh]orizontal|[Vv]ertical)(?:ly)?\b")

CP_RE = re.compile(r"[Cc]ircularly[\s-]*[Pp]olari[sz]ed|\(CP\)|\bCP\b(?:[\s-]*mode)?|[Qq]uadrature (?:mode|excitation)")
MC_RE = re.compile(r"\bMC-?2\b|[Mm]ulti-?\s?channel|[Pp]arallel[\s-]*[Tt]ransmi\w*|\bpTx\b")

TRANSMIT_CTX = re.compile(r"[Tt]ransmit|T/R\b|[Tt]ransmit/[Rr]eceive|RF\s+[Cc]oil\s+[Tt]ype")
TX_ANY = re.compile(
    r"[Nn]o\s+(?:RF\s+)?[Tt]ransmit[\s-]*[Cc]oil\s+restrictions?|[Nn]o\s+restrictions?\s+on\s+(?:the\s+)?"
    r"(?:RF\s+)?transmit|[Aa]ny\s+(?:RF\s+)?[Tt]ransmit|[Tt]ransmit[\s-]*[Cc]oil(?:\s+[Tt]ype)?\s*:?\s*"
    r"(?:[Aa]ny|[Nn]o restriction)"
)
TX_BODY = re.compile(
    r"[Ii]ntegrated\s+(?:[Ww]hole[\s-]*)?[Bb]ody|[Bb]ody\s+(?:RF\s+)?(?:[Tt]ransmit\w*\s+)?[Cc]oil|"
    r"[Bb]ody\s+[Tt]ransmit|[Ww]hole[\s-]*[Bb]ody\s+(?:RF\s+)?[Tt]ransmit|[Qq]uadrature\s+[Bb]ody"
)
TX_HEAD = re.compile(
    r"[Hh]ead\s+(?:RF\s+)?(?:[Tt]ransmit(?:/|-|\s+and\s+)?[Rr]eceive|[Tt]ransmit|T/R)?\s*[Cc]oil|"
    r"[Hh]ead\s+[Tt]ransmit"
)
TX_EXT = re.compile(
    r"[Ee]xtremity\s+(?:RF\s+)?(?:[Tt]ransmit(?:/|-)?[Rr]eceive|[Tt]ransmit|T/R)?\s*[Cc]oil|"
    r"[Kk]nee\s+(?:T/R\s+|[Tt]ransmit/[Rr]eceive\s+)?[Cc]oil"
)
TX_LOCAL = re.compile(r"[Ll]ocal\s+(?:RF\s+)?[Tt]ransmit")
RX_CTX = re.compile(r"[Rr]eceive")
RX_ANY = re.compile(
    r"[Rr]eceive[\s-]*(?:[Cc]oil)?(?:\s+[Tt]ype)?\s*:?\s*[Aa]ny\b|[Aa]ny\s+(?:RF\s+)?[Rr]eceive|"
    r"[Nn]o\s+(?:RF\s+)?[Rr]eceive[\s-]*[Cc]oil\s+restrictions?|[Rr]eceive[\s-]*only\s+coils?\s+"
    r"(?:may|can)\s+be\s+used|[Aa]ny\s+receive-only"
)
RX_TR_HEAD = re.compile(r"[Hh]ead\s+(?:RF\s+)?(?:[Tt]ransmit(?:/|-|\s+and\s+)[Rr]eceive|T/R)\s*[Cc]oil")
RX_TR_EXT = re.compile(r"[Ee]xtremity\s+(?:RF\s+)?(?:[Tt]ransmit(?:/|-|\s+and\s+)[Rr]eceive|T/R)\s*[Cc]oil")

MODE_NORMAL = re.compile(r"[Nn]ormal\s+(?:[Oo]perating\s+)?[Mm]ode")
MODE_FIRST = re.compile(r"[Ff]irst[\s-]*[Ll]evel(?:\s+[Cc]ontrolled)?(?:\s+[Oo]perating)?(?:\s+[Mm]ode)?")

DEVICE_CFG = re.compile(
    r"[Dd]evice\s+[Cc]onfiguration\s*:?\s*(?P<v>[^.;\n]{2,60})|"
    r"(?P<stim>[Ss]timulation\s+(?:OFF|[Oo]ff|turned off|must be (?:turned )?off|disabled))|"
    r"(?P<ther>[Tt]herapy\s+(?:OFF|[Oo]ff|turned off|disabled))|"
    r"(?P<mode>(?:MRI?\s+SureScan|SureScan\s+(?:mode|MRI)?|MRI?[\s-]+(?:[Mm]ode|[Ss]ettings)|MR[\s-]+[Cc]onditional\s+mode|"
    r"MRI\s+[Pp]rotection\s+[Mm]ode|MRI\s+[Mm]ode))|"
    r"(?P<prog>[Pp]rogram(?:med)?\s+(?:the\s+device\s+)?(?:to|into|in)\s+(?:the\s+)?[\w\s-]{0,30}?[Mm]ode)"
)
REGION_KW = re.compile(
    r"[Ss]can\s+[Rr]egions?|[Ee]xclusion\s+[Zz]ones?|[Ii]so-?cent(?:er|re)|[Ll]andmark|[Hh]ead\s+[Oo]nly|"
    r"[Ff]ull[\s-]*[Bb]ody\s+(?:scan|MRI|exam|eligib)\w*|[Ee]xtremit(?:y|ies)\s+only|[Aa]nywhere in the body|"
    r"[Pp]ositioned\s+(?:above|below|outside)|[Ii]maging\s+[Zz]one"
)
REGION_UNRESTRICTED = re.compile(
    r"[Aa]ny landmark|[Nn]o (?:exclusion zone|scan region restriction|landmark restriction)s?|"
    r"[Ff]ull[\s-]*[Bb]ody\s+(?:scan|MRI|exam|eligib)\w*|[Aa]nywhere in the body"
)
OTHER_PATTERNS = [
    re.compile(
        r"(?:at least|minimum of|wait)\s*\d+\s*(?:weeks?|days?|months?)\s*(?:after|following|post)[\s-]*"
        r"(?:implant\w*|surgery|placement|insertion)"
    ),
    re.compile(
        r"[Mm]onitor(?:ed|ing)?\s+(?:the\s+)?patient|[Pp]atient\s+(?:must|should)\s+be\s+(?:continuously\s+)?"
        r"monitored"
    ),
    re.compile(r"[Aa]bandoned\s+leads?|[Ff]ractured\s+leads?|[Ll]ead\s+fragments?|[Ll]ead\s+extenders?|[Aa]dapters?"),
]
SENTENCE_KW_FOR_OTHER = re.compile(r"MR|MRI|scan", re.IGNORECASE)

# which fields may be scoped by a sticky section heading ("At 3 T:") rather than an
# in-statement scope; everything else is treated as global (conservative)
STICKY_SCOPED_FIELDS = {
    "b1rms_max_uT",
    "wb_sar_max_w_per_kg",
    "head_sar_max_w_per_kg",
    "sar_is_b1rms_fallback",
    "max_active_scan_time_min",
    "wait_time_min",
    "operating_mode",
}

COLUMN_HEADER_WORDS = {
    "parameter",
    "parameters",
    "condition",
    "conditions",
    "field",
    "strength",
    "static",
    "magnetic",
    "b0",
    "(b0)",
    "b",
    "0",
    "value",
    "values",
    "limit",
    "limits",
    "|",
    "mr",
    "scanner",
    "scanners",
    "system",
    "systems",
    "-",
    "tesla",
    "t",
}


@dataclass
class _Ctx:
    sticky_scope: tuple[float, ...] | None = None
    sticky_left: int = 0
    columns: tuple[float, ...] | None = None
    columns_left: int = 0


@dataclass
class _B0Mention:
    q: Quantity
    role: str  # allowed | scope | ignore
    comparator: str = "in"


@dataclass
class _Analysis:
    mentions: list[_B0Mention] = field(default_factory=list)
    others: list[Quantity] = field(default_factory=list)


def _clause_start(t: str, pos: int) -> int:
    """Start of the clause containing `pos` (clauses end at ';' or '. ')."""
    k = max(t.rfind(";", 0, pos), t.rfind(". ", 0, pos))
    return k + 1 if k >= 0 else 0


def _negated(t: str, pos: int) -> bool:
    """A negation word within 45 characters before `pos`, in the same clause."""
    cs = max(t.rfind(";", 0, pos), t.rfind(",", 0, pos), t.rfind(". ", 0, pos))
    window = t[max(cs + 1, pos - 45) : pos]
    return bool(NEG_RE.search(window))


NEG_AFTER_RE = re.compile(
    r"^[^;,.]{0,25}?\b(?:(?:has|have|is|are|was|were)\s+not|not\s+(?:been\s+)?(?:evaluated|tested|permitted|allowed|"
    r"approved)|untested|unsafe|prohibited|contraindicated|must not|should not)\b",
    re.IGNORECASE,
)


def _negated_after(t: str, end: int) -> bool:
    """A negation right after a mention ("7 T has not been evaluated")."""
    return bool(NEG_AFTER_RE.search(t[end : end + 60]))


CONDITION_KW = re.compile(
    r"SAR|[Ss]pecific [Aa]bsorption|B\s*1|[Oo]perating [Mm]ode|[Cc]oil|[Ss]lew|[Gg]radient|"
    r"[Ss]can (?:time|duration)|[Ww]ait|[Ee]xcitation|[Ll]andmark|[Ee]xclusion"
)
PLACEHOLDER_RE = re.compile(r"\bnot (?:specified|stated|applicable|evaluated)\b|\bN/?A\b|(?<=\s)[—–-](?=\s|$)")


def _barrier(t: str, a: int, b: int) -> bool:
    """True if a clause boundary separates positions a < b (a ';' not followed by a B1+rms fallback clause)."""
    from ..verifier import FALLBACK_RE

    seg = t[a:b]
    if ". " in seg:
        return True
    for m in re.finditer(";", seg):
        k = a + m.end()
        nxt = t.find(";", k)
        clause = t[k : nxt if nxt >= 0 else len(t)]
        if not FALLBACK_RE.search(clause):
            return True
    return False


class RuleExtractor:
    """Deterministic regex/units extractor (no model)."""

    name = "rule"

    def extract(self, doc: DocumentText) -> ExtractionOutput:
        out = ExtractionOutput(doc_id=doc.doc_id, extractor=self.name)
        ctx = _Ctx()
        for seg in segment_document(doc):
            self._segment(doc, seg, ctx, out)
        return out

    # -----------------------------------------------------------------------------------------
    def _segment(self, doc: DocumentText, seg: Segment, ctx: _Ctx, out: ExtractionOutput) -> None:
        t = seg.text
        self._status(doc, seg, out)
        quantities = find_quantities(t)
        words = find_word_durations(t)
        b0_qs = [q for q in quantities if q.dimension == "field" and q.unit == "T"]
        others = [q for q in quantities if q.dimension in ("gradient", "slew", "b1", "sar", "time", "length")]
        others += [w for w in words if not any(o.start <= w.start < o.end for o in others)]
        others.sort(key=lambda q: q.start)
        test_ctx = bool(TEST_CONTEXT_RE.search(t))
        strong_cue = bool(STRONG_B0_CUE.search(t))

        mentions: list[_B0Mention] = []
        for q in b0_qs:
            before = t[max(0, q.start - 60) : q.start]
            after = t[q.end : q.end + 30]
            comp = "le" if (B0_LE_BEFORE.search(before) or B0_LE_AFTER.search(after)) else "in"
            pre = t[max(0, q.start - 80) : q.start]
            scope_cue = bool(
                SCOPE_PREP.search(t[max(0, q.start - 25) : q.start])
                or SCOPE_AFTER.match(after)
                or not t[: q.start].strip()
                or t[: q.start].rstrip().endswith("(")
            )
            heading_like = t.rstrip().endswith(":") and len(t.split()) <= 8
            if _negated(t, q.start) or _negated_after(t, q.end):
                role = "ignore"
            elif test_ctx and not strong_cue:
                role = "ignore"
            elif NEAR_ALLOWED_CUE.search(pre) or MRCOND_NEAR.search(pre) or comp == "le":
                role = "allowed"
            elif scope_cue and (others or (CONDITION_KW.search(t) and not heading_like)):
                role = "scope"
            else:
                role = "allowed"
            mentions.append(_B0Mention(q, role, comp))

        # column header / sticky heading detection (no other quantities in the statement)
        allowed_vals = [m for m in mentions if m.role == "allowed"]
        if allowed_vals and not others:
            residual = t
            for m in sorted(allowed_vals, key=lambda m: -m.q.start):
                residual = residual[: m.q.start] + " " + residual[m.q.end :]
            res_words = [w.lower().strip(":,()") for w in residual.split() if w.strip(":,()")]
            vals = tuple(sorted({m.q.canonical_value for m in allowed_vals}))
            if len(vals) >= 2 and all(w in COLUMN_HEADER_WORDS for w in res_words):
                ctx.columns, ctx.columns_left = tuple(m.q.canonical_value for m in allowed_vals), 30
                ctx.sticky_scope, ctx.sticky_left = None, 0
            elif (
                len(vals) == 1
                and not NEAR_ALLOWED_CUE.search(t[: allowed_vals[0].q.start])
                and (t.rstrip().endswith(":") or len(t.split()) <= 6)
                and not STRONG_B0_CUE.search(t.replace("MR Conditional", ""))
            ):
                ctx.sticky_scope, ctx.sticky_left = vals, 25
                ctx.columns, ctx.columns_left = None, 0
            elif STRONG_B0_CUE.search(t):
                # a new permitted-field-strength statement ends any section scope
                ctx.sticky_scope, ctx.sticky_left = None, 0

        for m in mentions:
            if m.role != "allowed":
                continue
            vals = [m.q.canonical_value]
            ev = evidence_from_segment(doc.doc_id, seg, max(0, m.q.start - 60), m.q.end + 20, (m.q.start, m.q.end))
            out.candidates.append(
                Candidate(
                    "b0_tesla",
                    extracted(vals if m.comparator == "in" else vals[0], "T", vals, "T", m.comparator, [ev]),
                    None,
                    "rule:b0",
                )
            )
        # merge evidence of several permitted values in one statement into one candidate
        self._merge_b0_candidates(out, seg)

        scopes = [m for m in mentions if m.role == "scope"]
        in_scope_style = "prefix"
        if scopes and others and scopes[0].q.start > others[0].start:
            in_scope_style = "postfix"

        def scope_for(pos: int, fieldname: str) -> tuple[float, ...] | None:
            if scopes:
                if in_scope_style == "prefix":
                    prev = [m for m in scopes if m.q.start < pos and not _barrier(t, m.q.end, pos)]
                    if prev:
                        return (prev[-1].q.canonical_value,)
                else:
                    nxt = [m for m in scopes if m.q.start > pos and not _barrier(t, pos, m.q.start)]
                    if nxt:
                        return (nxt[0].q.canonical_value,)
            if ctx.sticky_scope and fieldname in STICKY_SCOPED_FIELDS:
                return ctx.sticky_scope
            return None

        self._numeric(doc, seg, others, scope_for, ctx, out, test_ctx)
        self._categorical(doc, seg, scope_for, out, test_ctx)

        if ctx.sticky_left:
            ctx.sticky_left -= 1
            if ctx.sticky_left == 0:
                ctx.sticky_scope = None
        if ctx.columns_left:
            ctx.columns_left -= 1
            if ctx.columns_left == 0:
                ctx.columns = None

    # -----------------------------------------------------------------------------------------
    def _merge_b0_candidates(self, out: ExtractionOutput, seg: Segment) -> None:
        idx = [
            i
            for i, c in enumerate(out.candidates)
            if c.field == "b0_tesla"
            and c.fv.evidence
            and c.fv.evidence[0].page == seg.page
            and seg.start <= c.fv.evidence[0].char_start < seg.start + len(seg.text) + 1
            and c.source == "rule:b0"
        ]
        if len(idx) < 2:
            return
        group = [out.candidates[i] for i in idx]
        if len({c.fv.comparator for c in group}) != 1:
            return
        comp = group[0].fv.comparator
        vals = sorted({float(v) for c in group for v in c.fv.normalized_value})  # type: ignore[union-attr]
        s = min(c.fv.evidence[0].char_start for c in group)
        e = max(c.fv.evidence[0].char_end for c in group)
        rel_s, rel_e = s - seg.start, e - seg.start
        first_q = min(c.fv.evidence[0].char_start for c in group) - seg.start
        ev = evidence_from_segment(out.doc_id, seg, rel_s, rel_e, (first_q, rel_e))
        merged = Candidate(
            "b0_tesla",
            extracted(vals if comp == "in" else vals[0], "T", vals if comp == "in" else [min(vals)], "T", comp, [ev]),
            None,
            "rule:b0",
        )
        for i in sorted(idx, reverse=True):
            del out.candidates[i]
        out.candidates.append(merged)

    # -----------------------------------------------------------------------------------------
    def _status(self, doc: DocumentText, seg: Segment, out: ExtractionOutput) -> None:
        t = seg.text
        for status, pat in STATUS_PATTERNS:
            for m in pat.finditer(t):
                if status != MRStatus.NOT_STATED and _negated(t, m.start()):
                    continue
                if status == MRStatus.MR_SAFE and re.search(r"[Ss]afety", t[m.start() : m.end() + 3]):
                    continue
                ev = evidence_from_segment(doc.doc_id, seg, max(0, m.start() - 80), m.end() + 40, (m.start(), m.end()))
                out.status_candidates.append(
                    Candidate(
                        "mr_status",
                        extracted(status.value, None, status.value, None, "eq", [ev]),
                        None,
                        f"rule:status:{status.name}",
                    )
                )
                break

    # -----------------------------------------------------------------------------------------
    def _numeric(self, doc, seg, others, scope_for, ctx, out, test_ctx) -> None:
        t = seg.text
        assigned: list[tuple[str, Quantity, int, int]] = []  # field, q, ev_start, ev_end
        prev_end = 0
        for q in others:
            cs = _clause_start(t, q.start)
            win_start = max(cs, prev_end)
            before = t[win_start : q.start]
            after = t[q.end : q.end + 45]
            nxt = [o for o in others if o.start >= q.end]
            if nxt:
                after = t[q.end : min(nxt[0].start, q.end + 45)]
            fieldname, kw_span = self._classify(q, t, before, after, win_start, test_ctx)
            if (
                fieldname is None
                and assigned
                and assigned[-1][1].dimension == q.dimension
                and re.fullmatch(r"[\s≤<=≥>,;/|]*(?:(?:or|and)\s*[≤<=]?\s*)?", PLACEHOLDER_RE.sub(" ", before))
            ):
                # next cell of the same table row: "Whole-Body SAR ≤ 2 W/kg ≤ 0.4 W/kg"
                fieldname, kw_span = assigned[-1][0], None
            prev_end = q.end
            if fieldname is None:
                continue
            ks, ke = kw_span if kw_span else (q.start, q.end)
            ev_s = min(ks, q.start)
            if ev_s - cs <= 60:
                ev_s = cs
            ev_e = max(ke, q.end)
            assigned.append((fieldname, q, ev_s, ev_e))

        # column-wise tables: k values of one field in a row map to the k columns
        by_field: dict[str, list[tuple[Quantity, int, int]]] = {}
        for f, q, s, e in assigned:
            by_field.setdefault(f, []).append((q, s, e))
        for f, items in by_field.items():
            distinct: list[tuple[Quantity, int, int]] = []
            for q, s, e in items:
                if (
                    distinct
                    and abs(q.canonical_value - distinct[-1][0].canonical_value) < 1e-9
                    and re.fullmatch(r"[\s(/=]*", t[distinct[-1][0].end : q.start])
                ):
                    # same value in two unit systems: "30 T/m (3,000 G/cm)" -> one candidate, wider evidence
                    d = distinct[-1]
                    distinct[-1] = (d[0], d[1], max(d[2], e))
                    continue
                distinct.append((q, s, e))
            col_scopes: list[tuple[float, ...] | None] = [None] * len(distinct)
            if ctx.columns and scope_for(distinct[0][0].start, f) is None:
                row_start = _clause_start(t, distinct[0][0].start)
                holders = [m.start() for m in PLACEHOLDER_RE.finditer(t, row_start)]
                cells = sorted(
                    [("v", d[0].start, i) for i, d in enumerate(distinct)] + [("p", h, -1) for h in holders],
                    key=lambda c: c[1],
                )
                if len(cells) == len(ctx.columns) and len(cells) > 1:
                    for col, (kind, _, i) in zip(ctx.columns, cells):
                        if kind == "v":
                            col_scopes[i] = (col,)
            for (q, s, e), cscope in zip(distinct, col_scopes):
                spec = FIELD_SPECS[f]
                scope = cscope if cscope is not None else scope_for(q.start, f)
                focus = (q.start, q.end)
                if cscope is not None:
                    # evidence: row label + this value
                    s = _clause_start(t, distinct[0][0].start)
                ev = evidence_from_segment(doc.doc_id, seg, s, e, focus)
                comparator = {"upper": "le", "lower": "ge", "info": "eq"}[spec.direction or "info"]
                fv = extracted(q.value, q.raw_unit.strip(), q.canonical_value, spec.canonical_unit, comparator, [ev])
                out.candidates.append(Candidate(f, fv, scope, f"rule:{f}"))

        # explicit "no limit on scan duration"
        m = re.search(
            r"[Nn]o (?:time )?limit(?:s|ation)? on (?:the )?(?:scan|scanning|active scan)\w*\s*"
            r"(?:duration|time)?|[Ss]can [Dd]uration\s*:?\s*[Uu]nlimited",
            t,
        )
        if m:
            ev = evidence_from_segment(doc.doc_id, seg, 0, len(t), (m.start(), m.end()))
            out.candidates.append(
                Candidate(
                    "max_active_scan_time_min",
                    extracted("no limit", None, None, "min", "any", [ev]),
                    scope_for(m.start(), "max_active_scan_time_min"),
                    "rule:scan_time_any",
                )
            )
        # SAR applies only when B1+rms is not available
        from ..verifier import FALLBACK_RE

        fm = FALLBACK_RE.search(t)
        if fm and SAR_ANY.search(t):
            fs = _clause_start(t, fm.start())
            if fm.start() - fs > 60:
                fs = max(0, fm.start() - 20)
            ev = evidence_from_segment(doc.doc_id, seg, fs, fm.end() + 60, (fm.start(), fm.end()))
            out.candidates.append(
                Candidate(
                    "sar_is_b1rms_fallback",
                    extracted(True, None, True, None, "eq", [ev]),
                    scope_for(fm.start(), "sar_is_b1rms_fallback"),
                    "rule:fallback",
                )
            )

    def _classify(
        self, q: Quantity, t: str, before: str, after: str, win_start: int, test_ctx: bool
    ) -> tuple[str | None, tuple[int, int] | None]:
        def last(pat: re.Pattern[str], text: str, offset: int):
            ms = list(pat.finditer(text))
            return (ms[-1].start() + offset, ms[-1].end() + offset) if ms else None

        def first_after(pat: re.Pattern[str]):
            m = pat.match(after) if pat.pattern.startswith("^") else pat.search(after)
            return (q.end + m.start(), q.end + m.end()) if m else None

        d = q.dimension
        if d == "sar":
            if PARTIAL_KW.search(before):
                return None, None
            post = re.match(r"[ \t]*\(?[ \t]*([Hh]ead|[Ww]hole[\s-]*[Bb]ody|WB)", after)
            if post and not (HEAD_KW.search(before) or WB_KW.search(before) or SAR_ANY.search(before)):
                kw = (q.end + post.start(1), q.end + post.end(1))
                sar_after = SAR_ANY.search(after)
                if sar_after:
                    kw = (kw[0], max(kw[1], q.end + sar_after.end()))
                return (
                    "head_sar_max_w_per_kg" if post.group(1).lower().startswith("head") else "wb_sar_max_w_per_kg"
                ), kw
            h, w = last(HEAD_KW, before, win_start), last(WB_KW, before, win_start)
            s = last(SAR_ANY, before, win_start)
            if h and (not w or h[0] > w[0]):
                return "head_sar_max_w_per_kg", h
            if w:
                return "wb_sar_max_w_per_kg", w
            if s:
                return "wb_sar_max_w_per_kg", s
            # keyword after the value ("2 W/kg whole-body average SAR")
            w2 = first_after(WB_KW) or first_after(SAR_ANY)
            h2 = first_after(HEAD_KW)
            if h2 and (not w2 or h2[0] < w2[0]):
                return "head_sar_max_w_per_kg", h2
            if w2:
                sar_after = SAR_ANY.search(after)
                if sar_after:
                    w2 = (w2[0], max(w2[1], q.end + sar_after.end()))
                return "wb_sar_max_w_per_kg", w2
            return None, None
        if d == "b1":
            k = last(B1_KW, before, win_start) or first_after(B1_KW)
            return ("b1rms_max_uT", k) if k else (None, None)
        if d == "gradient":
            if PRODUCT_KW.search(before):
                return None, None
            k = last(SFG_KW, before, win_start)
            if k:
                return "max_sfg_t_per_m", k
            if q.unit == "G/cm" and GRAD_KW.search(t):
                g = last(GRAD_KW, t[: q.start], 0)
                return "max_sfg_t_per_m", g
            return None, None
        if d == "slew":
            k = last(re.compile(r"[Ss]lew"), before, win_start)
            return "max_slew_rate_t_per_m_per_s", k
        if d == "time":
            if test_ctx and re.search(r"[Tt]emperature|°C|ºC|[Hh]eating|[Rr]ise", t):
                return None, None
            if POST_IMPLANT.search(t[q.end : q.end + 40]):
                return None, None
            k = last(WAIT_KW, before, win_start) or first_after(
                re.compile(
                    r"^\s*(?:of\s+)?(?:wait|cool|break|pause|"
                    r"rest)\w*|^\s*between"
                )
            )
            if k:
                return "wait_time_min", k
            if WINDOW_AFTER.search(after) and re.search(r"(?:per|in an?|within|every|over an?)\s*$", before):
                return None, None
            k = last(SCAN_KW, before, win_start) or first_after(SCAN_KW)
            if k:
                return "max_active_scan_time_min", k
            return None, None
        if d == "length":
            if not ARTIFACT_KW.search(t) or DEVICE_DIM_KW.search(before[-30:]):
                return None, None
            k = last(
                re.compile(r"[Ee]xtend\w*|[Aa]pproximately|[Uu]p to|[Aa]rtifact|[Aa]rtefact|[Vv]oid"), before, win_start
            )
            if k or re.match(r"\s*(?:from|beyond|around|outside|past|relative)", after):
                a = last(ARTIFACT_KW, t[: q.start], 0)
                return "artifact_extent_mm", a or k
            return None, None
        return None, None

    # -----------------------------------------------------------------------------------------
    def _categorical(self, doc, seg, scope_for, out, test_ctx) -> None:
        t = seg.text

        def add(
            fieldname: str,
            value,
            normalized,
            m_start: int,
            m_end: int,
            comparator: str = "in",
            span: tuple[int, int] | None = None,
        ) -> None:
            s, e = span if span else (max(0, m_start - 40), m_end + 20)
            cs = _clause_start(t, m_start)
            s = max(s, cs) if m_start - cs < 60 else s
            ev = evidence_from_segment(doc.doc_id, seg, s, e, (m_start, m_end))
            out.candidates.append(
                Candidate(
                    fieldname,
                    extracted(value, None, normalized, None, comparator, [ev]),
                    scope_for(m_start, fieldname),
                    f"rule:{fieldname}",
                )
            )

        # nucleus
        if NUCLEUS_CTX.search(t):
            toks = ["1H"]
            for tok, pat in NUCLEUS_TOKENS:
                m = re.search(pat, t)
                if m and not _negated(t, m.start()):
                    toks.append(tok)
            m = NUCLEUS_CTX.search(t)
            assert m is not None
            add("nucleus", ", ".join(toks), sorted(set(toks)), m.start(), min(len(t), m.end() + 30))

        if not test_ctx:
            # scanner type / orientation
            m = SCANNER_CYL.search(t)
            if m and not _negated(t, m.start()):
                add("scanner_type", m.group(0), "cylindrical", m.start(), m.end(), "eq")
            else:
                m = SCANNER_OPEN.search(t)
                if m and not _negated(t, m.start()):
                    add("scanner_type", m.group(0), "open", m.start(), m.end(), "eq")
            m = ORIENT_RE.search(t)
            if (
                m
                and re.search(r"B\s*0|[Ff]ield|[Bb]ore|[Ss]canner|[Oo]rientation|[Ss]ystem", t)
                and not _negated(t, m.start())
            ):
                add("b0_orientation", m.group(0), m.group(1).lower(), m.start(), m.end(), "eq")

            # RF excitation
            cp, mc = CP_RE.search(t), MC_RE.search(t)
            if cp or mc:
                toks = []
                if cp and not _negated(t, cp.start()):
                    toks.append("CP")
                if mc and not _negated(t, mc.start()):
                    toks.append("MC-2")
                if toks and re.search(r"[Ee]xcitation|[Pp]olari[sz]|[Mm]ode|[Tt]ransmi|\bCP\b|[Oo]nly|RF", t):
                    first = min(x.start() for x in (cp, mc) if x)
                    last_end = max(x.end() for x in (cp, mc) if x)
                    add("rf_excitation", ", ".join(toks), toks, first, last_end)

            # transmit coil
            if TRANSMIT_CTX.search(t):
                toks, spans = [], []
                for tok, pat in (
                    ("any", TX_ANY),
                    ("body", TX_BODY),
                    ("head", TX_HEAD),
                    ("extremity", TX_EXT),
                    ("local", TX_LOCAL),
                ):
                    m = pat.search(t)
                    if m and (tok == "any" or not _negated(t, m.start())):
                        toks.append(tok)
                        spans.append((m.start(), m.end()))
                if "any" in toks and len(toks) > 1:
                    toks = [x for x in toks if x != "any"] if not TX_ANY.search(t) else ["any"]
                    spans = spans[:1]
                if toks:
                    add(
                        "transmit_coil",
                        ", ".join(toks),
                        sorted(set(toks)),
                        min(s for s, _ in spans),
                        max(e for _, e in spans),
                    )
            # receive coil
            if RX_CTX.search(t):
                toks, spans = [], []
                for tok, pat in (("any", RX_ANY), ("head", RX_TR_HEAD), ("extremity", RX_TR_EXT)):
                    m = pat.search(t)
                    if m:
                        toks.append(tok)
                        spans.append((m.start(), m.end()))
                if toks:
                    add(
                        "receive_coil",
                        ", ".join(toks),
                        sorted(set(toks)),
                        min(s for s, _ in spans),
                        max(e for _, e in spans),
                    )

        # operating mode (permitted maximum)
        mn, mf = MODE_NORMAL.search(t), MODE_FIRST.search(t)
        if mf and not _negated(t, mf.start()) and not re.search(r"[Ff]irst[\s-]*[Ll]evel\s+SAR", t):
            if mn and re.search(r"[Nn]ormal\s+(?:[Oo]perating\s+)?[Mm]ode\s+or\s+(?:the\s+)?[Ff]irst", t):
                add("operating_mode", t[mn.start() : mf.end()], "first_level", mn.start(), mf.end(), "le")
            elif not mn:
                add("operating_mode", mf.group(0), "first_level", mf.start(), mf.end(), "le")
            else:
                add("operating_mode", mn.group(0), "normal", mn.start(), mn.end(), "le")
        elif mn and not _negated(t, mn.start()):
            add("operating_mode", mn.group(0), "normal", mn.start(), mn.end(), "le")

        # device configuration
        m = DEVICE_CFG.search(t)
        if m and not test_ctx:
            if m.group("v"):
                v = m.group("v").strip()
                add("device_configuration", v, v, m.start(), m.end(), "eq", (m.start(), m.end()))
            else:
                v = m.group(0).strip()
                add("device_configuration", v, v, m.start(), m.end(), "eq")

        # scan regions / exclusion zones
        m = REGION_KW.search(t)
        if m and not test_ctx:
            unrestricted = REGION_UNRESTRICTED.search(t)
            label = re.match(r"\s*(?:[•·▪\-–*]\s*)?[Ss]can\s+[Rr]egions?\s*:?\s*", t)
            body = t[label.end() :] if label else t
            cleaned = re.sub(r"\(?\b[Nn]o exclusion zones?\)?", " ", REGION_UNRESTRICTED.sub(" ", body))
            restrictive = re.search(
                r"[Oo]nly|[Ee]xclu|[Mm]ust|\b[Nn]ot\b|[Aa]bove|[Bb]elow|[Aa]void|[Bb]etween", cleaned
            )
            norm = "unrestricted" if unrestricted and not restrictive else body.strip()[:200]
            add("scan_region_restrictions", body.strip()[:200], norm, m.start(), m.end(), "eq", (0, len(t)))

        # other conditions (free text, not evaluated in the benchmark)
        for pat in OTHER_PATTERNS:
            m = pat.search(t)
            if m and SENTENCE_KW_FOR_OTHER.search(t + " MR"):
                add("other_conditions", m.group(0), m.group(0), m.start(), m.end(), "eq", (0, len(t)))
