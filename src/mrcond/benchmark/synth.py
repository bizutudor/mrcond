"""Track S: synthetic MR Conditional labeling with exactly known ground truth.

A ground-truth specification (status, permitted field strengths and per-field
conditions, optionally different RF limits per field strength) is sampled and
rendered into labeling text in one of five styles:

* ``fda_table``      - two-column "Parameter / Condition" table (FDA 2023 guidance, Table 4a style)
* ``fda_columns``    - table with one column per field strength
* ``legacy_bullets`` - legacy prose with bullets (static field / spatial gradient / SAR)
* ``prose_sections`` - prose with general and per-field-strength sections
* ``compact_inline`` - one dense sentence

with paraphrase and format variation (units T/m vs G/cm, µT/μT/uT/microtesla,
comparator wording, thousands separators, hyphenated units), optional
SAR-fallback wording, missing fields, multi-page layout and distractors
(temperature-rise results, test descriptions mentioning other field
strengths, device dimensions, gradient amplitude, negated field strengths,
post-implantation waiting periods).

The generator records the character span of every rendered value, so gold
evidence (page, offsets, quote <= 25 words) is exact. Device names are
fictional.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from ..ingest import DocumentText, Page
from ..merge import merge_values
from ..schema import (
    FIELD_NAMES,
    FIELD_SPECS,
    ConditionSet,
    Evidence,
    FieldStatus,
    FieldValue,
    LabelDocument,
    MRStatus,
    Provenance,
)
from ..textnorm import trim_quote

__all__ = ["SynthDoc", "generate_corpus", "generate_doc", "save_corpus", "load_corpus", "STYLES"]

STYLES = ("fda_table", "fda_columns", "legacy_bullets", "prose_sections", "compact_inline")

DEVICE_TYPES: list[tuple[str, bool, list[str]]] = [
    ("coronary stent", False, ["Coronary Stent System", "Stent"]),
    ("peripheral stent", False, ["Peripheral Stent", "Vascular Stent System"]),
    ("orthopedic screw system", False, ["Bone Screw System", "Fixation System"]),
    ("intracranial aneurysm clip", False, ["Aneurysm Clip"]),
    ("vena cava filter", False, ["Vena Cava Filter"]),
    ("transcatheter heart valve", False, ["Transcatheter Valve"]),
    ("embolization coil", False, ["Embolization Coil System"]),
    ("spinal fixation system", False, ["Spinal System"]),
    ("spinal cord stimulation system", True, ["SCS System", "Neurostimulation System"]),
    ("deep brain stimulation system", True, ["DBS System"]),
    ("cardiac pacing system", True, ["Pacing System", "Pacemaker System"]),
    ("cochlear implant", True, ["Cochlear Implant"]),
    ("sacral neuromodulation system", True, ["Neuromodulation System"]),
    ("implantable drug pump", True, ["Infusion Pump"]),
]
NAME_PARTS = [
    "Aster",
    "Boreal",
    "Cobalt",
    "Delta",
    "Ember",
    "Fjord",
    "Garnet",
    "Helix",
    "Iris",
    "Juniper",
    "Kestrel",
    "Lumen",
    "Meridian",
    "Nova",
    "Orion",
    "Pylon",
    "Quartz",
    "Rhea",
    "Sable",
    "Tern",
    "Umbra",
    "Vega",
    "Willow",
    "Zephyr",
]
HEADINGS = [
    "MRI Safety Information",
    "MRI Safety Information",
    "Magnetic Resonance Imaging (MRI) Safety Information",
    "MR Safety",
    "MRI Information",
]
REGION_RESTRICTED = [
    "Head Only",
    "Exclusion zone: do not position the isocenter between the C7 vertebra and the iliac crest",
    "Only head and lower extremity landmarks are permitted",
    "Extremity scanning only (knee and below, elbow and below)",
    "Landmark the scan at least 30 cm away from the implanted pulse generator",
]
REGION_UNRESTRICTED = ["Any landmark is acceptable", "Full body scanning is permitted (no exclusion zone)"]
DEVICE_CFG = ["Stimulation OFF", "MRI mode ON", "Therapy OFF", "MRI SureScan mode"]


# --------------------------------------------------------------------------------------------
# ground truth
# --------------------------------------------------------------------------------------------


@dataclass
class Truth:
    status: MRStatus
    device: str
    device_type: str
    active: bool
    b0: list[float]
    b0_le: float | None
    per_b0: bool
    common: dict = field(default_factory=dict)  # field -> canonical value
    per: dict = field(default_factory=dict)  # b0 -> {field -> value}


def _sample_truth(rng: random.Random) -> Truth:
    dtype, active, suffixes = rng.choice(DEVICE_TYPES)
    device = f"{rng.choice(NAME_PARTS)} {rng.choice(suffixes)}"
    r = rng.random()
    status = (
        MRStatus.MR_CONDITIONAL
        if r < 0.85
        else MRStatus.MR_UNSAFE
        if r < 0.90
        else MRStatus.MR_SAFE
        if r < 0.94
        else MRStatus.NOT_STATED
    )
    if status == MRStatus.MR_SAFE:
        dtype, active, device = "polymer suture anchor", False, f"{rng.choice(NAME_PARTS)} PEEK Suture Anchor"
    t = Truth(status, device, dtype, active, [], None, False)
    if status != MRStatus.MR_CONDITIONAL:
        return t
    r = rng.random()
    if r < 0.5:
        t.b0 = [1.5, 3.0]
    elif r < 0.75:
        t.b0 = [1.5]
    elif r < 0.87:
        t.b0 = [3.0]
    else:
        t.b0, t.b0_le = [], 3.0
    t.per_b0 = len(t.b0) == 2 and rng.random() < 0.4
    c = t.common
    if rng.random() < 0.75:
        c["max_sfg_t_per_m"] = rng.choice([3.0, 7.2, 10.0, 12.0, 19.0, 20.0, 25.0, 30.0, 40.0])
    if rng.random() < 0.35:
        c["max_slew_rate_t_per_m_per_s"] = rng.choice([100.0, 150.0, 200.0])
    if rng.random() < 0.3:
        c["rf_excitation"] = ["CP"]
    if rng.random() < 0.45:
        c["transmit_coil"] = rng.choice([["any"], ["body"], ["body"], ["head"], ["body", "head"]])
        if c["transmit_coil"] == ["head"]:
            c["receive_coil"] = ["head"]  # "head transmit/receive coil" states the receive coil too
    if "receive_coil" not in c and rng.random() < 0.25:
        c["receive_coil"] = ["any"]
    if rng.random() < 0.7:
        c["operating_mode"] = "normal" if rng.random() < 0.8 else "first_level"
    if rng.random() < 0.1:
        c["nucleus"] = rng.choice([["1H"], ["1H", "31P"]])
    if rng.random() < 0.3:
        c["scanner_type"] = "cylindrical"
    if rng.random() < 0.25:
        c["b0_orientation"] = "horizontal"
    if rng.random() < (0.5 if active else 0.15):
        c["scan_region_restrictions"] = rng.choice(REGION_RESTRICTED + REGION_UNRESTRICTED)
    if active and rng.random() < 0.8:
        c["device_configuration"] = rng.choice(DEVICE_CFG)
    if rng.random() < 0.45:
        c["artifact_extent_mm"] = float(rng.choice([2, 3, 5, 7, 10, 15, 20, 25, 40]))

    def rf(b0: float | None) -> dict:
        d: dict = {}
        high = b0 is not None and b0 >= 3.0
        has_b1 = rng.random() < 0.45
        has_wb = rng.random() < 0.8 or not has_b1
        if has_b1:
            d["b1rms_max_uT"] = rng.choice([1.0, 1.2, 1.5, 1.7, 2.0] if high else [2.0, 2.4, 2.8, 3.2, 3.6])
        if has_wb:
            if c.get("operating_mode") == "first_level" and rng.random() < 0.5:
                d["wb_sar_max_w_per_kg"] = 4.0
            else:
                d["wb_sar_max_w_per_kg"] = rng.choice(
                    [0.1, 0.4, 0.5, 1.0, 1.2, 1.5] if high else [0.4, 1.0, 1.5, 2.0, 2.0, 2.0]
                )
        if rng.random() < 0.25:
            d["head_sar_max_w_per_kg"] = rng.choice([0.1, 0.4, 1.0, 3.2])
        if has_b1 and has_wb and rng.random() < 0.6:
            d["sar_is_b1rms_fallback"] = True
        if rng.random() < 0.55:
            d["max_active_scan_time_min"] = float(rng.choice([10, 15, 20, 30, 60]))
            if rng.random() < 0.35:
                d["wait_time_min"] = float(rng.choice([5, 10, 15, 30, 60]))
        return d

    if t.per_b0:
        for b in t.b0:
            t.per[b] = rf(b)
    else:
        c.update(rf(max(t.b0) if t.b0 else t.b0_le))
    return t


# --------------------------------------------------------------------------------------------
# rendering helpers
# --------------------------------------------------------------------------------------------


@dataclass
class Mark:
    field: str
    scope: float | None
    s: int  # value span (relative to the statement text)
    e: int
    value: object
    unit: str | None
    normalized: object
    comparator: str | None


@dataclass
class Stmt:
    text: str
    marks: list[Mark] = field(default_factory=list)

    def mark(
        self, needle: str, fieldname: str, scope: float | None, value, unit, normalized, comparator, start_at: int = 0
    ) -> Stmt:
        k = self.text.index(needle, start_at)
        self.marks.append(Mark(fieldname, scope, k, k + len(needle), value, unit, normalized, comparator))
        return self


def _g(x: float) -> str:
    return f"{x:g}"


def _thousands(x: float, rng: random.Random) -> str:
    s = _g(x)
    if x >= 1000 and float(x).is_integer() and rng.random() < 0.6:
        return f"{int(x):,}"
    return s


def _b1_unit(rng: random.Random) -> str:
    return rng.choice(["µT", "μT", "uT", "microtesla"])


class _Renderer:
    def __init__(self, rng: random.Random, t: Truth) -> None:
        self.rng, self.t = rng, t

    # --- numeric statements --------------------------------------------------------------------
    def sfg(self, v: float, scope: float | None, style: str) -> Stmt:
        rng = self.rng
        gcm = v * 100
        g_txt = _thousands(gcm, rng)
        form = rng.choice(["t", "g", "tg", "gt"])
        if style == "fda_table":
            label = "Maximum Spatial Field Gradient "
        elif style == "compact_inline":
            label = rng.choice(["SFG ≤ ", "spatial field gradient ≤ "])
        else:
            label = rng.choice(
                [
                    "Maximum spatial field gradient of ",
                    "Maximum spatial gradient magnetic field of ",
                    "Spatial field gradient: ≤ ",
                    "a maximum spatial field gradient of ",
                ]
            )
        sep = "-" if style == "legacy_bullets" and rng.random() < 0.5 else " "
        gunit = rng.choice(["G/cm", "gauss/cm", "Gauss/cm"])
        if form == "t":
            txt = f"{label}{_g(v)}{sep}T/m"
            st = Stmt(txt).mark(f"{_g(v)}{sep}T/m", "max_sfg_t_per_m", scope, v, "T/m", v, "le")
        elif form == "g":
            txt = f"{label}{g_txt}{sep}{gunit}"
            st = Stmt(txt).mark(f"{g_txt}{sep}{gunit}", "max_sfg_t_per_m", scope, gcm, gunit, v, "le")
        elif form == "tg":
            txt = f"{label}{_g(v)} T/m ({g_txt} {gunit})"
            st = Stmt(txt).mark(f"{_g(v)} T/m", "max_sfg_t_per_m", scope, v, "T/m", v, "le")
        else:
            txt = f"{label}{g_txt}{sep}{gunit} ({_g(v)}{sep}T/m)"
            st = Stmt(txt).mark(f"{g_txt}{sep}{gunit}", "max_sfg_t_per_m", scope, gcm, gunit, v, "le")
        if style == "legacy_bullets" and rng.random() < 0.4:
            st.text += " or less"
        return st

    def slew(self, v: float, scope, style: str) -> Stmt:
        if style == "fda_table":
            txt = f"Maximum Gradient Slew Rate {_g(v)} T/m/s per axis"
        elif style == "compact_inline":
            txt = f"slew rate ≤ {_g(v)} T/m/s per axis"
        else:
            txt = self.rng.choice(
                [
                    f"Maximum gradient slew rate of {_g(v)} T/m/s per axis",
                    f"Gradient slew rate ≤ {_g(v)} T/m/s (per axis)",
                    f"Gradient slew rate limited to {_g(v)} T/m/s per axis",
                ]
            )
        return Stmt(txt).mark(f"{_g(v)} T/m/s", "max_slew_rate_t_per_m_per_s", scope, v, "T/m/s", v, "le")

    def b1(self, v: float, scope) -> Stmt:
        u = _b1_unit(self.rng)
        val = f"{_g(v)} {u}"
        txt = self.rng.choice(
            [
                f"B1+rms ≤ {val}",
                f"B1+RMS of {val} or less",
                f"maximum B1+rms of {val}",
                f"B1+rms must not exceed {val}",
                f"B1+rms <= {val}",
            ]
        )
        return Stmt(txt).mark(val, "b1rms_max_uT", scope, v, u, v, "le")

    def wb(self, v: float, scope, style: str) -> Stmt:
        sep = "-" if style == "legacy_bullets" and self.rng.random() < 0.5 else " "
        val = f"{_g(v)}{sep}W/kg"
        if style == "legacy_bullets":
            txt = self.rng.choice(
                [
                    f"Maximum MR system reported, whole body averaged specific absorption rate (SAR) of {val}",
                    f"Maximum whole body averaged specific absorption rate (SAR) of {val}",
                ]
            )
        elif style == "fda_table":
            txt = self.rng.choice([f"Whole-Body SAR ≤ {val}", f"Maximum Whole-Body SAR {val}"])
        else:
            txt = self.rng.choice(
                [
                    f"Whole-Body SAR ≤ {val}",
                    f"whole body SAR not to exceed {val}",
                    f"WB-SAR ≤ {val}",
                    f"whole-body averaged SAR of {val} or less",
                ]
            )
        return Stmt(txt).mark(val, "wb_sar_max_w_per_kg", scope, v, "W/kg", v, "le")

    def head(self, v: float, scope) -> Stmt:
        val = f"{_g(v)} W/kg"
        txt = self.rng.choice([f"Maximum Head SAR {val}", f"head SAR ≤ {val}", f"head averaged SAR of {val}"])
        return Stmt(txt).mark(val, "head_sar_max_w_per_kg", scope, v, "W/kg", v, "le")

    def fallback_wb(self, v: float, scope) -> Stmt:
        val = f"{_g(v)} W/kg"
        txt = self.rng.choice(
            [
                f"for MR scanners that do not report B1+rms, Whole-Body SAR ≤ {val}",
                f"if B1+rms is not available, limit whole-body SAR to {val}",
                f"when B1+rms is not displayed, use a whole body SAR of {val} or less",
            ]
        )
        st = Stmt(txt).mark(val, "wb_sar_max_w_per_kg", scope, v, "W/kg", v, "le")
        k = 0
        st.marks.append(Mark("sar_is_b1rms_fallback", scope, k, len(txt), True, None, True, "eq"))
        return st

    def scan_time(self, v: float, scope, style: str) -> Stmt:
        if v == 60 and self.rng.random() < 0.3:
            txt = "Limit scanning to one hour of continuous active scan time"
            return Stmt(txt).mark("one hour", "max_active_scan_time_min", scope, "one hour", "hour", 60.0, "le")
        val = f"{_g(v)} minutes"
        txt = self.rng.choice(
            [
                f"Scan for up to {val}",
                f"Maximum active scan time of {val}",
                f"Limit total active scanning to {_g(v)} min",
                f"No more than {val} of active scan time per session",
            ]
        )
        needle = f"{_g(v)} min" if "min per" in txt or txt.endswith(" min") else val
        return Stmt(txt).mark(needle, "max_active_scan_time_min", scope, v, needle.split(" ", 1)[1], v, "le")

    def wait(self, v: float, scope) -> Stmt:
        val = f"{_g(v)} minutes"
        txt = self.rng.choice(
            [
                f"Wait {val} before the next imaging session",
                f"A cooling period of {val} is required between scans",
                f"followed by a wait time of {val}",
            ]
        )
        return Stmt(txt).mark(val, "wait_time_min", scope, v, "minutes", v, "ge")

    # --- categorical statements ----------------------------------------------------------------
    def mode(self, v: str, style: str) -> Stmt:
        if v == "normal":
            txt = "Normal Operating Mode"
            return Stmt(txt).mark(txt, "operating_mode", None, txt, None, "normal", "le")
        txt = "Normal Operating Mode or First Level Controlled Operating Mode"
        if self.rng.random() < 0.4:
            txt = "First Level Controlled Operating Mode"
        return Stmt(txt).mark(txt, "operating_mode", None, txt, None, "first_level", "le")

    def tx(self, toks: list[str], style: str) -> Stmt:
        key = tuple(toks)
        table = {
            ("any",): ["There are no Transmit Coil restrictions", "Any transmit coil may be used"],
            ("body",): ["Integrated Whole Body Transmit Coil", "Body transmit coil only", "Use the body transmit coil"],
            ("head",): ["Head Transmit/Receive Coil ONLY", "head transmit/receive coil only"],
            ("body", "head"): ["Body transmit coil or head transmit coil"],
        }
        txt = self.rng.choice(table[key])
        st = Stmt(txt).mark(txt, "transmit_coil", None, txt, None, list(toks), "in")
        if "ransmit/" in txt:  # a transmit/receive coil also states the receive coil
            st.mark(txt, "receive_coil", None, txt, None, list(toks), "in")
        if style == "fda_table":
            st.text = "RF Transmit Coil Type " + txt if key != ("head",) else "RF Coil Type " + txt
            for m in st.marks:
                m.s += len(st.text) - len(txt)
                m.e += len(st.text) - len(txt)
        return st

    def rx(self, toks: list[str], style: str) -> Stmt:
        txt = "RF Receive Coil Type Any" if style == "fda_table" else "Any receive-only coil may be used"
        if toks == ["head"]:
            txt = "Receive with the head transmit/receive coil"
        return Stmt(txt).mark(txt, "receive_coil", None, txt, None, list(toks), "in")


def _phrase_b0(t: Truth, rng: random.Random, style: str) -> tuple[str, list[float] | float, str]:
    if t.b0_le is not None:
        txt = rng.choice([f"{_g(t.b0_le)}-Tesla or less", f"{_g(t.b0_le)} Tesla or less", f"≤ {_g(t.b0_le)} T"])
        return txt, t.b0_le, "le"
    vals = t.b0
    if len(vals) == 1:
        v = vals[0]
        txt = rng.choice([f"{_g(v)} T", f"{_g(v)}T only", f"{_g(v)} Tesla only", f"{_g(v)}-Tesla"])
        if v == 3.0 and rng.random() < 0.4:
            txt = txt.replace("3 ", "3.0 ").replace("3T", "3.0T").replace("3-", "3.0-")
        return txt, vals, "in"
    a, b = "1.5", rng.choice(["3", "3.0"])
    txt = rng.choice(
        [
            f"{a}T and {b}T",
            f"{a} T or {b} T",
            f"{a} Tesla and {b} Tesla only",
            f"{a} and {b} Tesla",
            f"{a}-Tesla and {b}-Tesla",
        ]
    )
    return txt, vals, "in"


def _b0_label(b: float) -> str:
    return f"{_g(b)}T" if b != 3.0 else "3T"


# --------------------------------------------------------------------------------------------
# document builder
# --------------------------------------------------------------------------------------------


class _Builder:
    def __init__(self) -> None:
        self.pages: list[list[str]] = [[]]
        self.lens = [0]
        self.marks: list[tuple[int, int, int, int, int, Mark]] = []  # page, stmt_s, stmt_e, val_s, val_e, mark

    def add(self, *parts: Stmt | str, end: str = "\n") -> None:
        page = len(self.pages) - 1
        line_start = self.lens[page]
        cur = line_start
        texts = []
        pending = []
        for p in parts:
            st = Stmt(p) if isinstance(p, str) else p
            for m in st.marks:
                pending.append((cur, cur + len(st.text), cur + m.s, cur + m.e, m))
            texts.append(st.text)
            cur += len(st.text)
        line = "".join(texts) + end
        self.pages[page].append(line)
        self.lens[page] += len(line)
        for ss, se, vs, ve, m in pending:
            self.marks.append((page, ss, se, vs, ve, m))

    def page_break(self) -> None:
        self.pages.append([])
        self.lens.append(0)

    def texts(self) -> list[str]:
        return ["".join(p) for p in self.pages]


@dataclass
class SynthDoc:
    doc: DocumentText
    gold: LabelDocument
    style: str
    meta: dict


def _evidence(doc_id: str, page_text: str, page_no: int, ss: int, se: int, vs: int, ve: int) -> Evidence:
    s, e = trim_quote(page_text, ss, se, vs, ve, 25)
    return Evidence(doc_id=doc_id, page=page_no, char_start=s, char_end=e, quote=page_text[s:e], verified=True)


def _fv_from_marks(doc_id: str, texts: list[str], items) -> FieldValue:
    fvs = []
    for page, ss, se, vs, ve, m in items:
        ev = _evidence(doc_id, texts[page], page + 1, ss, se, vs, ve)
        fvs.append(
            FieldValue(
                status=FieldStatus.EXTRACTED,
                value=m.value,
                unit=m.unit,
                normalized_value=m.normalized,
                normalized_unit=FIELD_SPECS[m.field].canonical_unit if m.field in FIELD_SPECS else None,
                comparator=m.comparator,
                evidence=[ev],
            )
        )
    return merge_values(items[0][5].field, fvs) if items[0][5].field in FIELD_SPECS else fvs[0]


def _gold(doc_id: str, t: Truth, b: _Builder) -> LabelDocument:
    texts = b.texts()
    status_items = [x for x in b.marks if x[5].field == "mr_status"]
    if status_items:
        page, ss, se, vs, ve, m = status_items[0]
        st = FieldValue(
            status=FieldStatus.EXTRACTED if t.status != MRStatus.NOT_STATED else FieldStatus.NOT_STATED,
            value=m.value,
            normalized_value=m.normalized,
            comparator="eq",
            evidence=[_evidence(doc_id, texts[page], page + 1, ss, se, vs, ve)],
        )
    else:
        st = FieldValue()
    sets: list[ConditionSet] = []
    if t.status == MRStatus.MR_CONDITIONAL:
        b0_items = [x for x in b.marks if x[5].field == "b0_tesla"]
        keys: list[float | None] = list(t.b0) if t.per_b0 else [None]
        for i, key in enumerate(keys, start=1):
            cs = ConditionSet(set_id=f"cs{i}", scope=None if key is None else f"{_g(key)} T")
            evs = [_evidence(doc_id, texts[p], p + 1, ss, se, vs, ve) for p, ss, se, vs, ve, _ in b0_items]
            if key is None:
                if t.b0_le is not None:
                    cs.b0_tesla = FieldValue(
                        status=FieldStatus.EXTRACTED,
                        value=t.b0_le,
                        unit="T",
                        normalized_value=[t.b0_le],
                        normalized_unit="T",
                        comparator="le",
                        evidence=evs,
                    )
                else:
                    cs.b0_tesla = FieldValue(
                        status=FieldStatus.EXTRACTED,
                        value=list(t.b0),
                        unit="T",
                        normalized_value=list(t.b0),
                        normalized_unit="T",
                        comparator="in",
                        evidence=evs,
                    )
            else:
                cs.b0_tesla = FieldValue(
                    status=FieldStatus.EXTRACTED,
                    value=[key],
                    unit="T",
                    normalized_value=[key],
                    normalized_unit="T",
                    comparator="in",
                    evidence=evs,
                )
            for name in FIELD_NAMES:
                if name == "b0_tesla":
                    continue
                items = [
                    x
                    for x in b.marks
                    if x[5].field == name and (x[5].scope is None or key is None or abs(x[5].scope - key) < 1e-9)
                ]
                if items:
                    cs.set_field(name, _fv_from_marks(doc_id, texts, items))
            sets.append(cs)
    return LabelDocument(
        doc_id=doc_id,
        device_name=t.device,
        manufacturer=None,
        device_type=t.device_type,
        mr_status=st,
        condition_sets=sets,
        provenance=Provenance(
            extractor="synthetic-generator",
            verification="human-verified",
            annotator="generator (exact by construction)",
            text_extractor="plain",
        ),
    )


def _status_stmt(t: Truth, rng: random.Random) -> list[Stmt]:
    d = t.device
    if t.status == MRStatus.MR_UNSAFE:
        st = Stmt(f"The {d} is MR Unsafe.").mark("MR Unsafe", "mr_status", None, "MR Unsafe", None, "MR Unsafe", "eq")
        out = [st, Stmt("The device presents a projectile hazard. Do not bring it into the MR scanner room.")]
        if rng.random() < 0.5:
            out.append(Stmt("Patients with this device must not undergo MRI at 1.5 T or 3 T."))
        return out
    if t.status == MRStatus.MR_SAFE:
        st = Stmt(f"The {d} is MR Safe.").mark("MR Safe", "mr_status", None, "MR Safe", None, "MR Safe", "eq")
        return [
            st,
            Stmt("It is composed entirely of electrically nonconductive, nonmetallic, and nonmagnetic materials."),
        ]
    st = Stmt(f"The {d} has not been evaluated for safety in the MR environment.")
    st.mark(
        "not been evaluated for safety in the MR environment", "mr_status", None, "Not stated", None, "Not stated", "eq"
    )
    return [
        st,
        Stmt(
            f"It has not been tested for heating or unwanted movement in the MR environment. The safety of "
            f"the {d} in the MR environment is unknown. Performing an MR exam on a person who has this "
            f"medical device may result in injury or device malfunction."
        ),
    ]


def _preamble(b: _Builder, t: Truth, rng: random.Random) -> None:
    b.add("Instructions for Use")
    b.add(
        f"Device description: The {t.device} is a {t.device_type}. "
        f"It is available in lengths from {rng.choice([8, 12, 15])} mm to {rng.choice([28, 38, 48])} mm."
    )
    b.add("Warnings: For single use only. Do not resterilize. Store at 15 °C to 30 °C.")
    b.add("Refer to the surgical technique guide for implantation instructions.")


def _distractors(t: Truth, rng: random.Random) -> list[Stmt]:
    out = []
    if rng.random() < 0.35:
        out.append(Stmt("RF heating was evaluated using a body coil at 64 MHz and 128 MHz in a gel phantom."))
    if rng.random() < 0.25:
        out.append(Stmt("Testing used a maximum gradient amplitude of 40 mT/m."))
    if rng.random() < 0.2 and t.b0:
        out.append(
            Stmt("Scanning at 7 T has not been evaluated; do not scan at field strengths other than those listed.")
        )
    if rng.random() < 0.2:
        out.append(
            Stmt(
                rng.choice(
                    [
                        "Patients may be scanned immediately after implantation.",
                        "Wait at least 6 weeks after implantation before performing an MRI scan.",
                    ]
                )
            )
        )
    return out


def _render_conditional(t: Truth, rng: random.Random, style: str, b: _Builder) -> None:
    R = _Renderer(rng, t)
    c = t.common
    d = t.device
    b0_txt, b0_val, b0_cmp = _phrase_b0(t, rng, style)
    b0_norm = [b0_val] if isinstance(b0_val, float) else list(b0_val)

    def b0_stmt(prefix: str, suffix: str = "") -> Stmt:
        st = Stmt(prefix + b0_txt + suffix)
        st.mark(b0_txt, "b0_tesla", None, b0_val, "T", b0_norm, b0_cmp, start_at=len(prefix))
        return st

    def status_mark(text: str) -> Stmt:
        return Stmt(text).mark("MR Conditional", "mr_status", None, "MR Conditional", None, "MR Conditional", "eq")

    def rf_global_stmts(vals: dict, scope: float | None) -> list[Stmt]:
        out = []
        if "b1rms_max_uT" in vals:
            s1 = R.b1(vals["b1rms_max_uT"], scope)
            if vals.get("sar_is_b1rms_fallback"):
                s2 = R.fallback_wb(vals["wb_sar_max_w_per_kg"], scope)
                out.append(_join([s1, Stmt("; "), s2]))
            else:
                out.append(s1)
                if "wb_sar_max_w_per_kg" in vals:
                    out.append(R.wb(vals["wb_sar_max_w_per_kg"], scope, style))
        elif "wb_sar_max_w_per_kg" in vals:
            out.append(R.wb(vals["wb_sar_max_w_per_kg"], scope, style))
        if "head_sar_max_w_per_kg" in vals:
            out.append(R.head(vals["head_sar_max_w_per_kg"], scope))
        return out

    def time_stmts(vals: dict, scope: float | None) -> list[Stmt]:
        out = []
        if "max_active_scan_time_min" in vals:
            out.append(R.scan_time(vals["max_active_scan_time_min"], scope, style))
        if "wait_time_min" in vals:
            w = R.wait(vals["wait_time_min"], scope)
            if w.text.startswith("followed"):
                w.text = w.text[0].upper() + w.text[1:]
            out.append(w)
        return out

    if style == "fda_table":
        b.add(rng.choice(HEADINGS))
        b.add(
            status_mark(
                f"The {d} is MR Conditional. A person with the {d} may be safely scanned under the "
                f"following conditions. Failure to follow these conditions may result in injury."
            )
        )
        b.add("Parameter Condition")
        b.add(f"Device Name {d}")
        if "device_configuration" in c:
            v = c["device_configuration"]
            b.add(Stmt(f"Device Configuration {v}").mark(v, "device_configuration", None, v, None, v, "eq"))
        b.add(b0_stmt("Static Magnetic Field Strength (B0) "))
        if "nucleus" in c:
            v = c["nucleus"]
            txt = "No additional nuclei" if v == ["1H"] else "31P (phosphorus)"
            b.add(Stmt(f"Type of Nuclei {txt}").mark(txt, "nucleus", None, txt, None, v, "in"))
        if "scanner_type" in c:
            b.add(
                Stmt("MR Scanner Type Cylindrical").mark(
                    "Cylindrical", "scanner_type", None, "Cylindrical", None, "cylindrical", "eq"
                )
            )
        if "b0_orientation" in c:
            b.add(
                Stmt("B0 Field Orientation Horizontal").mark(
                    "Horizontal", "b0_orientation", None, "Horizontal", None, "horizontal", "eq"
                )
            )
        if "max_sfg_t_per_m" in c:
            b.add(R.sfg(c["max_sfg_t_per_m"], None, style))
        if "max_slew_rate_t_per_m_per_s" in c:
            b.add(R.slew(c["max_slew_rate_t_per_m_per_s"], None, style))
        if "rf_excitation" in c:
            b.add(
                Stmt("RF Excitation Circularly Polarized (CP)").mark(
                    "Circularly Polarized (CP)", "rf_excitation", None, "Circularly Polarized (CP)", None, ["CP"], "in"
                )
            )
        if "transmit_coil" in c:
            b.add(R.tx(c["transmit_coil"], style))
        if "receive_coil" in c:
            b.add(R.rx(c["receive_coil"], style))
        if "operating_mode" in c:
            b.add("Operating Mode ", R.mode(c["operating_mode"], style))
        if t.per_b0:
            first = True
            for bb in t.b0:
                stmts = rf_global_stmts(t.per[bb], bb)
                if stmts:
                    prefix = "RF Conditions " if first else ""
                    b.add(prefix + f"For {_b0_label(bb)} MR Scanner: ", _join(_sep(stmts, "; ")))
                    first = False
            for bb in t.b0:
                ts = time_stmts(t.per[bb], bb)
                if ts:
                    b.add(f"Scan Duration at {_b0_label(bb)}: ", _join(_sep(ts, ". ")), end=".\n")
        else:
            stmts = rf_global_stmts(c, None)
            if stmts:
                if len(stmts) == 1 or rng.random() < 0.5:
                    b.add("RF Conditions ", _join(_sep(stmts, "; ")))
                else:
                    for s in stmts:
                        b.add(s)
            ts = time_stmts(c, None)
            if ts:
                b.add("Scan Duration ", _join(_sep(ts, ". ")), end=".\n")
        if "scan_region_restrictions" in c:
            v = c["scan_region_restrictions"]
            norm = "unrestricted" if v in REGION_UNRESTRICTED else v
            b.add(Stmt(f"Scan Regions {v}").mark(v, "scan_region_restrictions", None, v, None, norm, "eq"))
        if "artifact_extent_mm" in c:
            a = c["artifact_extent_mm"]
            txt = f"The image artifact extends approximately {_g(a)} mm from the device."
            b.add(Stmt("Image Artifact " + txt).mark(f"{_g(a)} mm", "artifact_extent_mm", None, a, "mm", a, "eq"))

    elif style == "fda_columns":
        assert len(t.b0) == 2
        b.add(rng.choice(HEADINGS))
        b.add(status_mark(f"The {d} is MR Conditional. Scanning is possible under the conditions in the table below."))
        b.add(b0_stmt("Static magnetic field strength: "))
        b.add("Parameter 1.5 T 3 T")
        if "max_sfg_t_per_m" in c:
            b.add(R.sfg(c["max_sfg_t_per_m"], None, "fda_table"))
        if "max_slew_rate_t_per_m_per_s" in c:
            b.add(R.slew(c["max_slew_rate_t_per_m_per_s"], None, "fda_table"))
        if "operating_mode" in c:
            b.add("Operating Mode ", R.mode(c["operating_mode"], style))
        if "transmit_coil" in c:
            b.add(R.tx(c["transmit_coil"], "fda_table"))
        per = t.per if t.per_b0 else {bb: {k: v for k, v in c.items()} for bb in t.b0}
        rows = [
            ("b1rms_max_uT", "B1+rms ", lambda v: f"≤ {_g(v)} μT", "μT"),
            ("wb_sar_max_w_per_kg", "Whole-Body SAR ", lambda v: f"≤ {_g(v)} W/kg", "W/kg"),
            ("head_sar_max_w_per_kg", "Head SAR ", lambda v: f"≤ {_g(v)} W/kg", "W/kg"),
            ("max_active_scan_time_min", "Maximum active scan time ", lambda v: f"{_g(v)} minutes", "minutes"),
            ("wait_time_min", "Wait time between sessions ", lambda v: f"{_g(v)} minutes", "minutes"),
        ]
        for fname, label, fmt, unit in rows:
            vals = [per[bb].get(fname) for bb in t.b0]
            if all(v is None for v in vals):
                continue
            if any(v is None for v in vals) or (t.per_b0 and vals[0] != vals[1]) or not t.per_b0 and vals[0] != vals[1]:
                cells = [fmt(v) if v is not None else "not specified" for v in vals]
                st = Stmt(label + " ".join(cells))
                pos = len(label)
                for bb, v, cell in zip(t.b0, vals, cells):
                    if v is not None:
                        num = cell.replace("≤ ", "")
                        k = st.text.index(num, pos)
                        cmp = "ge" if fname == "wait_time_min" else "le"
                        st.marks.append(Mark(fname, bb if t.per_b0 else None, k, k + len(num), v, unit, v, cmp))
                    pos += len(cell) + 1
                b.add(st)
            else:
                cell = fmt(vals[0])
                num = cell.replace("≤ ", "")
                st = Stmt(label + cell)
                k = st.text.index(num, len(label))
                cmp = "ge" if fname == "wait_time_min" else "le"
                # same value at both field strengths
                st.marks.append(Mark(fname, None, k, k + len(num), vals[0], unit, vals[0], cmp))
                b.add(st)
        fb = [per[bb].get("sar_is_b1rms_fallback") for bb in t.b0]
        if any(fb):
            if all(fb):
                st = Stmt("Whole-body SAR limits apply only to MR scanners that do not report B1+rms.")
                st.marks.append(Mark("sar_is_b1rms_fallback", None, 0, len(st.text), True, None, True, "eq"))
                b.add(st)
            else:
                for bb, f in zip(t.b0, fb):
                    if f:
                        st = Stmt(
                            f"At {_b0_label(bb)}, the whole-body SAR limit applies only to MR scanners that do "
                            f"not report B1+rms."
                        )
                        st.marks.append(Mark("sar_is_b1rms_fallback", bb, 0, len(st.text), True, None, True, "eq"))
                        b.add(st)
        if "device_configuration" in c:
            v = c["device_configuration"]
            b.add(Stmt(f"Device Configuration {v}").mark(v, "device_configuration", None, v, None, v, "eq"))
        if "scan_region_restrictions" in c:
            v = c["scan_region_restrictions"]
            norm = "unrestricted" if v in REGION_UNRESTRICTED else v
            b.add(Stmt(f"Scan Regions {v}").mark(v, "scan_region_restrictions", None, v, None, norm, "eq"))
        if "artifact_extent_mm" in c:
            a = c["artifact_extent_mm"]
            b.add(
                Stmt(f"MR image artifact: the artifact may extend up to {_g(a)} mm beyond the implant.").mark(
                    f"{_g(a)} mm", "artifact_extent_mm", None, a, "mm", a, "eq"
                )
            )

    elif style == "legacy_bullets":
        b.add(rng.choice(HEADINGS))
        b.add(
            status_mark(
                f"Non-clinical testing has demonstrated that the {d} is MR Conditional. A patient with this "
                f"device can be safely scanned in an MR system meeting the following conditions:"
            )
        )
        bullet = rng.choice(["• ", "- ", "· "])
        b.add(bullet, b0_stmt("Static magnetic field of "))
        if "max_sfg_t_per_m" in c:
            b.add(bullet, R.sfg(c["max_sfg_t_per_m"], None, style))
        if "max_slew_rate_t_per_m_per_s" in c:
            b.add(bullet, R.slew(c["max_slew_rate_t_per_m_per_s"], None, style))
        if t.per_b0:
            for bb in t.b0:
                stmts = rf_global_stmts(t.per[bb], bb) + time_stmts(t.per[bb], bb)
                if stmts:
                    b.add(bullet, f"At {_g(bb)} T: ", _join(_sep(stmts, "; ")))
            if "operating_mode" in c:
                b.add(bullet, "Scanning in the ", R.mode(c["operating_mode"], style), " of operation for the MR system")
        else:
            stmts = rf_global_stmts(c, None)
            mode_done = False
            for i, s in enumerate(stmts):
                if i == 0 and "operating_mode" in c and s.marks and s.marks[0].field == "wb_sar_max_w_per_kg":
                    m = R.mode(c["operating_mode"], style)
                    b.add(bullet, s, " (", m, ")")
                    mode_done = True
                else:
                    b.add(bullet, s)
            if "operating_mode" in c and not mode_done:
                b.add(bullet, "Scanning in the ", R.mode(c["operating_mode"], style), " of operation for the MR system")
            for s in time_stmts(c, None):
                b.add(bullet, s)
        if "transmit_coil" in c:
            b.add(bullet, R.tx(c["transmit_coil"], style))
        if "receive_coil" in c:
            b.add(bullet, R.rx(c["receive_coil"], style))
        if "rf_excitation" in c:
            b.add(
                bullet,
                Stmt("RF excitation: circularly polarized (CP) only").mark(
                    "circularly polarized (CP)", "rf_excitation", None, "circularly polarized (CP)", None, ["CP"], "in"
                ),
            )
        if "device_configuration" in c:
            v = c["device_configuration"]
            b.add(bullet, Stmt(f"Device configuration: {v}").mark(v, "device_configuration", None, v, None, v, "eq"))
        if "scan_region_restrictions" in c:
            v = c["scan_region_restrictions"]
            norm = "unrestricted" if v in REGION_UNRESTRICTED else v
            b.add(bullet, Stmt(f"Scan regions: {v}").mark(v, "scan_region_restrictions", None, v, None, norm, "eq"))
        x = rng.choice(["1.6", "2.1", "2.4", "3.0"])
        b.add(
            f"Under the scan conditions defined above, the {d} is expected to produce a maximum temperature rise of "
            f"less than {x} °C after 15 minutes of continuous scanning."
        )
        if "artifact_extent_mm" in c:
            a = c["artifact_extent_mm"]
            tb = rng.choice(["3 T", "3-Tesla", "1.5 T"])
            b.add(
                Stmt(
                    f"In non-clinical testing, the image artifact caused by the device extends approximately {_g(a)} "
                    f"mm from the device when imaged with a gradient echo pulse sequence and a {tb} MR system."
                ).mark(f"{_g(a)} mm", "artifact_extent_mm", None, a, "mm", a, "eq")
            )

    elif style == "prose_sections":
        b.add(rng.choice(HEADINGS))
        b.add(
            status_mark(f"The {d} is MR Conditional."),
            Stmt(" "),
            b0_stmt("Patients with this device can be scanned at ", " when all of the following conditions are met."),
        )
        general = []
        if "max_sfg_t_per_m" in c:
            general.append(R.sfg(c["max_sfg_t_per_m"], None, style))
        if "max_slew_rate_t_per_m_per_s" in c:
            general.append(R.slew(c["max_slew_rate_t_per_m_per_s"], None, style))
        if "operating_mode" in c:
            general.append(_join([Stmt("Scan in "), R.mode(c["operating_mode"], style)]))
        if "transmit_coil" in c:
            general.append(R.tx(c["transmit_coil"], style))
        if "receive_coil" in c:
            general.append(R.rx(c["receive_coil"], style))
        if "rf_excitation" in c:
            general.append(
                Stmt("Use CP mode only (circularly polarized RF excitation)").mark(
                    "CP mode", "rf_excitation", None, "CP mode", None, ["CP"], "in"
                )
            )
        if not t.per_b0:
            general += rf_global_stmts(c, None) + time_stmts(c, None)
        if general:
            b.add("General conditions:")
            for s in general:
                b.add("- ", s, end=".\n")
        if t.per_b0:
            for bb in t.b0:
                stmts = rf_global_stmts(t.per[bb], bb) + time_stmts(t.per[bb], bb)
                if stmts:
                    b.add(f"{_g(bb)} T scanning conditions:")
                    for s in stmts:
                        b.add("- ", s, end=".\n")
        if "device_configuration" in c:
            v = c["device_configuration"]
            b.add(
                Stmt(f"Before scanning, set the device configuration: {v}.").mark(
                    v, "device_configuration", None, v, None, v, "eq"
                )
            )
        if "scan_region_restrictions" in c:
            v = c["scan_region_restrictions"]
            norm = "unrestricted" if v in REGION_UNRESTRICTED else v
            b.add(Stmt(f"Scan regions: {v}.").mark(v, "scan_region_restrictions", None, v, None, norm, "eq"))
        if "artifact_extent_mm" in c:
            a = c["artifact_extent_mm"]
            b.add(
                Stmt(f"The image artifact extends approximately {_g(a)} mm from the device.").mark(
                    f"{_g(a)} mm", "artifact_extent_mm", None, a, "mm", a, "eq"
                )
            )

    else:  # compact_inline
        parts: list[Stmt] = [status_mark("MR Conditional "), b0_stmt("(", "): ")]
        items: list[Stmt] = []
        if "max_sfg_t_per_m" in c:
            items.append(R.sfg(c["max_sfg_t_per_m"], None, style))
        if "max_slew_rate_t_per_m_per_s" in c:
            items.append(R.slew(c["max_slew_rate_t_per_m_per_s"], None, style))
        items += rf_global_stmts(c, None) if not t.per_b0 else []
        if t.per_b0:
            for bb in t.b0:
                stmts = rf_global_stmts(t.per[bb], bb) + time_stmts(t.per[bb], bb)
                if stmts:
                    items.append(_join([Stmt(f"at {_g(bb)} T: ")] + _sep(stmts, ", ")))
        if "operating_mode" in c:
            items.append(R.mode(c["operating_mode"], style))
        if not t.per_b0:
            items += time_stmts(c, None)
        if "transmit_coil" in c:
            items.append(R.tx(c["transmit_coil"], style))
        b.add(rng.choice(HEADINGS))
        b.add(_join(parts + _sep(items, "; ")), end=".\n")
        if "device_configuration" in c:
            v = c["device_configuration"]
            b.add(
                Stmt(f"Device configuration during MRI: {v}.").mark(v, "device_configuration", None, v, None, v, "eq")
            )
        if "artifact_extent_mm" in c:
            a = c["artifact_extent_mm"]
            b.add(
                Stmt(f"Image artifact: up to {_g(a)} mm.").mark(
                    f"{_g(a)} mm", "artifact_extent_mm", None, a, "mm", a, "eq"
                )
            )


def _sep(stmts: list[Stmt], sep: str) -> list[Stmt]:
    out: list[Stmt] = []
    for i, s in enumerate(stmts):
        if i:
            out.append(Stmt(sep))
        out.append(s)
    return out


def _join(stmts: list[Stmt]) -> Stmt:
    text = ""
    marks: list[Mark] = []
    for s in stmts:
        for m in s.marks:
            marks.append(
                Mark(m.field, m.scope, m.s + len(text), m.e + len(text), m.value, m.unit, m.normalized, m.comparator)
            )
        text += s.text
    return Stmt(text, marks)


def generate_doc(seed: int, index: int) -> SynthDoc:
    rng = random.Random(f"{seed}-{index}")
    t = _sample_truth(rng)
    doc_id = f"synth-{seed}-{index:04d}"
    b = _Builder()
    multipage = rng.random() < 0.3
    if multipage:
        _preamble(b, t, rng)
        b.page_break()
    if t.status == MRStatus.MR_CONDITIONAL:
        styles = list(STYLES)
        if len(t.b0) != 2:
            styles.remove("fda_columns")
        if t.b0_le is not None:
            styles = ["legacy_bullets", "compact_inline", "prose_sections"]
        style = rng.choice(styles)
        _render_conditional(t, rng, style, b)
    else:
        style = "status_only"
        b.add(rng.choice(HEADINGS))
        for s in _status_stmt(t, rng):
            b.add(s)
    for s in _distractors(t, rng):
        b.add(s)
    texts = b.texts()
    doc = DocumentText(doc_id, [Page(i + 1, tx) for i, tx in enumerate(texts)], "text", "synthetic", {"style": style})
    gold = _gold(doc_id, t, b)
    return SynthDoc(doc, gold, style, {"device_type": t.device_type, "active": t.active, "per_b0": t.per_b0})


def generate_corpus(n: int = 200, seed: int = 20261015) -> list[SynthDoc]:
    return [generate_doc(seed, i) for i in range(n)]


def save_corpus(docs: list[SynthDoc], out_dir: str | Path) -> None:
    out = Path(out_dir)
    (out / "docs").mkdir(parents=True, exist_ok=True)
    (out / "gold").mkdir(parents=True, exist_ok=True)
    index = []
    for sd in docs:
        (out / "docs" / f"{sd.doc.doc_id}.json").write_text(
            json.dumps(sd.doc.to_json(), ensure_ascii=False), encoding="utf-8"
        )
        (out / "gold" / f"{sd.doc.doc_id}.json").write_text(
            sd.gold.model_dump_json(exclude_defaults=True), encoding="utf-8"
        )
        index.append({"doc_id": sd.doc.doc_id, "style": sd.style, **sd.meta})
    (out / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")


def load_corpus(out_dir: str | Path) -> list[tuple[DocumentText, LabelDocument, dict]]:
    out = Path(out_dir)
    index = json.loads((out / "index.json").read_text(encoding="utf-8"))
    res = []
    for item in index:
        doc = DocumentText.load(out / "docs" / f"{item['doc_id']}.json")
        gold = LabelDocument.model_validate_json((out / "gold" / f"{item['doc_id']}.json").read_text(encoding="utf-8"))
        res.append((doc, gold, item))
    return res
