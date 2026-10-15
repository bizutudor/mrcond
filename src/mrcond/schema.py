"""MRCond data model (pydantic v2).

A :class:`LabelDocument` describes one MRI-safety labeling source for one
device: its MR status (MR Conditional / MR Safe / MR Unsafe / Not stated) and a
list of :class:`ConditionSet` objects. Each condition set is one alternative
set of conditions (typically one per static field strength). Every condition is
a :class:`FieldValue` carrying the value as stated, its unit, a normalised value
in a canonical unit, a status (EXTRACTED / NOT_STATED / NEEDS_REVIEW) and the
evidence (document id, page, character offsets and a short verbatim quote).

Field semantics (which fields are upper/lower limits, canonical units,
plausible ranges) live in :data:`FIELD_SPECS`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "0.1.0"


class FieldStatus(str, Enum):
    EXTRACTED = "EXTRACTED"
    NOT_STATED = "NOT_STATED"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class MRStatus(str, Enum):
    MR_CONDITIONAL = "MR Conditional"
    MR_SAFE = "MR Safe"
    MR_UNSAFE = "MR Unsafe"
    NOT_STATED = "Not stated"


Comparator = Literal["le", "ge", "eq", "in", "any"]
NormalizedValue = bool | float | list[float] | list[str] | str | None


class Evidence(BaseModel):
    """Pointer to a short verbatim quote in a source document (page is 1-based)."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    page: int = Field(ge=1)
    char_start: int = Field(ge=0, description="Offset of the quote in the page text (MRCond ingestion).")
    char_end: int = Field(ge=0)
    quote: str = Field(description="Verbatim quote, at most 25 words.")
    verified: bool | None = Field(default=None, description="Set by the verifier: quote found verbatim.")


class FieldValue(BaseModel):
    """One labeled condition with evidence."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    status: FieldStatus = FieldStatus.NOT_STATED
    value: NormalizedValue = Field(default=None, description="Value as stated in the labeling.")
    unit: str | None = Field(default=None, description="Unit as stated in the labeling.")
    normalized_value: NormalizedValue = Field(default=None, description="Value in the canonical unit.")
    normalized_unit: str | None = None
    comparator: Comparator | None = Field(
        default=None,
        description="le = upper limit, ge = lower limit, in = allowed set, eq = descriptive, "
        "any = explicitly unrestricted.",
    )
    evidence: list[Evidence] = Field(default_factory=list)
    review_reasons: list[str] = Field(default_factory=list)

    @property
    def is_stated(self) -> bool:
        return self.status == FieldStatus.EXTRACTED


def _fv() -> FieldValue:
    return FieldValue()


class ConditionSet(BaseModel):
    """One alternative set of MR Conditional conditions (e.g. per field strength)."""

    model_config = ConfigDict(extra="forbid")

    set_id: str = "cs1"
    scope: str | None = Field(default=None, description="Free-text scope, e.g. 'Whole-body examinations'.")
    b0_tesla: FieldValue = Field(default_factory=_fv)
    nucleus: FieldValue = Field(default_factory=_fv)
    scanner_type: FieldValue = Field(default_factory=_fv)
    b0_orientation: FieldValue = Field(default_factory=_fv)
    max_sfg_t_per_m: FieldValue = Field(default_factory=_fv)
    max_slew_rate_t_per_m_per_s: FieldValue = Field(default_factory=_fv)
    rf_excitation: FieldValue = Field(default_factory=_fv)
    transmit_coil: FieldValue = Field(default_factory=_fv)
    receive_coil: FieldValue = Field(default_factory=_fv)
    operating_mode: FieldValue = Field(default_factory=_fv)
    b1rms_max_uT: FieldValue = Field(default_factory=_fv)
    wb_sar_max_w_per_kg: FieldValue = Field(default_factory=_fv)
    head_sar_max_w_per_kg: FieldValue = Field(default_factory=_fv)
    sar_is_b1rms_fallback: FieldValue = Field(
        default_factory=_fv,
        description="True when the labeling states SAR limits only apply if B1+rms is not available.",
    )
    max_active_scan_time_min: FieldValue = Field(default_factory=_fv)
    wait_time_min: FieldValue = Field(default_factory=_fv)
    scan_region_restrictions: FieldValue = Field(default_factory=_fv)
    device_configuration: FieldValue = Field(default_factory=_fv)
    artifact_extent_mm: FieldValue = Field(default_factory=_fv)
    other_conditions: FieldValue = Field(default_factory=_fv)

    def field(self, name: str) -> FieldValue:
        return getattr(self, name)

    def set_field(self, name: str, fv: FieldValue) -> None:
        setattr(self, name, fv)

    def iter_fields(self):
        for name in FIELD_NAMES:
            yield name, getattr(self, name)


Verification = Literal["machine-extracted", "machine-verified", "draft-unverified", "human-verified"]


class Provenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    extractor: str = Field(description="e.g. 'rule', 'llm:openai-compat:qwen2.5-1.5b', 'human'.")
    verification: Verification = "machine-extracted"
    annotator: str | None = None
    reviewer: str | None = None
    reviewed_at: datetime | None = None
    created_at: datetime | None = None
    text_extractor: str | None = Field(default=None, description="e.g. 'pypdf 5.9.0'.")
    notes: list[str] = Field(default_factory=list)


class LabelDocument(BaseModel):
    """MRI safety labeling of one device, as extracted or annotated from one source document."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    doc_id: str
    device_name: str | None = None
    manufacturer: str | None = None
    device_type: str | None = None
    mr_status: FieldValue = Field(default_factory=_fv)
    condition_sets: list[ConditionSet] = Field(default_factory=list)
    provenance: Provenance = Field(default_factory=lambda: Provenance(extractor="unknown"))

    def status_value(self) -> MRStatus:
        if self.mr_status.status != FieldStatus.EXTRACTED or self.mr_status.normalized_value is None:
            return MRStatus.NOT_STATED
        try:
            return MRStatus(str(self.mr_status.normalized_value))
        except ValueError:
            return MRStatus.NOT_STATED


# --------------------------------------------------------------------------------------
# Field specifications
# --------------------------------------------------------------------------------------

FieldKind = Literal["b0", "numeric", "categorical", "text", "bool"]


@dataclass(frozen=True)
class FieldSpec:
    name: str
    label: str
    kind: FieldKind
    dimension: str | None = None  # units.CANONICAL_UNITS key
    canonical_unit: str | None = None
    direction: Literal["upper", "lower", "info"] | None = None
    plausible: tuple[float, float] | None = None
    context: tuple[str, ...] = ()  # regexes; at least one must occur in a supporting quote
    evaluated: bool = True  # included in benchmark field metrics
    vocabulary: tuple[str, ...] = ()  # allowed normalised tokens (categorical)


FIELD_SPECS: dict[str, FieldSpec] = {
    s.name: s
    for s in [
        FieldSpec(
            "b0_tesla",
            "Static field strength B0",
            "b0",
            "field",
            "T",
            None,
            (0.05, 7.0),
            (r"[Tt]esla|\bT\b|\dT\b|field|B0|B 0|scanner|system|MR",),
        ),
        FieldSpec(
            "nucleus",
            "Nucleus",
            "categorical",
            context=(r"nucle|1H|31P|23Na|13C|19F|proton|hydrogen|phosph|sodium",),
            vocabulary=("1H", "23Na", "31P", "13C", "19F", "129Xe", "any"),
        ),
        FieldSpec(
            "scanner_type",
            "Scanner type",
            "categorical",
            context=(r"[Cc]ylindric|closed|bore|open",),
            vocabulary=("cylindrical", "open", "any"),
        ),
        FieldSpec(
            "b0_orientation",
            "B0 orientation",
            "categorical",
            context=(r"[Hh]orizontal|[Vv]ertical",),
            vocabulary=("horizontal", "vertical", "any"),
        ),
        FieldSpec(
            "max_sfg_t_per_m",
            "Max spatial field gradient",
            "numeric",
            "gradient",
            "T/m",
            "upper",
            (0.1, 100.0),
            (r"[Ss]patial[^.;]{0,30}[Gg]radient|[Gg]radient[^.;]{0,20}[Ff]ield|SFG|dB/d",),
        ),
        FieldSpec(
            "max_slew_rate_t_per_m_per_s",
            "Max gradient slew rate (per axis)",
            "numeric",
            "slew",
            "T/m/s",
            "upper",
            (10.0, 500.0),
            (r"[Ss]lew|T/m/s|dB/dt",),
        ),
        FieldSpec(
            "rf_excitation",
            "RF excitation",
            "categorical",
            context=(
                r"[Cc]ircular|\bCP\b|[Qq]uadrature|MC-?2|[Mm]ulti-?channel|[Pp]arallel transmit|pTx|"
                r"[Ee]xcitation",
            ),
            vocabulary=("CP", "MC-2", "any"),
        ),
        FieldSpec(
            "transmit_coil",
            "RF transmit coil",
            "categorical",
            context=(r"[Tt]ransmit|T/R|[Cc]oil",),
            vocabulary=("body", "head", "extremity", "local", "any"),
        ),
        FieldSpec(
            "receive_coil",
            "RF receive coil",
            "categorical",
            context=(r"[Rr]eceive|T/R|[Cc]oil",),
            vocabulary=("any", "body", "head", "extremity", "local"),
        ),
        FieldSpec(
            "operating_mode",
            "Operating mode",
            "categorical",
            context=(r"[Nn]ormal|[Ff]irst [Ll]evel|[Oo]perating [Mm]ode",),
            vocabulary=("normal", "first_level"),
        ),
        FieldSpec("b1rms_max_uT", "Max B1+rms", "numeric", "b1", "uT", "upper", (0.1, 10.0), (r"B\s*1|B₁",)),
        FieldSpec(
            "wb_sar_max_w_per_kg",
            "Max whole-body SAR",
            "numeric",
            "sar",
            "W/kg",
            "upper",
            (0.01, 4.0),
            (r"SAR|[Ss]pecific [Aa]bsorption", r"[Ww]hole|[Bb]ody|WB|SAR"),
        ),
        FieldSpec(
            "head_sar_max_w_per_kg",
            "Max head SAR",
            "numeric",
            "sar",
            "W/kg",
            "upper",
            (0.01, 3.2),
            (r"SAR|[Ss]pecific [Aa]bsorption", r"[Hh]ead"),
        ),
        FieldSpec(
            "sar_is_b1rms_fallback",
            "SAR applies only if B1+rms unavailable",
            "bool",
            context=(r"B\s*1|B₁",),
            evaluated=True,
        ),
        FieldSpec(
            "max_active_scan_time_min",
            "Max active scan time",
            "numeric",
            "time",
            "min",
            "upper",
            (0.1, 600.0),
            (r"[Ss]can|RF|[Cc]ontinuous|[Ss]equence|[Ee]xam|[Ss]ession|[Dd]uration|[Aa]ctive|[Ii]maging",),
        ),
        FieldSpec(
            "wait_time_min",
            "Wait/cooling time",
            "numeric",
            "time",
            "min",
            "lower",
            (0.0, 1440.0),
            (r"[Ww]ait|[Cc]ool|[Bb]reak|[Pp]ause|between",),
        ),
        FieldSpec("scan_region_restrictions", "Scan region / exclusion zone", "text"),
        FieldSpec("device_configuration", "Device configuration", "text"),
        FieldSpec(
            "artifact_extent_mm",
            "Image artifact extent",
            "numeric",
            "length",
            "mm",
            "info",
            (0.0, 300.0),
            (r"[Aa]rtifact|[Aa]rtefact|[Vv]oid",),
        ),
        FieldSpec("other_conditions", "Other conditions", "text", evaluated=False),
    ]
}

FIELD_NAMES: tuple[str, ...] = tuple(FIELD_SPECS)


# --------------------------------------------------------------------------------------
# Checker models
# --------------------------------------------------------------------------------------

CheckStatus = Literal["PASS", "FAIL", "INSUFFICIENT_INFO", "NEEDS_REVIEW"]
OutcomeCode = Literal["NO_CONFLICT_FOUND", "CONFLICT_FOUND", "NEEDS_REVIEW"]

OUTCOME_TEXT: dict[str, str] = {
    "NO_CONFLICT_FOUND": "No conflict found with entered conditions — requires MR Safety Officer/Expert sign-off",
    "CONFLICT_FOUND": "Conflict found",
    "NEEDS_REVIEW": "Insufficient information / needs review",
}

DISCLAIMER = (
    "Research prototype — not a medical device. Never use as the sole basis for an MRI safety decision. "
    "MRCond compares entered values with human-verified labeling excerpts and makes no safety "
    "determination. Every result requires review of the cited manufacturer labeling and sign-off by the "
    "MR Safety Officer/Expert."
)


class ScannerProfile(BaseModel):
    """Scanner specifications entered by the user. Values must be finite; absurd values are rejected (HTTP 422)."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    profile_id: str
    name: str
    b0_tesla: float = Field(gt=0, le=25, description="Static field strength (T).")
    scanner_type: str | None = Field(default=None, description="e.g. 'cylindrical'; None = unknown (never assumed).")
    b0_orientation: str | None = Field(default=None, description="e.g. 'horizontal'; None = unknown.")
    max_slew_rate_t_per_m_per_s: float | None = Field(
        default=None, gt=0, le=1000, description="Maximum gradient slew rate per axis (T/m/s)."
    )
    transmit_coils: list[str] = Field(default_factory=lambda: ["body"])
    rf_excitation_modes: list[str] = Field(default_factory=lambda: ["CP"])
    notes: str | None = None


class SequencePlan(BaseModel):
    """RF exposure of one planned sequence as displayed by the scanner (None = not available)."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    name: str = "sequence"
    b1rms_uT: float | None = Field(default=None, ge=0, le=100)
    wb_sar_w_per_kg: float | None = Field(default=None, ge=0, le=20)
    head_sar_w_per_kg: float | None = Field(default=None, ge=0, le=20)


class ProtocolPlan(BaseModel):
    """Planned exam. Values must be finite and non-negative; None means "not entered" (never assumed)."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    operating_mode: Literal["normal", "first_level"] | None = None
    transmit_coil: str | None = None
    receive_coil: str | None = None
    rf_excitation: str | None = None
    nucleus: str | None = Field(default=None, description="Imaged nucleus, e.g. '1H'; None = not entered.")
    sequences: list[SequencePlan] = Field(default_factory=list)
    total_active_scan_time_min: float | None = Field(default=None, ge=0, le=1440)
    sessions: int = Field(default=1, ge=1, le=100)
    wait_between_sessions_min: float | None = Field(default=None, ge=0, le=100000)
    scan_region: str | None = None
    slew_rate_limit_t_per_m_per_s: float | None = Field(
        default=None, gt=0, le=1000, description="Protocol/system slew-rate limit if lower than the scanner maximum."
    )
    sfg_at_implant_t_per_m: float | None = Field(
        default=None, ge=0, le=1000, description="Spatial field gradient at the implant location, if known (T/m)."
    )
    device_configuration_confirmed: bool | None = Field(
        default=None,
        description="True: the user confirms the implant is configured as the labeling requires; "
        "False: it is not (conflict); None: not confirmed (needs review).",
    )


class ConditionCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    label: str
    status: CheckStatus
    reason: str
    label_value: str | None = None
    entered_value: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)


class ConditionSetResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    set_id: str
    scope: str | None = None
    applicable: bool
    status: CheckStatus
    checks: list[ConditionCheck] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ImplantResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    implant_id: str
    device_name: str | None = None
    mr_status: str
    verification: str
    status: CheckStatus
    selected_set_id: str | None = None
    implant_checks: list[ConditionCheck] = Field(default_factory=list)
    set_results: list[ConditionSetResult] = Field(default_factory=list)


class CheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: OutcomeCode
    outcome_text: str
    implants: list[ImplantResult]
    disclaimer: str = DISCLAIMER
    generated_at: datetime | None = None


class CheckRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    implants: list[LabelDocument]
    scanner: ScannerProfile
    protocol: ProtocolPlan


def export_json_schemas(out_dir: str | Path) -> list[Path]:
    """Write JSON Schemas for the public models to `out_dir`."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, model in [
        ("label_document", LabelDocument),
        ("check_request", CheckRequest),
        ("check_result", CheckResult),
    ]:
        p = out / f"{name}.schema.json"
        schema = model.model_json_schema()
        schema["$id"] = f"urn:mrcond:schema:{name}:{SCHEMA_VERSION}"
        p.write_text(json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written.append(p)
    return written
