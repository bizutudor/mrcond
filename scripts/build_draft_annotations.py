"""Build the Track R DRAFT annotations (data/annotations/track_r/*.json).

These annotations were written by an AI assistant ("assistant-draft") from the
documents listed in data/corpus/manifest.json and are marked
``"verification": "draft-unverified"``. They MUST be reviewed by a human (e.g. in
the MRCond review UI) before being called gold.

Every quote below is located verbatim (whitespace-normalised) in the fetched
text; the script fails if a quote is missing or longer than 25 words. Run
``mrcond fetch`` first.

Annotation guidelines (summary; see README "Contributing annotations"):
* only conditions for scanning patients; never test conditions or test results
  (temperature rise, phantom SAR, test durations)
* a duration attached to a condition ("SAR of 2 W/kg for 15 minutes of scanning")
  is a scan-time condition; a duration in a heating-result sentence is not
* MR status only when the text uses the term (MR Conditional / MR Safe / MR Unsafe)
* one condition set per field strength when limits differ; alternative sets for
  configuration- or landmark-dependent limits (scope text explains)
* configuration-dependent values inside one set: most conservative value, note
  in other_conditions
* strict "<" limits are recorded as upper limits (le)
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mrcond.corpus import annotation_path, load_manifest, load_text  # noqa: E402
from mrcond.schema import (  # noqa: E402
    ConditionSet,
    Evidence,
    FieldStatus,
    FieldValue,
    LabelDocument,
    Provenance,
)
from mrcond.textnorm import find_all, word_count  # noqa: E402
from mrcond.units import to_canonical  # noqa: E402

CREATED = datetime(2026, 10, 15, 12, 0, tzinfo=UTC)


class Doc:
    def __init__(self, doc_id: str) -> None:
        entry = load_manifest().get(doc_id)
        text = load_text(entry)
        if text is None:
            raise SystemExit(f"{doc_id}: text not available - run `mrcond fetch` first")
        self.entry, self.text = entry, text

    def ev(self, quote: str) -> Evidence:
        if word_count(quote) > 25:
            raise SystemExit(f"{self.entry.doc_id}: quote longer than 25 words: {quote!r}")
        for p in self.text.pages:
            spans = find_all(p.text, quote)
            if spans:
                s, e = spans[0]
                return Evidence(
                    doc_id=self.entry.doc_id, page=p.number, char_start=s, char_end=e, quote=p.text[s:e], verified=True
                )
        raise SystemExit(f"{self.entry.doc_id}: quote not found: {quote!r}")

    # value helpers -------------------------------------------------------------------------
    def num(self, value: float, unit: str, *quotes: str, comparator: str = "le") -> FieldValue:
        norm, nunit = to_canonical(value, unit)
        return FieldValue(
            status=FieldStatus.EXTRACTED,
            value=value,
            unit=unit,
            normalized_value=norm,
            normalized_unit=nunit,
            comparator=comparator,
            evidence=[self.ev(q) for q in quotes],
        )

    def unlimited(self, quote: str, unit: str = "min") -> FieldValue:
        return FieldValue(
            status=FieldStatus.EXTRACTED,
            value="no limit",
            normalized_unit=unit,
            comparator="any",
            evidence=[self.ev(quote)],
        )

    def b0(self, values: list[float], *quotes: str, le: bool = False) -> FieldValue:
        if le:
            return FieldValue(
                status=FieldStatus.EXTRACTED,
                value=values[0],
                unit="T",
                normalized_value=[values[0]],
                normalized_unit="T",
                comparator="le",
                evidence=[self.ev(q) for q in quotes],
            )
        return FieldValue(
            status=FieldStatus.EXTRACTED,
            value=values,
            unit="T",
            normalized_value=values,
            normalized_unit="T",
            comparator="in",
            evidence=[self.ev(q) for q in quotes],
        )

    def cat(self, stated: str, normalized, *quotes: str, comparator: str = "in") -> FieldValue:
        return FieldValue(
            status=FieldStatus.EXTRACTED,
            value=stated,
            normalized_value=normalized,
            comparator=comparator,
            evidence=[self.ev(q) for q in quotes],
        )

    def text_(self, value: str, *quotes: str, normalized: str | None = None) -> FieldValue:
        return FieldValue(
            status=FieldStatus.EXTRACTED,
            value=value,
            normalized_value=normalized or value,
            comparator="eq",
            evidence=[self.ev(q) for q in quotes],
        )

    def flag(self, quote: str) -> FieldValue:
        return FieldValue(
            status=FieldStatus.EXTRACTED, value=True, normalized_value=True, comparator="eq", evidence=[self.ev(quote)]
        )

    def status(self, value: str, quote: str | None) -> FieldValue:
        if quote is None:
            return FieldValue(status=FieldStatus.NOT_STATED, value=value, normalized_value=value)
        return FieldValue(
            status=FieldStatus.EXTRACTED,
            value=value,
            normalized_value=value,
            comparator="eq",
            evidence=[self.ev(quote)],
        )

    def save(self, status: FieldValue, sets: list[ConditionSet], notes: list[str] | None = None) -> LabelDocument:
        label = LabelDocument(
            doc_id=self.entry.doc_id,
            device_name=self.entry.device_name,
            manufacturer=self.entry.manufacturer,
            device_type=self.entry.device_type,
            mr_status=status,
            condition_sets=sets,
            provenance=Provenance(
                extractor="manual-draft",
                verification="draft-unverified",
                annotator="assistant-draft",
                created_at=CREATED,
                text_extractor=self.text.text_extractor,
                notes=["DRAFT - must be reviewed by a human before use as gold.", *(notes or [])],
            ),
        )
        p = annotation_path(self.entry.doc_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(label.model_dump_json(indent=1, exclude_defaults=True) + "\n", encoding="utf-8")
        return label


def cs(set_id: str = "cs1", scope: str | None = None, **fields: FieldValue) -> ConditionSet:
    c = ConditionSet(set_id=set_id, scope=scope)
    for k, v in fields.items():
        c.set_field(k, v)
    return c


# ---------------------------------------------------------------------------------------------
# annotations
# ---------------------------------------------------------------------------------------------


def fda_t3() -> None:
    d = Doc("fda-guidance-2023-t3")
    d.save(
        d.status(
            "MR Conditional", "Example MR Conditional labeling for a passive medical device called the Star implant."
        ),
        [
            cs(
                b0_tesla=d.b0([1.5, 3.0], "Static Magnetic Field Strength (B 0) 1.5T or 3.0T"),
                max_sfg_t_per_m=d.num(30, "T/m", "Maximum Spatial Field Gradient 30 T/m (3,000 gauss /cm)"),
                rf_excitation=d.cat("Circularly Polarized (CP)", ["CP"], "RF Excitation Circularly Polarized (CP)"),
                transmit_coil=d.cat(
                    "There are no Transmit Coil restrictions",
                    ["any"],
                    "RF Transmit Coil Type There are no Transmit Coil restrictions",
                ),
                operating_mode=d.cat(
                    "Normal Operating Mode", "normal", "Operating Mode Normal Operating Mode", comparator="le"
                ),
                wb_sar_max_w_per_kg=d.num(2, "W/kg", "Maximum Whole -Body SAR 2 W/kg (Normal Operating Mode)"),
                head_sar_max_w_per_kg=d.num(3.2, "W/kg", "Maximum Head SAR 3.2 W/kg (Normal Operating Mode)"),
                max_active_scan_time_min=d.num(
                    60,
                    "minutes",
                    "Scan Duration 2 W/kg whole -body average SAR for 60 minutes of continuous RF (a sequence or back to back series/scan without breaks)",
                ),
            )
        ],
    )


def fda_t4a() -> None:
    d = Doc("fda-guidance-2023-t4a")
    b0q = "Static Magnetic Field Strength (B0) 1.5T and 3T"
    common = dict(
        nucleus=d.cat("31P (phosphorus)", ["1H", "31P"], "Type of Nuclei 31P (phosphorus)"),
        scanner_type=d.cat("Cylindrical", "cylindrical", "MR Scanner Type Cylindrical", comparator="eq"),
        b0_orientation=d.cat("Horizontal", "horizontal", "B0 Field Orientation Horizontal", comparator="eq"),
        max_sfg_t_per_m=d.num(25, "T/m", "Maximum Spatial Field Gradient 25 T/m (2500 gauss/cm)"),
        max_slew_rate_t_per_m_per_s=d.num(200, "T/m/s", "Maximum Gradient Slew Rate 200 T/m/s per axis"),
        rf_excitation=d.cat("Circularly Polarized (CP)", ["CP"], "RF Excitation Circularly Polarized (CP)"),
        transmit_coil=d.cat(
            "Integrated Whole Body Transmit Coil", ["body"], "RF Transmit Coil Type Integrated Whole Body Transmit Coil"
        ),
        receive_coil=d.cat("Any", ["any"], "RF Receive Coil Type Any"),
        operating_mode=d.cat(
            "Normal Operating Mode", "normal", "Operating Mode Normal Operating Mode", comparator="le"
        ),
        max_active_scan_time_min=d.num(30, "minutes", "Scan Duration Scan for up to 30 minutes."),
        wait_time_min=d.num(30, "minutes", "Wait 30 minutes before the next imaging session.", comparator="ge"),
        scan_region_restrictions=d.text_(
            "Any landmark is acceptable", "Scan Regions Any landmark is acceptable", normalized="unrestricted"
        ),
        device_configuration=d.text_("Stimulation OFF", "Device Configuration Stimulation OFF"),
    )
    d.save(
        d.status("MR Conditional", "an MR Conditional active implant called the Star Active Implant System"),
        [
            cs(
                "cs1",
                "Whole-body MR examinations, 1.5 T",
                b0_tesla=d.b0([1.5], b0q),
                wb_sar_max_w_per_kg=d.num(2, "W/kg", "RF Conditions For 1.5T MR Scanner: Whole-Body SAR ≤ 2 W/kg"),
                **common,
            ),
            cs(
                "cs2",
                "Whole-body MR examinations, 3 T",
                b0_tesla=d.b0([3.0], b0q),
                b1rms_max_uT=d.num(1.7, "μT", "For 3T MR Scanner : B1+rms ≤ 1.7 μT"),
                wb_sar_max_w_per_kg=d.num(
                    1.2, "W/kg", "for MR scanners that do not report B1+rms, Whole-Body SAR ≤ 1.2 W/kg"
                ),
                sar_is_b1rms_fallback=d.flag("for MR scanners that do not report B1+rms, Whole-Body SAR ≤ 1.2 W/kg"),
                **common,
            ),
        ],
    )


def fda_t4b() -> None:
    d = Doc("fda-guidance-2023-t4b")
    coil = "RF Coil Type Detachable Head Transmit/Receive Coil ONLY"
    d.save(
        d.status("MR Conditional", "an MR Conditional active implant called the Star Active Implant System"),
        [
            cs(
                scope="Head MR examinations",
                b0_tesla=d.b0([1.5, 3.0], "Static Magnetic Field Strength (B0) 1.5T and 3T"),
                nucleus=d.cat("No additional nuclei", ["1H"], "Type of Nuclei No additional nuclei"),
                scanner_type=d.cat("Cylindrical", "cylindrical", "MR Scanner Type Cylindrical", comparator="eq"),
                b0_orientation=d.cat("Horizontal", "horizontal", "B0 Field Orientation Horizontal", comparator="eq"),
                max_sfg_t_per_m=d.num(25, "T/m", "Maximum Spatial Field Gradient 25 T/m (2500 gauss/cm)"),
                max_slew_rate_t_per_m_per_s=d.num(200, "T/m/s", "Maximum Gradient Slew Rate 200 T/m/s per axis"),
                rf_excitation=d.cat("Circularly Polarized (CP)", ["CP"], "RF Excitation Circularly Polarized (CP)"),
                transmit_coil=d.cat("Detachable Head Transmit/Receive Coil ONLY", ["head"], coil),
                receive_coil=d.cat("Detachable Head Transmit/Receive Coil ONLY", ["head"], coil),
                operating_mode=d.cat(
                    "Normal Operating Mode or First Level Controlled Operating Mode",
                    "first_level",
                    "Operating Mode Normal Operating Mode or First Level Controlled Operating Mode",
                    comparator="le",
                ),
                head_sar_max_w_per_kg=d.num(3.2, "W/kg", "Maximum Head SAR 3.2 W/kg"),
                max_active_scan_time_min=d.unlimited("Scan Duration There is no limit on scan duration."),
                scan_region_restrictions=d.text_("Head Only", "Scan Regions Head Only"),
                device_configuration=d.text_("Stimulation OFF", "Device Configuration Stimulation OFF"),
            )
        ],
    )


def gore_tigris() -> None:
    d = Doc("gore-tigris-mri-html")
    d.save(
        d.status("MR Conditional", "MR status MR Conditional"),
        [
            cs(
                b0_tesla=d.b0([1.5, 3.0], "Static magnetic fields strength 1.5 tesla or 3.0 tesla"),
                max_sfg_t_per_m=d.num(4000, "gauss/cm", "Maximum spatial gradient 4000 gauss/cm"),
                wb_sar_max_w_per_kg=d.num(
                    4.0, "W/kg", "Max whole-body-averaged specific absorption rate (SAR) 4.0 W/kg"
                ),
                max_active_scan_time_min=d.num(15, "minutes", "Scan duration 15 minutes of scanning"),
                artifact_extent_mm=d.num(
                    5, "mm", "The image artifact extends approximately 5 mm from both the device", comparator="eq"
                ),
            )
        ],
    )


def gore_viabahn() -> None:
    d = Doc("gore-viabahn-mri-html-emea")
    d.save(
        d.status("MR Conditional", "MR status MR Conditional"),
        [
            cs(
                b0_tesla=d.b0([1.5, 3.0], "Static magnetic field strengths 1.5 Tesla or 3.0 Tesla"),
                max_sfg_t_per_m=d.num(30, "T/m", "Maximum spatial gradient 30 T/m (3000 gauss/cm)"),
                wb_sar_max_w_per_kg=d.num(
                    2, "W/kg", "Max whole-body-averaged specific absorption rate (SAR) 2 W/kg (normal operating mode)"
                ),
                operating_mode=d.cat(
                    "normal operating mode", "normal", "2 W/kg (normal operating mode)", comparator="le"
                ),
                max_active_scan_time_min=d.num(
                    60,
                    "minutes",
                    "Scan duration 60 minutes of continuous RF (a sequence or back to back series/scan without breaks)",
                ),
                artifact_extent_mm=d.num(
                    20,
                    "mm",
                    "the artifact may extend up to 5 mm from the implant",
                    "When the VIABAHN ® Device was placed within a bare nitinol stent the image artifact extended up to 20 mm from the devices.",
                    comparator="eq",
                ),
                other_conditions=d.text_(
                    "Device within a bare nitinol stent: use the lesser of the bare stent's SAR limit or 2.0 W/kg",
                    "For a VIABAHN ® Device placed within a bare nitinol stent, the lesser of the maximum whole-body averaged SAR reported in the bare nitinol stent's",
                ),
            )
        ],
        notes=["Artifact: 5 mm for the device alone, 20 mm inside a bare nitinol stent; the larger value is recorded."],
    )


def gore_fortegra() -> None:
    d = Doc("gore-viabahn-fortegra-card")
    d.save(
        d.status("MR Conditional", "MR Conditional Caution Consult Instructions for Use"),
        [
            cs(
                b0_tesla=d.b0([1.5, 3.0], "Static Magnetic Field Strength (B0) 1.5T or 3.0T"),
                max_sfg_t_per_m=d.num(40, "T/m", "Maximum Spatial Field Gradient 40 T/m (4,000 gauss/cm)"),
                rf_excitation=d.cat(
                    "There are no RF excitation restrictions",
                    ["any"],
                    "RF Excitation There are no RF excitation restrictions",
                ),
                transmit_coil=d.cat(
                    "There are no Transmit Coil restrictions",
                    ["any"],
                    "RF Transmit Coil Type There are no Transmit Coil restrictions",
                ),
                operating_mode=d.cat(
                    "Normal Operating Mode", "normal", "Operating Mode Normal Operating Mode", comparator="le"
                ),
                wb_sar_max_w_per_kg=d.num(2, "W/kg", "Maximum Whole-Body SAR 2 W/kg (Normal Operating Mode)"),
                max_active_scan_time_min=d.num(
                    60,
                    "minutes",
                    "Scan Duration 2 W/kg whole-body averaged SAR for 60 minutes of continuous RF (a sequence or back to back series/scan without breaks)",
                ),
                artifact_extent_mm=d.num(
                    5, "mm", "the artifact may extend up to 5mm from the implant", comparator="eq"
                ),
            )
        ],
    )


def bsc_innova() -> None:
    d = Doc("bsc-innova-stent-ifu")
    b0 = d.b0([1.5, 3.0], "Static magnetic field of 1.5 or 3.0 Tesla")
    sfg = d.num(40, "Tesla/m", "Highest spatial gradient magnetic field of 40 Tesla/m (4,000 Gauss/cm) or less")
    art = d.num(
        12,
        "mm",
        "the image artifact caused by the device extends approximately 12 mm from the Innova™ stent",
        comparator="eq",
    )
    oth = d.text_(
        "MR Conditional for single and overlapping lengths up to 200 mm",
        "MR Conditional for single and overlapping lengths up to 200mm",
    )
    above = "≤2 W/kg for landmarks (i.e. center of RF coil) above the umbilicus"
    below = "≤1 W/kg for landmarks below the umbilicus"
    d.save(
        d.status("MR Conditional", "Non-clinical testing has demonstrated the Innova™ Stents are MR Conditional"),
        [
            cs(
                "cs1",
                "Landmark (center of RF coil) above the umbilicus",
                b0_tesla=b0,
                max_sfg_t_per_m=sfg,
                wb_sar_max_w_per_kg=d.num(2, "W/kg", above),
                scan_region_restrictions=d.text_("Landmarks (center of RF coil) above the umbilicus", above),
                artifact_extent_mm=art,
                other_conditions=oth,
            ),
            cs(
                "cs2",
                "Landmark below the umbilicus",
                b0_tesla=b0,
                max_sfg_t_per_m=sfg,
                wb_sar_max_w_per_kg=d.num(1, "W/kg", below),
                scan_region_restrictions=d.text_("Landmarks below the umbilicus", below),
                artifact_extent_mm=art,
                other_conditions=oth,
            ),
        ],
        notes=["'15 minutes of continuous scanning' belongs to the heating result, not a scan-time condition."],
    )


def bsc_promus() -> None:
    d = Doc("bsc-promus-elite-mri-letter")
    d.save(
        d.status("MR Conditional", "the Promus ELITE Stent has been shown to be MR Conditional"),
        [
            cs(
                b0_tesla=d.b0([3.0], "Field strengths of 3 Tesla or less", le=True),
                max_sfg_t_per_m=d.num(14, "T/m", "Static magnetic field gradient < 14 T/m (extrapolated)"),
                wb_sar_max_w_per_kg=d.num(
                    2.0, "W/kg", "A maximum whole body averaged specific absorption rate (SAR) of lower than 2.0 W/kg"
                ),
                max_active_scan_time_min=d.num(
                    15, "minutes", "for a total active MR scan time (with RF exposure) of 15 minutes or less"
                ),
                artifact_extent_mm=d.num(
                    8,
                    "mm",
                    "The calculated image artifact extends approximately 8 mm from the perimeter of the device diameter",
                    comparator="eq",
                ),
                other_conditions=d.text_(
                    "B0 x spatial gradient product < 25 T²/m; dB/dt <= 60 T/s",
                    "Product of static magnetic field and static magnetic field gradient < 25 T²/m (extrapolated)",
                    "A calculated rate of change of magnetic field (dB/dt) of 60 T/s or less",
                ),
            )
        ],
        notes=[
            "Strict '<' limits recorded as upper limits.",
            "Temperature-information paragraphs (2.3/2.1 W/kg phantom SAR, 15 min RF exposure) are test results.",
        ],
    )


def livanova_cphv() -> None:
    d = Doc("livanova-cphv-mri-us")
    sar_q = "Whole body averaged specific absorption rate (SAR) of 2-W/kg in the Normal Operating Mode"
    d.save(
        d.status("MR Conditional", "Table 2a: MR Conditional Products"),
        [
            cs(
                b0_tesla=d.b0([3.0], "Static magnetic field of 3-Tesla or less", le=True),
                max_sfg_t_per_m=d.num(
                    720, "Gauss/cm", "Maximum spatial gradient magnetic field of 720-Gauss/cm or less"
                ),
                wb_sar_max_w_per_kg=d.num(2, "W/kg", sar_q),
                operating_mode=d.cat("Normal Operating Mode", "normal", sar_q, comparator="le"),
                max_active_scan_time_min=d.num(
                    15, "minutes", "physiological stress to PATIENTS) for 15 minutes (i.e., per pulse sequence)."
                ),
                artifact_extent_mm=d.num(
                    10,
                    "mm",
                    "extends approximately 10-mm relative to the size and shape of the device",
                    comparator="eq",
                ),
            )
        ],
    )


def acumed_radial_head() -> None:
    d = Doc("acumed-radial-head-mr")
    d.save(
        d.status("Not stated", None),
        [
            cs(
                b0_tesla=d.b0([3.0], "Nominal value(s) of Static Magnetic Field 3.0 T"),
                max_sfg_t_per_m=d.num(19, "T/m", "Maximum Spatial Field Gradient 19 T/m (1900 gauss/cm)"),
                rf_excitation=d.cat("Circularly Polarized (CP)", ["CP"], "RF Excitation Circularly Polarized (CP)"),
                transmit_coil=d.cat(
                    "Whole body transmit coil / Head RF transmit-receive coil",
                    ["body", "head"],
                    "RF Transmit Coil Type Whole body transmit coilHead RF transmit-receive coil",
                ),
                wb_sar_max_w_per_kg=d.num(2.0, "W/kg", "Maximum Whole-Body SAR at 1.5 T and 3.0 T 2.0 W/kg"),
                max_active_scan_time_min=d.num(
                    12,
                    "minutes",
                    "Limits on Scan Duration 3.0 T 2.0 W/kg whole-body average SAR for twelve (12) minutes of continuous RF",
                ),
                artifact_extent_mm=d.num(
                    71, "mm", "The presence of this implant may produce an image artifact of 71 mm", comparator="eq"
                ),
            )
        ],
        notes=[
            "No 'MR Conditional' term in the text layer (status probably conveyed by a symbol image).",
            "Static-field row lists 3.0 T only, but SAR/scan-duration rows also mention 1.5 T (15 min): "
            "annotated conservatively as 3.0 T only - ambiguous label, needs human review.",
        ],
    )


def atricure() -> None:
    d = Doc("atricure-atriclip-implant-card")
    d.save(
        d.status("MR Conditional", "The LAA Exclusion System Clip is MR Conditional."),
        [
            cs(
                b0_tesla=d.b0([1.5, 3.0], "Nominal Values of Static Magnetic Field: 1.5-Tesla or 3.0-Tesla"),
                max_sfg_t_per_m=d.num(40, "T/m", "Maximum Spatial Field Gradient: 40-T/m (4,000-gauss/cm)"),
                rf_excitation=d.cat(
                    "Circularly Polarized (CP)",
                    ["CP"],
                    "Type of RF Excitation: Circularly Polarized (CP) (i.e., Quadrature Transmission)",
                ),
                transmit_coil=d.cat(
                    "There are no transmit RF coil restrictions",
                    ["any"],
                    "Transmit RF Coil Information: There are no transmit RF coil restrictions",
                ),
                operating_mode=d.cat(
                    "Normal Operating Mode",
                    "normal",
                    "Operating Mode of MR System: Normal Operating Mode",
                    comparator="le",
                ),
                wb_sar_max_w_per_kg=d.num(2, "W/kg", "Maximum Whole Body Averaged SAR: 2-W/kg (Normal Operating Mode)"),
                max_active_scan_time_min=d.num(
                    60,
                    "minutes",
                    "Limits on Scan Duration: Whole body averaged SAR of 2-W/kg for 60 minutes of continuous RF exposure",
                ),
            )
        ],
    )


def aesculap_plating() -> None:
    d = Doc("aesculap-neuro-plating-card")
    sar_q = "whole body averaged specific absorption rate (SAR) of 1 W/kg and head SAR of 1 W/kg for landmarks above the shoulder"
    cool = "A cool down period of 5 minutes is needed after each 7 minutes of continuous scanning."
    d.save(
        d.status("MR Conditional", "Non-clinical testing has demonstrated the Neuro Plating System is MR Conditional."),
        [
            cs(
                b0_tesla=d.b0([1.5, 3.0], "Static magnetic field of 1.5 T and 3.0 T"),
                max_sfg_t_per_m=d.num(
                    3000, "gauss/cm", "Maximum spatial magnetic field gradient of 3,000 gauss/cm (30T/m)"
                ),
                wb_sar_max_w_per_kg=d.num(1, "W/kg", sar_q),
                head_sar_max_w_per_kg=d.num(1, "W/kg", sar_q),
                scan_region_restrictions=d.text_("Landmarks above the shoulder", sar_q),
                operating_mode=d.cat(
                    "Normal Operating Mode for gradient output",
                    "normal",
                    "Normal Operating Mode for gradient output",
                    comparator="le",
                ),
                max_active_scan_time_min=d.num(7, "minutes", cool),
                wait_time_min=d.num(5, "minutes", cool, comparator="ge"),
                artifact_extent_mm=d.num(
                    2,
                    "mm",
                    "the image artifact caused by the device extends approximately 2 mm from the Neuro Plating System",
                    comparator="eq",
                ),
            )
        ],
        notes=[
            "The 7-minute limit and 5-minute cool-down are stated as requirements ('is needed').",
            "Operating-mode statement refers to gradient output.",
        ],
    )


def aesculap_yasargil() -> None:
    d = Doc("aesculap-yasargil-clip-card")
    d.save(
        d.status("Not stated", None),
        [
            cs(
                b0_tesla=d.b0([3.0], "Static magnetic field of 3 Tesla or less", le=True),
                max_sfg_t_per_m=d.num(720, "Gauss/cm", "Maximum spatial gradient field of 720 Gauss/cm or less"),
            )
        ],
        notes=[
            "The card lists scanning conditions but does not use the term 'MR Conditional' for the clip.",
            "'15 minutes of continuous scanning' belongs to the heating result.",
        ],
    )


def bioretec() -> None:
    d = Doc("bioretec-remeos-mri-html")
    d.save(
        d.status("MR Conditional", "Non-clinical testing demonstrated that the RemeOs™ Screw is MR Conditional."),
        [
            cs(
                b0_tesla=d.b0([1.5, 3.0], "Static Magnetic Field Strength (B0) 1.5T or 3.0T"),
                rf_excitation=d.cat("Circularly Polarized (CP)", ["CP"], "RF Excitation Circularly Polarized (CP)"),
                transmit_coil=d.cat("Volume RF body coil", ["body"], "RF Transmit Coil Type Volume RF body coil"),
                operating_mode=d.cat(
                    "Normal Operating Mode", "normal", "Operating Mode Normal Operating Mode", comparator="le"
                ),
                wb_sar_max_w_per_kg=d.num(2, "W/kg", "Maximum Whole-Body SAR 2 W/kg (Normal Operating Mode)"),
                head_sar_max_w_per_kg=d.num(3.2, "W/kg", "Maximum Head SAR 3.2W/kg"),
                max_active_scan_time_min=d.num(
                    25, "minutes", "the implant can be scanned continuously for 25 minutes."
                ),
                wait_time_min=d.num(
                    5, "minutes", "With 5 minutes break, this can be repeated twice in 60 minutes.", comparator="ge"
                ),
                other_conditions=d.text_(
                    "Repeat at most twice in 60 minutes",
                    "With 5 minutes break, this can be repeated twice in 60 minutes.",
                ),
            )
        ],
    )


def medtronic_linq() -> None:
    d = Doc("medtronic-reveal-linq-mri-html")
    geo = "Horizontal cylindrical bore magnet, clinical MRI systems"
    d.save(
        d.status("MR Conditional", "Reveal LINQ™ ICM is MR Conditional at 3.0 and 1.5 T"),
        [
            cs(
                b0_tesla=d.b0(
                    [1.5, 3.0],
                    "clinical MRI systems with a static magnetic field of 1.5 Tesla (T) or 3.0 T must be used.",
                ),
                scanner_type=d.cat("cylindrical bore", "cylindrical", geo, comparator="eq"),
                b0_orientation=d.cat("Horizontal", "horizontal", geo, comparator="eq"),
                nucleus=d.cat("Hydrogen proton", ["1H"], "Hydrogen proton MRI equipment must be used."),
                max_sfg_t_per_m=d.num(
                    25,
                    "T/m",
                    "Maximum spatial gradient of the static magnetic field specification must be ≤25 T/m (2500 gauss/ cm).",
                ),
                max_slew_rate_t_per_m_per_s=d.num(
                    200, "T/m/s", "gradient slew rate specification must be ≤200 T/m/s per axis."
                ),
                wb_sar_max_w_per_kg=d.num(
                    4.0,
                    "W/kg",
                    "The Whole Body Specific Absorption Rate (WB-SAR) as reported by the MRI equipment must be ≤4.0 W/kg",
                ),
                head_sar_max_w_per_kg=d.num(
                    3.2, "W/kg", "the head SAR as reported by the MRI equipment must be ≤3.2 W/kg."
                ),
                receive_coil=d.cat(
                    "no restrictions on receive-only coils",
                    ["any"],
                    "There are no restrictions on the placement of receive-only coils",
                ),
                other_conditions=d.text_(
                    "Do not use local transmit coils on the chest, trunk, or shoulder region",
                    "Do not use local transmit coils on the chest, trunk, or shoulder region.",
                ),
            )
        ],
        notes=[
            "Transmit-coil permission is location dependent (local transmit coils allowed for head/extremities "
            "only): recorded under other_conditions rather than as a permitted-coil list."
        ],
    )


def bsc_lux_dx() -> None:
    d = Doc("bsc-lux-dx-icm-mri-guide")
    geo = "Horizontal,1H proton,closed bore scanners only"
    d.save(
        d.status("MR Conditional", "The ICM device is an MRConditional device"),
        [
            cs(
                b0_tesla=d.b0(
                    [1.5, 3.0], "MRI magnetstrength of1.5 T (approximately 64 MHz)or 3 T (approximately 128MHz)"
                ),
                scanner_type=d.cat("closed bore", "cylindrical", geo, comparator="eq"),
                b0_orientation=d.cat("Horizontal", "horizontal", geo, comparator="eq"),
                nucleus=d.cat("1H proton", ["1H"], geo),
                max_sfg_t_per_m=d.num(30, "T/m", "Spatial gradientno greater than 30 T/m (3,000 G/cm)"),
                operating_mode=d.cat(
                    "First Level Controlled Operating Mode",
                    "first_level",
                    "SAR limits up to First Level ControlledOperating Mode1 may beapplied for the entireactive scan session",
                    comparator="le",
                ),
                wb_sar_max_w_per_kg=d.num(4.0, "W/kg", "Whole bodyaveraged,≤4.0 W/kg"),
                head_sar_max_w_per_kg=d.num(3.2, "W/kg", "Head, ≤3.2 W/kg"),
                max_slew_rate_t_per_m_per_s=d.num(
                    200, "T/m/s", "Maximum specified gradientslew rate≤200 T/m/s peraxis"
                ),
                receive_coil=d.cat(
                    "receive-only coils not restricted", ["any"], "Theuse of receive-onlycoilsis not restricted."
                ),
                other_conditions=d.text_(
                    "Local transmit coils not directly over the device; supine or prone position only; "
                    "models M301, M302, M312",
                    "Local transmit coilsmay be used,but should not be placed directly overthe device.",
                    "Patient in supineor prone position only",
                ),
            )
        ],
        notes=["Text layer drops spaces ('of1.5 T'); quotes are verbatim from the extracted text."],
    )


def biotronik() -> None:
    d = Doc("biotronik-promri-radiology-checklist")
    d.save(
        d.status("Not stated", None),
        [
            cs(
                b0_tesla=d.b0([1.5], "static magnetic field strength of 1.5 tesla"),
                scanner_type=d.cat(
                    "closed tube, cylindrical magnets",
                    "cylindrical",
                    "Use a clinical MRI system with a closed tube, cylindrical magnets",
                    comparator="eq",
                ),
                max_slew_rate_t_per_m_per_s=d.num(
                    200,
                    "T/m/s",
                    "The slew rate of the MRI scanner's gradient fields should not exceed 200 T/m/s per axis.",
                ),
                wb_sar_max_w_per_kg=d.num(
                    2.0,
                    "W/kg",
                    "The mean specific absorption rate for the whole body displayed by the MRI scanner must not exceed 2.0 W/kg.",
                ),
                head_sar_max_w_per_kg=d.num(
                    3.2, "W/kg", "The head absorption rate displayed by the MRI scanner must not exceed 3.2 W/kg."
                ),
                device_configuration=d.text_(
                    "Programmed to MRI mode",
                    "Confirm with the cardiology department that the pacemaker or ICD is programmed to MRI mode prior to the MRI scan.",
                ),
                other_conditions=d.text_(
                    "Supine position; continuous monitoring; no abandoned leads/extenders/adaptors; "
                    "other metal implants >5 cm at least 4 cm from the lead",
                    "The MR scan should be performed with the patient in supine position.",
                    "Continuously monitor the patient's condition during the entire MR scan",
                ),
            )
        ],
        notes=["Checklist never labels the ProMRI system itself 'MR Conditional' (status not stated in this text)."],
    )


def vivokey() -> None:
    d = Doc("vivokey-spark2-mri")
    d.save(
        d.status("MR Conditional", "The VivoKey Spark 2 is MR Conditional."),
        [
            cs(
                b0_tesla=d.b0([1.5, 3.0], "Nominal Values of Static Magnetic Field (T) 1.5-T and 3-T"),
                max_sfg_t_per_m=d.num(
                    40, "T/m", "Maximum Spatial Field Gradient (T/m and gauss/cm) 40-T/m (4,000-gauss/cm)"
                ),
                rf_excitation=d.cat(
                    "Circularly Polarized (CP)",
                    ["CP"],
                    "Type of RF Excitation Circularly Polarized (CP) (i.e., quadrature-driven)",
                ),
                transmit_coil=d.cat(
                    "There are no transmit RF coil restrictions",
                    ["any"],
                    "Transmit RF Coil Information There are no transmit RF coil restrictions.",
                ),
                operating_mode=d.cat(
                    "Normal Operating Mode",
                    "normal",
                    "Operating Mode of MR System Normal Operating Mode",
                    comparator="le",
                ),
                wb_sar_max_w_per_kg=d.num(2, "W/kg", "Maximum Whole Body Averaged SAR 2-W/kg (Normal Operating Mode)"),
                max_active_scan_time_min=d.num(
                    60,
                    "minutes",
                    "Limits on Scan Duration Whole body averaged SAR of 2-W/kg for 60 minutes of continuous RF exposure",
                ),
            )
        ],
        notes=["'3-Tesla or less' in the important note concerns displacement, not a permitted field strength."],
    )


def abbott() -> None:
    d = Doc("abbott-mri-ready-pacing-manual")
    img = "Imaging System 1.5 Tesla/64 MHz excitation frequency (hydrogen atom)"
    geo = "Scanner Type Horizontal Closed Bore Horizontal Closed Bore"
    wb_row = "Whole Body SAR (Specific Absorption Rate) ¢4 W/kg ¢2 W/kg"
    common = dict(
        b0_tesla=d.b0([1.5], img),
        nucleus=d.cat("hydrogen atom", ["1H"], img),
        scanner_type=d.cat("Closed Bore", "cylindrical", geo, comparator="eq"),
        b0_orientation=d.cat("Horizontal", "horizontal", geo, comparator="eq"),
        head_sar_max_w_per_kg=d.num(3.2, "W/kg", "Head SAR ¢3.2 W/kg ¢3.2 W/kg"),
        max_slew_rate_t_per_m_per_s=d.num(200, "T/m/s", "Gradient Slew Rate ¢200 T/m/s ¢200 T/m/s"),
        receive_coil=d.cat("Receive-only coils", ["any"], "Receive -only coils can be safely used."),
        device_configuration=d.text_("MRI Settings enabled", "Enable the MRI Settings to turn on a mode of operation"),
        other_conditions=d.text_(
            "Local transmit-only or transmit/receive coils not studied",
            "Use of local tra nsmit -only coils or local transmit/receive coils has not been studied.",
        ),
    )
    d.save(
        d.status(
            "MR Conditional",
            "The St. Jude Medical MR Conditional pacing system includes a St. Jude Medical MR Conditional pulse generator",
        ),
        [
            cs(
                "cs1",
                "Scan Set A (Accent MRI, Assurity MRI, Endurity MRI + Tendril MRI LPA1200M)",
                wb_sar_max_w_per_kg=d.num(4, "W/kg", wb_row),
                scan_region_restrictions=d.text_("Full Body", "Scan Regions Full Body", normalized="unrestricted"),
                **common,
            ),
            cs(
                "cs2",
                "Scan Set B (Endurity PM1162/PM2162 + Tendril MRI LPA1200M)",
                wb_sar_max_w_per_kg=d.num(2, "W/kg", wb_row),
                scan_region_restrictions=d.text_(
                    "Isocenter 10 cm superior to C1 vertebra or at/inferior to L4 vertebra",
                    "Superior: Isocenter 10 cm superior to C1 vertebra Inferior: Isocenter at or inferior to L4 vertebra",
                ),
                max_active_scan_time_min=d.num(
                    30,
                    "minutes",
                    "the accumulated imaging time per MRI procedure (as displayed by the MRI scanner) should not exceed 30 minutes.",
                ),
                wait_time_min=d.num(
                    30,
                    "minutes",
                    "Do not start another MRI procedure until 30 minutes after the end of the initial procedure.",
                    comparator="ge",
                ),
                **common,
            ),
        ],
        notes=[
            "The text layer renders the 'less than or equal' sign as '¢'.",
            "Scan set depends on the pulse generator/lead combination (Table 2).",
        ],
    )


def cook_celect() -> None:
    d = Doc("cook-celect-platinum-ivc-ifu")
    sar_q = "whole-body-averaged specific absorption rate (SAR) of 2.0 W/ kg for 15 minutes of scanning or less (i.e., per scanning sequence)."
    d.save(
        d.status(
            "MR Conditional", "Non-clinical testing has demonstrated that the Cook Celect Filter is MR Conditional"
        ),
        [
            cs(
                b0_tesla=d.b0([1.5, 3.0], "Static magnetic field of 3.0 Tesla or 1.5 Tesla."),
                max_sfg_t_per_m=d.num(
                    1600, "Gauss/cm", "Maximum spatial magnetic gradient of 1600 Gauss/cm (16.0 T/m) or less."
                ),
                operating_mode=d.cat("Normal operating mode", "normal", "Normal operating mode.", comparator="le"),
                wb_sar_max_w_per_kg=d.num(2.0, "W/kg", sar_q),
                max_active_scan_time_min=d.num(15, "minutes", sar_q),
                artifact_extent_mm=d.num(
                    21,
                    "mm",
                    "the area of interest is within approximately 21 mm of the position of the Cook Celect Filter",
                    comparator="eq",
                ),
            )
        ],
        notes=["Heating paragraphs (2.05/2.08 W/kg, 15 minutes of MRI) are test results."],
    )


def argon_option() -> None:
    d = Doc("argon-option-elite-ivc-ifu")
    sar_q = "Maximum whole body averaged specific absorption rate (SAR) of 3.0 W/kg for 15min of scanning"
    d.save(
        d.status(
            "MR Conditional", "Non-clinical testing has demonstrated that the Option™ELITE Filter is MR Conditional."
        ),
        [
            cs(
                b0_tesla=d.b0([3.0], "Static magnetic field of 3 Tesla or less", le=True),
                max_sfg_t_per_m=d.num(720, "Gauss/cm", "Spatial gradient magnetic field of 720 Gauss/cm or less"),
                wb_sar_max_w_per_kg=d.num(3.0, "W/kg", sar_q),
                max_active_scan_time_min=d.num(15, "min", sar_q),
            )
        ],
        notes=[
            "Heating paragraph (3.0 W/kg for 15 minutes in a 3.0 Tesla scanner, 2.8 W/kg calorimetry) is a test result."
        ],
    )


ALL = [
    fda_t3,
    fda_t4a,
    fda_t4b,
    gore_tigris,
    gore_viabahn,
    gore_fortegra,
    bsc_innova,
    bsc_promus,
    livanova_cphv,
    acumed_radial_head,
    atricure,
    aesculap_plating,
    aesculap_yasargil,
    bioretec,
    medtronic_linq,
    bsc_lux_dx,
    biotronik,
    vivokey,
    abbott,
    cook_celect,
    argon_option,
]


def main() -> None:
    for fn in ALL:
        fn()
        print("ok", fn.__name__)


if __name__ == "__main__":
    main()
