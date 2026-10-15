"""Checker scenario suite: synthetic implants x scanner x protocol with expected outcomes.

Expected outcomes are written by hand from the labeling semantics (they do not
call the checker). Most scenarios are *mutations* of a passing base case that
violate or remove exactly one condition, so the expected outcome is known by
construction. The key safety metric is the number of **unsafe PASSes**: the
checker returns "No conflict found" (or a per-condition PASS) where the
expected result is a conflict or needs-review. The target is zero.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from ..corpus import data_dir
from ..schema import (
    ConditionSet,
    Evidence,
    FieldStatus,
    FieldValue,
    LabelDocument,
    ProtocolPlan,
    Provenance,
    ScannerProfile,
)

__all__ = ["build_implants", "build_scenarios", "run_scenarios", "save_scenarios", "scenarios_path"]


def scenarios_path() -> Path:
    return data_dir() / "benchmark" / "checker_scenarios.json"


def _ev(doc_id: str, quote: str) -> list[Evidence]:
    return [Evidence(doc_id=doc_id, page=1, char_start=0, char_end=len(quote), quote=quote, verified=True)]


def _num(doc: str, v: float, unit: str, quote: str, comparator: str = "le") -> FieldValue:
    return FieldValue(
        status=FieldStatus.EXTRACTED,
        value=v,
        unit=unit,
        normalized_value=v,
        normalized_unit=unit,
        comparator=comparator,
        evidence=_ev(doc, quote),
    )


def _b0(doc: str, vals: list[float], quote: str, le: bool = False) -> FieldValue:
    return FieldValue(
        status=FieldStatus.EXTRACTED,
        value=vals,
        unit="T",
        normalized_value=vals,
        normalized_unit="T",
        comparator="le" if le else "in",
        evidence=_ev(doc, quote),
    )


def _cat(doc: str, norm, quote: str, comparator: str = "in") -> FieldValue:
    return FieldValue(
        status=FieldStatus.EXTRACTED,
        value=quote,
        normalized_value=norm,
        comparator=comparator,
        evidence=_ev(doc, quote),
    )


def _status(doc: str, value: str) -> FieldValue:
    return FieldValue(
        status=FieldStatus.EXTRACTED,
        value=value,
        normalized_value=value,
        comparator="eq",
        evidence=_ev(doc, f"The device is {value}."),
    )


def _label(
    doc_id: str, name: str, status: str, sets: list[ConditionSet], verification: str = "human-verified"
) -> LabelDocument:
    st = _status(doc_id, status) if status != "Not stated" else FieldValue()
    return LabelDocument(
        doc_id=doc_id,
        device_name=name,
        device_type="synthetic scenario implant",
        mr_status=st,
        condition_sets=sets,
        provenance=Provenance(extractor="scenario-fixture", verification=verification, reviewer="scenario fixture"),
    )


def build_implants() -> dict[str, LabelDocument]:
    imp: dict[str, LabelDocument] = {}

    d = "IMP-PASSIVE-STAR"  # modelled on FDA 2023 guidance Table 3
    imp[d] = _label(
        d,
        "Passive implant (FDA Table 3 style)",
        "MR Conditional",
        [
            ConditionSet(
                b0_tesla=_b0(d, [1.5, 3.0], "Static Magnetic Field Strength (B0) 1.5T or 3.0T"),
                max_sfg_t_per_m=_num(d, 30, "T/m", "Maximum Spatial Field Gradient 30 T/m (3,000 gauss/cm)"),
                rf_excitation=_cat(d, ["CP"], "RF Excitation Circularly Polarized (CP)"),
                transmit_coil=_cat(d, ["any"], "There are no Transmit Coil restrictions"),
                operating_mode=_cat(d, "normal", "Normal Operating Mode", "le"),
                wb_sar_max_w_per_kg=_num(d, 2, "W/kg", "Maximum Whole-Body SAR 2 W/kg"),
                head_sar_max_w_per_kg=_num(d, 3.2, "W/kg", "Maximum Head SAR 3.2 W/kg"),
                max_active_scan_time_min=_num(d, 60, "min", "60 minutes of continuous RF"),
            )
        ],
    )

    d = "IMP-STAR-ACTIVE"  # modelled on FDA Tables 4a + 4b (three alternative condition sets)
    common = dict(
        nucleus=_cat(d, ["1H", "31P"], "Type of Nuclei 31P (phosphorus)"),
        scanner_type=_cat(d, "cylindrical", "MR Scanner Type Cylindrical", "eq"),
        b0_orientation=_cat(d, "horizontal", "B0 Field Orientation Horizontal", "eq"),
        max_sfg_t_per_m=_num(d, 25, "T/m", "Maximum Spatial Field Gradient 25 T/m"),
        max_slew_rate_t_per_m_per_s=_num(d, 200, "T/m/s", "Maximum Gradient Slew Rate 200 T/m/s per axis"),
        rf_excitation=_cat(d, ["CP"], "RF Excitation Circularly Polarized (CP)"),
        device_configuration=_cat(d, "Stimulation OFF", "Device Configuration Stimulation OFF", "eq"),
    )
    wb_common = dict(
        transmit_coil=_cat(d, ["body"], "Integrated Whole Body Transmit Coil"),
        receive_coil=_cat(d, ["any"], "RF Receive Coil Type Any"),
        operating_mode=_cat(d, "normal", "Normal Operating Mode", "le"),
        max_active_scan_time_min=_num(d, 30, "min", "Scan for up to 30 minutes."),
        wait_time_min=_num(d, 30, "min", "Wait 30 minutes before the next imaging session.", "ge"),
        scan_region_restrictions=_cat(d, "unrestricted", "Any landmark is acceptable", "eq"),
    )
    sets = [
        ConditionSet(
            set_id="wb-1.5",
            scope="Whole body, 1.5 T",
            b0_tesla=_b0(d, [1.5], "1.5T and 3T"),
            wb_sar_max_w_per_kg=_num(d, 2, "W/kg", "For 1.5T MR Scanner: Whole-Body SAR ≤ 2 W/kg"),
            **common,
            **wb_common,
        ),
        ConditionSet(
            set_id="wb-3",
            scope="Whole body, 3 T",
            b0_tesla=_b0(d, [3.0], "1.5T and 3T"),
            b1rms_max_uT=_num(d, 1.7, "uT", "For 3T MR Scanner: B1+rms ≤ 1.7 μT"),
            wb_sar_max_w_per_kg=_num(d, 1.2, "W/kg", "do not report B1+rms, Whole-Body SAR ≤ 1.2 W/kg"),
            sar_is_b1rms_fallback=_cat(d, True, "for MR scanners that do not report B1+rms", "eq"),
            **common,
            **wb_common,
        ),
        ConditionSet(
            set_id="head",
            scope="Head examinations",
            b0_tesla=_b0(d, [1.5, 3.0], "1.5T and 3T"),
            transmit_coil=_cat(d, ["head"], "Detachable Head Transmit/Receive Coil ONLY"),
            receive_coil=_cat(d, ["head"], "Detachable Head Transmit/Receive Coil ONLY"),
            operating_mode=_cat(d, "first_level", "Normal Operating Mode or First Level", "le"),
            head_sar_max_w_per_kg=_num(d, 3.2, "W/kg", "Maximum Head SAR 3.2 W/kg"),
            max_active_scan_time_min=FieldValue(
                status=FieldStatus.EXTRACTED,
                value="no limit",
                comparator="any",
                evidence=_ev(d, "There is no limit on scan duration."),
            ),
            scan_region_restrictions=_cat(d, "Head Only", "Scan Regions Head Only", "eq"),
            **common,
        ),
    ]
    imp[d] = _label(d, "Active implant (FDA Tables 4a/4b style)", "MR Conditional", sets)

    d = "IMP-ACTIVE-WB"  # the whole-body sets only
    imp[d] = _label(d, "Active implant, whole-body sets only", "MR Conditional", copy.deepcopy(sets[:2]))
    d = "IMP-ACTIVE-HEAD"
    imp[d] = _label(d, "Active implant, head-only set", "MR Conditional", copy.deepcopy(sets[2:]))

    d = "IMP-LEGACY-3T"  # legacy wording: 3 T or less, 720 G/cm, 2 W/kg for 15 min
    imp[d] = _label(
        d,
        "Legacy passive implant",
        "MR Conditional",
        [
            ConditionSet(
                b0_tesla=_b0(d, [3.0], "Static magnetic field of 3-Tesla or less", le=True),
                max_sfg_t_per_m=_num(d, 7.2, "T/m", "Maximum spatial gradient magnetic field of 720-Gauss/cm or less"),
                operating_mode=_cat(d, "normal", "Normal Operating Mode", "le"),
                wb_sar_max_w_per_kg=_num(d, 2, "W/kg", "whole body averaged SAR of 2-W/kg"),
                max_active_scan_time_min=_num(d, 15, "min", "for 15 minutes of scanning (i.e., per pulse sequence)"),
            )
        ],
    )

    d = "IMP-UNSAFE"
    imp[d] = _label(d, "MR Unsafe device", "MR Unsafe", [])
    d = "IMP-SAFE"
    imp[d] = _label(d, "MR Safe device", "MR Safe", [])
    d = "IMP-NOTSTATED"
    imp[d] = _label(d, "Device without MR status", "Not stated", [])
    d = "IMP-DRAFT"
    imp[d] = imp["IMP-PASSIVE-STAR"].model_copy(deep=True)
    imp[d].doc_id = d
    imp[d].provenance.verification = "draft-unverified"

    d = "IMP-NO-RF"
    imp[d] = _label(
        d,
        "Conditional label without RF condition",
        "MR Conditional",
        [ConditionSet(b0_tesla=_b0(d, [1.5], "1.5 T only"))],
    )

    d = "IMP-B1-ONLY"
    imp[d] = _label(
        d,
        "3 T label with B1+rms limit only",
        "MR Conditional",
        [ConditionSet(b0_tesla=_b0(d, [3.0], "3 T only"), b1rms_max_uT=_num(d, 2.0, "uT", "B1+rms ≤ 2.0 µT"))],
    )

    d = "IMP-SAR-INDEP"
    imp[d] = _label(
        d,
        "3 T label with independent B1+rms and SAR limits",
        "MR Conditional",
        [
            ConditionSet(
                b0_tesla=_b0(d, [3.0], "3 T only"),
                b1rms_max_uT=_num(d, 2.0, "uT", "B1+rms ≤ 2.0 µT"),
                wb_sar_max_w_per_kg=_num(d, 1.0, "W/kg", "and whole-body SAR ≤ 1.0 W/kg"),
            )
        ],
    )
    return imp


def _scanner(b0: float, **kw) -> dict:
    base = {
        "profile_id": f"generic-{b0:g}T",
        "name": f"Generic {b0:g} T closed bore",
        "b0_tesla": b0,
        "scanner_type": "cylindrical",
        "b0_orientation": "horizontal",
        "max_slew_rate_t_per_m_per_s": 200.0,
        "transmit_coils": ["body", "head"],
        "rf_excitation_modes": ["CP"],
    }
    base.update(kw)
    return base


def _seq(name="seq1", b1=None, wb=None, head=None) -> dict:
    return {"name": name, "b1rms_uT": b1, "wb_sar_w_per_kg": wb, "head_sar_w_per_kg": head}


def build_scenarios() -> list[dict]:
    S: list[dict] = []

    def add(
        sid: str,
        implants: list[str],
        scanner: dict,
        protocol: dict,
        expected: str,
        note: str,
        expect_fields: dict | None = None,
    ) -> None:
        S.append(
            {
                "id": sid,
                "implants": implants,
                "scanner": scanner,
                "protocol": protocol,
                "expected": expected,
                "note": note,
                "expect_fields": expect_fields or {},
            }
        )

    NC, CF, NR = "NO_CONFLICT_FOUND", "CONFLICT_FOUND", "NEEDS_REVIEW"

    star = {
        "operating_mode": "normal",
        "transmit_coil": "body",
        "receive_coil": "body",
        "rf_excitation": "CP",
        "nucleus": "1H",
        "sequences": [_seq(wb=1.5, head=2.0)],
        "total_active_scan_time_min": 40,
        "sfg_at_implant_t_per_m": 10,
    }

    def m(base: dict, **kw) -> dict:
        out = copy.deepcopy(base)
        out.update(kw)
        return out

    s15, s3 = _scanner(1.5), _scanner(3.0)
    P = "IMP-PASSIVE-STAR"
    add("star-base-1.5T", [P], s15, star, NC, "all conditions met")
    add("star-base-3T", [P], s3, star, NC, "3 T also permitted")
    add("star-b0-0.55T", [P], _scanner(0.55), star, CF, "field strength not labeled", {P: {"b0_tesla": "FAIL"}})
    add(
        "star-wbsar-over",
        [P],
        s15,
        m(star, sequences=[_seq(wb=2.5, head=2.0)]),
        CF,
        "WB SAR 2.5 > 2",
        {P: {"wb_sar_max_w_per_kg": "FAIL"}},
    )
    add("star-wbsar-equal", [P], s15, m(star, sequences=[_seq(wb=2.0, head=2.0)]), NC, "limit is inclusive")
    add(
        "star-headsar-over",
        [P],
        s15,
        m(star, sequences=[_seq(wb=1.0, head=3.3)]),
        CF,
        "head SAR 3.3 > 3.2",
        {P: {"head_sar_max_w_per_kg": "FAIL"}},
    )
    add(
        "star-headsar-missing",
        [P],
        s15,
        m(star, sequences=[_seq(wb=1.0)]),
        NR,
        "head SAR not entered",
        {P: {"head_sar_max_w_per_kg": "INSUFFICIENT_INFO"}},
    )
    add(
        "star-mode-firstlevel",
        [P],
        s15,
        m(star, operating_mode="first_level"),
        CF,
        "Normal mode required",
        {P: {"operating_mode": "FAIL"}},
    )
    add(
        "star-mode-missing",
        [P],
        s15,
        m(star, operating_mode=None),
        NR,
        "operating mode not entered",
        {P: {"operating_mode": "INSUFFICIENT_INFO"}},
    )
    add(
        "star-time-over",
        [P],
        s15,
        m(star, total_active_scan_time_min=70),
        CF,
        "70 > 60 min",
        {P: {"max_active_scan_time_min": "FAIL"}},
    )
    add("star-time-missing", [P], s15, m(star, total_active_scan_time_min=None), NR, "scan time not entered")
    add(
        "star-sfg-missing",
        [P],
        s15,
        m(star, sfg_at_implant_t_per_m=None),
        NR,
        "spatial field gradient position question -> needs review",
        {P: {"max_sfg_t_per_m": "NEEDS_REVIEW"}},
    )
    add(
        "star-sfg-over",
        [P],
        s15,
        m(star, sfg_at_implant_t_per_m=35),
        CF,
        "35 > 30 T/m",
        {P: {"max_sfg_t_per_m": "FAIL"}},
    )
    add("star-sfg-equal", [P], s15, m(star, sfg_at_implant_t_per_m=30), NC, "inclusive SFG limit")
    add("star-excitation-mc2", [P], s3, m(star, rf_excitation="MC-2"), CF, "CP only", {P: {"rf_excitation": "FAIL"}})
    add("star-excitation-missing", [P], s15, m(star, rf_excitation=None), NR, "excitation not entered")
    add("star-headcoil", [P], s15, m(star, transmit_coil="head"), NC, "no transmit coil restriction")
    add("star-no-sequences", [P], s15, m(star, sequences=[]), NR, "no RF values entered")
    add(
        "star-nucleus-31P",
        [P],
        s15,
        m(star, nucleus="31P"),
        CF,
        "only 1H implied when nucleus not listed",
        {P: {"nucleus": "FAIL"}},
    )
    add(
        "star-two-seq-one-over",
        [P],
        s15,
        m(star, sequences=[_seq("t1", wb=1.0, head=1.0), _seq("dwi", wb=2.2, head=1.0)]),
        CF,
        "one sequence over the WB SAR limit",
    )

    A = "IMP-ACTIVE-WB"
    act3 = {
        "operating_mode": "normal",
        "transmit_coil": "body",
        "receive_coil": "local",
        "rf_excitation": "CP",
        "nucleus": "1H",
        "sequences": [_seq(b1=1.5)],
        "total_active_scan_time_min": 25,
        "sfg_at_implant_t_per_m": 10,
        "device_configuration_confirmed": True,
        "scan_region": "lumbar spine",
    }
    add("act3-base", [A], s3, act3, NC, "B1+rms within limit, SAR fallback not needed")
    add("act3-b1-over", [A], s3, m(act3, sequences=[_seq(b1=1.8)]), CF, "1.8 > 1.7 uT", {A: {"b1rms_max_uT": "FAIL"}})
    add("act3-b1-equal", [A], s3, m(act3, sequences=[_seq(b1=1.7)]), NC, "inclusive B1+rms limit")
    add("act3-fallback-ok", [A], s3, m(act3, sequences=[_seq(wb=1.1)]), NC, "no B1+rms -> SAR fallback 1.1 <= 1.2")
    add(
        "act3-fallback-over",
        [A],
        s3,
        m(act3, sequences=[_seq(wb=1.3)]),
        CF,
        "fallback SAR 1.3 > 1.2",
        {A: {"b1rms_max_uT": "FAIL"}},
    )
    add(
        "act3-no-rf-values",
        [A],
        s3,
        m(act3, sequences=[_seq()]),
        NR,
        "neither B1+rms nor SAR entered",
        {A: {"b1rms_max_uT": "INSUFFICIENT_INFO"}},
    )
    add(
        "act3-both-reported-sar-over",
        [A],
        s3,
        m(act3, sequences=[_seq(b1=1.5, wb=1.5)]),
        CF,
        "conservative: when both are reported both are checked",
        {A: {"wb_sar_max_w_per_kg": "FAIL"}},
    )
    add(
        "act3-slew-over",
        [A],
        _scanner(3.0, max_slew_rate_t_per_m_per_s=220.0),
        act3,
        CF,
        "scanner slew 220 > 200",
        {A: {"max_slew_rate_t_per_m_per_s": "FAIL"}},
    )
    add(
        "act3-slew-limited",
        [A],
        _scanner(3.0, max_slew_rate_t_per_m_per_s=220.0),
        m(act3, slew_rate_limit_t_per_m_per_s=150.0),
        NC,
        "protocol limits slew rate to 150",
    )
    add("act3-slew-unknown", [A], _scanner(3.0, max_slew_rate_t_per_m_per_s=None), act3, NR, "slew rate unknown")
    add(
        "act3-headcoil",
        [A],
        s3,
        m(act3, transmit_coil="head"),
        CF,
        "body transmit coil only",
        {A: {"transmit_coil": "FAIL"}},
    )
    add("act3-receive-any", [A], s3, m(act3, receive_coil="head"), NC, "any receive coil")
    add(
        "act3-config-unconfirmed",
        [A],
        s3,
        m(act3, device_configuration_confirmed=None),
        NR,
        "Stimulation OFF not confirmed",
        {A: {"device_configuration": "INSUFFICIENT_INFO"}},
    )
    add("act3-time-over", [A], s3, m(act3, total_active_scan_time_min=35), CF, "35 > 30 min")
    add(
        "act3-two-sessions-short-wait",
        [A],
        s3,
        m(act3, sessions=2, wait_between_sessions_min=20),
        CF,
        "wait 20 < 30 min",
        {A: {"wait_time_min": "FAIL"}},
    )
    add("act3-two-sessions-wait-ok", [A], s3, m(act3, sessions=2, wait_between_sessions_min=30), NC, "wait 30 min")
    add("act3-two-sessions-wait-missing", [A], s3, m(act3, sessions=2), NR, "wait not entered")
    add("act3-nucleus-31P", [A], s3, m(act3, nucleus="31P"), NC, "31P listed")
    add("act3-nucleus-23Na", [A], s3, m(act3, nucleus="23Na"), CF, "23Na not listed")
    add("act3-open-scanner", [A], _scanner(3.0, scanner_type="open"), act3, CF, "cylindrical required")
    add("act3-vertical-b0", [A], _scanner(3.0, b0_orientation="vertical"), act3, CF, "horizontal B0 required")
    add("act3-orientation-unknown", [A], _scanner(3.0, b0_orientation=None), act3, NR, "orientation not entered")
    act15 = m(act3, sequences=[_seq(wb=1.8)])
    add("act15-base", [A], s15, act15, NC, "1.5 T whole-body SAR 1.8 <= 2")
    add("act15-wbsar-over", [A], s15, m(act15, sequences=[_seq(wb=2.1)]), CF, "2.1 > 2 W/kg")
    add(
        "act15-only-b1",
        [A],
        s15,
        m(act15, sequences=[_seq(b1=2.0)]),
        NR,
        "1.5 T set has an independent SAR limit; B1+rms alone is insufficient",
    )

    H = "IMP-ACTIVE-HEAD"
    head = {
        "operating_mode": "first_level",
        "transmit_coil": "head",
        "receive_coil": "head",
        "rf_excitation": "CP",
        "nucleus": "1H",
        "sequences": [_seq(head=3.0)],
        "total_active_scan_time_min": 50,
        "sfg_at_implant_t_per_m": 5,
        "device_configuration_confirmed": True,
        "scan_region": "head",
    }
    add(
        "head-base",
        [H],
        s3,
        head,
        NR,
        "scan region 'Head Only' always needs human review",
        {H: {"scan_region_restrictions": "NEEDS_REVIEW"}},
    )
    add("head-bodycoil", [H], s3, m(head, transmit_coil="body"), CF, "head T/R coil only")
    add("head-headsar-over", [H], s3, m(head, sequences=[_seq(head=3.3)]), CF, "3.3 > 3.2 W/kg")

    X = "IMP-STAR-ACTIVE"
    add("star-active-head-scan", [X], s3, head, NR, "whole-body sets fail, head set applies -> region review")
    add("star-active-body-scan", [X], s3, act3, NC, "3 T whole-body set applies")
    add("star-active-body-scan-1.5", [X], s15, act15, NC, "1.5 T whole-body set applies")

    L = "IMP-LEGACY-3T"
    leg = {
        "operating_mode": "normal",
        "transmit_coil": "body",
        "rf_excitation": "CP",
        "nucleus": "1H",
        "sequences": [_seq(wb=1.9)],
        "total_active_scan_time_min": 10,
        "sfg_at_implant_t_per_m": 5,
    }
    add("legacy-3T", [L], s3, leg, NC, "3 T or less")
    add("legacy-1.5T", [L], s15, leg, NC, "1.5 T <= 3 T")
    add("legacy-7T", [L], _scanner(7.0), leg, CF, "7 T > 3 T", {L: {"b0_tesla": "FAIL"}})
    add("legacy-wbsar-over", [L], s3, m(leg, sequences=[_seq(wb=2.5)]), CF, "2.5 > 2 W/kg")
    add("legacy-time-over", [L], s3, m(leg, total_active_scan_time_min=20), CF, "20 > 15 min")
    add("legacy-sfg-over", [L], s3, m(leg, sfg_at_implant_t_per_m=8.0), CF, "8 > 7.2 T/m")

    add("unsafe", ["IMP-UNSAFE"], s15, star, CF, "MR Unsafe implant")
    add("safe", ["IMP-SAFE"], s3, star, NC, "MR Safe implant (still requires sign-off)")
    add("status-not-stated", ["IMP-NOTSTATED"], s15, star, NR, "no MR status")
    add("draft-annotation", ["IMP-DRAFT"], s15, star, NR, "not human-verified")
    add("no-rf-condition", ["IMP-NO-RF"], s15, star, NR, "label states no RF condition")
    add(
        "b1-only-label-sar-entered",
        ["IMP-B1-ONLY"],
        s3,
        m(star, sequences=[_seq(wb=1.0)]),
        NR,
        "SAR cannot substitute for B1+rms",
    )
    add("b1-only-label-ok", ["IMP-B1-ONLY"], s3, m(star, sequences=[_seq(b1=1.9)]), NC, "B1+rms 1.9 <= 2.0")
    add(
        "sar-indep-b1-only",
        ["IMP-SAR-INDEP"],
        s3,
        m(star, sequences=[_seq(b1=1.5)]),
        NR,
        "independent SAR limit needs the SAR value",
    )
    add("sar-indep-both-ok", ["IMP-SAR-INDEP"], s3, m(star, sequences=[_seq(b1=1.5, wb=0.9)]), NC, "both within")
    add("sar-indep-sar-over", ["IMP-SAR-INDEP"], s3, m(star, sequences=[_seq(b1=1.5, wb=1.2)]), CF, "SAR 1.2 > 1.0")

    add(
        "multi-pass",
        [P, L],
        s3,
        m(star, sequences=[_seq(wb=1.5, head=2.0)], total_active_scan_time_min=10, sfg_at_implant_t_per_m=5),
        NC,
        "both implants pass",
    )
    add("multi-unsafe", [P, "IMP-UNSAFE"], s15, star, CF, "one MR Unsafe implant")
    add("multi-notstated", [P, "IMP-NOTSTATED"], s15, star, NR, "one implant without status")
    add(
        "multi-legacy-time",
        [P, L],
        s3,
        m(star, total_active_scan_time_min=30, sfg_at_implant_t_per_m=5),
        CF,
        "legacy implant allows only 15 min",
        {L: {"max_active_scan_time_min": "FAIL"}},
    )
    add(
        "multi-star-active-b1-only",
        [P, A],
        s3,
        m(act3, sequences=[_seq(b1=1.5)]),
        NR,
        "passive implant needs SAR values",
    )
    add("no-implants", [], s3, star, NR, "nothing to check")
    return S


def save_scenarios(scenarios: list[dict], path: Path | None = None) -> Path:
    p = path or scenarios_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    implants = {k: json.loads(v.model_dump_json(exclude_defaults=True)) for k, v in build_implants().items()}
    p.write_text(
        json.dumps({"implants": implants, "scenarios": scenarios}, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return p


def run_scenarios(path: Path | None = None) -> dict:
    from ..checker import check

    p = path or scenarios_path()
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        implants = {k: LabelDocument.model_validate(v) for k, v in data["implants"].items()}
        scenarios = data["scenarios"]
    else:
        implants, scenarios = build_implants(), build_scenarios()
    failures, unsafe_pass, field_unsafe_pass, n_fields = [], 0, 0, 0
    by_expected: dict[str, int] = {}
    for sc in scenarios:
        res = check(
            [implants[i] for i in sc["implants"]], ScannerProfile(**sc["scanner"]), ProtocolPlan(**sc["protocol"])
        )
        by_expected[sc["expected"]] = by_expected.get(sc["expected"], 0) + 1
        if res.outcome != sc["expected"]:
            failures.append(f"{sc['id']}: expected {sc['expected']}, got {res.outcome}")
            if res.outcome == "NO_CONFLICT_FOUND":
                unsafe_pass += 1
        for imp_id, fields in sc.get("expect_fields", {}).items():
            ir = next(r for r in res.implants if r.implant_id == imp_id)
            checks = list(ir.implant_checks)
            sel = [s for s in ir.set_results if s.set_id == ir.selected_set_id] or ir.set_results
            for s in sel:
                checks += s.checks
            for fname, want in fields.items():
                n_fields += 1
                got = [c.status for c in checks if c.field == fname]
                if want not in got:
                    failures.append(f"{sc['id']}: {imp_id}.{fname} expected {want}, got {got}")
                    if "PASS" in got and want != "PASS":
                        field_unsafe_pass += 1
    return {
        "n_scenarios": len(scenarios),
        "by_expected": by_expected,
        "n_field_expectations": n_fields,
        "mismatches": len(failures),
        "unsafe_pass": unsafe_pass,
        "field_unsafe_pass": field_unsafe_pass,
        "failures": failures,
    }
