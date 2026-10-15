"""Checker: scenario suite (zero unsafe PASS) and targeted semantics."""

import json
import re

from mrcond.benchmark.scenarios import build_implants, build_scenarios, run_scenarios
from mrcond.checker import check
from mrcond.schema import OUTCOME_TEXT, ProtocolPlan, ScannerProfile

SAFE_CLAIM = re.compile(r"\b(safe to scan|is safe|are safe|scan is safe|cleared for scanning)\b", re.IGNORECASE)


def test_scenario_suite_has_zero_unsafe_pass():
    rep = run_scenarios()
    assert rep["n_scenarios"] >= 50
    assert rep["unsafe_pass"] == 0
    assert rep["field_unsafe_pass"] == 0
    assert rep["failures"] == [], rep["failures"]
    assert set(rep["by_expected"]) == {"NO_CONFLICT_FOUND", "CONFLICT_FOUND", "NEEDS_REVIEW"}


def test_committed_scenarios_match_generator():
    from mrcond.benchmark.scenarios import scenarios_path

    committed = json.loads(scenarios_path().read_text())["scenarios"]
    assert committed == build_scenarios()


def test_outputs_never_claim_safety():
    imps = build_implants()
    for sc in build_scenarios():
        res = check([imps[i] for i in sc["implants"]], ScannerProfile(**sc["scanner"]), ProtocolPlan(**sc["protocol"]))
        text = res.model_dump_json()
        assert not SAFE_CLAIM.search(text.replace("does not determine that a scan is safe", "")), sc["id"]
    for t in OUTCOME_TEXT.values():
        assert not SAFE_CLAIM.search(t)
    assert "sign-off" in OUTCOME_TEXT["NO_CONFLICT_FOUND"]


def _scanner(b0=3.0, **kw):
    # scanner type and orientation are entered explicitly: the checker never assumes them
    kw = {"scanner_type": "cylindrical", "b0_orientation": "horizontal", **kw}
    return ScannerProfile(profile_id="t", name="t", b0_tesla=b0, max_slew_rate_t_per_m_per_s=200.0, **kw)


def test_sfg_needs_review_without_value_at_implant():
    imp = build_implants()["IMP-PASSIVE-STAR"]
    p = ProtocolPlan(
        operating_mode="normal",
        rf_excitation="CP",
        transmit_coil="body",
        nucleus="1H",
        sequences=[{"name": "s", "wb_sar_w_per_kg": 1.0, "head_sar_w_per_kg": 1.0}],
        total_active_scan_time_min=10,
    )
    res = check([imp], _scanner(), p)
    assert res.outcome == "NEEDS_REVIEW"
    sfg = [c for s in res.implants[0].set_results for c in s.checks if c.field == "max_sfg_t_per_m"]
    assert sfg and sfg[0].status == "NEEDS_REVIEW"
    p.sfg_at_implant_t_per_m = 5
    assert check([imp], _scanner(), p).outcome == "NO_CONFLICT_FOUND"


def test_b1rms_fallback_semantics():
    imp = build_implants()["IMP-ACTIVE-WB"]
    base = dict(
        operating_mode="normal",
        rf_excitation="CP",
        transmit_coil="body",
        receive_coil="local",
        nucleus="1H",
        total_active_scan_time_min=10,
        sfg_at_implant_t_per_m=5,
        device_configuration_confirmed=True,
    )
    ok_sar_only = ProtocolPlan(**base, sequences=[{"name": "s", "wb_sar_w_per_kg": 1.0}])
    assert check([imp], _scanner(), ok_sar_only).outcome == "NO_CONFLICT_FOUND"
    over = ProtocolPlan(**base, sequences=[{"name": "s", "wb_sar_w_per_kg": 1.3}])
    assert check([imp], _scanner(), over).outcome == "CONFLICT_FOUND"
    nothing = ProtocolPlan(**base, sequences=[{"name": "s"}])
    assert check([imp], _scanner(), nothing).outcome == "NEEDS_REVIEW"


def test_multi_implant_most_conservative_and_empty():
    imps = build_implants()
    p = ProtocolPlan(
        operating_mode="normal",
        rf_excitation="CP",
        transmit_coil="body",
        nucleus="1H",
        sequences=[{"name": "s", "wb_sar_w_per_kg": 1.0, "head_sar_w_per_kg": 1.0}],
        total_active_scan_time_min=10,
        sfg_at_implant_t_per_m=5,
    )
    assert check([imps["IMP-PASSIVE-STAR"], imps["IMP-LEGACY-3T"]], _scanner(), p).outcome == "NO_CONFLICT_FOUND"
    assert check([imps["IMP-PASSIVE-STAR"], imps["IMP-UNSAFE"]], _scanner(), p).outcome == "CONFLICT_FOUND"
    assert check([imps["IMP-PASSIVE-STAR"], imps["IMP-DRAFT"]], _scanner(), p).outcome == "NEEDS_REVIEW"
    assert check([], _scanner(), p).outcome == "NEEDS_REVIEW"
