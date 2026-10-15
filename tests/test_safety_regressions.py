"""Regression tests for the safety issues found in the independent QA review (October 2026).

Each test names the defect it guards against. See docs/VALIDATION.md.
"""

from __future__ import annotations

import json
import re
import warnings

import pytest
from pydantic import ValidationError

from mrcond.checker import check
from mrcond.extractors.base import Candidate
from mrcond.ingest import DocumentText, Page
from mrcond.merge import merge_values
from mrcond.schema import (
    DISCLAIMER,
    OUTCOME_TEXT,
    ConditionSet,
    Evidence,
    FieldStatus,
    FieldValue,
    LabelDocument,
    ProtocolPlan,
    Provenance,
    ScannerProfile,
    SequencePlan,
)
from mrcond.units import find_quantities
from mrcond.verifier import verify_candidate

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from fastapi.testclient import TestClient


def ev(q: str = "x") -> list[Evidence]:
    return [Evidence(doc_id="d", page=1, char_start=0, char_end=len(q), quote=q)]


def fv(norm, comparator=None, **kw) -> FieldValue:
    return FieldValue(
        status=FieldStatus.EXTRACTED, value=norm, normalized_value=norm, comparator=comparator, evidence=ev(), **kw
    )


def label(*sets: ConditionSet, status: str = "MR Conditional") -> LabelDocument:
    return LabelDocument(
        doc_id="d",
        mr_status=fv(status, "eq"),
        condition_sets=list(sets),
        provenance=Provenance(extractor="t", verification="human-verified"),
    )


B0_3T = fv([3.0], "in")
SCANNER = ScannerProfile(
    profile_id="s",
    name="s",
    b0_tesla=3.0,
    scanner_type="cylindrical",
    b0_orientation="horizontal",
    max_slew_rate_t_per_m_per_s=200,
)


def protocol(**kw) -> ProtocolPlan:
    base = dict(
        operating_mode="normal",
        transmit_coil="body",
        receive_coil="body",
        rf_excitation="CP",
        nucleus="1H",
        total_active_scan_time_min=10,
        sfg_at_implant_t_per_m=1,
        sequences=[{"name": "s", "b1rms_uT": 9.0, "wb_sar_w_per_kg": 3.9, "head_sar_w_per_kg": 3.1}],
    )
    base.update(kw)
    return ProtocolPlan(**base)


# ---------------------------------------------------------------------------------------------- checker


@pytest.mark.parametrize("field", ["b1rms_max_uT", "wb_sar_max_w_per_kg", "head_sar_max_w_per_kg"])
@pytest.mark.parametrize("bad", [None, "abc", "2 W/kg"])
def test_unreadable_rf_limit_is_never_no_conflict(field, bad):
    """S1: an EXTRACTED RF limit that is not a number used to skip the comparison (No conflict)."""
    cs = ConditionSet(b0_tesla=B0_3T)
    cs.set_field(field, fv(bad, "le"))
    res = check([label(cs)], SCANNER, protocol())
    assert res.outcome == "NEEDS_REVIEW"
    checks = [c for c in res.implants[0].set_results[0].checks if c.field == field]
    assert checks and checks[0].status == "NEEDS_REVIEW"


def test_unreadable_wait_time_is_not_ignored():
    cs = ConditionSet(b0_tesla=B0_3T, operating_mode=fv("normal", "le"), wait_time_min=fv("thirty", "ge"))
    assert check([label(cs)], SCANNER, protocol(sequences=[])).outcome == "NEEDS_REVIEW"


@pytest.mark.parametrize(
    "payload",
    [
        {"protocol": {"sequences": [{"name": "s", "wb_sar_w_per_kg": float("nan")}]}},
        {"protocol": {"sequences": [{"name": "s", "b1rms_uT": -1.0}]}},
        {"protocol": {"sequences": [{"name": "s", "b1rms_uT": float("-inf")}]}},
        {"protocol": {"total_active_scan_time_min": -10}},
        {"protocol": {"sfg_at_implant_t_per_m": -1}},
        {"protocol": {"sessions": 0}},
        {"protocol": {"wait_between_sessions_min": -5}},
        {"protocol": {"slew_rate_limit_t_per_m_per_s": 0}},
        {"scanner": {"b0_tesla": 0}},
        {"scanner": {"b0_tesla": -3}},
        {"scanner": {"b0_tesla": float("nan")}},
        {"scanner": {"b0_tesla": 300}},
        {"scanner": {"max_slew_rate_t_per_m_per_s": float("inf")}},
    ],
)
def test_invalid_inputs_are_rejected(payload):
    """S2: NaN, negative and absurd inputs used to produce 'No conflict found' (e.g. NaN SAR, -10 min)."""
    scanner = {"profile_id": "s", "name": "s", "b0_tesla": 3.0, **payload.get("scanner", {})}
    proto = {"operating_mode": "normal", **payload.get("protocol", {})}
    with pytest.raises(ValidationError):
        ScannerProfile(**scanner)
        ProtocolPlan(**proto)


def test_api_rejects_nan_and_negative_values(data_copy):
    """S2 through HTTP: the live API returned 'No conflict found' for NaN SAR, -10 min and -1 T/m."""
    from mrcond.api import create_app

    client = TestClient(create_app())
    body = (
        '{"implant_ids": ["demo:IMP-LEGACY-3T"], "scanner": {"profile_id": "x", "name": "x", "b0_tesla": 3.0, '
        '"max_slew_rate_t_per_m_per_s": 200}, "protocol": {"operating_mode": "normal", "transmit_coil": "body", '
        '"rf_excitation": "CP", "nucleus": "1H", "sequences": [{"name": "s", "wb_sar_w_per_kg": NaN}], '
        '"total_active_scan_time_min": -10, "sfg_at_implant_t_per_m": -1}}'
    )
    r = client.post("/api/check", content=body, headers={"Content-Type": "application/json"})
    assert r.status_code == 422
    text = r.text
    assert "wb_sar_w_per_kg" in text and "total_active_scan_time_min" in text and "sfg_at_implant_t_per_m" in text


def test_checker_defends_against_unvalidated_inputs():
    """Callers that bypass validation (model_construct) still never get 'No conflict' from invalid values."""
    cs = ConditionSet(
        b0_tesla=B0_3T,
        operating_mode=fv("normal", "le"),
        b1rms_max_uT=fv(1.0, "le"),
        max_active_scan_time_min=fv(10.0, "le"),
        max_sfg_t_per_m=fv(3.0, "le"),
    )
    for seq, extra in [
        ({"b1rms_uT": float("nan")}, {}),
        ({"b1rms_uT": -1.0}, {}),
        ({"b1rms_uT": 0.5}, {"total_active_scan_time_min": -5.0}),
        ({"b1rms_uT": 0.5}, {"sfg_at_implant_t_per_m": -3.0}),
        ({"b1rms_uT": 0.5}, {"sfg_at_implant_t_per_m": float("nan")}),
    ]:
        seq_values = {"name": "s", "b1rms_uT": None, "wb_sar_w_per_kg": None, "head_sar_w_per_kg": None, **seq}
        p = ProtocolPlan.model_construct(
            **{**protocol().model_dump(), **extra, "sequences": [SequencePlan.model_construct(**seq_values)]}
        )
        assert check([label(cs)], SCANNER, p).outcome != "NO_CONFLICT_FOUND", (seq, extra)
    bad_scanner = ScannerProfile.model_construct(**{**SCANNER.model_dump(), "b0_tesla": -3.0})
    le = ConditionSet(b0_tesla=fv([3.0], "le"), operating_mode=fv("normal", "le"))
    assert check([label(le)], bad_scanner, protocol(sequences=[])).outcome != "NO_CONFLICT_FOUND"


def test_device_configuration_not_set_is_a_conflict():
    """S4: the user stating that the device is NOT configured as required is a conflict, not missing input."""
    cs = ConditionSet(b0_tesla=B0_3T, operating_mode=fv("normal", "le"), device_configuration=fv("Therapy OFF", "eq"))
    p = protocol(sequences=[])
    assert (
        check([label(cs)], SCANNER, p.model_copy(update={"device_configuration_confirmed": False})).outcome
        == "CONFLICT_FOUND"
    )
    assert check([label(cs)], SCANNER, p).outcome == "NEEDS_REVIEW"
    assert (
        check([label(cs)], SCANNER, p.model_copy(update={"device_configuration_confirmed": True})).outcome
        == "NO_CONFLICT_FOUND"
    )


def test_mr_safe_with_conditions_needs_review():
    """S5: a record labeled MR Safe that also holds condition sets is contradictory."""
    cs = ConditionSet(b0_tesla=fv([1.5], "in"))
    assert check([label(cs, status="MR Safe")], SCANNER, protocol()).outcome == "NEEDS_REVIEW"
    assert check([label(status="MR Safe")], SCANNER, protocol()).outcome == "NO_CONFLICT_FOUND"


def test_exam_values_are_never_assumed():
    """Nucleus, scanner type and B0 orientation must be entered; defaults are not assumed any more."""
    cs = ConditionSet(b0_tesla=B0_3T, operating_mode=fv("normal", "le"), scanner_type=fv("cylindrical", "eq"))
    p = protocol(sequences=[])
    assert ProtocolPlan().nucleus is None and ScannerProfile(profile_id="s", name="s", b0_tesla=3).scanner_type is None
    assert check([label(cs)], SCANNER, p).outcome == "NO_CONFLICT_FOUND"
    assert check([label(cs)], SCANNER, p.model_copy(update={"nucleus": None})).outcome == "NEEDS_REVIEW"
    assert check([label(cs)], SCANNER.model_copy(update={"scanner_type": None}), p).outcome == "NEEDS_REVIEW"


def test_boundaries_after_unit_conversion():
    """P5: values exactly at a limit pass whatever unit they came from; just above conflicts."""
    from mrcond.units import to_canonical

    limit = to_canonical(3000, "G/cm")[0]  # "3,000 G/cm" in the labeling -> 30 T/m
    cs = ConditionSet(
        b0_tesla=B0_3T,
        operating_mode=fv("normal", "le"),
        max_sfg_t_per_m=fv(limit, "le"),
        wb_sar_max_w_per_kg=fv(0.3, "le"),
    )
    ok = protocol(sfg_at_implant_t_per_m=3000 / 100, sequences=[{"name": "s", "wb_sar_w_per_kg": 0.1 + 0.2}])
    assert check([label(cs)], SCANNER, ok).outcome == "NO_CONFLICT_FOUND"
    over = protocol(sfg_at_implant_t_per_m=3000.4 / 100, sequences=[{"name": "s", "wb_sar_w_per_kg": 0.3}])
    assert check([label(cs)], SCANNER, over).outcome == "CONFLICT_FOUND"
    assert [q.canonical_value for q in find_quantities("1,5 T")] == [1.5]
    assert [q.canonical_value for q in find_quantities("3,000 gauss/cm")] == [30.0]


def test_outcome_texts_and_disclaimer_never_claim_safety():
    """P1: only the three outcomes; no 'safe', 'cleared' or 'approved' in outcome texts or the disclaimer."""
    words = re.compile(r"\bsafe(ly)?\b|\bcleared\b|\bapproved\b", re.IGNORECASE)
    assert set(OUTCOME_TEXT.values()) == {
        "No conflict found with entered conditions — requires MR Safety Officer/Expert sign-off",
        "Conflict found",
        "Insufficient information / needs review",
    }
    for t in [*OUTCOME_TEXT.values(), DISCLAIMER]:
        assert not words.search(t.replace("MR Safety", "")), t


# ---------------------------------------------------------------------------------------------- merge


def test_numeric_merge_never_drops_an_unreadable_limit():
    """S3: [no limit, unreadable] merged to 'no limit' while [unreadable, no limit] stayed a limit."""
    any_ = FieldValue(
        status=FieldStatus.EXTRACTED, value="no limit", normalized_value=None, comparator="any", evidence=ev()
    )
    bad = fv(None, "le")
    for order in ([any_, bad], [bad, any_]):
        assert merge_values("wb_sar_max_w_per_kg", order).status == FieldStatus.NEEDS_REVIEW
    good = fv(2.0, "le")
    assert merge_values("wb_sar_max_w_per_kg", [any_, good]).normalized_value == 2.0  # "no limit" never relaxes
    assert (
        merge_values("operating_mode", [fv("first_level", "le"), fv("turbo", "le")]).status == FieldStatus.NEEDS_REVIEW
    )


# ---------------------------------------------------------------------------------------------- verifier


def _verify(field, text, quote, **kw):
    doc = DocumentText("d", [Page(1, text)])
    c = Candidate(field, FieldValue(status=FieldStatus.EXTRACTED, evidence=ev(quote), **kw))
    verify_candidate(c, doc)
    return c.accepted


def test_verifier_rejects_numbers_that_belong_to_another_quantity():
    """V1: a SAR of 3 W/kg was 'supported' by the 3 of '3T' in 'For 3T: Whole-Body SAR <= 2 W/kg'."""
    t = "For 3T MR Scanner: Whole-Body SAR ≤ 2 W/kg"
    assert not _verify("wb_sar_max_w_per_kg", t, t, value=3, unit="W/kg", normalized_value=3.0, comparator="le")
    assert _verify("wb_sar_max_w_per_kg", t, t, value=2, unit="W/kg", normalized_value=2.0, comparator="le")
    t2 = "Scan for 15 minutes with a whole body SAR of 2 W/kg"
    assert not _verify("max_active_scan_time_min", t2, t2, value=2, unit="min", normalized_value=2.0, comparator="le")
    t3 = "Whole-Body SAR (W/kg) 2.0"  # table cell: unit in the header, bare number
    assert _verify("wb_sar_max_w_per_kg", t3, t3, value=2.0, unit="W/kg", normalized_value=2.0, comparator="le")


def test_verifier_requires_no_limit_wording_for_unlimited():
    """V3: 'no limit' claims were accepted from any quote mentioning the quantity."""
    t = "B1+rms ≤ 2.0 µT"
    assert not _verify("b1rms_max_uT", t, t, value="no limit", normalized_value=None, comparator="any")
    t2 = "There is no limit on scan duration."
    assert _verify("max_active_scan_time_min", t2, t2, value="no limit", normalized_value=None, comparator="any")


def test_verifier_any_coil_needs_unrestricted_wording():
    """V4: 'any' coil was supported by 'all other coils are prohibited' or 'do not use any local coil'."""
    for t in [
        "Use the body coil only; all other transmit coils are prohibited.",
        "Do not use any local transmit coil.",
    ]:
        assert not _verify("transmit_coil", t, t, value="any", normalized_value=["any"], comparator="in")
    for t in [
        "There are no Transmit Coil restrictions",
        "RF Receive Coil Type Any",
        "Any receive-only coil may be used",
    ]:
        field = "receive_coil" if "eceive" in t else "transmit_coil"
        assert _verify(field, t, t, value="any", normalized_value=["any"], comparator="in"), t


def test_verifier_rejects_negated_status_and_effective_slew():
    """V5/V6: 'is not MR Safe' supported MR Safe; an effective (vector) slew rate passed as per-axis."""
    t = "The device is not MR Safe."
    assert not _verify("mr_status", t, t, value="MR Safe", normalized_value="MR Safe", comparator="eq")
    t2 = "Maximum gradient slew rate 200 T/m/s (effective)"
    assert not _verify(
        "max_slew_rate_t_per_m_per_s", t2, t2, value=200, unit="T/m/s", normalized_value=200.0, comparator="le"
    )


# ---------------------------------------------------------------------------------------------- API edits

DOC = "fda-guidance-2023-t4a"


@pytest.fixture()
def client(data_copy):
    from mrcond.api import create_app

    return TestClient(create_app())


@pytest.mark.parametrize(
    "decision, fragment",
    [
        ({"field": "wb_sar_max_w_per_kg", "value": "nan", "unit": "W/kg"}, "finite"),
        ({"field": "wb_sar_max_w_per_kg", "value": 40, "unit": "W/kg"}, "plausible"),
        ({"field": "wb_sar_max_w_per_kg", "value": 2, "unit": "W/kg", "comparator": "ge"}, "comparator"),
        ({"field": "wb_sar_max_w_per_kg", "value": 2, "unit": "µT"}, "not a sar unit"),
        ({"field": "transmit_coil", "normalized_value": ["anything"]}, "options"),
        ({"field": "operating_mode", "normalized_value": ["normal", "first_level"]}, "exactly one"),
        ({"field": "b0_tesla", "value": [float("inf")]}, "finite"),
    ],
)
def test_reviewer_edits_are_validated(client, decision, fragment):
    """Human edits become 'human-verified' values the checker relies on: implausible input is refused."""
    d = {"set_id": "cs1", "action": "edit", "quote": "Whole-Body SAR ≤ 2 W/kg", "page": 30, **decision}
    r = client.post(
        f"/api/documents/{DOC}/verify",
        content=json.dumps({"reviewer": "QA", "decisions": [d]}, allow_nan=True),
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 422, r.text
    assert fragment in r.text.lower()


def test_reviewer_bool_text_is_parsed(client):
    """'false' sent as text used to become True (bool('false'))."""
    d = {
        "set_id": "cs2",
        "field": "sar_is_b1rms_fallback",
        "action": "edit",
        "value": "false",
        "quote": "for MR scanners that",
        "page": 30,
    }
    r = client.post(f"/api/documents/{DOC}/verify", json={"reviewer": "QA", "decisions": [d]})
    assert r.status_code == 200, r.text
    assert r.json()["condition_sets"][1]["sar_is_b1rms_fallback"]["normalized_value"] is False


# ---------------------------------------------------------------------------------------------- replays


def test_recordings_match_the_current_prompt():
    """Recorded LLM outputs are only valid for the prompt they were recorded with (prompt + context hash).

    A change to the prompt, retrieval/chunking or text normalisation silently invalidates every replay; this
    test fails instead. Documents whose text is not available locally (manufacturer PDFs before
    `mrcond fetch`) are skipped.
    """
    from mrcond.benchmark.runner import load_track_r, load_track_s
    from mrcond.corpus import repo_root
    from mrcond.extractors.backends import prompt_hash
    from mrcond.extractors.llm import LLMExtractor

    docs = {d.doc_id: d for d, _, _ in load_track_r()[0]}
    docs.update({d.doc_id: d for d, _, _ in load_track_s("test", 30)})

    class _Unused:
        name = "unused"

    checked = 0
    for path in sorted((repo_root() / "data" / "replay").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec["doc_id"] not in docs:
                continue
            run = int(rec.get("run", 0))
            ex = LLMExtractor(_Unused(), run=run, temperature=0.0 if run == 0 else 0.7, seed=run)
            assert prompt_hash(ex.request(docs[rec["doc_id"]])) == rec["prompt_hash"], (path.name, rec["doc_id"])
            checked += 1
    assert checked >= 30  # at least the Track S documents are always available


def test_verifier_checks_the_quantity_of_categorical_values():
    """A transmit-coil sentence must not support an RF-excitation value ('Any transmit coil' is not 'any excitation')."""
    t = "Any transmit coil may be used."
    assert not _verify("rf_excitation", t, t, value="any", normalized_value=["any"], comparator="in")
    t2 = "RF Excitation: Circularly Polarized (CP)"
    assert _verify("rf_excitation", t2, t2, value="CP", normalized_value=["CP"], comparator="in")


def test_slew_rate_in_gauss_per_cm_per_second_is_not_a_gradient():
    """U1: '5,000 G/cm/s' was read as a 50 T/m spatial field gradient."""
    qs = find_quantities("Maximum slew rate 5,000 G/cm/s")
    assert [(q.dimension, q.canonical_value) for q in qs] == [("slew", 50.0)]
    assert [(q.dimension, q.canonical_value) for q in find_quantities("720 G/cm")] == [("gradient", 7.2)]


def test_llm_coil_normalisation_does_not_read_all_as_any():
    from mrcond.extractors.llm import _norm_categorical

    assert _norm_categorical("transmit_coil", "Body coil only; all other coils are prohibited") == ["body"]
    assert _norm_categorical("receive_coil", "small local coil") == ["local"]
    assert _norm_categorical("receive_coil", "Any receive-only coil may be used") == ["any"]


# ---------------------------------------------------------------------------------------------- metrics


def test_more_permissive_mr_status_is_a_less_conservative_error():
    from mrcond.benchmark.metrics import compare_docs

    unsafe = LabelDocument(doc_id="d", mr_status=fv("MR Unsafe", "eq"))
    for pred_status in ("MR Conditional", "MR Safe"):
        pred = label(ConditionSet(b0_tesla=fv([1.5], "in")), status=pred_status)
        assert any(f == "mr_status" for _, f, _ in compare_docs(unsafe, pred).lc)
    flagged = LabelDocument(doc_id="d", mr_status=FieldValue(status=FieldStatus.NEEDS_REVIEW, evidence=ev()))
    assert compare_docs(unsafe, flagged).lc == []  # abstention is never less conservative
    cond = label(ConditionSet(b0_tesla=fv([1.5], "in")))
    assert not any(f == "mr_status" for _, f, _ in compare_docs(cond, unsafe).lc)  # more conservative


def test_field_strength_inside_a_gold_range_is_checked_against_its_limits():
    """Gold '3 T or less, SAR <= 1'; a predicted 1.5 T set with SAR 2 used to escape the less-conservative count."""
    from mrcond.benchmark.metrics import compare_docs

    gold = label(ConditionSet(b0_tesla=fv([3.0], "le"), wb_sar_max_w_per_kg=fv(1.0, "le")))
    pred = label(
        ConditionSet(set_id="a", b0_tesla=fv([1.5], "in"), wb_sar_max_w_per_kg=fv(2.0, "le")),
        ConditionSet(set_id="b", b0_tesla=fv([3.0], "in"), wb_sar_max_w_per_kg=fv(1.0, "le")),
    )
    lc = compare_docs(gold, pred).lc
    assert ("1.5", "wb_sar_max_w_per_kg", "higher limit (2 > 1)") in lc


def test_readme_results_match_the_benchmark_json():
    """The README tables are generated from reports/benchmark_results.json and must match it exactly."""
    from mrcond.benchmark.report import README_END, README_START, readme_results
    from mrcond.corpus import repo_root

    results = json.loads((repo_root() / "reports" / "benchmark_results.json").read_text(encoding="utf-8"))
    readme = (repo_root() / "README.md").read_text(encoding="utf-8")
    block = readme[readme.index(README_START) : readme.index(README_END) + len(README_END)]
    assert block == readme_results(results)
    assert "{{RESULTS}}" not in readme
