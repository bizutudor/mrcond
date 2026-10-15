"""Rule-based extractor: FDA guidance examples (public domain), targeted traps, Track S."""

import pytest

from mrcond.benchmark.metrics import aggregate, compare_docs
from mrcond.benchmark.runner import load_track_s
from mrcond.corpus import load_annotation, load_manifest, load_text
from mrcond.extractors.rule import RuleExtractor
from mrcond.ingest import DocumentText, Page
from mrcond.pipeline import run_extraction
from mrcond.schema import FieldStatus


def extract(text: str):
    label, _ = run_extraction(DocumentText("t", [Page(1, text)]), RuleExtractor(), verify=True)
    return label


@pytest.mark.parametrize("doc_id", ["fda-guidance-2023-t3", "fda-guidance-2023-t4a", "fda-guidance-2023-t4b"])
def test_fda_examples_match_draft_annotations(doc_id):
    doc = load_text(load_manifest().get(doc_id))
    gold = load_annotation(doc_id)
    pred, _ = run_extraction(doc, RuleExtractor(), verify=True)
    sc = compare_docs(gold, pred, doc)
    assert sc.status_correct
    assert sc.lc == [], sc.lc  # never more permissive than the labeling
    assert sc.n_unsupported == 0
    assert sc.tp >= 0.85 * (sc.tp + sc.fn)


def test_fda_table_4a_sets_and_fallback():
    doc = load_text(load_manifest().get("fda-guidance-2023-t4a"))
    label, _ = run_extraction(doc, RuleExtractor(), verify=True)
    s15, s3 = label.condition_sets
    assert s15.b0_tesla.normalized_value == [1.5] and s3.b0_tesla.normalized_value == [3.0]
    assert s15.wb_sar_max_w_per_kg.normalized_value == 2.0
    assert s3.b1rms_max_uT.normalized_value == 1.7
    assert s3.wb_sar_max_w_per_kg.normalized_value == 1.2
    assert s3.sar_is_b1rms_fallback.normalized_value is True
    assert s15.sar_is_b1rms_fallback.status == FieldStatus.NOT_STATED
    assert s3.max_sfg_t_per_m.normalized_value == 25.0
    for cs in label.condition_sets:
        for _, fv in cs.iter_fields():
            for ev in fv.evidence:
                assert len(ev.quote.split()) <= 25
                assert doc.page_text(ev.page)[ev.char_start : ev.char_end] == ev.quote


def test_legacy_wording_and_unit_conversion():
    label = extract(
        "Non-clinical testing has demonstrated that the device is MR Conditional. A patient with this device can be "
        "safely scanned in an MR system meeting the following conditions:\n"
        "• Static magnetic field of 3-Tesla or less\n"
        "• Maximum spatial gradient magnetic field of 720-Gauss/cm or less\n"
        "• Maximum MR system reported, whole body averaged specific absorption rate (SAR) of 2-W/kg for 15 minutes of "
        "scanning (i.e., per pulse sequence)\n"
        "Under the scan conditions defined above, the device is expected to produce a maximum temperature rise of "
        "less than 2.1 °C after 60 minutes of continuous scanning.\n"
    )
    cs = label.condition_sets[0]
    assert label.status_value().value == "MR Conditional"
    assert cs.b0_tesla.comparator == "le" and cs.b0_tesla.normalized_value == [3.0]
    assert cs.max_sfg_t_per_m.normalized_value == 7.2
    assert cs.wb_sar_max_w_per_kg.normalized_value == 2.0
    # the 15 min condition is kept, the 60 min heating-test duration is ignored
    assert cs.max_active_scan_time_min.normalized_value == 15.0


def test_negated_and_test_context_field_strengths_are_ignored():
    label = extract(
        "The device is MR Conditional at 1.5 T.\n"
        "Scanning at 7 T has not been evaluated.\n"
        "In non-clinical testing, the image artifact extends approximately 5 mm from the device when imaged in a "
        "3 T MR system.\n"
        "Whole-Body SAR ≤ 2 W/kg\n"
    )
    assert [cs.b0_tesla.normalized_value for cs in label.condition_sets] == [[1.5]]
    assert label.condition_sets[0].artifact_extent_mm.normalized_value == 5.0


def test_mr_unsafe_and_not_evaluated():
    assert (
        extract("The Foo pump is MR Unsafe. Do not bring it into the scanner room.").status_value().value == "MR Unsafe"
    )
    lab = extract("The Foo clip has not been evaluated for safety in the MR environment.")
    assert lab.status_value().value == "Not stated" and lab.condition_sets == []


def test_column_table_assigns_values_per_field_strength():
    label = extract(
        "The Bar system is MR Conditional.\n"
        "Static magnetic field strength: 1.5 T or 3 T\n"
        "Parameter 1.5 T 3 T\n"
        "B1+rms ≤ 2.8 μT ≤ 2.0 μT\n"
        "Whole-Body SAR ≤ 2 W/kg not specified\n"
    )
    by_b0 = {cs.b0_tesla.normalized_value[0]: cs for cs in label.condition_sets}
    assert by_b0[1.5].b1rms_max_uT.normalized_value == 2.8
    assert by_b0[3.0].b1rms_max_uT.normalized_value == 2.0
    assert by_b0[1.5].wb_sar_max_w_per_kg.normalized_value == 2.0
    assert by_b0[3.0].wb_sar_max_w_per_kg.status == FieldStatus.NOT_STATED


def test_track_s_dev_split_quality():
    """Regression guard on the synthetic dev split (templates are shared with the test split)."""
    items = load_track_s("dev")
    scores = []
    for doc, gold, _ in items:
        pred, _ = run_extraction(doc, RuleExtractor(), verify=True)
        scores.append(compare_docs(gold, pred, doc))
    m = aggregate(scores)
    assert m["n_docs"] == 100
    assert m["f1"] >= 0.95
    assert m["unsupported_values"] == 0
    assert m["status_accuracy"] == 1.0
    assert m["less_conservative_errors"] <= 2
