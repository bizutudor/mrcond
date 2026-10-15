"""Verifier (quote-or-abstain, parse check, plausibility) and conservative merge."""

from mrcond.assemble import assemble
from mrcond.extractors.base import Candidate, ExtractionOutput, extracted
from mrcond.ingest import DocumentText, Page
from mrcond.merge import merge_values
from mrcond.schema import Evidence, FieldStatus, FieldValue
from mrcond.verifier import verify_candidate, verify_output

TEXT = (
    "MRI Safety Information\n"
    "The Kestrel Stent is MR Conditional.\n"
    "Static magnetic field of 1.5 T and 3 T\n"
    "Maximum Spatial Field Gradient 25 T/m (2500 gauss/cm)\n"
    "Whole-Body SAR ≤ 2 W/kg\n"
    "Maximum Head SAR 3.2 W/kg\n"
    "B1+rms ≤ 1.7 μT\n"
    "Scan for up to 30 minutes.\n"
)
DOC = DocumentText("doc", [Page(1, "cover page"), Page(2, TEXT)])


def ev(quote: str, page: int = 2, start: int = 0) -> Evidence:
    return Evidence(doc_id="doc", page=page, char_start=start, char_end=start + len(quote), quote=quote)


def cand(field, value, unit, norm, quote, comparator="le", page=2) -> Candidate:
    return Candidate(field, extracted(value, unit, norm, None, comparator, [ev(quote, page)]))


def test_accepts_supported_value_and_reanchors_offsets():
    c = verify_candidate(cand("wb_sar_max_w_per_kg", 2, "W/kg", 2.0, "Whole-Body SAR ≤ 2 W/kg", page=1), DOC)
    assert c.accepted, c.reasons
    e = c.fv.evidence[0]
    assert e.page == 2 and TEXT[e.char_start : e.char_end] == "Whole-Body SAR ≤ 2 W/kg" and e.verified


def test_rejects_hallucinated_quote():
    c = verify_candidate(cand("wb_sar_max_w_per_kg", 4, "W/kg", 4.0, "Whole-Body SAR ≤ 4 W/kg"), DOC)
    assert c.accepted is False
    assert c.reasons == ["evidence: quote not found verbatim in source"]


def test_rejects_value_not_in_quote():
    c = verify_candidate(cand("wb_sar_max_w_per_kg", 3, "W/kg", 3.0, "Whole-Body SAR ≤ 2 W/kg"), DOC)
    assert c.accepted is False and c.reasons[0].startswith("parse:")


def test_rejects_wrong_unit_dimension():
    c = verify_candidate(cand("b1rms_max_uT", 1.7, "W/kg", 1.7, "B1+rms ≤ 1.7 μT"), DOC)
    assert c.accepted is False and c.reasons[0].startswith("unit:")


def test_rejects_unconverted_gauss_value():
    # an extractor that reads 2500 gauss/cm as 2500 T/m must abstain
    c = verify_candidate(
        cand("max_sfg_t_per_m", 2500, "T/m", 2500.0, "Maximum Spatial Field Gradient 25 T/m (2500 gauss/cm)"), DOC
    )
    assert c.accepted is False
    ok = verify_candidate(
        cand("max_sfg_t_per_m", 2500, "gauss/cm", 25.0, "Maximum Spatial Field Gradient 25 T/m (2500 gauss/cm)"), DOC
    )
    assert ok.accepted, ok.reasons


def test_rejects_misattributed_context():
    c = verify_candidate(cand("wb_sar_max_w_per_kg", 3.2, "W/kg", 3.2, "Maximum Head SAR 3.2 W/kg"), DOC)
    assert c.accepted is False and c.reasons[0].startswith("context:")


def test_rejects_implausible_value():
    doc = DocumentText("d", [Page(1, "Whole-Body SAR ≤ 20 W/kg")])
    c = verify_candidate(
        Candidate(
            "wb_sar_max_w_per_kg",
            extracted(
                20,
                "W/kg",
                20.0,
                None,
                "le",
                [Evidence(doc_id="d", page=1, char_start=0, char_end=24, quote="Whole-Body SAR ≤ 20 W/kg")],
            ),
        ),
        doc,
    )
    assert c.accepted is False and any(r.startswith("plausibility") for r in c.reasons)


def test_rejects_missing_evidence_and_bad_status_quote():
    c = verify_candidate(Candidate("wb_sar_max_w_per_kg", extracted(2, "W/kg", 2.0, None, "le", [])), DOC)
    assert c.accepted is False and c.reasons == ["evidence: no supporting quote"]
    s = verify_candidate(
        Candidate(
            "mr_status",
            extracted("MR Unsafe", None, "MR Unsafe", None, "eq", [ev("The Kestrel Stent is MR Conditional.")]),
        ),
        DOC,
    )
    assert s.accepted is False


def test_long_quotes_are_trimmed_to_25_words():
    long_text = " ".join(["filler"] * 30) + " whole-body SAR of 2 W/kg " + " ".join(["more"] * 30)
    doc = DocumentText("d", [Page(1, long_text)])
    c = verify_candidate(
        Candidate(
            "wb_sar_max_w_per_kg",
            extracted(
                2,
                "W/kg",
                2.0,
                None,
                "le",
                [Evidence(doc_id="d", page=1, char_start=0, char_end=len(long_text), quote=long_text)],
            ),
        ),
        doc,
    )
    assert c.accepted, c.reasons
    assert len(c.fv.evidence[0].quote.split()) <= 25 and "2 W/kg" in c.fv.evidence[0].quote


def fv(value, unit="W/kg", comparator="le", status=FieldStatus.EXTRACTED) -> FieldValue:
    return FieldValue(status=status, value=value, unit=unit, normalized_value=value, comparator=comparator)


def test_merge_most_conservative_numeric():
    assert merge_values("wb_sar_max_w_per_kg", [fv(2.0), fv(1.2), fv(4.0)]).normalized_value == 1.2
    assert merge_values("wait_time_min", [fv(5.0, "min", "ge"), fv(30.0, "min", "ge")]).normalized_value == 30.0
    assert merge_values("artifact_extent_mm", [fv(5.0, "mm", "eq"), fv(20.0, "mm", "eq")]).normalized_value == 20.0
    unlimited = FieldValue(status=FieldStatus.EXTRACTED, value="no limit", comparator="any")
    assert merge_values("max_active_scan_time_min", [unlimited, fv(15.0, "min")]).normalized_value == 15.0


def test_merge_categorical_and_b0():
    any_coil = FieldValue(status=FieldStatus.EXTRACTED, normalized_value=["any"], comparator="in")
    body = FieldValue(status=FieldStatus.EXTRACTED, normalized_value=["body"], comparator="in")
    head = FieldValue(status=FieldStatus.EXTRACTED, normalized_value=["head"], comparator="in")
    assert merge_values("transmit_coil", [any_coil, body]).normalized_value == ["body"]
    conflict = merge_values("transmit_coil", [body, head])
    assert conflict.status == FieldStatus.NEEDS_REVIEW
    normal = FieldValue(status=FieldStatus.EXTRACTED, normalized_value="normal", comparator="le")
    first = FieldValue(status=FieldStatus.EXTRACTED, normalized_value="first_level", comparator="le")
    assert merge_values("operating_mode", [first, normal]).normalized_value == "normal"
    both = FieldValue(status=FieldStatus.EXTRACTED, normalized_value=[1.5, 3.0], comparator="in")
    le3 = FieldValue(status=FieldStatus.EXTRACTED, normalized_value=[3.0], comparator="le")
    only15 = FieldValue(status=FieldStatus.EXTRACTED, normalized_value=[1.5], comparator="in")
    assert merge_values("b0_tesla", [both, only15]).normalized_value == [1.5]
    assert merge_values("b0_tesla", [both, le3]).normalized_value == [1.5, 3.0]
    yes = FieldValue(status=FieldStatus.EXTRACTED, normalized_value=True)
    no = FieldValue(status=FieldStatus.EXTRACTED, normalized_value=False)
    assert merge_values("sar_is_b1rms_fallback", [yes, no]).normalized_value is False


def test_merge_scan_region_keeps_restriction():
    free = FieldValue(status=FieldStatus.EXTRACTED, normalized_value="unrestricted")
    head = FieldValue(status=FieldStatus.EXTRACTED, normalized_value="Head Only")
    assert merge_values("scan_region_restrictions", [free, head]).normalized_value == "Head Only"


def test_rejected_candidate_turns_field_into_needs_review():
    out = ExtractionOutput("doc", "test")
    out.candidates = [
        cand("b0_tesla", [1.5, 3.0], "T", [1.5, 3.0], "Static magnetic field of 1.5 T and 3 T", "in"),
        cand("wb_sar_max_w_per_kg", 2, "W/kg", 2.0, "Whole-Body SAR ≤ 2 W/kg"),
        cand("wb_sar_max_w_per_kg", 0.5, "W/kg", 0.5, "Whole-Body SAR ≤ 0.5 W/kg"),  # hallucinated
    ]
    rep = verify_output(out, DOC)
    assert rep.n_accepted == 2 and rep.n_rejected == 1
    label = assemble(out)
    wb = label.condition_sets[0].wb_sar_max_w_per_kg
    assert wb.status == FieldStatus.NEEDS_REVIEW
    assert wb.normalized_value == 2.0  # verified suggestion kept for the reviewer
    assert any("not found verbatim" in r for r in wb.review_reasons)


def test_scoped_candidates_create_per_field_strength_sets():
    out = ExtractionOutput("doc", "test")
    b0 = cand("b0_tesla", [1.5, 3.0], "T", [1.5, 3.0], "Static magnetic field of 1.5 T and 3 T", "in")
    wb15 = cand("wb_sar_max_w_per_kg", 2, "W/kg", 2.0, "Whole-Body SAR ≤ 2 W/kg")
    wb15.scope_b0 = (1.5,)
    b1_3 = cand("b1rms_max_uT", 1.7, "μT", 1.7, "B1+rms ≤ 1.7 μT")
    b1_3.scope_b0 = (3.0,)
    sfg = cand("max_sfg_t_per_m", 25, "T/m", 25.0, "Maximum Spatial Field Gradient 25 T/m")
    out.candidates = [b0, wb15, b1_3, sfg]
    verify_output(out, DOC)
    label = assemble(out)
    assert [cs.b0_tesla.normalized_value for cs in label.condition_sets] == [[1.5], [3.0]]
    s15, s3 = label.condition_sets
    assert s15.wb_sar_max_w_per_kg.normalized_value == 2.0 and s15.b1rms_max_uT.status == FieldStatus.NOT_STATED
    assert s3.b1rms_max_uT.normalized_value == 1.7 and s3.wb_sar_max_w_per_kg.status == FieldStatus.NOT_STATED
    assert s15.max_sfg_t_per_m.normalized_value == 25.0 == s3.max_sfg_t_per_m.normalized_value


def test_scope_check_rejects_values_quoted_from_another_field_strength():
    text = (
        "Static Magnetic Field Strength (B0) 1.5T and 3T\n"
        "RF Conditions For 1.5T MR Scanner: Whole-Body SAR ≤ 2 W/kg\n"
        "For 3T MR Scanner : B1+rms ≤ 1.7 μT\n"
    )
    doc = DocumentText("doc", [Page(2, text)])

    def run(scope):
        out = ExtractionOutput("doc", "test")
        sar = cand("wb_sar_max_w_per_kg", 2, "W/kg", 2.0, "Whole-Body SAR ≤ 2 W/kg")
        sar.scope_b0 = scope
        b0 = cand("b0_tesla", [1.5, 3.0], "T", [1.5, 3.0], "Static Magnetic Field Strength (B0) 1.5T and 3T", "in")
        out.candidates = [b0, sar]
        verify_output(out, doc)
        return sar

    unscoped = run(None)  # the 1.5 T limit applied to both field strengths: less conservative at 3 T
    assert unscoped.accepted is False and unscoped.reasons[0].startswith("scope:")
    assert run((1.5,)).accepted
    assert run((3.0,)).accepted is False
