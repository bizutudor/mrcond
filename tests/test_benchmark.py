"""Benchmark metrics, the Track S generator and the Track R manifest/annotations."""

import json
import re

import pytest

from mrcond.benchmark.metrics import aggregate, compare_docs, consistency, flatten
from mrcond.benchmark.runner import load_track_s, run_benchmark
from mrcond.benchmark.synth import generate_corpus, generate_doc
from mrcond.corpus import annotation_path, load_annotation, load_manifest
from mrcond.ingest import DocumentText, Page
from mrcond.schema import ConditionSet, Evidence, FieldStatus, FieldValue, LabelDocument, MRStatus
from mrcond.textnorm import find_all


def label(sets, status="MR Conditional"):
    st = FieldValue(status=FieldStatus.EXTRACTED, value=status, normalized_value=status)
    return LabelDocument(doc_id="d", mr_status=st, condition_sets=sets)


def num(v, comparator="le", quote=None):
    ev = [Evidence(doc_id="d", page=1, char_start=0, char_end=len(quote), quote=quote)] if quote else []
    return FieldValue(status=FieldStatus.EXTRACTED, value=v, normalized_value=v, comparator=comparator, evidence=ev)


def b0(vals, comparator="in"):
    return FieldValue(status=FieldStatus.EXTRACTED, value=vals, normalized_value=vals, comparator=comparator)


def test_perfect_prediction():
    g = label([ConditionSet(b0_tesla=b0([1.5, 3.0]), wb_sar_max_w_per_kg=num(2.0))])
    sc = compare_docs(g, g.model_copy(deep=True))
    assert (sc.tp, sc.fp, sc.fn) == (4, 0, 0) and sc.lc == [] and sc.all_correct


def test_less_conservative_errors_are_detected():
    g = label(
        [
            ConditionSet(
                b0_tesla=b0([1.5]),
                wb_sar_max_w_per_kg=num(2.0),
                wait_time_min=num(30, "ge"),
                max_slew_rate_t_per_m_per_s=num(200.0),
            )
        ]
    )
    p = label([ConditionSet(b0_tesla=b0([1.5, 3.0]), wb_sar_max_w_per_kg=num(4.0), wait_time_min=num(10, "ge"))])
    sc = compare_docs(g, p)
    kinds = sorted(r.split(" (")[0] for _, _, r in sc.lc)
    assert kinds == ["extra field strength", "higher limit", "missed constraint", "shorter wait"]


def test_more_conservative_errors_are_not_less_conservative():
    g = label([ConditionSet(b0_tesla=b0([1.5, 3.0]), wb_sar_max_w_per_kg=num(2.0))])
    p = label([ConditionSet(b0_tesla=b0([1.5]), wb_sar_max_w_per_kg=num(1.0), head_sar_max_w_per_kg=num(3.2))])
    sc = compare_docs(g, p)
    assert sc.lc == [] and sc.fp > 0 and sc.fn > 0


def test_field_strength_ranges_are_aligned():
    le3 = b0([3.0], "le")  # "3 Tesla or less"
    g = label([ConditionSet(b0_tesla=le3, wb_sar_max_w_per_kg=num(2.0))])
    only3 = label([ConditionSet(b0_tesla=b0([3.0]), wb_sar_max_w_per_kg=num(2.0))])
    sc = compare_docs(g, only3)
    assert sc.lc == [] and sc.tp == 1  # narrower reading: conservative; SAR still compared and correct
    g2 = label([ConditionSet(b0_tesla=b0([1.5, 3.0]), wb_sar_max_w_per_kg=num(2.0))])
    wide = label([ConditionSet(b0_tesla=le3, wb_sar_max_w_per_kg=num(4.0))])
    kinds = sorted(r.split(" (")[0] for _, _, r in compare_docs(g2, wide).lc)
    assert kinds == ["extra field strength", "higher limit", "higher limit"]


def test_nucleus_1h_and_abstentions():
    nuc = FieldValue(status=FieldStatus.EXTRACTED, normalized_value=["1H"], comparator="in")
    g = label([ConditionSet(b0_tesla=b0([3.0]), nucleus=nuc, wb_sar_max_w_per_kg=num(2.0))])
    flagged = FieldValue(status=FieldStatus.NEEDS_REVIEW, review_reasons=["x"])
    p = label([ConditionSet(b0_tesla=b0([3.0]), wb_sar_max_w_per_kg=flagged)])
    sc = compare_docs(g, p)
    assert sc.lc == []  # 1H is the default nucleus; abstention is never "less conservative"
    assert sc.abstain == 1
    m = aggregate([sc])
    assert m["abstention_rate"] > 0 and m["selective_accuracy"] == 0.5


def test_unsupported_values_and_consistency():
    doc = DocumentText("d", [Page(1, "Whole-Body SAR ≤ 2 W/kg")])
    g = label([ConditionSet(b0_tesla=b0([1.5]), wb_sar_max_w_per_kg=num(2.0, quote="Whole-Body SAR ≤ 2 W/kg"))])
    p = label([ConditionSet(b0_tesla=b0([1.5]), wb_sar_max_w_per_kg=num(2.0, quote="Whole-Body SAR ≤ 9 W/kg"))])
    sc = compare_docs(g, p, doc)
    assert sc.n_unsupported >= 1
    assert consistency([g, g.model_copy(deep=True)]) == 1.0
    assert consistency([g, p]) == 1.0  # same values; evidence differences do not count
    q = label([ConditionSet(b0_tesla=b0([1.5]), wb_sar_max_w_per_kg=num(1.0))])
    assert consistency([g, q]) < 1.0


def test_flatten_merges_alternative_sets_conservatively():
    g = label(
        [
            ConditionSet(set_id="a", b0_tesla=b0([1.5]), wb_sar_max_w_per_kg=num(2.0)),
            ConditionSet(set_id="b", b0_tesla=b0([1.5]), wb_sar_max_w_per_kg=num(1.0)),
        ]
    )
    assert flatten(g)[("1.5", "wb_sar_max_w_per_kg")].fv.normalized_value == 1.0


def test_generator_is_deterministic_and_gold_quotes_are_verbatim():
    assert generate_doc(7, 3).doc.full_text == generate_doc(7, 3).doc.full_text
    for sd in generate_corpus(60, seed=11):
        for cs in sd.gold.condition_sets:
            for _, fv in cs.iter_fields():
                for ev in fv.evidence:
                    assert len(ev.quote.split()) <= 25
                    assert sd.doc.page_text(ev.page)[ev.char_start : ev.char_end] == ev.quote


def test_committed_track_s_matches_generator():
    items = load_track_s("all")
    assert len(items) == 200
    regenerated = generate_doc(20261015, 150)
    doc, gold, _ = items[150]
    assert doc.full_text == regenerated.doc.full_text
    assert gold.model_dump(exclude={"provenance"}) == regenerated.gold.model_dump(exclude={"provenance"})


def test_manifest_integrity():
    man = load_manifest()
    ids = [e.doc_id for e in man.entries]
    assert 15 <= len(ids) <= 25 and len(set(ids)) == len(ids)
    for e in man.entries:
        assert e.url.startswith("https://")
        assert re.fullmatch(r"[0-9a-f]{64}", e.sha256 or "")
        assert e.retrieved == "2026-10-15"
    assert len({e.manufacturer for e in man.entries}) >= 10


def test_track_r_draft_annotations():
    man = load_manifest()
    annotated = [e for e in man.entries if annotation_path(e.doc_id).exists()]
    assert len(annotated) >= 10
    for e in annotated:
        lab = load_annotation(e.doc_id, prefer_verified=False)
        assert lab.provenance.verification == "draft-unverified"
        assert lab.provenance.annotator == "assistant-draft"
        assert lab.status_value().value == (e.status_expected or lab.status_value().value)
        for cs in lab.condition_sets:
            for _, fv in cs.iter_fields():
                for ev in fv.evidence:
                    assert len(ev.quote.split()) <= 25, (e.doc_id, ev.quote)
        if e.redistributable:  # public-domain text is in the repository: quotes must be verbatim
            from mrcond.corpus import load_text

            doc = load_text(e)
            for cs in lab.condition_sets:
                for _, fv in cs.iter_fields():
                    for ev in fv.evidence:
                        assert find_all(doc.page_text(ev.page), ev.quote)


def test_runner_writes_reports(data_copy, tmp_path):
    res = run_benchmark(["S"], ["rule"], split="test", limit=5, out_dir=tmp_path)
    m = res["tracks"]["S"]["results"]["rule"]
    assert m["n_docs"] == 5 and 0 <= m["f1"] <= 1
    assert (tmp_path / "benchmark_results.md").read_text().startswith("# MRCond benchmark results")
    assert json.loads((tmp_path / "benchmark_results.json").read_text())["tracks"]["S"]


@pytest.mark.network
def test_fetch_public_fda_guidance(data_copy):
    from mrcond.corpus import fetch_entry

    st = fetch_entry(load_manifest().get("fda-guidance-2023-t3"))
    assert st["sha256_matches_manifest"] and st["pages"] == [29]
    assert MRStatus.MR_CONDITIONAL.value  # import used
