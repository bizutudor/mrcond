"""Benchmark runner for Track S (synthetic) and Track R (real public labeling)."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ..corpus import data_dir, load_annotation, load_manifest, load_text
from ..extractors.backends import LLMBackend, ReplayMiss
from ..extractors.base import Extractor
from ..ingest import DocumentText
from ..pipeline import run_extraction
from ..schema import LabelDocument
from .metrics import DocScore, aggregate, compare_docs, consistency
from .synth import generate_corpus, load_corpus, save_corpus

__all__ = ["load_track_s", "load_track_r", "evaluate", "run_benchmark", "results_markdown", "TRACK_S_DIR"]

TRACK_S_SEED = 20261015
TRACK_S_N = 200


def TRACK_S_DIR() -> Path:  # noqa: N802 - function so MRCOND_DATA is honoured at call time
    return data_dir() / "benchmark" / "track_s"


def load_track_s(split: str = "test", limit: int | None = None) -> list[tuple[DocumentText, LabelDocument, dict]]:
    d = TRACK_S_DIR()
    if not (d / "index.json").exists():
        save_corpus(generate_corpus(TRACK_S_N, TRACK_S_SEED), d)
    items = load_corpus(d)
    half = len(items) // 2
    if split == "dev":
        items = items[:half]
    elif split == "test":
        items = items[half:]
    return items[:limit] if limit else items


def load_track_r(limit: int | None = None) -> tuple[list[tuple[DocumentText, LabelDocument, dict]], list[str]]:
    """Documents with a (draft or verified) annotation and available text. Returns (items, skipped ids)."""
    man = load_manifest()
    items, skipped = [], []
    for e in man.entries:
        gold = load_annotation(e.doc_id)
        if gold is None:
            continue
        doc = load_text(e)
        if doc is None:
            skipped.append(e.doc_id)
            continue
        items.append(
            (
                doc,
                gold,
                {"doc_id": e.doc_id, "verification": gold.provenance.verification, "device_type": e.device_type},
            )
        )
    return (items[:limit] if limit else items), skipped


@dataclass
class EvalResult:
    extractor: str
    metrics: dict
    scores: list[DocScore]
    predictions: dict[str, LabelDocument]
    seconds: float
    skipped: list[str]


def evaluate(
    items: list[tuple[DocumentText, LabelDocument, dict]], extractor: Extractor, verify: bool, label: str
) -> EvalResult:
    t0 = time.time()
    scores, preds, skipped = [], {}, []
    for doc, gold, _meta in items:
        try:
            pred, _ = run_extraction(doc, extractor, verify=verify)
        except ReplayMiss:
            skipped.append(doc.doc_id)
            continue
        preds[doc.doc_id] = pred
        scores.append(compare_docs(gold, pred, doc))
    m = aggregate(scores)
    return EvalResult(label, m, scores, preds, round(time.time() - t0, 2), skipped)


def run_benchmark(
    tracks: list[str],
    extractors: list[str],
    *,
    split: str = "test",
    llm_backend_factory: Callable[[int], LLMBackend] | None = None,
    llm_tag: str | None = None,
    llm_runs: int = 1,
    limit: int | None = None,
    llm_limit: int | None = None,
    out_dir: str | Path | None = None,
) -> dict:
    """Run extractors on tracks and write reports/benchmark_results.{json,md}."""
    from ..extractors.llm import LLMExtractor
    from ..extractors.rule import RuleExtractor

    results: dict = {"generated_at": datetime.now(UTC).isoformat(timespec="seconds"), "tracks": {}}
    for track in tracks:
        if track == "S":
            items = load_track_s(split, limit)
            tinfo = {
                "description": f"Track S (synthetic, exact ground truth), split={split}",
                "n_docs": len(items),
                "results": {},
            }
        else:
            items, missing = load_track_r(limit)
            tinfo = {
                "description": "Track R (real public labeling) - on DRAFT annotations pending human verification",
                "n_docs": len(items),
                "missing_text": missing,
                "results": {},
            }
        for name in extractors:
            base, _, suffix = name.partition("+")
            verify = suffix == "verifier"
            if base == "rule":
                res = evaluate(items, RuleExtractor(), verify, name)
                tinfo["results"][name] = {**res.metrics, "seconds": res.seconds}
            elif base == "llm":
                if llm_backend_factory is None:
                    continue
                sub = items[:llm_limit] if llm_limit else items
                runs_preds = []
                res = None
                for run in range(llm_runs):
                    ex = LLMExtractor(
                        llm_backend_factory(run), llm_tag, run=run, temperature=0.0 if run == 0 else 0.7, seed=run
                    )
                    r = evaluate(sub, ex, verify, name)
                    runs_preds.append(r.predictions)
                    if run == 0:
                        res = r
                assert res is not None
                entry = {**res.metrics, "seconds": res.seconds, "skipped_no_recording": res.skipped, "model": llm_tag}
                if llm_runs > 1:
                    cons = []
                    for doc_id in runs_preds[0]:
                        docs = [rp[doc_id] for rp in runs_preds if doc_id in rp]
                        c = consistency(docs)
                        if c is not None:
                            cons.append(c)
                    entry["run_to_run_consistency"] = round(sum(cons) / len(cons), 4) if cons else None
                    entry["consistency_runs"] = llm_runs
                    entry["consistency_docs"] = len(cons)
                tinfo["results"][name] = entry
        results["tracks"][track] = tinfo
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "benchmark_results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
        (out / "benchmark_results.md").write_text(results_markdown(results), encoding="utf-8")
    return results


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.1f}%"


def results_markdown(results: dict) -> str:
    lines = ["# MRCond benchmark results", "", f"Generated: {results['generated_at']}", ""]
    for track, t in results["tracks"].items():
        lines += [f"## Track {track}", "", t["description"], "", f"Documents: {t['n_docs']}", ""]
        if t.get("missing_text"):
            lines += [f"Not fetched locally (skipped): {', '.join(t['missing_text'])}", ""]
        lines += [
            "| Extractor | Docs | Precision | Recall | F1 | Status acc. | Unsupported values | "
            "Less-conservative errors (rate) | Abstention | Selective acc. | Consistency |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for name, m in t["results"].items():
            cons = m.get("run_to_run_consistency")
            cons_txt = (
                _pct(cons) if cons is not None else ("100% (deterministic)" if not name.startswith("llm") else "—")
            )
            lines.append(
                f"| {name} | {m['n_docs']} | {_pct(m['precision'])} | {_pct(m['recall'])} | {_pct(m['f1'])} | "
                f"{_pct(m['status_accuracy'])} | {m['unsupported_values']}/{m['n_values']} "
                f"({_pct(m['unsupported_rate'])}) | {m['less_conservative_errors']} "
                f"({_pct(m['less_conservative_rate'])}) | {_pct(m['abstention_rate'])} | "
                f"{_pct(m['selective_accuracy'])} | {cons_txt} |"
            )
        lines.append("")
        for name, m in t["results"].items():
            if m["less_conservative_errors"]:
                lines += [
                    f"Less-conservative errors for `{name}` by kind: "
                    + ", ".join(f"{k}: {v}" for k, v in sorted(m["less_conservative_by_kind"].items())),
                    "",
                ]
    return "\n".join(lines) + "\n"
