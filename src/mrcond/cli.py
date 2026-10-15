"""Command line interface: ``mrcond <command>``.

Commands
--------
fetch          download Track R documents listed in the manifest (data/raw, data/text; git-ignored)
extract        run an extractor on a corpus document or a local file and print the LabelDocument JSON
bench          run the benchmark (tracks S and/or R) and write reports/quick/benchmark_results.{json,md}
synth          (re)generate the Track S synthetic corpus
scenarios      run the checker scenario suite (or regenerate it with --write)
gudid          openFDA/GUDID gap analysis (CSV + figure under reports/)
export-schema  write JSON Schemas to schemas/
serve          start the REST API (and the built web UI, if present)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _llm_backend_factory(args) -> tuple[object | None, str | None]:
    if not args.llm_backend:
        return None, None
    from .extractors.backends import OllamaBackend, OpenAICompatBackend, RecordingBackend, ReplayBackend

    tag = args.llm_tag or args.llm_model or "replay"

    def factory(run: int):
        if args.llm_backend == "replay":
            path = Path(args.llm_replay)
            if run > 0:
                alt = path.with_name(path.stem + f".run{run}" + path.suffix)
                if alt.exists():
                    path = alt
            return ReplayBackend(path)
        if args.llm_backend == "ollama":
            be = OllamaBackend(args.llm_model, args.llm_host)
        else:
            be = OpenAICompatBackend(args.llm_model, args.llm_base_url, args.llm_api_key, args.llm_schema_mode)
        if args.llm_record:
            path = Path(args.llm_record)
            if run > 0:
                path = path.with_name(path.stem + f".run{run}" + path.suffix)
            return RecordingBackend(be, path, tag)
        return be

    return factory, tag


def _add_llm_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("LLM backend")
    g.add_argument("--llm-backend", choices=["openai", "ollama", "replay"], help="enable the LLM extractor")
    g.add_argument("--llm-model", default="local-model", help="model name as served (e.g. qwen2.5:7b-instruct)")
    g.add_argument("--llm-tag", help="label for results/recordings (default: model name)")
    g.add_argument("--llm-base-url", default="http://localhost:8080/v1", help="OpenAI-compatible base URL")
    g.add_argument("--llm-host", default="http://localhost:11434", help="Ollama host")
    g.add_argument("--llm-api-key", default=None)
    g.add_argument("--llm-schema-mode", default="json_schema", choices=["json_schema", "json_object", "none"])
    g.add_argument("--llm-replay", help="replay JSONL file (backend=replay)")
    g.add_argument("--llm-record", help="record live outputs to this JSONL file (replayable later)")


def cmd_fetch(args) -> int:
    from .corpus import fetch_entry, load_manifest

    man = load_manifest()
    ok = True
    for e in man.entries:
        if args.only and e.doc_id not in args.only:
            continue
        try:
            st = fetch_entry(e, force=args.force)
            flag = "" if st["sha256_matches_manifest"] else "  [SHA-256 differs from manifest: document changed?]"
            print(f"{e.doc_id:42s} ok  pages={st['pages']} chars={st['chars']}{flag}")
        except Exception as exc:  # network errors etc.
            ok = False
            print(f"{e.doc_id:42s} FAILED: {exc}", file=sys.stderr)
    return 0 if ok else 1


def cmd_extract(args) -> int:
    from .corpus import load_manifest, load_text
    from .ingest import load_document
    from .pipeline import get_extractor, run_extraction

    path = Path(args.target)
    if path.exists():
        doc = load_document(path)
    else:
        doc = (
            load_text(load_manifest().get(args.target))
            if args.target in {e.doc_id for e in load_manifest().entries}
            else None
        )
        if doc is None:
            print(f"not found (fetch first?): {args.target}", file=sys.stderr)
            return 1
    kwargs = {}
    if args.extractor.startswith("llm"):
        factory, tag = _llm_backend_factory(args)
        if factory is None:
            print("--llm-backend is required for the LLM extractor", file=sys.stderr)
            return 2
        kwargs = {"backend": factory(0), "model_tag": tag}
    ex, verify = get_extractor(args.extractor, **kwargs)
    label, out = run_extraction(doc, ex, verify=verify)
    print(label.model_dump_json(indent=2, exclude_defaults=True))
    return 0


def cmd_bench(args) -> int:
    from .benchmark.runner import results_markdown, run_benchmark

    factory, tag = _llm_backend_factory(args)
    res = run_benchmark(
        args.tracks,
        args.extractors,
        split=args.split,
        llm_backend_factory=factory,
        llm_tag=tag,
        llm_runs=args.llm_runs,
        limit=args.limit,
        llm_limit=args.llm_limit,
        out_dir=args.out,
    )
    print(results_markdown(res))
    return 0


def cmd_synth(args) -> int:
    from .benchmark.synth import generate_corpus, save_corpus

    docs = generate_corpus(args.n, args.seed)
    save_corpus(docs, args.out)
    print(f"wrote {len(docs)} synthetic documents to {args.out}")
    return 0


def cmd_scenarios(args) -> int:
    from .benchmark.scenarios import build_scenarios, run_scenarios, save_scenarios, scenarios_path

    if args.write:
        sc = build_scenarios()
        save_scenarios(sc)
        print(f"wrote {len(sc)} scenarios to {scenarios_path()}")
    rep = run_scenarios()
    print(json.dumps({k: v for k, v in rep.items() if k != "failures"}, indent=1))
    for f in rep["failures"]:
        print("MISMATCH:", f)
    return 0 if rep["unsafe_pass"] == 0 and not rep["failures"] else 1


def cmd_gudid(args) -> int:
    from .gudid import run_gap_analysis

    out = run_gap_analysis(args.out, offline=args.offline, by_year=not args.no_years)
    print(json.dumps(out, indent=1))
    return 0


def cmd_export_schema(args) -> int:
    from .schema import export_json_schemas

    for p in export_json_schemas(args.out):
        print(p)
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    uvicorn.run("mrcond.api:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="mrcond", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("fetch", help="download Track R documents")
    p.add_argument("--only", nargs="*", help="doc ids")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("extract", help="extract MR conditions from a document")
    p.add_argument("target", help="corpus doc_id or path to a PDF/HTML/TXT file")
    p.add_argument("--extractor", default="rule+verifier", choices=["rule", "rule+verifier", "llm", "llm+verifier"])
    _add_llm_args(p)
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("bench", help="run the benchmark")
    p.add_argument("--tracks", nargs="+", default=["S", "R"], choices=["S", "R"])
    p.add_argument("--extractors", nargs="+", default=["rule", "rule+verifier"])
    p.add_argument("--split", default="test", choices=["dev", "test", "all"])
    p.add_argument("--limit", type=int)
    p.add_argument("--llm-limit", type=int, help="evaluate the LLM on the first N documents of each track")
    p.add_argument("--llm-runs", type=int, default=1, help=">1 adds sampled runs for run-to-run consistency")
    p.add_argument(
        "--out",
        default="reports/quick",
        help="output directory (the full report in reports/ is written by scripts/benchmark_report.py)",
    )
    _add_llm_args(p)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("synth", help="generate the Track S corpus")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=20261015)
    p.add_argument("--out", default="data/benchmark/track_s")
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("scenarios", help="run the checker scenario suite")
    p.add_argument("--write", action="store_true", help="regenerate data/benchmark/checker_scenarios.json")
    p.set_defaults(func=cmd_scenarios)

    p = sub.add_parser("gudid", help="openFDA GUDID MRI-safety gap analysis")
    p.add_argument("--out", default="reports")
    p.add_argument("--offline", action="store_true", help="use the 15 Oct 2026 snapshot only")
    p.add_argument("--no-years", action="store_true", help="skip the per-publish-year queries")
    p.set_defaults(func=cmd_gudid)

    p = sub.add_parser("export-schema", help="write JSON Schemas")
    p.add_argument("--out", default="schemas")
    p.set_defaults(func=cmd_export_schema)

    p = sub.add_parser("serve", help="start the REST API + web UI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(func=cmd_serve)

    args = ap.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
