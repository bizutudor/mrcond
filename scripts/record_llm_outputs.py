"""Run a live local LLM over benchmark documents and record its raw outputs (resumable).

The recordings (data/replay/<tag>.jsonl, plus .runN.jsonl for sampled runs)
contain only the model output (with the short quotes the model copied) and a
prompt hash - never the prompt or the document text - and let anyone re-score
the run without the model:

    mrcond bench --tracks S R --extractors llm llm+verifier \
        --llm-backend replay --llm-replay data/replay/<tag>.jsonl --llm-tag <tag> --llm-runs 3

Example (llama.cpp server or llama-cpp-python server on port 8088):

    python scripts/record_llm_outputs.py --base-url http://127.0.0.1:8088/v1 \
        --model qwen2.5-1.5b-instruct-q4_k_m --schema-mode json_object \
        --tag qwen2.5-1.5b-instruct-q4_k_m --track-s 30 --track-r --runs 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mrcond.benchmark.runner import load_track_r, load_track_s  # noqa: E402
from mrcond.extractors.backends import OllamaBackend, OpenAICompatBackend, RecordingBackend  # noqa: E402
from mrcond.extractors.llm import LLMExtractor  # noqa: E402


def recorded(path: Path) -> set[tuple[str, int]]:
    out = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out.add((r["doc_id"], int(r.get("run", 0))))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="openai", choices=["openai", "ollama"])
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--model", required=True)
    ap.add_argument("--schema-mode", default="json_schema")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--track-s", type=int, default=0, help="number of Track S test documents")
    ap.add_argument("--track-r", action="store_true")
    ap.add_argument("--runs", type=int, nargs="+", default=[0], help="0 = temperature 0; k>0 = T 0.7, seed k")
    ap.add_argument("--only", nargs="*", help="restrict to these doc ids")
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "replay"))
    args = ap.parse_args()

    if args.backend == "ollama":
        inner = OllamaBackend(args.model, args.host)
    else:
        inner = OpenAICompatBackend(args.model, args.base_url, schema_mode=args.schema_mode)

    docs = []
    if args.track_r:
        items, missing = load_track_r()
        if missing:
            print("Track R documents without local text (run `mrcond fetch`):", missing)
        docs += [d for d, _, _ in items]
    if args.track_s:
        docs += [d for d, _, _ in load_track_s("test", args.track_s)]
    if args.only:
        docs = [d for d in docs if d.doc_id in args.only]

    for run in args.runs:
        path = Path(args.out_dir) / (f"{args.tag}.jsonl" if run == 0 else f"{args.tag}.run{run}.jsonl")
        done = recorded(path)
        be = RecordingBackend(inner, path, args.tag)
        ex = LLMExtractor(be, args.tag, run=run, temperature=0.0 if run == 0 else 0.7, seed=run)
        for i, doc in enumerate(docs, 1):
            if (doc.doc_id, run) in done:
                continue
            t0 = time.time()
            try:
                out = ex.extract(doc)
                n = len(out.candidates)
                status = "ok"
            except Exception as exc:  # keep going; failures are reported
                n, status = 0, f"ERROR {exc!r}"
            print(
                f"[run {run}] {i}/{len(docs)} {doc.doc_id}: {status} candidates={n} {time.time() - t0:.0f}s", flush=True
            )


if __name__ == "__main__":
    main()
