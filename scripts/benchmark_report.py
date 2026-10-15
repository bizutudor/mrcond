"""Run the complete benchmark from recorded LLM outputs and refresh the reports and the README tables.

    python scripts/benchmark_report.py            # writes reports/benchmark_results.{json,md}, updates README.md

No model is needed: LLM results are re-scored from data/replay/*.jsonl (see scripts/record_llm_outputs.py).
Track R uses the locally fetched manufacturer documents (`mrcond fetch`); documents without local text are listed
as missing and skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mrcond.benchmark.report import README_END, README_START, readme_results, report_markdown, run_full_benchmark  # noqa: E402

SETUP = {
    "models": {
        "qwen2.5-1.5b-instruct-q4_k_m": "Qwen2.5-1.5B-Instruct, GGUF Q4_K_M (Qwen/Qwen2.5-1.5B-Instruct-GGUF on Hugging Face)",
        "qwen2.5-3b-instruct-q4_k_m": "Qwen2.5-3B-Instruct, GGUF Q4_K_M (Qwen/Qwen2.5-3B-Instruct-GGUF on Hugging Face)",
    },
    "runtime": "llama-cpp-python 0.3.36 OpenAI-compatible server, n_ctx 8192, 1 generation thread, 2 batch threads",
    "hardware": "2 vCPU (Intel Xeon 2.1 GHz, AVX2), 8 GB RAM, no GPU; about 1.5-2 min per document",
    "decoding": "run 0: temperature 0 (greedy), seed 0; runs 1-2 (consistency, 1.5B only, first 10 Track R documents): "
    "temperature 0.7, seeds 1 and 2; max_tokens 1000; JSON schema passed as response_format json_object",
    "context": "keyword-retrieved statements, at most 6000 characters, page markers kept",
    "llm_documents": "Track R: all annotated documents; Track S: the first 30 test documents (synth-20261015-0100..0129)",
    "notes": "The 3B server was OOM-killed once (after 26 of 51 documents) and restarted with identical settings "
    "(plus use_mlock false); the remaining documents were recorded afterwards.",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay-dir", default=str(ROOT / "data" / "replay"))
    ap.add_argument("--out", default=str(ROOT / "reports"))
    ap.add_argument("--readme", default=str(ROOT / "README.md"))
    ap.add_argument("--no-readme", action="store_true")
    args = ap.parse_args()

    results = run_full_benchmark(args.replay_dir, SETUP)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark_results.json").write_text(
        json.dumps(results, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out / "benchmark_results.md").write_text(report_markdown(results), encoding="utf-8")
    print(report_markdown(results))
    if not args.no_readme:
        readme = Path(args.readme)
        text = readme.read_text(encoding="utf-8")
        block = readme_results(results)
        if README_START in text and README_END in text:
            head, rest = text.split(README_START, 1)
            _, tail = rest.split(README_END, 1)
            text = head + block + tail
        elif "{{RESULTS}}" in text:
            text = text.replace("{{RESULTS}}", block)
        else:
            raise SystemExit("README has neither result markers nor {{RESULTS}}")
        readme.write_text(text, encoding="utf-8")
        print(f"updated {readme}")


if __name__ == "__main__":
    main()
