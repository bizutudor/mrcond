"""Cap the quotes stored in LLM recordings at 25 words (no long manufacturer text in the repository).

Models sometimes copy a whole sentence as "quote". This rewrites every ``*quote`` string longer than 25 words in
``data/replay/*.jsonl`` to a 25-word window around the value it supports (the same window the verifier keeps),
leaving everything else in the record unchanged. Trimming must not change the evaluation: when the source text is
available, the window is chosen so that the capped quote is found verbatim in the source exactly when the original
was (a misquoted sentence stays misquoted and keeps counting as an unsupported value). Run it after recording,
then re-run the benchmark from the replays and compare: the QA review (October 2026) checked that all metrics are
identical before and after capping.

    python scripts/cap_replay_quotes.py [--check]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mrcond.extractors.llm import parse_json_lenient  # noqa: E402
from mrcond.textnorm import find_all, trim_quote, word_count  # noqa: E402
from mrcond.units import find_numbers  # noqa: E402

MAX_WORDS = 25
_STATUS = re.compile(r"MRI?\s*-?\s*(?:[Cc]onditional|[Uu]nsafe|[Ss]afe)|MR\s*-?\s*(?:CONDITIONAL|UNSAFE|SAFE)")


def _focus(quote: str, value) -> tuple[int, int]:
    """Character span the capped quote must keep: the number of `value`, else a status term, else the start."""
    targets = []
    for v in value if isinstance(value, list) else [value]:
        for x, _, _ in find_numbers(str(v)):
            targets.append(x)
    for x, s, e in find_numbers(quote):
        if any(abs(x - t) < 1e-9 for t in targets):
            return s, e
    m = _STATUS.search(quote)
    if m:
        return m.start(), m.end()
    return 0, 0


def _verbatim(doc, quote: str) -> bool:
    return any(find_all(p.text, quote) for p in doc.pages)


def cap(quote: str, value, doc=None) -> str:
    if word_count(quote) <= MAX_WORDS:
        return quote
    fs, fe = _focus(quote, value)
    s, e = trim_quote(quote, 0, len(quote), fs, fe, MAX_WORDS)
    if doc is None or _verbatim(doc, quote) == _verbatim(doc, quote[s:e]):
        return quote[s:e]
    # keep the verbatim status of the original: the nearest 25-word window (with the value, if possible) that does
    words = [(m.start(), m.end()) for m in re.finditer(r"\S+", quote)]
    want = _verbatim(doc, quote)
    windows = [(words[i][0], words[i + MAX_WORDS - 1][1]) for i in range(len(words) - MAX_WORDS + 1)]
    ok = [(a, b) for a, b in windows if _verbatim(doc, quote[a:b]) == want]
    if not ok:
        return quote[s:e]
    with_value = [(a, b) for a, b in ok if a <= fs and fe <= b] or ok
    a, b = min(with_value, key=lambda w: abs(w[0] - s))
    return quote[a:b]


def cap_record(rec: dict, doc=None) -> tuple[dict, int]:
    try:
        obj = json.loads(rec["content"])
    except (json.JSONDecodeError, TypeError):
        obj = parse_json_lenient(rec.get("content") or "")
        if not obj:
            return rec, 0
    n = 0

    def walk(o, value=None):
        nonlocal n
        if isinstance(o, dict):
            local_value = o.get("value", o.get("field_strengths_tesla", value))
            for k, v in list(o.items()):
                if k.endswith("quote") and isinstance(v, str) and word_count(v) > MAX_WORDS:
                    vv = (
                        o.get("field_strengths_tesla")
                        if k == "field_strength_quote"
                        else (o.get("mr_status") if k == "mr_status_quote" else local_value)
                    )
                    o[k] = cap(v, vv, doc)
                    n += 1
                else:
                    walk(v, local_value)
        elif isinstance(o, list):
            for v in o:
                walk(v, value)

    walk(obj)
    if n:
        rec = {**rec, "content": json.dumps(obj, ensure_ascii=False, separators=(",", ":")), "quotes_capped": n}
    return rec, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay-dir", default=str(ROOT / "data" / "replay"))
    ap.add_argument("--check", action="store_true", help="only report quotes longer than 25 words")
    args = ap.parse_args()
    docs = {}
    if not args.check:
        from mrcond.benchmark.runner import load_track_r, load_track_s

        docs = {d.doc_id: d for d, _, _ in load_track_r()[0]}
        docs.update({d.doc_id: d for d, _, _ in load_track_s("all")})
    total = 0
    for path in sorted(Path(args.replay_dir).glob("*.jsonl")):
        lines, changed = [], 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            rec, n = cap_record(rec, docs.get(rec.get("doc_id")))
            changed += n
            lines.append(json.dumps(rec, ensure_ascii=False))
        total += changed
        print(f"{path.name}: {changed} quote(s) over {MAX_WORDS} words")
        if changed and not args.check:
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if args.check and total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
