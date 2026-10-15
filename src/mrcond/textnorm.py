"""Whitespace/punctuation-normalised text matching with offset mapping.

The verifier accepts a quote only if it occurs verbatim in the source text
after *whitespace normalisation*: runs of whitespace collapse to one space,
whitespace after a hyphen is dropped (PDF line-break hyphenation), and a small
set of typographic variants are unified (micro sign vs Greek mu, curly vs
straight quotes, dash variants, non-breaking space, ligatures). Matching is
case-sensitive: a quote that changes letters, digits or units is not verbatim.
"""

from __future__ import annotations

import re

__all__ = ["normalize_with_map", "normalize", "find_all", "find_best", "word_count", "trim_quote"]

_CHAR_EQUIV: dict[str, str] = {
    "µ": "μ",  # micro sign -> greek mu
    "“": '"',
    "”": '"',
    "„": '"',
    "‘": "'",
    "’": "'",
    "′": "'",
    "‐": "-",
    "‑": "-",
    "‒": "-",
    "–": "-",
    "—": "-",
    "−": "-",
    " ": " ",
    " ": " ",
    " ": " ",
    "­": "",  # soft hyphen
    "ﬁ": "fi",
    "ﬂ": "fl",
    "≤": "≤",
    "≥": "≥",
}


def normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Return (normalised text, map from normalised index -> original index)."""
    out: list[str] = []
    idx: list[int] = []
    prev_space = True  # strip leading whitespace
    prev_hyphen = False
    for i, ch in enumerate(text):
        rep = _CHAR_EQUIV.get(ch, ch)
        if rep == "":
            continue
        if rep.isspace():
            if prev_space or prev_hyphen:
                continue
            out.append(" ")
            idx.append(i)
            prev_space = True
            continue
        for r in rep:
            out.append(r)
            idx.append(i)
        prev_space = False
        prev_hyphen = rep == "-"
    # strip trailing space
    while out and out[-1] == " ":
        out.pop()
        idx.pop()
    return "".join(out), idx


def normalize(text: str) -> str:
    return normalize_with_map(text)[0]


def find_all(haystack: str, needle: str) -> list[tuple[int, int]]:
    """All (start, end) original-offset spans in `haystack` matching `needle` (normalised)."""
    n_needle = normalize(needle)
    if not n_needle:
        return []
    n_hay, idx = normalize_with_map(haystack)
    spans = []
    start = 0
    while True:
        k = n_hay.find(n_needle, start)
        if k < 0:
            break
        s = idx[k]
        e = idx[k + len(n_needle) - 1] + 1
        spans.append((s, e))
        start = k + 1
    return spans


def find_best(haystack: str, needle: str, hint_start: int | None = None) -> tuple[int, int] | None:
    """The match closest to `hint_start` (or the first match)."""
    spans = find_all(haystack, needle)
    if not spans:
        return None
    if hint_start is None:
        return spans[0]
    return min(spans, key=lambda sp: abs(sp[0] - hint_start))


_WORD_RE = re.compile(r"\S+")


def word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


def trim_quote(
    text: str, start: int, end: int, focus_start: int, focus_end: int, max_words: int = 25
) -> tuple[int, int]:
    """Shrink [start, end) to at most `max_words` words while keeping [focus_start, focus_end)."""
    words = [(m.start() + start, m.end() + start) for m in _WORD_RE.finditer(text[start:end])]
    if len(words) <= max_words:
        return start, end
    # index of words overlapping the focus
    fi = [i for i, (s, e) in enumerate(words) if e > focus_start and s < focus_end]
    if not fi:
        fi = [0]
    lo, hi = fi[0], fi[-1]
    while hi - lo + 1 < max_words and (lo > 0 or hi < len(words) - 1):
        if lo > 0:
            lo -= 1
        if hi - lo + 1 < max_words and hi < len(words) - 1:
            hi += 1
    if hi - lo + 1 > max_words:
        hi = lo + max_words - 1
    return words[lo][0], words[hi][1]
