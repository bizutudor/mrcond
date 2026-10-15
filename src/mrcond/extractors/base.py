"""Common types for extractors.

Extractors do not build :class:`~mrcond.schema.LabelDocument` objects
directly. They emit *candidates*: one statement-level value per field, each
with its own evidence and an optional field-strength scope. The verifier then
accepts or rejects each candidate, and :func:`mrcond.assemble.assemble` merges
the candidates into condition sets using the most conservative value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..ingest import DocumentText, Segment
from ..schema import Evidence, FieldStatus, FieldValue
from ..textnorm import trim_quote

__all__ = ["Candidate", "ExtractionOutput", "Extractor", "evidence_from_segment"]


@dataclass
class Candidate:
    field: str  # a FIELD_NAMES entry, or "mr_status"
    fv: FieldValue
    scope_b0: tuple[float, ...] | None = None  # None: applies to every permitted field strength
    source: str = ""  # pattern / model id (debugging)
    accepted: bool | None = None  # set by the verifier
    reasons: list[str] = field(default_factory=list)

    def key(self) -> tuple:
        return (self.field, self.scope_b0, repr(self.fv.normalized_value))


@dataclass
class ExtractionOutput:
    doc_id: str
    extractor: str
    candidates: list[Candidate] = field(default_factory=list)
    status_candidates: list[Candidate] = field(default_factory=list)
    raw: Any = None  # raw model output (LLM) for audit
    meta: dict = field(default_factory=dict)
    verified: bool = False

    def all_candidates(self) -> list[Candidate]:
        return [*self.status_candidates, *self.candidates]


class Extractor(Protocol):
    name: str

    def extract(self, doc: DocumentText) -> ExtractionOutput: ...


def evidence_from_segment(
    doc_id: str,
    seg: Segment,
    rel_start: int,
    rel_end: int,
    focus: tuple[int, int] | None = None,
    max_words: int = 25,
) -> Evidence:
    """Build evidence for seg.text[rel_start:rel_end] (trimmed to <= max_words around `focus`)."""
    text = seg.text
    rel_start = max(0, rel_start)
    rel_end = min(len(text), rel_end)
    # snap to word boundaries (never cut a word in half)
    while 0 < rel_start < len(text) and text[rel_start - 1].isalnum() and text[rel_start].isalnum():
        rel_start -= 1
    while 0 < rel_end < len(text) and text[rel_end - 1].isalnum() and text[rel_end].isalnum():
        rel_end += 1
    fs, fe = focus if focus else (rel_start, rel_end)
    s, e = trim_quote(seg.text, rel_start, rel_end, fs, fe, max_words)
    # strip dangling punctuation/whitespace at the edges
    while s < e and seg.text[s] in " \t\n,;:":
        s += 1
    while e > s and seg.text[e - 1] in " \t\n,;:(":
        e -= 1
    return Evidence(
        doc_id=doc_id,
        page=seg.page,
        char_start=seg.start + s,
        char_end=seg.start + e,
        quote=seg.text[s:e],
    )


def extracted(
    value: Any,
    unit: str | None,
    normalized: Any,
    normalized_unit: str | None,
    comparator: str | None,
    evidence: list[Evidence],
) -> FieldValue:
    return FieldValue(
        status=FieldStatus.EXTRACTED,
        value=value,
        unit=unit,
        normalized_value=normalized,
        normalized_unit=normalized_unit,
        comparator=comparator,  # type: ignore[arg-type]
        evidence=evidence,
    )
