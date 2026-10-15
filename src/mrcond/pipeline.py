"""High-level pipeline: document text -> extractor -> verifier -> conservative merge."""

from __future__ import annotations

from .assemble import assemble
from .extractors.base import ExtractionOutput, Extractor
from .ingest import DocumentText
from .schema import LabelDocument
from .verifier import verify_output

__all__ = ["run_extraction", "get_extractor", "EXTRACTOR_NAMES"]

EXTRACTOR_NAMES = ("rule", "rule+verifier", "llm", "llm+verifier")


def run_extraction(
    doc: DocumentText,
    extractor: Extractor,
    *,
    verify: bool = True,
    device_name: str | None = None,
    manufacturer: str | None = None,
    device_type: str | None = None,
) -> tuple[LabelDocument, ExtractionOutput]:
    """Extract candidates, optionally verify them, and assemble a LabelDocument."""
    out = extractor.extract(doc)
    if verify:
        verify_output(out, doc)
    label = assemble(
        out,
        device_name=device_name,
        manufacturer=manufacturer,
        device_type=device_type,
        text_extractor=doc.text_extractor,
    )
    if verify:
        label.provenance.extractor = f"{out.extractor}+verifier"
        label.provenance.notes.append(
            "verifier: {n_accepted}/{n_candidates} candidate values accepted".format(**out.meta["verification"])
        )
    return label, out


def get_extractor(name: str, **kwargs) -> tuple[Extractor, bool]:
    """Resolve an extractor name ("rule", "rule+verifier", "llm", "llm+verifier") -> (extractor, verify)."""
    base, _, suffix = name.partition("+")
    verify = suffix == "verifier"
    if base == "rule":
        from .extractors.rule import RuleExtractor

        return RuleExtractor(), verify
    if base == "llm":
        from .extractors.llm import LLMExtractor

        return LLMExtractor.from_config(**kwargs), verify
    raise ValueError(f"unknown extractor: {name}")
