"""Track R corpus: manifest of public labeling documents, fetching, text cache, annotations.

MRCond does not redistribute manufacturer documents. The repository contains a
manifest (URL, retrieval date, SHA-256, relevant pages) and this fetch code;
fetched files go to ``data/raw/`` and extracted text to ``data/text/`` (both
git-ignored). Annotations store page, character offsets and quotes of at most
25 words. US federal (FDA) documents are public domain; their extracted pages
are kept in ``data/public/`` so the demo and tests work offline.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from .ingest import DocumentText, load_document
from .schema import LabelDocument

__all__ = [
    "CorpusEntry",
    "Manifest",
    "repo_root",
    "data_dir",
    "load_manifest",
    "fetch_entry",
    "load_text",
    "load_annotation",
    "annotation_path",
    "verified_path",
]

USER_AGENT = "Mozilla/5.0 (compatible; MRCond/0.1 research benchmark fetcher)"


def repo_root() -> Path:
    env = os.environ.get("MRCOND_ROOT")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists() and (parent / "data").exists():
            return parent
    return Path.cwd()


def data_dir() -> Path:
    return Path(os.environ.get("MRCOND_DATA", repo_root() / "data"))


@dataclass
class CorpusEntry:
    doc_id: str
    title: str
    url: str
    file: str
    format: str
    manufacturer: str | None = None
    device_name: str | None = None
    device_type: str | None = None
    retrieved: str | None = None
    sha256: str | None = None
    pages: list[int] | None = None
    license: str | None = None
    redistributable: bool = False
    section: dict | None = None  # {"start": verbatim text, "end": verbatim text}: keep only this span
    status_expected: str | None = None
    annotation_status: str | None = None
    notes: str | None = None
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> CorpusEntry:
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d and k != "extra"}
        extra = {k: v for k, v in d.items() if k not in cls.__dataclass_fields__}
        return cls(**known, extra=extra)


@dataclass
class Manifest:
    entries: list[CorpusEntry]
    meta: dict

    def get(self, doc_id: str) -> CorpusEntry:
        for e in self.entries:
            if e.doc_id == doc_id:
                return e
        raise KeyError(doc_id)


def load_manifest(path: str | Path | None = None) -> Manifest:
    p = Path(path) if path else data_dir() / "corpus" / "manifest.json"
    if not p.exists():
        return Manifest([], {})
    raw = json.loads(p.read_text(encoding="utf-8"))
    return Manifest(
        [CorpusEntry.from_dict(d) for d in raw.get("documents", [])], {k: v for k, v in raw.items() if k != "documents"}
    )


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def raw_path(entry: CorpusEntry) -> Path:
    return data_dir() / "raw" / entry.file


def text_cache_path(doc_id: str) -> Path:
    return data_dir() / "text" / f"{doc_id}.json"


def public_text_path(doc_id: str) -> Path:
    return data_dir() / "public" / f"{doc_id}.json"


def apply_section(doc: DocumentText, section: dict | None) -> DocumentText:
    """Blank out (with spaces, keeping newlines and offsets) all text outside `section`.

    `section` = {"start": text, "end": text}; both anchors are matched with whitespace-normalised
    search. Used for documents that cover several devices or many languages.
    """
    if not section:
        return doc
    from .textnorm import find_all

    start_anchor, end_anchor = section.get("start"), section.get("end")
    pages = doc.pages
    s_loc = None
    for i, p in enumerate(pages):
        spans = find_all(p.text, start_anchor) if start_anchor else [(0, 0)]
        if spans:
            s_loc = (i, spans[0][0])
            break
    if s_loc is None:
        raise ValueError(f"{doc.doc_id}: section start anchor not found: {start_anchor!r}")
    e_loc = None
    for i in range(s_loc[0], len(pages)):
        spans = find_all(pages[i].text, end_anchor) if end_anchor else [(len(pages[i].text),) * 2]
        spans = [sp for sp in spans if i > s_loc[0] or sp[0] >= s_loc[1]]
        if spans:
            e_loc = (i, spans[0][1])
            break
    if e_loc is None:
        raise ValueError(f"{doc.doc_id}: section end anchor not found: {end_anchor!r}")

    def blank(t: str) -> str:
        return "".join(c if c == "\n" else " " for c in t)

    for i, p in enumerate(pages):
        if i < s_loc[0] or i > e_loc[0]:
            p.text = blank(p.text)
            continue
        a = s_loc[1] if i == s_loc[0] else 0
        b = e_loc[1] if i == e_loc[0] else len(p.text)
        p.text = blank(p.text[:a]) + p.text[a:b] + blank(p.text[b:])
    doc.pages = [p for p in pages if p.text.strip()]
    doc.meta["section"] = section
    return doc


def fetch_entry(entry: CorpusEntry, *, force: bool = False, timeout: float = 60.0) -> dict:
    """Download `entry` (if needed), check SHA-256, extract and cache its text. Returns a status dict."""
    import httpx

    dest = raw_path(entry)
    dest.parent.mkdir(parents=True, exist_ok=True)
    status: dict = {"doc_id": entry.doc_id, "url": entry.url}
    if force or not dest.exists():
        r = httpx.get(
            entry.url, follow_redirects=True, timeout=timeout, headers={"User-Agent": USER_AGENT, "Accept": "*/*"}
        )
        r.raise_for_status()
        dest.write_bytes(r.content)
        status["downloaded"] = True
        status["retrieved"] = date.today().isoformat()
    digest = sha256_file(dest)
    status["sha256"] = digest
    status["sha256_matches_manifest"] = (entry.sha256 is None) or (digest == entry.sha256)
    doc = apply_section(load_document(dest, entry.doc_id, pages=entry.pages), entry.section)
    doc.meta.update({"url": entry.url, "sha256": digest})
    doc.save(text_cache_path(entry.doc_id))
    status["pages"] = [p.number for p in doc.pages]
    status["chars"] = sum(len(p.text) for p in doc.pages)
    return status


def load_text(entry_or_id: CorpusEntry | str) -> DocumentText | None:
    """Text of a corpus document: public cache, else local text cache, else parse the raw file."""
    doc_id = entry_or_id if isinstance(entry_or_id, str) else entry_or_id.doc_id
    for p in (public_text_path(doc_id), text_cache_path(doc_id)):
        if p.exists():
            return DocumentText.load(p)
    if not isinstance(entry_or_id, str):
        rp = raw_path(entry_or_id)
        if rp.exists():
            return apply_section(load_document(rp, doc_id, pages=entry_or_id.pages), entry_or_id.section)
    return None


def annotation_path(doc_id: str) -> Path:
    return data_dir() / "annotations" / "track_r" / f"{doc_id}.json"


def verified_path(doc_id: str) -> Path:
    return data_dir() / "annotations" / "verified" / f"{doc_id}.json"


def load_annotation(doc_id: str, prefer_verified: bool = True) -> LabelDocument | None:
    paths = [verified_path(doc_id), annotation_path(doc_id)] if prefer_verified else [annotation_path(doc_id)]
    for p in paths:
        if p.exists():
            return LabelDocument.model_validate_json(p.read_text(encoding="utf-8"))
    return None
