"""Document ingestion: PDF / HTML / plain text -> page-indexed text -> statements.

Offsets used everywhere in MRCond (evidence ``char_start``/``char_end``) are
offsets into :attr:`Page.text` as produced here. Text normalisation is kept
minimal and deterministic (line endings, NUL removal) so that offsets are
reproducible for a given extractor version; quotes are the primary anchor and
are re-located with whitespace-normalised matching when offsets drift.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

__all__ = [
    "DocumentText",
    "Page",
    "Segment",
    "load_document",
    "load_html",
    "load_pdf",
    "load_text",
    "segment_document",
    "segment_page",
    "text_extractor_version",
]


@dataclass
class Page:
    number: int  # 1-based page number in the source document
    text: str


@dataclass
class DocumentText:
    doc_id: str
    pages: list[Page]
    source_type: str = "text"  # pdf | html | text
    text_extractor: str = "plain"
    meta: dict = field(default_factory=dict)

    def page(self, number: int) -> Page | None:
        for p in self.pages:
            if p.number == number:
                return p
        return None

    def page_text(self, number: int) -> str:
        p = self.page(number)
        return p.text if p else ""

    @property
    def full_text(self) -> str:
        return "\n\f\n".join(p.text for p in self.pages)

    def to_json(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "source_type": self.source_type,
            "text_extractor": self.text_extractor,
            "meta": self.meta,
            "pages": [asdict(p) for p in self.pages],
        }

    @classmethod
    def from_json(cls, data: dict) -> DocumentText:
        return cls(
            doc_id=data["doc_id"],
            pages=[Page(int(p["number"]), p["text"]) for p in data["pages"]],
            source_type=data.get("source_type", "text"),
            text_extractor=data.get("text_extractor", "plain"),
            meta=data.get("meta", {}),
        )

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_json(), ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> DocumentText:
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))


def _clean(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")


def text_extractor_version() -> str:
    try:
        import pypdf

        return f"pypdf {pypdf.__version__}"
    except Exception:  # pragma: no cover
        return "pypdf (unknown)"


def load_pdf(path: str | Path, doc_id: str, pages: list[int] | None = None) -> DocumentText:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    wanted = set(pages) if pages else None
    out = []
    for i, pg in enumerate(reader.pages, start=1):
        if wanted is not None and i not in wanted:
            continue
        try:
            txt = pg.extract_text() or ""
        except Exception:  # pragma: no cover - malformed page
            txt = ""
        out.append(Page(i, _clean(txt)))
    return DocumentText(doc_id, out, "pdf", text_extractor_version(), {"n_pages_total": len(reader.pages)})


_BLOCK_TAGS = ("p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "table", "section", "br", "td", "th")


def html_to_text(html: str) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "header", "form", "iframe"]):
        tag.decompose()
    for td in soup.find_all(["td", "th"]):
        td.insert_after(" ")
    text = soup.get_text("\n")
    lines = [re.sub(r"[ \t ]+", " ", ln).strip() for ln in _clean(text).split("\n")]
    out: list[str] = []
    for ln in lines:
        if not ln:
            if out and out[-1] != "":
                out.append("")
            continue
        out.append(ln)
    return "\n".join(out).strip() + "\n"


def load_html(path: str | Path, doc_id: str) -> DocumentText:
    raw = Path(path).read_bytes()
    try:
        html = raw.decode("utf-8")
    except UnicodeDecodeError:
        html = raw.decode("latin-1")
    from bs4 import __version__ as bs4_version

    return DocumentText(doc_id, [Page(1, html_to_text(html))], "html", f"beautifulsoup4 {bs4_version}")


def load_text(path: str | Path, doc_id: str) -> DocumentText:
    text = _clean(Path(path).read_text(encoding="utf-8"))
    chunks = text.split("\f")
    return DocumentText(doc_id, [Page(i, c) for i, c in enumerate(chunks, start=1)], "text", "plain")


def load_document(path: str | Path, doc_id: str | None = None, pages: list[int] | None = None) -> DocumentText:
    p = Path(path)
    doc_id = doc_id or p.stem
    suffix = p.suffix.lower()
    head = p.read_bytes()[:1024]
    if suffix == ".pdf" or head.startswith(b"%PDF"):
        return load_pdf(p, doc_id, pages)
    if suffix in (".html", ".htm") or b"<html" in head.lower() or b"<!doctype html" in head.lower():
        doc = load_html(p, doc_id)
    elif suffix == ".json":
        doc = DocumentText.load(p)
    else:
        doc = load_text(p, doc_id)
    if pages:
        doc.pages = [pg for pg in doc.pages if pg.number in set(pages)]
    return doc


# --------------------------------------------------------------------------------------
# Statement segmentation
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Segment:
    page: int
    start: int
    end: int
    text: str
    line_start: bool = True  # segment begins at the start of a line


_TERMINAL = re.compile(r"[.:;!?]\s*$")
_CONNECTOR_END = re.compile(
    r"(?:\b(?:of|the|and|or|to|with|in|a|an|for|at|than|by|is|are|be|on|from|that|this|per|if|when|not|"
    r"maximum|max|minimum|up|under)|[≤≥<>=,(\-–/+])\s*$",
    re.IGNORECASE,
)
_BULLET = re.compile(r"^\s*(?:[•·▪◦●■\-–*]|\d{1,2}[.)]|\(?[a-z]\))\s+")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z•·\"“(])")
_ABBREV = re.compile(r"(?:\b(?:e\.g|i\.e|approx|Fig|No|vs|Dr|St|Inc|Ltd|Co|ca|cf|etc|Ref|Vol|p|pp))\.$")


def _lines_with_offsets(text: str) -> list[tuple[int, int]]:
    spans = []
    pos = 0
    for line in text.split("\n"):
        spans.append((pos, pos + len(line)))
        pos += len(line) + 1
    return spans


def segment_page(text: str, page: int) -> list[Segment]:
    """Split page text into statements (table rows, list items, sentences) with offsets."""
    lines = [(s, e) for s, e in _lines_with_offsets(text) if text[s:e].strip()]
    blocks: list[tuple[int, int]] = []
    for s, e in lines:
        line = text[s:e]
        stripped = line.strip()
        if blocks:
            ps, pe = blocks[-1]
            prev = text[ps:pe].rstrip()
            cont = False
            if not _BULLET.match(line):
                if prev.endswith("-") and not prev.endswith(" -"):
                    cont = True
                elif not _TERMINAL.search(prev):
                    first = stripped[:1]
                    if first.islower() or (first.isdigit() and not re.match(r"^\d{1,2}[.)]\s", stripped)):
                        cont = True
                    elif first in "(≤≥<>=/":
                        cont = True
                    elif _CONNECTOR_END.search(prev):
                        cont = True
            if cont:
                blocks[-1] = (ps, e)
                continue
        blocks.append((s, e))

    segments: list[Segment] = []
    for bs, be in blocks:
        block = text[bs:be]
        # strip leading whitespace
        lead = len(block) - len(block.lstrip())
        pieces = []
        last = lead
        for m in _SENT_SPLIT.finditer(block):
            before = block[last : m.start()]
            if _ABBREV.search(before.rstrip()):
                continue
            pieces.append((bs + last, bs + m.start()))
            last = m.end()
        pieces.append((bs + last, be))
        first = True
        for ps, pe in pieces:
            seg_text = text[ps:pe]
            stripped_r = seg_text.rstrip()
            pe2 = ps + len(stripped_r)
            ls = len(seg_text) - len(seg_text.lstrip())
            if pe2 <= ps + ls:
                continue
            segments.append(Segment(page, ps + ls, pe2, text[ps + ls : pe2], line_start=first))
            first = False
    return segments


def segment_document(doc: DocumentText) -> list[Segment]:
    out: list[Segment] = []
    for p in doc.pages:
        out.extend(segment_page(p.text, p.number))
    return out
