import zlib

from mrcond.ingest import DocumentText, Page, html_to_text, load_document, segment_page
from mrcond.textnorm import find_all, find_best, normalize, trim_quote, word_count


def test_normalize_whitespace_and_typography():
    assert normalize("  B1+rms  ≤\n 1.7 µT ") == "B1+rms ≤ 1.7 μT"
    assert normalize("disease-\nmimicking") == "disease-mimicking"
    assert normalize("“quoted” and ‘single’ – dash") == "\"quoted\" and 'single' -dash"


def test_find_all_maps_back_to_original_offsets():
    text = "Maximum Whole -Body SAR 2\nW/kg (Normal\tOperating Mode)"
    spans = find_all(text, "Whole -Body SAR 2 W/kg")
    assert len(spans) == 1
    s, e = spans[0]
    assert text[s:e] == "Whole -Body SAR 2\nW/kg"
    assert find_all(text, "Whole-Body SAR 3 W/kg") == []  # changed digit: not verbatim
    assert find_all(text, "whole -body sar 2 w/kg") == []  # matching is case-sensitive


def test_find_best_prefers_hint():
    text = "2 W/kg here and 2 W/kg there"
    assert find_best(text, "2 W/kg", hint_start=16) == (16, 22)


def test_trim_quote_keeps_focus():
    text = " ".join(f"w{i}" for i in range(60))
    focus = text.index("w40")
    s, e = trim_quote(text, 0, len(text), focus, focus + 3, 25)
    assert word_count(text[s:e]) == 25 and "w40" in text[s:e]


def test_segment_table_rows_and_continuations():
    text = (
        "RF Conditions For 1.5T MR Scanner: Whole-Body SAR ≤ 2 W/kg\n"
        "For 3T MR Scanner : B1+rms ≤ 1.7 μT; for MR scanners that \n"
        "do not report B1+rms, Whole-Body SAR ≤ 1.2 W/kg\n"
        "Scan Duration Scan for up to 30 minutes. Wait 30 minutes before the next \nimaging session.\n"
    )
    segs = [s.text for s in segment_page(text, 1)]
    assert segs[0] == "RF Conditions For 1.5T MR Scanner: Whole-Body SAR ≤ 2 W/kg"
    assert segs[1].startswith("For 3T MR Scanner") and segs[1].endswith("SAR ≤ 1.2 W/kg")
    assert segs[2] == "Scan Duration Scan for up to 30 minutes."
    assert segs[3].startswith("Wait 30 minutes") and "imaging session." in segs[3]
    for s in segment_page(text, 1):
        assert text[s.start : s.end] == s.text


def test_html_to_text_drops_scripts_and_keeps_cells():
    html = (
        "<html><head><script>var x=1;</script><style>p{}</style></head><body><nav>menu</nav>"
        "<table><tr><td>Maximum spatial gradient</td><td>30 T/m</td></tr></table><p>MR Conditional</p>"
        "</body></html>"
    )
    txt = html_to_text(html)
    assert "var x" not in txt and "menu" not in txt
    assert "Maximum spatial gradient" in txt and "30 T/m" in txt and "MR Conditional" in txt


def _minimal_pdf(lines: list[str]) -> bytes:
    """A tiny valid one-page PDF with a text layer (no external dependencies)."""
    content = "BT /F1 11 Tf 50 750 Td 14 TL " + " ".join(f"({ln}) Tj T*" for ln in lines) + " ET"
    stream = zlib.compress(content.encode("latin-1"))
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d /Filter /FlateDecode >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, o in enumerate(objs, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + o + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return bytes(out)


def test_load_pdf_and_text(tmp_path):
    pdf = tmp_path / "label.pdf"
    pdf.write_bytes(
        _minimal_pdf(
            ["MRI Safety Information", "Static magnetic field of 1.5 T", "Maximum whole body averaged SAR of 2 W/kg"]
        )
    )
    doc = load_document(pdf, "mini")
    assert doc.source_type == "pdf" and len(doc.pages) == 1
    assert "1.5 T" in doc.pages[0].text and "2 W/kg" in doc.pages[0].text

    txt = tmp_path / "label.txt"
    txt.write_text("page one\fpage two")
    doc = load_document(txt)
    assert [p.number for p in doc.pages] == [1, 2] and doc.pages[1].text == "page two"


def test_document_text_roundtrip(tmp_path):
    d = DocumentText("x", [Page(3, "abc")], "pdf", "pypdf test")
    d.save(tmp_path / "x.json")
    assert DocumentText.load(tmp_path / "x.json").pages[0].number == 3
