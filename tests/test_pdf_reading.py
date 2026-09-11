"""PDFs must be readable end to end.

Before: a 79-page transcript reached the model as its first ~11 pages, the
viewer document it was pointed to held the same 15,000 characters, a
four-PDF upload dropped the last two files, and read_file returned raw PDF
bytes. Test PDFs are built from raw PDF syntax so no optional library is needed.
"""
import asyncio
import json

from src import document_processor as dp
from src.agent_tools.filesystem_tools import ReadFileTool


def make_pdf(path, pages):
    """Write a minimal text PDF: one Helvetica text block per page."""
    n = len(pages)
    page_ids = [4 + 2 * i for i in range(n)]
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        ("<< /Type /Pages /Kids [%s] /Count %d >>" % (" ".join(f"{p} 0 R" for p in page_ids), n)).encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for i, text in enumerate(pages):
        ops = ["BT", "/F1 9 Tf", "11 TL", "30 810 Td"]
        for line in text.split("\n"):
            esc = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            ops.append(f"({esc}) Tj T*")
        ops.append("ET")
        stream = "\n".join(ops).encode("latin-1")
        objs.append(("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
                     "/Resources << /Font << /F1 3 0 R >> >> /Contents %d 0 R >>" % (page_ids[i] + 1)).encode())
        objs.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for idx, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{idx} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))
    return path


def page_text(n, lines=40):
    return "\n".join(f"Page {n} line {i}: the court heard argument on the objection." for i in range(lines))


def test_extraction_keeps_every_page_when_asked(tmp_path):
    pdf = make_pdf(tmp_path / "t.pdf", [page_text(n) for n in range(1, 11)])  # ~25k chars
    capped = dp._process_pdf(str(pdf))
    full = dp._process_pdf(str(pdf), max_chars=None)
    assert "[PDF content truncated]" in capped and "Page 10 line" not in capped
    assert "[Page 10 text]" in full and "Page 10 line 39" in full


def test_inline_budget_scales_with_context_and_keeps_the_old_floor():
    assert dp.inline_attachment_budget(None) == dp.MAX_INLINE_ATTACHMENT_CHARS
    assert dp.inline_attachment_budget(8192) == dp.MAX_INLINE_ATTACHMENT_CHARS
    # 262,144 tokens holds a whole 207,799-character transcript.
    assert dp.inline_attachment_budget(262144) > 207_799


def _pdf_block(title, body):
    return f"\n\n[PDF attached: {title} — opened in document viewer.]\n\n[PDF content — {title}]:\n{body}"


def _body(pages):
    return "\n\n".join(f"[Page {n} text]:\n{page_text(n)}" for n in range(1, pages + 1))


def test_a_pdf_that_fits_is_inlined_whole():
    body = _body(3)
    text = _pdf_block("t", body)
    kept, used = dp._fit_pdf_inline(text, len(text) + 10, "t.pdf", {"doc_id": "D", "body": body, "doc_body_start": 50})
    assert kept == text and used == len(text)


def test_truncation_says_which_pages_and_where_to_read_on():
    body = _body(12)
    header = '<!-- pdf_source upload_id="x.pdf" -->\n\n# t\n\n'
    doc_content = header + body + "\n"
    info = {"doc_id": "DOC1", "body": body, "doc_body_start": doc_content.find(body[:200])}
    kept, used = dp._fit_pdf_inline(_pdf_block("t", body), 8000, "t.pdf", info)
    assert used <= 8000
    note = kept[kept.rindex("[Showing"):]
    assert "of 12" in note and "document_id=DOC1" in note
    offset = int(note.split("offset=")[1].split()[0])
    shown = kept[kept.index("[Page 1 text]"):kept.rindex("\n\n[Showing")]
    # The offset lands exactly where the inline text stopped.
    assert doc_content[offset - 40:offset] == shown[-40:]
    assert body.startswith(shown)


def test_a_pdf_with_no_room_left_is_named_with_a_way_to_read_it():
    body = _body(5)
    kept, _ = dp._fit_pdf_inline(_pdf_block("t", body), 300, "t.pdf", {"doc_id": "DOC2", "body": body, "doc_body_start": 42})
    assert "[Not shown inline: t.pdf (5 pages" in kept
    assert "document_id=DOC2 offset=42" in kept
    assert "[Page 1 text]" not in kept


def _read(tmp_path, monkeypatch, args):
    import src.tool_execution as te
    monkeypatch.setattr(te, "_resolve_tool_path", lambda p: p)
    return asyncio.run(ReadFileTool().execute(json.dumps(args), {}))


def test_read_file_extracts_pdf_text_instead_of_raw_bytes(tmp_path, monkeypatch):
    pdf = make_pdf(tmp_path / "doc.pdf", [page_text(1), page_text(2)])
    out = _read(tmp_path, monkeypatch, {"path": str(pdf)})["output"]
    assert "%PDF" not in out and "stream" not in out
    assert "[PDF: 2 pages" in out and "[Page 2]" in out and "Page 2 line 39" in out


def test_read_file_pages_through_a_pdf_by_line(tmp_path, monkeypatch):
    pdf = make_pdf(tmp_path / "doc.pdf", [page_text(n) for n in range(1, 4)])
    first = _read(tmp_path, monkeypatch, {"path": str(pdf), "offset": 1, "limit": 5})["output"]
    later = _read(tmp_path, monkeypatch, {"path": str(pdf), "offset": 90, "limit": 5})["output"]
    assert "[PDF: 3 pages" in first and "Page 3" not in first
    assert "Page 3 line" in later
