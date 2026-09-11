"""The open-document context for PDFs.

find_source_upload_id() matches both PDF markers, so every plain PDF copy was
treated as a fillable form: its whole text went into the prompt every turn
under FORM MODE instructions telling the model not to read the PDF. A 79-page
transcript was sent twice on the upload turn and re-read on every follow-up.
"""
from types import SimpleNamespace

from src import agent_loop as al

PLAIN_HEAD = '<!-- pdf_source upload_id="0123456789abcdef0123456789abcdef.pdf" -->\n\n# Record\n\n'
FORM_HEAD = '<!-- pdf_form_source upload_id="0123456789abcdef0123456789abcdef.pdf" fields="3" -->\n\n# Form\n\n'


def _body(pages, per_page=900):
    return "\n\n".join(f"[Page {n} text]:\n" + ("word " * (per_page // 5)).strip() + f" end of page {n}." for n in range(1, pages + 1))


def test_plain_and_form_pdf_documents_are_told_apart():
    assert al._pdf_doc_kind(PLAIN_HEAD + "text") == "plain"
    assert al._pdf_doc_kind(FORM_HEAD + "- **Name:** x <!-- field=name type=text -->") == "form"
    assert al._pdf_doc_kind("# Just a note\n\nhello") is None


def test_a_long_pdf_gets_a_bounded_preview_and_an_exact_offset():
    body = _body(79)
    doc = SimpleNamespace(id="DOC1", title="Record", current_content=PLAIN_HEAD + body + "\n")
    ctx = al._plain_pdf_doc_context(doc)
    assert len(ctx) < al._PDF_DOC_PREVIEW_CHARS + 1200
    assert "79 pages" in ctx and "document_id=DOC1" in ctx
    assert "FORM" not in ctx
    offset = int(ctx.split("offset=")[1].split()[0])
    preview = ctx.split("```\n", 1)[1].split("\n```", 1)[0]
    # The offset lands exactly where the preview stopped.
    assert doc.current_content[offset - 40:offset] == preview[-40:]
    assert body.startswith(preview)


def test_a_short_pdf_is_shown_whole_with_no_paging_note():
    body = _body(2, per_page=300)
    doc = SimpleNamespace(id="DOC2", title="Short", current_content=PLAIN_HEAD + body + "\n")
    ctx = al._plain_pdf_doc_context(doc)
    assert body in ctx
    assert "offset=" not in ctx
