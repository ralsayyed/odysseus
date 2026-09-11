# src/document_processor.py
"""Document processing: PDF/OCR extraction, text file handling, image VL analysis, user content building."""

import os
import logging
import mimetypes
import base64
import re
import tempfile
from typing import List, Dict, Any

from src.llm_core import llm_call

logger = logging.getLogger(__name__)

MAX_INLINE_ATTACHMENT_CHARS = 24000
MIN_INLINE_ATTACHMENT_SLICE = 500

# Share of the model's context window one message's attachments may fill
# inline, and a conservative characters-per-token rate for extracted text. The
# fixed 24,000-character budget (~6k tokens) suited small-context models; on a
# 262k-token model it showed 11 of 79 pages of one PDF and dropped the last two
# files of a four-PDF upload entirely. 0.3 still left most of a large window
# unused: a 64k-token file fit but a second one did not, and the point of a
# 262k window is to hold big documents whole. Half the window leaves ample
# room for system prompt, tools, history, and the reply while letting large
# uploads through; per-file extraction no longer truncates before this budget
# (see _process_text_file), so the budget is the single size authority.
INLINE_ATTACHMENT_CONTEXT_SHARE = 0.5
INLINE_CHARS_PER_TOKEN = 3.5


def inline_attachment_budget(context_tokens: int | None) -> int:
    """Characters of attachment text one message may carry inline.

    Scales with the model's context window, and never drops below the old
    fixed budget, so small-context models behave exactly as before.
    """
    if not context_tokens or context_tokens <= 0:
        return MAX_INLINE_ATTACHMENT_CHARS
    scaled = int(context_tokens * INLINE_ATTACHMENT_CONTEXT_SHARE * INLINE_CHARS_PER_TOKEN)
    return max(MAX_INLINE_ATTACHMENT_CHARS, scaled)


def _is_text_file(path: str) -> bool:
    """Check if file has text extension."""
    return any(
        path.lower().endswith(ext)
        for ext in (".txt", ".py", ".html", ".htm", ".md", ".json", ".csv", ".log", ".js", ".nix")
    )


def _process_text_file(path: str) -> str:
    """Process text file with enhanced formatting and metadata."""
    language_map = {
        ".py": "python", ".js": "javascript", ".html": "html", ".css": "css",
        ".json": "json", ".md": "markdown", ".txt": "text", ".csv": "csv",
        ".log": "log", ".sh": "bash", ".bash": "bash", ".nix": "nix",
        ".yml": "yaml", ".yaml": "yaml",
        ".xml": "xml", ".sql": "sql", ".cpp": "cpp", ".c": "c",
        ".java": "java", ".go": "go", ".rs": "rust", ".php": "php",
        ".rb": "ruby", ".ts": "typescript", ".jsx": "javascript", ".tsx": "typescript",
    }

    filename = os.path.basename(path)
    _, ext = os.path.splitext(path.lower())
    language = language_map.get(ext, "text")

    try:
        from src.personal_docs import read_text_file
        content = read_text_file(path)
    except Exception:
        try:
            with open(path, "rb") as f:
                raw_data = f.read()
            try:
                content = raw_data.decode("utf-8")
            except UnicodeDecodeError:
                from charset_normalizer import detect
                encoding = (detect(raw_data) or {}).get("encoding") or "utf-8"
                content = raw_data.decode(encoding, errors="replace")
        except Exception as e:
            logger.error(f"Failed to read file {path}: {e}")
            return "\n\n[Failed to read attached file]"

    try:
        file_size = os.path.getsize(path)
        size_str = f"{file_size:,}"
    except OSError:
        size_str = "unknown"

    # No per-file cap here: text files go through whole and the shared inline
    # attachment budget (inline_attachment_budget, which scales with the
    # model's context) is the single authority that trims — with a marker
    # telling the model exactly how to read the remainder via `read_file`.
    # The old 30,000-character hard cap truncated a large upload twice (once
    # here, once in the budget) and made big-context models useless for
    # whole-file questions.
    lines = content.split("\n")
    line_count = len(lines)
    content_length = len(content)

    header = f"\n=== File: {filename} ===\n"
    header += f"[Type: {language}, Lines: {line_count}, Size: {size_str} bytes]"

    code_extensions = {
        ".py", ".js", ".html", ".css", ".json", ".md", ".sh", ".bash", ".nix",
        ".yml", ".yaml", ".xml", ".sql", ".cpp", ".c", ".java", ".go", ".rs", ".php", ".rb",
        ".ts", ".jsx", ".tsx",
    }
    if ext in code_extensions:
        code_block = f"```{language}\n{content}"
        code_block += "\n```"
        return header + "\n\n" + code_block
    else:
        return header + "\n\n" + content


def _process_pdf(path: str, owner: str | None = None, max_chars: int | None = 15000) -> str:
    """Process PDF file with text extraction (pypdf). Uses VL model for image-heavy pages.

    ``max_chars`` caps the returned text; the default suits callers that paste
    it into a short prompt. Pass None for every page: the chat attachment and
    the viewer document need the whole PDF, not the first ~11 pages of 79.
    """
    try:
        from pypdf import PdfReader
        pdf_text = ""
        reader = PdfReader(path)

        for page_num, page in enumerate(reader.pages):
            page_text = (page.extract_text() or "").strip()
            if page_text:
                pdf_text += f"\n\n[Page {page_num + 1} text]:\n{page_text}"

            # For pages with images but little text, try VL model
            try:
                images = list(page.images)
            except Exception:
                images = []
            if images and len(page_text) < 50:
                for img_index, img in enumerate(images[:3]):  # cap at 3 images per page
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                            temp_img_path = tmp.name
                        try:
                            img.image.save(temp_img_path, "PNG")  # pypdf -> PIL image
                            ocr_text = analyze_image_with_vl(temp_img_path, owner=owner)
                            if ocr_text and "unavailable" not in ocr_text.lower():
                                pdf_text += f"\n\n[Page {page_num + 1} image {img_index + 1} text]: {ocr_text}"
                        finally:
                            try:
                                os.unlink(temp_img_path)
                            except OSError:
                                pass
                    except Exception as e:
                        logger.warning(f"Failed to analyze image in PDF: {e}")
                        continue

        if pdf_text:
            if max_chars is not None and len(pdf_text) > max_chars:
                pdf_text = pdf_text[:max_chars] + "\n[PDF content truncated]"
            return f"\n\n[PDF content]:{pdf_text}"
        else:
            return "\n\n[PDF processed but no readable content found]"

    except Exception as e:
        return f"\n\n[PDF processing failed: {str(e)}]"


def _fit_inline_attachment_text(
    text: str,
    remaining: int,
    display_name: str,
) -> tuple[str, int]:
    """Fit extracted attachment text into the shared inline attachment budget.

    Extraction is whole (text, PDF, Office); this is the single size authority
    for one attachment, and multi-file batches share the remaining budget so
    every file stays visible by name. Marks exactly where inline content was
    reduced so the model does not silently miss attachments.
    """
    text = text or ""
    if len(text) <= remaining:
        return text, remaining - len(text)

    name = os.path.basename(display_name or "attachment")
    if remaining < MIN_INLINE_ATTACHMENT_SLICE:
        return (
            f"\n\n[Attachment not shown inline: {name}. This message's "
            "attachments already filled the inline budget. Read the file with "
            "`read_file` (its path is in the uploaded-files list; use "
            "offset/limit to page).]",
            0,
        )
    marker = (
        f"\n\n[Attachment truncated: {name}. Only the first {remaining:,} "
        "characters are shown inline. Read the rest with `read_file` (its "
        "path is in the uploaded-files list; use offset/limit to page).]"
    )
    return text[:remaining] + marker, 0


_PAGE_MARK_RE = None


def _page_numbers(text: str) -> list[int]:
    """Page numbers from the ``[Page N text]`` / ``[Page N image k text]`` markers."""
    global _PAGE_MARK_RE
    if _PAGE_MARK_RE is None:
        import re
        _PAGE_MARK_RE = re.compile(r"\[Page (\d+) (?:text|image)")
    return [int(n) for n in _PAGE_MARK_RE.findall(text or "")]


def _doc_body_start(doc_id: str, body: str) -> int | None:
    """Offset of ``body`` inside a document's stored content, for paged reads."""
    if not doc_id or not body:
        return None
    try:
        from src.database import SessionLocal, Document
        db = SessionLocal()
        try:
            doc = db.query(Document).filter(Document.id == doc_id).first()
            content = (doc.current_content or "") if doc else ""
        finally:
            db.close()
        start = content.find(body[:200])
        return start if start >= 0 else None
    except Exception:
        return None


def _fit_pdf_inline(text: str, allowance: int, display_name: str, info: Dict[str, Any]) -> tuple[str, int]:
    """Fit a PDF attachment's inline text into ``allowance`` characters.

    Returns (text, characters used). When the PDF doesn't fit, the text is cut
    at a line boundary and ends with a note saying which pages are shown and
    exactly where to continue in the viewer document -- which holds every
    page -- with ``manage_documents``. The old note sent the model to "the
    document viewer", which itself held only the first 15,000 characters.
    """
    if len(text) <= allowance:
        return text, len(text)
    name = os.path.basename(display_name or "attachment")
    body = info.get("body") or ""
    doc_id = info.get("doc_id")
    base = info.get("doc_body_start")
    pages = _page_numbers(body)
    total_pages = max(pages) if pages else None
    body_at = text.find(body[:200]) if body else -1
    head = text[:body_at] if body_at >= 0 else ""
    how = (
        f"read it with `manage_documents` action=read document_id={doc_id}"
        + (f" offset={{offset}}" if base is not None else " (start at offset=0)")
        + " -- each read returns next_offset."
    )
    reserve = 420  # room for the note itself
    room = allowance - len(head) - reserve
    if body_at < 0 or room < MIN_INLINE_ATTACHMENT_SLICE:
        size = f"{total_pages} pages, " if total_pages else ""
        note = (
            f"\n\n[Not shown inline: {name} ({size}{len(body):,} characters). "
            "This message's attachments already filled the inline budget. "
            "Every page is in the viewer document: "
            + how.format(offset=base or 0) + "]"
        )
        kept = head.rstrip() + note
        return kept, min(allowance, len(kept))
    cut = body.rfind("\n", 0, room)
    if cut < int(room * 0.8):
        cut = room
    shown_pages = _page_numbers(body[:cut])
    last = max(shown_pages) if shown_pages else None
    span = f"pages 1-{last} of {total_pages}" if last and total_pages else f"the first {cut:,} characters"
    note = (
        f"\n\n[Showing {span} of {name} ({cut:,} of {len(body):,} characters). "
        "Every page is in the viewer document; to read on, "
        + how.format(offset=(base or 0) + cut) + "]"
    )
    kept = head + body[:cut] + note
    return kept, min(allowance, len(kept))


def _process_office_document(
    path: str,
    display_name: str,
    session_id: str | None = None,
    auto_opened_docs: list[Dict[str, Any]] | None = None,
    owner: str | None = None,
    out: dict | None = None,
) -> str:
    """Extract an Office/EPUB document to Markdown via the optional markitdown dep.

    Falls back to a friendly banner when markitdown is unavailable or finds no
    text, so a missing optional dependency never breaks the chat path. When a
    session_id is provided AND the extraction succeeded, the FULL text is also
    saved as a Document so the agent can page through it via
    `manage_documents action=read offset=…` when the inline budget trims it.

    When ``out`` is given, it receives "full_len" (chars of the whole
    extraction) and "doc_id" so callers can report honest ingestion numbers.
    """
    from src.markitdown_runtime import (
        is_markitdown_format,
        convert_to_markdown,
        load_markitdown,
    )

    if not is_markitdown_format(path):
        return "\n\n[Attached document file]"

    markdown = convert_to_markdown(path)
    if markdown and markdown.strip():
        title = os.path.splitext(os.path.basename(path))[0]
        if out is not None:
            out["full_len"] = len(markdown)

        # Persist the full extracted text as a Document. The agent's existing
        # manage_documents tool can then read past the inline cap with offset.
        doc_id = None
        if session_id:
            try:
                from src.office_doc import create_office_document
                doc_id = create_office_document(
                    session_id=session_id,
                    upload_id=os.path.basename(path),
                    title=title,
                    body_text=markdown,
                )
                if doc_id and auto_opened_docs is not None:
                    from src.database import SessionLocal, Document
                    _db = SessionLocal()
                    try:
                        _d = _db.query(Document).filter(Document.id == doc_id).first()
                        if _d:
                            auto_opened_docs.append({
                                "doc_id": _d.id,
                                "title": _d.title,
                                "language": _d.language,
                                "content": _d.current_content,
                                "version": _d.version_count,
                            })
                    finally:
                        _db.close()
            except Exception as e:
                logger.warning("Office auto-doc creation failed for %s: %s", path, e)
        if out is not None:
            out["doc_id"] = doc_id

        # Return the FULL markdown — no inline truncation here. The shared
        # inline attachment budget trims it (and records what was cut); the
        # old _truncate_inline(15k) cap fought the budget and hid most of a
        # large docx from big-context models even when the budget allowed it.
        return f"\n\n[Document content — {title}]:\n{markdown}"

    # No content: tell the user whether to install the optional dep or whether
    # the document simply had no extractable text.
    try:
        load_markitdown()
        return f"\n\n[Attached document: {display_name} — no extractable text found.]"
    except RuntimeError as exc:
        return f"\n\n[Attached document: {display_name} — {exc}]"


# Marker that _process_pdf prepends to extracted text.
_PDF_CONTENT_MARKER = "\n\n[PDF content]:"


def strip_pdf_content_marker(text: str) -> str:
    """Remove the leading ``[PDF content]:`` wrapper that ``_process_pdf`` adds.

    Uses ``str.removeprefix`` rather than ``str.lstrip(chars)``: ``lstrip``
    treats its argument as a *set of characters*, so ``lstrip("\\n[PDF content]:")``
    keeps chewing into the page text that follows the marker. For example
    ``"\\n\\n[PDF content]:\\n\\n[Page 1 text]:\\nto the board"`` would lose the
    leading "to" because 't' and 'o' are in the marker's character set.
    """
    return (text or "").removeprefix(_PDF_CONTENT_MARKER).strip()


def _load_vl_settings() -> dict:
    """Load admin settings from disk."""
    try:
        from src.settings import load_settings
        return load_settings()
    except Exception:
        return {}


def _resolve_vl_model(configured: str, owner: str | None = None) -> tuple:
    """Resolve the vision model to (url, model_id, headers).

    Uses admin-configured model if set, otherwise tries auto-detection
    of known vision-capable models across configured endpoints.
    """
    from src.ai_interaction import _resolve_model

    if configured:
        return _resolve_model(configured, owner=owner)

    # Auto-detect: try known vision-capable models in priority order
    candidates = [
        "gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini",
        "claude-sonnet-4-5-20250929", "claude-opus-4-20250514",
        "gemini-2.0-flash", "gemini-2.5-pro",
        "llava", "pixtral", "qwen2-vl",
    ]
    for candidate in candidates:
        try:
            return _resolve_model(candidate, owner=owner)
        except (ValueError, Exception):
            continue

    raise ValueError("No vision model available")


def analyze_image_with_vl_result(image_path: str, owner: str | None = None) -> dict:
    """Analyze an image and return both text and the model that produced it."""
    logger.info(f"Analyzing image with VL model: {image_path}")
    try:
        settings = _load_vl_settings()
        if not settings.get("vision_enabled", True):
            return {"text": "[Vision is disabled — enable it in Settings → Vision]", "model": ""}
        vl_model = settings.get("vision_model", "")

        try:
            url, model_id, headers = _resolve_vl_model(vl_model, owner=owner)
        except ValueError:
            return {"text": "[No vision model configured — set one in Settings → Vision]", "model": vl_model or ""}

        with open(image_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode("utf-8")

        ext = os.path.splitext(image_path)[1].lower()
        mime_map = {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png", ".gif": "gif", ".webp": "webp"}
        img_format = mime_map.get(ext, "jpeg")

        vl_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in detail"},
                    {"type": "image_url", "image_url": {"url": f"data:image/{img_format};base64,{img_data}"}},
                ],
            }
        ]
        # Vision-specific fallback chain (Settings → Vision → Fallbacks). A
        # downed vision endpoint can fall through to the next configured model
        # — same shape as task/chat but its own list (`vision_model_fallbacks`).
        try:
            from src.endpoint_resolver import resolve_vision_fallback_candidates
            _vl_candidates = [(url, model_id, headers)] + resolve_vision_fallback_candidates(owner=owner)
        except Exception:
            _vl_candidates = [(url, model_id, headers)]

        last_err = None
        for i, (_url, _model, _headers) in enumerate([c for c in _vl_candidates if c and c[0] and c[1]]):
            try:
                description = llm_call(_url, _model, vl_messages, headers=_headers, timeout=120)
                logger.info("VL analysis complete with model %s", _model)
                return {"text": description, "model": _model}
            except Exception as e:
                last_err = e
                tag = "primary" if i == 0 else "candidate"
                logger.warning(f"[vision fallback] {tag} {_model} failed ({type(e).__name__}); trying next")
                continue
        raise last_err if last_err else RuntimeError("No vision model endpoint configured")

    except Exception as e:
        logger.error(f"VL model unavailable: {e}")
        return {"text": "[VL model unavailable - image not analyzed]", "model": ""}


def analyze_image_with_vl(image_path: str, owner: str | None = None) -> str:
    """Analyze an image using the admin-configured Vision-Language model."""
    return analyze_image_with_vl_result(image_path, owner=owner).get("text", "")


def _classify_ingestion(fitted_text: str, original_len: int) -> str:
    """Classify a fitted attachment as 'full', 'partial' or 'omitted'.

    Reads the note markers the fitters append; falls back to a length
    comparison when a marker was lost (e.g. an unusual extraction path).
    """
    if "[Attachment not shown inline:" in fitted_text or "[Not shown inline:" in fitted_text:
        return "omitted"
    if "[Attachment truncated:" in fitted_text or "[Showing " in fitted_text:
        return "partial"
    # The fitters keep the note inside the allowance, so a fitted length far
    # below the original also means content was cut even if a marker was lost
    # (e.g. an unusual extraction path).
    if fitted_text and original_len and len(fitted_text) + 256 < original_len:
        return "partial"
    return "full"


def _ingestion_report(notices: list, context_tokens: int | None, budget: int) -> str:
    """Model-facing summary of what was inlined vs dropped from the uploads.

    The per-file fitters already leave a note next to each cut; this block
    consolidates the accounting so the model cannot mistake a message with a
    partial file for one that read everything, and says what to do about it.
    """
    lines = ["\n\n[Attachment ingestion report — read before answering]"]
    if context_tokens:
        lines.append(
            f"Inline budget for this message: {budget:,} characters "
            f"(50% of the {context_tokens:,}-token context window)."
        )
    else:
        lines.append(f"Inline budget for this message: {budget:,} characters.")
    for n in notices:
        name = n["name"]
        status = n["status"]
        if status == "full":
            lines.append(f"- {name}: SHOWN IN FULL ({n['inline_chars']:,} chars inline)")
        elif status == "omitted":
            lines.append(
                f"- {name}: NOT SHOWN — the inline budget was exhausted before "
                f"this file ({n['extracted_chars']:,} chars extracted). None of "
                "its body is in this conversation. Read it with `read_file` "
                "(path is in the uploaded-files list) before relying on its content."
            )
        else:
            lines.append(
                f"- {name}: PARTIAL — {n['inline_chars']:,} of {n['extracted_chars']:,} "
                "chars shown inline; the rest was NOT ingested. Continue reading "
                "with `read_file` (use offset/limit)"
                + (f" or `manage_documents` (document_id={n['doc_id']})" if n.get("doc_id") else "")
                + "."
            )
    lines.append(
        "Un-shown content is unread: do not answer as if you had seen it. "
        "Fetch it with the tools above if the answer needs it."
    )
    return "\n" + "\n".join(lines)


def build_user_content(
    text: str,
    attachment_ids: list[str] | None,
    upload_dir: str,
    upload_handler,
    session_id: str | None = None,
    auto_opened_docs: list[Dict[str, Any]] | None = None,
    owner: str | None = None,
    resolved_uploads: dict[str, Dict[str, Any]] | None = None,
    context_tokens: int | None = None,
    ingestion_notices: list | None = None,
) -> str | List[Dict[str, Any]]:
    """Build user content with attachments (text, images, audio, documents).

    If session_id is provided and an attached PDF contains AcroForm fields,
    a markdown Document is auto-created so the user can edit the form in the
    editor. When `auto_opened_docs` is supplied, an entry is appended for each
    such doc so the chat route can emit a `doc_update` SSE event and the
    frontend can switch to the new doc immediately.
    """
    content = [{"type": "text", "text": text}]
    inline_attachment_remaining = inline_attachment_budget(context_tokens)

    _attachment_ids = list(attachment_ids or [])
    for _att_index, fid in enumerate(_attachment_ids):
        upload_info = (resolved_uploads or {}).get(fid)
        if upload_info is None and hasattr(upload_handler, "resolve_upload"):
            upload_info = upload_handler.resolve_upload(fid, owner=owner)
        if upload_info is None:
            logger.warning(f"Attachment {fid} not found or not authorized")
            continue

        path = upload_info.get("path")
        if not path or not os.path.exists(path):
            logger.warning(f"Attachment {fid} path is missing")
            continue
        if hasattr(upload_handler, "_inside_upload_dir") and not upload_handler._inside_upload_dir(path):
            logger.warning(f"Attachment {fid} path is outside upload directory: {path}")
            continue
        if not hasattr(upload_handler, "_inside_upload_dir") and not upload_handler.inside_base_dir(path):
            logger.warning(f"Attachment {fid} path is outside base directory: {path}")
            continue

        _, ext = os.path.splitext(path.lower())
        mime = upload_info.get("mime") or mimetypes.guess_type(path)[0] or "application/octet-stream"
        display_name = upload_info.get("name") or upload_info.get("original_name") or path
        pdf_inline = None  # set by the PDF branch: where its full text lives
        _office_info = None  # set by the office branch: full length + doc_id

        if upload_handler.is_image_file(display_name, mime):
            try:
                with open(path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                # Extensionless uploads (e.g. a pasted screenshot) have no ext,
                # so fall back to the resolved MIME subtype rather than emitting
                # an invalid "data:image/;base64," with an empty subtype.
                image_format = ext[1:] or (mime.split("/", 1)[1] if mime.startswith("image/") else "png")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{image_format};base64,{encoded_string}"},
                })
            except Exception as e:
                logger.error(f"Failed to encode image {fid}: {e}")
                if content and content[0]["type"] == "text":
                    content[0]["text"] += "\n\n[Image attached but could not be processed]"
                else:
                    content.insert(0, {"type": "text", "text": "[Image attached but could not be processed]"})

        elif upload_handler.is_audio_file(display_name, mime):
            try:
                with open(path, "rb") as audio_file:
                    encoded_string = base64.b64encode(audio_file.read()).decode("utf-8")
                audio_format = ext[1:] or (mime.split("/", 1)[1] if mime.startswith("audio/") else "mpeg")
                content.append({
                    "type": "audio",
                    "audio": {"url": f"data:audio/{audio_format};base64,{encoded_string}"},
                })
            except Exception as e:
                logger.error(f"Failed to encode audio {fid}: {e}")
                if content and content[0]["type"] == "text":
                    content[0]["text"] += "\n\n[Audio attached but could not be processed]"
                else:
                    content.insert(0, {"type": "text", "text": "[Audio attached but could not be processed]"})

        elif upload_handler.is_document_file(display_name, mime):
            if mime == "application/pdf":
                extracted_text = None
                if session_id:
                    try:
                        from src.pdf_forms import has_form_fields, extract_fields
                        from src.pdf_form_doc import (
                            save_field_sidecar,
                            create_form_markdown_document,
                            create_plain_pdf_document,
                        )
                        title = os.path.splitext(os.path.basename(display_name))[0]
                        # Pull the PDF prose once — used as either intro_text
                        # (form path) or the doc body (plain path).
                        try:
                            pdf_body_text = strip_pdf_content_marker(_process_pdf(path, owner=owner, max_chars=None))
                        except Exception:
                            pdf_body_text = None

                        is_form = False
                        try:
                            is_form = has_form_fields(path)
                        except Exception as e:
                            logger.warning(f"PDF form detection failed for {path}: {e}")

                        # Inline the PDF body in the chat content too. Without
                        # this, the assistant only saw the "PDF attached"
                        # banner and had no idea what was inside. The full body
                        # goes in here; the shared inline budget trims it below
                        # (_fit_pdf_inline) and points at the viewer document,
                        # which holds every page, for the rest.
                        body_for_chat = (pdf_body_text or "").strip()
                        truncated_marker = ""

                        if is_form:
                            fields = extract_fields(path)
                            save_field_sidecar(path, fields)
                            doc_id = create_form_markdown_document(
                                session_id=session_id,
                                fields=fields,
                                upload_id=os.path.basename(path),
                                title=title,
                                # A form doc carries its fields after the intro and is
                                # sent whole while open, so keep the intro at the size
                                # it always had.
                                intro_text=(
                                    pdf_body_text[:15000] + "\n[PDF content truncated]"
                                    if pdf_body_text and len(pdf_body_text) > 15000
                                    else pdf_body_text
                                ),
                            )
                            if doc_id:
                                extracted_text = (
                                    f"\n\n[Form attached: {title} — {len(fields)} fields. "
                                    f"Opened in editor — edit the values there and use "
                                    f"the Export PDF button when done.]"
                                )
                                if body_for_chat:
                                    extracted_text += (
                                        f"\n\n[PDF content — {title}]:\n{body_for_chat}{truncated_marker}"
                                    )
                        else:
                            doc_id = create_plain_pdf_document(
                                session_id=session_id,
                                upload_id=os.path.basename(path),
                                title=title,
                                body_text=pdf_body_text,
                            )
                            if doc_id:
                                extracted_text = (
                                    f"\n\n[PDF attached: {title} — opened in document viewer.]"
                                )
                                if body_for_chat:
                                    extracted_text += (
                                        f"\n\n[PDF content — {title}]:\n{body_for_chat}{truncated_marker}"
                                    )

                        # Plain PDF docs hold every page, so the inline note can
                        # point at an exact offset. Form docs hold only the intro
                        # and then the fields; they fall back to read_file.
                        if doc_id and body_for_chat and not is_form:
                            pdf_inline = {
                                "doc_id": doc_id,
                                "body": body_for_chat,
                                "doc_body_start": _doc_body_start(doc_id, body_for_chat),
                            }

                        if doc_id and auto_opened_docs is not None:
                            from src.database import SessionLocal, Document
                            _db = SessionLocal()
                            try:
                                _d = _db.query(Document).filter(
                                    Document.id == doc_id
                                ).first()
                                if _d:
                                    auto_opened_docs.append({
                                        "doc_id": _d.id,
                                        "title": _d.title,
                                        "language": _d.language,
                                        "content": _d.current_content,
                                        "version": _d.version_count,
                                    })
                            finally:
                                _db.close()
                    except Exception as e:
                        logger.warning(f"PDF auto-doc creation failed for {path}: {e}")
                if extracted_text is None:
                    # max_chars=None: the inline budget is the single size
                    # authority — the old 15,000-char default here silently
                    # truncated PDFs whenever viewer-doc creation failed (or
                    # no session existed) and nothing flagged the loss.
                    extracted_text = _process_pdf(path, owner=owner, max_chars=None)
            elif mime.startswith("text/") or _is_text_file(path):
                extracted_text = _process_text_file(path)
            else:
                _office_info = {}
                extracted_text = _process_office_document(
                    path,
                    display_name,
                    session_id=session_id,
                    auto_opened_docs=auto_opened_docs,
                    owner=owner,
                    out=_office_info,
                )

            # Split what's left evenly over the attachments still to come, so a
            # multi-file upload shows the start of every file instead of all of
            # the first and none of the last. Unused share rolls forward.
            _allowance = inline_attachment_remaining // max(1, len(_attachment_ids) - _att_index)
            _original_len = (
                len(pdf_inline["body"]) if pdf_inline else len(extracted_text or "")
            )
            if pdf_inline:
                extracted_text, _used = _fit_pdf_inline(extracted_text, _allowance, display_name, pdf_inline)
            else:
                extracted_text, _ = _fit_inline_attachment_text(extracted_text, _allowance, display_name)
                # Charge what actually went inline (an omitted file costs only
                # its note), capped at this file's share.
                _used = min(len(extracted_text), _allowance)
            inline_attachment_remaining -= _used
            if ingestion_notices is not None:
                _status = _classify_ingestion(extracted_text, _original_len)
                _notice = {
                    "id": fid,
                    "name": os.path.basename(display_name or "attachment"),
                    "status": _status,
                    "extracted_chars": _original_len,
                    "inline_chars": len(extracted_text or ""),
                }
                if pdf_inline:
                    _pages = _page_numbers(pdf_inline["body"])
                    if _pages:
                        _notice["pages_total"] = max(_pages)
                    if _status == "partial":
                        _m = re.search(r"pages 1-(\d+) of", extracted_text or "")
                        if _m:
                            _notice["pages_inline"] = int(_m.group(1))
                    _notice["doc_id"] = pdf_inline.get("doc_id")
                elif _office_info is not None and _office_info.get("doc_id"):
                    # Office/EPUB continuations live in a viewer document, not
                    # in the upload dir — point the report at it.
                    _notice["doc_id"] = _office_info["doc_id"]
                ingestion_notices.append(_notice)
            if content and content[0]["type"] == "text":
                content[0]["text"] += extracted_text
            else:
                content.insert(0, {"type": "text", "text": extracted_text.lstrip()})
        else:
            if content and content[0]["type"] == "text":
                content[0]["text"] += "\n\n[Attached non-text file]"
            else:
                content.insert(0, {"type": "text", "text": "[Attached non-text file]"})

    # When anything was cut, tell the model explicitly, once, what it did and
    # did not receive — and how to fetch the rest.
    if ingestion_notices is not None and any(n["status"] != "full" for n in ingestion_notices):
        _report = _ingestion_report(
            ingestion_notices,
            context_tokens,
            inline_attachment_budget(context_tokens),
        )
        if content and content[0].get("type") == "text":
            content[0]["text"] += _report
        else:
            content.insert(0, {"type": "text", "text": _report.lstrip()})

    has_media = any(item.get("type") in ["image_url", "audio"] for item in content if isinstance(item, dict))
    if not has_media and content:
        combined_text = ""
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                combined_text += item.get("text", "")
        return combined_text.strip()

    return content
