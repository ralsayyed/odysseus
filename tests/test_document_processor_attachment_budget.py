from pathlib import Path


class _UploadHandler:
    def __init__(self, uploads):
        self.uploads = uploads

    def resolve_upload(self, fid, owner=None):
        return self.uploads.get(fid)

    def _inside_upload_dir(self, path):
        return True

    def is_image_file(self, display_name, mime):
        return False

    def is_audio_file(self, display_name, mime):
        return False

    def is_document_file(self, display_name, mime):
        return True


def _text_upload(tmp_path: Path, fid: str, body: str):
    path = tmp_path / f"{fid}.txt"
    path.write_text(body, encoding="utf-8")
    return {
        "path": str(path),
        "name": path.name,
        "mime": "text/plain",
    }


def _three_files(tmp_path):
    return {
        "a": _text_upload(tmp_path, "a", "alpha\n" + ("A" * 1000)),
        "b": _text_upload(tmp_path, "b", "bravo\n" + ("B" * 1000)),
        "c": _text_upload(tmp_path, "c", "charlie\n" + ("C" * 1000)),
    }


def test_multifile_inline_attachment_budget_keeps_later_files_visible(tmp_path, monkeypatch):
    import src.document_processor as dp

    monkeypatch.setattr(dp, "MAX_INLINE_ATTACHMENT_CHARS", 1200)
    monkeypatch.setattr(dp, "MIN_INLINE_ATTACHMENT_SLICE", 200)

    content = dp.build_user_content(
        "How many files do you see?",
        ["a", "b", "c"],
        str(tmp_path),
        _UploadHandler(_three_files(tmp_path)),
        owner="tester",
    )

    # The budget is split evenly across the files, so each one's start is
    # shown -- not all of the first and none of the rest.
    for name in ("a.txt", "b.txt", "c.txt"):
        assert f"=== File: {name} ===" in content
        assert f"Attachment truncated: {name}" in content
    # The note says how to read the rest.
    assert "read_file" in content
    assert len(content) < 2200


def test_files_are_named_with_a_way_to_read_them_when_nothing_fits(tmp_path, monkeypatch):
    import src.document_processor as dp

    monkeypatch.setattr(dp, "MAX_INLINE_ATTACHMENT_CHARS", 450)
    monkeypatch.setattr(dp, "MIN_INLINE_ATTACHMENT_SLICE", 200)

    content = dp.build_user_content(
        "How many files do you see?",
        ["a", "b", "c"],
        str(tmp_path),
        _UploadHandler(_three_files(tmp_path)),
        owner="tester",
    )

    for name in ("a.txt", "b.txt", "c.txt"):
        assert f"Attachment not shown inline: {name}" in content
    assert "=== File:" not in content
    assert "read_file" in content


def test_inline_attachment_budget_does_not_truncate_small_batches(tmp_path, monkeypatch):
    import src.document_processor as dp

    monkeypatch.setattr(dp, "MAX_INLINE_ATTACHMENT_CHARS", 5000)
    uploads = {
        "a": _text_upload(tmp_path, "a", "alpha"),
        "b": _text_upload(tmp_path, "b", "bravo"),
    }

    content = dp.build_user_content(
        "Summarize these.",
        ["a", "b"],
        str(tmp_path),
        _UploadHandler(uploads),
        owner="tester",
    )

    assert "=== File: a.txt ===" in content
    assert "=== File: b.txt ===" in content
    assert "Attachment truncated" not in content
    assert "not shown inline" not in content
