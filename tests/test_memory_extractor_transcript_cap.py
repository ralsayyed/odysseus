"""Auto memory extraction caps each message in the transcript it sends.

A message carrying an attached PDF holds the whole file text after what the
user typed. Sent uncapped, one extraction prefilled ~60k tokens (128s on a
local server) and the user's next message waited for it.
"""
import asyncio
import tempfile

import src.event_bus
import src.llm_core
from src.memory import MemoryManager
from services.memory import memory_extractor
from services.memory.memory_extractor import extract_and_store

TYPED = "Summarise this record for me. I work on the Nicaragua v. Germany case."
PDF_TEXT = "[Page 1 text]:\n" + "The Court resumed its sitting. " * 7000  # ~210k chars


class _Session:
    owner = "alice"
    session_id = "sess-1"

    def get_context_messages(self):
        return [
            {"role": "user", "content": TYPED + "\n\n[PDF content]:\n" + PDF_TEXT},
            {"role": "assistant", "content": "It is the verbatim record of a public sitting."},
        ]


def test_attached_file_text_is_capped_but_typed_text_is_kept(monkeypatch):
    sent = []

    async def _fake_llm(url, model, messages, **kwargs):
        sent.append(messages)
        return "[]"

    monkeypatch.setattr(src.llm_core, "llm_call_async", _fake_llm)
    monkeypatch.setattr(src.event_bus, "fire_event", lambda *a, **k: None)

    with tempfile.TemporaryDirectory() as data_dir:
        asyncio.new_event_loop().run_until_complete(extract_and_store(
            _Session(), MemoryManager(data_dir), None,
            endpoint_url="http://x", model="m",
        ))

    transcript = sent[0][-1]["content"]
    assert TYPED in transcript
    assert "It is the verbatim record" in transcript
    assert len(transcript) < 2 * memory_extractor.MAX_MESSAGE_CHARS + 500
