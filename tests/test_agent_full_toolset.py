"""Opt-in full tool set on local endpoints (agent_full_toolset_local).

Local servers (mlx-serve, llama.cpp) reuse a cached prompt only up to its first
changed token. The start of the agent prompt is built from the offered tools,
so with the setting on the tool list is fixed for every turn; a turn's
deliberate removals are refused when the tool is called instead of being
dropped from the list (dropping them made the first follow-up after a PDF
upload re-read the whole conversation).
"""
import pytest

from src import agent_loop as al
from src.settings import DEFAULT_SETTINGS

BASE = {"manage_memory", "ask_user", "update_plan"}  # tool_index.ALWAYS_AVAILABLE
SCHEMA_NAMES = {s["function"]["name"] for s in al.FUNCTION_TOOL_SCHEMAS}


class FakeMcp:
    def get_all_tools(self):
        return [
            {"qualified_name": "mcp__builtin_browser__browser_navigate", "server_id": "builtin_browser"},
            {"qualified_name": "mcp__email__read_email", "server_id": "email"},
            {"server_id": "broken"},  # no qualified_name: ignored
        ]


@pytest.fixture(autouse=True)
def image_gen(monkeypatch):
    settings = {"image_gen_enabled": True}
    monkeypatch.setattr(al, "get_setting", lambda key, default=None: settings.get(key, default))
    return settings


def test_off_by_default():
    assert DEFAULT_SETTINGS["agent_full_toolset_local"] is False


def test_the_set_is_the_same_every_turn():
    assert al._full_tool_set(FakeMcp()) == al._full_tool_set(FakeMcp())


def test_the_same_set_builds_a_byte_identical_prompt():
    full = al._full_tool_set(None)
    forward, backward = set(sorted(full)), set(sorted(full, reverse=True))
    for compact in (True, False):
        assert al._assemble_prompt(forward, set(), compact=compact) == al._assemble_prompt(backward, set(), compact=compact)


def test_every_schema_plus_mcp_tools_are_offered():
    full = al._full_tool_set(FakeMcp())
    assert SCHEMA_NAMES <= full
    assert {"mcp__builtin_browser__browser_navigate", "mcp__email__read_email"} <= full


def test_admin_turns_do_not_change_the_set():
    # _ADMIN_TOOLS is unioned in when admin intent fires; already being in the
    # full set means those turns keep the same prompt start.
    assert al._ADMIN_TOOLS & SCHEMA_NAMES <= al._full_tool_set(None)


def test_user_facing_tools_retrieval_can_offer_are_present():
    # search_chats and manage_tasks sit in _ADMIN_SCHEMA_NAMES; leaving them
    # out made "find my earlier chat" and reminders unreachable.
    assert {"search_chats", "manage_tasks"} <= al._full_tool_set(None)


def test_image_generation_follows_its_setting(image_gen):
    if "generate_image" in SCHEMA_NAMES:
        assert "generate_image" in al._full_tool_set(None)
    image_gen["image_gen_enabled"] = False
    assert "generate_image" not in al._full_tool_set(None)


# -- a turn's removals are refused at call time, with a reason ---------------

def test_open_document_turns_refuse_file_tools_and_point_at_the_document():
    excluded = dict.fromkeys({"read_file", "bash"}, al._EXCLUSION_HINTS["document"])
    blocks = al._turn_tool_blocks(BASE | {"edit_document"}, excluded)
    assert set(blocks) == {"read_file", "bash"}
    assert "manage_documents" in blocks["read_file"]


def test_an_open_email_draft_refuses_fetching_the_email():
    excluded = dict.fromkeys({"read_email", "list_emails"}, al._EXCLUSION_HINTS["email_draft"])
    blocks = al._turn_tool_blocks(BASE, excluded)
    assert "edit the draft" in blocks["read_email"]


def test_a_contact_save_refuses_memory_and_names_manage_contact():
    # Selection drops manage_memory on purpose for a contact save.
    blocks = al._turn_tool_blocks({"ask_user", "update_plan", "manage_contact"}, {})
    assert "manage_contact" in blocks["manage_memory"]


def test_nothing_is_refused_on_an_ordinary_turn():
    assert al._turn_tool_blocks(BASE | {"web_search"}, {}) == {}


def test_no_selection_refuses_no_always_available_tools():
    # Retrieval unavailable (_relevant_tools is None) is not a deliberate drop.
    assert al._turn_tool_blocks(None, {}) == {}
