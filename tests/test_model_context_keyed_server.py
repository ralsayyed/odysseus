"""Context-window lookup against a local server that requires an API key.

mlx-serve started with --api-key answers /slots and /v1/models with 401 when
asked without one. The unauthenticated lookup then fell back to the name table,
which maps any "qwen3*" id to 131072 -- half the 262144 the server reports.
"""
import pytest

from src import model_context as mc

URL = "http://100.64.161.47:8099/v1"
SERVED = "Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit"


class _Resp:
    def __init__(self, status, body=None):
        self.status_code = status
        self.is_success = 200 <= status < 300
        self._body = body or {}

    def json(self):
        return self._body


def _keyed_server(url, headers=None, timeout=None, **kw):
    if (headers or {}).get("Authorization") != "Bearer k":
        return _Resp(401)
    if url.endswith("/models"):
        return _Resp(200, {"data": [{"id": SERVED, "context_length": 262144}]})
    return _Resp(404)  # mlx-serve has no /slots


@pytest.fixture
def local_endpoint(monkeypatch):
    monkeypatch.setattr(mc, "is_local_endpoint", lambda url: True)
    monkeypatch.setattr(mc, "_configured_endpoint_kind", lambda url: "local")
    monkeypatch.setattr(mc.httpx, "get", _keyed_server)
    mc._context_cache.clear()


def test_lookup_sends_the_endpoint_key_and_reads_the_real_window(monkeypatch, local_endpoint):
    monkeypatch.setattr(mc, "_endpoint_headers", lambda url: {"Authorization": "Bearer k"})
    assert mc.get_context_length(URL, SERVED) == 262144


def test_stale_model_id_on_a_single_model_local_server(monkeypatch, local_endpoint):
    # The endpoint still lists the id from before the backend swap; the
    # server answers it with its one loaded model.
    monkeypatch.setattr(mc, "_endpoint_headers", lambda url: {"Authorization": "Bearer k"})
    assert mc.get_context_length(URL, "Qwen3.8-Flash-Next") == 262144


def test_without_a_key_the_name_table_wins(monkeypatch, local_endpoint):
    # Why the key matters: a 401 leaves only the name table, which halves it.
    monkeypatch.setattr(mc, "_endpoint_headers", lambda url: {})
    assert mc.get_context_length(URL, SERVED) == 131072


def test_single_model_fallback_is_local_only(monkeypatch, local_endpoint):
    # A cloud catalog listing one unrelated model must not lend its window.
    monkeypatch.setattr(mc, "is_local_endpoint", lambda url: False)
    monkeypatch.setattr(mc, "_configured_endpoint_kind", lambda url: None)
    monkeypatch.setattr(mc, "_endpoint_headers", lambda url: {"Authorization": "Bearer k"})
    assert mc.get_context_length(URL, "some-other-model") == mc.DEFAULT_CONTEXT
