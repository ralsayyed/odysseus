"""Prefill and prompt-cache metrics from llama.cpp- and mlx-serve-shaped usage.

The two servers disagree on timings.prompt_n: llama.cpp counts only the
uncached tokens, mlx-serve the whole prompt. The final chunks below were
captured from each server; the mlx-serve one is a warm turn whose own
prefill_tokens_total counter moved by 31.
"""
import asyncio
import copy
import json

from src import llm_core


class _FakeResp:
    def __init__(self, lines):
        self._lines = lines
        self.status_code = 200

    async def aiter_lines(self):
        for ln in self._lines:
            yield ln

    async def aread(self):
        return b""


class _FakeStreamCtx:
    def __init__(self, lines):
        self._lines = lines

    async def __aenter__(self):
        return _FakeResp(self._lines)

    async def __aexit__(self, *a):
        return False


class _FakeClient:
    def __init__(self, lines):
        self._lines = lines

    def stream(self, method, url, **kw):
        return _FakeStreamCtx(self._lines)


def _drive(monkeypatch, lines):
    monkeypatch.setattr(llm_core, "_get_http_client", lambda: _FakeClient(lines))
    monkeypatch.setattr(llm_core, "_is_host_dead", lambda u: False)
    monkeypatch.setattr(llm_core, "note_model_activity", lambda *a, **k: None)
    monkeypatch.setattr(llm_core, "_clear_host_dead", lambda *a, **k: None)
    monkeypatch.setattr(llm_core, "_mark_host_dead", lambda *a, **k: False, raising=False)

    async def run():
        out = []
        async for chunk in llm_core.stream_llm(
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            "gpt-4o-test", [{"role": "user", "content": "hi"}],
            headers={"Authorization": "Bearer k"},
        ):
            out.append(chunk)
        return "".join(out)

    return asyncio.run(run())


def _usage_events(blob):
    events = []
    for ln in blob.split("\n"):
        ln = ln.strip()
        if not ln.startswith("data: ") or ln == "data: [DONE]":
            continue
        try:
            j = json.loads(ln[6:])
        except ValueError:
            continue
        if j.get("type") == "usage":
            events.append(j["data"])
    return events


def _stream(final):
    """An OpenAI-style stream as mlx-serve sends it: usage is null on every
    chunk except the last."""
    opening = {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}], "usage": None}
    token = {"choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": None}], "usage": None}
    finish = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": None}
    chunks = [opening, token, finish] + ([final] if final else [])
    return [f"data: {json.dumps(c)}" for c in chunks] + ["data: [DONE]"]


MLX_WARM = {
    "choices": [],
    "usage": {"prompt_tokens": 7050, "completion_tokens": 44, "total_tokens": 7094,
              "prompt_tokens_details": {"cached_tokens": 7019}},
    "timings": {"prompt_n": 7050, "cached_n": 7019, "prompt_ms": 471.206, "prompt_per_second": 65.789,
                "predicted_n": 44, "predicted_ms": 1433.51, "predicted_per_second": 30.694},
}

LLAMACPP_WARM = {
    "choices": [],
    "usage": {"prompt_tokens": 58, "completion_tokens": 32, "total_tokens": 90,
              "prompt_tokens_details": {"cached_tokens": 42}},
    "timings": {"cache_n": 42, "prompt_n": 16, "prompt_ms": 1088.631, "prompt_per_second": 14.697,
                "predicted_n": 32, "predicted_ms": 1267.063, "predicted_per_second": 24.466},
}


def test_mlx_serve_prefill_is_prompt_minus_cache(monkeypatch):
    events = _usage_events(_drive(monkeypatch, _stream(MLX_WARM)))
    assert len(events) == 1  # the null-usage chunks emit nothing
    u = events[0]
    assert u["cached_tokens"] == 7019
    assert u["prefill_tokens"] == 31  # not timings.prompt_n, which is 7,050 here
    assert u["prefill_ms"] == 471
    assert u["prefill_tps"] == 65.79
    assert u["gen_tps"] == 30.69


def test_mlx_serve_cached_n_is_used_without_prompt_tokens_details(monkeypatch):
    final = copy.deepcopy(MLX_WARM)
    del final["usage"]["prompt_tokens_details"]
    u = _usage_events(_drive(monkeypatch, _stream(final)))[0]
    assert u["cached_tokens"] == 7019
    assert u["prefill_tokens"] == 31


def test_llamacpp_prefill_matches_its_prompt_n(monkeypatch):
    u = _usage_events(_drive(monkeypatch, _stream(LLAMACPP_WARM)))[0]
    assert u["cached_tokens"] == 42
    assert u["prefill_tokens"] == 16 == LLAMACPP_WARM["timings"]["prompt_n"]


def test_llamacpp_cache_n_is_used_without_prompt_tokens_details(monkeypatch):
    final = copy.deepcopy(LLAMACPP_WARM)
    del final["usage"]["prompt_tokens_details"]
    u = _usage_events(_drive(monkeypatch, _stream(final)))[0]
    assert u["cached_tokens"] == 42
    assert u["prefill_tokens"] == 16


def test_null_usage_chunks_emit_no_usage_event(monkeypatch):
    # A server that never reports usage must leave chat mode free to fall back
    # to its own estimate instead of recording all-zero metrics.
    assert _usage_events(_drive(monkeypatch, _stream(None))) == []


def _final_metrics(**overrides):
    from src.agent_loop import _compute_final_metrics
    args = dict(
        messages=[{"role": "user", "content": "x" * 100}], full_response="ok",
        total_duration=70.0, time_to_first_token=67.0, context_length=262144,
        real_input_tokens=28921, real_output_tokens=1, has_real_usage=True,
        tool_events=[], round_texts=[], model="m",
        last_round_input_tokens=28921, request_context_tokens=11300,
    )
    args.update(overrides)
    return _compute_final_metrics(**args)


def test_context_percent_uses_the_servers_prompt_count_when_it_reports_cache():
    # Live agent turn on mlx-serve: the estimate said 11,300 tokens (4.3%);
    # the server's own prompt_tokens was 28,921 (11.0%).
    m = _final_metrics(backend_prefill_tokens=28921, backend_cached_tokens=0, backend_prefill_ms=66978)
    assert m["request_context_tokens"] == 28921
    assert m["context_percent"] == 11.0


def test_context_percent_keeps_the_estimate_without_cache_accounting():
    # Providers whose input count may be cache-adjusted report no cached_tokens
    # mapping, and keep the assembled-prompt estimate as before.
    m = _final_metrics()
    assert m["request_context_tokens"] == 11300


def test_turn_prefill_rate_comes_from_the_turns_totals():
    # Matches the live run: 28,921 tokens in 66,978 ms.
    m = _final_metrics(backend_prefill_tokens=28921, backend_cached_tokens=0,
                       backend_prefill_ms=66978, backend_prefill_tps=12.0)
    assert m["prefill_ms"] == 66978
    assert m["prefill_tps"] == 431.8  # not the last round's 12.0
