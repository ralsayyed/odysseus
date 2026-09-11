"""Live prefill progress: src/prefill_progress.py and its plumbing in llm_core.

Figures in comments come from measurements against mlx-serve with --metrics.
"""
import asyncio
import gc
import inspect
import time

import pytest

from src import llm_core
from src import prefill_progress as pp


def run(coro):
    return asyncio.run(coro)


def lone(live):
    return {"requests_running": 1, "requests_waiting": 0, "requests_prefilling": 1, "prefill_tokens_live": live}


# -- attribution --------------------------------------------------------------

@pytest.mark.parametrize("gauges, expected", [
    (lone(4096), 4096),
    # Another request in flight: the gauge can't be attributed. This exact
    # state made a 15-token prompt report 16,384 tokens "processed".
    ({"requests_running": 2, "requests_waiting": 1, "requests_prefilling": 1, "prefill_tokens_live": 16384}, None),
    ({"requests_running": 2, "requests_waiting": 0, "requests_prefilling": 1, "prefill_tokens_live": 4096}, None),
    ({"requests_running": 1, "requests_prefilling": 0, "prefill_tokens_live": 0}, None),
    # Prefilling but not yet through the first 4096-token step.
    (lone(0), None),
    ({}, None),
    (None, None),
])
def test_progress_only_trusted_for_a_lone_request(gauges, expected):
    assert pp.parse_progress(gauges) == expected


# -- which servers get probed -------------------------------------------------

@pytest.mark.parametrize("origin, expected", [
    ("http://100.64.161.47:8099", True),   # Tailscale / CGNAT
    ("http://127.0.0.1:7000", True),
    ("http://192.168.1.20:8080", True),
    ("http://[::1]:8099", True),
    ("https://8.8.8.8", False),
    ("", False),
])
def test_is_local_origin(origin, expected):
    assert run(pp.is_local_origin(origin)) is expected


def test_public_hosts_are_never_probed():
    class Client:
        async def get(self, url, **kw):
            raise AssertionError(f"probed {url}")

    pp._SUPPORT_CACHE.clear()
    assert run(pp.supported(Client(), "https://8.8.8.8")) is False


# -- denominator --------------------------------------------------------------

def test_prompt_text_includes_tool_schemas():
    # 16 tools measured 2,968 real prompt tokens against 263 from the messages
    # alone; the schemas have to be counted.
    tools = [{"type": "function", "function": {"name": "read_email", "parameters": {}}}]
    text = pp.prompt_text([{"role": "user", "content": "hi"}], tools)
    assert "hi" in text and "read_email" in text


def test_prompt_text_reads_text_blocks_of_multimodal_messages():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {}}]}]
    assert pp.prompt_text(msgs) == "look"


# -- stream_with_progress -----------------------------------------------------

class Lines:
    """Async line source built from (delay_seconds, line) pairs."""

    def __init__(self, items):
        self.items = list(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.items:
            raise StopAsyncIteration
        delay, line = self.items.pop(0)
        await asyncio.sleep(delay)
        return line


def test_first_token_is_not_held_behind_an_in_flight_poll():
    async def slow_poll():
        await asyncio.sleep(0.9)
        return {"processed": 1, "total": 2}

    async def main():
        t0 = time.monotonic()
        async for kind, _ in pp.stream_with_progress(Lines([(0.5, "tok")]), slow_poll, 0.2):
            if kind == "line":
                return time.monotonic() - t0

    # The poll starts at 0.2s and ends at 1.1s; the token is ready at 0.5s.
    # Awaiting the poll inline delivered it at ~1.1s.
    assert run(main()) < 0.75


def test_progress_is_relayed_while_the_stream_is_idle():
    count = iter(range(1, 100))

    async def poll():
        return next(count)

    async def main():
        return [x async for x in pp.stream_with_progress(Lines([(0.35, "tok")]), poll, 0.1)]

    out = run(main())
    assert out[-1] == ("line", "tok")
    assert len(out) >= 2 and all(kind == "progress" for kind, _ in out[:-1])


def test_should_poll_false_skips_polling():
    calls = []

    async def poll():
        calls.append(1)
        return 1

    async def main():
        return [x async for x in pp.stream_with_progress(Lines([(0.3, "a")]), poll, 0.05, lambda: False)]

    assert run(main()) == [("line", "a")]
    assert calls == []


def test_without_a_poll_lines_are_relayed_unchanged():
    async def main():
        return [x async for x in pp.stream_with_progress(Lines([(0, "a"), (0, "b")]), None, pp.POLL_INTERVAL)]

    assert run(main()) == [("line", "a"), ("line", "b")]


def test_abandoned_stream_leaves_no_unretrieved_task_exception():
    class ClosesUnderneath:
        closed = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            while not self.closed:
                await asyncio.sleep(0.01)
            raise RuntimeError("stream closed")  # what httpx raises once the response closes

    async def main():
        reports = []
        asyncio.get_running_loop().set_exception_handler(lambda loop, ctx: reports.append(ctx.get("message")))
        src = ClosesUnderneath()

        async def poll():
            return 1

        gen = pp.stream_with_progress(src, poll, 0.05)
        async for _ in gen:
            break              # Stop / disconnect / agent round deadline
        src.closed = True      # the caller's `async with client.stream` closes r
        await asyncio.sleep(0.1)
        del gen
        for _ in range(3):
            gc.collect()
            await asyncio.sleep(0.05)
        return reports

    assert "Task exception was never retrieved" not in run(main())


# -- PrefillTracker -----------------------------------------------------------

class Resp:
    def __init__(self, status, body):
        self.status_code, self._body = status, body

    def json(self):
        return self._body


class FakeServer:
    """Stands in for mlx-serve: each GET /metrics.json returns the next gauges."""

    def __init__(self, gauges, metrics=True, tokens=20000):
        self.gauges, self.metrics, self.tokens, self.tokenized = list(gauges), metrics, tokens, 0

    async def get(self, url, **kw):
        if not self.metrics:
            return Resp(503, {})
        g = self.gauges.pop(0) if len(self.gauges) > 1 else self.gauges[0]
        return Resp(200, {"gauges": g})

    async def post(self, url, json=None, **kw):
        self.tokenized += 1
        return Resp(200, {"tokens": list(range(self.tokens))})


def test_tracker_reports_forward_progress_against_the_tokenized_total():
    pp._SUPPORT_CACHE.clear()
    # First read is the capability probe, then one read per poll.
    server = FakeServer([{}, lone(4096), lone(4096), lone(8192)], tokens=20000)

    async def main():
        t = pp.PrefillTracker(server, "http://127.0.0.1:8099", "prompt", {})
        return [await t.poll() for _ in range(3)]

    assert run(main()) == [{"processed": 4096, "total": 20000}, None, {"processed": 8192, "total": 20000}]


def test_tracker_goes_quiet_when_the_server_has_no_metrics():
    pp._SUPPORT_CACHE.clear()
    server = FakeServer([{}], metrics=False)

    async def main():
        t = pp.PrefillTracker(server, "http://127.0.0.1:8099", "prompt", {})
        return await t.poll(), t.should_poll(), server.tokenized

    assert run(main()) == (None, False, 0)


def test_tracker_stop_ends_polling():
    pp._SUPPORT_CACHE.clear()

    async def main():
        t = pp.PrefillTracker(FakeServer([{}]), "http://127.0.0.1:8099", "prompt", {})
        t.stop()
        return t.should_poll()

    assert run(main()) is False


# -- llm_core plumbing --------------------------------------------------------

def test_progress_is_opt_in():
    # Background llm_call_async users never pass it, so they skip the probe,
    # /tokenize and polling.
    for fn in (llm_core.stream_llm, llm_core._stream_llm_inner):
        assert inspect.signature(fn).parameters["prefill_progress"].default is False


def test_fallback_forwards_the_opt_in(monkeypatch):
    seen = {}

    async def fake_stream(url, model, messages, **kw):
        seen.update(kw)
        yield 'data: {"delta": "x"}\n\n'
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(llm_core, "stream_llm", fake_stream)

    async def main():
        return [c async for c in llm_core.stream_llm_with_fallback(
            [("u", "m", {})], [{"role": "user", "content": "hi"}], prefill_progress=True)]

    run(main())
    assert seen.get("prefill_progress") is True


def test_fallback_passes_progress_through_immediately_without_committing(monkeypatch):
    prog = 'data: {"type": "prefill_progress", "data": {"processed": 4096, "total": 9000}}\n\n'

    async def fake_stream(url, model, messages, **kw):
        if model == "primary":
            yield prog
            yield 'event: error\ndata: {"status": 500, "text": "boom"}\n\n'
        else:
            yield 'data: {"delta": "hello"}\n\n'
            yield "data: [DONE]\n\n"

    monkeypatch.setattr(llm_core, "stream_llm", fake_stream)

    async def main():
        return [c async for c in llm_core.stream_llm_with_fallback(
            [("u1", "primary", {}), ("u2", "backup", {})], [{"role": "user", "content": "hi"}])]

    chunks = run(main())
    assert chunks[0] == prog                          # not buffered until output
    assert any('"fallback"' in c for c in chunks)     # the primary still failed over
    assert any('"delta": "hello"' in c for c in chunks)
