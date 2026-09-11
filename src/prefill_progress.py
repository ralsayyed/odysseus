"""Live prefill progress for mlx-serve endpoints.

mlx-serve started with ``--metrics`` publishes server-wide gauges at
``/metrics.json``. While a cold prefill runs, ``requests_prefilling`` reads 1
and ``prefill_tokens_live`` climbs in ubatch-sized steps (4096 by default), so a
20k prompt yields ~5 discrete updates. The server never publishes the target
count mid-request, so the denominator comes from ``POST /tokenize``, which
accepts plain text only -- no messages, no tools, no template endpoint.

Measured behaviours this module is built around:

* The gauges are **server-wide**. ``requests_running`` counts every in-flight
  request, ours included, so ``== 1`` is the only state where the prefill gauge
  is provably ours. Gating on ``requests_prefilling == 1`` alone let a 15-token
  prompt queued behind another client's 20k prefill report that prefill as its
  own (gauges read ``running=2, waiting=1, prefilling=1``).
* Tool schemas travel outside the messages but dominate agent prompts: 16 tools
  measured 2,968 real prompt tokens against 263 from the messages alone.
  Tokenizing ``messages + json.dumps(tools)`` lands within ~5% of the server's
  count; the remainder is chat-template scaffolding.
* ``requests_prefilling`` takes ~2s to flip to 1 on a cold prefill, and a
  prefix-cache hit answers in ~0.4s, so polling starts only once the stream has
  been idle for ``POLL_INTERVAL``. Warm turns finish before the first poll and
  never show a bar.
* The final sub-ubatch chunk is never published, so progress stops short of
  the total (~80% on a 20k prompt); the first token is the real completion.
* Polls answer in ~20ms even mid-prefill (40ms max over 172 samples), but the
  first token must never wait on one, so polls run alongside the stream read.

Everything is best-effort: any failure means no progress for that request, and
lines are relayed exactly as they would be without this module.
"""

import asyncio
import ipaddress
import json
import logging
import time
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

# Per-origin capability cache: origin -> (supported, checked_at). Re-probed
# every few minutes so restarting the server with --metrics is picked up
# without restarting Odysseus, and a server that loses the flag stops being
# polled.
_SUPPORT_CACHE: Dict[str, Tuple[bool, float]] = {}
_SUPPORT_TTL = 300.0

# Short timeouts throughout: this runs while someone is waiting, and progress
# must never be the reason a reply is late.
_PROBE_TIMEOUT = 2.0
_POLL_TIMEOUT = 2.0
_TOKENIZE_TIMEOUT = 5.0

POLL_INTERVAL = 1.0  # the gauge moves in 4096-token steps (~9s apart); 1s is plenty


def origin_of(url: str) -> str:
    """Return scheme://host:port for a completions URL."""
    parts = urlsplit(url or "")
    if not parts.scheme or not parts.netloc:
        return ""
    return f"{parts.scheme}://{parts.netloc}"


def _is_local_ip(addr: str) -> bool:
    try:
        # is_global is False for private, loopback and link-local space, and
        # for the 100.64.0.0/10 CGNAT range Tailscale hands out.
        return not ipaddress.ip_address(addr.split("%", 1)[0]).is_global
    except ValueError:
        return False


async def is_local_origin(origin: str) -> bool:
    """True when every address the origin resolves to is non-public.

    Only self-hosted servers expose /metrics.json and /tokenize. Probing a
    cloud API for them would send its key to paths it never asked for, so
    public hosts are ruled out before any request is made.
    """
    host = urlsplit(origin or "").hostname
    if not host:
        return False
    try:
        ipaddress.ip_address(host)
        return _is_local_ip(host)
    except ValueError:
        pass
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(host, None)
    except OSError:
        return False
    addrs = {info[4][0] for info in infos}
    return bool(addrs) and all(_is_local_ip(a) for a in addrs)


async def supported(client, origin: str, headers: Optional[Dict] = None) -> bool:
    """True if this origin serves mlx-serve's /metrics.json gauges.

    A server without ``--metrics`` answers 503 ("metrics not enabled") and
    anything that isn't mlx-serve 404s, so one probe separates all three
    cases. Cached per origin for ``_SUPPORT_TTL``.
    """
    if not origin:
        return False
    hit = _SUPPORT_CACHE.get(origin)
    if hit and (time.monotonic() - hit[1]) < _SUPPORT_TTL:
        return hit[0]
    ok = False
    try:
        if await is_local_origin(origin):
            r = await client.get(f"{origin}/metrics.json", headers=headers or {}, timeout=_PROBE_TIMEOUT)
            ok = r.status_code == 200 and isinstance((r.json() or {}).get("gauges"), dict)
    except Exception:
        ok = False
    _SUPPORT_CACHE[origin] = (ok, time.monotonic())
    if ok:
        logger.info("Prefill progress available at %s", origin)
    return ok


def prompt_text(messages: List[Dict], tools: Optional[List[Dict]] = None) -> str:
    """Flatten a request to the text whose token count approximates its prompt.

    Tool schemas are sent beside the messages, not in them, yet dominate agent
    prompts; leaving them out undercounted a 16-tool request ~10x. With them,
    the count runs ~5% under the server's (chat-template scaffolding).
    """
    out = []
    for m in messages or []:
        c = m.get("content")
        if isinstance(c, str):
            out.append(c)
        elif isinstance(c, list):
            # Multimodal content blocks: only the text parts are tokenized here.
            for part in c:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    out.append(part["text"])
    if tools:
        out.append(json.dumps(tools))
    return "\n".join(out)


async def token_count(client, origin: str, text: str, headers: Optional[Dict] = None) -> Optional[int]:
    """Denominator via POST /tokenize. Returns None if unavailable."""
    if not origin or not text:
        return None
    try:
        r = await client.post(
            f"{origin}/tokenize",
            json={"content": text},
            headers=headers or {},
            timeout=_TOKENIZE_TIMEOUT,
        )
        if r.status_code != 200:
            return None
        # mlx-serve returns the token IDs themselves, not a count.
        toks = (r.json() or {}).get("tokens")
        return len(toks) if isinstance(toks, list) else None
    except Exception:
        return None


def parse_progress(gauges) -> Optional[int]:
    """Tokens prefilled so far for *our* request, or None if that can't be known.

    ``requests_running`` includes us, so 1 means nobody else is in flight and
    the prefill gauge is ours. Any other state -- someone queued behind us, us
    queued behind them, two batched together -- can't be attributed, and
    showing nothing beats showing another request's progress.
    """
    if not isinstance(gauges, dict):
        return None
    if gauges.get("requests_running") != 1 or gauges.get("requests_prefilling") != 1:
        return None
    live = gauges.get("prefill_tokens_live")
    # 0 while still inside the first ubatch step.
    return live if isinstance(live, int) and live > 0 else None


async def read_progress(client, origin: str, headers: Optional[Dict] = None) -> Optional[int]:
    """One gauge read, interpreted by :func:`parse_progress`."""
    try:
        r = await client.get(f"{origin}/metrics.json", headers=headers or {}, timeout=_POLL_TIMEOUT)
        if r.status_code != 200:
            return None
        return parse_progress((r.json() or {}).get("gauges"))
    except Exception:
        return None


def _retrieve(task: asyncio.Future) -> None:
    """Done-callback marking a task's exception as seen.

    When the consumer abandons the stream (Stop, disconnect, agent round
    deadline) the pending read is left behind, then fails once the caller's
    ``async with client.stream`` closes the response. Without this, asyncio
    logs "Task exception was never retrieved" for every such stream.
    """
    if not task.cancelled():
        task.exception()


class PrefillTracker:
    """Per-request prefill progress, shaped for :func:`stream_with_progress`.

    Setup (capability probe + denominator) starts at once as a background task
    rather than before the request, so the fast path never waits on it: no
    poll happens until the stream has been idle for ``POLL_INTERVAL``, and a
    warm turn answers well inside that.
    """

    def __init__(self, client, origin: str, prompt: str, headers: Optional[Dict] = None):
        self._client = client
        self._origin = origin
        self._headers = headers or {}
        self._last = 0
        self.stopped = False
        self._setup = asyncio.ensure_future(self._arm(prompt))
        self._setup.add_done_callback(_retrieve)

    async def _arm(self, prompt: str) -> Optional[int]:
        try:
            if not await supported(self._client, self._origin, self._headers):
                return None
            return await token_count(self._client, self._origin, prompt, self._headers)
        except Exception:
            return None

    def should_poll(self) -> bool:
        return not self.stopped

    def stop(self) -> None:
        """Output has started: prefill is over, and any later pause is decode."""
        self.stopped = True

    async def poll(self) -> Optional[Dict[str, int]]:
        total = await self._setup
        if not total:
            self.stopped = True  # no metrics or no denominator: stay quiet
            return None
        live = await read_progress(self._client, self._origin, self._headers)
        if live is None or live <= self._last:
            return None
        self._last = live
        return {"processed": live, "total": total}


async def stream_with_progress(aiter, poll, interval: Optional[float], should_poll=None):
    """Relay ``aiter`` as ``("line", str)``; while it is idle, poll for progress.

    After the stream has been silent for ``interval`` seconds, ``poll()`` runs
    and non-None results are relayed as ``("progress", value)``. The poll runs
    *alongside* the pending read rather than blocking it, so a line arriving
    mid-poll is yielded at once -- awaiting the poll inline held the first
    token back by the whole poll duration. ``should_poll`` is consulted before
    each poll; returning False skips it.

    With ``poll`` or ``interval`` None, lines are relayed with no tasks at all,
    so callers without progress keep their original behaviour.
    """
    it = aiter.__aiter__()
    if poll is None or interval is None:
        async for line in it:
            yield ("line", line)
        return

    loop = asyncio.get_running_loop()

    def spawn(awaitable):
        task = asyncio.ensure_future(awaitable)
        task.add_done_callback(_retrieve)
        return task

    pending = spawn(it.__anext__())
    poll_task = None
    next_poll = loop.time() + interval
    try:
        while True:
            waiting = {pending} if poll_task is None else {pending, poll_task}
            timeout = None if poll_task is not None else max(0.0, next_poll - loop.time())
            done, _ = await asyncio.wait(waiting, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
            if pending in done:
                try:
                    line = pending.result()
                except StopAsyncIteration:
                    return
                yield ("line", line)
                pending = spawn(it.__anext__())
                next_poll = loop.time() + interval
                continue
            if poll_task is not None:
                if poll_task in done:
                    value = None
                    if not poll_task.cancelled() and poll_task.exception() is None:
                        value = poll_task.result()
                    poll_task = None
                    next_poll = loop.time() + interval
                    if value is not None:
                        yield ("progress", value)
                continue
            # Idle for a full interval with no poll running.
            if should_poll is None or should_poll():
                poll_task = spawn(poll())
            else:
                next_poll = loop.time() + interval
    finally:
        for task in (pending, poll_task):
            if task is not None and not task.done():
                task.cancel()
