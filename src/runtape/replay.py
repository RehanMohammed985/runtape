"""Replay a recorded run through your real agent code.

    with runtape.replay("traces/run.jsonl") as rp:
        client = rp.wrap(Anthropic())

        @rp.tool
        def lookup_order(order_id): ...   # not executed: returns the recorded result

        run_agent(client)

Model replies and tool results come from the recording, so the agent runs
offline, for free, with no side effects, and the same way every time: set a
breakpoint anywhere in your code and step through a production failure.

When your code sends a request that differs from the recorded one (you changed
a prompt, a tool, the message handling), replay stops at the first difference
and tells you exactly what changed. With on_diverge="live" it switches to the
real model from that point instead, so you can see what your change does.
"""
from __future__ import annotations

import functools
import inspect
import json
import time
from dataclasses import dataclass
from typing import Any, Callable

from .recorder import Recorder, _bind, _safe_signature
from .serialize import to_jsonable
from .trace import Event, Trace


class ReplayDiverged(RuntimeError):
    def __init__(self, divergence: "Divergence"):
        self.divergence = divergence
        super().__init__(divergence.describe())


@dataclass
class Divergence:
    kind: str  # model_call | tool_call | extra_call
    step: int  # 1-based count of calls of this kind at the point of divergence
    recorded_event: int | None
    field: str  # what differs: model, system, tools, messages[3], arguments, name, ...
    recorded: Any = None
    now: Any = None

    def describe(self) -> str:
        where = f" (recorded #{self.recorded_event})" if self.recorded_event is not None else ""
        if self.kind == "extra_call" or self.field == "call":
            head = f"Replay diverged at {self.kind.replace('_', ' ')} {self.step}: the recording has no call here."
        else:
            head = f"Replay diverged at {self.kind.replace('_', ' ')} {self.step}{where}: {self.field} differs."
        return f"{head}\n  recorded: {_short(self.recorded)}\n  now:      {_short(self.now)}"


def _restore(value: Any, tag: str | None) -> Any:
    """Rebuild a recorded tool result as the type the tool originally returned, when possible."""
    if not tag or value is None:
        return value
    try:
        if tag == "tuple":
            return tuple(value)
        if tag == "set":
            return set(value)
        kind, mod, qual = tag.split(":", 2)
        if "<locals>" in qual:
            return value
        import sys

        obj = sys.modules.get(mod)  # only types the running code already imported: no import side effects
        if obj is None:
            return value
        for part in qual.split("."):
            obj = getattr(obj, part)
        if kind == "dataclass":
            return obj(**value)
        if kind == "namedtuple":
            return obj(*value)
        if kind == "pydantic":
            return obj.model_validate(value)
    except Exception:
        pass
    return value


def _short(v: Any, n: int = 300) -> str:
    s = v if isinstance(v, str) else json.dumps(to_jsonable(v), ensure_ascii=False)
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 3] + "..."


def _canon(v: Any) -> str:
    return json.dumps(to_jsonable(v), sort_keys=True, ensure_ascii=False)


@dataclass
class ReplayStats:
    served_model_calls: int = 0
    served_tool_calls: int = 0
    live_model_calls: int = 0
    live_tool_calls: int = 0
    divergence: Divergence | None = None
    unused_model_calls: int = 0  # recorded calls the new run never reached


class Replayer(Recorder):
    """A Recorder that answers from an earlier trace instead of calling models and tools."""

    def __init__(self, source: str | Trace, path=None, *, on_diverge: str = "stop", **kwargs):
        if on_diverge not in ("stop", "live"):
            raise ValueError("on_diverge must be 'stop' or 'live'")
        self.source = source if isinstance(source, Trace) else Trace.load(source)
        kwargs.setdefault("name", (self.source.header.get("name") or "run") + "-replay")
        tags = dict(kwargs.pop("tags", None) or {})
        tags["replay_of"] = self.source.header.get("run_id")
        super().__init__(path, tags=tags, **kwargs)
        self.on_diverge = on_diverge
        self.stats = ReplayStats()
        self._reqs = self.source.of_type("llm_request")
        self._tool_calls = self.source.of_type("tool_call")
        self._ri = 0
        self._used_tools: set[int] = set()
        self._turn_start = -1  # id of the recorded model call served most recently
        self._live = False

    # ------------------------------------------------------------ helpers

    @property
    def live(self) -> bool:
        return self._live

    def _diverge(self, d: Divergence) -> None:
        if self.stats.divergence is None:
            self.stats.divergence = d
            self.log("log", {"message": "replay diverged", "divergence": d.__dict__})
        if self.on_diverge == "stop":
            raise ReplayDiverged(d)
        self._live = True

    def _compare_request(self, req: dict, provider: str, api: str) -> Divergence | None:
        step = self._ri + 1
        if self._ri >= len(self._reqs):
            return Divergence("model_call", step, None, "call", "(no more recorded model calls)", (req.get("messages") or [])[-1:])
        rec_ev = self._reqs[self._ri]
        ctx = self.source.context(rec_ev.id)
        recorded = {
            "model": rec_ev.payload.get("model"),
            "system": ctx.system,
            "tools": ctx.tools,
            "messages": self.source.messages(rec_ev.id),
        }
        for key in ("model", "system", "tools"):
            if _canon(recorded[key]) != _canon(req.get(key)):
                return Divergence("model_call", step, rec_ev.id, key, recorded[key], req.get(key))
        from .rerun import _DROP_PARAMS

        def settings(p):
            return {k: v for k, v in (to_jsonable(p or {})).items() if k not in _DROP_PARAMS and v is not None}

        if _canon(settings(rec_ev.payload.get("params"))) != _canon(settings(req.get("params"))):
            return Divergence("model_call", step, rec_ev.id, "params",
                              settings(rec_ev.payload.get("params")), settings(req.get("params")))
        a, b = recorded["messages"], to_jsonable(req.get("messages") or [])
        for i in range(max(len(a), len(b))):
            ra = a[i] if i < len(a) else "(missing)"
            rb = b[i] if i < len(b) else "(missing)"
            if _canon(ra) != _canon(rb):
                return Divergence("model_call", step, rec_ev.id, f"messages[{i}]", ra, rb)
        return None

    def _recorded_response(self, provider: str, api: str) -> Any:
        rec_ev = self._reqs[self._ri]
        resp = next((e for e in self.source.children(rec_ev.id) if e.type == "llm_response"), None)
        if resp is None or resp.payload.get("raw") in (None, {}) or (isinstance(resp.payload.get("raw"), dict) and resp.payload["raw"].get("streamed")):
            return None, resp
        raw = resp.payload["raw"]
        try:
            if api == "messages":
                from anthropic.types import Message

                return Message.model_validate(raw), resp
            if api == "chat.completions":
                from openai.types.chat import ChatCompletion

                return ChatCompletion.model_validate(raw), resp
            if api == "responses":
                from openai.types.responses import Response

                return Response.model_validate(raw), resp
        except Exception:
            return None, resp
        return None, resp

    # ------------------------------------------------------------ models

    def wrap(self, client: Any) -> Any:
        from .integrations import _Anthropic, _OpenAIChat, _OpenAIResponses

        mod = type(client).__module__.split(".")[0]
        if mod == "openai":
            chat = getattr(getattr(client, "chat", None), "completions", None)
            if chat is not None:
                self._patch(chat, "openai", _OpenAIChat)
            if getattr(client, "responses", None) is not None:
                self._patch(client.responses, "openai", _OpenAIResponses)
            return client
        if mod == "anthropic":
            self._patch(client.messages, "anthropic", _Anthropic)
            return client
        raise TypeError(f"replay can't wrap {type(client).__name__}; supported: OpenAI and Anthropic clients")

    def _patch(self, resource: Any, provider: str, adapter: type) -> None:
        original = resource.create
        rp = self

        def serve(kwargs):
            """(recorded response or None, request id). None means: call the real model."""
            req = adapter.request(kwargs)
            rid = rp.log_llm_request(provider=provider, api=adapter.api, **req)
            if rp._live:
                return None, rid
            d = rp._compare_request(req, provider, adapter.api)
            if d is not None:
                rp._diverge(d)
                return None, rid
            obj, resp_ev = rp._recorded_response(provider, adapter.api)
            if obj is None or kwargs.get("stream"):
                rp._diverge(Divergence("model_call", rp._ri + 1, rp._reqs[rp._ri].id, "response",
                                       "(recorded reply can't be served: streamed or not stored)", None))
                return None, rid
            rp._turn_start = rp._reqs[rp._ri].id
            rp._ri += 1
            rp.stats.served_model_calls += 1
            out = adapter.response(obj)
            rp.log_llm_response(rid, latency_ms=0.0, **out)
            return obj, rid

        def go_live(rid, kwargs, t0, resp):
            out = adapter.response(resp)
            rp.stats.live_model_calls += 1
            rp.log_llm_response(rid, latency_ms=(time.perf_counter() - t0) * 1000, **out)
            return resp

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            obj, rid = serve(kwargs)
            if obj is not None:
                if inspect.iscoroutinefunction(original) or _is_async_client(resource):
                    async def done():
                        return obj
                    return done()
                return obj
            from .integrations import _AsyncStream, _SyncStream

            t0 = time.perf_counter()
            resp = original(*args, **kwargs)
            stream = bool(kwargs.get("stream"))
            if inspect.isawaitable(resp):
                async def finish():
                    r = await resp
                    if stream:
                        rp.stats.live_model_calls += 1
                        return _AsyncStream(r, adapter.stream(), rp, rid, t0)
                    return go_live(rid, kwargs, t0, r)
                return finish()
            if stream:
                rp.stats.live_model_calls += 1
                return _SyncStream(resp, adapter.stream(), rp, rid, t0)
            return go_live(rid, kwargs, t0, resp)

        resource.create = wrapped

    # ------------------------------------------------------------- tools

    def tool(self, fn: Callable | None = None, *, name: str | None = None):
        """Like Recorder.tool, but returns recorded results instead of running the tool,
        until the run diverges. Recorded errors are raised again as RuntimeError."""

        def decorate(f: Callable) -> Callable:
            tool_name = name or f.__name__
            sig = _safe_signature(f)

            def serve(args, kwargs):
                arguments = _bind(sig, args, kwargs)
                cid = self.log("tool_call", {"name": tool_name, "arguments": arguments})
                if self._live:
                    return False, None, cid
                # match any not-yet-used recorded call in the current turn with the same name and
                # arguments: order within a turn and tools left unwrapped don't matter
                # the current turn: after the last model call served, before the next recorded one
                limit = self._reqs[self._ri].id if self._ri < len(self._reqs) else float("inf")
                window = [e for e in self._tool_calls
                          if self._turn_start < e.id < limit and e.id not in self._used_tools]
                want = _canon(arguments)
                rec = next((e for e in window if e.payload.get("name") == tool_name
                            and _canon(e.payload.get("arguments")) == want), None)
                step = len(self._used_tools) + 1
                if rec is None:
                    same = next((e for e in window if e.payload.get("name") == tool_name), None)
                    if same is not None:
                        self._diverge(Divergence("tool_call", step, same.id, "arguments", same.payload.get("arguments"), arguments))
                    else:
                        self._diverge(Divergence("tool_call", step, None, "call",
                                                 "(no recorded call to this tool in this turn)",
                                                 {"name": tool_name, "arguments": arguments}))
                    return False, None, cid
                res = next((e for e in self.source.children(rec.id) if e.type == "tool_result"), None)
                if res is not None and "__truncated__" in json.dumps(res.payload.get("result"))[:10_000_000]:
                    self._diverge(Divergence("tool_call", step, rec.id, "result",
                                             "(recorded result was too large and was truncated)", None))
                    return False, None, cid
                self._used_tools.add(rec.id)
                self.stats.served_tool_calls += 1
                return True, res, cid

            def finish_served(res: Event | None, cid: int):
                if res is None:
                    self.log("tool_result", {"name": tool_name, "result": None}, parent=cid)
                    return None
                meta = {"latency_ms": 0.0, "replayed_from": res.id}
                if res.meta.get("result_type"):
                    meta["result_type"] = res.meta["result_type"]
                self.log("tool_result", dict(res.payload), meta=meta, parent=cid)
                if "error" in res.payload:
                    err = res.payload["error"]
                    raise RuntimeError(f"(replayed) {err.get('type')}: {err.get('message')}")
                return _restore(res.payload.get("result"), res.meta.get("result_type"))

            def run_live(call, cid):
                self.stats.live_tool_calls += 1
                t = time.perf_counter()
                try:
                    out = call()
                except BaseException as e:
                    from .recorder import _exc_payload

                    self.log("tool_result", {"name": tool_name, "error": _exc_payload(e)},
                             meta={"latency_ms": round((time.perf_counter() - t) * 1000, 2)}, parent=cid)
                    raise
                self.log("tool_result", {"name": tool_name, "result": out},
                         meta={"latency_ms": round((time.perf_counter() - t) * 1000, 2)}, parent=cid)
                return out

            if inspect.iscoroutinefunction(f):

                @functools.wraps(f)
                async def awrapper(*args, **kwargs):
                    served, res, cid = serve(args, kwargs)
                    if served:
                        return finish_served(res, cid)
                    self.stats.live_tool_calls += 1
                    t = time.perf_counter()
                    out = await f(*args, **kwargs)
                    self.log("tool_result", {"name": tool_name, "result": out},
                             meta={"latency_ms": round((time.perf_counter() - t) * 1000, 2)}, parent=cid)
                    return out

                return awrapper

            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                served, res, cid = serve(args, kwargs)
                if served:
                    return finish_served(res, cid)
                return run_live(lambda: f(*args, **kwargs), cid)

            return wrapper

        if fn is not None and callable(fn):
            return decorate(fn)
        return decorate

    def close(self, status: str = "ok") -> None:
        self.stats.unused_model_calls = max(0, len(self._reqs) - self._ri - self.stats.live_model_calls) if not self._live else 0
        super().close(status)

    def __exit__(self, exc_type, exc, tb) -> bool:
        if isinstance(exc, ReplayDiverged):
            self.close("diverged")
            return False
        return super().__exit__(exc_type, exc, tb)


def _is_async_client(resource: Any) -> bool:
    return type(resource).__name__.startswith("Async")


def replay(source, path=None, *, on_diverge: str = "stop", **kwargs) -> Replayer:
    """Replay a recorded run through your agent code. See module docstring."""
    return Replayer(source, path, on_diverge=on_diverge, **kwargs)
