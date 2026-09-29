"""The recorder: writes events to an append-only JSONL trace."""
from __future__ import annotations

import atexit
import datetime as _dt
import functools
import inspect
import json
import os
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

from .serialize import dumps, to_jsonable

FORMAT_VERSION = 1
_RECENT_REQUESTS = 16

_current: Optional["Recorder"] = None


def current() -> Optional["Recorder"]:
    """The most recently started recorder that is still open, if any."""
    return _current


class Recorder:
    """Records one agent run to a JSONL trace file.

    rec = Recorder(name="support-bot")
    client = rec.wrap(OpenAI())

    @rec.tool
    def lookup_order(order_id: str): ...
    """

    def __init__(
        self,
        path: str | os.PathLike | None = None,
        *,
        name: str | None = None,
        dir: str | os.PathLike = "traces",
        tags: dict | None = None,
        redact: Callable[[dict], dict] | None = None,
        fsync: bool = False,
    ):
        global _current
        self.run_id = uuid.uuid4().hex[:12]
        self.name = name or "run"
        if path is None:
            stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
            safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in self.name)
            path = Path(dir) / f"{stamp}-{safe}-{self.run_id[:6]}.jsonl"
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # one run per file: an existing file at this path is replaced, never appended to
        self._fh = open(self.path, "w", encoding="utf-8")
        self._lock = threading.RLock()
        self._next_id = 0
        self._redact = redact
        self._fsync = fsync
        self._closed = False
        self._t0 = time.time()
        # delta-encoding state
        self._recent: list[tuple[int, list[str]]] = []  # (request id, serialized messages)
        self._last_system: dict[str, str] = {}
        self._last_tools: dict[str, str] = {}
        # tool calls the model asked for that no tool has picked up yet
        self._pending_tool_calls: list[dict] = []
        self._responses_seen = 0
        self._redact_failed = False

        self.log(
            "run_start",
            {
                "run_id": self.run_id,
                "format": "runtape",
                "version": FORMAT_VERSION,
                "name": self.name,
                "started_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                "python": sys.version.split()[0],
                "argv": list(sys.argv),
                "tags": tags or {},
            },
        )
        _current = self
        _install_excepthook()
        atexit.register(self._atexit)

    # ------------------------------------------------------------------ core

    def log(
        self,
        type: str,
        payload: Any = None,
        *,
        meta: dict | None = None,
        parent: int | None = None,
    ) -> int:
        """Append one event. Returns its id. Never raises, and ids never skip."""
        body = to_jsonable(payload if payload is not None else {})  # outside the lock: can be slow
        extra = to_jsonable(meta or {})
        with self._lock:
            if self._closed:
                return -1
            eid = self._next_id
            self._next_id += 1
            event = {"id": eid, "ts": time.time(), "type": type, "parent": parent, "payload": body, "meta": extra}
            if self._redact is not None:
                event = self._apply_redact(event)
            try:
                line = dumps(event)
            except (TypeError, ValueError) as e:
                line = dumps({**{k: event.get(k) for k in ("id", "ts", "type", "parent")},
                              "payload": {"__unserializable__": str(e)}, "meta": {}})
            self._fh.write(line + "\n")
            self._fh.flush()
            if self._fsync:
                os.fsync(self._fh.fileno())
            return eid

    def _apply_redact(self, event: dict) -> dict:
        """Run the user's redactor. If it fails, drop the content rather than write it unredacted."""
        fixed = {k: event[k] for k in ("id", "ts", "type", "parent")}
        try:
            out = self._redact(dict(event))
            if not isinstance(out, dict):
                raise TypeError(f"redact returned {type(out).__name__}, expected the event dict")
            return {**fixed, "payload": to_jsonable(out.get("payload", {})), "meta": to_jsonable(out.get("meta", {}))}
        except Exception as e:
            self._redact_failed = True
            return {**fixed, "payload": {"__redacted__": "redact function failed; content dropped"},
                    "meta": {"redact_error": repr(e)}}

    def state(self, key: str, value: Any) -> int:
        """Record a state change: memory write, plan update, scratchpad, etc."""
        return self.log("state", {"key": key, "value": value})

    def note(self, message: str, **fields: Any) -> int:
        return self.log("log", {"message": message, **fields})

    def error(self, exc: BaseException, parent: int | None = None) -> int:
        return self.log("error", _exc_payload(exc), parent=parent)

    def close(self, status: str = "ok") -> None:
        global _current
        with self._lock:
            if self._closed:
                return
            self.log(
                "run_end",
                {"status": status, "duration_ms": round((time.time() - self._t0) * 1000, 1)},
            )
            self._closed = True
            self._fh.close()
            if _current is self:
                _current = None

    def _atexit(self) -> None:
        if not self._closed:
            self.close(status="error" if self._crashed else "ok")

    _crashed = False

    def __enter__(self) -> "Recorder":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is not None:
            self.error(exc)
            self.close("error")
        else:
            self.close("ok")
        return False

    # ----------------------------------------------------------------- tools

    def tool(self, fn: Callable | None = None, *, name: str | None = None):
        """Decorator that records a tool's arguments, result, errors and latency.

        Works on sync and async functions, with or without arguments:
            @rec.tool
            @rec.tool(name="search")
        """

        def decorate(f: Callable) -> Callable:
            tool_name = name or f.__name__
            sig = _safe_signature(f)

            def start(args, kwargs) -> int:
                arguments = _bind(sig, args, kwargs)
                payload: dict = {"name": tool_name, "arguments": arguments}
                meta: dict = {}
                linked = self._claim_tool_call(tool_name, arguments)
                if linked:
                    payload["call_id"] = linked["id"]
                    meta["requested_by"] = linked["response_event"]
                return self.log("tool_call", payload, meta=meta)

            def finish(cid: int, t: float, result=None, exc: BaseException | None = None):
                meta = {"latency_ms": round((time.perf_counter() - t) * 1000, 2)}
                if exc is not None:
                    self.log(
                        "tool_result",
                        {"name": tool_name, "error": _exc_payload(exc)},
                        meta=meta,
                        parent=cid,
                    )
                else:
                    tag = _type_tag(result)
                    if tag:
                        meta["result_type"] = tag  # lets replay hand back the original type
                    self.log(
                        "tool_result", {"name": tool_name, "result": result}, meta=meta, parent=cid
                    )

            if inspect.iscoroutinefunction(f):

                @functools.wraps(f)
                async def awrapper(*args, **kwargs):
                    cid = start(args, kwargs)
                    t = time.perf_counter()
                    try:
                        out = await f(*args, **kwargs)
                    except BaseException as e:
                        finish(cid, t, exc=e)
                        raise
                    finish(cid, t, result=out)
                    return out

                return awrapper

            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                cid = start(args, kwargs)
                t = time.perf_counter()
                try:
                    out = f(*args, **kwargs)
                except BaseException as e:
                    finish(cid, t, exc=e)
                    raise
                finish(cid, t, result=out)
                return out

            return wrapper

        if fn is not None and callable(fn):
            return decorate(fn)
        return decorate

    def _claim_tool_call(self, tool_name: str, arguments: Any = None) -> dict | None:
        """Link a tool run to the model reply that asked for it.

        Matches on name and arguments, newest request first. A run whose arguments don't match
        anything the model asked for (called by code, or with changed arguments) is left unlinked
        rather than guessed.
        """
        want = _canon_args(arguments)
        with self._lock:
            same_name = [i for i, tc in enumerate(self._pending_tool_calls) if tc["name"] == tool_name]
            for i in reversed(same_name):
                if _canon_args(self._pending_tool_calls[i]["arguments"]) == want:
                    return self._pending_tool_calls.pop(i)
            # the code converted types ("2400" -> 2400.0) or filled in defaults
            loose = _loose(arguments)
            for i in reversed(same_name):
                asked = _loose(self._pending_tool_calls[i]["arguments"])
                if asked == loose or (isinstance(asked, dict) and isinstance(loose, dict) and asked
                                      and all(loose.get(k) == v for k, v in asked.items())):
                    return self._pending_tool_calls.pop(i)
        return None

    # ------------------------------------------------------------------- llm

    def wrap(self, client: Any) -> Any:
        """Patch an OpenAI or Anthropic client (sync or async) in place and return it."""
        from .integrations import wrap_client

        return wrap_client(self, client)

    def langchain(self):
        """Callback handler for LangChain runnables and LangGraph graphs:
        graph.invoke(inputs, config={"callbacks": [rec.langchain()]})"""
        from .langchain import RuntapeCallbackHandler

        return RuntapeCallbackHandler(self)

    def log_llm_request(
        self,
        *,
        provider: str,
        model: Any,
        messages: list,
        system: Any = None,
        tools: Any = None,
        params: dict | None = None,
        api: str | None = None,
    ) -> int:
        """Record a model call. Messages are delta-encoded against earlier requests.

        api names the endpoint so the call can be replayed later:
        "messages" (Anthropic), "chat.completions" or "responses" (OpenAI)."""
        msgs = to_jsonable(messages or [])
        ser = [json.dumps(m, sort_keys=True) for m in msgs]
        payload: dict = {"provider": provider, "model": model}
        if api:
            payload["api"] = api

        with self._lock:
            # system and tools are written only when they change, including a change to none,
            # so readers can carry them forward without inventing a prompt that wasn't sent
            sys_ser = json.dumps(to_jsonable(system), sort_keys=True)
            if self._last_system.get(provider, "null") != sys_ser:
                payload["system"] = system
                self._last_system[provider] = sys_ser
            tools_ser = json.dumps(to_jsonable(tools), sort_keys=True)
            if self._last_tools.get(provider, "null") != tools_ser:
                payload["tools"] = tools
                self._last_tools[provider] = tools_ser

            base = None
            best = 0
            for rid, prev in self._recent:
                n = len(prev)
                if 0 < n <= len(ser) and n >= best and ser[:n] == prev:
                    base, best = rid, n
            if base is not None:
                payload["base"] = base
                payload["messages_append"] = msgs[best:]
            else:
                payload["messages"] = msgs
            payload["params"] = params or {}

            self._redact_failed = False
            rid = self.log("llm_request", payload, meta={"model": model, "provider": provider})
            if self._redact_failed:
                # this request's content was dropped: later requests must not build on it, and must
                # write their system prompt and tools again rather than point back at it
                self._recent.clear()
                self._last_system.pop(provider, None)
                self._last_tools.pop(provider, None)
                self._last_system[provider] = "\x00dropped"
                self._last_tools[provider] = "\x00dropped"
                return rid
            self._recent.append((rid, ser))
            if len(self._recent) > _RECENT_REQUESTS:
                self._recent.pop(0)
        return rid

    def log_llm_response(
        self,
        request_id: int,
        *,
        text: str | None,
        tool_calls: list[dict],
        stop_reason: Any,
        raw: Any,
        latency_ms: float,
        tokens: dict | None = None,
        model: Any = None,
    ) -> int:
        with self._lock:
            eid = self.log(
                "llm_response",
                {"text": text, "tool_calls": tool_calls, "stop_reason": stop_reason, "raw": raw},
                meta={"latency_ms": round(latency_ms, 2), "tokens": tokens or {}, "model": model},
                parent=request_id,
            )
            # keep unclaimed requests from the last few replies (several agents can share a recorder),
            # but let old ones expire so a later call made by code isn't linked to a stale request
            self._responses_seen += 1
            self._pending_tool_calls = [tc for tc in self._pending_tool_calls
                                        if self._responses_seen - tc["seq"] < 5]
            self._pending_tool_calls.extend(
                {"id": tc.get("id"), "name": tc.get("name"), "arguments": tc.get("arguments"),
                 "response_event": eid, "seq": self._responses_seen}
                for tc in tool_calls
            )
        return eid


# ---------------------------------------------------------------- helpers

_hook_installed = False


def _install_excepthook() -> None:
    """Log uncaught exceptions to the open recorder before the process dies."""
    global _hook_installed
    if _hook_installed:
        return
    _hook_installed = True
    prev = sys.excepthook

    def hook(exc_type, exc, tb):
        rec = _current
        if rec is not None and not rec._closed:
            try:
                rec.error(exc)
                rec._crashed = True
                rec.close("error")
            except Exception:
                pass
        prev(exc_type, exc, tb)

    sys.excepthook = hook

    import threading

    prev_thread = threading.excepthook

    def thread_hook(args):
        rec = _current
        if rec is not None and not rec._closed and args.exc_value is not None:
            try:
                rec.log("error", {**_exc_payload(args.exc_value),
                                  "thread": getattr(args.thread, "name", None)})
            except Exception:
                pass
        prev_thread(args)

    threading.excepthook = thread_hook



def _type_tag(v: Any) -> str | None:
    """How to rebuild a tool result that JSON can't represent exactly."""
    import dataclasses as _dc

    if isinstance(v, tuple) and hasattr(v, "_fields"):
        return f"namedtuple:{type(v).__module__}:{type(v).__qualname__}"
    if isinstance(v, tuple):
        return "tuple"
    if isinstance(v, (set, frozenset)):
        return "set"
    cls = type(v)
    where = f"{cls.__module__}:{cls.__qualname__}"
    if _dc.is_dataclass(v) and not isinstance(v, type):
        return "dataclass:" + where
    if hasattr(cls, "model_validate") and hasattr(v, "model_dump"):
        return "pydantic:" + where
    return None


def _loose(v: Any) -> Any:
    """Arguments normalized for matching: numbers and numeric strings compare equal."""
    v = to_jsonable(v)
    if isinstance(v, dict):
        return {k: _loose(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_loose(x) for x in v]
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.strip())
        except ValueError:
            return v.strip()
    return v


def _canon_args(v: Any) -> str:
    try:
        return json.dumps(to_jsonable(v), sort_keys=True, ensure_ascii=False)
    except Exception:
        return repr(v)


def _exc_payload(exc: BaseException) -> dict:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }


def _safe_signature(f: Callable):
    try:
        return inspect.signature(f)
    except (TypeError, ValueError):
        return None


def _bind(sig, args, kwargs) -> dict:
    if sig is not None:
        try:
            bound = sig.bind(*args, **kwargs)
            out: dict = {}
            for name, val in bound.arguments.items():
                kind = sig.parameters[name].kind
                if kind is inspect.Parameter.VAR_KEYWORD:
                    out.update(val)  # f(**kw): record the keywords themselves, as the model sent them
                elif name not in ("self", "cls"):
                    out[name] = val
            return out
        except TypeError:
            pass
    return {"args": list(args), "kwargs": kwargs}
