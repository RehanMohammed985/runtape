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
        self._fh = open(self.path, "a", encoding="utf-8")
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
        """Append one event. Returns its id. Never raises on bad payloads."""
        with self._lock:
            if self._closed:
                return -1
            eid = self._next_id
            self._next_id += 1
            event = {
                "id": eid,
                "ts": time.time(),
                "type": type,
                "parent": parent,
                "payload": to_jsonable(payload if payload is not None else {}),
                "meta": to_jsonable(meta or {}),
            }
            if self._redact is not None:
                try:
                    event = self._redact(event)
                except Exception as e:  # a broken redactor must not kill the run
                    event["meta"]["redact_error"] = repr(e)
            self._fh.write(dumps(event) + "\n")
            self._fh.flush()
            if self._fsync:
                os.fsync(self._fh.fileno())
            return eid

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
                linked = self._claim_tool_call(tool_name)
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

    def _claim_tool_call(self, tool_name: str) -> dict | None:
        with self._lock:
            for i, tc in enumerate(self._pending_tool_calls):
                if tc["name"] == tool_name:
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
    ) -> int:
        """Record a model call. Messages are delta-encoded against earlier requests."""
        msgs = to_jsonable(messages or [])
        ser = [json.dumps(m, sort_keys=True) for m in msgs]
        payload: dict = {"provider": provider, "model": model}

        with self._lock:
            sys_ser = json.dumps(to_jsonable(system), sort_keys=True)
            if system is not None and self._last_system.get(provider) != sys_ser:
                payload["system"] = system
                self._last_system[provider] = sys_ser
            tools_ser = json.dumps(to_jsonable(tools), sort_keys=True)
            if tools is not None and self._last_tools.get(provider) != tools_ser:
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

            rid = self.log("llm_request", payload, meta={"model": model, "provider": provider})
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
            self._pending_tool_calls = [
                {"id": tc.get("id"), "name": tc.get("name"), "response_event": eid}
                for tc in tool_calls
            ]
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
            out = dict(bound.arguments)
            out.pop("self", None)
            out.pop("cls", None)
            return out
        except TypeError:
            pass
    return {"args": list(args), "kwargs": kwargs}
