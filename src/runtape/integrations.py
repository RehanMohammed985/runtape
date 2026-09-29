"""Client wrappers for OpenAI and Anthropic SDKs.

We patch `create` on the client's resource objects in place, so the object you
get back is the same client (isinstance checks and type hints keep working).
"""
from __future__ import annotations

import functools
import inspect
import json
import time
from typing import TYPE_CHECKING, Any

from .serialize import to_jsonable

if TYPE_CHECKING:
    from .recorder import Recorder

_MARK = "__runtape_wrapped__"


def wrap_client(rec: "Recorder", client: Any) -> Any:
    mod = type(client).__module__.split(".")[0]
    if mod == "openai":
        # .stream() helpers call .create(stream=True) internally, so they are covered by create
        chat = getattr(getattr(client, "chat", None), "completions", None)
        if chat is not None:
            _patch(rec, chat, "create", "openai", _OpenAIChat)
            if hasattr(chat, "parse"):
                _patch(rec, chat, "parse", "openai", _OpenAIChat)
        beta_chat = getattr(getattr(getattr(client, "beta", None), "chat", None), "completions", None)
        if beta_chat is not None and beta_chat is not chat and hasattr(beta_chat, "parse"):
            _patch(rec, beta_chat, "parse", "openai", _OpenAIChat)
        responses = getattr(client, "responses", None)
        if responses is not None and hasattr(responses, "create"):
            _patch(rec, responses, "create", "openai", _OpenAIResponses)
            if hasattr(responses, "parse"):
                _patch(rec, responses, "parse", "openai", _OpenAIResponses)
        return client
    if mod == "anthropic":
        for res in (client.messages, getattr(getattr(client, "beta", None), "messages", None)):
            if res is None:
                continue
            _patch(rec, res, "create", "anthropic", _Anthropic)
            if hasattr(res, "stream"):
                _patch_manager(rec, res, "stream", "anthropic", _Anthropic, "get_final_message")
        return client
    raise TypeError(
        f"runtape.wrap doesn't know {type(client).__module__}.{type(client).__name__}. "
        "Supported: openai.OpenAI/AsyncOpenAI, anthropic.Anthropic/AsyncAnthropic. "
        "For anything else use rec.log_llm_request / rec.log_llm_response."
    )


# --------------------------------------------------------------- patching


def _patch(rec: "Recorder", resource: Any, attr: str, provider: str, adapter: type) -> None:
    original = getattr(resource, attr)
    if getattr(original, _MARK, False):
        return

    endpoint = _endpoint(resource)

    def before(kwargs) -> tuple[int, float]:
        req = adapter.request(kwargs)
        rid = rec.log_llm_request(provider=provider, api=adapter.api, endpoint=endpoint, **req)
        return rid, time.perf_counter()

    def after(rid: int, t0: float, resp: Any) -> None:
        out = adapter.response(resp)
        rec.log_llm_response(rid, latency_ms=(time.perf_counter() - t0) * 1000, **out)

    # SDK async methods aren't always declared `async def` (decorators hide it),
    # so decide sync vs async by what the call returns, not by the signature.
    async def finish_async(coro, rid, t0, stream):
        try:
            resp = await coro
        except BaseException as e:
            rec.error(e, parent=rid)
            raise
        if stream:
            return _AsyncStream(resp, adapter.stream(), rec, rid, t0)
        after(rid, t0, resp)
        return resp

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        rid, t0 = before(kwargs)
        try:
            resp = original(*args, **kwargs)
        except BaseException as e:
            rec.error(e, parent=rid)
            raise
        stream = bool(kwargs.get("stream"))
        if inspect.isawaitable(resp):
            return finish_async(resp, rid, t0, stream)
        if stream:
            return _SyncStream(resp, adapter.stream(), rec, rid, t0)
        after(rid, t0, resp)
        return resp

    setattr(wrapped, _MARK, True)
    setattr(resource, attr, wrapped)


_DEFAULT_ENDPOINTS = ("https://api.openai.com/v1", "https://api.anthropic.com")


def _endpoint(resource: Any) -> str | None:
    """The server a client talks to, when it isn't the provider's default (Ollama, vLLM, a proxy...)."""
    url = str(getattr(getattr(resource, "_client", None), "base_url", "") or "").rstrip("/")
    return None if not url or url in _DEFAULT_ENDPOINTS else url


def _patch_manager(rec: "Recorder", resource: Any, attr: str, provider: str, adapter: type, final: str) -> None:
    """Record SDK stream helpers used as context managers (anthropic messages.stream())."""
    original = getattr(resource, attr)
    if getattr(original, _MARK, False):
        return

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        return _ManagerProxy(rec, original(*args, **kwargs), provider, adapter, kwargs, final)

    setattr(wrapped, _MARK, True)
    setattr(resource, attr, wrapped)


class _ManagerProxy:
    def __init__(self, rec, mgr, provider, adapter, kwargs, final):
        self._rec, self._mgr, self._provider, self._adapter = rec, mgr, provider, adapter
        self._kwargs, self._final = kwargs, final
        self._rid = None
        self._stream = None

    def _start(self):
        req = self._adapter.request(self._kwargs)
        self._rid = self._rec.log_llm_request(provider=self._provider, api=self._adapter.api,
                                              endpoint=_endpoint(self._mgr) or None, **req)
        self._t0 = time.perf_counter()

    def _log(self, final_obj, exc):
        if exc is not None:
            self._rec.error(exc, parent=self._rid)
        if final_obj is not None:
            out = self._adapter.response(final_obj)
        else:
            out = {"text": None, "tool_calls": [], "stop_reason": "interrupted", "raw": {"streamed": True}}
        self._rec.log_llm_response(self._rid, latency_ms=(time.perf_counter() - self._t0) * 1000, **out)

    def __enter__(self):
        self._start()
        self._stream = self._mgr.__enter__()
        return self._stream

    def __exit__(self, et, ev, tb):
        final_obj = None
        if ev is None:
            try:
                final_obj = getattr(self._stream, self._final)()
            except Exception:
                final_obj = None
        self._log(final_obj, ev)
        return self._mgr.__exit__(et, ev, tb)

    async def __aenter__(self):
        self._start()
        self._stream = await self._mgr.__aenter__()
        return self._stream

    async def __aexit__(self, et, ev, tb):
        final_obj = None
        if ev is None:
            try:
                final_obj = await getattr(self._stream, self._final)()
            except Exception:
                final_obj = None
        self._log(final_obj, ev)
        return await self._mgr.__aexit__(et, ev, tb)

    def __getattr__(self, name):
        return getattr(self._mgr, name)


# ---------------------------------------------------------------- streams


class _StreamBase:
    def __init__(self, inner, acc, rec, rid, t0):
        self._inner = inner
        self._acc = acc
        self._rec = rec
        self._rid = rid
        self._t0 = t0
        self._done = False
        self._gen = None

    def _finish(self, exc: BaseException | None = None):
        if self._done:
            return
        self._done = True
        if exc is not None:
            self._rec.error(exc, parent=self._rid)
        out = self._acc.result()
        if exc is not None:
            out["stop_reason"] = out.get("stop_reason") or "interrupted"
        self._rec.log_llm_response(
            self._rid, latency_ms=(time.perf_counter() - self._t0) * 1000, **out
        )

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _SyncStream(_StreamBase):
    def __next__(self):  # next(stream) works like on the SDK's own stream
        if self._gen is None:
            self._gen = self.__iter__()
        return next(self._gen)

    def __iter__(self):
        try:
            for chunk in self._inner:
                self._acc.feed(chunk)
                yield chunk
        except GeneratorExit:  # caller stopped iterating early
            self._finish()
            raise
        except BaseException as e:
            self._finish(e)
            raise
        self._finish()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._finish()
        close = getattr(self._inner, "close", None)
        if close:
            close()
        return False

    def close(self):
        self._finish()
        close = getattr(self._inner, "close", None)
        if close:
            close()


class _AsyncStream(_StreamBase):
    async def __anext__(self):
        if self._gen is None:
            self._gen = self.__aiter__()
        return await self._gen.__anext__()

    async def __aiter__(self):
        try:
            async for chunk in self._inner:
                self._acc.feed(chunk)
                yield chunk
        except GeneratorExit:  # caller stopped iterating early
            self._finish()
            raise
        except BaseException as e:
            self._finish(e)
            raise
        self._finish()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self._finish()
        close = getattr(self._inner, "close", None)
        if close:
            r = close()
            if inspect.isawaitable(r):
                await r
        return False


# --------------------------------------------------------------- adapters


def _g(obj, *path, default=None):
    """getattr/getitem chain that tolerates dicts, objects and missing keys."""
    for p in path:
        if obj is None:
            return default
        if isinstance(obj, dict):
            obj = obj.get(p)
        elif isinstance(p, int):
            try:
                obj = obj[p]
            except (IndexError, TypeError, KeyError):
                return default
        else:
            obj = getattr(obj, p, None)
    return default if obj is None else obj


def _parse_args(s: Any) -> Any:
    if isinstance(s, str):
        try:
            return json.loads(s)
        except ValueError:
            return s
    return s


_SKIP = {"messages", "tools", "model", "system", "input", "instructions"}


def _params(kwargs: dict) -> dict:
    return {k: v for k, v in kwargs.items() if k not in _SKIP and type(v).__name__ not in ("NotGiven", "Omit")}


class _OpenAIChat:
    api = "chat.completions"

    @staticmethod
    def request(kw: dict) -> dict:
        return {
            "model": kw.get("model"),
            "messages": list(kw.get("messages") or []),
            "system": None,  # system lives inside messages for chat completions
            "tools": kw.get("tools"),
            "params": _params(kw),
        }

    @staticmethod
    def response(resp: Any) -> dict:
        msg = _g(resp, "choices", 0, "message")
        calls = [
            {
                "id": _g(tc, "id"),
                "name": _g(tc, "function", "name"),
                "arguments": _parse_args(_g(tc, "function", "arguments")),
            }
            for tc in (_g(msg, "tool_calls") or [])
        ]
        usage = _g(resp, "usage")
        return {
            "text": _g(msg, "content"),
            "tool_calls": calls,
            "stop_reason": _g(resp, "choices", 0, "finish_reason"),
            "raw": to_jsonable(resp),
            "tokens": {
                "input": _g(usage, "prompt_tokens"),
                "output": _g(usage, "completion_tokens"),
            },
            "model": _g(resp, "model"),
        }

    @staticmethod
    def stream():
        return _OpenAIChatAcc()


class _OpenAIChatAcc:
    def __init__(self):
        self.text: list[str] = []
        self.calls: dict[int, dict] = {}
        self.stop = None
        self.usage = None
        self.model = None
        self.n = 0

    def feed(self, chunk):
        self.n += 1
        self.model = _g(chunk, "model") or self.model
        if _g(chunk, "usage"):
            self.usage = _g(chunk, "usage")
        delta = _g(chunk, "choices", 0, "delta")
        if delta is None:
            return
        if _g(delta, "content"):
            self.text.append(_g(delta, "content"))
        for tc in _g(delta, "tool_calls") or []:
            slot = self.calls.setdefault(_g(tc, "index", default=0), {"id": None, "name": "", "arguments": ""})
            slot["id"] = _g(tc, "id") or slot["id"]
            slot["name"] += _g(tc, "function", "name") or ""
            slot["arguments"] += _g(tc, "function", "arguments") or ""
        self.stop = _g(chunk, "choices", 0, "finish_reason") or self.stop

    def result(self) -> dict:
        calls = [
            {"id": c["id"], "name": c["name"], "arguments": _parse_args(c["arguments"])}
            for _, c in sorted(self.calls.items())
        ]
        return {
            "text": "".join(self.text) or None,
            "tool_calls": calls,
            "stop_reason": self.stop,
            "raw": {"streamed": True, "chunks": self.n},
            "tokens": {
                "input": _g(self.usage, "prompt_tokens"),
                "output": _g(self.usage, "completion_tokens"),
            },
            "model": self.model,
        }


class _OpenAIResponses:
    api = "responses"

    @staticmethod
    def request(kw: dict) -> dict:
        inp = kw.get("input")
        if isinstance(inp, str):
            msgs = [{"role": "user", "content": inp}]
        else:
            msgs = list(inp or [])
        return {
            "model": kw.get("model"),
            "messages": msgs,
            "system": kw.get("instructions"),
            "tools": kw.get("tools"),
            "params": _params(kw),
        }

    @staticmethod
    def response(resp: Any) -> dict:
        calls = [
            {
                "id": _g(item, "call_id") or _g(item, "id"),
                "name": _g(item, "name"),
                "arguments": _parse_args(_g(item, "arguments")),
            }
            for item in (_g(resp, "output") or [])
            if _g(item, "type") == "function_call"
        ]
        text = getattr(resp, "output_text", None) if not isinstance(resp, dict) else None
        usage = _g(resp, "usage")
        return {
            "text": text or None,
            "tool_calls": calls,
            "stop_reason": _g(resp, "status"),
            "raw": to_jsonable(resp),
            "tokens": {"input": _g(usage, "input_tokens"), "output": _g(usage, "output_tokens")},
            "model": _g(resp, "model"),
        }

    @staticmethod
    def stream():
        return _OpenAIResponsesAcc()


class _OpenAIResponsesAcc:
    def __init__(self):
        self.final = None
        self.text: list[str] = []
        self.n = 0

    def feed(self, event):
        self.n += 1
        t = _g(event, "type")
        if t == "response.output_text.delta":
            self.text.append(_g(event, "delta") or "")
        elif t in ("response.completed", "response.incomplete", "response.failed"):
            self.final = _g(event, "response")

    def result(self) -> dict:
        if self.final is not None:
            out = _OpenAIResponses.response(self.final)
            out["raw"] = {"streamed": True, "chunks": self.n, "response": out["raw"]}
            return out
        return {
            "text": "".join(self.text) or None,
            "tool_calls": [],
            "stop_reason": None,
            "raw": {"streamed": True, "chunks": self.n},
            "tokens": {},
            "model": None,
        }


class _Anthropic:
    api = "messages"

    @staticmethod
    def request(kw: dict) -> dict:
        return {
            "model": kw.get("model"),
            "messages": list(kw.get("messages") or []),
            "system": kw.get("system"),
            "tools": kw.get("tools"),
            "params": _params(kw),
        }

    @staticmethod
    def response(resp: Any) -> dict:
        texts, calls = [], []
        for block in _g(resp, "content") or []:
            bt = _g(block, "type")
            if bt == "text":
                texts.append(_g(block, "text") or "")
            elif bt == "tool_use":
                calls.append(
                    {"id": _g(block, "id"), "name": _g(block, "name"), "arguments": to_jsonable(_g(block, "input"))}
                )
        usage = _g(resp, "usage")
        return {
            "text": "".join(texts) or None,
            "tool_calls": calls,
            "stop_reason": _g(resp, "stop_reason"),
            "raw": to_jsonable(resp),
            "tokens": {"input": _g(usage, "input_tokens"), "output": _g(usage, "output_tokens")},
            "model": _g(resp, "model"),
        }

    @staticmethod
    def stream():
        return _AnthropicAcc()


class _AnthropicAcc:
    def __init__(self):
        self.blocks: dict[int, dict] = {}
        self.stop = None
        self.tokens = {"input": None, "output": None}
        self.model = None
        self.n = 0

    def feed(self, ev):
        self.n += 1
        t = _g(ev, "type")
        if t == "message_start":
            self.model = _g(ev, "message", "model")
            self.tokens["input"] = _g(ev, "message", "usage", "input_tokens")
        elif t == "content_block_start":
            cb = _g(ev, "content_block")
            self.blocks[_g(ev, "index", default=0)] = {
                "type": _g(cb, "type"),
                "id": _g(cb, "id"),
                "name": _g(cb, "name"),
                "text": "",
                "json": "",
            }
        elif t == "content_block_delta":
            b = self.blocks.setdefault(_g(ev, "index", default=0), {"type": "text", "text": "", "json": ""})
            d = _g(ev, "delta")
            if _g(d, "type") == "text_delta":
                b["text"] += _g(d, "text") or ""
            elif _g(d, "type") == "input_json_delta":
                b["json"] += _g(d, "partial_json") or ""
        elif t == "message_delta":
            self.stop = _g(ev, "delta", "stop_reason") or self.stop
            out = _g(ev, "usage", "output_tokens")
            if out is not None:
                self.tokens["output"] = out

    def result(self) -> dict:
        texts, calls = [], []
        for _, b in sorted(self.blocks.items()):
            if b.get("type") == "tool_use":
                calls.append({"id": b.get("id"), "name": b.get("name"), "arguments": _parse_args(b["json"] or "{}")})
            elif b.get("text"):
                texts.append(b["text"])
        return {
            "text": "".join(texts) or None,
            "tool_calls": calls,
            "stop_reason": self.stop,
            "raw": {"streamed": True, "chunks": self.n},
            "tokens": self.tokens,
            "model": self.model,
        }
