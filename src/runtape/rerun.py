"""Rebuild recorded model calls and send them again.

This is what makes runtape an experimental debugger instead of a log viewer:
every decision in a trace can be re-executed, as recorded or modified, without
running the agent or repeating any of its side effects.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import re
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from .serialize import to_jsonable
from .trace import Trace

# request params that only matter to the original transport, never replayed
_DROP_PARAMS = {"stream", "stream_options", "extra_headers", "extra_query", "extra_body", "timeout", "n"}


# ---------------------------------------------------------------- requests


def build_request(trace: Trace, request_id: int) -> dict:
    """The exact request behind an llm_request event, fully resolved."""
    ev = trace[request_id]
    if ev.type != "llm_request":
        raise ValueError(f"#{request_id} is a {ev.type}, not an llm_request")
    p = ev.payload
    ctx = trace.context(request_id)
    provider = p.get("provider") or "unknown"
    api = p.get("api") or {"anthropic": "messages", "openai": "chat.completions"}.get(provider, provider)
    params = {k: v for k, v in (p.get("params") or {}).items() if k not in _DROP_PARAMS and v is not None}
    return {
        "provider": provider,
        "api": api,
        "endpoint": p.get("endpoint"),
        "model": p.get("model"),
        "system": copy.deepcopy(ctx.system),
        "tools": copy.deepcopy(ctx.tools),
        "messages": copy.deepcopy(ctx.messages),
        "params": params,
    }


def request_for(trace: Trace, event_id: int) -> tuple[int, int | None]:
    """(request id, response id) for the decision behind any event.

    Accepts an llm_request, an llm_response, or a tool_call the model asked for.
    """
    ev = trace[event_id]
    if ev.type == "llm_request":
        resp = next((e.id for e in trace.children(ev.id) if e.type == "llm_response"), None)
        return ev.id, resp
    if ev.type == "llm_response":
        return ev.parent, ev.id
    if ev.type == "tool_call":
        rid = ev.meta.get("requested_by")
        if rid is None:
            # never guess: an unlinked call was run by code, or with arguments the model didn't ask for
            prev = next((e.id for e in reversed(trace.events) if e.id < ev.id and e.type == "llm_response"), None)
            hint = f" Point at the model reply instead, e.g. #{prev}." if prev is not None else ""
            raise ValueError(
                f"#{event_id} ({ev.payload.get('name')}) wasn't requested by a model reply in this trace: it was "
                f"called by code, or with arguments different from what the model asked for.{hint}"
            )
        return trace[rid].parent, rid
    if ev.type == "tool_result" and ev.parent is not None:
        return request_for(trace, ev.parent)
    raise ValueError(f"#{event_id} is a {ev.type}; point at a model call, model reply, or tool call")


def request_key(req: dict) -> str:
    blob = json.dumps(to_jsonable(req), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


# ------------------------------------------------------------------ replies


@dataclass
class Reply:
    text: str | None
    tool_calls: list[dict] = field(default_factory=list)
    stop_reason: Any = None
    raw: Any = None

    def describe(self, limit: int = 90) -> str:
        """Human description of what the model did."""
        if self.tool_calls:
            parts = []
            for tc in self.tool_calls:
                args = tc.get("arguments")
                if isinstance(args, dict):
                    a = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in args.items())
                else:
                    a = json.dumps(args, ensure_ascii=False) if args is not None else ""
                parts.append(f"{tc.get('name')}({a})")
            s = "calls " + ", ".join(parts)
        elif (self.text or "").strip():
            s = "replies: " + " ".join(self.text.split())
        else:
            s = "stops with an empty reply"
        return s if len(s) <= limit else s[: limit - 3] + "..."

    def to_dict(self) -> dict:
        return {"text": self.text, "tool_calls": self.tool_calls, "stop_reason": self.stop_reason}

    @classmethod
    def from_any(cls, obj: Any) -> "Reply":
        if isinstance(obj, Reply):
            return obj
        if isinstance(obj, str):
            return cls(text=obj)
        if isinstance(obj, dict):
            return cls(
                text=obj.get("text"),
                tool_calls=list(obj.get("tool_calls") or []),
                stop_reason=obj.get("stop_reason"),
                raw=obj.get("raw"),
            )
        raise TypeError(f"model returned {type(obj).__name__}; expected Reply, dict or str")


# ---------------------------------------------------------- format bridges

# parameters each API accepts; LangChain traces carry many client-only settings besides these
_ANTHROPIC_PARAMS = {"max_tokens", "temperature", "top_k", "top_p", "stop_sequences", "thinking", "tool_choice", "metadata"}
_OPENAI_PARAMS = {
    "temperature", "max_tokens", "max_completion_tokens", "top_p", "seed", "tool_choice", "response_format",
    "presence_penalty", "frequency_penalty", "stop", "reasoning_effort", "parallel_tool_calls",
}


def _filter_params(params: dict, allowed: set, renames: dict | None = None) -> dict:
    out = {}
    for k, v in (params or {}).items():
        k = (renames or {}).get(k, k)
        if k in allowed and v is not None:
            out[k] = v
    return out


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return "" if content is None else str(content)


def openai_to_anthropic(messages: list) -> tuple[str | None, list]:
    """Convert OpenAI-style messages (as LangChain records them) to Anthropic's format."""
    system: list[str] = []
    out: list[dict] = []
    for m in messages or []:
        role = m.get("role")
        if role in ("system", "developer"):
            system.append(_text_of(m.get("content")))
        elif role == "tool":
            block = {"type": "tool_result", "tool_use_id": m.get("tool_call_id"), "content": _text_of(m.get("content")) or "(empty)"}
            prev = out[-1] if out else None
            if prev and prev["role"] == "user" and isinstance(prev["content"], list) and all(
                b.get("type") == "tool_result" for b in prev["content"]
            ):
                prev["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
        elif role == "assistant":
            blocks = []
            text = _text_of(m.get("content"))
            if text.strip():
                blocks.append({"type": "text", "text": text})
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                args = fn.get("arguments")
                try:
                    args = json.loads(args) if isinstance(args, str) else (args or {})
                except ValueError:
                    args = {"_raw": args}
                blocks.append({"type": "tool_use", "id": tc.get("id"), "name": fn.get("name"), "input": args})
            out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": "(empty)"}]})
        else:
            content = m.get("content")
            if isinstance(content, list):
                content = [b for b in content if isinstance(b, dict) and b.get("type") == "text"] or _text_of(content)
            out.append({"role": "user", "content": content})
    return ("\n\n".join(x for x in system if x) or None), out


# ------------------------------------------------------------------- models

Model = Callable[[dict], Reply]


def _has_cache_control(obj: Any) -> bool:
    if isinstance(obj, dict):
        return "cache_control" in obj or any(_has_cache_control(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_has_cache_control(v) for v in obj)
    return False


def _mark(content: Any) -> Any:
    """The content with a cache breakpoint on its last block."""
    if isinstance(content, str):
        return [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}] if content else content
    if isinstance(content, list) and content and isinstance(content[-1], dict):
        return content[:-1] + [{**content[-1], "cache_control": {"type": "ephemeral"}}]
    return content


def cache_breakpoints(system: Any, messages: list) -> tuple[Any, list]:
    """Anthropic prompt caching for reruns: why sends the same context many times, so mark the system
    prompt and the end of the conversation as cacheable. Repeats then read the context at a tenth of
    the input price. Requests that already set their own breakpoints are left alone, and a context
    shorter than the provider's minimum is simply not cached."""
    if _has_cache_control(system) or _has_cache_control(messages) or not messages:
        return system, messages
    system = _mark(system) if system else system
    messages = list(messages)
    last = messages[-1]
    if isinstance(last, dict) and "content" in last:
        messages[-1] = {**last, "content": _mark(last["content"])}
    return system, messages


class AnthropicModel:
    prefix_cache = True  # repeats of a request are cheaper once the first has been sent

    def __init__(self, client: Any = None, endpoint: str | None = None):
        self._client = client
        self.endpoint = endpoint  # created on first call, so no API key is needed until then

    @property
    def client(self):
        if self._client is None:
            self._client = self._make()
        return self._client


    def _make(self):
        import anthropic

        if self.endpoint:
            return anthropic.Anthropic(base_url=self.endpoint, max_retries=8, timeout=client_timeout(anthropic))
        return anthropic.Anthropic(max_retries=8, timeout=client_timeout(anthropic))  # why runs many calls; ride out rate limits

    def __call__(self, req: dict) -> Reply:
        from .integrations import _Anthropic

        messages, system = req["messages"], req.get("system")
        if req.get("api") == "langchain":
            system, messages = openai_to_anthropic(messages)
            kw = _filter_params(req["params"], _ANTHROPIC_PARAMS, {"stop": "stop_sequences"})
        else:
            kw = dict(req["params"])
        kw.setdefault("max_tokens", 1024)
        system, messages = cache_breakpoints(system, messages)
        if system is not None:
            kw["system"] = system
        if req.get("tools"):
            kw["tools"] = req["tools"]
        resp = self.client.messages.create(model=req["model"], messages=messages, **kw)
        out = _Anthropic.response(resp)
        return Reply(out["text"], out["tool_calls"], out["stop_reason"], out["raw"])


def client_timeout(sdk: Any) -> Any:
    """How long a rerun waits: 10 seconds to connect, and RUNTAPE_TIMEOUT seconds (default 300) for the reply.
    The SDKs' default of 10 minutes, retried, let one request cut off by a network drop stall a run for an hour;
    a slow local model can need more, so it's settable. Built with the SDK's own Timeout class (the SDKs have
    moved between HTTP libraries)."""
    read = float(os.environ.get("RUNTAPE_TIMEOUT") or 300)
    make = getattr(sdk, "Timeout", None)
    try:
        return make(read, connect=10.0) if make else read
    except TypeError:
        return read


class OpenAIChatModel:
    prefix_cache = True  # OpenAI caches long repeated prefixes automatically

    def __init__(self, client: Any = None, endpoint: str | None = None):
        self._client = client
        self.endpoint = endpoint  # created on first call, so no API key is needed until then

    @property
    def client(self):
        if self._client is None:
            self._client = self._make()
        return self._client


    def _make(self):
        import os

        import openai

        if self.endpoint:  # a local or self-hosted server: it usually ignores the key
            return openai.OpenAI(base_url=self.endpoint, max_retries=8, timeout=client_timeout(openai),
                                 api_key=os.environ.get("OPENAI_API_KEY") or "local")
        return openai.OpenAI(max_retries=8, timeout=client_timeout(openai))

    def _kw(self, req: dict) -> dict:
        kw = _filter_params(req["params"], _OPENAI_PARAMS) if req.get("api") == "langchain" else dict(req["params"])
        if req.get("tools"):
            kw["tools"] = req["tools"]
        return kw

    def __call__(self, req: dict) -> Reply:
        from .integrations import _OpenAIChat

        resp = self.client.chat.completions.create(model=req["model"], messages=req["messages"], **self._kw(req))
        out = _OpenAIChat.response(resp)
        return Reply(out["text"], out["tool_calls"], out["stop_reason"], out["raw"])

    def batch(self, req: dict, n: int) -> list[Reply]:
        """n samples of one request in a single call (the API's `n`): the context is sent and billed
        once. A server that ignores `n` returns fewer choices; the sampler notices and stops asking."""
        from .integrations import _OpenAIChat

        resp = self.client.chat.completions.create(model=req["model"], messages=req["messages"], n=n, **self._kw(req))
        raw = to_jsonable(resp)
        out = []
        for choice in (raw.get("choices") or [])[:n]:
            one = _OpenAIChat.response({**raw, "choices": [choice]})
            out.append(Reply(one["text"], one["tool_calls"], one["stop_reason"], one["raw"]))
        return out


class OpenAIResponsesModel:
    prefix_cache = True

    def __init__(self, client: Any = None, endpoint: str | None = None):
        self._client = client
        self.endpoint = endpoint  # created on first call, so no API key is needed until then

    @property
    def client(self):
        if self._client is None:
            self._client = self._make()
        return self._client


    def _make(self):
        import os

        import openai

        if self.endpoint:  # a local or self-hosted server: it usually ignores the key
            return openai.OpenAI(base_url=self.endpoint, max_retries=8, timeout=client_timeout(openai),
                                 api_key=os.environ.get("OPENAI_API_KEY") or "local")
        return openai.OpenAI(max_retries=8, timeout=client_timeout(openai))

    def __call__(self, req: dict) -> Reply:
        from .integrations import _OpenAIResponses

        kw = dict(req["params"])
        if req.get("system") is not None:
            kw["instructions"] = req["system"]
        if req.get("tools"):
            kw["tools"] = req["tools"]
        resp = self.client.responses.create(model=req["model"], input=req["messages"], **kw)
        out = _OpenAIResponses.response(resp)
        return Reply(out["text"], out["tool_calls"], out["stop_reason"], out["raw"])


class FunctionModel:
    """Wrap any function(request) -> Reply | dict | str. Useful for local models and tests."""

    def __init__(self, fn: Callable[[dict], Any]):
        self.fn = fn

    def __call__(self, req: dict) -> Reply:
        return Reply.from_any(self.fn(req))


def model_for(req: dict) -> Model:
    """The live backend matching how a request was originally sent."""
    api, provider, endpoint = req.get("api"), req.get("provider"), req.get("endpoint")
    if api == "messages" or provider == "anthropic":
        return AnthropicModel(endpoint=endpoint)
    if api == "responses":
        return OpenAIResponsesModel(endpoint=endpoint)
    if api == "chat.completions" or (api == "langchain" and provider == "openai"):
        return OpenAIChatModel(endpoint=endpoint)
    raise ValueError(
        f"Don't know how to resend a {provider}/{api} call. Pass a model function "
        "(--model-fn module:function) that takes the request dict and returns a reply."
    )


def load_model_fn(spec: str) -> Model:
    """'package.module:function' -> FunctionModel. Also accepts 'path/to/file.py:function'."""
    mod_name, _, attr = spec.partition(":")
    if not attr:
        raise ValueError(f"--model-fn '{spec}' should look like file.py:function or module:function")
    if mod_name.endswith(".py") or os.sep in mod_name:
        import importlib.util
        import sys

        if not Path(mod_name).is_file():
            raise ValueError(f"--model-fn: no file {mod_name} (from {Path.cwd()})")
        folder = str(Path(mod_name).resolve().parent)
        if folder not in sys.path:
            sys.path.insert(0, folder)  # so the file can import its neighbors
        spec_ = importlib.util.spec_from_file_location(Path(mod_name).stem, mod_name)
        mod = importlib.util.module_from_spec(spec_)
        spec_.loader.exec_module(mod)
    else:
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError as e:
            raise ValueError(f"--model-fn: can't import {mod_name} ({e}). For a file, give its path: "
                             f"path/to/{mod_name}.py:{attr}") from None
    if not hasattr(mod, attr):
        raise ValueError(f"--model-fn: {mod_name} has no '{attr}'")
    obj = getattr(mod, attr)
    if isinstance(obj, type):
        obj = obj()
    if isinstance(obj, (AnthropicModel, OpenAIChatModel, OpenAIResponsesModel, FunctionModel)):
        return obj
    return FunctionModel(obj)


# ------------------------------------------------------------------ sampler


def model_identity(model: Any) -> str:
    """A stable name for a backend, including the source of a model function."""
    if isinstance(model, FunctionModel):
        fn = model.fn
        name = f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__qualname__', type(fn).__name__)}"
        parts = []
        for obj in (fn, inspect.getmodule(fn)):  # the function itself, and the file around it (its helpers)
            try:
                parts.append(inspect.getsource(obj) if obj is not None else "")
            except (OSError, TypeError):
                parts.append(repr(obj))
        src = "\n".join(parts)
        return "fn:" + name + ":" + hashlib.sha256(src.encode()).hexdigest()[:16]
    return "api:" + type(model).__name__ + ":" + str(getattr(model, "endpoint", None) or "")


def server_side_context(req: dict) -> str | None:
    """A warning when part of a call's context lives on the provider's servers, out of reach."""
    if (req.get("params") or {}).get("previous_response_id") or (req.get("params") or {}).get("conversation"):
        return ("This call continues a conversation stored on OpenAI's servers (previous_response_id). "
                "Earlier turns are not in the trace, so they can't be tested or removed; they stay in every rerun.")
    return None


class BudgetExceeded(RuntimeError):
    pass


def safe_workers(model, workers: int) -> int:
    """One request at a time for a model on this machine. A local server (Ollama, LM Studio) keeps a
    separate context in memory for each parallel request, which can exhaust a laptop's memory."""
    ep = getattr(model, "endpoint", None) or ""
    host = re.sub(r"^\w+://", "", ep).split("/")[0].rsplit(":", 1)[0].strip("[]")
    if host in ("localhost", "127.0.0.1", "0.0.0.0", "::1") or host.endswith(".local"):
        return 1
    return workers


class Sampler:
    """Sends requests through a model with caching, a call budget and parallelism.

    Each (request, sample index) pair is cached on disk, so repeating an
    experiment is free and resumable.
    """

    def __init__(
        self,
        model: Model,
        *,
        cache_dir: str | os.PathLike | None = ".runtape/cache",
        budget: int | None = 300,
        workers: int = 8,
        on_call: Callable[[], None] | None = None,
    ):
        self.model = model
        self.model_id = model_identity(model)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.budget = budget
        self.workers = workers
        self.calls = 0  # live samples (model calls) made
        self.requests = 0  # live API requests; fewer than calls when several samples share one request
        self.hits = 0  # served from cache
        self._lock = threading.Lock()
        self.on_call = on_call
        self.max_n = 20  # samples asked for in one request, where the API supports it
        self.n_ok: bool | None = None  # does this backend honor `n`? unknown until the first try

    def _cache_path(self, req: dict, i: int) -> Path | None:
        if not self.cache_dir:
            return None
        # the key covers the backend too, so switching models never reuses another model's replies
        key = hashlib.sha256((self.model_id + "\n" + request_key(req)).encode()).hexdigest()
        return self.cache_dir / f"{key[:40]}-{i}.json"

    def _read(self, req: dict, i: int) -> Reply | None:
        path = self._cache_path(req, i)
        if path and path.exists():
            try:
                reply = Reply.from_any(json.loads(path.read_text()))
            except ValueError:
                return None
            with self._lock:
                self.hits += 1
            return reply
        return None

    def _reserve(self, n: int) -> int:
        """Take up to n samples from the budget; how many were granted (at least 1, or BudgetExceeded)."""
        with self._lock:
            if self.budget is not None:
                n = min(n, self.budget - self.calls)
                if n <= 0:
                    raise BudgetExceeded(f"hit the budget of {self.budget} model calls")
            self.calls += n
            self.requests += 1
        return n

    def _call(self, req: dict) -> Reply:
        try:
            return Reply.from_any(self.model(req))
        except Exception as e:
            if isinstance(self.model, FunctionModel):
                raise RuntimeError(f"the model function raised {type(e).__name__}: {e} "
                                   "(it must handle any variant of the context, including removed content)") from e
            raise

    def _live(self, req: dict, idx: list[int], got: list[Reply]) -> None:
        """Fetch samples `idx` of one request, in as few API requests as the backend allows. Replies are
        cached and appended to `got` as they arrive, so a budget stop keeps what was fetched."""
        batch = getattr(self.model, "batch", None)
        while len(got) < len(idx):
            want = len(idx) - len(got)
            if batch is not None and self.n_ok is not False and want > 1:
                n = self._reserve(min(want, self.max_n))
                try:
                    replies = [Reply.from_any(r) for r in batch(req, n)][:n]
                except Exception:
                    if self.n_ok is None:  # the server may just not support `n`: fall back to one at a time
                        with self._lock:
                            self.n_ok, self.calls, self.requests = False, self.calls - n, self.requests - 1
                        continue
                    raise
                if not replies:
                    with self._lock:
                        self.n_ok, self.calls, self.requests = False, self.calls - n, self.requests - 1
                    continue
                with self._lock:
                    if len(replies) < n:  # `n` was ignored: count what came back, and stop asking for more
                        self.n_ok, self.calls = False, self.calls - (n - len(replies))
                    elif n > 1:
                        self.n_ok = True
            else:
                self._reserve(1)
                replies = [self._call(req)]
            for reply in replies:
                path = self._cache_path(req, idx[len(got)])
                if path:
                    path.write_text(json.dumps(to_jsonable(reply.to_dict()), ensure_ascii=False))
                got.append(reply)
                if self.on_call:
                    self.on_call()

    def one(self, req: dict, i: int = 0) -> Reply:
        return self.many([(req, i)])[0]

    def many(self, jobs: Iterable[tuple[dict, int]]) -> list[Reply]:
        """Run jobs (request, sample index), in order. Cached samples are read from disk; the rest are
        grouped by request, so a backend that supports `n` gets one API request per distinct context.
        For a backend with prompt caching, the first sample of each context is sent before its repeats,
        so the repeats find the context cached. If the budget runs out, the replies that did arrive
        (a leading run of them) are attached to the exception as .partial."""
        jobs = list(jobs)
        res: list[Reply | None] = [None] * len(jobs)
        groups: dict[str, tuple[dict, list[int]]] = {}
        for j, (r, i) in enumerate(jobs):
            hit = self._read(r, i)
            if hit is not None:
                res[j] = hit
            else:
                groups.setdefault(request_key(r), (r, []))[1].append(j)

        batching = getattr(self.model, "batch", None) is not None and self.n_ok is not False
        first: list[tuple[dict, list[int]]] = []
        rest: list[tuple[dict, list[int]]] = []
        for r, js in groups.values():
            if batching:
                first.append((r, js))
            elif getattr(self.model, "prefix_cache", False) and len(js) > 1:
                first.append((r, js[:1]))
                rest.extend((r, [j]) for j in js[1:])
            else:
                first.extend((r, [j]) for j in js)

        err: list[BudgetExceeded] = []

        def run(unit: tuple[dict, list[int]]) -> None:
            r, js = unit
            got: list[Reply] = []
            try:
                self._live(r, [jobs[j][1] for j in js], got)
            except BudgetExceeded as e:
                err.append(e)
            finally:
                for j, reply in zip(js, got):
                    res[j] = reply

        for phase in (first, rest):
            if not phase or err:
                continue
            if self.workers <= 1 or len(phase) <= 1:
                for unit in phase:
                    if err:
                        break
                    run(unit)
            else:
                with ThreadPoolExecutor(max_workers=self.workers) as pool:
                    list(pool.map(run, phase))
        if err:
            e = err[0]
            lead: list[Reply] = []
            for x in res:
                if x is None:
                    break
                lead.append(x)
            e.partial = lead
            raise e
        return res  # type: ignore[return-value]

    def samples(self, req: dict, k: int, start: int = 0) -> list[Reply]:
        return self.many((req, i) for i in range(start, start + k))


def is_deterministic(req: dict) -> bool:
    return req.get("params", {}).get("temperature") == 0


# ------------------------------------------------------------ distributions


@dataclass
class Distribution:
    """What a model did across repeated runs of one request."""

    replies: list[Reply]
    notes: list[str] = field(default_factory=list)
    recorded: Reply | None = None
    request_id: int | None = None
    response_id: int | None = None
    model_calls: int = 0

    def counts(self, key: Callable[[Reply], str] | None = None) -> Counter:
        key = key or (lambda r: r.describe(200))
        return Counter(key(r) for r in self.replies)

    def calls(self, tool: str) -> int:
        return sum(any(tc.get("name") == tool for tc in r.tool_calls) for r in self.replies)

    def rate(self, tool: str) -> float:
        return self.calls(tool) / len(self.replies) if self.replies else 0.0

    def never_calls(self, tool: str) -> "Distribution":
        n = self.calls(tool)
        if n:
            raise AssertionError(f"{tool} was called in {n}/{len(self.replies)} runs")
        return self

    def always_calls(self, tool: str) -> "Distribution":
        n = self.calls(tool)
        if n != len(self.replies):
            raise AssertionError(f"{tool} was called in only {n}/{len(self.replies)} runs")
        return self

    def matching(self, pattern: str) -> int:
        """Runs whose reply matches a regex (on the text and on each tool call as `name {json args}`)."""
        rx = re.compile(pattern, re.I)
        return sum(bool(rx.search(_reply_text(r))) for r in self.replies)

    def calls_matching(self, pattern: str) -> int:
        """Runs with a tool call matching a regex (on `name {json args}`), ignoring the reply text."""
        rx = re.compile(pattern, re.I)
        return sum(any(rx.search(f"{tc.get('name')} {json.dumps(tc.get('arguments'), sort_keys=True, ensure_ascii=False)}")
                       for tc in r.tool_calls) for r in self.replies)

    def never_calls_matching(self, pattern: str) -> "Distribution":
        n = self.calls_matching(pattern)
        if n:
            raise AssertionError(f"a tool call matching /{pattern}/ was made in {n}/{len(self.replies)} runs")
        return self

    def never_matches(self, pattern: str) -> "Distribution":
        n = self.matching(pattern)
        if n:
            raise AssertionError(f"/{pattern}/ matched in {n}/{len(self.replies)} runs")
        return self

    def __len__(self) -> int:
        return len(self.replies)


def _reply_text(r: Reply) -> str:
    parts = [r.text or ""]
    for tc in r.tool_calls:
        parts.append(f"{tc.get('name')} {json.dumps(tc.get('arguments'), sort_keys=True, ensure_ascii=False)}")
    return "\n".join(parts)


def current_system(req: dict) -> str:
    """The request's system prompt as text ("" if none)."""
    if req.get("api") in ("chat.completions", "langchain"):
        parts = [m.get("content") for m in req.get("messages") or []
                 if isinstance(m, dict) and m.get("role") in ("system", "developer")]
    else:
        parts = [req.get("system")]
    out = []
    for p in parts:
        if isinstance(p, str):
            out.append(p)
        elif isinstance(p, list):
            out.extend(b.get("text", "") for b in p if isinstance(b, dict))
    return "\n\n".join(x for x in out if x)


# ---------------------------------------------------------------- what-if


def edited_request(
    trace: Trace,
    request_id: int,
    *,
    drop: Iterable[str] = (),
    replace: dict[str, str] | None = None,
    system: str | None = None,
    model_name: str | None = None,
    fill: str | None = None,
    add_system: str | None = None,
) -> tuple[dict, list[str]]:
    """The recorded request with edits applied. Returns (request, notes on what changed)."""
    from .segments import ablate, extract, fill_text, find, replace_text

    req = build_request(trace, request_id)
    notes = []
    drop = list(drop)
    if drop:
        segs = extract(req, trace, request_id)
        chosen = []
        for ref in drop:
            found = find(segs, ref)
            chosen.extend(found)
            notes.append(f"dropped {', '.join(s.where for s in found)}")
        req = ablate(req, chosen, fill_text(fill))
    for old, new in (replace or {}).items():
        if not old:
            raise ValueError("replace needs non-empty text to find")
        req, n = replace_text(req, old, new)
        if n == 0:
            raise ValueError(f"'{old}' doesn't appear anywhere in the context")
        notes.append(f"replaced '{old}' ({n}x)")
    if add_system and system is None and req.get("api") in ("chat.completions", "langchain"):
        # add to the first system (or developer) message and leave any others where they are
        msgs = req.get("messages") or []
        first = next((m for m in msgs if isinstance(m, dict) and m.get("role") in ("system", "developer")), None)
        if first is None:
            req["messages"] = [{"role": "system", "content": add_system}] + list(msgs)
        elif isinstance(first.get("content"), list):
            first["content"] = list(first["content"]) + [{"type": "text", "text": add_system}]
        else:
            first["content"] = ((first.get("content") or "") + "\n\n" + add_system).strip()
        notes.append("added to the system prompt")
        add_system = None
    elif add_system:
        base = system if system is not None else current_system(req)
        system = (base + "\n\n" + add_system).strip() if base else add_system
        notes.append("added to the system prompt")
    if system is not None:
        if req.get("api") == "chat.completions" or (req.get("api") == "langchain"):
            msgs = [m for m in req["messages"] if not (isinstance(m, dict) and m.get("role") in ("system", "developer"))]
            req["messages"] = [{"role": "system", "content": system}] + msgs
        else:
            req["system"] = system
        if not add_system:
            notes.append("new system prompt")
    if model_name:
        req["model"] = model_name
        notes.append(f"model {model_name}")
    return req, notes


def rerun(
    trace: "Trace | str | os.PathLike",
    event_id: int,
    *,
    runs: int = 5,
    drop: "str | Iterable[str]" = (),
    replace: dict[str, str] | None = None,
    system: str | None = None,
    model_name: str | None = None,
    model: Model | Callable[[dict], Any] | None = None,
    cache_dir: str | None = ".runtape/cache",
    budget: int | None = 100,
    workers: int = 8,
    fill: str | None = None,
    add_system: str | None = None,
) -> Distribution:
    """Re-run one decision from a trace, as recorded or with edits, and return what the model did.

    Works as a regression test for agent bugs:

        runtape.rerun("traces/bad-refund.jsonl", 30, system=FIXED_PROMPT).never_calls("issue_refund")
    """
    if not isinstance(trace, Trace):
        trace = Trace.load(trace)
    if isinstance(drop, (str, int)):  # one reference, not a list of characters
        drop = [str(drop)]
    rid, resp = request_for(trace, event_id)
    req, notes = edited_request(trace, rid, drop=drop, replace=replace, system=system, model_name=model_name,
                                fill=fill, add_system=add_system)
    if model is None:
        model = model_for(req)
    elif not isinstance(model, (AnthropicModel, OpenAIChatModel, OpenAIResponsesModel, FunctionModel)):
        model = FunctionModel(model)
    sampler = Sampler(model, cache_dir=cache_dir, budget=budget, workers=safe_workers(model, workers))
    if runs < 1:
        raise ValueError("runs must be at least 1")
    k = min(runs, 2) if is_deterministic(req) else runs
    dist = Distribution(sampler.samples(req, k))
    warn = server_side_context(req)
    dist.notes = notes + ([warn] if warn else [])
    dist.request_id, dist.response_id = rid, resp
    rp = trace[resp].payload if resp is not None else {}
    dist.recorded = Reply(rp.get("text"), list(rp.get("tool_calls") or []), rp.get("stop_reason")) if rp else None
    dist.model_calls = sampler.calls
    return dist
