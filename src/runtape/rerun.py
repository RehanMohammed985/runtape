"""Rebuild recorded model calls and send them again.

This is what makes runtape an experimental debugger instead of a log viewer:
every decision in a trace can be re-executed, as recorded or modified, without
running the agent or repeating any of its side effects.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
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
            # unlinked tool: the latest model reply that asked for a tool of this name
            for e in reversed(trace.events):
                if e.id < ev.id and e.type == "llm_response" and any(
                    tc.get("name") == ev.payload.get("name") for tc in e.payload.get("tool_calls") or []
                ):
                    rid = e.id
                    break
        if rid is None:
            raise ValueError(f"#{event_id}: can't find the model reply that requested this tool call")
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
        else:
            s = "replies: " + " ".join((self.text or "").split())
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


class AnthropicModel:
    def __init__(self, client: Any = None):
        self._client = client  # created on first call, so no API key is needed until then

    @property
    def client(self):
        if self._client is None:
            self._client = self._make()
        return self._client


    @staticmethod
    def _make():
        import anthropic

        client = anthropic.Anthropic(max_retries=8)  # why runs many calls; ride out rate limits
        return client

    def __call__(self, req: dict) -> Reply:
        from .integrations import _Anthropic

        messages, system = req["messages"], req.get("system")
        if req.get("api") == "langchain":
            system, messages = openai_to_anthropic(messages)
            kw = _filter_params(req["params"], _ANTHROPIC_PARAMS, {"stop": "stop_sequences"})
        else:
            kw = dict(req["params"])
        kw.setdefault("max_tokens", 1024)
        if system is not None:
            kw["system"] = system
        if req.get("tools"):
            kw["tools"] = req["tools"]
        resp = self.client.messages.create(model=req["model"], messages=messages, **kw)
        out = _Anthropic.response(resp)
        return Reply(out["text"], out["tool_calls"], out["stop_reason"], out["raw"])


class OpenAIChatModel:
    def __init__(self, client: Any = None):
        self._client = client  # created on first call, so no API key is needed until then

    @property
    def client(self):
        if self._client is None:
            self._client = self._make()
        return self._client


    @staticmethod
    def _make():
        import openai

        client = openai.OpenAI(max_retries=8)
        return client

    def __call__(self, req: dict) -> Reply:
        from .integrations import _OpenAIChat

        kw = _filter_params(req["params"], _OPENAI_PARAMS) if req.get("api") == "langchain" else dict(req["params"])
        if req.get("tools"):
            kw["tools"] = req["tools"]
        resp = self.client.chat.completions.create(model=req["model"], messages=req["messages"], **kw)
        out = _OpenAIChat.response(resp)
        return Reply(out["text"], out["tool_calls"], out["stop_reason"], out["raw"])


class OpenAIResponsesModel:
    def __init__(self, client: Any = None):
        self._client = client  # created on first call, so no API key is needed until then

    @property
    def client(self):
        if self._client is None:
            self._client = self._make()
        return self._client


    @staticmethod
    def _make():
        import openai

        client = openai.OpenAI(max_retries=8)
        return client

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
    api, provider = req.get("api"), req.get("provider")
    if api == "messages" or provider == "anthropic":
        return AnthropicModel()
    if api == "responses":
        return OpenAIResponsesModel()
    if api == "chat.completions" or (api == "langchain" and provider == "openai"):
        return OpenAIChatModel()
    raise ValueError(
        f"Don't know how to resend a {provider}/{api} call. Pass a model function "
        "(--model-fn module:function) that takes the request dict and returns a reply."
    )


def load_model_fn(spec: str) -> Model:
    """'package.module:function' -> FunctionModel. Also accepts 'path/to/file.py:function'."""
    mod_name, _, attr = spec.partition(":")
    if not attr:
        raise ValueError("model function must look like module:function")
    if mod_name.endswith(".py") or os.sep in mod_name:
        import importlib.util

        spec_ = importlib.util.spec_from_file_location(Path(mod_name).stem, mod_name)
        mod = importlib.util.module_from_spec(spec_)
        spec_.loader.exec_module(mod)
    else:
        mod = importlib.import_module(mod_name)
    obj = getattr(mod, attr)
    if isinstance(obj, type):
        obj = obj()
    if isinstance(obj, (AnthropicModel, OpenAIChatModel, OpenAIResponsesModel, FunctionModel)):
        return obj
    return FunctionModel(obj)


# ------------------------------------------------------------------ sampler


class BudgetExceeded(RuntimeError):
    pass


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
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.budget = budget
        self.workers = workers
        self.calls = 0  # live calls made
        self.hits = 0  # served from cache
        self._lock = threading.Lock()
        self.on_call = on_call

    def _cache_path(self, req: dict, i: int) -> Path | None:
        if not self.cache_dir:
            return None
        return self.cache_dir / f"{request_key(req)[:40]}-{i}.json"

    def one(self, req: dict, i: int = 0) -> Reply:
        path = self._cache_path(req, i)
        if path and path.exists():
            try:
                with self._lock:
                    self.hits += 1
                return Reply.from_any(json.loads(path.read_text()))
            except ValueError:
                pass
        with self._lock:
            if self.budget is not None and self.calls >= self.budget:
                raise BudgetExceeded(f"hit the budget of {self.budget} model calls")
            self.calls += 1
        reply = Reply.from_any(self.model(req))
        if path:
            path.write_text(json.dumps(to_jsonable(reply.to_dict()), ensure_ascii=False))
        if self.on_call:
            self.on_call()
        return reply

    def many(self, jobs: Iterable[tuple[dict, int]]) -> list[Reply]:
        jobs = list(jobs)
        if self.workers <= 1 or len(jobs) <= 1:
            return [self.one(r, i) for r, i in jobs]
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(lambda job: self.one(*job), jobs))

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

    def __len__(self) -> int:
        return len(self.replies)


# ---------------------------------------------------------------- what-if


def edited_request(
    trace: Trace,
    request_id: int,
    *,
    drop: Iterable[str] = (),
    replace: dict[str, str] | None = None,
    system: str | None = None,
    model_name: str | None = None,
) -> tuple[dict, list[str]]:
    """The recorded request with edits applied. Returns (request, notes on what changed)."""
    from .segments import ablate, extract, find, replace_text

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
        req = ablate(req, chosen)
    for old, new in (replace or {}).items():
        req, n = replace_text(req, old, new)
        if n == 0:
            raise ValueError(f"'{old}' doesn't appear anywhere in the context")
        notes.append(f"replaced '{old}' ({n}x)")
    if system is not None:
        if req.get("api") == "chat.completions" or (req.get("api") == "langchain"):
            msgs = [m for m in req["messages"] if not (isinstance(m, dict) and m.get("role") in ("system", "developer"))]
            req["messages"] = [{"role": "system", "content": system}] + msgs
        else:
            req["system"] = system
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
    drop: Iterable[str] = (),
    replace: dict[str, str] | None = None,
    system: str | None = None,
    model_name: str | None = None,
    model: Model | Callable[[dict], Any] | None = None,
    cache_dir: str | None = ".runtape/cache",
    budget: int | None = 100,
    workers: int = 8,
) -> Distribution:
    """Re-run one decision from a trace, as recorded or with edits, and return what the model did.

    Works as a regression test for agent bugs:

        runtape.rerun("traces/bad-refund.jsonl", 30, system=FIXED_PROMPT).never_calls("issue_refund")
    """
    if not isinstance(trace, Trace):
        trace = Trace.load(trace)
    rid, resp = request_for(trace, event_id)
    req, notes = edited_request(trace, rid, drop=drop, replace=replace, system=system, model_name=model_name)
    if model is None:
        model = model_for(req)
    elif not isinstance(model, (AnthropicModel, OpenAIChatModel, OpenAIResponsesModel, FunctionModel)):
        model = FunctionModel(model)
    sampler = Sampler(model, cache_dir=cache_dir, budget=budget, workers=workers)
    k = 1 if is_deterministic(req) else runs
    dist = Distribution(sampler.samples(req, k))
    dist.notes = notes
    dist.request_id, dist.response_id = rid, resp
    rp = trace[resp].payload if resp is not None else {}
    dist.recorded = Reply(rp.get("text"), list(rp.get("tool_calls") or []), rp.get("stop_reason")) if rp else None
    dist.model_calls = sampler.calls
    return dist
