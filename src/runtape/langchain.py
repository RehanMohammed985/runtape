"""LangChain / LangGraph integration.

    rec = runtape.record(name="my-graph")
    graph.invoke(inputs, config={"callbacks": [rec.langchain()]})

Records every chat model call and tool run inside the chain or graph, in the
same format as the OpenAI/Anthropic wrappers.
"""
from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from .serialize import to_jsonable

if TYPE_CHECKING:
    from .recorder import Recorder


def _to_dicts(messages: list) -> list[dict]:
    try:
        from langchain_core.messages import convert_to_openai_messages

        return convert_to_openai_messages(messages)
    except Exception:
        out = []
        for m in messages:
            d = {"role": getattr(m, "type", "?"), "content": to_jsonable(getattr(m, "content", m))}
            if getattr(m, "tool_calls", None):
                d["tool_calls"] = to_jsonable(m.tool_calls)
            out.append(d)
        return out


def _model_name(serialized: dict | None, kwargs: dict) -> str | None:
    inv = kwargs.get("invocation_params") or {}
    meta = kwargs.get("metadata") or {}
    return (
        inv.get("model")
        or inv.get("model_name")
        or meta.get("ls_model_name")
        or (serialized or {}).get("name")
        or inv.get("_type")
    )


class RuntapeCallbackHandler(BaseCallbackHandler):
    """Pass as a callback to any LangChain runnable or LangGraph graph."""

    raise_error = False  # a recording problem must never break the user's chain

    def __init__(self, rec: "Recorder"):
        self.rec = rec
        self._runs: dict[UUID, tuple[int, float]] = {}  # langchain run id -> (event id, start)
        self._tool_names: dict[UUID, str] = {}
        self._lock = threading.Lock()

    # ---------------------------------------------------------- model calls

    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs):
        inv = kwargs.get("invocation_params") or {}
        rid = self.rec.log_llm_request(
            provider=(metadata or {}).get("ls_provider") or "langchain",
            model=_model_name(serialized, {**kwargs, "metadata": metadata}),
            messages=_to_dicts(messages[0] if messages else []),
            tools=inv.get("tools"),
            params={k: v for k, v in inv.items() if k not in ("tools", "model", "model_name", "_type")},
        )
        with self._lock:
            self._runs[run_id] = (rid, time.perf_counter())

    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs):
        # plain (non-chat) LLMs: treat each prompt as a user message
        rid = self.rec.log_llm_request(
            provider=(metadata or {}).get("ls_provider") or "langchain",
            model=_model_name(serialized, {**kwargs, "metadata": metadata}),
            messages=[{"role": "user", "content": p} for p in prompts[:1]],
        )
        with self._lock:
            self._runs[run_id] = (rid, time.perf_counter())

    def on_llm_end(self, response, *, run_id, parent_run_id=None, **kwargs):
        with self._lock:
            started = self._runs.pop(run_id, None)
        if started is None:
            return
        rid, t0 = started
        gen = (response.generations or [[None]])[0][0]
        msg = getattr(gen, "message", None)
        text = None
        calls: list[dict] = []
        tokens: dict = {}
        stop = None
        if msg is not None:
            content = msg.content
            if isinstance(content, str):
                text = content or None
            else:
                parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
                text = "".join(parts) or None
            calls = [
                {"id": tc.get("id"), "name": tc.get("name"), "arguments": to_jsonable(tc.get("args"))}
                for tc in (getattr(msg, "tool_calls", None) or [])
            ]
            usage = getattr(msg, "usage_metadata", None) or {}
            tokens = {"input": usage.get("input_tokens"), "output": usage.get("output_tokens")}
            rm = getattr(msg, "response_metadata", None) or {}
            stop = rm.get("finish_reason") or rm.get("stop_reason")
        elif gen is not None:
            text = getattr(gen, "text", None)
        if not tokens.get("input") and response.llm_output:
            u = (response.llm_output or {}).get("token_usage") or {}
            tokens = {"input": u.get("prompt_tokens"), "output": u.get("completion_tokens")}
        self.rec.log_llm_response(
            rid,
            text=text,
            tool_calls=calls,
            stop_reason=stop,
            raw=to_jsonable(msg if msg is not None else gen),
            latency_ms=(time.perf_counter() - t0) * 1000,
            tokens=tokens,
            model=(getattr(msg, "response_metadata", None) or {}).get("model_name"),
        )

    def on_llm_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        with self._lock:
            started = self._runs.pop(run_id, None)
        self.rec.error(error, parent=started[0] if started else None)

    # ----------------------------------------------------------------- tools

    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, tags=None, metadata=None, inputs=None, **kwargs):
        name = (serialized or {}).get("name") or kwargs.get("name") or "tool"
        payload: dict[str, Any] = {"name": name, "arguments": inputs if inputs is not None else input_str}
        meta: dict[str, Any] = {}
        linked = self.rec._claim_tool_call(name)
        if linked:
            payload["call_id"] = linked["id"]
            meta["requested_by"] = linked["response_event"]
        cid = self.rec.log("tool_call", payload, meta=meta)
        with self._lock:
            self._runs[run_id] = (cid, time.perf_counter())
            self._tool_names[run_id] = name

    def on_tool_end(self, output, *, run_id, parent_run_id=None, **kwargs):
        with self._lock:
            started = self._runs.pop(run_id, None)
            name = self._tool_names.pop(run_id, None)
        if started is None:
            return
        cid, t0 = started
        result = getattr(output, "content", output)  # ToolMessage -> its content
        self.rec.log(
            "tool_result",
            {"name": name, "result": result},
            meta={"latency_ms": round((time.perf_counter() - t0) * 1000, 2)},
            parent=cid,
        )

    def on_tool_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        with self._lock:
            started = self._runs.pop(run_id, None)
            name = self._tool_names.pop(run_id, None)
        from .recorder import _exc_payload

        cid, t0 = started if started else (None, time.perf_counter())
        self.rec.log(
            "tool_result",
            {"name": name, "error": _exc_payload(error)},
            meta={"latency_ms": round((time.perf_counter() - t0) * 1000, 2)},
            parent=cid,
        )

    # --------------------------------------------------------------- chains

    def on_chain_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        if parent_run_id is None:  # only the top-level failure, not every layer it bubbles through
            self.rec.error(error)
