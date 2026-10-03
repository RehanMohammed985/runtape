"""Import traces from other tools, so `why`, `fix` and `test` run on an agent you already trace.

    runtape import spans.json                 # an OpenTelemetry export
    runtape import langfuse-trace.json        # a Langfuse trace, as its API returns it
    runtape import langfuse:<trace id>        # straight from Langfuse (LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY,
                                              # LANGFUSE_HOST)

OpenTelemetry: an OTLP JSON export (one {"resourceSpans": ...} object, or one per line, as the collector's file
exporter writes them), or spans as the Python SDK's console exporter prints them. The model calls are read from
whichever convention the instrumentation used:

- the GenAI semantic conventions: gen_ai.input.messages, gen_ai.output.messages, gen_ai.system_instructions,
  gen_ai.tool.definitions
- OpenLLMetry / Traceloop: gen_ai.prompt.N.*, gen_ai.completion.N.*, llm.request.functions.N.*
- OpenInference / Arize Phoenix: input.value (the request as sent), llm.input_messages.N.*,
  llm.output_messages.N.*, llm.tools.N.tool.json_schema

Langfuse: the generations of a trace (its OpenAI and LangChain integrations log the messages, and the tools
when there are any).

Every model call becomes a request and a reply, and every tool call the model made becomes a tool call and its
result (read from the next request's history), so the trace reads as if runtape had recorded it. A rerun sends
the request again, so it needs what was sent: without the tool definitions the model can't call a tool, and a
decision to call one can't be repeated. The import says when they are missing; --tools supplies them.
"""
from __future__ import annotations

import base64
import datetime as _dt
import json
import os
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .recorder import Recorder


@dataclass
class Call:
    """One model call, in OpenAI chat format whatever the source."""

    start: float
    model: str | None
    provider: str | None
    messages: list[dict]
    tools: list[dict] | None
    params: dict
    text: str | None
    tool_calls: list[dict]  # [{"id", "name", "arguments": dict}]
    stop_reason: str | None = None
    trace_id: str | None = None
    endpoint: str | None = None
    source: str = ""
    span_id: str | None = None
    parent_span: str | None = None
    end: float | None = None  # seconds, like start
    tokens: dict | None = None  # {"input": n, "output": n}


@dataclass
class Imported:
    path: Path
    calls: int
    tool_calls: int
    missing_tools: int  # model calls whose tool definitions weren't in the source
    provider: str
    notes: list[str] = field(default_factory=list)


# ------------------------------------------------------------------ messages


_ROLES = {"human": "user", "ai": "assistant", "model": "assistant", "developer": "system", "function": "tool"}
_OPENAI_PARTS = {"text", "image_url", "input_audio", "file", "refusal"}


def _args_text(a: Any) -> str:
    return a if isinstance(a, str) else json.dumps(a if a is not None else {})


def _args_dict(a: Any) -> Any:
    if isinstance(a, str):
        try:
            return json.loads(a) if a.strip() else {}
        except ValueError:
            return a
    return a if a is not None else {}


def _tool_call(tc: dict) -> dict | None:
    """Any tool call shape -> OpenAI's {"id", "type": "function", "function": {"name", "arguments": str}}."""
    if not isinstance(tc, dict):
        return None
    if "tool_call" in tc and isinstance(tc["tool_call"], dict):  # OpenInference
        tc = tc["tool_call"]
    fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
    name = fn.get("name") or tc.get("name")
    if not name:
        return None
    args = fn.get("arguments", tc.get("arguments", tc.get("input", tc.get("args"))))
    return {"id": tc.get("id") or tc.get("call_id") or "", "type": "function",
            "function": {"name": name, "arguments": _args_text(args)}}


def _text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for p in content:
            if isinstance(p, str):
                out.append(p)
            elif isinstance(p, dict):
                v = p.get("text", p.get("content"))
                if isinstance(v, str):
                    out.append(v)
                elif v is not None:
                    out.append(json.dumps(v))
        return "\n".join(out)
    return json.dumps(content)


def _openai_part(x: Any) -> bool:
    if not isinstance(x, dict) or x.get("type") not in _OPENAI_PARTS:
        return False
    return x["type"] != "text" or "text" in x  # {"type": "text", "content": ...} is another convention's part


def openai_messages(messages: list) -> list[dict]:
    """Messages in any common shape (OpenAI, Anthropic blocks, GenAI semantic convention parts, LangChain roles)
    -> OpenAI chat messages."""
    out: list[dict] = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role = _ROLES.get(m.get("role"), m.get("role") or "user")
        content = m.get("content")
        parts = m.get("parts") if isinstance(m.get("parts"), list) else None
        calls = [c for c in (_tool_call(tc) for tc in (m.get("tool_calls") or [])) if c]
        results: list[dict] = []
        text_parts: list[str] = []
        blocks = parts if parts is not None else (content if isinstance(content, list) else None)
        if parts is None and isinstance(content, list) and all(_openai_part(x) for x in content):
            blocks = None  # already OpenAI content parts (text, images, audio): sent back as they were
        if blocks is not None:
            plain = []
            for b in blocks:
                if not isinstance(b, dict):
                    plain.append(b)
                    continue
                t = b.get("type")
                if t in ("tool_call", "tool_use", "function_call"):
                    c = _tool_call(b)
                    if c:
                        calls.append(c)
                elif t in ("tool_call_response", "tool_result", "function_call_output"):
                    res = b.get("result", b.get("response", b.get("content", b.get("output"))))
                    results.append({"role": "tool", "tool_call_id": b.get("id") or b.get("tool_use_id")
                                    or b.get("call_id") or "", "content": _text(res) if not isinstance(res, str)
                                    else res})
                elif t in ("thinking", "redacted_thinking", "reasoning"):
                    continue  # not something a rerun can send back as text
                else:
                    plain.append(b)
            text_parts.append(_text(plain))
            content = "\n".join(x for x in text_parts if x)
        if role == "tool" and not results:
            out.append({"role": "tool", "tool_call_id": m.get("tool_call_id") or m.get("id") or "",
                        "content": content if isinstance(content, str) else _text(content)})
            continue
        if results and not calls and role in ("user", "tool"):
            out.extend(results)  # Anthropic and the semantic conventions carry results inside a message
            rest = content if isinstance(content, str) else _text(content)
            if rest.strip():
                out.append({"role": "user", "content": rest})
            continue
        msg: dict = {"role": role, "content": content if isinstance(content, (str, list)) or content is None
                     else _text(content)}
        if calls:
            msg["tool_calls"] = calls
            if not msg["content"]:
                msg["content"] = None
        out.append(msg)
        out.extend(results)
    return out


def openai_tools(tools: Any) -> list[dict] | None:
    """Tool definitions in any common shape -> OpenAI's [{"type": "function", "function": {...}}]."""
    if not tools:
        return None
    if isinstance(tools, str):
        try:
            tools = json.loads(tools)
        except ValueError:
            return None
    out = []
    for t in tools if isinstance(tools, list) else [tools]:
        if isinstance(t, str):
            try:
                t = json.loads(t)
            except ValueError:
                continue
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if isinstance(t.get("function"), dict) else t
        if not fn.get("name"):
            continue
        params = fn.get("parameters", fn.get("input_schema", fn.get("inputSchema")))
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except ValueError:
                params = None
        d = {"name": fn["name"]}
        if fn.get("description"):
            d["description"] = fn["description"]
        d["parameters"] = params or {"type": "object", "properties": {}}
        out.append({"type": "function", "function": d})
    return out or None


def to_anthropic(messages: list[dict], tools: list[dict] | None) -> tuple[Any, list[dict], list[dict] | None]:
    """OpenAI chat messages -> (system, Anthropic messages, Anthropic tools), so a call made to Claude is resent
    to Claude in its own format."""
    system = "\n\n".join(_text(m.get("content")) for m in messages if m.get("role") == "system") or None
    out: list[dict] = []

    def add(role: str, blocks: list[dict]) -> None:
        if out and out[-1]["role"] == role:
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": role, "content": blocks})

    for m in messages:
        role = m.get("role")
        if role == "system":
            continue
        if role == "tool":
            add("user", [{"type": "tool_result", "tool_use_id": m.get("tool_call_id") or "",
                          "content": _text(m.get("content"))}])
            continue
        blocks: list[dict] = []
        t = _text(m.get("content"))
        if t:
            blocks.append({"type": "text", "text": t})
        for tc in m.get("tool_calls") or []:
            blocks.append({"type": "tool_use", "id": tc.get("id") or "", "name": tc["function"]["name"],
                           "input": _args_dict(tc["function"].get("arguments")) or {}})
        if blocks:
            add("assistant" if role == "assistant" else "user", blocks)
    atools = None
    if tools:
        atools = [{"name": t["function"]["name"], "description": t["function"].get("description", ""),
                   "input_schema": t["function"].get("parameters") or {"type": "object", "properties": {}}}
                  for t in tools]
    return system, out, atools


# ------------------------------------------------------------------ OpenTelemetry


def _any_value(v: Any) -> Any:
    if not isinstance(v, dict):
        return v
    for k in ("stringValue", "boolValue", "doubleValue"):
        if k in v:
            return v[k]
    if "intValue" in v:
        return int(v["intValue"])
    if "arrayValue" in v:
        return [_any_value(x) for x in (v["arrayValue"].get("values") or [])]
    if "kvlistValue" in v:
        return {x["key"]: _any_value(x.get("value")) for x in (v["kvlistValue"].get("values") or [])}
    if "bytesValue" in v:
        return v["bytesValue"]
    return None


def _attrs(a: Any) -> dict:
    if isinstance(a, dict):
        return dict(a)
    return {x["key"]: _any_value(x.get("value")) for x in a or [] if isinstance(x, dict) and "key" in x}


def _time(v: Any) -> float:
    if v is None:
        return 0.0
    if isinstance(v, (int, float)) or (isinstance(v, str) and v.isdigit()):
        n = float(v)
        return n / 1e9 if n > 1e14 else n  # OTLP gives nanoseconds
    try:
        return _dt.datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _spans(data: Any) -> list[dict]:
    """Spans from an OTLP JSON export, the console exporter's output, or a plain list of spans."""
    out: list[dict] = []
    items = data if isinstance(data, list) else [data]
    for item in items:
        if not isinstance(item, dict):
            continue
        if "resourceSpans" in item or "resource_spans" in item:
            for rs in item.get("resourceSpans") or item.get("resource_spans") or []:
                res = _attrs((rs.get("resource") or {}).get("attributes"))
                for ss in rs.get("scopeSpans") or rs.get("scope_spans") or rs.get("instrumentationLibrarySpans") or []:
                    for s in ss.get("spans") or []:
                        out.append({"trace_id": s.get("traceId") or s.get("trace_id"),
                                    "span_id": s.get("spanId") or s.get("span_id"),
                                    "parent": s.get("parentSpanId") or s.get("parent_span_id"),
                                    "name": s.get("name"), "start": _time(s.get("startTimeUnixNano")),
                                    "end": _time(s.get("endTimeUnixNano")),
                                    "attrs": _attrs(s.get("attributes")), "events": s.get("events") or [],
                                    "resource": res})
        elif "attributes" in item and ("name" in item or "context" in item):
            ctx = item.get("context") or {}
            out.append({"trace_id": ctx.get("trace_id") or item.get("trace_id") or item.get("traceId"),
                        "span_id": ctx.get("span_id") or item.get("span_id") or item.get("spanId"),
                        "parent": item.get("parent_id") or item.get("parentSpanId"),
                        "name": item.get("name"),
                        "start": _time(item.get("start_time") or item.get("startTimeUnixNano")),
                        "end": _time(item.get("end_time") or item.get("endTimeUnixNano")),
                        "attrs": _attrs(item.get("attributes")), "events": item.get("events") or [],
                        "resource": _attrs((item.get("resource") or {}).get("attributes"))})
    return out


def _unflatten(attrs: dict, prefix: str) -> Any:
    """{"p.0.role": "user", "p.0.tool_calls.1.name": "x"} -> [{"role": "user", "tool_calls": [None, {"name": "x"}]}]"""
    tree: dict = {}
    for k, v in attrs.items():
        if not k.startswith(prefix + "."):
            continue
        node = tree
        keys = k[len(prefix) + 1:].split(".")
        for key in keys[:-1]:
            node = node.setdefault(key, {})
            if not isinstance(node, dict):
                break
        else:
            node[keys[-1]] = v

    def fix(n: Any) -> Any:
        if not isinstance(n, dict):
            return n
        if n and all(k.isdigit() for k in n):
            return [fix(n[k]) for k in sorted(n, key=int)]
        return {k: fix(v) for k, v in n.items()}

    return fix(tree) if tree else None


def _json(v: Any) -> Any:
    if isinstance(v, str):
        try:
            return json.loads(v)
        except ValueError:
            return None
    return v


def _provider(a: dict) -> str | None:
    p = a.get("gen_ai.provider.name") or a.get("gen_ai.system") or a.get("llm.provider") or a.get("llm.system")
    if not p:
        return None
    p = str(p).lower()
    if "anthropic" in p or "claude" in p:
        return "anthropic"
    if "openai" in p or p in ("azure", "az.ai.openai", "azure.ai.openai"):
        return "openai"
    return p


_GENAI_PARAMS = {"temperature": "temperature", "max_tokens": "max_tokens", "top_p": "top_p", "seed": "seed",
                 "frequency_penalty": "frequency_penalty", "presence_penalty": "presence_penalty",
                 "stop_sequences": "stop"}


def _span_call(s: dict) -> Call | None:
    a = s["attrs"]
    kind = a.get("openinference.span.kind")
    has_content = any(k.startswith(("gen_ai.prompt.", "llm.input_messages.")) for k in a) or \
        "gen_ai.input.messages" in a or (kind == "LLM" and "input.value" in a)
    if not has_content and not _event_messages(s):
        return None
    if kind and kind != "LLM":
        return None  # a chain or agent span that repeats its LLM child's content
    model = a.get("gen_ai.request.model") or a.get("llm.model_name") or a.get("gen_ai.response.model")
    params = {p: a[f"gen_ai.request.{k}"] for k, p in _GENAI_PARAMS.items() if f"gen_ai.request.{k}" in a}
    messages: list | None = None
    tools = None
    text, calls, stop = None, [], None
    source = ""

    # OpenInference: the request exactly as sent, when the instrumentation kept it
    raw_in = _json(a.get("input.value")) if kind == "LLM" else None
    if isinstance(raw_in, dict) and isinstance(raw_in.get("messages"), list):
        messages = raw_in["messages"]
        tools = raw_in.get("tools") or raw_in.get("functions")
        params.update({k: v for k, v in raw_in.items() if k not in ("messages", "tools", "functions", "model",
                                                                     "stream", "stream_options", "n")})
        model = model or raw_in.get("model")
        source = "OpenInference (request as sent)"
    if messages is None and "gen_ai.input.messages" in a:
        messages = list(_json(a["gen_ai.input.messages"]) or [])
        sys_parts = _json(a.get("gen_ai.system_instructions"))
        if sys_parts:
            messages.insert(0, {"role": "system", "content": _text(sys_parts)})
        source = "GenAI semantic conventions"
    if messages is None and any(k.startswith("gen_ai.prompt.") for k in a):
        messages = [m for m in (_unflatten(a, "gen_ai.prompt") or []) if m]
        source = "OpenLLMetry"
    if messages is None and any(k.startswith("llm.input_messages.") for k in a):
        messages = [(m or {}).get("message") or {} for m in (_unflatten(a, "llm.input_messages") or [])]
        for m in messages:
            if m.get("contents") and not m.get("content"):
                m["content"] = [c.get("message_content") or c for c in m["contents"] if c]
        source = "OpenInference"
    if messages is None:
        messages = _event_messages(s)
        source = "GenAI events"

    if tools is None:
        tools = _json(a.get("gen_ai.tool.definitions"))
    if tools is None and any(k.startswith("llm.request.functions.") for k in a):
        tools = _unflatten(a, "llm.request.functions")
    if tools is None and any(k.startswith("llm.tools.") for k in a):
        tools = [(t or {}).get("tool", {}).get("json_schema") for t in (_unflatten(a, "llm.tools") or [])]
    inv = _json(a.get("llm.invocation_parameters"))
    if isinstance(inv, dict):
        tools = tools or inv.get("tools")
        params.update({k: v for k, v in inv.items() if k not in ("tools", "model", "messages", "stream", "n")})

    # the reply
    out_msgs = None
    if "gen_ai.output.messages" in a:
        out_msgs = _json(a["gen_ai.output.messages"])
    elif any(k.startswith("gen_ai.completion.") for k in a):
        out_msgs = []
        for c in _unflatten(a, "gen_ai.completion") or []:
            if not c:
                continue
            if c.get("function_call") and not c.get("tool_calls"):
                c["tool_calls"] = [c["function_call"]]
            out_msgs.append({"role": "assistant", **c})
    elif any(k.startswith("llm.output_messages.") for k in a):
        out_msgs = [(m or {}).get("message") or {} for m in (_unflatten(a, "llm.output_messages") or [])]
    else:
        raw_out = _json(a.get("output.value"))
        if isinstance(raw_out, dict) and raw_out.get("choices"):
            ch = raw_out["choices"][0]
            out_msgs = [{**(ch.get("message") or {}), "finish_reason": ch.get("finish_reason")}]
    if not out_msgs:
        out_msgs = _event_choices(s)
    if out_msgs:
        first = out_msgs[0] if isinstance(out_msgs[0], dict) else {}
        reply = openai_messages([{**first, "role": "assistant"}])
        r = next((x for x in reply if x.get("role") == "assistant"), {})
        text = _text(r.get("content")) or None
        calls = [{"id": tc["id"], "name": tc["function"]["name"], "arguments": _args_dict(tc["function"]["arguments"])}
                 for tc in r.get("tool_calls") or []]
        stop = first.get("finish_reason")
    reasons = a.get("gen_ai.response.finish_reasons")
    if not stop and isinstance(reasons, list) and reasons:
        stop = reasons[0]
    # server.address names the host but not the path (/v1, /openai/v1), so a non-default server is set with
    # --base-url rather than guessed
    return Call(start=s["start"], model=model, provider=_provider(a), messages=openai_messages(messages),
                tools=openai_tools(tools), params=_clean_params(params), text=text, tool_calls=calls,
                stop_reason=stop, trace_id=s.get("trace_id"), source=source, span_id=s.get("span_id"),
                parent_span=s.get("parent"), end=s.get("end"), tokens=_usage(a))


def _usage(a: dict) -> dict | None:
    """Token counts from the GenAI, OpenLLMetry or OpenInference attributes."""
    def first(*keys):
        for k in keys:
            v = a.get(k)
            if isinstance(v, (int, float)) or (isinstance(v, str) and v.isdigit()):
                return int(v)
        return None
    tin = first("gen_ai.usage.input_tokens", "gen_ai.usage.prompt_tokens", "llm.token_count.prompt")
    tout = first("gen_ai.usage.output_tokens", "gen_ai.usage.completion_tokens", "llm.token_count.completion")
    return {"input": tin, "output": tout} if tin is not None or tout is not None else None


def _event_messages(s: dict) -> list[dict]:
    """The older GenAI convention: one span event per message (gen_ai.user.message, ...)."""
    out = []
    for e in s.get("events") or []:
        name = e.get("name") or ""
        m = re.match(r"gen_ai\.(system|user|assistant|tool)\.message$", name)
        if not m:
            continue
        body = _attrs(e.get("attributes"))
        d = _json(body.get("body")) if "body" in body else body
        d = d if isinstance(d, dict) else {"content": body.get("content")}
        out.append({"role": m.group(1), **{k: v for k, v in d.items() if k != "role"}})
    return out


def _event_choices(s: dict) -> list[dict]:
    for e in s.get("events") or []:
        if e.get("name") == "gen_ai.choice":
            body = _attrs(e.get("attributes"))
            d = _json(body.get("body")) if "body" in body else body
            if isinstance(d, dict):
                msg = d.get("message") if isinstance(d.get("message"), dict) else d
                return [{**msg, "finish_reason": d.get("finish_reason")}]
    return []


def _clean_params(params: dict) -> dict:
    out = {}
    for k, v in params.items():
        if v is None or v == "inf" or v == float("inf"):
            continue
        if isinstance(v, str):
            try:
                v = int(v) if re.fullmatch(r"-?\d+", v) else float(v) if re.fullmatch(r"-?\d*\.\d+", v) else v
            except ValueError:
                pass
        out[k] = v
    return out


# ------------------------------------------------------------------ Langfuse


def _langfuse_calls(data: Any) -> list[Call]:
    obs = data.get("observations") if isinstance(data, dict) else data
    out = []
    for o in obs or []:
        if not isinstance(o, dict) or o.get("type") != "GENERATION":
            continue
        inp, tools = o.get("input"), None
        if isinstance(inp, dict) and isinstance(inp.get("messages"), list):
            tools = inp.get("tools") or inp.get("functions")
            msgs = inp["messages"]
        elif isinstance(inp, list):
            msgs = inp
        elif isinstance(inp, str):
            msgs = [{"role": "user", "content": inp}]
        else:
            continue
        outp = o.get("output")
        if isinstance(outp, dict) and isinstance(outp.get("choices"), list) and outp["choices"]:
            outp = outp["choices"][0].get("message") or outp["choices"][0]
        if isinstance(outp, list) and outp and isinstance(outp[0], dict):
            outp = outp[0]
        if isinstance(outp, str):
            outp = {"content": outp}
        reply = openai_messages([{**(outp or {}), "role": "assistant"}]) if isinstance(outp, dict) else []
        r = next((x for x in reply if x.get("role") == "assistant"), {})
        meta = o.get("metadata") or {}
        model = o.get("model") or (o.get("modelParameters") or {}).get("model")
        provider = meta.get("ls_provider") if isinstance(meta, dict) else None
        if not provider and model and "claude" in str(model).lower():
            provider = "anthropic"
        out.append(Call(
            start=_time(o.get("startTime")), model=model, provider=_provider({"gen_ai.system": provider})
            if provider else None, messages=openai_messages(msgs), tools=openai_tools(tools),
            params=_clean_params({k: v for k, v in (o.get("modelParameters") or {}).items() if k != "model"}),
            text=_text(r.get("content")) or None,
            tool_calls=[{"id": tc["id"], "name": tc["function"]["name"], "arguments": _args_dict(tc["function"]["arguments"])}
                        for tc in r.get("tool_calls") or []],
            stop_reason=(outp or {}).get("finish_reason") if isinstance(outp, dict) else None,
            trace_id=o.get("traceId"), source="Langfuse", end=_time(o.get("endTime")) if o.get("endTime") else None,
            tokens=_langfuse_usage(o)))
    return out


def fetch_langfuse(trace_id: str) -> dict:
    """A trace from the Langfuse API, with the keys from the environment."""
    host = (os.environ.get("LANGFUSE_HOST") or os.environ.get("LANGFUSE_BASE_URL") or "https://cloud.langfuse.com").rstrip("/")
    pk, sk = os.environ.get("LANGFUSE_PUBLIC_KEY"), os.environ.get("LANGFUSE_SECRET_KEY")
    if not (pk and sk):
        raise ValueError("Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY (and LANGFUSE_HOST if not on Langfuse "
                         "Cloud EU) to import from Langfuse, or export the trace and import the file.")
    req = urllib.request.Request(f"{host}/api/public/traces/{trace_id}")
    req.add_header("Authorization", "Basic " + base64.b64encode(f"{pk}:{sk}".encode()).decode())
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


# ------------------------------------------------------------------ reading


def load_source(source: str) -> Any:
    if source.startswith("langfuse:"):
        return fetch_langfuse(source.split(":", 1)[1])
    text = Path(source).read_text()
    try:
        return json.loads(text)
    except ValueError:
        items = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                try:
                    items.append(json.loads(line))
                except ValueError:
                    continue
        if not items:
            raise ValueError(f"{source} is not JSON or JSON lines")
        return items


def _langfuse_usage(o: dict) -> dict | None:
    u = o.get("usageDetails") or o.get("usage") or {}
    tin = u.get("input") if isinstance(u, dict) else None
    tout = u.get("output") if isinstance(u, dict) else None
    tin = tin if tin is not None else o.get("promptTokens")
    tout = tout if tout is not None else o.get("completionTokens")
    return {"input": tin, "output": tout} if tin is not None or tout is not None else None


def read_calls(data: Any) -> tuple[str, list[Call]]:
    """(the format found, the model calls in it, in order)."""
    obs = data.get("observations") if isinstance(data, dict) else data if isinstance(data, list) else None
    if isinstance(obs, list) and any(isinstance(o, dict) and o.get("type") == "GENERATION" for o in obs):
        calls = _langfuse_calls(data)
        fmt = "Langfuse"
    else:
        flat = []
        for d in data if isinstance(data, list) else [data]:
            flat.extend(d if isinstance(d, list) else [d])
        spans = _spans(flat)
        calls = [c for c in (_span_call(s) for s in spans) if c]
        if not spans:
            return "nothing", []
        # a framework's span around the provider's span for the same call: keep the innermost
        parent_of = {s["span_id"]: s.get("parent") for s in spans if s.get("span_id")}
        llm = {c.span_id for c in calls if c.span_id}
        outer = set()
        for c in calls:
            p, hops = c.parent_span, 0
            while p and hops < 50:
                if p in llm:
                    outer.add(p)
                p, hops = parent_of.get(p), hops + 1
        calls = [c for c in calls if c.span_id not in outer]
        fmt = "OpenTelemetry"
    calls.sort(key=lambda c: c.start)
    # the same call recorded twice (a framework span and the provider span inside it): keep one
    seen, unique = set(), []
    for c in calls:
        key = json.dumps([c.trace_id, c.messages, c.text, c.tool_calls], sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return fmt, unique


# ------------------------------------------------------------------ writing


def write_trace(calls: list[Call], path: str | os.PathLike, *, name: str, provider: str | None = None,
                endpoint: str | None = None, tools: list[dict] | None = None, source: str = "") -> Imported:
    """Write model calls as a runtape trace."""
    seen_providers = {c.provider for c in calls if c.provider}
    prov = provider or (seen_providers.pop() if len(seen_providers) == 1 else None) or "openai"
    rec = Recorder(path, name=name, tags={"imported_from": source} if source else None)
    pending: dict[str, tuple[dict, int]] = {}  # tool call id -> (call, reply event)
    n_tools = missing = 0
    notes: list[str] = []
    try:
        for c in calls:
            # tool calls answered in this request's history: run between the reply that asked and this request
            for m in c.messages:
                if m.get("role") == "tool" and m.get("tool_call_id") in pending:
                    tc, reply = pending.pop(m["tool_call_id"])
                    cid = rec.log("tool_call", {"name": tc["name"], "arguments": tc["arguments"], "call_id": tc["id"]},
                                  meta={"requested_by": reply})
                    rec.log("tool_result", {"name": tc["name"], "result": m.get("content")}, parent=cid)
                    n_tools += 1
            defs = tools or c.tools
            uses_tools = bool(c.tool_calls) or any(m.get("tool_calls") or m.get("role") == "tool" for m in c.messages)
            if uses_tools and not defs:
                missing += 1
            params = dict(c.params)
            if prov == "anthropic":
                system, msgs, atools = to_anthropic(c.messages, defs)
                params.setdefault("max_tokens", 4096)  # Anthropic requires it; the source may not say
                rid = rec.log_llm_request(provider="anthropic", api="messages", model=c.model, system=system,
                                          messages=msgs, tools=atools, params=params, endpoint=endpoint or c.endpoint)
            else:
                rid = rec.log_llm_request(provider="openai", api="chat.completions", model=c.model, messages=c.messages,
                                          tools=defs, params=params, endpoint=endpoint or c.endpoint)
            took = (c.end - c.start) * 1000 if c.end and c.start and c.end >= c.start else 0
            reply = rec.log_llm_response(rid, text=c.text, tool_calls=c.tool_calls, stop_reason=c.stop_reason,
                                         raw=None, latency_ms=took, model=c.model, tokens=c.tokens)
            for tc in c.tool_calls:
                pending[tc["id"]] = (tc, reply)
        # tool calls with no later request: the run ended on them (or the result was never sent back)
        for tc, reply in pending.values():
            rec.log("tool_call", {"name": tc["name"], "arguments": tc["arguments"], "call_id": tc["id"]},
                    meta={"requested_by": reply})
            n_tools += 1
    finally:
        rec.close()
    if len({c.provider for c in calls if c.provider}) > 1 and not provider:
        notes.append("The calls went to more than one provider; reruns go to " + prov + ". Pass --provider to choose.")
    return Imported(Path(path), len(calls), n_tools, missing, prov, notes)


def import_trace(source: str, out: str | os.PathLike | None = None, *, trace_id: str | None = None,
                 provider: str | None = None, endpoint: str | None = None, tools_file: str | None = None,
                 dir: str = "traces") -> list[Imported]:
    """Import a trace (or every trace in an export) and write runtape traces. Returns one entry per trace."""
    data = load_source(source)
    fmt, calls = read_calls(data)
    if fmt == "nothing":
        raise ValueError(f"No OpenTelemetry spans or Langfuse generations in {source}. runtape import reads an "
                         "OTLP JSON export, the OpenTelemetry console exporter's output, or a Langfuse trace; a "
                         "runtape trace needs no import.")
    if not calls:
        raise ValueError(
            f"No model calls with their messages in {source}. runtape needs the prompts and replies: turn on "
            "content capture in the instrumentation (OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true for "
            "the OpenTelemetry instrumentations; TRACELOOP_TRACE_CONTENT stays on by default for OpenLLMetry).")
    tools = openai_tools(json.loads(Path(tools_file).read_text())) if tools_file else None
    groups: dict[str | None, list[Call]] = {}
    for c in calls:
        groups.setdefault(c.trace_id, []).append(c)
    if trace_id:
        groups = {k: v for k, v in groups.items() if k and (k == trace_id or k.lower().endswith(trace_id.lower()))}
        if not groups:
            raise ValueError(f"No trace {trace_id} in {source}.")
    stem = re.sub(r"[^\w.-]+", "-", Path(source.split(":", 1)[-1]).stem if not source.startswith("langfuse:")
                  else "langfuse-" + source.split(":", 1)[1])[:60]
    results = []
    for i, (tid, cs) in enumerate(groups.items()):
        if out and len(groups) == 1:
            path = Path(out)
        else:
            suffix = f"-{str(tid)[-8:]}" if len(groups) > 1 and tid else (f"-{i}" if len(groups) > 1 else "")
            path = (Path(out).parent if out else Path(dir)) / f"{stem}{suffix}.jsonl"
        results.append(write_trace(cs, path, name=f"imported {stem}{' ' + str(tid) if tid and len(groups) > 1 else ''}",
                                   provider=provider, endpoint=endpoint, tools=tools, source=f"{fmt}: {source}"))
    return results
