"""Importing traces from other tools: the same support run, written the way each one records it, imports into a
trace that `why` explains the same way (the stale doc is what makes the agent refund $900)."""
import json

import pytest

from runtape import Trace
from runtape.cli import main
from runtape.importers import import_trace, openai_messages
from runtape.rerun import FunctionModel, build_request
from runtape.why import why

from .test_why import POLICY, STALE, support_model

SYSTEM = "You are support. Follow policy."
ORDER = {"order_id": "Z-9", "total": 900}
KB = [POLICY, STALE]


def fn(name, desc, props):
    return {"type": "function", "function": {"name": name, "description": desc,
                                             "parameters": {"type": "object", "properties": props}}}


TOOLS = [fn("lookup_order", "Look up an order", {"order_id": {"type": "string"}}),
         fn("search_kb", "Search the help docs", {"q": {"type": "string"}}),
         fn("issue_refund", "Refund an order", {"order_id": {"type": "string"}, "amount": {"type": "number"}}),
         fn("escalate_to_manager", "Hand to a manager", {"order_id": {"type": "string"}})]


def run_calls():
    """The run as OpenAI chat calls: (request messages, reply tool call)."""
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Refund my order Z-9 please."}]
    steps = [("c0", "lookup_order", {"order_id": "Z-9"}, ORDER), ("c1", "search_kb", {"q": "refund"}, KB)]
    calls = []
    for cid, name, args, result in steps:
        calls.append((list(msgs), {"id": cid, "name": name, "arguments": args}))
        msgs = msgs + [{"role": "assistant", "content": None, "tool_calls": [
            {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]},
            {"role": "tool", "tool_call_id": cid, "content": json.dumps(result)}]
    calls.append((list(msgs), {"id": "x", "name": "issue_refund", "arguments": {"order_id": "Z-9", "amount": 900}}))
    return calls


def otlp(spans):
    """OTLP JSON, as the collector's file exporter writes it: typed attribute values."""
    def val(v):
        if isinstance(v, bool):
            return {"boolValue": v}
        if isinstance(v, int):
            return {"intValue": str(v)}
        if isinstance(v, float):
            return {"doubleValue": v}
        if isinstance(v, list):
            return {"arrayValue": {"values": [val(x) for x in v]}}
        return {"stringValue": v if isinstance(v, str) else json.dumps(v)}

    out = []
    for i, (attrs, extra) in enumerate(spans):
        out.append({"traceId": "5b8efff798038103d269b633813fc60c", "spanId": f"{i + 1:016x}",
                    "parentSpanId": extra.get("parent", ""), "name": extra.get("name", "chat"),
                    "startTimeUnixNano": str(1_790_000_000_000_000_000 + i * 1_000_000_000),
                    "attributes": [{"key": k, "value": val(v)} for k, v in attrs.items()],
                    "events": extra.get("events", [])})
    return {"resourceSpans": [{"resource": {"attributes": [{"key": "service.name", "value": {"stringValue": "bot"}}]},
                               "scopeSpans": [{"scope": {"name": "x"}, "spans": out}]}]}


def openllmetry(msgs, reply):
    a = {"gen_ai.system": "openai", "gen_ai.request.model": "gpt-4o-mini", "llm.request.type": "chat",
         "gen_ai.request.temperature": 0.7}
    for i, m in enumerate(msgs):
        a[f"gen_ai.prompt.{i}.role"] = m["role"]
        if m.get("content") is not None:
            a[f"gen_ai.prompt.{i}.content"] = m["content"]
        if m.get("tool_call_id"):
            a[f"gen_ai.prompt.{i}.tool_call_id"] = m["tool_call_id"]
        for j, tc in enumerate(m.get("tool_calls") or []):
            a[f"gen_ai.prompt.{i}.tool_calls.{j}.id"] = tc["id"]
            a[f"gen_ai.prompt.{i}.tool_calls.{j}.name"] = tc["function"]["name"]
            a[f"gen_ai.prompt.{i}.tool_calls.{j}.arguments"] = tc["function"]["arguments"]
    for i, t in enumerate(TOOLS):
        a[f"llm.request.functions.{i}.name"] = t["function"]["name"]
        a[f"llm.request.functions.{i}.description"] = t["function"]["description"]
        a[f"llm.request.functions.{i}.parameters"] = json.dumps(t["function"]["parameters"])
    a["gen_ai.completion.0.role"] = "assistant"
    a["gen_ai.completion.0.finish_reason"] = "tool_calls"
    a["gen_ai.completion.0.tool_calls.0.id"] = reply["id"]
    a["gen_ai.completion.0.tool_calls.0.name"] = reply["name"]
    a["gen_ai.completion.0.tool_calls.0.arguments"] = json.dumps(reply["arguments"])
    return a


def openinference(msgs, reply, raw=False):
    a = {"openinference.span.kind": "LLM", "llm.model_name": "gpt-4o-mini", "llm.provider": "openai",
         "llm.invocation_parameters": json.dumps({"temperature": 0.7})}
    if raw:
        a["input.value"] = json.dumps({"model": "gpt-4o-mini", "messages": msgs, "tools": TOOLS, "temperature": 0.7})
        a["input.mime_type"] = "application/json"
    for i, m in enumerate(msgs):
        a[f"llm.input_messages.{i}.message.role"] = m["role"]
        if m.get("content") is not None:
            a[f"llm.input_messages.{i}.message.content"] = m["content"]
        if m.get("tool_call_id"):
            a[f"llm.input_messages.{i}.message.tool_call_id"] = m["tool_call_id"]
        for j, tc in enumerate(m.get("tool_calls") or []):
            p = f"llm.input_messages.{i}.message.tool_calls.{j}.tool_call"
            a[p + ".id"] = tc["id"]
            a[p + ".function.name"] = tc["function"]["name"]
            a[p + ".function.arguments"] = tc["function"]["arguments"]
    for i, t in enumerate(TOOLS):
        a[f"llm.tools.{i}.tool.json_schema"] = json.dumps(t)
    p = "llm.output_messages.0.message"
    a[p + ".role"] = "assistant"
    a[p + ".tool_calls.0.tool_call.id"] = reply["id"]
    a[p + ".tool_calls.0.tool_call.function.name"] = reply["name"]
    a[p + ".tool_calls.0.tool_call.function.arguments"] = json.dumps(reply["arguments"])
    return a


def genai(msgs, reply):
    def parts(m):
        if m["role"] == "tool":
            return {"role": "tool", "parts": [{"type": "tool_call_response", "id": m["tool_call_id"],
                                               "result": m["content"]}]}
        ps = [{"type": "text", "content": m["content"]}] if m.get("content") else []
        ps += [{"type": "tool_call", "id": tc["id"], "name": tc["function"]["name"],
                "arguments": json.loads(tc["function"]["arguments"])} for tc in m.get("tool_calls") or []]
        return {"role": m["role"], "parts": ps}

    return {"gen_ai.operation.name": "chat", "gen_ai.provider.name": "openai", "gen_ai.request.model": "gpt-4o-mini",
            "gen_ai.request.temperature": 0.7,
            "gen_ai.system_instructions": json.dumps([{"type": "text", "content": SYSTEM}]),
            "gen_ai.input.messages": json.dumps([parts(m) for m in msgs if m["role"] != "system"]),
            "gen_ai.output.messages": json.dumps([{"role": "assistant", "finish_reason": "tool_call", "parts": [
                {"type": "tool_call", "id": reply["id"], "name": reply["name"], "arguments": reply["arguments"]}]}]),
            "gen_ai.tool.definitions": json.dumps([{"type": "function", **t["function"]} for t in TOOLS])}


def langfuse(model="gpt-4o-mini"):
    obs = [{"id": "trace-span", "type": "SPAN", "name": "agent", "startTime": "2026-10-02T20:00:00.000Z"}]
    for i, (msgs, reply) in enumerate(run_calls()):
        obs.append({"id": f"gen-{i}", "type": "GENERATION", "name": "OpenAI-generation", "traceId": "lf-1",
                    "startTime": f"2026-10-02T20:00:0{i + 1}.000Z", "model": model,
                    "modelParameters": {"temperature": 0.7, "max_tokens": "inf"},
                    "input": {"messages": msgs, "tools": TOOLS},
                    "output": {"role": "assistant", "content": None, "tool_calls": [
                        {"id": reply["id"], "type": "function", "function": {
                            "name": reply["name"], "arguments": json.dumps(reply["arguments"])}}]}})
    return {"id": "lf-1", "name": "support", "observations": obs}


def explain(path):
    t = Trace.load(path)
    rep = why(t, t.of_type("llm_response")[-1].id, model=FunctionModel(support_model), cache_dir=None)
    decisive = [c for c in rep.causes if c.kind == "decisive"]
    return t, rep, decisive


def check(path, provider="openai"):
    t, rep, decisive = explain(path)
    assert rep.baseline.kept == rep.baseline.n, rep.warnings
    assert decisive and "any amount" in decisive[0].finest.removed[-1].text, rep.to_dict()
    # the tool calls were linked to the replies that asked for them, so pieces carry their origin
    calls = t.of_type("tool_call")
    assert [c.payload["name"] for c in calls] == ["lookup_order", "search_kb", "issue_refund"]
    assert all(c.meta.get("requested_by") is not None for c in calls)
    req = build_request(t, t.of_type("llm_request")[-1].id)
    assert req["provider"] == provider and req["tools"], req
    return t


@pytest.mark.parametrize("convention", ["openllmetry", "openinference", "openinference_raw", "genai"])
def test_opentelemetry_conventions(tmp_path, convention):
    make = {"openllmetry": openllmetry, "openinference": openinference, "genai": genai,
            "openinference_raw": lambda m, r: openinference(m, r, raw=True)}[convention]
    spans = [(make(m, r), {"name": "openai.chat"}) for m, r in run_calls()]
    src = tmp_path / "spans.json"
    src.write_text(json.dumps(otlp(spans)))
    [r] = import_trace(str(src), tmp_path / "t.jsonl")
    assert r.calls == 3 and r.tool_calls == 3 and r.missing_tools == 0
    t = check(r.path)
    assert t.of_type("llm_request")[0].payload["params"].get("temperature") == 0.7


def test_collector_file_and_console_exporter(tmp_path):
    # the collector writes one OTLP object per line; the SDK's console exporter prints one span per object
    spans = [(openllmetry(m, r), {}) for m, r in run_calls()]
    lines = tmp_path / "lines.jsonl"
    lines.write_text("\n".join(json.dumps(otlp([s])) for s in spans))
    [r] = import_trace(str(lines), tmp_path / "a.jsonl")
    check(r.path)
    console = [{"name": "openai.chat", "context": {"trace_id": "0xabc", "span_id": f"0x{i}"}, "parent_id": None,
                "start_time": f"2026-10-02T20:00:0{i}.000000Z", "attributes": a} for i, (a, _) in enumerate(spans)]
    src = tmp_path / "console.json"
    src.write_text(json.dumps(console))
    [r] = import_trace(str(src), tmp_path / "b.jsonl")
    check(r.path)


def test_framework_span_around_provider_span_counts_once(tmp_path):
    # LangChain's span and the OpenAI client's span inside it record the same call
    spans = []
    for i, (m, r) in enumerate(run_calls()):
        outer = openllmetry(m, r)
        spans.append((outer, {"name": "ChatOpenAI.chat"}))
        spans.append((dict(outer), {"name": "openai.chat", "parent": f"{2 * i + 1:016x}"}))
    src = tmp_path / "spans.json"
    src.write_text(json.dumps(otlp(spans)))
    [r] = import_trace(str(src), tmp_path / "t.jsonl")
    assert r.calls == 3
    check(r.path)


def test_langfuse_trace(tmp_path):
    src = tmp_path / "lf.json"
    src.write_text(json.dumps(langfuse()))
    [r] = import_trace(str(src), tmp_path / "t.jsonl")
    assert r.calls == 3 and r.provider == "openai"
    t = check(r.path)
    assert "max_tokens" not in t.of_type("llm_request")[0].payload["params"]  # Langfuse's "inf" means unset


def test_claude_calls_are_resent_to_claude(tmp_path):
    src = tmp_path / "lf.json"
    src.write_text(json.dumps(langfuse("claude-sonnet-5-5")))
    [r] = import_trace(str(src), tmp_path / "t.jsonl")
    assert r.provider == "anthropic"
    t = check(r.path, provider="anthropic")
    req = build_request(t, t.of_type("llm_request")[-1].id)
    assert req["system"] == SYSTEM and req["tools"][0]["input_schema"]
    kinds = [b["type"] for m in req["messages"] for b in m["content"]]
    assert "tool_use" in kinds and "tool_result" in kinds


def test_missing_tool_definitions_are_reported_and_can_be_supplied(tmp_path, capsys):
    spans = []
    for m, r in run_calls():
        a = openllmetry(m, r)
        spans.append(({k: v for k, v in a.items() if not k.startswith("llm.request.functions")}, {}))
    src = tmp_path / "spans.json"
    src.write_text(json.dumps(otlp(spans)))
    code = main(["--no-color", "import", str(src), "-o", str(tmp_path / "t.jsonl")])
    out = capsys.readouterr().out
    assert code == 0 and "no tool" in out and "--tools" in out
    tools = tmp_path / "tools.json"
    tools.write_text(json.dumps(TOOLS))
    code = main(["--no-color", "import", str(src), "-o", str(tmp_path / "u.jsonl"), "--tools", str(tools)])
    out = capsys.readouterr().out
    assert code == 0 and "no tool" not in out and "runtape why" in out
    check(tmp_path / "u.jsonl")


def test_no_content_says_how_to_turn_it_on(tmp_path):
    src = tmp_path / "spans.json"
    src.write_text(json.dumps(otlp([({"gen_ai.system": "openai", "gen_ai.request.model": "gpt-4o"}, {})])))
    with pytest.raises(ValueError, match="CAPTURE_MESSAGE_CONTENT"):
        import_trace(str(src), tmp_path / "t.jsonl")


def test_several_traces_in_one_export(tmp_path):
    doc = otlp([(openllmetry(m, r), {}) for m, r in run_calls()])
    other = json.loads(json.dumps(doc))
    for s in other["resourceSpans"][0]["scopeSpans"][0]["spans"]:
        s["traceId"] = "ffffffffffffffffffffffffffffffff"
    src = tmp_path / "spans.jsonl"
    src.write_text(json.dumps(doc) + "\n" + json.dumps(other))
    done = import_trace(str(src), dir=str(tmp_path / "out"))
    assert len(done) == 2 and all(r.calls == 3 for r in done)
    [one] = import_trace(str(src), tmp_path / "x.jsonl", trace_id="ffffffff")
    assert one.calls == 3


def test_message_shapes():
    # Anthropic blocks, semantic-convention parts and LangChain roles all become OpenAI chat messages
    out = openai_messages([
        {"role": "human", "content": "hi"},
        {"role": "assistant", "content": [{"type": "text", "text": "looking"},
                                          {"type": "tool_use", "id": "t1", "name": "f", "input": {"a": 1}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]},
        {"role": "tool", "parts": [{"type": "tool_call_response", "id": "t2", "result": {"v": 2}}]},
    ])
    assert out[0] == {"role": "user", "content": "hi"}
    assert out[1]["tool_calls"][0]["function"] == {"name": "f", "arguments": '{"a": 1}'}
    assert out[2] == {"role": "tool", "tool_call_id": "t1", "content": "ok"}
    assert out[3]["role"] == "tool" and out[3]["tool_call_id"] == "t2" and json.loads(out[3]["content"]) == {"v": 2}


FIXTURES = __import__("pathlib").Path(__file__).parent / "fixtures" / "imports"


@pytest.mark.parametrize("name", ["openllmetry-0.62.4-otlp.json", "openllmetry-0.62.4-console.json",
                                  "openinference-0.1.63-otlp.json"])
def test_spans_from_real_instrumentations(tmp_path, name):
    """Spans written by the real instrumentation packages (fixtures/imports/make_fixtures.py ran the same support
    agent under each): the rebuilt request is exactly what the agent sent, and why finds the stale doc."""
    [r] = import_trace(str(FIXTURES / name), tmp_path / "t.jsonl")
    assert r.calls == 3 and r.missing_tools == 0
    t = check(r.path)
    req = build_request(t, t.of_type("llm_request")[-1].id)
    sent = run_calls()[-1][0]
    for m in sent:  # the fixture agent's tool call ids
        for tc in m.get("tool_calls") or []:
            tc["id"] = {"c0": "call_0", "c1": "call_1"}[tc["id"]]
        if m.get("tool_call_id"):
            m["tool_call_id"] = {"c0": "call_0", "c1": "call_1"}[m["tool_call_id"]]
    assert req["messages"] == sent
    assert req["params"] == {"temperature": 0.7}
    # the span's duration and token counts come along, for summary and the cost estimate
    resp = t.of_type("llm_response")[0]
    assert resp.meta["tokens"] == {"input": 10, "output": 5}
    assert resp.meta["latency_ms"] > 0


def test_a_file_with_no_spans_says_so(tmp_path):
    src = tmp_path / "other.json"
    src.write_text(json.dumps({"hello": "world"}))
    with pytest.raises(ValueError, match="No OpenTelemetry spans or Langfuse generations"):
        import_trace(str(src), tmp_path / "t.jsonl")



def test_openai_content_parts_are_kept():
    img = {"role": "user", "content": [{"type": "text", "text": "what is this?"},
                                       {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}
    assert openai_messages([img]) == [img]
