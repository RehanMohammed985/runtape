import pytest

from runtape import Recorder, Trace

from .conftest import Script, an_msg, anthropic_client, oa_chat, openai_client


# ----------------------------------------------------------------- openai


def test_openai_chat_multiturn_delta_and_context(tpath):
    script = Script(
        [
            oa_chat(tool_calls=[("call_1", "get_weather", {"city": "Raleigh"})], finish="tool_calls"),
            oa_chat("It's 72F in Raleigh."),
        ]
    )
    rec = Recorder(tpath)
    client = rec.wrap(openai_client(script))

    @rec.tool
    def get_weather(city):
        return {"temp_f": 72}

    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "weather in raleigh?"},
    ]
    r1 = client.chat.completions.create(model="gpt-test", messages=msgs, tools=[{"type": "function", "function": {"name": "get_weather"}}])
    tc = r1.choices[0].message.tool_calls[0]
    out = get_weather(city="Raleigh")
    msgs += [r1.choices[0].message.model_dump(exclude_none=True), {"role": "tool", "tool_call_id": tc.id, "content": str(out)}]
    client.chat.completions.create(model="gpt-test", messages=msgs, tools=[{"type": "function", "function": {"name": "get_weather"}}])
    rec.close()

    t = Trace.load(tpath)
    reqs = t.of_type("llm_request")
    assert len(reqs) == 2
    # second request is stored as a delta on the first
    assert "messages" in reqs[0].payload
    assert reqs[1].payload["base"] == reqs[0].id
    assert len(reqs[1].payload["messages_append"]) == 2
    # tools unchanged -> not repeated
    assert "tools" in reqs[0].payload and "tools" not in reqs[1].payload
    # rebuilt context matches exactly what hit the wire
    assert t.messages(reqs[1].id) == script.requests[1]["messages"]
    ctx = t.context(reqs[1].id)
    assert ctx.tools == [{"type": "function", "function": {"name": "get_weather"}}]

    resp1 = t.of_type("llm_response")[0]
    assert resp1.payload["tool_calls"] == [{"id": "call_1", "name": "get_weather", "arguments": {"city": "Raleigh"}}]
    assert resp1.meta["tokens"] == {"input": 10, "output": 5}
    # the tool execution is linked back to the model's request for it
    call = t.of_type("tool_call")[0]
    assert call.payload["call_id"] == "call_1"
    assert call.meta["requested_by"] == resp1.id
    assert t.of_type("llm_response")[1].payload["text"] == "It's 72F in Raleigh."


def test_openai_api_error_recorded(tpath):
    import openai

    script = Script([(400, {"error": {"message": "bad model", "type": "invalid_request_error"}})])
    rec = Recorder(tpath)
    client = rec.wrap(openai_client(script))
    with pytest.raises(openai.BadRequestError):
        client.chat.completions.create(model="nope", messages=[{"role": "user", "content": "hi"}])
    rec.close()
    t = Trace.load(tpath)
    err = t.of_type("error")[0]
    assert err.parent == t.of_type("llm_request")[0].id
    assert err.payload["type"] == "BadRequestError"


def test_openai_stream(tpath):
    def chunk(delta, finish=None):
        return {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "gpt-test",
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    script = Script([[
        chunk({"role": "assistant", "content": "Hel"}),
        chunk({"content": "lo"}),
        chunk({"tool_calls": [{"index": 0, "id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"a"'}}]}),
        chunk({"tool_calls": [{"index": 0, "function": {"arguments": ': 1}'}}]}, finish="tool_calls"),
        "[DONE]",
    ]])
    rec = Recorder(tpath)
    client = rec.wrap(openai_client(script))
    stream = client.chat.completions.create(model="gpt-test", messages=[{"role": "user", "content": "x"}], stream=True)
    got = "".join(c.choices[0].delta.content or "" for c in stream)
    rec.close()
    assert got == "Hello"
    resp = Trace.load(tpath).of_type("llm_response")[0]
    assert resp.payload["text"] == "Hello"
    assert resp.payload["tool_calls"] == [{"id": "c1", "name": "f", "arguments": {"a": 1}}]
    assert resp.payload["stop_reason"] == "tool_calls"


def test_openai_stream_break_early(tpath):
    def chunk(c):
        return {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m",
                "choices": [{"index": 0, "delta": {"content": c}, "finish_reason": None}]}

    script = Script([[chunk("a"), chunk("b"), chunk("c"), "[DONE]"]])
    rec = Recorder(tpath)
    client = rec.wrap(openai_client(script))
    for c in client.chat.completions.create(model="m", messages=[], stream=True):
        break
    rec.close()
    t = Trace.load(tpath)
    assert len(t.of_type("llm_response")) == 1
    assert not t.of_type("error")


def test_openai_responses_api(tpath):
    body = {
        "id": "resp_1", "object": "response", "created_at": 0, "model": "gpt-test", "status": "completed",
        "output": [
            {"type": "function_call", "id": "fc_1", "call_id": "call_9", "name": "search",
             "arguments": '{"q": "refund policy"}', "status": "completed"},
        ],
        "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
        "usage": {"input_tokens": 12, "output_tokens": 3, "total_tokens": 15,
                  "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}},
    }
    script = Script([body])
    rec = Recorder(tpath)
    client = rec.wrap(openai_client(script))
    client.responses.create(model="gpt-test", input="what's the refund policy", instructions="be brief")
    rec.close()
    t = Trace.load(tpath)
    req = t.of_type("llm_request")[0]
    assert req.payload["system"] == "be brief"
    assert req.payload["messages"] == [{"role": "user", "content": "what's the refund policy"}]
    resp = t.of_type("llm_response")[0]
    assert resp.payload["tool_calls"] == [{"id": "call_9", "name": "search", "arguments": {"q": "refund policy"}}]
    assert resp.meta["tokens"] == {"input": 12, "output": 3}


async def test_openai_async(tpath):
    script = Script([oa_chat("async hi")])
    rec = Recorder(tpath)
    client = rec.wrap(openai_client(script, async_=True))
    r = await client.chat.completions.create(model="gpt-test", messages=[{"role": "user", "content": "hi"}])
    rec.close()
    assert r.choices[0].message.content == "async hi"
    assert Trace.load(tpath).of_type("llm_response")[0].payload["text"] == "async hi"


def test_wrap_is_idempotent_and_returns_same_client(tpath):
    script = Script([oa_chat("x")])
    rec = Recorder(tpath)
    c = openai_client(script)
    assert rec.wrap(rec.wrap(c)) is c
    c.chat.completions.create(model="m", messages=[])
    rec.close()
    assert len(Trace.load(tpath).of_type("llm_request")) == 1


def test_wrap_unknown_client_raises(tpath):
    rec = Recorder(tpath)
    with pytest.raises(TypeError):
        rec.wrap(object())
    rec.close()


# -------------------------------------------------------------- anthropic


def test_anthropic_tool_loop(tpath):
    script = Script(
        [
            an_msg("Let me check.", tool_uses=[("toolu_1", "lookup_order", {"order_id": "A1"})], stop="tool_use"),
            an_msg("Your order shipped."),
        ]
    )
    rec = Recorder(tpath)
    client = rec.wrap(anthropic_client(script))

    @rec.tool
    def lookup_order(order_id):
        return {"status": "shipped"}

    msgs = [{"role": "user", "content": "where is order A1"}]
    r = client.messages.create(model="claude-test", max_tokens=100, system="You are support.", messages=msgs)
    tu = [b for b in r.content if b.type == "tool_use"][0]
    res = lookup_order(**tu.input)
    msgs += [
        {"role": "assistant", "content": [b.model_dump(exclude_none=True) for b in r.content]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tu.id, "content": str(res)}]},
    ]
    client.messages.create(model="claude-test", max_tokens=100, system="You are support.", messages=msgs)
    rec.close()

    t = Trace.load(tpath)
    r1, r2 = t.of_type("llm_request")
    assert r1.payload["system"] == "You are support."
    assert "system" not in r2.payload  # unchanged, not repeated
    assert t.context(r2.id).system == "You are support."
    assert t.messages(r2.id) == script.requests[1]["messages"]
    assert r2.payload["params"]["max_tokens"] == 100
    resp = t.of_type("llm_response")[0]
    assert resp.payload["text"] == "Let me check."
    assert resp.payload["tool_calls"][0]["arguments"] == {"order_id": "A1"}
    assert t.of_type("tool_call")[0].payload["call_id"] == "toolu_1"


def test_anthropic_system_change_is_recorded(tpath):
    script = Script([an_msg("a"), an_msg("b")])
    rec = Recorder(tpath)
    client = rec.wrap(anthropic_client(script))
    client.messages.create(model="m", max_tokens=5, system="v1", messages=[{"role": "user", "content": "x"}])
    client.messages.create(model="m", max_tokens=5, system="v2", messages=[{"role": "user", "content": "x"}])
    rec.close()
    r1, r2 = Trace.load(tpath).of_type("llm_request")
    assert r2.payload["system"] == "v2"


def test_anthropic_stream(tpath):
    events = [
        {"type": "message_start", "message": {"id": "m", "type": "message", "role": "assistant", "model": "claude-test",
                                              "content": [], "stop_reason": None, "stop_sequence": None,
                                              "usage": {"input_tokens": 9, "output_tokens": 0}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Refund "}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "issued."}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "tu", "name": "refund", "input": {}}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"amount":'}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": " 500}"}},
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None}, "usage": {"output_tokens": 14}},
        {"type": "message_stop"},
    ]
    script = Script([events])
    rec = Recorder(tpath)
    client = rec.wrap(anthropic_client(script))
    with client.messages.create(model="claude-test", max_tokens=50, messages=[{"role": "user", "content": "x"}], stream=True) as s:
        for _ in s:
            pass
    rec.close()
    resp = Trace.load(tpath).of_type("llm_response")
    assert len(resp) == 1
    p = resp[0].payload
    assert p["text"] == "Refund issued."
    assert p["tool_calls"] == [{"id": "tu", "name": "refund", "arguments": {"amount": 500}}]
    assert resp[0].meta["tokens"] == {"input": 9, "output": 14}


async def test_anthropic_async(tpath):
    script = Script([an_msg("hi")])
    rec = Recorder(tpath)
    client = rec.wrap(anthropic_client(script, async_=True))
    await client.messages.create(model="m", max_tokens=5, messages=[{"role": "user", "content": "x"}])
    rec.close()
    assert Trace.load(tpath).of_type("llm_response")[0].payload["text"] == "hi"


# ------------------------------------------------------------------ trace


def test_grep_first_hit_is_where_term_entered(tpath):
    script = Script([oa_chat("ok"), oa_chat("ok"), oa_chat("I'll refund you")])
    rec = Recorder(tpath)
    client = rec.wrap(openai_client(script))

    @rec.tool
    def search_kb(q):
        return "Policy: refunds always approved"

    msgs = [{"role": "user", "content": "hi"}]
    client.chat.completions.create(model="m", messages=msgs)
    kb = search_kb("policy")
    msgs += [{"role": "assistant", "content": "ok"}, {"role": "user", "content": kb}]
    client.chat.completions.create(model="m", messages=msgs)
    msgs += [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "can I get money back"}]
    client.chat.completions.create(model="m", messages=msgs)
    rec.close()

    t = Trace.load(tpath)
    hits = t.grep("REFUND")
    assert hits[0].first
    assert hits[0].event.type == "tool_result"  # entered via the tool, not the user
    # the later delta-encoded requests only match where the term is newly added
    types = [h.event.type for h in hits]
    assert types.count("llm_request") == 1


def test_summary(tpath):
    script = Script([oa_chat("a"), oa_chat("b")])
    rec = Recorder(tpath, name="sum")
    client = rec.wrap(openai_client(script))

    @rec.tool
    def boom():
        raise ValueError

    client.chat.completions.create(model="m", messages=[])
    client.chat.completions.create(model="m", messages=[])
    with pytest.raises(ValueError):
        boom()
    rec.close()
    s = Trace.load(tpath).summary()
    assert s["llm_calls"] == 2
    assert s["tokens"] == {"input": 20, "output": 10}
    assert s["tool_calls"] == {"boom": 1}
    assert s["tool_errors"] == 1
    assert s["status"] == "ok"


# ------------------------------------------------------- more entry points


def _anthropic_sse(text="Hi there", tool=None):
    ev = [
        {"type": "message_start", "message": {"id": "m", "type": "message", "role": "assistant", "model": "claude-test",
                                              "content": [], "stop_reason": None, "stop_sequence": None,
                                              "usage": {"input_tokens": 4, "output_tokens": 0}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 3}},
        {"type": "message_stop"},
    ]
    return ev


def test_anthropic_messages_stream_helper_is_recorded(tpath):
    script = Script([_anthropic_sse("streamed hello")])
    rec = Recorder(tpath)
    client = rec.wrap(anthropic_client(script))
    with client.messages.stream(model="claude-test", max_tokens=20, messages=[{"role": "user", "content": "x"}]) as s:
        got = "".join(s.text_stream)
    rec.close()
    assert got == "streamed hello"
    t = Trace.load(tpath)
    assert len(t.of_type("llm_request")) == 1
    r = t.of_type("llm_response")[0]
    assert r.payload["text"] == "streamed hello" and r.meta["tokens"]["output"] == 3


def test_anthropic_beta_create_is_recorded(tpath):
    script = Script([an_msg("beta hi")])
    rec = Recorder(tpath)
    client = rec.wrap(anthropic_client(script))
    client.beta.messages.create(model="claude-test", max_tokens=20, messages=[{"role": "user", "content": "x"}])
    rec.close()
    assert Trace.load(tpath).of_type("llm_response")[0].payload["text"] == "beta hi"


def test_openai_stream_helper_and_parse_recorded_once(tpath):
    def chunk(delta, finish=None):
        return {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "gpt-test",
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    script = Script([[chunk({"role": "assistant", "content": "str"}), chunk({"content": "eamed"}, "stop"), "[DONE]"],
                     oa_chat("parsed")])
    rec = Recorder(tpath)
    client = rec.wrap(openai_client(script))
    with client.chat.completions.stream(model="gpt-test", messages=[{"role": "user", "content": "x"}]) as s:
        for _ in s:
            pass
    client.chat.completions.parse(model="gpt-test", messages=[{"role": "user", "content": "y"}])
    rec.close()
    t = Trace.load(tpath)
    assert len(t.of_type("llm_request")) == 2  # not double-logged
    texts = [e.payload["text"] for e in t.of_type("llm_response")]
    assert texts == ["streamed", "parsed"]


def test_next_on_recorded_stream(tpath):
    def chunk(c):
        return {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m",
                "choices": [{"index": 0, "delta": {"content": c}, "finish_reason": None}]}

    script = Script([[chunk("a"), chunk("b"), "[DONE]"]])
    rec = Recorder(tpath)
    s = rec.wrap(openai_client(script)).chat.completions.create(model="m", messages=[], stream=True)
    assert next(s).choices[0].delta.content == "a"
    assert next(s).choices[0].delta.content == "b"
    with pytest.raises(StopIteration):
        next(s)
    rec.close()
    assert Trace.load(tpath).of_type("llm_response")[0].payload["text"] == "ab"
