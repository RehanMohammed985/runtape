"""Replaying a recorded run through the real agent code."""
from dataclasses import dataclass

import anthropic
import pytest

import runtape
from runtape import ReplayDiverged, Trace

from .conftest import Script, an_msg, anthropic_client, oa_chat, openai_client
from .test_example import load_example


@pytest.fixture
def recorded(tmp_path):
    ex = load_example()
    return ex, ex.main(tmp_path / "orig.jsonl")


def offline_client():
    # any real call during replay would hit this and fail loudly
    def boom(request):
        raise AssertionError("replay made a live model call")

    import httpx2
    return anthropic.Anthropic(api_key="x", http_client=httpx2.Client(transport=httpx2.MockTransport(boom)))


def test_exact_replay_serves_everything(recorded, tmp_path):
    ex, orig = recorded
    executed = []
    real_orders = dict(ex.ORDERS)
    with runtape.replay(orig, tmp_path / "replay.jsonl") as rp:
        client = rp.wrap(offline_client())
        ex.ORDERS.clear()  # tools must not run: the recording supplies their results
        try:
            msgs = ex.run_agent(rp, client)
        finally:
            ex.ORDERS.update(real_orders)
    assert rp.stats.served_model_calls == 9
    assert rp.stats.served_tool_calls == 6
    assert rp.stats.live_model_calls == rp.stats.live_tool_calls == 0
    assert rp.stats.divergence is None
    assert "2,400.00" in msgs[-1]["content"][0]["text"]
    # the replay is itself a trace, tagged with what it replayed
    new = Trace.load(tmp_path / "replay.jsonl")
    assert new.header["tags"]["replay_of"] == Trace.load(orig).header["run_id"]
    assert len(new.of_type("llm_request")) == 9 and new.status == "ok"
    assert new.grep("any amount")[0].event.type == "tool_result"


def test_prompt_change_stops_at_first_difference(recorded, tmp_path):
    ex, orig = recorded
    with pytest.raises(ReplayDiverged) as err:
        with runtape.replay(orig, tmp_path / "r.jsonl") as rp:
            ex.run_agent(rp, rp.wrap(offline_client()), system=ex.SYSTEM + " Never refund over $200.")
    d = err.value.divergence
    assert d.kind == "model_call" and d.step == 1 and d.field == "system"
    assert "Never refund over $200" in str(err.value)
    assert Trace.load(tmp_path / "r.jsonl").status == "diverged"


def _record_one_turn(path, *, tools):
    """Record: one model call that asks for tools, then those tools run."""
    calls = [(f"t{i}", name, args) for i, (name, args) in enumerate(tools)]
    s = Script([an_msg(tool_uses=calls, stop="tool_use")])
    rec = runtape.record(path)
    c = rec.wrap(anthropic_client(s))
    fns = {}
    for name in {n for n, _ in tools}:
        fns[name] = rec.tool(name=name)(lambda **kw: {"echo": kw})
    r = c.messages.create(model="m", max_tokens=5, messages=[{"role": "user", "content": "go"}])
    for b in r.content:
        fns[b.name](**b.input)
    rec.close()


def test_tool_argument_change_is_caught(tmp_path):
    _record_one_turn(tmp_path / "o.jsonl", tools=[("lookup", {"order_id": "A"})])
    with pytest.raises(ReplayDiverged) as err:
        with runtape.replay(tmp_path / "o.jsonl", tmp_path / "r.jsonl") as rp:
            c = rp.wrap(anthropic_client(Script([])))
            c.messages.create(model="m", max_tokens=5, messages=[{"role": "user", "content": "go"}])
            rp.tool(name="lookup")(lambda **kw: None)(order_id="WRONG")
    d = err.value.divergence
    assert d.kind == "tool_call" and d.field == "arguments"
    assert d.recorded == {"order_id": "A"} and d.now == {"order_id": "WRONG"}


def test_parallel_tools_in_any_order_and_unwrapped_tools(tmp_path):
    _record_one_turn(tmp_path / "o.jsonl", tools=[("search", {"q": "cats"}), ("search", {"q": "dogs"}), ("log", {"m": "x"})])
    with runtape.replay(tmp_path / "o.jsonl", tmp_path / "r.jsonl") as rp:
        c = rp.wrap(anthropic_client(Script([])))
        c.messages.create(model="m", max_tokens=5, messages=[{"role": "user", "content": "go"}])
        search = rp.tool(name="search")(lambda **kw: None)
        dogs = search(q="dogs")  # opposite order from the recording
        cats = search(q="cats")
        # "log" is never wrapped or called in the replay: fine
    assert dogs == {"echo": {"q": "dogs"}} and cats == {"echo": {"q": "cats"}}
    assert rp.stats.divergence is None and rp.stats.served_tool_calls == 2


def test_changed_request_settings_diverge(tmp_path):
    s = Script([an_msg("x")])
    rec = runtape.record(tmp_path / "o.jsonl")
    rec.wrap(anthropic_client(s)).messages.create(model="m", max_tokens=100, messages=[{"role": "user", "content": "x"}])
    rec.close()
    with pytest.raises(ReplayDiverged) as err:
        with runtape.replay(tmp_path / "o.jsonl", tmp_path / "r.jsonl") as rp:
            rp.wrap(anthropic_client(Script([]))).messages.create(model="m", max_tokens=5, messages=[{"role": "user", "content": "x"}])
    assert err.value.divergence.field == "params"


@dataclass
class Order:
    id: str
    total: float


def test_replay_returns_original_types(tmp_path):
    rec = runtape.record(tmp_path / "o.jsonl")

    @rec.tool
    def get_order():
        return Order("A", 18.0)

    @rec.tool
    def pair():
        return (1, 2)

    get_order()
    pair()
    rec.close()
    src = Trace.load(tmp_path / "o.jsonl")
    # no model calls in this recording, so every tool call is in the same "turn"
    with runtape.replay(src, tmp_path / "r.jsonl") as rp:
        o = rp.tool(name="get_order")(lambda: None)()
        p = rp.tool(name="pair")(lambda: None)()
    assert isinstance(o, Order) and o.total == 18.0
    assert p == (1, 2) and isinstance(p, tuple)


def test_diverge_live_goes_to_real_model(tmp_path):
    # record a 2-call run, then replay with a changed second message: first call served, second live
    s1 = Script([an_msg("first"), an_msg("second")])
    rec = runtape.record(tmp_path / "o.jsonl")
    c = rec.wrap(anthropic_client(s1))
    c.messages.create(model="m", max_tokens=5, messages=[{"role": "user", "content": "a"}])
    c.messages.create(model="m", max_tokens=5, messages=[{"role": "user", "content": "b"}])
    rec.close()

    s2 = Script([an_msg("LIVE REPLY")])
    with runtape.replay(tmp_path / "o.jsonl", tmp_path / "r.jsonl", on_diverge="live") as rp:
        c = rp.wrap(anthropic_client(s2))
        r1 = c.messages.create(model="m", max_tokens=5, messages=[{"role": "user", "content": "a"}])
        r2 = c.messages.create(model="m", max_tokens=5, messages=[{"role": "user", "content": "CHANGED"}])
    assert r1.content[0].text == "first"  # served
    assert r2.content[0].text == "LIVE REPLY"  # real call
    assert len(s2.requests) == 1
    assert rp.stats.divergence.field == "messages[0]"
    assert rp.stats.served_model_calls == 1 and rp.stats.live_model_calls == 1


def test_openai_replay_and_async(tmp_path):
    s = Script([oa_chat("hello")])
    rec = runtape.record(tmp_path / "o.jsonl")
    rec.wrap(openai_client(s)).chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])
    rec.close()
    with runtape.replay(tmp_path / "o.jsonl", tmp_path / "r.jsonl") as rp:
        r = rp.wrap(openai_client(Script([]))).chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])
    assert r.choices[0].message.content == "hello"


async def test_async_replay(tmp_path):
    s = Script([an_msg("async one")])
    rec = runtape.record(tmp_path / "o.jsonl")
    await rec.wrap(anthropic_client(s, async_=True)).messages.create(model="m", max_tokens=5, messages=[{"role": "user", "content": "x"}])
    rec.close()
    with runtape.replay(tmp_path / "o.jsonl", tmp_path / "r.jsonl") as rp:
        r = await rp.wrap(anthropic_client(Script([]), async_=True)).messages.create(
            model="m", max_tokens=5, messages=[{"role": "user", "content": "x"}])
    assert r.content[0].text == "async one"


def test_recorded_tool_errors_are_raised_again(tmp_path):
    rec = runtape.record(tmp_path / "o.jsonl")

    @rec.tool
    def charge(amount):
        raise ValueError("card declined")

    with pytest.raises(ValueError):
        charge(5)
    rec.close()
    with runtape.replay(tmp_path / "o.jsonl", tmp_path / "r.jsonl") as rp:
        @rp.tool
        def charge(amount):
            raise AssertionError("should not run")

        with pytest.raises(RuntimeError, match="card declined"):
            charge(5)


def test_extra_call_beyond_recording(tmp_path):
    s = Script([an_msg("only")])
    rec = runtape.record(tmp_path / "o.jsonl")
    rec.wrap(anthropic_client(s)).messages.create(model="m", max_tokens=5, messages=[{"role": "user", "content": "x"}])
    rec.close()
    with pytest.raises(ReplayDiverged, match="no call here"):
        with runtape.replay(tmp_path / "o.jsonl", tmp_path / "r.jsonl") as rp:
            c = rp.wrap(anthropic_client(Script([])))
            c.messages.create(model="m", max_tokens=5, messages=[{"role": "user", "content": "x"}])
            c.messages.create(model="m", max_tokens=5, messages=[{"role": "user", "content": "y"}])


def test_tools_from_an_earlier_turn_are_not_reused(tmp_path):
    # turn 1: get_time -> 10:00, send_email ; turn 2: get_time -> 10:05
    s = Script([an_msg(tool_uses=[("t1", "get_time", {}), ("t2", "send_email", {"to": "a"})], stop="tool_use"),
                an_msg(tool_uses=[("t3", "get_time", {})], stop="tool_use"),
                an_msg("done")])
    rec = runtape.record(tmp_path / "o.jsonl")
    c = rec.wrap(anthropic_client(s))
    times = iter(["10:00", "10:05"])
    get_time = rec.tool(name="get_time")(lambda: next(times))
    send_email = rec.tool(name="send_email")(lambda to: "sent")
    msgs = [{"role": "user", "content": "go"}]
    for _ in range(3):
        r = c.messages.create(model="m", max_tokens=5, messages=msgs)
        msgs = msgs + [{"role": "assistant", "content": [b.model_dump(exclude_none=True) for b in r.content]}]
        outs = []
        for b in r.content:
            if b.type == "tool_use":
                out = {"get_time": get_time, "send_email": send_email}[b.name](**b.input)
                outs.append({"type": "tool_result", "tool_use_id": b.id, "content": str(out)})
        if not outs:
            break
        msgs = msgs + [{"role": "user", "content": outs}]
    rec.close()

    # replay with code that skips turn 1's tools entirely: turn 2's get_time must not get 10:00
    with pytest.raises(ReplayDiverged):
        with runtape.replay(tmp_path / "o.jsonl", tmp_path / "r.jsonl") as rp:
            c = rp.wrap(anthropic_client(Script([])))
            gt = rp.tool(name="get_time")(lambda: None)
            m = [{"role": "user", "content": "go"}]
            r = c.messages.create(model="m", max_tokens=5, messages=m)
            m = m + [{"role": "assistant", "content": [b.model_dump(exclude_none=True) for b in r.content]},
                     {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "skipped"}]}]
            c.messages.create(model="m", max_tokens=5, messages=m)  # differs from recording: diverges here


def test_replay_never_imports_modules_named_in_a_trace(tmp_path, capsys):
    from runtape.replay import _restore

    assert _restore({"a": 1}, "dataclass:this:Anything") == {"a": 1}
    assert "Zen of Python" not in capsys.readouterr().out
