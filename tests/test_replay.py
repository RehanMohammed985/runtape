"""Replaying a recorded run through the real agent code."""
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


def test_tool_argument_change_is_caught(recorded, tmp_path):
    ex, orig = recorded
    src = Trace.load(orig)
    with pytest.raises(ReplayDiverged) as err:
        with runtape.replay(src, tmp_path / "r.jsonl") as rp:
            @rp.tool
            def lookup_order(order_id):
                return {}

            lookup_order("A-1042")  # matches recorded tool call 1
            lookup_order("WRONG")  # recorded call 2 is search_kb
    d = err.value.divergence
    assert d.kind == "tool_call" and d.step == 2 and d.field == "name"


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
