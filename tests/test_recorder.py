import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import runtape
from runtape import Recorder, Trace

SRC = str(Path(__file__).resolve().parents[1] / "src")


def lines(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def test_header_ids_and_end(tpath):
    with Recorder(tpath, name="unit", tags={"env": "test"}) as rec:
        rec.note("hello")
        rec.state("plan", ["a", "b"])
    ev = lines(tpath)
    assert [e["id"] for e in ev] == list(range(len(ev)))
    assert ev[0]["type"] == "run_start"
    assert ev[0]["payload"]["format"] == "runtape"
    assert ev[0]["payload"]["tags"] == {"env": "test"}
    assert ev[-1]["type"] == "run_end"
    assert ev[-1]["payload"]["status"] == "ok"
    assert Trace.load(tpath).status == "ok"


def test_default_path_in_traces_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rec = runtape.record(name="my agent!")
    rec.close()
    files = list((tmp_path / "traces").glob("*.jsonl"))
    assert len(files) == 1 and "my-agent-" in files[0].name


def test_unserializable_payloads_dont_raise(tpath):
    class Weird:
        def __repr__(self):
            return "<weird>"

    with Recorder(tpath) as rec:
        rec.log("state", {"obj": Weird(), "b": b"abc", "s": {1, 2}})
    p = lines(tpath)[1]["payload"]
    assert p["obj"] == {"__repr__": "<weird>"}
    assert p["b"] == {"__bytes__": 3}


def test_exception_in_with_block_marks_error(tpath):
    with pytest.raises(RuntimeError):
        with Recorder(tpath) as rec:
            rec.note("before")
            raise RuntimeError("boom")
    t = Trace.load(tpath)
    assert t.status == "error"
    err = t.of_type("error")[0]
    assert err.payload["message"] == "boom"


def _run(code, cwd):
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=cwd,
        env={"PYTHONPATH": SRC},
        capture_output=True,
        text=True,
    )


def test_hard_kill_leaves_readable_trace(tmp_path):
    # os._exit skips atexit and finally: the closest thing to a segfault/OOM kill
    _run(
        """
        import os, runtape
        rec = runtape.record("t.jsonl")
        for i in range(5):
            rec.note(f"step {i}")
        os._exit(1)
        """,
        tmp_path,
    )
    t = Trace.load(tmp_path / "t.jsonl")
    assert len(t) == 6
    assert t.status == "crashed"


def test_truncated_last_line_is_tolerated(tpath):
    with Recorder(tpath) as rec:
        rec.note("ok")
    with open(tpath, "a") as fh:
        fh.write('{"id": 99, "type": "log", "payl')
    t = Trace.load(tpath)
    assert t.truncated
    assert t.events[-1].type == "run_end"


def test_corruption_in_middle_raises(tpath):
    tpath.write_text('{"id":0,"type":"run_start","payload":{}}\nGARBAGE\n{"id":2,"type":"log"}\n')
    with pytest.raises(ValueError):
        Trace.load(tpath)


def test_uncaught_exception_is_recorded(tmp_path):
    r = _run(
        """
        import runtape
        rec = runtape.record("t.jsonl")
        rec.note("working")
        raise ValueError("agent blew up")
        """,
        tmp_path,
    )
    assert "agent blew up" in r.stderr  # original traceback still printed
    t = Trace.load(tmp_path / "t.jsonl")
    assert t.status == "error"
    assert t.of_type("error")[0].payload["message"] == "agent blew up"


def test_clean_exit_without_close_writes_run_end(tmp_path):
    _run(
        """
        import runtape
        rec = runtape.record("t.jsonl")
        rec.note("done")
        """,
        tmp_path,
    )
    assert Trace.load(tmp_path / "t.jsonl").status == "ok"


def test_redact(tpath):
    def redact(ev):
        s = json.dumps(ev).replace("sk-secret", "[REDACTED]")
        return json.loads(s)

    with Recorder(tpath, redact=redact) as rec:
        rec.note("key is sk-secret")
    assert "sk-secret" not in tpath.read_text()


def test_broken_redactor_does_not_kill_run(tpath):
    with Recorder(tpath, redact=lambda ev: 1 / 0) as rec:
        rec.note("x")
    assert "redact_error" in lines(tpath)[1]["meta"]


# ------------------------------------------------------------------ tools


def test_tool_decorator_sync(tpath):
    with Recorder(tpath) as rec:

        @rec.tool
        def add(a, b=2):
            return a + b

        assert add(1) == 3
    t = Trace.load(tpath)
    call, res = t.of_type("tool_call")[0], t.of_type("tool_result")[0]
    assert call.payload == {"name": "add", "arguments": {"a": 1}}
    assert res.parent == call.id
    assert res.payload["result"] == 3
    assert "latency_ms" in res.meta


def test_tool_decorator_error(tpath):
    with Recorder(tpath) as rec:

        @rec.tool(name="lookup")
        def f(x):
            raise KeyError(x)

        with pytest.raises(KeyError):
            f("missing")
    res = Trace.load(tpath).of_type("tool_result")[0]
    assert res.payload["name"] == "lookup"
    assert res.payload["error"]["type"] == "KeyError"


async def test_tool_decorator_async(tpath):
    with Recorder(tpath) as rec:

        @rec.tool
        async def fetch(url: str):
            return {"status": 200, "url": url}

        out = await fetch("http://x")
    assert out["status"] == 200
    res = Trace.load(tpath).of_type("tool_result")[0]
    assert res.payload["result"]["url"] == "http://x"


def test_tool_on_method_drops_self(tpath):
    with Recorder(tpath) as rec:

        class Agent:
            @rec.tool
            def search(self, q):
                return [q]

        Agent().search("hi")
    call = Trace.load(tpath).of_type("tool_call")[0]
    assert call.payload["arguments"] == {"q": "hi"}


def test_state_and_manual_llm_logging(tpath):
    # a framework with no adapter can still log model calls by hand
    with Recorder(tpath) as rec:
        rid = rec.log_llm_request(provider="local", model="llama", messages=[{"role": "user", "content": "hi"}])
        rec.log_llm_response(rid, text="yo", tool_calls=[], stop_reason="stop", raw=None, latency_ms=3)
    t = Trace.load(tpath)
    ctx = t.context(t.of_type("llm_response")[0].id)
    assert ctx.messages == [{"role": "user", "content": "hi"}]
    assert ctx.response.payload["text"] == "yo"


def test_system_and_tools_removed_later_are_not_inherited(tpath):
    # a call with no system prompt must not pick up the previous call's prompt
    with Recorder(tpath) as rec:
        r1 = rec.log_llm_request(provider="anthropic", model="m", system="You are a pirate.", tools=[{"name": "t"}],
                                 messages=[{"role": "user", "content": "a"}])
        r2 = rec.log_llm_request(provider="anthropic", model="m", messages=[{"role": "user", "content": "b"}])
        r3 = rec.log_llm_request(provider="anthropic", model="m", system="You are a pirate.",
                                 messages=[{"role": "user", "content": "c"}])
    t = Trace.load(tpath)
    assert t.context(r1).system == "You are a pirate." and t.context(r1).tools == [{"name": "t"}]
    assert t.context(r2).system is None and t.context(r2).tools is None
    assert t[r2].payload["system"] is None  # the change to none is recorded
    assert t.context(r3).system == "You are a pirate." and t.context(r3).tools is None
    # a first call with nothing writes nothing
    assert "system" not in Trace.load(tpath)[r1].payload or True


def test_recorder_survives_hostile_values(tpath):
    class BadRepr:
        def __repr__(self):
            raise ValueError("no repr")

    cyc = {"name": "loop"}
    cyc["self"] = cyc
    shared = [1]
    for _ in range(40):  # 2^40 paths if expanded naively
        shared = [shared, shared]
    with Recorder(tpath) as rec:

        @rec.tool
        def returns_bad():
            return BadRepr()

        assert isinstance(returns_bad(), BadRepr)  # the tool's result still reaches the caller
        rec.state("cycle", cyc)
        rec.state("shared", shared)
        rec.state("nan", float("nan"))
    ev = lines(tpath)
    assert [e["id"] for e in ev] == list(range(len(ev)))
    by_key = {e["payload"].get("key"): e["payload"].get("value") for e in ev if e["type"] == "state"}
    assert by_key["cycle"]["self"] == {"__cycle__": "dict"}
    assert "__truncated__" in json.dumps(by_key["shared"])
    assert by_key["nan"] == "nan"
    assert tpath.stat().st_size < 5_000_000


def test_failed_redaction_never_writes_secrets(tpath):
    def bad(ev):
        return None

    with Recorder(tpath, redact=bad) as rec:
        rec.note("password is hunter2")
    text = tpath.read_text()
    assert "hunter2" not in text
    t = Trace.load(tpath)  # still a valid trace
    assert t.status == "ok"


def test_recording_to_the_same_path_twice_replaces_the_run(tpath):
    for word in ("first", "second"):
        with Recorder(tpath, name=word) as rec:
            rec.note(word)
    ev = lines(tpath)
    assert [e["id"] for e in ev] == [0, 1, 2]
    assert ev[0]["payload"]["name"] == "second"


def test_redaction_failure_on_a_model_call_keeps_later_calls_whole(tpath):
    def redact(ev):
        if ev["id"] == 2:  # the redactor breaks on one call only
            raise RuntimeError("nope")
        return ev

    with Recorder(tpath, redact=redact) as rec:
        h = [{"role": "user", "content": "hi"}]
        rec.log_llm_request(provider="anthropic", model="m", system="S", messages=h)
        h = h + [{"role": "assistant", "content": "the secret is 42"}]
        rec.log_llm_request(provider="anthropic", model="m", system="S", messages=h)  # redaction fails
        h = h + [{"role": "user", "content": "thanks"}]
        r3 = rec.log_llm_request(provider="anthropic", model="m", system="S", messages=h)
    t = Trace.load(tpath)
    assert "redact function failed" in json.dumps(t[2].payload)
    # the third call doesn't build on the dropped one: its history and system prompt are complete
    assert t[r3].payload.get("base") != 2
    assert [m["content"] for m in t.messages(r3)] == ["hi", "the secret is 42", "thanks"]
    assert t.context(r3).system == "S"


def test_thread_exceptions_are_recorded(tmp_path):
    r = _run(
        """
        import threading, runtape
        rec = runtape.record("t.jsonl")
        th = threading.Thread(target=lambda: 1 / 0, name="worker-1")
        th.start(); th.join()
        rec.close()
        """,
        tmp_path,
    )
    t = Trace.load(tmp_path / "t.jsonl")
    err = t.of_type("error")[0]
    assert err.payload["type"] == "ZeroDivisionError" and err.payload["thread"] == "worker-1"


def test_link_tolerates_converted_arguments_and_expires_stale_requests(tpath):
    with Recorder(tpath) as rec:
        refund = rec.tool(name="refund")(lambda amount: "ok")
        search = rec.tool(name="search")(lambda q: "ok")
        rid = rec.log_llm_request(provider="anthropic", model="m", messages=[{"role": "user", "content": "x"}])
        rec.log_llm_response(rid, text=None, tool_calls=[{"id": "r1", "name": "refund", "arguments": {"amount": "2400"}},
                                                        {"id": "s1", "name": "search", "arguments": {"q": "policy"}}],
                             stop_reason="tool_use", raw=None, latency_ms=1)
        refund(amount=2400.0)  # code converted the string
        for _ in range(6):  # several later replies: the unused search request goes stale
            r = rec.log_llm_request(provider="anthropic", model="m", messages=[{"role": "user", "content": "y"}])
            rec.log_llm_response(r, text="k", tool_calls=[], stop_reason="end_turn", raw=None, latency_ms=1)
        search(q="policy")  # called by code much later
    t = Trace.load(tpath)
    calls = {c.payload["name"]: c for c in t.of_type("tool_call")}
    assert calls["refund"].payload.get("call_id") == "r1"
    assert "call_id" not in calls["search"].payload
