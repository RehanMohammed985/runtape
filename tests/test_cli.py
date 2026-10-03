import io
import json
import os
from pathlib import Path

import pytest
from rich.console import Console

from runtape import Recorder, cli

from .test_example import load_example


@pytest.fixture
def demo(tmp_path, monkeypatch):
    """The refund-bot trace, recorded into ./traces of a temp dir."""
    monkeypatch.chdir(tmp_path)
    path = load_example().main()
    return path


def run(*argv) -> tuple[int, str]:
    buf = io.StringIO()
    c = Console(file=buf, width=140, no_color=True, highlight=False)
    code = cli.main(list(argv), console=c)
    return code, buf.getvalue()


def replay(path, script: str, width=140) -> str:
    buf = io.StringIO()
    c = Console(file=buf, width=width, no_color=True, highlight=False)
    from runtape import Trace

    r = cli.Replay(Trace.load(path), c)
    r.use_rawinput = False
    r.stdin = io.StringIO(script)
    r.stdout = buf
    r.cmdloop()
    return buf.getvalue()


def test_ls_and_last(demo):
    code, out = run("ls")
    assert code == 0 and "refund-bot" in out and "ok" in out
    code, out = run("summary")  # no path: newest trace
    assert "model calls  9" in out
    assert "issue_refund x2" in out


def test_partial_name_resolves(demo):
    code, out = run("summary", demo.name[-12:])
    assert code == 0 and "refund-bot" in out


def test_missing_trace_is_a_clean_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    code, out = run("summary")
    assert code == 1 and "No traces" in out
    code, out = run("show", "nope.jsonl", "1")
    assert code == 1 and "No trace file" in out


def test_show_and_bad_event(demo):
    code, out = run("show", str(demo), "30")
    assert "issue_refund" in out and "2400" in out
    code, out = run("show", str(demo), "-1")
    assert "run_end" in out
    code, out = run("show", str(demo), "500")
    assert code == 1 and "No event #500" in out
    code, out = run("show", str(demo), "9", "--raw")
    assert '"type": "tool_result"' in out


def test_grep_finds_entry_point(demo):
    code, out = run("grep", str(demo), "any amount")
    assert "First appears at #9" in out
    assert "entered here" in out
    assert "result[1].text" in out


def test_context(demo):
    code, out = run("context", str(demo), "30")
    assert "You are the support agent" in out
    assert "community/forum/post-8812.md" in out  # poison still in the window at the decision
    assert "model reply #30" in out
    code, out = run("context", str(demo), "1")
    assert "Nothing in context yet" in out


def test_diff(demo):
    code, out = run("diff", str(demo), "25", "30")
    assert "+2 messages added, nothing removed" in out
    assert "standing desk" in out
    code, out = run("diff", str(demo), "27", "28")
    assert "did not change" in out


def test_diff_detects_rewritten_context(tmp_path):
    # context trimming/summarization: the second request is not a prefix extension
    p = tmp_path / "t.jsonl"
    with Recorder(p) as rec:
        rec.log_llm_request(provider="x", model="m", system="v1", messages=[
            {"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "user", "content": "c"}])
        rec.log_llm_request(provider="x", model="m", system="v2", messages=[
            {"role": "user", "content": "summary of a/b"}, {"role": "user", "content": "c"}])
    code, out = run("diff", str(p), "1", "2")
    assert "system prompt changed" in out
    assert "context rewritten: 0 messages kept" in out
    assert "summary of a/b" in out


def test_timeline(demo):
    code, out = run("timeline")
    assert out.count("\n") >= 37


def test_replay_session(demo):
    out = replay(
        demo,
        "\n".join(
            [
                "",  # enter steps
                "step tool_call",
                "back 1",
                "grep any amount",
                "next",
                "next",
                "prev",
                "context",
                "diff",
                "goto 30",
                "30",
                "list 1",
                "summary",
                "errors",
                "step bogus",
                "goto 999",
                "last",
                "step",
                "help",
                "quit",
            ]
        ),
    )
    assert "#1 state" in out
    assert "#4 tool_call" in out
    assert "First appears at #9" in out
    assert "#10 llm_request" in out
    assert "context at #9" in out
    assert ">  30 llm_response" in out
    assert "No errors." in out
    assert "Unknown event type 'bogus'" in out
    assert "No event #999" in out
    assert "At the end of the trace." in out
    assert "Commands" in out


def test_replay_type_aliases_and_empty_hits(demo):
    out = replay(demo, "step err\nnext\ngrep zzzqqq\nstep reply\nq\n")
    assert "No error after" in out
    assert "No search yet" in out
    assert 'No events contain "zzzqqq"' in out
    assert "#3 llm_response" in out


def test_crashed_trace_summary(tmp_path):
    p = tmp_path / "t.jsonl"
    rec = Recorder(p)
    rec.note("x")
    rec._fh.write('{"id": 2, "ty')
    rec._fh.flush()
    code, out = run("summary", str(p))
    assert "crashed" in out and "truncated" in out


def test_python_dash_m(demo):
    import subprocess
    import sys

    r = subprocess.run([sys.executable, "-m", "runtape", "--no-color", "summary"], capture_output=True, text=True,
                       env={**os.environ, "COLUMNS": "120"})
    assert r.returncode == 0 and "refund-bot" in r.stdout


def test_replay_opens_on_overview(demo):
    buf = io.StringIO()
    c = Console(file=buf, width=80, no_color=True, highlight=False)
    from runtape import Trace

    cli.Replay(Trace.load(demo), c).overview()
    out = buf.getvalue()
    assert "refund-bot  ok  |  37 events, 9 model calls, 6 tool calls, 0 errors" in out
    assert "36 run_end" in out
    assert '"run_id"' not in out  # no raw metadata dump
    assert all(len(line) <= 80 for line in out.splitlines())  # rows never wrap


SIM = str(Path(__file__).resolve().parents[1] / "examples" / "refund_bot.py") + ":simulated_model"


def test_cli_why(demo):
    code, out = run("why", str(demo), "31", "--model-fn", SIM, "--no-cache")
    assert code == 0
    assert 'Why does the agent call issue_refund(order_id="B-2290", amount=2400.0)?' in out
    assert "On the identical context it calls issue_refund in 10/10 reruns" in out
    assert "CAUSE  #9 search_kb result[1].text sentence 2" in out
    assert "masked" in out
    assert 'Without it the agent calls issue_refund in 0/10 reruns and instead calls escalate_to_manager(order_id="B-2290") (10/10)' in out
    assert "also required" in out and "#28 lookup_order" in out
    assert "significant after correcting for" in out


def test_cli_why_json_and_cache(demo, tmp_path):
    out_json = tmp_path / "why.json"
    run("why", str(demo), "31", "--model-fn", SIM, "--json", str(out_json))
    code, out = run("why", str(demo), "31", "--model-fn", SIM)  # second run: all cached
    assert "0 model calls" in out
    data = json.loads(out_json.read_text())
    cause = data["causes"][0]
    assert cause["kind"] == "decisive" and cause["masked"] is True
    assert cause["chain"][-1]["removed"][0]["origin"] == 9
    assert data["baseline"] == {"happens": 10, "runs": 10, "intermittent": False}


def test_cli_rerun_and_odds(demo):
    code, out = run("rerun", str(demo), "30", "--model-fn", SIM, "--drop", "9[1]")
    assert "dropped #9 search_kb result[1]" in out
    assert '5/5  calls escalate_to_manager(order_id="B-2290")' in out
    code, out = run("rerun", str(demo), "30", "--model-fn", SIM, "--replace", "any amount=>up to $200")
    assert "escalate_to_manager" in out
    code, out = run("odds", str(demo), "30", "--model-fn", SIM)
    assert "(same as recorded)" in out
    code, out = run("rerun", str(demo), "30", "--model-fn", SIM, "--drop", "77")
    assert code == 1 and "nothing in this context came from event #77" in out
    code, out = run("why", str(demo), "0", "--model-fn", SIM)
    assert code == 1 and "run_start" in out


def test_replay_why(demo):
    from runtape import Trace

    buf = io.StringIO()
    c = Console(file=buf, width=140, no_color=True, highlight=False)
    r = cli.Replay(Trace.load(demo), c)
    r.model_fn = SIM
    r.use_rawinput = False
    r.stdin = io.StringIO("goto 31\nwhy\nrerun --drop 9[1]\nodds 30 --runs 3\nwhy 0\njump 5\nq\n")
    r.stdout = buf
    r.cmdloop()
    out = buf.getvalue()
    assert "CAUSE  #9 search_kb result[1].text sentence 2" in out
    assert "dropped #9 search_kb result[1]" in out
    assert "3/3  calls issue_refund" in out
    assert "run_start" in out  # why 0 reports the error and keeps the session alive
    assert "#5 tool_result" in out


def test_bare_runtape_opens_replay(demo, monkeypatch):
    # `runtape` with no arguments is the README's first command: it must open the replay
    monkeypatch.setattr("sys.stdin", io.StringIO("q\n"))
    code, out = run()
    assert code == 0 and "refund-bot  ok" in out


def test_bad_negative_and_last_references(demo):
    code, out = run("show", str(demo), "-999")
    assert code == 1 and "No event #-999" in out
    code, out = run("why", str(demo), "last", "--model-fn", SIM, "--dry")
    assert code == 0 and "Suspects for the decision at #35" in out  # last model reply, not run_end
    r_out = replay(demo, "goto -999\nshow\nq\n")
    assert "No event #-999" in r_out and "#0 run_start" in r_out  # session survives


def test_trace_path_alone_opens_replay(demo, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("q\n"))
    code, out = run(str(demo))
    assert code == 0 and "refund-bot  ok" in out


def test_trace_path_after_global_option(demo, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("q\n"))
    code, out = run("--no-color", str(demo))
    assert code == 0 and "refund-bot  ok" in out


# ---- what a new user runs into (from a fresh-install pass over the README)


def test_next_hint_keeps_the_options_that_were_passed(demo):
    """The suggested `fix` command runs as printed: same event spec, same model function."""
    code, out = run("why", str(demo), "tool:issue_refund", "--model-fn", SIM)
    hint = " ".join(out[out.index("Next:"):].split())
    assert f"runtape fix {demo} tool:issue_refund --model-fn {SIM}" in hint, hint
    code, out = run("fix", str(demo), "tool:issue_refund", "--model-fn", SIM)
    hint = " ".join(out[out.index("Turn it into a test:"):].split())
    assert f"tool:issue_refund --model-fn {SIM} --write-test" in hint, hint


def test_reruns_that_cannot_start_say_why_before_asking(demo, monkeypatch):
    # the example's stand-in model has no API: point at --model-fn instead of asking for a key
    code, out = run("why", str(demo), "31", "-y")
    assert code == 1 and "stand-in model" in out and "--model-fn" in out
    # a real trace with no key set: name the variable, before any prompt or spinner
    t = Recorder(Path("traces/real.jsonl"))
    rid = t.log_llm_request(provider="openai", api="chat.completions", model="gpt-4o-mini",
                            messages=[{"role": "user", "content": "hi"}])
    t.log_llm_response(rid, text="hello", tool_calls=[], stop_reason="stop", raw=None, latency_ms=1)
    t.close()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    code, out = run("why", "traces/real.jsonl", "last")
    assert code == 1 and "OPENAI_API_KEY isn't set" in out and "Continue?" not in out
    code, out = run("rerun", "traces/real.jsonl", "last")
    assert code == 1 and "OPENAI_API_KEY isn't set" in out


def test_errors_name_what_exists(demo):
    code, out = run("why", str(demo), "issue_refund", "--model-fn", SIM)
    assert code == 1 and "tool:issue_refund" in out
    code, out = run("why", str(demo), "tool:Issue_Refund", "--model-fn", SIM)
    assert code == 1 and "It has:" in out and "issue_refund" in out
    code, out = run("why", str(demo), "31", "--model-fn", SIM, "--match", "[")
    assert code == 1 and "--match '[' isn't a valid regular expression" in out
    code, out = run("why", str(demo), "31", "--model-fn", "nope.py:f")
    assert code == 1 and "no file nope.py" in out
    code, out = run("why", str(demo), "31", "--model-fn", SIM.rsplit(":", 1)[0] + ":nope")
    assert code == 1 and "has no 'nope'" in out
    code, out = run("import", "missing.json")
    assert code == 1 and out.strip() == "No such file: missing.json"


def test_a_generic_tool_gets_a_match_suggestion(tmp_path, monkeypatch):
    """Explaining one of many run_command calls: say that any call counts, and how to explain this one."""
    monkeypatch.chdir(tmp_path)
    t = Recorder(Path("traces/ops.jsonl"))
    msgs = [{"role": "user", "content": "fix the deploy"}]
    for i, cmd in enumerate(["make test", "make db-reset"]):
        rid = t.log_llm_request(provider="openai", api="chat.completions", model="m", messages=list(msgs))
        call = {"id": f"c{i}", "name": "run_command", "arguments": {"command": cmd}}
        reply = t.log_llm_response(rid, text=None, tool_calls=[call], stop_reason="tool_calls", raw=None,
                                   latency_ms=1)
        t.log("tool_call", {"name": "run_command", "arguments": {"command": cmd}, "call_id": f"c{i}"},
              meta={"requested_by": reply})
        msgs += [{"role": "assistant", "content": None, "tool_calls": [
            {"id": f"c{i}", "type": "function", "function": {"name": "run_command", "arguments": "{}"}}]},
                 {"role": "tool", "tool_call_id": f"c{i}", "content": "ok"}]
    t.close()
    from runtape import Trace

    tr = Trace.load("traces/ops.jsonl")
    note = cli.generic_tool_note(tr, cli.resolve_event(tr, "tool:run_command"))
    assert "called 2 times" in note and "--match 'make db-reset'" in note
    assert cli.generic_tool_note(tr, 2, match="db-reset") is None


def test_replay_why_takes_a_model_function(demo):
    out = replay(demo, f"why 31 --model-fn {SIM}\nwhy 31 --bogus\nq\n")
    assert "CAUSE  #9 search_kb result[1].text sentence 2" in out
    assert "unrecognized arguments: --bogus" in out and "why: 2" not in out


def test_last_and_part_of_a_name_open_the_replay(demo, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("q\n"))
    assert run("last")[1].count("refund-bot  ok") == 1
    monkeypatch.setattr("sys.stdin", io.StringIO("q\n"))
    assert "refund-bot  ok" in run("refund-bot")[1]


def test_a_test_finds_its_model_function_relative_to_itself(demo, tmp_path):
    code, out = run("fix", str(demo), "31", "--model-fn", SIM, "--write-test", "tests/test_refund.py")
    text = Path("tests/test_refund.py").read_text()
    assert "Path(__file__).parent / " in text and str(tmp_path) not in text and code == 0


def test_rerun_takes_one_reference_as_a_string(demo):
    import runtape

    d = runtape.rerun(demo, 30, drop="9[1]", model=__import__("runtape.rerun", fromlist=["load_model_fn"])
                      .load_model_fn(SIM), cache_dir=None, runs=2)
    assert d.notes == ["dropped #9 search_kb result[1]"]
