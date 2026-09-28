import io
import os

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
    assert "community/faq-2019.md" in out  # poison still in the window at the decision
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
