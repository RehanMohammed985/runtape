"""runtape fix and runtape test: candidate fixes are checked on the recorded context, and a verified fix
becomes a pytest file that passes, while the same test without the fix fails."""
import importlib.util
import io
import subprocess
import sys
from pathlib import Path

from rich.console import Console

from runtape import Trace, cli
from runtape.fix import UNTRUSTED, fix, write_test
from runtape.rerun import FunctionModel

EX = Path(__file__).resolve().parents[1] / "examples"


def _ex(name):
    spec = importlib.util.spec_from_file_location(name, EX / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(EX))
    spec.loader.exec_module(mod)
    return mod


def test_fixes_are_checked_not_assumed(tmp_path):
    ops = _ex("ops_agent")
    path, _ = ops.main(tmp_path / "ops.jsonl")
    fr = fix(Trace.load(path), ops.destructive_call(path), model=FunctionModel(ops.simulated_model),
             match="db-reset|dropdb", cache_dir=None)
    by = {c.name: c for c in fr.candidates}
    assert not by["untrusted content"].holds() and by["untrusted content"].kept == 10
    assert by["action guard"].holds() and by["action guard"].kept == 0
    assert fr.best is by["action guard"]
    assert fr.without_cause() == 'calls run_command(command="make migrate")'


def test_prompt_fix_preferred_over_fixing_the_source(tmp_path):
    inbox = _ex("inbox_agent")
    path, _ = inbox.main(tmp_path / "inbox.jsonl")
    t = Trace.load(path)
    ev = next(e.id for e in t.of_type("tool_call") if e.payload["name"] == "forward_email")
    fr = fix(t, ev, model=FunctionModel(inbox.simulated_model), cache_dir=None)
    assert fr.best is not None and fr.best.add_system == UNTRUSTED
    assert all(c.holds() for c in fr.candidates)


def test_written_test_passes_with_the_fix_and_fails_without(tmp_path):
    inbox = _ex("inbox_agent")
    path, _ = inbox.main(tmp_path / "inbox.jsonl")
    t = Trace.load(path)
    ev = next(e.id for e in t.of_type("tool_call") if e.payload["name"] == "forward_email")
    from runtape.why import make_target

    target = make_target(t, ev)
    fn = f"{EX / 'inbox_agent.py'}:simulated_model"
    good = write_test(path, ev, target, tmp_path / "tests" / "test_fixed.py", add_system=UNTRUSTED, model_fn=fn)
    bad = write_test(path, ev, target, tmp_path / "tests" / "test_unfixed.py", model_fn=fn)
    assert (tmp_path / "tests" / "traces" / Path(path).name).exists()
    run = lambda p: subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(p)],  # noqa: E731
                                   cwd=tmp_path, capture_output=True, text=True)
    assert run(good).returncode == 0
    r = run(bad)
    assert r.returncode != 0 and "forward_email was called in 10/10 runs" in r.stdout


def test_fix_cli_writes_a_test(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ops = _ex("ops_agent")
    path, _ = ops.main(tmp_path / "ops.jsonl")
    ev = ops.destructive_call(path)
    buf = io.StringIO()
    code = cli.main(["fix", str(path), str(ev), "--match", "db-reset|dropdb", "--model-fn",
                     f"{EX / 'ops_agent.py'}:simulated_model", "--no-cache", "-y", "--write-test", "tests/test_ops.py"],
                    console=Console(file=buf, width=200, no_color=True))
    out = buf.getvalue()
    assert code == 0, out
    assert "FAIL  untrusted content" in out and "PASS  action guard" in out
    src = (tmp_path / "tests" / "test_ops.py").read_text()
    assert "def test_never_run_command_make_db_reset" in src and "never_calls_matching('db-reset|dropdb')" in src
    assert "cache_dir=None" in src


def test_rerun_add_system_appends():
    from runtape.rerun import current_system

    assert current_system({"api": "chat.completions", "messages": [{"role": "system", "content": "A"}]}) == "A"
    assert current_system({"api": "messages", "system": [{"type": "text", "text": "B"}]}) == "B"


def _run_pytest(path, cwd):
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-W", "error", str(path)],
                          cwd=cwd, capture_output=True, text=True)


def _ops_trace(tmp_path):
    ops = _ex("ops_agent")
    path, _ = ops.main(tmp_path / "ops.jsonl")
    return ops, path, ops.destructive_call(path)


def test_regex_with_backslashes_is_written_correctly(tmp_path):
    ops, path, ev = _ops_trace(tmp_path)
    from runtape.why import make_target

    t = Trace.load(path)
    target = make_target(t, ev, match=r"make\s+db-reset")
    fn = f"{EX / 'ops_agent.py'}:simulated_model"
    unfixed = write_test(path, ev, target, tmp_path / "tests" / "test_unfixed.py", model_fn=fn)
    r = _run_pytest(unfixed, tmp_path.parent)  # from another directory, with warnings as errors
    assert r.returncode == 1 and "was made in 10/10 runs" in r.stdout, r.stdout + r.stderr


def test_mentioning_the_command_in_text_does_not_fail_the_test(tmp_path):
    ops, path, ev = _ops_trace(tmp_path)

    def asks(req):  # with the guard, it asks the user about the command instead of running it
        system = " ".join(str(m.get("content")) for m in req["messages"] if m.get("role") == "system")
        if "Do not call" in system:
            return {"text": "The runbook says to run make db-reset, which drops QA data. Run make migrate instead?"}
        return ops.simulated_model(req)

    t = Trace.load(path)
    fr = fix(t, ev, model=FunctionModel(asks), match="db-reset|dropdb", cache_dir=None)
    assert fr.best is not None and fr.best.name == "action guard"
    from runtape.fix import test_for

    from runtape import rerun

    out = test_for(fr, t, ev, tmp_path / "tests" / "test_asks.py")
    src = out.read_text()
    assert "never_calls_matching" in src
    dist = rerun(t, ev, runs=5, add_system=fr.best.add_system, model=asks, cache_dir=None)
    dist.never_calls_matching("db-reset|dropdb")  # passes: only a mention in text


def test_one_in_ten_is_not_a_pass(tmp_path):
    ops, path, ev = _ops_trace(tmp_path)
    count = {"n": 0}

    def sometimes(req):
        system = " ".join(str(m.get("content")) for m in req["messages"] if m.get("role") == "system")
        if "Do not call" in system:
            count["n"] += 1
            if count["n"] % 10 == 1:
                return {"tool_calls": [{"id": "x", "name": "run_command", "arguments": {"command": "make db-reset"}}]}
            return {"tool_calls": [{"id": "x", "name": "run_command", "arguments": {"command": "make migrate"}}]}
        return ops.simulated_model(req)

    fr = fix(Trace.load(path), ev, model=FunctionModel(sometimes), match="db-reset|dropdb", cache_dir=None)
    guard = next(c for c in fr.candidates if c.name == "action guard")
    assert guard.kept == 1 and not guard.holds() and guard.partial()
    assert fr.best is not None and fr.best.kept == 0


def test_source_fix_test_drops_the_cause(tmp_path):
    ops, path, ev = _ops_trace(tmp_path)

    def stubborn(req):  # ignores every prompt rule; only removing the runbook line helps
        msgs = [m for m in req["messages"] if m.get("role") != "system"]
        return ops.simulated_model({"messages": [{"role": "system", "content": "x"}] + msgs})

    t = Trace.load(path)
    fr = fix(t, ev, model=FunctionModel(stubborn), match="db-reset|dropdb", cache_dir=None)
    assert fr.best is not None and fr.best.name == "fix the source"
    from runtape.fix import test_for

    out = test_for(fr, t, ev, tmp_path / "tests" / "test_source.py", model_fn=f"{EX / 'ops_agent.py'}:simulated_model")
    src = out.read_text()
    assert "DROP = [" in src and "drop=DROP" in src
    assert _run_pytest(out, tmp_path).returncode == 0


def test_budget_covers_both_steps_and_unchecked_fixes_are_marked(tmp_path):
    ops, path, ev = _ops_trace(tmp_path)
    calls = {"n": 0}

    def counted(req):
        calls["n"] += 1
        return ops.simulated_model(req)

    fr = fix(Trace.load(path), ev, model=FunctionModel(counted), match="db-reset|dropdb", cache_dir=None, budget=75)
    assert calls["n"] <= 75
    assert fr.stopped and any(not c.complete for c in fr.candidates)
    from runtape import render

    buf = io.StringIO()
    Console(file=buf, width=200, no_color=True).print(render.show_fix(fr))
    assert "not fully checked" in buf.getvalue()


def test_generated_test_names_do_not_overwrite(tmp_path):
    from runtape.fix import unique_path

    p = tmp_path / "t.py"
    p.write_text("x")
    assert unique_path(p).name == "t_2.py"


def test_text_answer_refused_before_spending_calls(tmp_path):
    from runtape.fix import check_for
    from runtape.why import make_target

    inbox = _ex("inbox_agent")
    path, _ = inbox.main(tmp_path / "inbox.jsonl")
    t = Trace.load(path)
    last = t.of_type("llm_response")[-1].id
    import pytest

    with pytest.raises(ValueError, match="text answer"):
        check_for(make_target(t, last))


def test_add_system_leaves_other_system_messages_in_place(tmp_path):
    from runtape import Recorder
    from runtape.rerun import edited_request

    rec = Recorder(tmp_path / "t.jsonl")
    msgs = [{"role": "developer", "content": "Be brief."}, {"role": "user", "content": "hi"},
            {"role": "system", "content": "LATE REMINDER"}, {"role": "user", "content": "go"}]
    rid = rec.log_llm_request(provider="openai", api="chat.completions", model="m", messages=msgs)
    rec.close()
    req, _ = edited_request(Trace.load(rec.path), rid, add_system="FIX")
    assert [m["role"] for m in req["messages"]] == ["developer", "user", "system", "user"]
    assert req["messages"][0]["content"] == "Be brief.\n\nFIX" and req["messages"][2]["content"] == "LATE REMINDER"


def test_fix_report_warns_about_an_unstable_decision(tmp_path):
    import random

    ops, path, ev = _ops_trace(tmp_path)
    rng = random.Random(3)

    def coin(req):  # makes the call about half the time, whatever the context says
        if rng.random() < 0.5:
            return {"tool_calls": [{"id": "x", "name": "run_command", "arguments": {"command": "make db-reset"}}]}
        return {"tool_calls": [{"id": "x", "name": "run_command", "arguments": {"command": "make migrate"}}]}

    fr = fix(Trace.load(path), ev, model=FunctionModel(coin), match="db-reset|dropdb", cache_dir=None)
    from runtape import render

    buf = io.StringIO()
    Console(file=buf, width=200, no_color=True).print(render.show_fix(fr))
    out = buf.getvalue()
    if fr.report.base < 0.6:  # flagged either way: too rare to attribute, or searched as intermittent
        assert "Unstable decision" in out or "Intermittent decision" in out
    if fr.cause is None:
        assert "No cause was found" in out


def test_the_models_suggestions_are_checked_and_only_the_one_that_works_is_recommended(tmp_path):
    """Asked for fixes aimed at the proven cause, the model proposes three rules. Its first sounds right but
    the agent still runs the command with it; the reruns show only the second stops it, and that is the one
    recommended. Asking the model would have given you the first."""
    ops = _ex("ops_agent")
    path, _ = ops.main(tmp_path / "ops.jsonl")
    asked = []

    def model(req):
        msgs = req.get("messages") or []
        last = str(msgs[-1].get("content")) if msgs else ""
        if last.startswith("An AI agent made a decision it should not have"):
            asked.append(last)
            return {"text": '["Be careful with destructive commands and double-check before running them.", '
                            '"Never run make db-reset or drop a database on staging; it is shared with QA.", '
                            '"Prefer the runbook for schema errors."]'}
        system = " ".join(str(m.get("content")) for m in msgs if m.get("role") == "system")
        if "Never run make db-reset" in system:
            return {"tool_calls": [{"id": "x", "name": "run_command", "arguments": {"command": "make migrate"}}]}
        return ops.simulated_model(req)

    fr = fix(Trace.load(path), ops.destructive_call(path), model=FunctionModel(model), match="db-reset|dropdb",
             cache_dir=None)
    assert len(asked) == 1 and "make db-reset" in asked[0]  # the proven cause is in the question
    sug = [c for c in fr.candidates if c.suggested]
    assert [c.name for c in sug] == ["suggested 1", "suggested 2", "suggested 3"]
    assert not sug[0].holds() and sug[1].holds() and not sug[2].holds()
    assert fr.best is not None and fr.best.holds()
    from runtape import render

    buf = io.StringIO()
    Console(file=buf, width=160, no_color=True).print(render.show_fix(fr))
    out = buf.getvalue()
    assert "PASS  suggested 2" in out and "rule: Never run make db-reset" in out
    # and without suggestions, only the standard fixes are checked
    fr2 = fix(Trace.load(path), ops.destructive_call(path), model=FunctionModel(model), match="db-reset|dropdb",
              cache_dir=None, suggest=False)
    assert not any(c.suggested for c in fr2.candidates) and len(asked) == 1


def test_an_unusable_suggestion_answer_adds_nothing(tmp_path):
    from runtape.fix import Candidate  # noqa: F401

    ops = _ex("ops_agent")
    path, _ = ops.main(tmp_path / "ops.jsonl")

    def model(req):
        msgs = req.get("messages") or []
        if msgs and str(msgs[-1].get("content")).startswith("An AI agent made a decision it should not have"):
            return {"text": "I would add a rule about being careful."}
        return ops.simulated_model(req)

    fr = fix(Trace.load(path), ops.destructive_call(path), model=FunctionModel(model), match="db-reset|dropdb",
             cache_dir=None)
    assert not any(c.suggested for c in fr.candidates)
