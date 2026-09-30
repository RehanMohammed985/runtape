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
    assert "def test_never_run_command_make_db_reset" in src and 'never_matches(r"db-reset|dropdb")' in src


def test_rerun_add_system_appends():
    from runtape.rerun import current_system

    assert current_system({"api": "chat.completions", "messages": [{"role": "system", "content": "A"}]}) == "A"
    assert current_system({"api": "messages", "system": [{"type": "text", "text": "B"}]}) == "B"
