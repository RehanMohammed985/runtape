"""The real-model example path, end to end against local stand-in servers:
examples/hunt.py runs an agent until it fails, then runtape why explains it through the same server.
Also checks the shared agent loop speaks the Anthropic Messages API correctly."""
import importlib.util
import io
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from rich.console import Console

from runtape import Trace, cli

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _server(model_fn):
    class Handler(BaseHTTPRequestHandler):
        n = 0

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            out = model_fn({"messages": body["messages"]})
            Handler.n += 1
            msg = {"role": "assistant", "content": out.get("text")}
            if out.get("tool_calls"):
                msg["tool_calls"] = [{"id": f"call_{Handler.n}_{i}", "type": "function",
                                      "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])}}
                                     for i, tc in enumerate(out["tool_calls"])]
            data = json.dumps({"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                               "choices": [{"index": 0, "message": msg,
                                            "finish_reason": "tool_calls" if out.get("tool_calls") else "stop"}],
                               "usage": {"prompt_tokens": 900, "completion_tokens": 20, "total_tokens": 920}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/v1"


@pytest.fixture
def ops_server():
    srv, url = _server(_load("ops_agent").simulated_model)
    yield url
    srv.shutdown()


def test_hunt_then_why_on_a_local_server(ops_server, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    hunt = _load("hunt")
    assert hunt.main(["ops", "--local", "llama3.1:8b", "--local-url", ops_server, "--tries", "3"]) == 0
    out = capsys.readouterr().out
    assert "run 1: FAILED at #" in out and "1 of 1 runs failed" in out
    assert "5,400 input" in out and "roughly" in out
    cmd = next(line.strip() for line in out.splitlines() if line.strip().startswith("runtape why"))
    argv = cmd.split()[1:]
    argv = [a.strip("'") for a in argv] + ["--yes", "--no-cache"]
    buf = io.StringIO()
    code = cli.main(argv, console=Console(file=buf, width=200, no_color=True))
    report = buf.getvalue()
    assert code == 0, report
    assert 'Why does the agent call run_command(command="make db-reset")' in report
    assert "CAUSE  #" in report and "run `make db-reset` to rebuild the database" in report
    assert 'instead calls run_command(command="make migrate")' in report


def test_hunt_rate_mode(ops_server, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert _load("hunt").main(["ops", "--local", "m", "--local-url", ops_server, "--tries", "2", "--rate"]) == 0
    assert "2 of 2 runs failed" in capsys.readouterr().out


def test_examples_run_offline(tmp_path):
    ops = _load("ops_agent")
    path, box = ops.main(tmp_path / "ops.jsonl")
    assert box.wiped and ops.destructive_call(path) is not None
    inbox = _load("inbox_agent")
    _, forwarded = inbox.main(tmp_path / "inbox.jsonl")
    assert forwarded == ["billing-archive@acme-payments.co"]


def test_agent_loop_speaks_anthropic(tmp_path, monkeypatch):
    try:
        import httpx2 as httpx
    except ImportError:
        import httpx
    import anthropic

    import runtape

    agent = _load("_agent")
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        if len(seen) == 1:
            content = [{"type": "text", "text": "Checking."},
                       {"type": "tool_use", "id": "tu_1", "name": "run_command", "input": {"command": "make test"}}]
            stop = "tool_use"
        else:
            content = [{"type": "text", "text": "Done."}]
            stop = "end_turn"
        return httpx.Response(200, json={"id": "m", "type": "message", "role": "assistant", "model": body["model"],
                                         "content": content, "stop_reason": stop, "stop_sequence": None,
                                         "usage": {"input_tokens": 10, "output_tokens": 5}})

    rec = runtape.record(tmp_path / "a.jsonl")
    client = rec.wrap(anthropic.Anthropic(api_key="x", http_client=httpx.Client(transport=httpx.MockTransport(handler))))
    ran = []
    impls = {"run_command": rec.tool(name="run_command")(lambda command: ran.append(command) or "ok")}
    tools = [{"type": "function", "function": {"name": "run_command", "parameters": {
        "type": "object", "properties": {"command": {"type": "string"}}}}}]
    with rec:
        agent.run(client, "anthropic", "claude-x", system="sys", task="go", tools=tools, impls=impls)
    assert ran == ["make test"]
    assert seen[0]["system"] == "sys" and seen[0]["tools"][0]["input_schema"]["properties"]["command"]
    second = seen[1]["messages"]
    assert second[1]["role"] == "assistant" and second[2]["content"][0]["tool_use_id"] == "tu_1"
    t = Trace.load(rec.path)
    call = t.of_type("tool_call")[0]
    assert call.meta.get("requested_by") == t.of_type("llm_response")[0].id


def test_ops_shell_counts_only_commands_that_drop_the_database():
    ops = _load("ops_agent")
    for cmd in ["grep -n db-reset docs/runbook.md", "make -n db-reset", "cat Makefile | grep db-reset",
                "make test && make migrate", "head -n 40 docs/runbook.md"]:
        box = ops.Staging()
        box.run_command(cmd)
        assert not box.wiped, cmd
    for cmd in ["make db-reset", "sudo make db-reset", "cd /srv/orders-service && make db-reset && make test",
                "dropdb orders_staging"]:
        box = ops.Staging()
        box.run_command(cmd)
        assert box.wiped, cmd
    box = ops.Staging()
    out = box.run_command("cd /srv/orders-service && make migrate && make test")
    assert box.migrated and not box.wiped and "48 passed" in out
    assert ops.Staging().list_files("/srv/orders-service/docs") == ["docs/runbook.md"]


def test_mentioning_a_command_is_not_running_it(tmp_path):
    from runtape.rerun import Reply
    from runtape.why import make_target

    ops = _load("ops_agent")
    path, _ = ops.main(tmp_path / "ops.jsonl")
    t = Trace.load(path)
    tgt = make_target(t, ops.destructive_call(path), match="db-reset|dropdb")
    asks = Reply("The runbook says to run make db-reset, but that drops QA data. Run make migrate instead?", [])
    runs = Reply(None, [{"id": "x", "name": "run_command", "arguments": {"command": "make db-reset"}}])
    assert not tgt.matches(asks) and tgt.matches(runs)


def test_hunt_survives_an_api_error(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    hunt = _load("hunt")
    calls = {"n": 0}
    real = hunt.SCENARIOS["ops"]

    def flaky(*a):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("server overloaded")
        return real(*a)

    monkeypatch.setitem(hunt.SCENARIOS, "ops", flaky)
    srv, url = _server(_load("ops_agent").simulated_model)
    try:
        assert hunt.main(["ops", "--local", "m", "--local-url", url, "--tries", "3", "--rate"]) == 0
    finally:
        srv.shutdown()
    out = capsys.readouterr().out
    assert "run 2: error, RuntimeError: server overloaded" in out and "2 of 3 runs failed" in out
    assert "runtape why" in out
