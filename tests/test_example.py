"""The refund-bot example is the launch demo, so it's tested like a feature."""
import importlib.util
from pathlib import Path

from runtape import Trace

EX = Path(__file__).resolve().parents[1] / "examples" / "refund_bot.py"


def load_example():
    spec = importlib.util.spec_from_file_location("refund_bot", EX)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_refund_bot_trace_explains_the_bug(tpath):
    load_example().main(tpath)
    t = Trace.load(tpath)
    assert t.status == "ok"

    # the bad action is in the trace
    refunds = [e for e in t.of_type("tool_call") if e.payload["name"] == "issue_refund"]
    assert refunds[-1].payload["arguments"]["amount"] == 2400.0

    # grep finds where the poison entered: the KB search result, well before the refund
    hits = t.grep("any amount")
    first = hits[0].event
    assert first.type == "tool_result" and first.payload["name"] == "search_kb"
    assert refunds[-1].id - first.id > 15

    # and the context at the bad decision still contains it
    decision = t[refunds[-1].meta["requested_by"]]
    ctx = t.context(decision.id)
    assert "any amount" in str(ctx.messages)
    assert ctx.system.startswith("You are the support agent")
