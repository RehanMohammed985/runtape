"""Cheaper why: early-stopping confirmation, narrowing that stops at the first confirmed part, several
samples per API request, and prompt caching. The results must not change, only the cost."""
import importlib.util
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import runtape
from runtape.rerun import BudgetExceeded, FunctionModel, OpenAIChatModel, Reply, Sampler
from runtape.cli import resolve_event

EX = Path(__file__).resolve().parents[1] / "examples"


def _ex(name):
    spec = importlib.util.spec_from_file_location(name, EX / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(EX))
    spec.loader.exec_module(mod)
    return mod


class Batchy:
    """A backend with an API-level `n`. honors_n=False behaves like a server that ignores it."""

    def __init__(self, honors_n=True, fail_first=False):
        self.honors_n, self.fail_first = honors_n, fail_first
        self.requests = []
        self.lock = threading.Lock()

    def _reply(self, req, k):
        return Reply(f"{req['messages'][-1]['content']}#{k}", [], "stop", None)

    def __call__(self, req):
        with self.lock:
            self.requests.append((req["messages"][-1]["content"], 1))
        return self._reply(req, 0)

    def batch(self, req, n):
        with self.lock:
            if self.fail_first:
                self.fail_first = False
                raise ValueError("n is not supported")
            self.requests.append((req["messages"][-1]["content"], n))
        return [self._reply(req, k) for k in range(n if self.honors_n else 1)]


def _req(text):
    return {"api": "chat.completions", "model": "m", "messages": [{"role": "user", "content": text}], "params": {}}


def test_samples_of_one_context_share_a_request(tmp_path):
    model = Batchy()
    s = Sampler(model, cache_dir=tmp_path, budget=None, workers=4)
    jobs = [(_req("a"), i) for i in range(10)] + [(_req("b"), i) for i in range(3)]
    out = s.many(jobs)
    assert len(out) == 13 and s.calls == 13 and s.requests == 2
    assert sorted(model.requests) == [("a", 10), ("b", 3)]
    # each sample is cached under its own index: asking again is free, and a longer run only fetches the rest
    s2 = Sampler(model, cache_dir=tmp_path, budget=None)
    assert [r.text for r in s2.many(jobs)] == [r.text for r in out] and s2.calls == 0
    s2.many([(_req("a"), i) for i in range(15)])
    assert s2.calls == 5 and model.requests[-1] == ("a", 5)


def test_server_that_ignores_n_falls_back_to_single_requests(tmp_path):
    model = Batchy(honors_n=False)
    s = Sampler(model, cache_dir=None, budget=None, workers=1)
    out = s.many([(_req("a"), i) for i in range(5)])
    assert len(out) == 5 and s.calls == 5 and s.n_ok is False
    assert model.requests[0] == ("a", 5) and all(n == 1 for _, n in model.requests[1:])


def test_server_that_rejects_n_falls_back(tmp_path):
    model = Batchy(fail_first=True)
    s = Sampler(model, cache_dir=None, budget=None, workers=1)
    assert len(s.many([(_req("a"), i) for i in range(4)])) == 4
    assert s.n_ok is False and s.calls == 4 and s.requests == 4


def test_budget_counts_samples_and_keeps_what_arrived(tmp_path):
    s = Sampler(Batchy(), cache_dir=None, budget=7, workers=1)
    with pytest.raises(BudgetExceeded) as e:
        s.many([(_req("a"), i) for i in range(5)] + [(_req("b"), i) for i in range(5)])
    assert s.calls == 7 and [r.text for r in e.value.partial] == ["a#0", "a#1", "a#2", "a#3", "a#4", "b#0", "b#1"]


def test_prompt_cached_backend_sends_one_sample_before_the_repeats(tmp_path):
    order = []
    lock = threading.Lock()

    class Cached:
        prefix_cache = True

        def __call__(self, req):
            with lock:
                order.append(req["messages"][-1]["content"])
            return Reply("x", [], "stop", None)

    s = Sampler(Cached(), cache_dir=None, budget=None, workers=8)
    s.many([(_req(c), i) for c in "abc" for i in range(4)])
    assert sorted(order[:3]) == ["a", "b", "c"]  # one of each context first, then the 9 repeats
    assert len(order) == 12


def _openai_fake(fn):
    """A fake OpenAI client answering chat.completions.create(n=...) with a model function."""
    seen = []

    class Completions:
        def create(self, model, messages, n=1, **kw):
            seen.append(n)
            choices = []
            for k in range(n):
                r = Reply.from_any(fn({"messages": messages, "tools": kw.get("tools"), "params": {}}))
                msg = {"role": "assistant", "content": r.text,
                       "tool_calls": [{"id": f"c{k}", "type": "function",
                                       "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])}}
                                      for tc in r.tool_calls] or None}
                choices.append({"index": k, "message": msg, "finish_reason": "tool_calls" if r.tool_calls else "stop"})
            return {"choices": choices, "usage": {"prompt_tokens": 1, "completion_tokens": 1}, "model": model}

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    return OpenAIChatModel(client), seen


def test_why_gives_the_same_answer_with_batched_samples(tmp_path):
    inbox = _ex("inbox_agent")
    path, _ = inbox.main(tmp_path / "inbox.jsonl")
    t = runtape.load(path)
    ev = resolve_event(t, "tool:forward_email")
    plain = runtape.why(t, ev, model=FunctionModel(inbox.simulated_model), cache_dir=None)
    model, seen = _openai_fake(inbox.simulated_model)
    batched = runtape.why(t, ev, model=model, cache_dir=None)
    assert batched.calls == plain.calls
    assert batched.requests < plain.calls / 2 and max(seen) > 1
    head = lambda r: [s.text for s in (r.causes[0].refined or r.causes[0].finest).removed]  # noqa: E731
    assert head(batched) == head(plain)


def test_early_stop_still_reports_full_evidence_for_the_headline(tmp_path):
    inbox = _ex("inbox_agent")
    path, _ = inbox.main(tmp_path / "inbox.jsonl")
    t = runtape.load(path)
    rep = runtape.why(t, resolve_event(t, "tool:forward_email"), model=FunctionModel(inbox.simulated_model),
                      cache_dir=None)
    top = rep.causes[0].finest
    assert top.n == 10 and top.kept == 0 and top.p < 1e-4
    assert rep.calls < 190  # was 206 before narrowing stopped early
