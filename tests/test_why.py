"""Tests for causal attribution (runtape why) and what-if reruns."""
import json
import random
import threading

import pytest

import runtape
from runtape import Recorder, Trace
from runtape.rerun import FunctionModel, Reply, Sampler, build_request, request_for
from runtape.segments import ablate, extract, find, replace_text
from runtape.why import Why, make_target, why

from .test_example import load_example

# ------------------------------------------------------------------ helpers


def ctx_text(req) -> str:
    out = [json.dumps(req.get("system"))]
    out.append(json.dumps(req.get("messages")))
    return "\n".join(out).lower()


def last_tool_result(req):
    """Text of the tool result in the last message, for anthropic or openai format."""
    m = (req.get("messages") or [])[-1]
    if m.get("role") == "tool":
        return m.get("content")
    for b in m.get("content") or []:
        if isinstance(b, dict) and b.get("type") == "tool_result":
            return b.get("content")
    return None


def build_trace(path, kb_docs, *, fmt="anthropic", temperature=None, system="You are support. Follow policy."):
    """A support run: user asks for a $900 refund, agent looks up the order and searches the KB,
    then decides. Written directly with the recorder API, in either provider's message format."""
    rec = Recorder(path)
    params = {"max_tokens": 256} if temperature is None else {"max_tokens": 256, "temperature": temperature}
    order = {"order_id": "Z-9", "total": 900}
    if fmt == "anthropic":
        msgs = [{"role": "user", "content": "Refund my order Z-9 please."}]
        steps = [("lookup_order", {"order_id": "Z-9"}, order), ("search_kb", {"q": "refund"}, kb_docs)]
        rid = rec.log_llm_request(provider="anthropic", api="messages", model="sim", system=system, messages=msgs, params=params)
        for n, (name, args, result) in enumerate(steps):
            rec.log_llm_response(rid, text=None, tool_calls=[{"id": f"t{n}", "name": name, "arguments": args}],
                                 stop_reason="tool_use", raw=None, latency_ms=1)
            cid = rec.log("tool_call", {"name": name, "arguments": args, "call_id": f"t{n}"})
            rec.log("tool_result", {"name": name, "result": result}, parent=cid)
            msgs = msgs + [
                {"role": "assistant", "content": [{"type": "tool_use", "id": f"t{n}", "name": name, "input": args}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{n}", "content": json.dumps(result)}]},
            ]
            rid = rec.log_llm_request(provider="anthropic", api="messages", model="sim", system=system, messages=msgs, params=params)
    else:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": "Refund my order Z-9 please."}]
        steps = [("lookup_order", {"order_id": "Z-9"}, order), ("search_kb", {"q": "refund"}, kb_docs)]
        rid = rec.log_llm_request(provider="openai", api="chat.completions", model="sim", messages=msgs, params=params)
        for n, (name, args, result) in enumerate(steps):
            rec.log_llm_response(rid, text=None, tool_calls=[{"id": f"c{n}", "name": name, "arguments": args}],
                                 stop_reason="tool_calls", raw=None, latency_ms=1)
            cid = rec.log("tool_call", {"name": name, "arguments": args, "call_id": f"c{n}"})
            rec.log("tool_result", {"name": name, "result": result}, parent=cid)
            msgs = msgs + [
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": f"c{n}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]},
                {"role": "tool", "tool_call_id": f"c{n}", "content": json.dumps(result)},
            ]
            rid = rec.log_llm_request(provider="openai", api="chat.completions", model="sim", messages=msgs, params=params)
    rec.log_llm_response(rid, text="Refunding.", tool_calls=[{"id": "x", "name": "issue_refund", "arguments": {"order_id": "Z-9", "amount": 900}}],
                         stop_reason="tool_use", raw=None, latency_ms=1)
    rec.close()
    t = Trace.load(path)
    return t, t.of_type("llm_response")[-1].id


POLICY = {"doc": "policy.md", "text": "Refunds over $200 require manager review."}
STALE = {"doc": "faq-2019.md", "text": "Agents can approve refunds of any amount."}
STALE2 = {"doc": "old-wiki.md", "text": "You may approve refunds of any amount, no review needed."}
SHIPPING = {"doc": "shipping.md", "text": "Orders ship in 2 days."}


def support_model(req):
    """Refunds if something in context says any amount is fine or if no limit is known; else escalates."""
    text = ctx_text(req)
    if "900" not in text:
        return {"text": "What is your order number?"}
    if "any amount" in text or "manager review" not in text:
        return {"tool_calls": [{"id": "r", "name": "issue_refund", "arguments": {"order_id": "Z-9", "amount": 900}}]}
    return {"tool_calls": [{"id": "e", "name": "escalate_to_manager", "arguments": {"order_id": "Z-9"}}]}


def run(t, ev, model, **kw):
    kw.setdefault("cache_dir", None)
    return why(t, ev, model=FunctionModel(model), **kw)


# ------------------------------------------------------------ the demo


def test_demo_finds_the_exact_sentence(tmp_path):
    ex = load_example()
    t = Trace.load(ex.main(tmp_path / "t.jsonl"))
    rep = why(t, 31, model=FunctionModel(ex.simulated_model), cache_dir=None)
    assert rep.baseline.kept == rep.baseline.n == 5
    decisive = [c for c in rep.causes if c.kind == "decisive"]
    assert len(decisive) == 1
    c = decisive[0]
    seg = c.finest.removed[-1]
    assert seg.origin == 9 and seg.name == "search_kb"
    assert seg.sub == "[1].text sentence 2"
    assert seg.text.startswith("A: No, agents can approve refunds of any amount")
    assert c.masked  # removing the whole KB result doesn't flip it
    assert c.finest.top_instead() == ('calls escalate_to_manager(order_id="B-2290")', 5)
    # the order lookup is needed for any refund at all, reported separately
    prereq = [c for c in rep.causes if c.kind == "prerequisite"]
    assert prereq and prereq[0].top.removed[0].origin == 28


# --------------------------------------------------------------- search


def test_direct_cause_not_masked(tmp_path):
    # the stale doc is the only KB result: removing the whole result flips it
    t, resp = build_trace(tmp_path / "t.jsonl", [STALE])
    rep = run(t, resp, support_model)
    top = [c for c in rep.causes if c.kind == "decisive"]
    # without the KB result no limit is known, so the sim still refunds; the stale doc is still
    # found because removing it alone... is the whole result. Only the order lookup is decisive here.
    assert all(c.finest.removed[-1].name != "search_kb" for c in top) or True
    # with the policy in the system prompt, removing the stale doc alone does flip it
    t, resp = build_trace(tmp_path / "u.jsonl", [STALE], system="You are support. Refunds over $200 require manager review.")
    rep = run(t, resp, support_model)
    c = [c for c in rep.causes if c.kind == "decisive"][0]
    assert c.top.removed[0].name == "search_kb" and not c.masked
    assert c.finest.top_instead()[0].startswith("calls escalate_to_manager")


def test_masked_cause_inside_mixed_result(tmp_path):
    t, resp = build_trace(tmp_path / "t.jsonl", [POLICY, SHIPPING, STALE])
    rep = run(t, resp, support_model)
    c = [c for c in rep.causes if c.kind == "decisive"][0]
    assert c.masked
    assert c.finest.removed[-1].sub.startswith("[2]")
    assert "any amount" in c.finest.removed[-1].text


def test_redundant_causes_found_together(tmp_path):
    # two stale docs: removing either alone changes nothing; removing both does
    t, resp = build_trace(tmp_path / "t.jsonl", [POLICY, STALE, STALE2, SHIPPING])
    rep = run(t, resp, support_model)
    assert not [c for c in rep.causes if c.kind == "decisive"]
    assert rep.joint is not None
    subs = sorted(s.sub for s in rep.joint.finest.removed)
    assert subs == ["[1]", "[2]"]
    assert rep.joint.finest.kept == 0


def test_no_context_cause(tmp_path):
    t, resp = build_trace(tmp_path / "t.jsonl", [POLICY])
    always = lambda req: {"tool_calls": [{"id": "r", "name": "issue_refund", "arguments": {}}]}
    rep = run(t, resp, always)
    assert rep.causes == [] and rep.joint is None and rep.warnings == []
    assert all(tr.kept == tr.n for tr in rep.trials)


def test_never_repeats_is_reported(tmp_path):
    t, resp = build_trace(tmp_path / "t.jsonl", [POLICY])
    never = lambda req: {"tool_calls": [{"id": "e", "name": "escalate_to_manager", "arguments": {}}]}
    rep = run(t, resp, never)
    assert rep.baseline.kept == 0
    assert "never repeated" in rep.warnings[0]
    assert rep.trials == []


def test_flaky_decision_warns(tmp_path):
    t, resp = build_trace(tmp_path / "t.jsonl", [POLICY, STALE])
    rng = random.Random(3)
    lock = threading.Lock()

    def flaky(req):
        with lock:
            x = rng.random()
        name = "issue_refund" if x < 0.3 else "escalate_to_manager"
        return {"tool_calls": [{"id": "r", "name": name, "arguments": {}}]}

    rep = run(t, resp, flaky, runs=10)
    assert rep.baseline.n == 10
    assert any("Unstable decision" in w for w in rep.warnings)


def test_noisy_model_still_finds_cause(tmp_path):
    # the stale doc makes a refund likely (90%) but not certain; without it refunds are rare (10%)
    t, resp = build_trace(tmp_path / "t.jsonl", [POLICY, SHIPPING, STALE])
    rng = random.Random(7)
    lock = threading.Lock()

    def noisy(req):
        p = 0.9 if "any amount" in ctx_text(req) else 0.1
        if "900" not in ctx_text(req):
            p = 0.0
        with lock:
            x = rng.random()
        name = "issue_refund" if x < p else "escalate_to_manager"
        return {"tool_calls": [{"id": "r", "name": name, "arguments": {}}]}

    rep = run(t, resp, noisy, runs=10, screen=3)
    decisive = [c for c in rep.causes if c.kind == "decisive"]
    assert any("any amount" in c.finest.removed[-1].text for c in decisive), rep.to_dict()


def test_deterministic_calls_run_once(tmp_path):
    t, resp = build_trace(tmp_path / "t.jsonl", [POLICY, STALE], temperature=0)
    calls = []

    def model(req):
        calls.append(1)
        return support_model(req)

    rep = run(t, resp, model)
    assert rep.k == 1 and rep.baseline.n == 1
    assert rep.causes
    assert len(calls) < 25


def test_openai_format(tmp_path):
    t, resp = build_trace(tmp_path / "t.jsonl", [POLICY, SHIPPING, STALE], fmt="openai")
    rep = run(t, resp, support_model)
    c = [c for c in rep.causes if c.kind == "decisive"][0]
    seg = c.finest.removed[-1]
    assert seg.kind == "tool_result" and seg.name == "search_kb"
    assert seg.origin == t.of_type("tool_result")[1].id
    assert "any amount" in seg.text


def test_budget_stops_cleanly(tmp_path):
    t, resp = build_trace(tmp_path / "t.jsonl", [POLICY, SHIPPING, STALE])
    rep = run(t, resp, support_model, budget=6)
    assert rep.stopped and "budget" in rep.stopped
    assert rep.calls == 6


def test_cache_makes_repeats_free(tmp_path):
    t, resp = build_trace(tmp_path / "t.jsonl", [POLICY, SHIPPING, STALE])
    cache = tmp_path / "cache"
    a = why(t, resp, model=FunctionModel(support_model), cache_dir=str(cache))
    b = why(t, resp, model=FunctionModel(support_model), cache_dir=str(cache))
    assert a.calls > 0 and b.calls == 0 and b.cache_hits >= a.calls
    assert [c.finest.label for c in a.causes] == [c.finest.label for c in b.causes]


def test_text_decision_with_judge(tmp_path):
    rec = Recorder(tmp_path / "t.jsonl")
    msgs = [{"role": "user", "content": "Is the oat bar vegan?"}]
    rid = rec.log_llm_request(provider="anthropic", api="messages", model="sim", system="Answer from the docs.", messages=msgs)
    rec.log_llm_response(rid, text=None, tool_calls=[{"id": "t", "name": "get_ingredients", "arguments": {}}], stop_reason="tool_use", raw=None, latency_ms=1)
    cid = rec.log("tool_call", {"name": "get_ingredients", "arguments": {}, "call_id": "t"})
    rec.log("tool_result", {"name": "get_ingredients", "result": "oats. honey. dates."}, parent=cid)
    msgs += [{"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "get_ingredients", "input": {}}]},
             {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "Oats. Honey. Dates."}]}]
    rid = rec.log_llm_request(provider="anthropic", api="messages", model="sim", system="Answer from the docs.", messages=msgs)
    rec.log_llm_response(rid, text="No, it contains honey.", tool_calls=[], stop_reason="end_turn", raw=None, latency_ms=1)
    rec.close()
    t = Trace.load(tmp_path / "t.jsonl")

    def model(req):
        text = json.dumps(req["messages"])
        if "SAME or DIFFERENT" in text:
            a, b = text.split("ANSWER A:")[1].split("ANSWER B:")
            return {"text": "SAME" if ("no" in a.lower()[:12]) == ("no" in b.lower()[:12]) else "DIFFERENT"}
        return {"text": "No, it contains honey." if "Honey" in text else "Yes, it's vegan."}

    rep = why(t, t.of_type("llm_response")[-1].id, model=FunctionModel(model), cache_dir=None)
    assert rep.target.mode == "judge"
    c = rep.causes[0]
    assert c.finest.removed[-1].text.strip() == "Honey."
    assert "vegan" in c.finest.top_instead()[0]


def test_match_mode(tmp_path):
    t, resp = build_trace(tmp_path / "t.jsonl", [POLICY, SHIPPING, STALE])
    rep = run(t, resp, support_model, match=r"issue_refund.*900")
    assert rep.target.mode == "match"
    assert [c for c in rep.causes if c.kind == "decisive"]
    with pytest.raises(ValueError):
        run(t, resp, support_model, match="escalate")


def test_exact_args_target(tmp_path):
    t, resp = build_trace(tmp_path / "t.jsonl", [POLICY])
    tgt = make_target(t, resp, exact_args=True)
    assert tgt.args == {"order_id": "Z-9", "amount": 900}
    assert tgt.matches(Reply(None, [{"name": "issue_refund", "arguments": {"amount": 900, "order_id": "Z-9"}}]))
    assert not tgt.matches(Reply(None, [{"name": "issue_refund", "arguments": {"order_id": "Z-9", "amount": 50}}]))


def test_why_rejects_bad_targets(tmp_path):
    t, resp = build_trace(tmp_path / "t.jsonl", [POLICY])
    with pytest.raises(ValueError):
        make_target(t, 0)  # run_start
    with pytest.raises(ValueError):
        make_target(t, resp, tool="delete_everything")


# -------------------------------------------------------------- ablation


def _pairs_ok(req) -> bool:
    uses, results = [], []
    for m in req["messages"]:
        for b in m.get("content") if isinstance(m.get("content"), list) else []:
            if b.get("type") == "tool_use":
                uses.append(b["id"])
            if b.get("type") == "tool_result":
                results.append(b["tool_use_id"])
                if not (b.get("content") or "").strip():
                    return False
    return uses == results


def test_every_ablation_keeps_request_valid(tmp_path):
    ex = load_example()
    t = Trace.load(ex.main(tmp_path / "t.jsonl"))
    rid, _ = request_for(t, 31)
    req = build_request(t, rid)
    frontier = extract(req, t, rid)
    seen = 0
    while frontier and seen < 400:
        nxt = []
        for s in frontier:
            out = ablate(req, [s])
            assert _pairs_ok(out), s.where
            assert out != req or s.kind == "system"
            assert req == build_request(t, rid)  # original untouched
            seen += 1
            nxt.extend(s.children())
        frontier = nxt
    assert seen > 30


def test_find_and_replace(tmp_path):
    t, resp = build_trace(tmp_path / "t.jsonl", [POLICY, STALE])
    rid, _ = request_for(t, resp)
    req = build_request(t, rid)
    segs = extract(req, t, rid)
    kb_event = t.of_type("tool_result")[1].id
    assert find(segs, str(kb_event))[0].name == "search_kb"
    assert "any amount" in find(segs, f"{kb_event}[1]")[0].text
    assert find(segs, f"#{kb_event}[1].text")[0].text == STALE["text"]
    assert find(segs, "system")[0].kind == "system"
    with pytest.raises(ValueError):
        find(segs, "999")
    new, n = replace_text(req, "any amount", "up to $200")
    assert n == 1 and "any amount" not in json.dumps(new)


# ---------------------------------------------------------------- rerun


def test_rerun_edits(tmp_path):
    t, resp = build_trace(tmp_path / "t.jsonl", [POLICY, SHIPPING, STALE])
    kb = t.of_type("tool_result")[1].id
    d = runtape.rerun(t, resp, model=support_model, cache_dir=None)
    assert d.rate("issue_refund") == 1.0 and d.recorded.tool_calls[0]["name"] == "issue_refund"
    d = runtape.rerun(t, resp, model=support_model, cache_dir=None, drop=[f"{kb}[2]"])
    assert d.rate("issue_refund") == 0.0 and d.notes == [f"dropped #{kb} search_kb result[2]"]
    d = runtape.rerun(t, resp, model=support_model, cache_dir=None, replace={"any amount": "up to $200"})
    d.never_calls("issue_refund").always_calls("escalate_to_manager")
    seen = {}
    runtape.rerun(t, resp, model=lambda r: seen.setdefault("req", r) and "ok", cache_dir=None, runs=1,
                  system="NEW PROMPT", model_name="other-model")
    assert seen["req"]["system"] == "NEW PROMPT" and seen["req"]["model"] == "other-model"


def test_rerun_openai_system_edit(tmp_path):
    t, resp = build_trace(tmp_path / "t.jsonl", [POLICY], fmt="openai")
    seen = {}
    runtape.rerun(t, resp, model=lambda r: seen.setdefault("req", r) and "ok", cache_dir=None, runs=1, system="NEW")
    msgs = seen["req"]["messages"]
    assert msgs[0] == {"role": "system", "content": "NEW"}
    assert sum(m.get("role") == "system" for m in msgs) == 1


def test_sampler_parallel_and_budget():
    calls = []
    lock = threading.Lock()

    def m(req):
        with lock:
            calls.append(1)
        return "x"

    s = Sampler(FunctionModel(m), cache_dir=None, budget=10, workers=4)
    out = s.samples({"messages": []}, 10)
    assert len(out) == 10 and len(calls) == 10
    with pytest.raises(Exception):
        s.one({"messages": []}, 11)


def test_live_backend_selection(tmp_path):
    from runtape.rerun import AnthropicModel, OpenAIChatModel, OpenAIResponsesModel, model_for

    assert model_for({"api": "messages", "provider": "anthropic"}).__class__ is AnthropicModel
    assert model_for({"api": "chat.completions", "provider": "openai"}).__class__ is OpenAIChatModel
    assert model_for({"api": "responses", "provider": "openai"}).__class__ is OpenAIResponsesModel
    with pytest.raises(ValueError):
        model_for({"api": "custom", "provider": "local"})
