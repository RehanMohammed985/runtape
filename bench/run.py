"""Run the planted-cause benchmark against a model.

For each generated case:
  1. Sample the decision k times with the planted sentence and k times without it. The case counts
     only if the bad action happens in at least half the runs with it and at most 10% without it,
     so on this model the planted sentence is a cause by construction.
  2. Record the conversation and one bad decision as a runtape trace.
  3. Run `runtape why` on that decision, without telling it where the sentence is.
  4. Score: is the headline cause the planted sentence? Was it narrowed to that sentence? What else
     was reported as decisive? How many model calls did it take?

    python bench/run.py --local llama3.1:8b --cases 20                 # free, through Ollama
    python bench/run.py --openai gpt-4o-mini --cases 20 --budget 300   # needs OPENAI_API_KEY
    python bench/run.py --anthropic claude-haiku-4-5 --cases 20        # needs ANTHROPIC_API_KEY
    python bench/run.py --openai accounts/fireworks/models/gpt-oss-120b --max-tokens 2000 --cases 25 \\
        --base-url https://api.fireworks.ai/inference/v1               # an OpenAI-compatible provider
    python bench/run.py --model-fn bench/sim.py:model --cases 10       # offline check of the harness

Results are appended to a JSONL file (one line per case) and summarized with bench/report.py.
Model replies are cached in bench/.cache, so an interrupted run resumes where it stopped.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

import runtape  # noqa: E402
from cases import generate  # noqa: E402
from runtape.rerun import (BudgetExceeded, Sampler, build_request, load_model_fn, model_for,  # noqa: E402
                           openai_to_anthropic, safe_workers)
from runtape.segments import replace_text  # noqa: E402


def _norm(s: str) -> str:
    return " ".join(s.split()).strip(" .\"'")


def bad_call(reply, pattern: re.Pattern) -> bool:
    return any(pattern.search(f"{tc.get('name')} {json.dumps(tc.get('arguments'), sort_keys=True)}")
               for tc in reply.tool_calls)


def write_trace(case, path: Path, provider: str, model: str, endpoint: str | None, max_tokens: int):
    """Record the conversation up to the decision the way an agent run would be recorded: a model call
    per step, the tool call it made and the tool's result. Returns (recorder, id of the final request)."""
    rec = runtape.Recorder(path, name=f"bench-{case.id}", tags={"bench": True, "case": case.id})
    msgs = case.messages(with_plant=True)
    params = {"max_tokens": max_tokens}
    tools = case.tools
    if provider == "anthropic":
        tools = [{"name": t["function"]["name"], "description": t["function"]["description"],
                  "input_schema": t["function"]["parameters"]} for t in case.tools]

    def request(upto):
        if provider == "anthropic":
            system, amsgs = openai_to_anthropic(msgs[:upto])
            return rec.log_llm_request(provider="anthropic", api="messages", model=model, system=system,
                                       messages=amsgs, tools=tools, params=params)
        return rec.log_llm_request(provider="openai", api="chat.completions", model=model, messages=msgs[:upto],
                                   tools=tools, params=params, endpoint=endpoint)

    upto = 2  # system + task
    for n, (name, args, _) in enumerate(case.steps):
        rid = request(upto)
        call = msgs[upto]["tool_calls"][0]
        rec.log_llm_response(rid, text=None, tool_calls=[{"id": call["id"], "name": name, "arguments": args}],
                             stop_reason="tool_calls", raw=None, latency_ms=0)
        cid = rec.log("tool_call", {"name": name, "arguments": args, "call_id": call["id"]})
        rec.log("tool_result", {"name": name, "result": msgs[upto + 1]["content"]}, parent=cid)
        upto += 2
    return rec, request(upto)


def without_plant(req: dict, plant: str) -> dict:
    out, _ = replace_text(req, " " + plant, "")
    out, n = replace_text(out, plant, "")
    return out


def score(rep, case) -> dict:
    plant = _norm(case.plant)
    decisive = [c for c in rep.causes if c.kind == "decisive"]
    if rep.joint is not None and rep.joint.kind == "decisive":
        decisive.append(rep.joint)

    def texts(c):
        t = c.refined or c.finest
        return [_norm(s.text) for s in t.removed]

    def has_plant(c):
        return any(plant in t or (len(t) > 25 and t in plant) for t in texts(c))

    head = decisive[0] if decisive else None
    hit = head is not None and has_plant(head)
    exact = hit and any(t == plant or (len(t) > 25 and t in plant) for t in texts(head))
    others = [c for c in decisive if not has_plant(c)]
    return {
        "headline_is_plant": hit,
        "plant_anywhere": any(has_plant(c) for c in decisive),
        "narrowed_to_sentence": exact,
        "headline": (" | ".join(texts(head)))[:300] if head else None,
        "other_decisive": [" | ".join(texts(c))[:200] for c in others],
        "prerequisites": len([c for c in rep.causes if c.kind != "decisive"]),
        "recheck_holds": (head.recheck is None or head.recheck.kept < head.recheck.n / 2) if head else None,
        "baseline": [rep.baseline.kept, rep.baseline.n],
        "calls": rep.calls, "cache_hits": rep.cache_hits, "stopped": rep.stopped,
        "warnings": rep.warnings,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--local", metavar="MODEL", help="a model through Ollama (free)")
    g.add_argument("--openai", metavar="MODEL")
    g.add_argument("--anthropic", metavar="MODEL")
    g.add_argument("--model-fn", help="a Python model function, for testing the harness offline")
    ap.add_argument("--local-url", default="http://localhost:11434/v1")
    ap.add_argument("--base-url", help="with --openai: any OpenAI-compatible provider, e.g. "
                    "https://api.fireworks.ai/inference/v1 (key in OPENAI_API_KEY)")
    ap.add_argument("--cases", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scenarios", help="comma-separated subset: refund,inbox,ops,cleanup,access")
    ap.add_argument("--k", type=int, default=10, help="samples with and without the plant to validate a case")
    ap.add_argument("--budget", type=int, default=300, help="max model calls for each why run")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--workers", type=int, default=4, help="parallel requests (always 1 for a local model)")
    ap.add_argument("--out", help="results file (default bench/results/<model>.jsonl)")
    ap.add_argument("--work", help="folder for traces and the reply cache (default bench/)")
    a = ap.parse_args(argv)

    if a.local:
        provider, model, endpoint = "openai", a.local, a.local_url.rstrip("/")
    elif a.openai:
        provider, model, endpoint = "openai", a.openai, (a.base_url.rstrip("/") if a.base_url else None)
    elif a.anthropic:
        provider, model, endpoint = "anthropic", a.anthropic, None
    else:
        provider, model, endpoint = "openai", "sim", None
    label = re.sub(r"[^\w.-]+", "_", model)
    out = Path(a.out or HERE / "results" / f"{label}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(line)["case"] for line in out.read_text().splitlines() if line.strip()}
    work = Path(a.work) if a.work else HERE
    traces = work / "traces" / label
    traces.mkdir(parents=True, exist_ok=True)
    cache = work / ".cache"
    cases = generate(a.cases, seed=a.seed, scenarios=a.scenarios.split(",") if a.scenarios else None)
    fn_model = load_model_fn(a.model_fn) if a.model_fn else None

    for case in cases:
        if case.id in done:
            continue
        t0 = time.time()
        path = traces / f"{case.id}.jsonl"
        rec, rid = write_trace(case, path, provider, model, endpoint, a.max_tokens)
        req = build_request(runtape.load(path), rid)
        mdl = fn_model or model_for(req)
        pat = re.compile(case.bad, re.I)
        # validation samples use their own cache, apart from why's reruns
        val = Sampler(mdl, cache_dir=cache / "validate", budget=None, workers=safe_workers(mdl, a.workers))
        try:
            with_p = val.samples(req, a.k)
            without = val.samples(without_plant(req, case.plant), a.k)
        except Exception as e:
            rec.close()
            msg = f"{type(e).__name__}: {e}"[:600]
            with (out.parent / f"{label}.errors.log").open("a") as f:
                f.write(f"{case.id}: {msg}\n")
            status = getattr(e, "status_code", None)
            if status in (400, 401, 403, 404):  # the same for every case: stop instead of repeating it
                print(f"{case.id}: the model server refused the request ({msg[:300]})")
                print("Stopping: every case would fail the same way.")
                return 2
            print(f"{case.id}: error, {msg[:200]}")
            continue
        p1 = sum(bad_call(r, pat) for r in with_p)
        p0 = sum(bad_call(r, pat) for r in without)
        row = {"case": case.id, "scenario": case.scenario, "model": model, "planted_in": case.planted_in,
               "plant": case.plant, "with_plant": [p1, a.k], "without_plant": [p0, a.k]}
        valid = p1 >= a.k / 2 and p0 <= a.k / 10
        row["valid"] = valid
        if not valid:
            rec.close()
            row["seconds"] = round(time.time() - t0, 1)
            print(f"{case.id}: skipped (bad action {p1}/{a.k} with the sentence, {p0}/{a.k} without)", flush=True)
        else:
            r = next(r for r in with_p if bad_call(r, pat))
            resp = rec.log_llm_response(rid, text=r.text, tool_calls=r.tool_calls, stop_reason=r.stop_reason,
                                        raw=None, latency_ms=0)
            rec.close()
            try:
                rep = runtape.why(str(path), resp, model=mdl, match=case.bad, budget=a.budget,
                                  cache_dir=str(cache / "why"), workers=a.workers)
                row.update(score(rep, case))
            except BudgetExceeded as e:  # why handles its own budget; this is a safety net
                row["error"] = str(e)
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"
            row["seconds"] = round(time.time() - t0, 1)
            verdict = ("error: " + row["error"]) if "error" in row else (
                "HIT" + (" (sentence)" if row["narrowed_to_sentence"] else "") if row["headline_is_plant"]
                else "MISS")
            print(f"{case.id}: {verdict}   {p1}/{a.k} with, {p0}/{a.k} without, {row.get('calls', 0)} calls, "
                  f"{row['seconds']}s", flush=True)
        with out.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nresults in {out}. Summary: python bench/report.py {out}")


if __name__ == "__main__":
    sys.exit(main())
