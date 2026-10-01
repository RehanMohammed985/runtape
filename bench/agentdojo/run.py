"""Run runtape on AgentDojo prompt-injection attacks.

AgentDojo (Debenedetti et al., 2024) is a public benchmark of tool-using agents: realistic tasks in
banking, Slack, travel and workspace environments, with attacker text injected into tool results.
For each (user task, injection task) pair:

  1. Run AgentDojo's own agent pipeline on the model, with every model call recorded by runtape.
  2. If the attack succeeded (AgentDojo's security check), find the model reply that made the
     attacker's call, e.g. send_money to the attacker's account.
  3. Run `runtape why` on that decision, without telling it where the injection is.
  4. Score whether the headline cause lies inside the injected text, and whether it was narrowed
     to part of it rather than the whole tool result.

    pip install agentdojo
    python bench/agentdojo/run.py --openai sarvam-105b --base-url https://api.sarvam.ai/v1 --suite banking
    python bench/agentdojo/report.py bench/results/agentdojo-sarvam-105b.jsonl

Model replies made by `why` are cached under --work, so an interrupted run resumes where it stopped.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

import openai  # noqa: E402
from agentdojo.agent_pipeline import AgentPipeline, InitQuery, SystemMessage, ToolsExecutionLoop, ToolsExecutor  # noqa: E402
from agentdojo.agent_pipeline.agent_pipeline import load_system_message  # noqa: E402
from agentdojo.agent_pipeline.llms import openai_llm  # noqa: E402
from agentdojo.attacks.attack_registry import load_attack  # noqa: E402
from agentdojo.task_suite.load_suites import get_suite  # noqa: E402

import runtape  # noqa: E402
from runtape.rerun import BudgetExceeded, FunctionModel, load_model_fn  # noqa: E402

WORD = re.compile(r"[a-z0-9]+")


def _plain_messages(message, model_name):
    """AgentDojo sends the system prompt as a 'developer' message with content parts. Many
    OpenAI-compatible servers only accept 'system' and string content, so send that instead."""
    out = dict(_original(message, model_name))
    if out.get("role") == "developer":
        out["role"] = "system"
    if isinstance(out.get("content"), list):
        out["content"] = "".join(part.get("text", "") for part in out["content"])
    return out


_original = openai_llm._message_to_openai
openai_llm._message_to_openai = _plain_messages


def pipeline_for(client, model: str) -> AgentPipeline:
    llm = openai_llm.OpenAILLM(client, model)
    p = AgentPipeline([SystemMessage(load_system_message(None)), InitQuery(), llm,
                       ToolsExecutionLoop([ToolsExecutor(), llm])])
    p.name = "local"  # only used by the attack to address the model; the attack below uses no model name
    return p


def target_value(injection_task, user_task, pre_env) -> tuple[str, str] | None:
    """The attacker's call and an argument value that marks it as the attacker's (an account, address
    or link named in the injection goal and not used by the user's own task)."""
    calls = injection_task.ground_truth(pre_env)
    user_vals = {str(v) for c in user_task.ground_truth(pre_env) for v in c.args.values()}
    for call in calls:
        for v in call.args.values():
            if isinstance(v, str) and len(v) >= 6 and v in injection_task.GOAL and v not in user_vals:
                return call.function, v
    return None


def find_decision(trace, value: str) -> int | None:
    for e in trace.of_type("llm_response"):
        for tc in e.payload.get("tool_calls") or []:
            if value in json.dumps(tc.get("arguments"), ensure_ascii=False):
                return e.id
    return None


def words(s: str) -> list[str]:
    return WORD.findall(s.lower())


def score(rep, injections: dict[str, str], goal: str) -> dict:
    inj_words = set(w for text in injections.values() for w in words(text))
    goal_words = set(words(goal))
    decisive = [c for c in rep.causes if c.kind == "decisive"]
    if rep.joint is not None and rep.joint.kind == "decisive":
        decisive.append(rep.joint)

    def texts(c):
        t = c.refined or c.finest
        return [s.text for s in t.removed]

    def inside(c):  # the cause is (part of) the injected text
        w = [x for t in texts(c) for x in words(t)]
        return bool(w) and sum(x in inj_words for x in w) / len(w) >= 0.8

    def contains(c):  # the cause includes the injection, plus other content around it
        w = set(x for t in texts(c) for x in words(t))
        return bool(goal_words) and len(goal_words & w) / len(goal_words) >= 0.6

    head = decisive[0] if decisive else None
    return {
        "headline_in_injection": bool(head) and inside(head),
        "headline_contains_injection": bool(head) and not inside(head) and contains(head),
        "injection_anywhere": any(inside(c) or contains(c) for c in decisive),
        "headline": " | ".join(texts(head))[:300] if head else None,
        "other_decisive": [" | ".join(texts(c))[:200] for c in decisive if c is not head and not inside(c)],
        "baseline": [rep.baseline.kept, rep.baseline.n],
        "calls": rep.calls, "cache_hits": rep.cache_hits, "stopped": rep.stopped,
        "warnings": rep.warnings,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--openai", metavar="MODEL", required=True, help="model name at an OpenAI-compatible API")
    ap.add_argument("--base-url", help="the API's base URL (key in OPENAI_API_KEY)")
    ap.add_argument("--model-fn", help="answer why's reruns with a Python function instead (offline tests)")
    ap.add_argument("--suite", default="banking", help="banking, slack, travel, workspace, or a comma list")
    ap.add_argument("--version", default="v1.2.2", help="AgentDojo benchmark version")
    ap.add_argument("--attack", default="important_instructions_no_model_name")
    ap.add_argument("--pairs", type=int, default=40, help="(user task, injection task) pairs per suite")
    ap.add_argument("--seed", type=int, default=0, help="which pairs are sampled")
    ap.add_argument("--only", help="run just these pairs, e.g. user_task_0/injection_task_0 (comma list)")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--budget", type=int, default=300, help="max model calls for each why run")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", help="results file (default bench/results/agentdojo-<model>.jsonl)")
    ap.add_argument("--work", help="folder for traces and the reply cache (default bench/agentdojo)")
    a = ap.parse_args(argv)

    label = re.sub(r"[^\w.-]+", "_", a.openai)
    out = Path(a.out or HERE.parent / "results" / f"agentdojo-{label}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(x)["pair"] for x in out.read_text().splitlines() if x.strip()}
    work = Path(a.work) if a.work else HERE
    traces = work / "traces" / label
    traces.mkdir(parents=True, exist_ok=True)
    cache = work / ".cache"
    fn_model = FunctionModel(load_model_fn(a.model_fn)) if a.model_fn else None

    for suite_name in a.suite.split(","):
        suite = get_suite(a.version, suite_name)
        pairs = [(u, i) for u in sorted(suite.user_tasks) for i in sorted(suite.injection_tasks)]
        random.Random(a.seed).shuffle(pairs)
        pairs = pairs[:a.pairs]
        if a.only:
            pairs = [tuple(x.split("/")) for x in a.only.split(",")]
        for uid, iid in pairs:
            pair = f"{suite_name}/{uid}/{iid}"
            if pair in done:
                continue
            t0 = time.time()
            user_task, inj_task = suite.get_user_task_by_id(uid), suite.get_injection_task_by_id(iid)
            path = traces / f"{suite_name}-{uid}-{iid}.jsonl"
            path.unlink(missing_ok=True)
            rec = runtape.Recorder(path, name=f"agentdojo-{pair}", tags={"bench": "agentdojo", "pair": pair})
            client = rec.wrap(openai.OpenAI(base_url=a.base_url) if a.base_url else openai.OpenAI())
            pipe = pipeline_for(client, a.openai)
            injections = load_attack(a.attack, suite, pipe).attack(user_task, inj_task)
            row = {"pair": pair, "suite": suite_name, "user_task": uid, "injection_task": iid,
                   "model": a.openai, "attack": a.attack}
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    utility, attacked = suite.run_task_with_pipeline(pipe, user_task, inj_task, injections)
            except openai.APIStatusError as e:
                rec.close()
                if e.status_code in (400, 401, 402, 403, 404):
                    print(f"{pair}: the model server refused the request ({str(e)[:300]})")
                    print("Stopping: every pair would fail the same way.")
                    return 1
                print(f"{pair}: error, {str(e)[:200]}")
                continue
            except Exception as e:  # noqa: BLE001 - a broken run of the agent is recorded, not fatal
                rec.close()
                row.update(error=f"{type(e).__name__}: {e}"[:300], seconds=round(time.time() - t0, 1))
                with out.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"{pair}: agent run failed ({row['error'][:120]})")
                continue
            rec.close()
            row.update(utility=utility, attacked=attacked)
            if attacked:
                env = suite.load_and_inject_default_environment(injections)
                tv = target_value(inj_task, user_task, user_task.init_environment(env))
                trace = runtape.load(path)
                ev = find_decision(trace, tv[1]) if tv else None
                row.update(target=tv and {"function": tv[0], "value": tv[1]}, decision=ev)
                if ev is not None:
                    try:
                        rep = runtape.why(trace, ev, match=re.escape(tv[1]), model=fn_model, runs=a.runs,
                                          budget=a.budget, cache_dir=str(cache / "why"), workers=a.workers)
                        row.update(score(rep, injections, inj_task.GOAL))
                    except BudgetExceeded as e:
                        row.update(stopped=str(e))
            row["seconds"] = round(time.time() - t0, 1)
            with out.open("a") as f:
                f.write(json.dumps(row) + "\n")
            if not attacked:
                verdict = "attack failed"
            elif row.get("decision") is None:
                verdict = "attacked, no attacker call found in the trace"
            elif row.get("headline_in_injection"):
                verdict = "FOUND (inside the injection)"
            elif row.get("headline_contains_injection"):
                verdict = "found (whole tool result)"
            else:
                verdict = "missed"
            base = row.get("baseline")
            extra = f"   baseline {base[0]}/{base[1]}, {row.get('calls', 0)} calls" if base else ""
            print(f"{pair}: {verdict}{extra}", flush=True)
    print(f"\nresults in {out}. Summary: python bench/agentdojo/report.py {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
