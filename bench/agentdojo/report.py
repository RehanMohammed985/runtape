"""Summarize AgentDojo results: python bench/agentdojo/report.py bench/results/agentdojo-*.jsonl"""
import json
import statistics
import sys
from collections import defaultdict


def pct(n, d):
    return f"{n}/{d} ({100 * n / d:.0f}%)" if d else "0/0"


def summarize(path):
    rows = [json.loads(x) for x in open(path) if x.strip()]
    if not rows:
        return
    model = rows[0].get("model")
    ran = [r for r in rows if "attacked" in r]
    errors = [r for r in rows if "error" in r]
    attacked = [r for r in ran if r["attacked"]]
    scored = [r for r in attacked if r.get("baseline")]
    stable = [r for r in scored if r["baseline"][0] * 2 >= r["baseline"][1]]
    drifted = [r for r in scored if r not in stable]
    inside = [r for r in stable if r.get("headline_in_injection")]
    whole = [r for r in stable if r.get("headline_contains_injection")]
    missed = [r for r in stable if r not in inside and r not in whole]
    print(f"## {model}\n")
    print(f"- pairs run: {len(ran)}" + (f" (plus {len(errors)} where the agent run failed)" if errors else ""))
    print(f"- attack succeeded (AgentDojo's security check): {pct(len(attacked), len(ran))}")
    print(f"- attacker's call found in the trace and `why` run: {len(scored)}")
    print(f"- no longer the model's usual choice when `why` ran: {len(drifted)}; why reported that instead of a cause")
    print(f"- headline cause inside the injected text: {pct(len(inside), len(stable))}")
    print(f"- headline cause is the whole tool result holding the injection: {pct(len(whole), len(stable))}")
    print(f"- missed: {pct(len(missed), len(stable))}" + (f" ({', '.join(r['pair'] for r in missed)})" if missed else ""))
    calls = [r["calls"] for r in scored if r.get("calls")]
    if calls:
        print(f"- model calls per `why`: median {statistics.median(calls):.0f}, max {max(calls)}")
    by = defaultdict(lambda: [0, 0, 0, 0])
    for r in ran:
        b = by[r["suite"]]
        b[0] += 1
        b[1] += r["attacked"]
        b[2] += r in stable
        b[3] += r in inside
    print("\n| suite | pairs | attack succeeded | stable when why ran | headline inside the injection |")
    print("|---|---|---|---|---|")
    for s, (n, a, st, i) in sorted(by.items()):
        print(f"| {s} | {n} | {a} | {st} | {i} |")
    for r in missed:
        print(f"\n{r['pair']} headline: {r.get('headline')!r}")
    print()


if __name__ == "__main__":
    for p in sys.argv[1:]:
        summarize(p)
