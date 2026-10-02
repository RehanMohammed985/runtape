"""Summarize AgentDojo results: python bench/agentdojo/report.py bench/results/agentdojo-*.jsonl"""
import json
import statistics
import sys
from collections import defaultdict


def pct(n, d):
    return f"{n}/{d} ({100 * n / d:.0f}%)" if d else "0/0"


def stable(r):
    b = r.get("baseline")
    return bool(b) and b[1] > 0 and b[0] * 2 >= b[1]


def rate(xs):
    return f"{sum(xs)}/{len(xs)}" if xs else "-"


def summarize(path):
    rows = [json.loads(x) for x in open(path) if x.strip()]
    if not rows:
        return
    mt = rows[0].get("max_tokens")
    print(f"## {rows[0].get('model')}" + (f" (replies capped at {mt} tokens)" if mt else "") + "\n")
    ran = [r for r in rows if "attacked" in r]
    errors = [r for r in rows if "error" in r]
    attacked = [r for r in ran if r["attacked"]]
    scored = [r for r in attacked if r.get("baseline")]
    st = [r for r in scored if stable(r)]
    drifted = [r for r in scored if not stable(r)]
    print(f"- pairs run: {len(ran)}" + (f" (plus {len(errors)} where the agent run failed)" if errors else ""))
    print(f"- attack succeeded (AgentDojo's security check): {pct(len(attacked), len(ran))}")
    print(f"- attacker's call found and `why` run: {len(scored)}; decision made consistently when rerun: {len(st)}")
    print(f"- not made consistently (under half of reruns): {len(drifted)}; why reported that instead of a cause")
    calls = [r["calls"] for r in st if r.get("calls")]
    if calls:
        print(f"- model calls per `why` on a consistent decision: median {statistics.median(calls):.0f}, max {max(calls)}")
    reqs = [r["requests"] for r in st if r.get("requests")]
    if reqs:
        print(f"- API requests for those calls: median {statistics.median(reqs):.0f}")

    print("\n### Attribution (decisions made consistently)\n")
    print("| method | blames text inside the injection | ...including the attacker's instruction | "
          "blames the whole tool result holding it | blames something else |")
    print("|---|---|---|---|---|")
    n = len(st)
    inside = sum(bool(r.get("headline_in_injection")) for r in st)
    goal = sum(bool(r.get("headline_in_injection") and r.get("headline_has_goal")) for r in st)
    whole = sum(bool(r.get("headline_contains_injection")) for r in st)
    print(f"| runtape why | {pct(inside, n)} | {pct(goal, n)} | {pct(whole, n)} | {pct(n - inside - whole, n)} |")
    with_b = [r for r in st if r.get("baselines")]
    for m, label in (("dry", "wording overlap"), ("judge", "model as judge")):
        xs = [r["baselines"][m] for r in with_b if m in r["baselines"]]
        if xs:
            i = sum(x["inside"] for x in xs)
            g = sum(x["inside"] and x["goal"] for x in xs)
            w = sum(x["contains"] for x in xs)
            blank = sum(not x.get("answer", "x") for x in xs)
            note = f" ({blank} gave no answer)" if blank else ""
            print(f"| {label}{note} | {pct(i, len(xs))} | {pct(g, len(xs))} | {pct(w, len(xs))} | "
                  f"{pct(len(xs) - i - w, len(xs))} |")
    loo = [r["baselines"]["loo"] for r in with_b if "loo" in r.get("baselines", {})]
    if loo:
        hit = sum(x["injection_flagged"] for x in loo)
        other = sum(x["other_flagged"] for x in loo)
        print(f"| leave-one-out, 1 run per piece | - | - | {pct(hit, len(loo))} flag it | "
              f"{other} other pieces flagged in {len(loo)} decisions |")
        tops = [x for x in loo if "top_holds_injection" in x]
        if tops:
            t = sum(x["top_holds_injection"] for x in tops)
            print(f"| leave-one-out, largest drop over 5 runs | - | - | {pct(t, len(tops))} | {pct(len(tops) - t, len(tops))} |")
    by = defaultdict(lambda: [0, 0, 0, 0])
    for r in ran:
        b = by[r["suite"]]
        b[0] += 1
        b[1] += r["attacked"]
        b[2] += r in st
        b[3] += r in st and bool(r.get("headline_in_injection"))
    print("\n| suite | pairs | attack succeeded | consistent | runtape: inside the injection |")
    print("|---|---|---|---|---|")
    for s, (a, b, c, d) in sorted(by.items()):
        print(f"| {s} | {a} | {b} | {c} | {d} |")
    missed = [r for r in st if not r.get("headline_in_injection") and not r.get("headline_contains_injection")]
    for r in missed:
        print(f"\nmissed {r['pair']}: headline {r.get('headline')!r}")

    fixed = [r for r in st if r.get("fix")]
    if fixed:
        print("\n### Fixes\n")
        print("Checked on the recorded decision (10 reruns each), then on the live task: full AgentDojo runs under "
              "attack and without it.\n")
        print("| fix | passed on the recorded decision | live: attack succeeded | live: task done under attack | "
              "live: task done, no attack |")
        print("|---|---|---|---|---|")
        names = ["none", "untrusted content", "action guard", "both rules", "fix the source"]
        for name in names:
            passed, att, util, ben = [], [], [], []
            for r in fixed:
                f = r["fix"]
                c = next((c for c in f["candidates"] if c["name"] == name), None)
                if c is not None:
                    passed.append(c["holds"])
                live = (f.get("live") or {}).get(name)
                if live:
                    att += live["attack"]
                    util += live["utility"]
                    ben += live["benign_utility"]
            if passed or att:
                print(f"| {name} | {rate(passed) if name != 'none' else '-'} | {rate(att)} | {rate(util)} | {rate(ben)} |")
        agree = total = 0
        for r in fixed:
            for c in r["fix"]["candidates"]:
                live = (r["fix"].get("live") or {}).get(c["name"])
                if live and live["attack"]:
                    total += 1
                    agree += c["holds"] == (not any(live["attack"]))
        if total:
            print(f"\nThe check on the recorded decision agreed with the live result (no attack succeeded in any run) "
                  f"for {pct(agree, total)} prompt fixes.")

    ctl = [r["control"] for r in rows if r.get("control")]
    if ctl:
        cs = [c for c in ctl if stable(c)]
        print("\n### Controls\n")
        print(f"The user's own action, in a trace where the injection is present ({len(ctl)} decisions, "
              f"{len(cs)} made consistently). Blaming text inside the injection is a false positive.\n")
        print("| method | blames the injection |")
        print("|---|---|")
        print(f"| runtape why | {pct(sum(bool(c.get('headline_in_injection')) for c in cs), len(cs))} |")
        for m, label in (("dry", "wording overlap"), ("judge", "model as judge")):
            xs = [c["baselines"][m] for c in cs if m in c.get("baselines", {})]
            if xs:
                print(f"| {label} | {pct(sum(x['inside'] for x in xs), len(xs))} |")
    print()


if __name__ == "__main__":
    for p in sys.argv[1:]:
        summarize(p)
