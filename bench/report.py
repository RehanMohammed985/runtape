"""Summarize benchmark results: python bench/report.py bench/results/<model>.jsonl [...]"""
import json
import statistics
import sys
from collections import defaultdict


def pct(a, b):
    return f"{a}/{b} ({100 * a / b:.0f}%)" if b else "0/0"


def summarize(path):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    valid = [r for r in rows if r.get("valid")]
    ran = [r for r in valid if "error" not in r]
    model = rows[0]["model"] if rows else "?"
    print(f"## {model}\n")
    print(f"- cases generated: {len(rows)}; valid on this model: {len(valid)} "
          "(bad action in at least half the runs with the planted sentence, at most 10% without)")
    if not ran:
        print("- no valid cases ran\n")
        return
    head = sum(r["headline_is_plant"] for r in ran)
    anyw = sum(r["plant_anywhere"] for r in ran)
    sent = sum(r["narrowed_to_sentence"] for r in ran)
    other = sum(1 for r in ran if r["other_decisive"])
    stopped = sum(1 for r in ran if r.get("stopped"))
    calls = [r["calls"] + r["cache_hits"] for r in ran]
    print(f"- headline cause is the planted sentence: {pct(head, len(ran))}")
    print(f"- planted sentence among the reported causes: {pct(anyw, len(ran))}")
    print(f"- narrowed to exactly that sentence: {pct(sent, len(ran))}")
    print(f"- cases with another piece also reported as decisive: {pct(other, len(ran))}")
    print(f"- stopped by the budget: {pct(stopped, len(ran))}")
    print(f"- model calls per case: median {statistics.median(calls):.0f}, max {max(calls)}")
    errors = [r for r in valid if "error" in r]
    if errors:
        print(f"- errors: {len(errors)}")
    print("\n| scenario | valid | headline = plant | exact sentence |\n|---|---|---|---|")
    by = defaultdict(list)
    for r in ran:
        by[r["scenario"]].append(r)
    for s, rs in sorted(by.items()):
        print(f"| {s} | {len(rs)} | {sum(r['headline_is_plant'] for r in rs)} | "
              f"{sum(r['narrowed_to_sentence'] for r in rs)} |")
    misses = [r for r in ran if not r["headline_is_plant"]]
    if misses:
        print("\nMisses:")
        for r in misses:
            print(f"- {r['case']}: headline was {r['headline']!r}; planted in {r['planted_in']}")
    if other:
        print("\nOther pieces reported as decisive:")
        for r in ran:
            for o in r["other_decisive"]:
                print(f"- {r['case']}: {o!r}")
    print()


if __name__ == "__main__":
    for p in sys.argv[1:]:
        summarize(p)
