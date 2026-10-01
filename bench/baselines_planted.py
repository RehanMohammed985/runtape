"""Score the baselines on the planted-cause benchmark, from saved replies (no model calls).

    python bench/baselines_planted.py

For every counted case of every run in bench/results, the decision is attributed by wording overlap and by
plain leave-one-out (each piece removed once, using the first rerun `why` already saved for it), and scored
the way bench/run.py scores `why`: is the blamed text the planted sentence, or a larger piece holding it?
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "src"), str(HERE)]

import runtape  # noqa: E402
from baselines import dry, loo  # noqa: E402
from cases import generate  # noqa: E402
from runtape.rerun import BudgetExceeded, OpenAIChatModel, build_request, request_for  # noqa: E402

RUNS = [("openai_gpt-oss-120b", HERE / "traces/openai_gpt-oss-120b", 0),
        ("sarvam-105b", HERE / "traces/sarvam-105b", 0),
        ("meta-llama_llama-3.1-8b-instruct", HERE / "traces/meta-llama_llama-3.1-8b-instruct", 0),
        ("sarvam-105b-seed1", HERE / "seed1/traces/sarvam-105b", 1)]


def _uncached(**kw):
    raise BudgetExceeded("a reply that was never saved")


OFFLINE = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=_uncached)))


def _norm(s: str) -> str:
    return " ".join(s.split()).strip(" .\"'")


def classify(text: str, plant: str) -> str:
    t, p = _norm(text), _norm(plant)
    if t == p or (len(t) > 25 and t in p):
        return "sentence"
    if p in t:
        return "piece"
    return "other"


def main(cache: str | None = None) -> None:
    cache = cache or str(HERE / ".cache" / "why")
    out = HERE / "results" / "baselines-planted.jsonl"
    rows = []
    for label, tdir, seed in RUNS:
        results = HERE / "results" / f"{label}.jsonl"
        if not results.exists():
            continue
        src = [json.loads(x) for x in results.read_text().splitlines() if x.strip()]
        cases = {c.id: c for c in generate(max(int(r["case"].rsplit("-", 1)[1]) for r in src) + 1, seed=seed)}
        for r in src:
            if not r.get("valid") or "error" in r:
                continue
            case = cases[r["case"]]
            t = runtape.load(tdir / f"{r['case']}.jsonl")
            ev = t.of_type("llm_response")[-1].id
            req = build_request(t, request_for(t, ev)[0])
            d = dry(t, ev)
            lo = loo(t, ev, OpenAIChatModel(client=OFFLINE, endpoint=req.get("endpoint")), cache_dir=cache,
                     match=case.bad)
            flags = [classify(x, case.plant) for x in lo["texts"]]
            b = r.get("baseline") or [0, 0]
            rows.append({"run": label, "case": r["case"], "stable": b[1] > 0 and 2 * b[0] >= b[1],
                         "dry": classify(d["texts"][0], case.plant) if d["texts"] else "none",
                         "loo_plant_flagged": any(f != "other" for f in flags),
                         "loo_other_flagged": sum(f == "other" for f in flags), "loo_complete": lo["complete"],
                         "loo_top": classify(lo["top"], case.plant) if lo["top"] else "none"})
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    for name, sub in (("decisions made consistently", [r for r in rows if r["stable"]]),
                      ("decisions made less than half the time", [r for r in rows if not r["stable"]])):
        print(f"{name}: {len(sub)} cases")
        print(f"  wording overlap: the planted sentence {sum(r['dry'] == 'sentence' for r in sub)}, the piece "
              f"holding it {sum(r['dry'] == 'piece' for r in sub)}, something else "
              f"{sum(r['dry'] in ('other', 'none') for r in sub)}")
        lc = [r for r in sub if r["loo_complete"]]
        print(f"  leave-one-out ({len(lc)} with every first rerun saved): flags the piece holding the plant in "
              f"{sum(r['loo_plant_flagged'] for r in lc)}, and {sum(r['loo_other_flagged'] for r in lc)} other "
              f"pieces in all; cases with at least one other piece flagged: "
              f"{sum(r['loo_other_flagged'] > 0 for r in lc)}")
        print(f"  leave-one-out, largest drop over up to 5 reruns: the piece holding the plant "
              f"{sum(r['loo_top'] == 'piece' for r in sub)}, something else {sum(r['loo_top'] != 'piece' for r in sub)}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
