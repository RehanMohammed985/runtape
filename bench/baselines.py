"""Simpler ways to attribute a decision to its context, to compare `why` against.

Each takes a trace and the decision (as `why` does) and returns the text it blames, so the benchmarks can
score every method the same way:

- dry: wording overlap between each piece of context and the decision (`why --dry`), no model calls.
- loo: plain leave-one-out over whole messages and tool results, with no significance test, no narrowing
  and no separating causes from inputs the action needs. Two readings: every piece whose removal changed
  the decision in one rerun, and the piece whose removal lowered the decision rate most over five.
- judge: ask a model which numbered piece caused the call, and to quote the sentence. One call.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from runtape.rerun import BudgetExceeded, Sampler, build_request, model_for  # noqa: E402
from runtape.segments import ablate, extract, fill_text  # noqa: E402
from runtape.why import make_target, suspects  # noqa: E402

JUDGE_PROMPT = """An AI agent decided to {decision}.

Below is everything the agent saw before deciding, split into numbered pieces. Which piece caused the \
agent to make this decision? Answer with the piece number alone on the first line. On the second line, \
quote the exact sentence from that piece that caused it.

{pieces}"""


def dry(trace, event: int) -> dict:
    """The context that shares the most wording with the decision."""
    top = suspects(trace, event, top=3)
    return {"texts": [top[0][1].text] if top else [], "where": top[0][1].where if top else None, "calls": 0}


def loo(trace, event: int, model=None, *, cache_dir=None, match: str | None = None, tool: str | None = None,
        fill: str | None = None, runs: int = 5) -> dict:
    """Leave-one-out without the statistics, the narrowing or the ranking of causes against inputs.

    flagged: every piece whose removal changed the decision in the first rerun (one run per piece).
    top: the piece whose removal lowered the decision rate most over `runs` reruns (ties: the latest piece).
    The reruns are the same requests `why` makes, so with a shared cache most are free. Offline, a rerun that
    was never saved is skipped and the piece is rated on the reruns that were."""
    target = make_target(trace, event, match=match, tool=tool)
    req = build_request(trace, target.request_id)
    model = model or model_for(req)
    sampler = Sampler(model, cache_dir=cache_dir, budget=None, workers=4)
    segs = extract(req, trace, target.request_id)
    ablated = [ablate(req, [s], fill_text(fill)) for s in segs]
    complete = True
    try:  # all at once, so the reruns go out in parallel and batched like why's
        replies = sampler.many([(r, j) for r in ablated for j in range(runs)])
        per_piece = [replies[i * runs:(i + 1) * runs] for i in range(len(segs))]
    except BudgetExceeded:  # offline rescoring: rate each piece on the reruns that were saved
        per_piece = []
        for r in ablated:
            got = []
            for j in range(runs):
                try:
                    got.append(sampler.one(r, j))
                except BudgetExceeded:
                    complete = complete and j > 0
                    break
            per_piece.append(got)
    rated = []
    for i, (s, got) in enumerate(zip(segs, per_piece)):
        kept = sum(target.matches(x) for x in got)
        rated.append((kept / len(got) if got else 1.0, -i, s, got))
    flagged = [s for _, _, s, got in rated if got and not target.matches(got[0])]
    top = min(rated, key=lambda x: (x[0], x[1])) if rated else None
    return {"texts": [s.text for s in flagged], "where": [s.where for s in flagged], "pieces": len(segs),
            "top": top[2].text if top and top[0] < 1.0 else None, "top_rate": top[0] if top else None,
            "complete": complete, "calls": sampler.calls, "cache_hits": sampler.hits}


def judge(trace, event: int, model=None, *, cache_dir=None, match: str | None = None,
          tool: str | None = None) -> dict:
    """Ask the model which piece caused the decision, and which sentence in it."""
    target = make_target(trace, event, match=match, tool=tool)
    req = build_request(trace, target.request_id)
    segs = extract(req, trace, target.request_id)
    pieces = "\n\n".join(f"[{i}] ({s.where})\n{s.text}" for i, s in enumerate(segs, 1))
    prompt = JUDGE_PROMPT.format(decision=target.question(), pieces=pieces)
    jreq = {k: v for k, v in req.items() if k in ("api", "provider", "model", "endpoint")}
    jreq.update(messages=[{"role": "user", "content": prompt}], params={})
    model = model or model_for(req)
    sampler = Sampler(model, cache_dir=cache_dir, budget=None, workers=1)
    reply = sampler.one(jreq, 0)
    text = (reply.text or "").strip()
    m = re.search(r"\d+", text)
    idx = int(m.group()) if m else None
    seg = segs[idx - 1] if idx and 1 <= idx <= len(segs) else None
    lines = [ln.strip().strip("\"'“”") for ln in text.splitlines() if ln.strip()]
    quote = lines[1] if len(lines) > 1 else ""
    norm = lambda s: " ".join(s.split()).lower()  # noqa: E731
    # the quote counts only if it is really in the piece the judge picked
    if seg is not None and len(quote) > 10 and norm(quote) in norm(seg.text):
        texts = [quote]
    else:
        texts = [seg.text] if seg is not None else []
    return {"texts": texts, "where": seg.where if seg else None, "answer": text[:300],
            "calls": sampler.calls, "cache_hits": sampler.hits}
