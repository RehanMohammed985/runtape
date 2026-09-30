"""Score finished benchmark cases again with the current code, using only saved model replies.

    python bench/rescore.py bench/results/openai_gpt-oss-120b.jsonl

No model is called and no API key is needed: every rerun `why` asks for is read from bench/.cache.
When the current code needs a reply that was never saved (it explores differently than the code that
ran the case), the search stops there and the case is marked incomplete. Writes <results>.rescored.jsonl.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "src"), str(HERE)]

import runtape  # noqa: E402
from cases import generate  # noqa: E402
from run import score  # noqa: E402
from runtape.rerun import BudgetExceeded, OpenAIChatModel, build_request, request_for  # noqa: E402


def _uncached(**kw):
    raise BudgetExceeded("a reply that was never saved")


OFFLINE = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=_uncached)))


def main(path: str, work: str | None = None, model_fn: str | None = None) -> Path:
    src = Path(path)
    work_dir = Path(work) if work else HERE
    rows = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
    label = src.name.removesuffix(".jsonl")
    cases = {c.id: c for c in generate(max(int(r["case"].rsplit("-", 1)[1]) for r in rows) + 1)}
    out = src.with_name(label + ".rescored.jsonl")
    with out.open("w") as f:
        for r in rows:
            if r.get("valid") and "error" not in r:
                case = cases[r["case"]]
                t = runtape.load(work_dir / "traces" / label / f"{r['case']}.jsonl")
                resp = t.of_type("llm_response")[-1].id
                req = build_request(t, request_for(t, resp)[0])
                if model_fn:  # offline runs (bench/sim.py) are rescored with the same function
                    from runtape.rerun import load_model_fn

                    model = load_model_fn(model_fn)
                else:
                    model = OpenAIChatModel(client=OFFLINE, endpoint=req.get("endpoint"))
                rep = runtape.why(t, resp, model=model, match=case.bad, budget=None,
                                  cache_dir=str(work_dir / ".cache" / "why"), workers=1)
                r = {**r, **score(rep, case), "rescored": True}
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+")
    ap.add_argument("--work", help="folder with traces/ and .cache/ (default bench/)")
    ap.add_argument("--model-fn", help="for offline runs made with --model-fn")
    a = ap.parse_args()
    for p in a.results:
        main(p, a.work, a.model_fn)
