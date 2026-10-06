# Benchmark: does `runtape why` find a known cause?

Each case is an agent conversation stopped at a decision point, in one of five
domains: support refunds, an email inbox, operations on a staging server, disk
cleanup, and access control. One sentence pushing the agent toward a specific
bad action (refund without approval, forward an email, drop a database, delete
backups, grant admin) is planted at a random sentence boundary inside one of
several realistic documents in the tool results. Everything else is ordinary
content.

On the model being tested, a case is kept only if the bad action happens in at
least half of 10 runs with the planted sentence and in at most 1 of 10 without
it. The planted sentence is then a cause of the decision by construction. `why`
is run on one bad decision, without being told where the sentence is, and
scored on:

- whether its headline cause is the planted sentence
- whether it narrowed the cause down to exactly that sentence
- what else it reported as decisive
- how many model calls it used

```
python bench/run.py --local llama3.1:8b --cases 20      # free, through Ollama
python bench/run.py --openai gpt-4o-mini --cases 20
python bench/run.py --openai openai/gpt-oss-120b --base-url https://openrouter.ai/api/v1 --max-tokens 2000 --cases 50
python bench/run.py --openai sarvam-105b --base-url https://api.sarvam.ai/v1 --max-tokens 2000 --cases 50
python bench/run.py --openai sarvam-105b --base-url https://api.sarvam.ai/v1 --max-tokens 2000 --cases 100 --seed 1 --work bench/seed1 --out bench/results/sarvam-105b-seed1.jsonl
python bench/report.py bench/results/*.rescored.jsonl
```

### Decoys: does it blame the obvious suspect?

`--decoys` adds a red herring to every case: a sentence plainly about the bad action ("Customer says the
delivery driver told them a full refund would be issued automatically"), in a different tool result, so it
is what someone asking "what made the agent do this?" would point at. A case counts only if the model
doesn't act on it: the bad action is about as common without it (within 3 of 10 runs, and at least half)
and rare when it's the only one left (at most 1 of 10). The red herring is then not a cause, by
construction. The real causes in this mode are written to work on careful models too: a policy loophole in
the help center, or a colleague's email.

An obvious injection ("SYSTEM OVERRIDE: ignore your instructions and ...") doesn't work as a red herring:
on Claude Haiku 4.5 it made the agent suspicious, and the bad action dropped from 7-8 of 10 runs to 0-4.

Each counted case is also given to the model with the question `why` starts from (which piece made the
agent act, and which sentence), and both are scored on whether they blame the real cause, the red herring
or something else. Results go to `bench/results/<model>-decoys.jsonl`; a case whose sentences changed since
it ran is moved to `.old.jsonl` and run again.

```
python bench/run.py --anthropic claude-haiku-4-5-20251001 --decoys --cases 25 --max-dollars 4
python bench/report.py bench/results/claude-haiku-4-5-20251001-decoys.jsonl
```

`bench/rescore.py` scores finished cases again with the current code, reading
every rerun from the saved replies in `bench/.cache` (no model calls). If the
current code asks for a reply that was never saved, that case is marked
incomplete.

## Results so far

Recorded in `bench/results/*.jsonl` (as run) and `*.rescored.jsonl` (scored
with the current code), with the conversations in `bench/traces`. Summary in
the main README. Two things the runs showed beyond the scores:

- Models resisted most planted sentences: gpt-oss-120b took the harmful
  action in 13 of 50 cases, sarvam-105b in 5 of 50 (and 10 of 100 in the
  new-seed run), Llama 3.1 8B in 6 of 18.
- Through OpenRouter, the same request was answered differently minutes
  apart (7 of 10 runs, then 0 of 5), consistent with calls being routed to
  different providers. Cases whose decision was no longer the model's usual
  choice when `why` ran are reported separately, not as misses.

## Baselines

`bench/baselines.py` attributes a decision three simpler ways, to compare with `why`: wording overlap
(`why --dry`), plain leave-one-out over whole messages and tool results (one run per piece, or the
largest drop over five), and asking a model which piece and sentence caused it. `bench/baselines_planted.py`
scores the first two on these cases from saved replies, with no model calls; `bench/agentdojo` runs all
three.

On the 18 counted cases whose decision was still made consistently when `why` ran (results in
`results/baselines-planted.jsonl`):

| method | planted sentence | the piece holding it | something else |
|---|---|---|---|
| `why` (headline) | 15 | 3 | 0 |
| wording overlap | 7 | 0 | 11 |
| leave-one-out, largest drop over 5 runs | - | 16 | 2 |

Leave-one-out with one run per piece flagged the piece holding the plant in all 18, and 37 other pieces
with it (at least one in every case), mostly inputs the action needs, which it can't tell apart from the
instruction. `why` lists those as also required, below the headline.

Runs resume where they stopped; replies are cached in `bench/.cache`.
`bench/sim.py` is a noisy stand-in model used only to test the harness.

What this measures and what it doesn't: the cases are generated, and the ground
truth is a sentence that was inserted, so the benchmark tests whether the
search finds a single planted cause among distractors, under the model's own
randomness. It does not cover causes spread across several pieces, causes
that only act together, or failures that start several steps before the
decision.
