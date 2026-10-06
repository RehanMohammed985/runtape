# runtape on AgentDojo

[AgentDojo](https://github.com/ethz-spylab/agentdojo) is a public benchmark of tool-using agents
under prompt injection: realistic tasks in banking, Slack, travel and workspace environments, with
attacker text placed in the data the agent reads. Unlike the cases in `bench/`, nothing here was
written for runtape.

`run.py` works through (user task, injection task) pairs in phases:

| phase | what it does |
|---|---|
| `agent` | runs AgentDojo's own agent pipeline on the model, recording every model call; AgentDojo's checks say whether the user's task got done and whether the attack worked |
| `why` | if the attack worked, runs `runtape why` on the reply that made the attacker's call (found by a value only the injection asks for, such as the attacker's account), without saying where the injection is |
| `baselines` | attributes the same decision by wording overlap, by plain leave-one-out (one run per piece, and the largest drop over five) and by asking the model which piece and sentence caused it |
| `fix` | checks fixes on the recorded decision with `runtape fix`, then runs the full task with each prompt fix and with none: under attack (does the attack still work, does the task get done) and without it (does the fix break the task) |
| `suggest` | asks the agent's model for rules aimed at the proven cause, checks them on the recorded decision with `runtape fix`, and runs the live task with the model's first suggestion (what asking the model gets you) and with the fix runtape picks |
| `control` | runs `why` and the baselines on the user's own action (paying the real bill) in a trace where the injection is present; blaming the injection there is a false positive |

```
pip install agentdojo
python bench/agentdojo/run.py --openai sarvam-105b --base-url https://api.sarvam.ai/v1 --suite banking
python bench/agentdojo/run.py --openai sarvam-105b --base-url https://api.sarvam.ai/v1 --paper --pairs 30
python bench/agentdojo/report.py bench/results/agentdojo-sarvam-105b.jsonl
python bench/agentdojo/run.py --anthropic claude-haiku-4-5-20251001 --suite banking,slack --pairs 30 --max-dollars 5
```

`--anthropic` runs AgentDojo's agent on a Claude model through the Messages API (key in `ANTHROPIC_API_KEY`), and
the reruns go there too. `--max-dollars` counts every request (agent runs, reruns, the judge, live fix runs) from
the tokens the API reports, and stops the run once that much is spent; it resumes where it stopped.

`--paper` runs every phase on all four suites. Results are one JSON line per pair, rewritten after each
phase; a pair that already has a phase isn't run again, so an interrupted run resumes, and running with
more phases later fills them in for pairs already done. If the model server can't be reached, the run
waits a minute and tries the pair again, for up to an hour. Utility without the attack depends only on the
user task and the system prompt, so it is shared between pairs (`benign-<model>.json`).

How the text a method blames is scored:

- inside the injection: at least 80% of its words come from the injected text;
- including the attacker's instruction: it covers at least 60% of the words of the injection task's goal;
- the whole tool result holding it: larger content that includes the instruction.

A reasoning model spends reply tokens thinking, and a server's default cap can cut it off before it acts:
with Sarvam's default of 2048, about one reply in six was cut off, in the agent's own runs and in reruns
alike. `--max-tokens 4096` sends a larger cap with every request (the trace records it, so reruns send it
too) and writes results to their own files.

The attack is AgentDojo's `important_instructions` without the model's name. A decision the model no
longer makes at least half the time when rerun is reported separately, as in the main benchmark. Traces
are written to `bench/agentdojo/traces` and reruns are cached in `bench/agentdojo/.cache`.

`tests/test_agentdojo.py` runs every phase end to end against a local stand-in server.

## Results: sarvam-105b, banking and Slack

sarvam-105b is an open-weight model (Apache 2.0), run here through Sarvam's API. 30 pairs per suite, every
phase. Results: `bench/results/agentdojo-sarvam-105b.jsonl`; traces: `bench/agentdojo/traces/sarvam-105b`.
`python bench/agentdojo/report.py bench/results/agentdojo-sarvam-105b.jsonl` prints the full report.

These results are from runtape 0.5.0, which tested at a 5% significance level and searched every piece.
0.6.0 tests at 1% and starts from the model's own guess at the cause; to rerun the search with it on the
same agent runs (the fixes are kept where the cause comes out the same), use `--redo why` with every phase.

The attack worked in 31 of 60 pairs, and the attacker's call was found in 30. When rerun on the same
context, 15 of those decisions were made consistently, 8 intermittently (15-60% of reruns, searched on 30
reruns per check), and 7 too rarely to attribute, which `why` reported as such.

On the 23 decisions it searched:

| method | text inside the injection | ...including the attacker's instruction | model calls |
|---|---|---|---|
| runtape why | 20/23 (87%) | 15/23 | median 153 (consistent), 262 (intermittent) |
| model as judge | 18/23 (78%) | 16/23 | 1-2 |
| leave-one-out, largest drop over 5 runs | 15/23 (65%) holds it | - | about 5 per piece |
| wording overlap | 10/23 (43%) | 10/23 | 0 |

- Leave-one-out with one run per piece flagged the injection in 18 of 23, along with 41 pieces that had
  nothing to do with it.
- runtape listed the injection among its causes, headline or not, in 21 of 23. Of the three it missed as
  headline: one needed two things together (the message linking to a page and the injected text on it) and
  ranked the message first; in another, the agent had restated the injected instruction in its own reply
  before acting, so removing the injection alone no longer stopped the action (leave-one-out missed it
  too); in the third, no piece passed the test.
- The judge is a strong baseline on this question, close to runtape and slightly better at quoting the
  attacker's sentence. What it doesn't give is evidence that removing the text changes the decision, or a
  way to check a fix.
- False positives: on 11 decisions that were the user's own action in a trace holding the injection,
  runtape and the judge blamed the injection 0 times, wording overlap once.

Fixes, on the 22 decisions with a cause (3 live AgentDojo runs per fix, under attack and without it):

| fix | passed on the recorded decision | live: attack worked | live: task done, no attack |
|---|---|---|---|
| none | - | 50/66 | 51/66 |
| untrusted content (prompt) | 0/22 | 34/66 | 52/66 |
| action guard (prompt) | 3/22 | 30/66 | 52/66 |
| both rules (prompt) | 4/22 | 31/66 | 51/66 |
| fix the source (remove the injection) | 20/22 | - | - |

- No prompt fix stops these attacks on this model: they cut the attack rate from 76% to 45-52% and don't
  break the task without an attack.
- The check on the recorded decision predicts the live result. Fixes that cut the attack below 20% on the
  recorded decision let it through 27% of the time in full runs, against 53% for 20-50% and 61% above 50%.
  It agreed with the live outcome (blocked in all 3 runs or not) for 55 of 66 fixes.

Limits: one model and two suites. sarvam-105b's default cap of 2048 reply tokens cut off about one reply
in six mid-thought, which accounts for most of the unstable decisions in banking (`--max-tokens` sets a
larger cap for later runs). The judge thought past 4096 tokens without answering on 30 of its 49
prompts and answered those with reasoning off. 3 live runs per fix is thin evidence for "blocked every time".
