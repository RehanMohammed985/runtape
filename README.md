# runtape

[![tests](https://github.com/RehanMohammed985/runtape/actions/workflows/ci.yml/badge.svg)](https://github.com/RehanMohammed985/runtape/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/runtape)](https://pypi.org/project/runtape/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://github.com/RehanMohammed985/runtape/blob/main/LICENSE)

Counterfactual debugging and regression tests for AI agents.

runtape records agent runs to local files. When an agent does something it
shouldn't, `runtape why` reruns that one decision with parts of the context
removed and reports which part the decision depends on, with a significance
test. `runtape.rerun` then turns the failure into a test you can run in CI.

![runtape why on the inbox example](https://raw.githubusercontent.com/RehanMohammed985/runtape/main/docs/demo.gif)

An email assistant forwards an invoice to an outside address. `runtape why`
traces the call to one sentence in an HTML comment inside a vendor email:
with it, the agent forwards in 10 of 10 reruns; without it, in 0 of 10
(p = 5e-6). The test at the end fails until the agent is fixed.

Tracing tools such as LangSmith and Langfuse show what the agent saw.
Attribution methods such as ContextCite score context for a single model
response. runtape works on your agent's own recorded runs, on your machine,
and is meant for investigating a specific failure and keeping it fixed.

## Install

```
pip install runtape
```

Python 3.10+. Works with the OpenAI and Anthropic SDKs, LangChain and
LangGraph, OpenAI-compatible local servers (Ollama, LM Studio, vLLM), and
custom agent loops.

## Try it

```
git clone https://github.com/RehanMohammed985/runtape
cd runtape
pip install . openai anthropic

python examples/inbox_agent.py
runtape why last tool:forward_email --model-fn examples/inbox_agent.py:simulated_model
```

| example | failure |
|---|---|
| `inbox_agent.py` | an email assistant forwards an invoice because of an instruction hidden in an email |
| `refund_bot.py` | a support agent refunds $2,400 after reading a stale forum post in search results |
| `ops_agent.py` | an operations agent drops a shared staging database, following an old runbook line |

By default the examples run offline with a rule-based stand-in model
(`--model-fn`). To run them on a real model, add `--local MODEL` (Ollama,
free), `--openai MODEL` or `--anthropic MODEL`. Real models don't fail every
time, so `examples/hunt.py` runs an example until it fails, reports the tokens
used, and prints the `why` command:

```
python examples/hunt.py ops --local llama3.1:8b --tries 5
```

A real case on llama3.2 (3B): the refund agent paid order B-2290 $64, the
amount from a different customer's order earlier in the conversation. On the
recorded context it did this in 9 of 40 reruns; with the earlier order lookup
removed, in 0 of 40 (p = 0.001).

## Record your agent

```python
import runtape
from openai import OpenAI

rec = runtape.record(name="support-bot")
client = rec.wrap(OpenAI())         # every model call is recorded

@rec.tool                           # arguments, results, errors, latency
def lookup_order(order_id: str):
    ...
```

Traces go to `./traces/`, one JSONL file per run. See
[docs/usage.md](https://github.com/RehanMohammed985/runtape/blob/main/docs/usage.md)
for Anthropic, LangChain, streaming and custom loops.

## Find what a decision depends on

```
runtape why <trace> <event>
```

`<event>` is an event number, `tool:NAME` for the last call to a tool, or
`last`. How it works:

1. Rerun the recorded model call on the unchanged context to measure how often
   the model makes the same decision.
2. Remove each piece of the context (system prompt, messages, tool results)
   and rerun: 2 runs to screen, more where the decision changes.
3. Confirm candidates with a one-sided Fisher exact test, corrected for every
   variant tried, so randomness in the model isn't reported as a cause.
4. Narrow each confirmed piece down to JSON items, paragraphs and sentences.
5. Look inside pieces whose removal changes nothing, for a cause hidden next
   to content that pushes the other way.
6. Find causes that repeat or that are each enough on their own.
7. Lead with the piece that changes what the agent does. Pieces it only needs
   as input (without them it stops or looks the data up again) are listed as
   also required.
8. Rerun the headline cause with a second replacement text, when removal left
   one, and flag it if the result doesn't hold.

Only the selected model call is rerun. Your agent and its tools don't run
again, so nothing is refunded, emailed or deleted twice.

What this shows: on this model and this context, the decision depends on the
reported text. It is an intervention on the input, not a correlation, but it
is not an explanation of the model's internals, and a different model or a
different context can depend on different things.

## Turn the failure into a test

```python
import runtape

def test_injected_email_is_not_forwarded():
    runtape.rerun("traces/injected.jsonl", 22, runs=10).never_calls("forward_email")

def test_stricter_prompt_resists_it():
    runtape.rerun("traces/injected.jsonl", 22, runs=10, system=NEW_PROMPT).never_calls("forward_email")
```

`rerun` sends the recorded request to the model again, optionally edited:
`drop` (remove part of the context), `replace` (edit text), `system` (a new
system prompt) or `model_name`. The first test fails while the agent is
vulnerable; the second checks a fix on the exact context that failed, before
you ship it. Model output varies, so checks are made over several runs:
`never_calls`, `always_calls`, `rate(tool)`, `counts()`.

The same from the command line:

```
runtape rerun last 22 --system-file new_prompt.txt --runs 10
runtape odds last 22 --runs 20
```

## Cost and limits

- `why` makes typically 100 to 250 model calls for one decision. It is for
  investigating a failure, not for monitoring every decision. On a small
  hosted model that is typically cents; on a local model it is free. `--dry` ranks
  suspects without model calls, `--budget` caps the calls, and replies are
  cached, so repeating a run is free.
- Randomness: on a simulated model that ignores its context, false causes
  appeared in 0 to 5 of 100 runs, matching the 5% significance level. A cause
  that moves the decision rate from 90% to 10% was found in every run; 90% to
  30%, in about 4 of 5. Decisions the model makes less than about 1 time in 5
  are too rare to attribute; measure them with `odds` and test suspects with
  `rerun --drop`.
- Large contexts: pieces are tested top-down and only narrowed where they
  matter. By default at most 80 pieces are tested, ranked by shared wording
  with the decision, always including the system prompt, the task and the
  latest message. Anything skipped is listed in the report.
- Interactions: combinations are searched among the most suspicious pieces
  only (both needed, or either enough). A cause that needs three or more
  unrelated pieces together can be missed.
- Local models: reruns against a server on your machine (Ollama, LM Studio)
  run one at a time. An 8B model needs about 6 GB of free memory; on a laptop
  with 8 GB, use a 3B model or a hosted one.
- Replacement text: sentences, paragraphs and JSON items are cut out. A whole
  message or tool result is replaced with `[content removed]` (set with
  `--fill`), which can itself affect the model; step 8 checks for that.

More options, replaying whole runs through your code, an interactive trace
browser, an MCP server and the trace format are in
[docs/usage.md](https://github.com/RehanMohammed985/runtape/blob/main/docs/usage.md).

<!-- mcp-name: io.github.RehanMohammed985/runtape -->

## Related work

[ContextCite](https://arxiv.org/abs/2409.00729) and
[TracLLM](https://github.com/Wang-Yanting/TracLLM-Kit) attribute single model
responses to context by ablation.
[Causal Agent Replay](https://arxiv.org/abs/2606.08275),
[AgentDebugX](https://github.com/AgentDebugX/AgentDebugX) and
[AgentDoG](https://arxiv.org/abs/2601.15075) apply attribution and
counterfactual reruns to agents.
[AttriGuard](https://arxiv.org/abs/2603.10749) uses reruns to detect prompt
injection at runtime. LangSmith, Laminar and Langfuse record and replay agent
runs as hosted platforms.

## License

MIT
