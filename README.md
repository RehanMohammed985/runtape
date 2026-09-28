# runtape

[![tests](https://github.com/RehanMohammed985/runtape/actions/workflows/ci.yml/badge.svg)](https://github.com/RehanMohammed985/runtape/actions/workflows/ci.yml)

**git blame for AI agents.**

runtape records everything your agent does to a local file. When it does
something wrong, **runtape why** tells you which part of what it read caused
it, and proves it: it removes pieces of the context, reruns that one decision
several times per variant, and shows what the agent does instead.

![runtape why](docs/why.svg)

Here the agent refunded $2,400 without a manager. The cause was one sentence
in one search result: a community forum post that had been indexed into the
help center. Without that sentence the agent escalates, 5 out of 5 times.

The cause is also **masked**. The same search result contained the real $200
policy, so removing the whole result changes nothing. Tools that only show
you the trace, or that test removing whole messages, would not find it.

## Install

```
pip install runtape
```

Python 3.10+. Works with the OpenAI and Anthropic SDKs, LangChain and
LangGraph, or any custom loop.

## Try it in one minute

The repo includes a support agent with this exact bug. It runs offline.

```
git clone https://github.com/RehanMohammed985/runtape && cd runtape && pip install -e .
python examples/refund_bot.py
runtape why last tool:issue_refund --model-fn examples/refund_bot.py:simulated_model
```

The offline demo uses a small simulated model. With an Anthropic API key,
run the same agent on real Claude and debug that run instead:

```
python examples/refund_bot.py --live
runtape why last tool:issue_refund
```

## Record your agent

```python
import runtape
from anthropic import Anthropic

rec = runtape.record(name="support-bot")
client = rec.wrap(Anthropic())      # every model call is recorded

@rec.tool                           # arguments, result, errors, latency
def lookup_order(order_id: str):
    ...
```

**OpenAI:** **rec.wrap(OpenAI())** records chat completions and the Responses API.
**LangChain / LangGraph:** **graph.invoke(inputs, config={"callbacks": [rec.langchain()]})**.
**Anything else:** **rec.log_llm_request(...)** and **rec.log_llm_response(...)**.

Traces go to **./traces/**, one JSONL file per run, written as the run
happens, so a crash still leaves a readable file. Sync, async and streaming
calls are supported.

## Find out why

```
runtape why <trace> <event>
```

Point it at a tool call, a model reply, or a model call. **last** is the
newest trace; **tool:NAME** is the last call of a tool.

What it does:

1. Reruns the recorded decision on the identical context, to check the model
   actually makes it reliably. An unstable decision is reported as such.
2. Splits the context into pieces: system prompt, messages, tool results.
3. Removes each piece and reruns the decision. A quick 2-run screen, extended
   to the full count only where something changes.
4. Narrows every cause down through JSON items, paragraphs and sentences.
5. Looks inside the most suspicious pieces even when removing them whole does
   nothing, to find masked causes.
6. If no single piece matters, searches for the smallest set that does.
7. Reports what the agent does instead, and separates **causes** (without
   them it acts differently) from **needed inputs** (without them it can't act).

Only the single decision is rerun. Your agent and its tools never run again,
so nothing is refunded, emailed or deleted twice.

Options: **--runs** (reruns per variant, default 5), **--budget** (max model
calls, default 300), **--dry** (a free ranked list of suspects, no model
calls), **--match REGEX** (explain a text answer), **--json** (write the report).
Replies are cached in **./.runtape/**, so running it again is free.

## Test a fix before you ship it

![runtape rerun](docs/rerun.svg)

```
runtape rerun <trace> <event> --drop "9[1]"             # remove part of the context
runtape rerun <trace> <event> --replace "old=>new"      # edit text the model saw
runtape rerun <trace> <event> --system-file fixed.txt   # try a new system prompt
runtape rerun <trace> <event> --model <name>            # try another model
runtape odds  <trace> <event>                           # how often it makes this call at all
```

## Turn the bug into a test

```python
def test_no_big_refunds_without_a_manager():
    runtape.rerun("traces/bad-refund.jsonl", 30, system=FIXED_PROMPT).never_calls("issue_refund")
```

This reruns the decision that failed in production, on its exact context,
with your fix. **always_calls**, **rate(tool)** and **counts()** are also available.

## Replay a whole run through your code

```python
with runtape.replay("traces/bad-refund.jsonl") as rp:
    client = rp.wrap(Anthropic())

    @rp.tool
    def lookup_order(order_id): ...     # not executed: returns the recorded result

    run_agent(client)
```

Model replies and tool results come from the recording, so the run is free,
offline and identical every time. Set a breakpoint anywhere in your agent. If
your code sends a request that differs from the recording, replay stops and
shows the first difference. With **on_diverge="live"** it switches to the real
model from that point.

## Browse a run

```
runtape                  # replay the newest trace interactively
```

| command | what it does |
|---|---|
| Enter, **step [N or TYPE]** | move forward, e.g. **step tool_call** |
| **back**, **goto N** | move around |
| **show [N]** | an event in full |
| **context [N]** | everything the model saw at that point |
| **grep TERM**, **next**, **prev** | find where something entered the run |
| **diff [A] [B]** | how the context changed between two points |
| **why**, **rerun**, **odds** | the experiments above, on the current event |

## Use it from Claude Code or Cursor

```
pip install "runtape[mcp]"
claude mcp add runtape -- runtape mcp
```

Then ask your coding agent why your agent did something. It can read
timelines and context windows, grep runs, and run **why** and **rerun** itself.

## Trace format

One JSON object per line, append-only, with message history stored as deltas
so long runs stay small. See [SPEC.md](SPEC.md).

To keep secrets out of traces, pass **redact=fn** to **runtape.record**. It
receives each event as a dict before it is written.

## Limits

- **why** reruns a real model, so it costs tokens. A typical run makes 50 to
  100 calls on one context. **--dry**, **--budget** and **--max-pieces** keep
  that in check, and results are cached.
- Removing a piece replaces it with **[content removed]**. The marker itself
  can occasionally influence a model.
- Rerunning needs the model the agent used. Other providers work through
  **--model-fn**, which takes any function from a request to a reply.

## License

MIT
