# runtape

A flight recorder for AI agents. Records every model call, tool call, and state
change to one local JSONL file so you can replay the run and find out why the
agent did what it did.

Local file, no server, no account, works with any agent code.

> Status: v0.1, recorder only. Replay CLI is next.

## Install

    pip install runtape

## Record

```python
import runtape
from anthropic import Anthropic   # or: from openai import OpenAI

rec = runtape.record(name="support-bot")
client = rec.wrap(Anthropic())    # every messages.create call is now recorded

@rec.tool                         # arguments, result, errors, latency
def lookup_order(order_id): ...

rec.state("plan", ["look up order", "check policy"])   # anything else
```

Trace lands in **./traces/**. If the process crashes, everything up to the
crash is already on disk.

Supported out of the box: OpenAI (chat completions + responses) and Anthropic
(messages), sync and async, streaming included. Anything else: call
**rec.log_llm_request** / **rec.log_llm_response** yourself.

## Read a trace

```python
t = runtape.load("traces/....jsonl")
t.summary()             # calls, tokens, errors, status
t.context(30)           # exactly what the model saw at event 30
t.grep("any amount")    # every event with this text; first hit = where it entered
```

## Example

**examples/refund_bot.py** is a support agent that issues a $2,400 refund it
should have escalated. **grep** shows the cause entered 21 events earlier, from
a stale FAQ returned by the knowledge base search.

## Trace format

See [SPEC.md](SPEC.md). One JSON event per line, append-only, message history
delta-encoded so long runs stay small.
