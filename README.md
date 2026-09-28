# runtape

[![tests](https://github.com/RehanMohammed985/runtape/actions/workflows/ci.yml/badge.svg)](https://github.com/RehanMohammed985/runtape/actions/workflows/ci.yml)

A flight recorder for AI agents.

runtape records every model call, tool call, and state change your agent makes
into a single local file, then lets you replay the run step by step from the
terminal. When an agent does something wrong, you can see exactly what it knew
at the moment it decided, and where that information came from.

- Local JSONL files. No server, no account, no hosted dashboard.
- Works with the OpenAI and Anthropic SDKs, LangChain, LangGraph, or any custom loop.
- Crash-safe: every event is written as it happens.

## Install

```
pip install runtape
```

Requires Python 3.10+.

## Record a run

```python
import runtape
from anthropic import Anthropic

rec = runtape.record(name="support-bot")
client = rec.wrap(Anthropic())      # records every messages.create call

@rec.tool                           # records arguments, result, errors, latency
def lookup_order(order_id: str):
    ...

rec.state("plan", ["look up order", "check policy"])   # anything else worth keeping
```

The trace is written to **./traces/**.

**OpenAI:** **rec.wrap(OpenAI())** records chat completions and the Responses API.

**LangChain / LangGraph:** pass the callback handler.

```python
graph.invoke(inputs, config={"callbacks": [rec.langchain()]})
```

**Anything else:** log model calls directly.

```python
rid = rec.log_llm_request(provider="local", model="llama-3", messages=messages)
rec.log_llm_response(rid, text=reply, tool_calls=[], stop_reason="stop", raw=None, latency_ms=412)
```

Sync, async, and streaming calls are all supported. Tool calls are linked to
the model response that requested them.

## Replay a run

```
runtape                 # replay the newest trace interactively
runtape replay <trace>  # replay a specific trace
```

Inside the replay:

| command | what it does |
|---|---|
| Enter, **step [N or TYPE]** | move forward, e.g. **step tool_call** |
| **back [N or TYPE]** | move backward |
| **goto N** | jump to event N |
| **show [N]** | full detail of an event (**--raw** for JSON) |
| **context [N]** | everything the model saw at that point |
| **grep TERM** | every event containing TERM, and where it first appeared |
| **next**, **prev** | jump between grep hits |
| **diff [A] [B]** | events between two points and how the context changed |
| **list**, **summary**, **errors** | timeline, run totals, failures |

The same views work as one-off commands: **runtape ls**, **summary**,
**timeline**, **show**, **context**, **grep**, **diff**.

## Example

**examples/refund_bot.py** is a support agent that issues a $2,400 refund it
should have escalated to a manager. It runs offline with a scripted model.

```
python examples/refund_bot.py
runtape grep last "any amount"
```

```
4 events contain "any amount". First appears at #9.
    9 tool_result   search_kb  result[1].text   <- entered here
      ...refunds need approval? A: No, agents can approve refunds of any amount without manager review.
   10 llm_request   claude-scripted  messages_append[1].content[0].content
      ...refunds need approval? A: No, agents can approve refunds of any amount without manager review."}]
   30 llm_response  text
      Per our policy, agents can approve refunds of any amount without manager review, so I'll process this now.
   34 llm_request   claude-scripted  messages_append[0].content[0].text
      Per our policy, agents can approve refunds of any amount without manager review, so I'll process this now.
```

The knowledge base search returned a stale 2019 FAQ at event 9. It stayed in
context, and the agent acted on it at event 30. **runtape context last 30**
shows the full context window at the moment of the decision.

## Reading traces in Python

```python
t = runtape.load("traces/20260928-183022-refund-bot-7889a8.jsonl")
t.summary()          # model calls, tokens, tool calls, errors, status
t.context(30)        # system prompt, tools, messages, and reply at event 30
t.grep("any amount") # matching events, first one marked
```

## Trace format

One JSON object per line, append-only. Message history is delta-encoded, so
each model call stores only what is new. The full format is documented in
[SPEC.md](SPEC.md).

## Redacting secrets

```python
rec = runtape.record(redact=lambda event: scrub(event))
```

The function receives each event as a dict before it is written.

## License

MIT
