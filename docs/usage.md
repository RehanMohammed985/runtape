# runtape usage

Reference for everything beyond the core workflow in the [README](../README.md).

## Recording

```python
import runtape
from anthropic import Anthropic

rec = runtape.record(name="support-bot")
client = rec.wrap(Anthropic())

@rec.tool
def lookup_order(order_id: str):
    ...
```

- OpenAI: `rec.wrap(OpenAI())` records Chat Completions and the Responses API.
- LangChain / LangGraph: `graph.invoke(inputs, config={"callbacks": [rec.langchain()]})`.
- Other frameworks: `rec.log_llm_request(...)` and `rec.log_llm_response(...)`.

Traces are written to `./traces/`, one file per run, as events happen, so a
crash still leaves a readable file. Sync, async and streaming calls are
recorded, including Anthropic's `messages.stream()`, OpenAI's `.stream()` and
`.parse()`, and beta endpoints. For OpenAI-compatible servers (Ollama, LM
Studio, vLLM), the base URL is recorded so reruns go to the same server.

Pass `redact=fn` to `runtape.record` to filter events before they are
written. If `fn` raises, the event content is dropped rather than written
unredacted.

## runtape why options

```
runtape why <trace> <event> [options]
```

`<trace>` is a path or `last`. `<event>` is an event number, `tool:NAME` (the
last call to that tool), or `last` (the last model response).

| option | |
|---|---|
| `--runs N` | reruns per variant (default 5) |
| `--budget N` | maximum model calls (default 400) |
| `--dry` | rank suspects by shared wording, no model calls |
| `--match REGEX` | explain a text answer, or a tool call with specific arguments |
| `--tool NAME` | explain whether this tool is called, when a reply calls several |
| `--exact-args` | count a rerun as the same decision only if the arguments match |
| `--fill TEXT` | replacement for a removed message or tool result: `marker` (default), `empty`, or any text |
| `--max-pieces N` | test at most N pieces, most suspicious first (default 80) |
| `--expand N` | look inside N pieces for masked causes (default 6) |
| `--all` | list every piece tested |
| `--json FILE` | also write the report as JSON |
| `--model-fn module:function` | rerun with a Python function instead of the recorded model |
| `--no-cache` | don't reuse cached replies |

## runtape rerun and odds

```
runtape rerun <trace> <event> --drop "9[1]"             # remove part of the context
runtape rerun <trace> <event> --replace "old=>new"      # edit text the model saw
runtape rerun <trace> <event> --system-file fixed.txt   # try a new system prompt
runtape rerun <trace> <event> --model <name>            # try another model
runtape odds  <trace> <event>                           # the unchanged context
```

References for `--drop`: an event number (`9`), a part of one (`9[1]`,
`9[1].text`, `12.body para 4`), or `system`. The labels in a `why` report use
the same form.

In Python, `runtape.rerun(trace, event, drop=..., replace=..., system=...,
model_name=..., runs=...)` returns a distribution with `never_calls(tool)`,
`always_calls(tool)`, `rate(tool)` and `counts()`.

## runtape fix and runtape test

```
runtape fix <trace> <event> [--runs 10] [--write-test PATH] [--match REGEX] [--tool NAME] [--budget 600]
runtape test <trace> <event> [--out PATH] [--add-system TEXT | --add-system-file F] [--runs 10]
```

`fix` runs `why`, then checks each candidate fix with `--runs` reruns of the
recorded decision: a rule that tool results are data, a rule that the call
needs the user's own request, both, and removing the cause at its source.
`--write-test` writes a pytest file for the best passing fix and copies the
trace next to it under `traces/`. `test` writes the same file for a fix you
pass yourself, or with no fix (a test that fails until the agent changes).

In Python: `runtape.fix(trace, event, model=..., runs=10)` returns a report with
`candidates` (each with `kept`, `n`, `p` and `holds()`) and `best`;
`runtape.write_test(trace_path, event, target, out, add_system=...)` writes the
test.

## Replay a whole run

```python
with runtape.replay("traces/bad-refund.jsonl") as rp:
    client = rp.wrap(Anthropic())

    @rp.tool
    def lookup_order(order_id): ...     # returns the recorded result

    run_agent(client)
```

Model responses and tool results are served from the trace, so the run is
offline and deterministic, and you can set breakpoints anywhere in your agent.
Tool results are restored to their original types (dataclasses, Pydantic
models, tuples). If a request differs from the recording, replay stops and
reports the first difference, or continues against the live model with
`on_diverge="live"`. Streamed calls are not served from the recording, and
LangChain runs can't be replayed yet.

## Browsing a trace

```
runtape                    # newest trace
runtape traces/run.jsonl
```

| command | |
|---|---|
| Enter, `step [N or TYPE]` | next event |
| `back`, `goto N` | move |
| `show [N]` | full event |
| `context [N]` | the model's input at that point |
| `grep TERM`, `next`, `prev` | search |
| `diff [A] [B]` | context changes between two events |
| `why`, `rerun`, `odds` | run on the current event |

Non-interactive subcommands: `ls`, `summary`, `timeline`, `show`, `context`,
`grep`, `diff`.

## MCP server

```
pip install "runtape[mcp]"
claude mcp add runtape -- runtape mcp
```

Exposes trace inspection, `why` and `rerun` as MCP tools for coding agents.

## Trace format

Append-only JSONL, one event per line. Message history is delta-encoded. See
[SPEC.md](../SPEC.md).
