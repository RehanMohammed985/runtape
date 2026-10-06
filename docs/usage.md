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

## Importing traces from other tools

```
runtape import spans.json [-o traces/run.jsonl] [--trace-id ID] [--provider openai|anthropic]
                          [--base-url URL] [--tools tools.json]
runtape import langfuse:<trace id>
```

Sources:

- OpenTelemetry: an OTLP JSON export (a whole file, or one object per line
  as the collector's file exporter writes them), or spans as the Python SDK's
  console exporter prints them. Model calls are read from the GenAI semantic
  conventions (`gen_ai.input.messages`, `gen_ai.output.messages`,
  `gen_ai.system_instructions`, `gen_ai.tool.definitions`), the older
  OpenLLMetry attributes (`gen_ai.prompt.N.*`, `llm.request.functions.N.*`),
  OpenInference (`input.value`, `llm.input_messages.N.*`, `llm.tools.N.*`),
  and the older per-message span events. When a framework span wraps the
  provider's span for the same call, the call is imported once.
- Langfuse: a trace as `GET /api/public/traces/{id}` returns it, or
  `langfuse:<trace id>` to fetch it with `LANGFUSE_PUBLIC_KEY`,
  `LANGFUSE_SECRET_KEY` and `LANGFUSE_HOST`.

An export with several traces is written to one file per trace, unless
`--trace-id` picks one. Each model call becomes a request and a reply, and
each tool call the model made becomes a tool call and its result, read from
the next request's history.

A rerun resends the request, so it needs what was sent:

- Content capture must be on. For the OpenTelemetry instrumentations set
  `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true`; OpenLLMetry
  captures content unless `TRACELOOP_TRACE_CONTENT=false`.
- Tool definitions: without them the model can't call a tool, so a decision
  to call one won't repeat. The import says when they are missing; pass the
  tools as sent to the model with `--tools tools.json`.
- Where to send reruns: calls are resent to OpenAI's API, or Anthropic's when
  the source says the call went to Claude (the messages are converted to
  Anthropic's format). Spans name the server's host but not its path, so for
  any other server pass `--base-url`.

Images and audio in OpenAI-format messages are kept as they were. Other non-text content (Anthropic image
blocks, reasoning blocks) is dropped, so a decision that depends on it won't repeat on rerun.

## Timeouts

Reruns wait 10 seconds to connect and 300 seconds for a reply, then retry. A slow local model may need longer:
set `RUNTAPE_TIMEOUT` (seconds).

## runtape why options

```
runtape why <trace> <event> [options]
```

`<trace>` is a path, part of a file name in `./traces` (a run id, say), or
`last` for the newest. `<event>` is an event number, `tool:NAME` (the last call
to that tool), or `last` (the last model response). When the agent calls the
same tool many times (`run_command`, `search`), `tool:NAME` asks whether it
calls that tool at all; add `--match` with part of the arguments to explain one
call, as `why` suggests when this applies.

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
| `--full` | test every piece and keep searching after the main cause is settled: hidden causes and combinations |
| `--no-guess` | don't start from the model's own guess at the cause; test every piece |
| `--all` | list every piece tested |
| `--json FILE` | also write the report as JSON |
| `--model-fn module:function` | rerun with a Python function instead of the recorded model (below) |
| `--no-cache` | don't reuse cached replies |
| `-y` | don't ask before making model calls |

Replies are cached in `.runtape/cache` in the folder you run from, so a repeat
run, or `fix` after `why`, doesn't pay twice. Add `.runtape/` to `.gitignore`.

### Model functions

`--model-fn path/to/file.py:name` (or `package.module:name`) reruns with your
own function instead of the API the trace was recorded with: a local model, a
proxy, or a stand-in for tests. It gets the request as a dict (`model`,
`messages`, `system`, `tools` and `params`, in the format the agent sent them)
and returns a string, or a dict:

```python
def my_model(req):
    return {"text": "...", "tool_calls": [{"id": "call_1", "name": "issue_refund",
                                            "arguments": {"order_id": "Z-9", "amount": 900}}]}
```

The examples' `simulated_model` functions are small instances of this.
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
runtape fix <trace> <event> [--runs 10] [--write-test PATH] [--match REGEX] [--tool NAME] [--budget 600] [--full] [--no-suggest]
runtape test <trace> <event> [--out PATH] [--add-system TEXT | --add-system-file F] [--runs 10]
```

`fix` runs `why`, then checks each candidate fix with `--runs` reruns of the
recorded decision: a rule that tool results are data, a rule that the call
needs the user's own request, both, removing the cause at its source, and up
to three rules the agent's model proposes when shown the proven cause (one
extra model call; `--no-suggest` skips it).
`--write-test` writes a pytest file for the best passing fix and copies the
trace next to it under `traces/`. A fix passes when the bad call never happens
in its reruns and the drop is significant. `test` writes the same file for a
fix you pass yourself, or with no fix: a test that fails while the model still
makes the decision on the recorded context. Generated tests call the model on
every run (no cache).

In Python: `runtape.fix(trace, event, model=..., runs=10, suggest=True)` returns a report with
`candidates` (each with `kept`, `n`, `p`, `holds()` and `suggested`) and `best`;
`runtape.write_test(trace_path, event, target, out, add_system=...)` writes the
test. `target` is the decision: the report's `report.target`, or
`make_target(trace, event, tool=..., match=...)` from `runtape.why`.

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
