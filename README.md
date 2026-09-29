# runtape

[![tests](https://github.com/RehanMohammed985/runtape/actions/workflows/ci.yml/badge.svg)](https://github.com/RehanMohammed985/runtape/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/runtape)](https://pypi.org/project/runtape/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://github.com/RehanMohammed985/runtape/blob/main/LICENSE)

runtape records AI agent runs to local JSONL files and finds which part of
the model's context caused a given decision.

`runtape why` removes parts of the context, reruns the recorded model call
several times per variant, and reports the parts whose removal changes the
decision, with a significance test.

![runtape why on the inbox example](https://raw.githubusercontent.com/RehanMohammed985/runtape/main/docs/demo.gif)

In the example above, an email assistant forwards an invoice to an external
address. `runtape why` attributes the call to one sentence in an HTML comment
inside a vendor email. With that sentence removed, the agent does not forward
in 10 of 10 reruns (p = 5e-6).

## Install

```
pip install runtape
```

Requires Python 3.10+. Supports the OpenAI and Anthropic SDKs, LangChain and
LangGraph, OpenAI-compatible local servers (Ollama, LM Studio, vLLM), and
custom agent loops.

## Examples

The repository includes two example agents. Both run offline with a
rule-based stand-in model, passed to `why` with `--model-fn`.

```
git clone https://github.com/RehanMohammed985/runtape
cd runtape
pip install . openai anthropic

python examples/inbox_agent.py
runtape why last tool:forward_email --model-fn examples/inbox_agent.py:simulated_model

python examples/refund_bot.py
runtape why last tool:issue_refund --model-fn examples/refund_bot.py:simulated_model
```

To run the refund example against a local model:

```
ollama pull llama3.1:8b
python examples/refund_bot.py --local llama3.1:8b
runtape why last tool:issue_refund
```

With llama3.2 (3B), the refund agent paid order B-2290 $64, the amount from a
different customer's order earlier in the conversation. On the recorded
context this happened in 9 of 40 reruns. With the earlier order lookup
removed, 0 of 40 (Fisher exact test, p = 0.001).

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

Traces are written to `./traces/`, one file per run, as events happen. Sync,
async and streaming calls are recorded, including Anthropic's
`messages.stream()`, OpenAI's `.stream()` and `.parse()`, and beta endpoints.
For OpenAI-compatible servers, the base URL is recorded so reruns go to the
same server.

## runtape why

```
runtape why <trace> <event>
```

`<trace>` is a path or `last`. `<event>` is an event number, `tool:NAME` (the
last call to that tool), or `last` (the last model response).

Procedure:

1. Rerun the recorded call on the unchanged context to measure how often the
   model makes the same decision.
2. Split the context into pieces: system prompt, messages, tool results.
3. Remove each piece and rerun (2 runs to screen, more where the decision
   changes).
4. Confirm candidates with a one-sided Fisher exact test, Bonferroni-corrected
   over all variants tried.
5. Narrow each confirmed piece to JSON items, paragraphs and sentences.
6. Search inside pieces whose removal has no effect, for causes masked by
   other content in the same piece.
7. Detect causes that are duplicated or individually sufficient.
8. Separate pieces that change the action from pieces the agent only needs as
   input (without them it stops or repeats a lookup).

Only the selected model call is rerun. Tools are not executed.

Options: `--runs` (reruns per variant, default 5), `--budget` (max model
calls, default 400), `--dry` (rank suspects without model calls), `--match
REGEX` (explain a text answer), `--max-pieces`, `--expand`, `--json`.
Replies are cached in `./.runtape/`.

## runtape rerun

![runtape rerun](https://raw.githubusercontent.com/RehanMohammed985/runtape/main/docs/rerun.svg)

```
runtape rerun <trace> <event> --drop "9[1]"
runtape rerun <trace> <event> --replace "old=>new"
runtape rerun <trace> <event> --system-file fixed.txt
runtape rerun <trace> <event> --model <name>
runtape odds  <trace> <event>
```

`rerun` applies an edit to the recorded context and reports the distribution
of decisions. `odds` reports the distribution on the unchanged context.

The same check can be used as a test:

```python
def test_no_forwarding_from_injected_email():
    runtape.rerun("traces/injected.jsonl", 22, drop=["12.body para 4"]).never_calls("forward_email")
```

`always_calls`, `rate(tool)` and `counts()` are also available.

## Replay

```python
with runtape.replay("traces/bad-refund.jsonl") as rp:
    client = rp.wrap(Anthropic())

    @rp.tool
    def lookup_order(order_id): ...     # returns the recorded result

    run_agent(client)
```

Model responses and tool results are served from the trace, so the run is
offline and deterministic. Tool results are restored to their original types
(dataclasses, Pydantic models, tuples). If a request differs from the
recording, replay stops and reports the first difference, or continues
against the live model with `on_diverge="live"`.

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

Other subcommands: `ls`, `summary`, `timeline`, `show`, `context`, `grep`,
`diff`.

## MCP server

```
pip install "runtape[mcp]"
claude mcp add runtape -- runtape mcp
```

Exposes trace inspection, `why` and `rerun` as MCP tools.

<!-- mcp-name: io.github.RehanMohammed985/runtape -->

## Trace format

Append-only JSONL, one event per line. Message history is delta-encoded. See
[SPEC.md](https://github.com/RehanMohammed985/runtape/blob/main/SPEC.md).

Pass `redact=fn` to `runtape.record` to filter events before they are
written. If `fn` raises, the event content is dropped.

## Limitations

- `why` calls the model, typically 100 to 250 times per decision. Use
  `--dry`, `--budget` and `--max-pieces` to limit cost.
- On a simulated model that ignores its context, false positives occurred in
  0 to 5 of 100 runs (alpha = 0.05). A cause that moves the decision rate from
  90% to 10% was found in all runs; 90% to 30%, in about 4 of 5.
- Decisions made in fewer than about 1 in 5 reruns are too rare to attribute
  automatically. Use `odds` and `rerun --drop` instead.
- At temperature 0, each variant is run once and each candidate twice.
- Removed content is replaced with `[content removed]`, which can itself
  affect the model.
- By default, 80 pieces are tested (ranked by word overlap with the decision,
  always including the system prompt, the first user message and the latest
  message), and 6 are searched for masked causes. Untested pieces are listed
  in the report.
- Replay does not serve streamed calls. LangChain runs can be recorded and
  analyzed but not replayed.
- The `--model-fn` cache key includes the function's source file. Use
  `--no-cache` if the function depends on other code that changed.
- Providers other than OpenAI, Anthropic and OpenAI-compatible servers need
  `--model-fn`.

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
