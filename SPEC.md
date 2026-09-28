# runtape trace format, v1

A trace is one UTF-8 JSONL file per run. One JSON object per line. Lines are
only ever appended, never rewritten, so a process that dies mid-run still
leaves a readable file. Readers must ignore a truncated final line.

## Event envelope

Every line has these fields:

| field     | type            | notes                                                     |
|-----------|-----------------|-----------------------------------------------------------|
| id        | int             | 0-based, increments by 1, unique within the file          |
| ts        | float           | unix seconds                                              |
| type      | string          | see event types below                                     |
| payload   | object          | type-specific content                                     |
| meta      | object          | optional: latency_ms, tokens, model, provider, etc.       |
| parent    | int or null     | id of the event this one belongs to (e.g. result -> call) |

Unknown fields must be preserved by tools and ignored by readers.

## Event types

### run_start (always id 0)
payload: run_id, format ("runtape"), version (int, currently 1), name, started_at (ISO 8601), python, argv, tags (object).

### run_end
payload: status ("ok" or "error"), duration_ms. Absent if the process crashed.

### llm_request
payload:
- provider: "openai" | "anthropic" | other string
- model
- system: string or list, present only if it changed since the previous request
- tools: list, present only if it changed since the previous request
- messages: full message list, OR
- base + messages_append: base is the id of an earlier llm_request whose
  resolved message list is a prefix of this one; messages_append holds the rest
- params: other request args (temperature, max_tokens, ...)

To resolve an llm_request's full messages, follow base recursively and
concatenate. To resolve system/tools, walk back to the most recent
llm_request that set them.

### llm_response
parent: the llm_request id.
payload:
- text: concatenated text output, or null
- tool_calls: list of {id, name, arguments} normalized across providers
- stop_reason
- raw: provider response object as JSON
meta: latency_ms, tokens {input, output}

### tool_call
payload: name, arguments (object)

### tool_result
parent: the tool_call id.
payload: name, result (any JSON), or error {type, message, traceback}
meta: latency_ms

### error
payload: type, message, traceback. parent points at the event that failed, if any.

### state
Free-form: memory writes, plan updates, anything. payload: key, value.

### log
Free-form note. payload: message, plus anything.

## Values

Non-JSON values are converted: pydantic models via model_dump, dataclasses via
asdict, bytes as {"__bytes__": len}, anything else via repr() as
{"__repr__": "..."}.
