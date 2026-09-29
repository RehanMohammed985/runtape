"""A minimal tool-calling agent loop shared by the examples.

Works with OpenAI (Chat Completions), Anthropic (Messages), and any
OpenAI-compatible local server such as Ollama. Tools are plain Python
functions recorded with rec.tool; their schemas are given in OpenAI format and
converted for Anthropic.
"""
import json
import os

PROVIDERS = ("openai", "anthropic", "ollama")


def make_client(rec, provider: str, *, local_url: str = "http://localhost:11434/v1", http_client=None):
    """A recorded SDK client for the provider."""
    if provider == "anthropic":
        import anthropic

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("Set ANTHROPIC_API_KEY to use --anthropic.")
        return rec.wrap(anthropic.Anthropic())
    import openai

    if provider == "ollama":
        return rec.wrap(openai.OpenAI(base_url=local_url, api_key="ollama"))
    if provider == "simulated":
        return rec.wrap(openai.OpenAI(api_key="simulated", http_client=http_client))
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY to use --openai.")
    return rec.wrap(openai.OpenAI())


def _anthropic_tools(tools):
    return [{"name": t["function"]["name"], "description": t["function"].get("description", ""),
             "input_schema": t["function"].get("parameters") or {"type": "object", "properties": {}}} for t in tools]


def _call(impls, name, args):
    try:
        return impls[name](**(args or {}))
    except Exception as e:  # models sometimes send bad arguments; report it like a real tool would
        return {"error": f"{type(e).__name__}: {e}"}


def run(client, provider: str, model: str, *, system: str, task: str, tools: list, impls: dict,
        max_turns: int = 12, max_tokens: int = 1024) -> list:
    """Run the agent until it stops calling tools. Returns the final message list."""
    if provider == "anthropic":
        msgs = [{"role": "user", "content": task}]
        atools = _anthropic_tools(tools)
        for _ in range(max_turns):
            resp = client.messages.create(model=model, system=system, messages=msgs, tools=atools,
                                          max_tokens=max_tokens)
            msgs.append({"role": "assistant", "content": [b.model_dump(exclude_none=True) for b in resp.content]})
            uses = [b for b in resp.content if b.type == "tool_use"]
            if not uses:
                break
            msgs.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": b.id, "content": json.dumps(_call(impls, b.name, b.input))}
                for b in uses]})
        return msgs

    msgs = [{"role": "system", "content": system}, {"role": "user", "content": task}]
    for _ in range(max_turns):
        resp = client.chat.completions.create(model=model, messages=msgs, tools=tools)
        msg = resp.choices[0].message
        msgs.append(msg.model_dump(exclude_none=True))
        if not msg.tool_calls:
            break
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except ValueError:
                args = None
            out = _call(impls, tc.function.name, args) if args is not None else {"error": "arguments were not valid JSON"}
            msgs.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(out)})
    return msgs


def simulated_backend(model_fn):
    """An in-process OpenAI-compatible HTTP backend answered by model_fn, for offline runs."""
    try:  # newer SDKs use httpx2
        import httpx2 as httpx
    except ImportError:
        import httpx

    def handler(request):
        body = json.loads(request.content)
        out = model_fn({"messages": body["messages"]})
        msg = {"role": "assistant", "content": out.get("text")}
        if out.get("tool_calls"):
            msg["tool_calls"] = [{"id": f"call_{len(body['messages'])}_{i}", "type": "function",
                                  "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])}}
                                 for i, tc in enumerate(out["tool_calls"])]
        return httpx.Response(200, json={
            "id": "sim", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if out.get("tool_calls") else "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}})

    return httpx.Client(transport=httpx.MockTransport(handler))


def tool_results(messages):
    """[(tool name, parsed result)] in order, from OpenAI-format messages."""
    names, out = {}, []
    for m in messages:
        for tc in m.get("tool_calls") or []:
            names[tc.get("id")] = (tc.get("function") or {}).get("name")
        if m.get("role") == "tool":
            try:
                val = json.loads(m.get("content") or "")
            except ValueError:
                val = m.get("content")
            out.append((names.get(m.get("tool_call_id")), val))
    return out


def add_provider_args(ap):
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--openai", metavar="MODEL", help="use an OpenAI model, e.g. gpt-4o-mini (needs OPENAI_API_KEY)")
    g.add_argument("--anthropic", metavar="MODEL",
                   help="use a Claude model, e.g. claude-haiku-4-5 (needs ANTHROPIC_API_KEY)")
    g.add_argument("--local", metavar="MODEL", help="free: a local model through Ollama, e.g. llama3.1:8b")
    ap.add_argument("--local-url", default="http://localhost:11434/v1", help="OpenAI-compatible server for --local")


def provider_from_args(a) -> tuple[str, str]:
    """(provider, model); ('simulated', 'simulated') when no model was chosen."""
    if a.openai:
        return "openai", a.openai
    if a.anthropic:
        return "anthropic", a.anthropic
    if a.local:
        return "ollama", a.local
    return "simulated", "simulated"
