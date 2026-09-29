"""Tiny harness: an OpenAI-SDK agent loop over a MockTransport answered by a model function."""
import json
import sys

sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "..", "src"))
try:
    import httpx2 as httpx  # noqa
except ImportError:
    import httpx  # noqa
import openai  # noqa
import runtape  # noqa


def tool_results(msgs):
    names, out = {}, []
    for m in msgs:
        for tc in m.get("tool_calls") or []:
            names[tc.get("id")] = (tc.get("function") or {}).get("name")
        if m.get("role") == "tool":
            try:
                val = json.loads(m.get("content") or "")
            except ValueError:
                val = m.get("content")
            out.append((names.get(m.get("tool_call_id")), val))
    return out


def called(msgs):
    return [(tc.get("function") or {}).get("name") for m in msgs for tc in (m.get("tool_calls") or [])]


def ctx(req):
    return json.dumps(req.get("messages") or []).lower()


def call(name, **args):
    return {"tool_calls": [{"id": f"sim_{name}", "name": name, "arguments": args}]}


def backend(model_fn):
    def handler(request):
        body = json.loads(request.content)
        out = model_fn({"messages": body["messages"]})
        if isinstance(out, str):
            out = {"text": out}
        msg = {"role": "assistant", "content": out.get("text")}
        if out.get("tool_calls"):
            msg["tool_calls"] = [{"id": f"call_{len(body['messages'])}_{i}", "type": "function",
                                  "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])}}
                                 for i, tc in enumerate(out["tool_calls"])]
        return httpx.Response(200, json={
            "id": "sim", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "message": msg,
                         "finish_reason": "tool_calls" if out.get("tool_calls") else "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}})
    return httpx.Client(transport=httpx.MockTransport(handler))


def run(path, model_fn, tools, system, users, max_steps=80, temperature=None):
    rec = runtape.record(path, name="t")
    client = rec.wrap(openai.OpenAI(api_key="x", http_client=backend(model_fn)))
    wrapped = {n: rec.tool(name=n)(f) for n, f in tools.items()}
    schema = [{"type": "function", "function": {"name": n, "description": n, "parameters": {"type": "object", "properties": {}}}}
              for n in tools]
    messages = [{"role": "system", "content": system}] if system else []
    kw = {} if temperature is None else {"temperature": temperature}
    with rec:
        for u in users:
            messages.append({"role": "user", "content": u})
            for _ in range(max_steps):
                resp = client.chat.completions.create(model="sim", messages=messages, tools=schema, **kw)
                msg = resp.choices[0].message
                messages.append(msg.model_dump(exclude_none=True))
                if not msg.tool_calls:
                    break
                for tc in msg.tool_calls:
                    out = wrapped[tc.function.name](**json.loads(tc.function.arguments or "{}"))
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(out)})
    return rec.path
