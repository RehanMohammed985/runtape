"""Run a small support agent with a real OpenTelemetry instrumentation against a local stand-in OpenAI server,
and write the spans as the SDK's console exporter prints them and as OTLP JSON."""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

which = sys.argv[1]
out_dir = sys.argv[2]
os.environ.setdefault("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true")
os.environ.setdefault("TRACELOOP_TRACE_CONTENT", "true")
os.environ["OPENAI_API_KEY"] = "test"

from opentelemetry import trace  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # noqa: E402

exp = InMemorySpanExporter()
tp = TracerProvider()
tp.add_span_processor(SimpleSpanProcessor(exp))
trace.set_tracer_provider(tp)

if which == "openllmetry":
    from opentelemetry.instrumentation.openai import OpenAIInstrumentor
    OpenAIInstrumentor().instrument()
elif which == "openinference":
    from openinference.instrumentation.openai import OpenAIInstrumentor
    OpenAIInstrumentor().instrument(tracer_provider=tp)
elif which == "otel-v2":
    from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor
    OpenAIInstrumentor().instrument(tracer_provider=tp)

import openai  # noqa: E402

KB = [{"doc": "policy.md", "text": "Refunds over $200 require manager review."},
      {"doc": "faq-2019.md", "text": "Agents can approve refunds of any amount."}]


def reply(messages):
    tools_done = [m for m in messages if m.get("role") == "tool"]
    if len(tools_done) == 0:
        return ("lookup_order", {"order_id": "Z-9"})
    if len(tools_done) == 1:
        return ("search_kb", {"q": "refund"})
    return ("issue_refund", {"order_id": "Z-9", "amount": 900})


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        name, args = reply(body["messages"])
        n = len([m for m in body["messages"] if m.get("role") == "tool"])
        msg = {"role": "assistant", "content": None, "tool_calls": [
            {"id": f"call_{n}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}
        out = {"id": f"chatcmpl-{n}", "object": "chat.completion", "created": 0, "model": body["model"],
               "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls"}],
               "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


srv = HTTPServer(("127.0.0.1", 0), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
client = openai.OpenAI(base_url=f"http://127.0.0.1:{srv.server_port}/v1")


def fn(name, desc, props):
    return {"type": "function", "function": {"name": name, "description": desc,
                                             "parameters": {"type": "object", "properties": props}}}


TOOLS = [fn("lookup_order", "Look up an order", {"order_id": {"type": "string"}}),
         fn("search_kb", "Search the help docs", {"q": {"type": "string"}}),
         fn("issue_refund", "Refund an order", {"order_id": {"type": "string"}, "amount": {"type": "number"}}),
         fn("escalate_to_manager", "Hand to a manager", {"order_id": {"type": "string"}})]
RESULTS = {"lookup_order": {"order_id": "Z-9", "total": 900}, "search_kb": KB}

messages = [{"role": "system", "content": "You are support. Follow policy."},
            {"role": "user", "content": "Refund my order Z-9 please."}]
tracer = trace.get_tracer("agent")
with tracer.start_as_current_span("support-agent"):
    for _ in range(3):
        r = client.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=TOOLS, temperature=0.7)
        m = r.choices[0].message
        tc = m.tool_calls[0]
        if tc.function.name == "issue_refund":
            break
        messages.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}]})
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(RESULTS[tc.function.name])})

spans = exp.get_finished_spans()
with open(os.path.join(out_dir, f"{which}-console.json"), "w") as f:
    json.dump([json.loads(s.to_json()) for s in spans], f)
try:
    from google.protobuf.json_format import MessageToDict
    from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
    with open(os.path.join(out_dir, f"{which}-otlp.json"), "w") as f:
        json.dump(MessageToDict(encode_spans(spans)), f)
except Exception as e:  # noqa: BLE001
    print("otlp encode failed", e)
print(which, len(spans), "spans:", [s.name for s in spans])
for s in spans:
    keys = sorted(k for k in (s.attributes or {}))
    print("  ", s.name, len(keys), "attrs;", [k for k in keys][:12], "... events:", [e.name for e in s.events][:6])
