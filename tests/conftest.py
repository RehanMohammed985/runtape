"""Fake HTTP backends so the real OpenAI/Anthropic SDKs run offline."""
import json

import pytest


def _hx(sdk):
    """The HTTP library the SDK itself uses (newer SDKs moved from httpx to httpx2)."""
    base = sdk._base_client
    return getattr(base, "httpx2", None) or getattr(base, "httpx", None) or __import__("httpx")


class Script:
    """Returns queued JSON bodies (or SSE event lists) in order and records requests."""

    def __init__(self, bodies):
        self.bodies = list(bodies)
        self.requests = []

    def handler(self, request):
        hx = __import__(type(request).__module__.split(".")[0])
        self.requests.append(json.loads(request.content))
        body = self.bodies.pop(0)
        if isinstance(body, tuple):  # (status, json)
            return hx.Response(body[0], json=body[1])
        if isinstance(body, list):  # SSE stream
            text = "".join(_sse(ev) for ev in body)
            return hx.Response(200, text=text, headers={"content-type": "text/event-stream"})
        return hx.Response(200, json=body)


def _sse(ev):
    if ev == "[DONE]":
        return "data: [DONE]\n\n"
    head = "" if "choices" in ev else f"event: {ev['type']}\n"
    return f"{head}data: {json.dumps(ev)}\n\n"


def openai_client(script, async_=False):
    import openai

    hx = _hx(openai)
    if async_:
        return openai.AsyncOpenAI(
            api_key="test", http_client=hx.AsyncClient(transport=hx.MockTransport(script.handler))
        )
    return openai.OpenAI(api_key="test", http_client=hx.Client(transport=hx.MockTransport(script.handler)))


def anthropic_client(script, async_=False):
    import anthropic

    hx = _hx(anthropic)

    if async_:
        return anthropic.AsyncAnthropic(
            api_key="test", http_client=hx.AsyncClient(transport=hx.MockTransport(script.handler))
        )
    return anthropic.Anthropic(
        api_key="test", http_client=hx.Client(transport=hx.MockTransport(script.handler))
    )


def oa_chat(content=None, tool_calls=None, finish="stop"):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = [
            {"id": tid, "type": "function", "function": {"name": n, "arguments": json.dumps(a)}}
            for tid, n, a in tool_calls
        ]
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-test",
        "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def an_msg(text=None, tool_uses=None, stop="end_turn"):
    content = []
    if text:
        content.append({"type": "text", "text": text})
    for tid, n, a in tool_uses or []:
        content.append({"type": "tool_use", "id": tid, "name": n, "input": a})
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-test",
        "content": content,
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": {"input_tokens": 20, "output_tokens": 7},
    }


@pytest.fixture
def tpath(tmp_path):
    return tmp_path / "t.jsonl"
