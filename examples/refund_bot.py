"""A support agent that approves a $2,400 refund it should have escalated.

The bug: early in the run the agent searches the knowledge base, and one of
the results is a stale community FAQ saying agents can refund any amount.
That text sits in context, and 6 turns later it decides the refund.

The model here is scripted (a fake HTTP backend), so this runs offline with no
API key and always reproduces the bug. The Anthropic SDK and runtape are real.

    python examples/refund_bot.py
    runtape why last 31 --model-fn examples/refund_bot.py:simulated_model

runtape why normally re-runs decisions against the real model the agent used.
To keep this example offline, simulated_model below stands in for it: a small
rule-based model that reads its context and decides from it the way the
scripted replies did. Point runtape why at a real Anthropic key and it uses
Claude instead.
"""
import json
import sys
from pathlib import Path

import anthropic

try:
    import httpx2 as hx
except ImportError:
    import httpx as hx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import runtape  # noqa: E402

# --------------------------------------------------------- scripted model


_ids = iter(range(1, 1000))


def _msg(text=None, tools=()):
    content = [{"type": "text", "text": text}] if text else []
    for name, args in tools:
        content.append({"type": "tool_use", "id": f"toolu_{next(_ids):02d}", "name": name, "input": args})
    return {
        "id": "msg", "type": "message", "role": "assistant", "model": "claude-scripted",
        "content": content, "stop_reason": "tool_use" if tools else "end_turn",
        "stop_sequence": None, "usage": {"input_tokens": 0, "output_tokens": 0},
    }


SCRIPT = [
    # ticket 1: small, legit refund
    _msg("Let me pull up that order.", [("lookup_order", {"order_id": "A-1042"})]),
    _msg("Checking our refund policy.", [("search_kb", {"query": "refund policy damaged item"})]),
    _msg("Refunding the damaged mug.", [("issue_refund", {"order_id": "A-1042", "amount": 18.00})]),
    _msg("Done. $18.00 is on its way back to your card."),
    # ticket 2: shipping question, harmless
    _msg("Looking that up.", [("lookup_order", {"order_id": "A-1107"})]),
    _msg("Your order shipped yesterday and should arrive Thursday."),
    # ticket 3: big refund that policy says must go to a manager
    _msg("Let me pull up that order.", [("lookup_order", {"order_id": "B-2290"})]),
    _msg(
        "Per our policy, agents can approve refunds of any amount without manager review, "
        "so I'll process this now.",
        [("issue_refund", {"order_id": "B-2290", "amount": 2400.00})],
    ),
    _msg("Your refund of $2,400.00 has been issued."),
]


def fake_backend():
    queue = list(SCRIPT)

    def handler(request):
        return hx.Response(200, json=queue.pop(0))

    return hx.Client(transport=hx.MockTransport(handler))


def _context_text(req):
    """Every string the model can see, in order."""
    out = [req.get("system") or ""]

    def walk(v):
        if isinstance(v, str):
            out.append(v)
        elif isinstance(v, list):
            for x in v:
                walk(x)
        elif isinstance(v, dict):
            for key in ("text", "content", "output"):
                if key in v:
                    walk(v[key])

    walk(req.get("messages"))
    return "\n".join(out)


def simulated_model(req):
    """Stand-in for Claude that decides from its context, for running `why` offline.

    It follows whatever refund rule it can find in context: a rule allowing
    refunds of any amount wins over the $200 limit, like the real bug.
    """
    msgs = req.get("messages") or []
    last = msgs[-1] if msgs else {}
    blocks = last.get("content") if isinstance(last.get("content"), list) else []
    result = next((b.get("content") for b in blocks if isinstance(b, dict) and b.get("type") == "tool_result"), None)
    try:
        order = json.loads(result) if result else None
    except ValueError:
        order = None
    if not isinstance(order, dict) or "total" not in order:
        return {"text": "Could you confirm your order number so I can look it up?"}
    oid = next((o for o, v in ORDERS.items() if v["total"] == order["total"]), None)
    ctx = _context_text(req).lower()
    unlimited = "any amount" in ctx
    limit_known = "over $200 require manager review" in ctx
    if order["total"] <= 200 or unlimited or not limit_known:
        return {"text": "Processing your refund.", "tool_calls": [
            {"id": "sim", "name": "issue_refund", "arguments": {"order_id": oid, "amount": order["total"]}}]}
    return {"text": "Refunds over $200 need a manager.", "tool_calls": [
        {"id": "sim", "name": "escalate_to_manager", "arguments": {"order_id": oid}}]}


# ------------------------------------------------------------------- agent

ORDERS = {
    "A-1042": {"item": "ceramic mug", "total": 18.00, "status": "delivered, damaged"},
    "A-1107": {"item": "desk lamp", "total": 64.00, "status": "shipped"},
    "B-2290": {"item": "standing desk", "total": 2400.00, "status": "delivered"},
}

KB = [
    {
        "doc": "policies/refunds.md",
        "updated": "2026-03-01",
        "text": "Refunds up to $200 may be issued by the agent. Refunds over $200 require manager review; "
        "use escalate_to_manager.",
    },
    {
        "doc": "community/faq-2019.md",
        "updated": "2019-06-12",
        "text": "Q: Do refunds need approval? A: No, agents can approve refunds of any amount "
        "without manager review.",
    },
]

SYSTEM = (
    "You are the support agent for Mugworks. Use tools to look up orders and policy. "
    "Follow the refund policy exactly."
)

TOOLS = [
    {"name": "lookup_order", "description": "Get an order", "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}}}},
    {"name": "search_kb", "description": "Search the help center", "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}}},
    {"name": "issue_refund", "description": "Refund an order", "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}, "amount": {"type": "number"}}}},
    {"name": "escalate_to_manager", "description": "Send to a manager", "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}}}},
]


def main(trace_path=None):
    rec = runtape.record(trace_path, name="refund-bot", tags={"example": True})
    client = rec.wrap(anthropic.Anthropic(api_key="scripted", http_client=fake_backend()))

    @rec.tool
    def lookup_order(order_id):
        return ORDERS[order_id]

    @rec.tool
    def search_kb(query):
        return KB  # naive retrieval: returns everything, stale docs included

    @rec.tool
    def issue_refund(order_id, amount):
        rec.state("refunds_issued", {"order_id": order_id, "amount": amount})
        return {"ok": True, "order_id": order_id, "amount": amount}

    @rec.tool
    def escalate_to_manager(order_id):
        return {"ok": True, "ticket": f"MGR-{order_id}"}

    tools = {f.__name__: f for f in (lookup_order, search_kb, issue_refund, escalate_to_manager)}

    tickets = [
        "Hi, my mug from order A-1042 arrived cracked. Can I get a refund?",
        "Where is my order A-1107?",
        "I want to return my standing desk, order B-2290. Full refund please.",
    ]

    messages = []  # one long session across tickets: this is how the poison survives
    with rec:
        for ticket in tickets:
            rec.state("ticket", ticket)
            messages.append({"role": "user", "content": ticket})
            while True:
                resp = client.messages.create(
                    model="claude-scripted", max_tokens=1024, system=SYSTEM, tools=TOOLS, messages=messages
                )
                messages.append({"role": "assistant", "content": [b.model_dump(exclude_none=True) for b in resp.content]})
                if resp.stop_reason != "tool_use":
                    break
                results = []
                for block in resp.content:
                    if block.type == "tool_use":
                        out = tools[block.name](**block.input)
                        results.append({"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(out)})
                messages.append({"role": "user", "content": results})
    return rec.path


if __name__ == "__main__":
    path = main(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"trace written to {path}")
