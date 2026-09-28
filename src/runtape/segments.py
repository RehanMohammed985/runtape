"""Split a request's context into pieces that can be removed one at a time.

A segment is any span of text the model saw: the system prompt, a user message,
an assistant turn, a tool result, one item of a JSON tool result, a paragraph,
a sentence. Each segment knows which trace event it came from, and can be
removed from a request without making the request invalid for the API.
"""
from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from typing import Any

from .trace import Trace

REMOVED = "[content removed]"
_MIN_CHARS = 2


@dataclass(eq=False)
class Segment:
    base: tuple  # path to a string inside {"system": ..., "messages": [...]}
    kind: str  # system | user | assistant | tool_result
    text: str
    origin: int | None = None  # trace event this came from
    name: str | None = None  # tool name, for tool results
    msg_index: int | None = None
    jpath: tuple | None = None  # path inside the JSON parsed from the base string
    span: tuple[int, int] | None = None  # character range inside the string at base/jpath
    sub: str = ""  # human description of the part: "[1]", ".docs[0]", "para 2", "sentence 3"
    parent: "Segment | None" = field(default=None, repr=False)
    _node: Any = field(default=None, repr=False)  # parsed JSON value for jpath segments

    # ------------------------------------------------------------- labels

    @property
    def where(self) -> str:
        if self.kind == "tool_result":
            head = f"#{self.origin} {self.name or 'tool'} result" if self.origin is not None else f"{self.name or 'tool'} result"
        elif self.kind == "system":
            head = "system prompt"
        else:
            head = f"{self.kind} message [{self.msg_index}]"
            if self.origin is not None:
                head = f"#{self.origin} " + head
        return head + (f" {self.sub}" if self.sub and not self.sub.startswith(("[", ".")) else self.sub)

    def preview(self, limit: int = 80) -> str:
        s = " ".join(self.text.split())
        return s if len(s) <= limit else s[: limit - 3] + "..."

    def depth(self) -> int:
        d, p = 0, self.parent
        while p is not None:
            d, p = d + 1, p.parent
        return d

    # ----------------------------------------------------------- children

    def children(self) -> list["Segment"]:
        """Smaller pieces of this segment: JSON items, then paragraphs, lines, sentences."""
        if self.span is None:
            node = self._node if self.jpath is not None else _parse_json(self.text)
            if node is not None and isinstance(node, (list, dict)):
                kids = self._json_children(node, self.jpath or ())
                if kids:
                    return kids
            if self.jpath is not None and isinstance(node, (list, dict)):
                return []  # a single scalar-free JSON container; nothing smaller to try
        return self._text_children()

    def _json_children(self, node: Any, path: tuple) -> list["Segment"]:
        # collapse single-element wrappers like {"results": [...]} or [[...]]
        while isinstance(node, (list, dict)) and len(node) == 1:
            key = next(iter(node)) if isinstance(node, dict) else 0
            node, path = node[key], path + (key,)
        if isinstance(node, str):
            seg = self._derive(jpath=path, text=node, sub=_fmt_path(path), node=node)
            return seg._text_children() if len(node) > 40 else []
        if not isinstance(node, (list, dict)) or len(node) < 2:
            return []
        items = list(enumerate(node)) if isinstance(node, list) else list(node.items())
        out = []
        for key, val in items:
            text = val if isinstance(val, str) else json.dumps(val, ensure_ascii=False)
            if len(text.strip()) < _MIN_CHARS:
                continue
            out.append(self._derive(jpath=path + (key,), text=text, sub=_fmt_path(path + (key,)), node=val))
        return out

    def _text_children(self) -> list["Segment"]:
        text = self.text
        offset = self.span[0] if self.span else 0
        pieces = _split(text)
        if len(pieces) < 2:
            return []
        unit = pieces[0][2]
        out = []
        for n, (a, b, _) in enumerate(pieces, 1):
            chunk = text[a:b]
            if len(chunk.strip()) < _MIN_CHARS:
                continue
            label = (self.sub + " " if self.sub else "") + f"{unit} {n}"
            out.append(self._derive(span=(offset + a, offset + b), text=chunk, sub=label, jpath=self.jpath, node=None))
        return out

    def _derive(self, *, text: str, sub: str, jpath=None, span=None, node=None) -> "Segment":
        return Segment(
            base=self.base, kind=self.kind, text=text, origin=self.origin, name=self.name,
            msg_index=self.msg_index, jpath=jpath, span=span, sub=sub, parent=self, _node=node,
        )


def _fmt_path(path: tuple) -> str:
    out = ""
    for k in path:
        out += f"[{k}]" if isinstance(k, int) else f".{k}"
    return out


def _parse_json(s: str) -> Any:
    s = s.strip()
    if not s or s[0] not in "[{":
        return None
    try:
        return json.loads(s)
    except ValueError:
        return None


_PARA = re.compile(r"\n\s*\n")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def _split(text: str) -> list[tuple[int, int, str]]:
    """Split into paragraphs, else lines, else sentences. Returns (start, end, unit)."""
    for unit, pattern in (("para", _PARA), ("line", re.compile(r"\n")), ("sentence", _SENT)):
        spans, start = [], 0
        for m in pattern.finditer(text):
            spans.append((start, m.start(), unit))
            start = m.end()
        spans.append((start, len(text), unit))
        spans = [s for s in spans if text[s[0] : s[1]].strip()]
        if len(spans) >= 2:
            return spans
    return []


# --------------------------------------------------------------- extraction


def extract(req: dict, trace: Trace | None = None, request_id: int | None = None) -> list[Segment]:
    """Top-level segments of a request, in context order."""
    origins = _origin_maps(trace, request_id) if trace is not None and request_id is not None else None
    segs: list[Segment] = []

    system = req.get("system")
    if isinstance(system, str) and system.strip():
        segs.append(Segment(("system",), "system", system, origin=origins and origins["system"]))
    elif isinstance(system, list):
        for j, block in enumerate(system):
            if isinstance(block, dict) and isinstance(block.get("text"), str) and block["text"].strip():
                segs.append(Segment(("system", j, "text"), "system", block["text"], origin=origins and origins["system"], sub=f"block {j}"))

    for i, m in enumerate(req.get("messages") or []):
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        mtype = m.get("type")
        intro = origins["intro"][i] if origins and i < len(origins["intro"]) else None

        def add(base, kind, text, call_id=None):
            if not isinstance(text, str) or len(text.strip()) < _MIN_CHARS:
                return
            origin, name = intro, None
            if kind == "tool_result" and origins and call_id in origins["calls"]:
                # ids can repeat across a run; take the latest result recorded before this message
                cands = [c for c in origins["calls"][call_id] if intro is None or c[0] < intro]
                origin, name = (cands or origins["calls"][call_id])[-1]
            elif kind == "assistant" and origins:
                origin = origins["assistant"].get(i, intro)
            segs.append(Segment(base, kind, text, origin=origin, name=name, msg_index=i))

        if mtype == "function_call_output":
            add(("messages", i, "output"), "tool_result", m.get("output"), m.get("call_id"))
            continue
        if mtype == "function_call" or mtype == "reasoning":
            continue
        kind = {"tool": "tool_result", "system": "system", "developer": "system", "assistant": "assistant"}.get(role, "user")
        content = m.get("content")
        if isinstance(content, str):
            add(("messages", i, "content"), kind, content, m.get("tool_call_id"))
        elif isinstance(content, list):
            for j, block in enumerate(content):
                if not isinstance(block, dict):
                    continue
                bt = block.get("type")
                if bt in ("text", "input_text", "output_text"):
                    add(("messages", i, "content", j, "text"), kind, block.get("text"), m.get("tool_call_id"))
                elif bt == "tool_result":
                    inner = block.get("content")
                    if isinstance(inner, str):
                        add(("messages", i, "content", j, "content"), "tool_result", inner, block.get("tool_use_id"))
                    elif isinstance(inner, list):
                        for q, ib in enumerate(inner):
                            if isinstance(ib, dict) and ib.get("type") == "text":
                                add(("messages", i, "content", j, "content", q, "text"), "tool_result", ib.get("text"), block.get("tool_use_id"))
    return segs


def _origin_maps(trace: Trace, request_id: int) -> dict:
    """Where each message of a request came from in the trace."""
    # message index -> request event that first included it (walk the delta chain)
    chain = []
    cur = request_id
    while cur is not None:
        chain.append(cur)
        cur = trace[cur].payload.get("base")
    chain.reverse()
    intro: list[int] = []
    for rid in chain:
        n = len(trace.messages(rid))
        intro.extend([rid] * (n - len(intro)))
    # call id -> (tool_result event, tool name)
    calls: dict[str, list[tuple[int, str]]] = {}
    for e in trace.events:
        if e.type == "tool_call" and e.payload.get("call_id"):
            res = next((c for c in trace.children(e.id) if c.type == "tool_result"), None)
            calls.setdefault(e.payload["call_id"], []).append(((res or e).id, e.payload.get("name")))
    # assistant message index -> the llm_response that produced it
    assistant: dict[int, int] = {}
    for i, rid in enumerate(intro):
        prev = [e.id for e in trace.events if e.type == "llm_response" and e.id < rid]
        if prev:
            assistant[i] = prev[-1]
    system_origin = None
    for e in trace.events:
        if e.id > request_id:
            break
        if e.type == "llm_request" and "system" in e.payload:
            system_origin = e.id
    return {"intro": intro, "calls": calls, "assistant": assistant, "system": system_origin}


# ------------------------------------------------------------------ removal


def ablate(req: dict, segments: list[Segment], replacement: str = REMOVED) -> dict:
    """A copy of the request with the given segments removed.

    Whole messages and tool results are replaced with a short marker (removing
    them outright would break tool call / result pairing). JSON items are
    deleted from their list or object. Text spans are cut out.
    """
    out = copy.deepcopy(req)
    groups: dict[tuple, list[Segment]] = {}
    for s in segments:
        groups.setdefault(s.base, []).append(s)
    for base, segs in groups.items():
        cur = _get(out, base)
        if not isinstance(cur, str):
            continue
        if any(s.jpath is None and s.span is None for s in segs):
            if base == ("system",):
                out["system"] = None
            else:
                _set(out, base, replacement)
            continue
        new = cur
        json_segs = [s for s in segs if s.jpath is not None]
        if json_segs:
            obj = _parse_json(cur)
            if obj is not None:
                obj = _apply_json(obj, json_segs)
                new = json.dumps(obj, ensure_ascii=False)
        text_spans = [s.span for s in segs if s.jpath is None and s.span is not None]
        new = _cut(new, text_spans)
        if not new.strip():
            new = replacement
        _set(out, base, new)
    return out


def _apply_json(obj: Any, segs: list[Segment]) -> Any:
    # cut spans inside JSON string values first, then delete whole items
    by_path: dict[tuple, list] = {}
    deletes = []
    for s in segs:
        if s.span is not None:
            by_path.setdefault(s.jpath, []).append(s.span)
        else:
            deletes.append(s.jpath)
    for path, spans in by_path.items():
        try:
            val = _get(obj, path)
            if isinstance(val, str):
                obj = _set_in(obj, path, _cut(val, spans))
        except (KeyError, IndexError, TypeError):
            pass
    for path in sorted(deletes, key=lambda p: [(0, k) if isinstance(k, int) else (1, str(k)) for k in p], reverse=True):
        if not path:
            continue
        try:
            parent = _get(obj, path[:-1])
            if isinstance(parent, list):
                del parent[path[-1]]
            elif isinstance(parent, dict):
                parent.pop(path[-1], None)
        except (KeyError, IndexError, TypeError):
            pass
    return obj


def _cut(text: str, spans: list[tuple[int, int]]) -> str:
    if not spans:
        return text
    merged: list[list[int]] = []
    for a, b in sorted(spans):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    for a, b in reversed(merged):
        text = text[:a] + text[b:]
    return text


def _get(obj: Any, path: tuple) -> Any:
    for k in path:
        obj = obj[k]
    return obj


def _set(obj: Any, path: tuple, val: Any) -> None:
    _get(obj, path[:-1])[path[-1]] = val


def _set_in(obj: Any, path: tuple, val: Any) -> Any:
    if not path:
        return val
    _set(obj, path, val)
    return obj


# ---------------------------------------------------------------- suspicion

_STOP = set(
    "a an the and or but if then of to in on at by for with from as is are was were be been it its this that these "
    "those i you he she we they me my your our their not no yes do does did so can could would should will just "
    "about into over than too very also any all some such only own same other more most per via id ok true false null "
    "please thanks hi hello let me".split()
)
_WORD = re.compile(r"[a-z0-9$][a-z0-9$.\-]*[a-z0-9]|[a-z0-9]")


def _tokens(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP]


def overlap(segment_text: str, decision_text: str) -> float:
    """How much of the decision's wording appears in a segment. Cheap, no model calls."""
    a, b = _tokens(segment_text), _tokens(decision_text)
    if not a or not b:
        return 0.0
    ua, ub = set(a), set(b)
    ba = set(zip(a, a[1:]))
    bb = set(zip(b, b[1:]))
    shared_bi = len(ba & bb)
    shared_uni = len(ua & ub)
    return (3 * shared_bi + shared_uni) / (len(ub) ** 0.5 + 1)


def find(segs: list[Segment], ref: str) -> list[Segment]:
    """Resolve a reference like '9', '#9[1]', '9[1].text', 'system' to segments.

    A bare event number means every piece that came from that event.
    """
    ref = ref.strip()
    if ref in ("system", "system prompt"):
        return [s for s in segs if s.kind == "system"]
    m = re.match(r"#?(\d+)\s*(.*)$", ref)
    if not m:
        raise ValueError(f"can't read '{ref}': use an event number like 9, or 9[1], or 9[1].text")
    origin, rest = int(m.group(1)), m.group(2).strip()
    top = [s for s in segs if s.origin == origin]
    if not top:
        raise ValueError(f"nothing in this context came from event #{origin}")
    if not rest:
        return top
    frontier = list(top)
    for _ in range(5):
        nxt = []
        for s in frontier:
            for k in s.children():
                if k.sub == rest:
                    return [k]
                nxt.append(k)
        frontier = nxt
    raise ValueError(f"no part '{rest}' inside #{origin}")


def replace_text(req: dict, old: str, new: str) -> tuple[dict, int]:
    """Replace text everywhere the model can read it. Returns (request, occurrences)."""
    out = copy.deepcopy(req)
    count = 0

    def walk(v):
        nonlocal count
        if isinstance(v, str):
            count += v.count(old)
            return v.replace(old, new)
        if isinstance(v, list):
            return [walk(x) for x in v]
        if isinstance(v, dict):
            return {k: (walk(x) if k in ("text", "content", "output") or isinstance(x, (list, dict)) else x) for k, x in v.items()}
        return v

    if isinstance(out.get("system"), (str, list)):
        out["system"] = walk(out["system"])
    out["messages"] = walk(out.get("messages") or [])
    return out, count
