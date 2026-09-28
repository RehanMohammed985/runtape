"""Read a trace file and answer questions about it.

This is what the replay CLI is built on. It is kept separate from the
recorder so other tools can read runtape traces without recording anything.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


@dataclass
class Event:
    id: int
    ts: float
    type: str
    payload: dict
    meta: dict = field(default_factory=dict)
    parent: int | None = None
    raw: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, d: dict) -> "Event":
        return cls(
            id=d["id"],
            ts=d.get("ts", 0.0),
            type=d.get("type", "unknown"),
            payload=d.get("payload") or {},
            meta=d.get("meta") or {},
            parent=d.get("parent"),
            raw=d,
        )

    def text(self) -> str:
        """Everything in this event as one searchable string."""
        return json.dumps(self.raw.get("payload", {}), ensure_ascii=False)


@dataclass
class Context:
    """What the model had in front of it at a point in the run."""

    at: int  # event id asked about
    request_id: int | None  # the llm_request this context comes from
    system: Any = None
    tools: Any = None
    messages: list = field(default_factory=list)
    response: Event | None = None  # the model's reply, if it happened by `at`
    since: list[Event] = field(default_factory=list)  # events after the request, up to `at`


@dataclass
class Hit:
    event: Event
    first: bool  # first event in the run where the term appears
    snippets: list[str]
    paths: list[str] = field(default_factory=list)  # which fields matched, e.g. result[1].text


def _leaves(obj: Any, path: str = "") -> Iterator[tuple[str, str]]:
    """(path, string) for every scalar in a JSON value."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _leaves(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _leaves(v, f"{path}[{i}]")
    elif obj is not None:
        yield path, obj if isinstance(obj, str) else json.dumps(obj)


class Trace:
    def __init__(self, events: list[Event], path: Path | None = None, truncated: bool = False):
        self.events = events
        self.path = path
        self.truncated = truncated  # last line was cut off (crash mid-write)
        self._by_id = {e.id: e for e in events}
        self._msg_cache: dict[int, list] = {}

    # ---------------------------------------------------------------- load

    @classmethod
    def load(cls, path: str | Path) -> "Trace":
        path = Path(path)
        events: list[Event] = []
        truncated = False
        with open(path, encoding="utf-8") as fh:
            lines = fh.read().split("\n")
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                events.append(Event.from_dict(json.loads(line)))
            except (ValueError, KeyError):
                # only the final line may be broken; anything else is corruption
                if any(l.strip() for l in lines[i + 1 :]):
                    raise ValueError(f"{path}: corrupt line {i + 1}")
                truncated = True
        return cls(events, path=path, truncated=truncated)

    # --------------------------------------------------------------- basics

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self) -> Iterator[Event]:
        return iter(self.events)

    def __getitem__(self, eid: int) -> Event:
        return self._by_id[eid]

    @property
    def header(self) -> dict:
        if self.events and self.events[0].type == "run_start":
            return self.events[0].payload
        return {}

    @property
    def status(self) -> str:
        """ok, error, or crashed (no run_end written)."""
        for e in reversed(self.events):
            if e.type == "run_end":
                return e.payload.get("status", "ok")
        return "crashed"

    def of_type(self, *types: str) -> list[Event]:
        return [e for e in self.events if e.type in types]

    def children(self, eid: int) -> list[Event]:
        return [e for e in self.events if e.parent == eid]

    # -------------------------------------------------------------- context

    def messages(self, request_id: int) -> list:
        """Full message list sent in an llm_request, resolving delta encoding."""
        if request_id in self._msg_cache:
            return self._msg_cache[request_id]
        chain = []
        cur: int | None = request_id
        seen = set()
        while cur is not None:
            if cur in seen:
                raise ValueError(f"cycle in base chain at event {cur}")
            seen.add(cur)
            ev = self._by_id[cur]
            if "messages" in ev.payload:
                chain.append(ev.payload["messages"])
                cur = None
            else:
                chain.append(ev.payload.get("messages_append", []))
                cur = ev.payload.get("base")
        out: list = []
        for part in reversed(chain):
            out.extend(part)
        self._msg_cache[request_id] = out
        return out

    def _sticky(self, request_id: int, key: str) -> Any:
        """system/tools are only written when they change; walk back to the last one."""
        provider = self._by_id[request_id].payload.get("provider")
        for e in reversed(self.events):
            if e.id > request_id or e.type != "llm_request":
                continue
            if e.payload.get("provider") == provider and key in e.payload:
                return e.payload[key]
        return None

    def context(self, at: int) -> Context:
        """What the model knew as of event `at`: the latest request at or before it."""
        req = None
        for e in self.events:
            if e.id > at:
                break
            if e.type == "llm_request":
                req = e
        if req is None:
            return Context(at=at, request_id=None, since=[e for e in self.events if e.id <= at])
        resp = next(
            (e for e in self.events if e.type == "llm_response" and e.parent == req.id and e.id <= at),
            None,
        )
        return Context(
            at=at,
            request_id=req.id,
            system=self._sticky(req.id, "system"),
            tools=self._sticky(req.id, "tools"),
            messages=self.messages(req.id),
            response=resp,
            since=[e for e in self.events if req.id < e.id <= at],
        )

    # ----------------------------------------------------------------- grep

    def grep(self, term: str, *, ignore_case: bool = True, width: int = 60) -> list[Hit]:
        """Every event containing `term`. The first hit is where it entered the run."""
        # Searches string values, not the serialized line, so snippets read cleanly.
        # A delta-encoded request only "contains" the term if it's in the new part,
        # so the first hit really is where the term entered.
        needle = term.lower() if ignore_case else term
        hits: list[Hit] = []
        for e in self.events:
            snippets: list[str] = []
            paths: list[str] = []
            for path, val in _leaves(e.payload):
                h = val.lower() if ignore_case else val
                i = h.find(needle)
                if i < 0:
                    continue
                a, b = max(0, i - width), min(len(val), i + len(term) + width)
                snippets.append(("..." if a else "") + val[a:b] + ("..." if b < len(val) else ""))
                paths.append(path)
            if snippets:
                hits.append(Hit(event=e, first=not hits, snippets=snippets[:3], paths=paths))
        return hits

    # ---------------------------------------------------------------- stats

    def summary(self) -> dict:
        tool_counts: dict[str, int] = {}
        tool_errors = 0
        tin = tout = 0
        llm_ms = 0.0
        for e in self.events:
            if e.type == "tool_call":
                n = e.payload.get("name", "?")
                tool_counts[n] = tool_counts.get(n, 0) + 1
            elif e.type == "tool_result" and "error" in e.payload:
                tool_errors += 1
            elif e.type == "llm_response":
                tok = e.meta.get("tokens") or {}
                tin += tok.get("input") or 0
                tout += tok.get("output") or 0
                llm_ms += e.meta.get("latency_ms") or 0
        dur = (self.events[-1].ts - self.events[0].ts) if len(self.events) > 1 else 0
        return {
            "name": self.header.get("name"),
            "run_id": self.header.get("run_id"),
            "status": self.status,
            "events": len(self.events),
            "llm_calls": len(self.of_type("llm_request")),
            "tool_calls": tool_counts,
            "tool_errors": tool_errors,
            "errors": len(self.of_type("error")),
            "tokens": {"input": tin, "output": tout},
            "llm_ms": round(llm_ms, 1),
            "duration_s": round(dur, 3),
            "truncated": self.truncated,
        }
