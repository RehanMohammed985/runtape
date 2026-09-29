"""Turn arbitrary Python values into JSON-safe data.

This runs on every event the recorder writes, so it must never raise, never
hang, and never blow up in size: cycles are cut, a node budget caps huge or
heavily shared structures, and objects whose repr() fails are still described.
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import enum
import json
import math
from typing import Any

MAX_DEPTH = 40
MAX_NODES = 5_000_000  # distinct values per payload: large real results are kept whole
MAX_SHARED = 200_000  # values reached again through a shared reference; stops DAG blowups


def _safe_repr(obj: Any) -> str:
    try:
        return repr(obj)[:2000]
    except Exception:
        return f"<unrepresentable {type(obj).__name__}>"


def to_jsonable(obj: Any) -> Any:
    state = {"nodes": 0, "shared": 0}
    stack: set[int] = set()
    done: set[int] = set()  # containers already converted once; meeting one again is a shared reference
    alive: list = []  # keep them referenced so a freed temporary's id can't be reused and misread as shared

    def conv(o: Any, depth: int, again: bool = False) -> Any:
        state["nodes"] += 1
        if again:
            state["shared"] += 1
            if state["shared"] > MAX_SHARED:
                return {"__truncated__": "shared value repeated too many times"}
        if state["nodes"] > MAX_NODES:
            return {"__truncated__": "value too large"}
        if o is None or isinstance(o, (bool, int, str)):
            return o
        if isinstance(o, float):
            return o if math.isfinite(o) else str(o)  # NaN/Infinity aren't valid JSON
        if depth > MAX_DEPTH:
            return {"__repr__": "<max depth>"}
        if isinstance(o, (bytes, bytearray)):
            return {"__bytes__": len(o)}
        if isinstance(o, enum.Enum):
            return conv(o.value, depth + 1)
        if isinstance(o, (_dt.datetime, _dt.date)):
            return o.isoformat()
        oid = id(o)
        if oid in stack:
            return {"__cycle__": type(o).__name__}
        again = again or oid in done
        if oid not in done:
            done.add(oid)
            alive.append(o)
        stack.add(oid)
        try:
            if isinstance(o, dict):
                return {str(k): conv(v, depth + 1, again) for k, v in o.items()}
            if isinstance(o, (list, tuple, set, frozenset)):
                return [conv(v, depth + 1, again) for v in o]
            # pydantic v2 (openai / anthropic response objects)
            dump = getattr(o, "model_dump", None)
            if callable(dump) and not isinstance(o, type):
                try:
                    return conv(dump(mode="json", exclude_unset=False), depth + 1)
                except Exception:
                    pass
            if dataclasses.is_dataclass(o) and not isinstance(o, type):
                try:
                    return conv({f.name: getattr(o, f.name) for f in dataclasses.fields(o)}, depth + 1)
                except Exception:
                    pass
            # SDK "NotGiven" sentinels and similar
            if type(o).__name__ in ("NotGiven", "Omit"):
                return None
            return {"__repr__": _safe_repr(o)}
        finally:
            stack.discard(oid)

    try:
        return conv(obj, 0)
    except Exception as e:  # last resort: never let recording break the agent
        return {"__unserializable__": f"{type(obj).__name__}: {e}"}


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
