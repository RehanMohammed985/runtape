"""Turn arbitrary Python values into JSON-safe data."""
from __future__ import annotations

import dataclasses
import datetime as _dt
import enum
import json
from typing import Any

_MAX_DEPTH = 40


def to_jsonable(obj: Any, _depth: int = 0) -> Any:
    if _depth > _MAX_DEPTH:
        return {"__repr__": "<max depth>"}
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    d = _depth + 1
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v, d) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v, d) for v in obj]
    if isinstance(obj, (bytes, bytearray)):
        return {"__bytes__": len(obj)}
    if isinstance(obj, enum.Enum):
        return to_jsonable(obj.value, d)
    if isinstance(obj, (_dt.datetime, _dt.date)):
        return obj.isoformat()
    # pydantic v2 (openai / anthropic response objects)
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        try:
            return to_jsonable(dump(mode="json", exclude_unset=False), d)
        except Exception:
            pass
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return to_jsonable(dataclasses.asdict(obj), d)
    # SDK "NotGiven" sentinels and similar
    if type(obj).__name__ in ("NotGiven", "Omit"):
        return None
    return {"__repr__": repr(obj)[:2000]}


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
