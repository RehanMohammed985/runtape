"""Dollars spent on Claude requests by a benchmark run, with a limit (--max-dollars).

Every request goes through the Anthropic SDK's Messages.create, so that is where it's counted: the agent's
calls, reruns, the judge, live runs. Cached replies make no request and cost nothing. The total is kept in a
file, so a resumed run keeps counting from where it stopped.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

# $ per million tokens (input, output), for --max-dollars. Cache writes cost 1.25x input, cache reads 0.1x.
PRICES = {"claude-haiku-4-5": (1.0, 5.0), "claude-3-5-haiku": (0.8, 4.0), "claude-sonnet-5": (2.0, 10.0),
          "claude-opus-5": (4.0, 20.0)}


class SpendLimit(BaseException):
    """Raised before a request once --max-dollars is spent. A BaseException, so it passes through the
    handlers that record a broken agent run, and out of the worker threads reruns are made in."""


class Spend:
    """Dollars spent on Anthropic requests by this benchmark, kept in a file so a resumed run keeps counting.
    Every request goes through the SDK's Messages.create, so that is where it's counted; cached reruns make no
    request and cost nothing."""

    def __init__(self, path: Path, model: str, limit: float | None):
        self.path, self.limit = path, limit
        self.price = next((v for k, v in PRICES.items() if model.startswith(k)), None)
        if limit is not None and self.price is None:
            raise SystemExit(f"--max-dollars doesn't know the price of {model}; add it to PRICES in run.py.")
        self.total = json.loads(path.read_text()).get("dollars", 0.0) if path.exists() else 0.0
        self._lock = threading.Lock()

    def check(self) -> None:
        if self.limit is not None and self.total >= self.limit:
            raise SpendLimit(f"${self.total:.2f} spent, the --max-dollars limit is ${self.limit:g}")

    def add(self, usage) -> None:
        if self.price is None or usage is None:
            return
        pin, pout = self.price
        tokens_in = (getattr(usage, "input_tokens", 0) or 0) + 1.25 * (getattr(usage, "cache_creation_input_tokens", 0)
                                                                        or 0) + 0.1 * (getattr(usage, "cache_read_input_tokens", 0) or 0)
        cost = (tokens_in * pin + (getattr(usage, "output_tokens", 0) or 0) * pout) / 1e6
        with self._lock:
            self.total += cost
            self.path.write_text(json.dumps({"dollars": round(self.total, 4)}))


def count_spend(spend: Spend) -> None:
    """Count every Anthropic request (the agent's, the live fix runs', the reruns', the judge's) against spend."""
    from anthropic.resources.messages import Messages

    original = Messages.create

    def create(self, *args, **kw):
        spend.check()
        out = original(self, *args, **kw)
        spend.add(getattr(out, "usage", None))
        return out

    Messages.create = create
