"""MCP server: lets a coding agent (Claude Code, Cursor, ...) debug your agent's runs.

    claude mcp add runtape -- runtape mcp

Then ask it things like "why did my agent issue that refund in the last run?"
It can list traces, read timelines and context windows, search them, and run
why / rerun experiments, and answer with evidence instead of guesses.
"""
from __future__ import annotations

import io
import os
from pathlib import Path

from rich.console import Console

from . import render
from .trace import Trace

INSTRUCTIONS = """runtape records AI agent runs to trace files and can run experiments on them.
To debug why an agent did something: find the trace (list_traces), read the timeline to locate the
decision, then call why on that event. why proves which part of the context caused the decision by
removing pieces and re-running that one decision; it costs model calls, so use why_dry first for a
free guess. Use rerun to test a fix (drop a piece, replace text, new system prompt) on the exact
failing context. Event references: a number, 'last', or 'tool:NAME' for the last call of a tool."""


def _text(renderable, width: int = 120) -> str:
    buf = io.StringIO()
    Console(file=buf, width=width, no_color=True, highlight=False, force_terminal=False).print(renderable)
    return buf.getvalue()


def _trace(trace: str) -> Trace:
    from .cli import resolve_trace

    try:
        return Trace.load(resolve_trace(trace or None))
    except SystemExit as e:  # CLI helpers exit; tools should report
        raise ValueError(str(e.code)) from None


def _event(t: Trace, ref: str | int) -> int:
    from .cli import resolve_event

    try:
        return resolve_event(t, str(ref))
    except SystemExit as e:
        raise ValueError(str(e.code)) from None


def build_server(model_fn: str | None = None):
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("runtape", instructions=INSTRUCTIONS)

    def model():
        if not model_fn:
            return None
        from .rerun import load_model_fn

        return load_model_fn(model_fn)

    @mcp.tool()
    def list_traces(directory: str = "traces") -> str:
        """List recorded traces, newest first, with status and size."""
        from .cli import find_traces

        rows = []
        for p in reversed(find_traces(directory)):
            try:
                s = Trace.load(p).summary()
                rows.append(f"{s['status']:<8} {s['events']:>5} events {s['llm_calls']:>3} model calls  {p}")
            except Exception as e:
                rows.append(f"broken   {p} ({e})")
        return "\n".join(rows) or f"No traces in {directory}."

    @mcp.tool()
    def summary(trace: str = "last") -> str:
        """Totals for a run: status, model calls, tokens, tool calls, errors."""
        return _text(render.show_summary(_trace(trace)))

    @mcp.tool()
    def timeline(trace: str = "last", start: int = 0, end: int | None = None) -> str:
        """One line per event. Use start/end to page through long runs."""
        t = _trace(trace)
        evs = [e for e in t.events if e.id >= start and (end is None or e.id <= end)]
        return _text(render.timeline(t, evs, width=100))

    @mcp.tool()
    def show_event(event: str, trace: str = "last") -> str:
        """Full detail of one event."""
        t = _trace(trace)
        return _text(render.show_event(t, t[_event(t, event)], full=True))

    @mcp.tool()
    def context(event: str, trace: str = "last") -> str:
        """Everything the model had in its context window at an event."""
        t = _trace(trace)
        return _text(render.show_context(t, _event(t, event), full=True))

    @mcp.tool()
    def grep(term: str, trace: str = "last") -> str:
        """Every event containing a term, and where it first entered the run."""
        return _text(render.show_grep(_trace(trace), term))

    @mcp.tool()
    def why_dry(event: str, trace: str = "last") -> str:
        """Free first guess at what caused a decision: context ranked by shared wording. No model calls."""
        from .why import suspects

        t = _trace(trace)
        rows = suspects(t, _event(t, event))
        return "\n".join(f"{sc:5.2f}  {seg.where}  {render.compact(seg.text, 120)}" for sc, seg in rows) or "No suspects."

    @mcp.tool()
    def why(event: str, trace: str = "last", runs: int = 5, budget: int = 150) -> str:
        """Prove which part of the context caused a decision (a tool call, model reply, or model call).
        Re-runs that one decision with pieces removed; costs model calls (capped by budget), cached on disk."""
        from .why import why as run_why

        t = _trace(trace)
        rep = run_why(t, _event(t, event), model=model(), runs=runs, budget=budget)
        return _text(render.show_why(rep, show_all=True))

    @mcp.tool()
    def rerun(
        event: str,
        trace: str = "last",
        runs: int = 5,
        drop: list[str] | None = None,
        replace: dict[str, str] | None = None,
        system: str | None = None,
        model_name: str | None = None,
    ) -> str:
        """Re-run one decision as recorded or with edits, and report what the model does across runs.
        drop: context to remove, e.g. ["9"] or ["9[1]"]. replace: {old: new} text edits.
        system: a new system prompt. Use this to test a fix against the exact failing context."""
        from .rerun import rerun as do_rerun

        t = _trace(trace)
        dist = do_rerun(t, _event(t, event), runs=runs, drop=drop or [], replace=replace, system=system,
                        model_name=model_name, model=model())
        what = "; ".join(dist.notes) if dist.notes else "unchanged context"
        return _text(render.show_distribution(dist, dist.recorded, title=f"rerun ({what}), {len(dist)} runs"))

    return mcp


def serve(model_fn: str | None = None) -> None:
    os.environ.setdefault("RUNTAPE_DIR", "traces")
    build_server(model_fn).run()
