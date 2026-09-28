"""runtape command line: inspect and replay trace files."""
from __future__ import annotations

import argparse
import cmd
import os
import shlex
import sys
from pathlib import Path

from rich.console import Console
from rich.text import Text

from . import __version__, render
from .trace import Trace

DEFAULT_DIR = os.environ.get("RUNTAPE_DIR", "traces")

EVENT_TYPES = (
    "llm_request", "llm_response", "tool_call", "tool_result", "error", "state", "log", "run_start", "run_end",
)
TYPE_ALIASES = {
    "llm": "llm_request", "request": "llm_request", "req": "llm_request",
    "response": "llm_response", "resp": "llm_response", "reply": "llm_response",
    "tool": "tool_call", "call": "tool_call", "result": "tool_result",
    "err": "error", "errors": "error",
}


# ------------------------------------------------------------ resolution


def find_traces(directory: str | Path = DEFAULT_DIR) -> list[Path]:
    d = Path(directory)
    if not d.is_dir():
        return []
    return sorted(d.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)


def resolve_trace(arg: str | None) -> Path:
    """A path, or 'last'/None for the newest trace in ./traces."""
    if arg and arg != "last":
        p = Path(arg)
        if p.is_file():
            return p
        # allow a unique substring of a trace name, like a run id
        matches = [t for t in find_traces() if arg in t.name]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise SystemExit(f"'{arg}' matches {len(matches)} traces; be more specific.")
        raise SystemExit(f"No trace file '{arg}'.")
    traces = find_traces()
    if not traces:
        raise SystemExit(f"No traces in ./{DEFAULT_DIR}. Pass a path, or record a run first.")
    return traces[-1]


def resolve_event(trace: Trace, arg: str | int) -> int:
    try:
        n = int(arg)
    except (TypeError, ValueError):
        if arg in ("last", "end", "$"):
            return trace.events[-1].id
        if arg in ("first", "start"):
            return trace.events[0].id
        raise SystemExit(f"'{arg}' is not an event number.")
    if n < 0:
        n = trace.events[n].id
    if n not in {e.id for e in trace.events}:
        raise SystemExit(f"No event #{n}. Trace has events 0-{trace.events[-1].id}.")
    return n


# ------------------------------------------------------------ interactive


class Replay(cmd.Cmd):
    intro = ""
    doc_header = "Commands (type help <command> for details)"

    def __init__(self, trace: Trace, console: Console, start: int = 0):
        super().__init__()
        self.trace = trace
        self.c = console
        self.ids = [e.id for e in trace.events]
        self.pos = self.ids.index(start) if start in self.ids else 0
        self.term: str | None = None  # last grep term, highlighted everywhere
        self.last_cmd_was_move = True

    # -- helpers

    @property
    def cur(self) -> int:
        return self.ids[self.pos]

    @property
    def prompt(self) -> str:  # type: ignore[override]
        return f"(runtape #{self.cur}/{self.ids[-1]}) "

    def show_current(self) -> None:
        self.c.print(render.show_event(self.trace, self.trace[self.cur], highlight=self.term))

    def _event_arg(self, arg: str) -> int | None:
        try:
            return resolve_event(self.trace, arg)
        except SystemExit as e:
            self.c.print(Text(str(e), style="red"))
            return None

    def _move(self, arg: str, direction: int) -> None:
        arg = arg.strip()
        if arg and not arg.lstrip("-").isdigit():
            want = TYPE_ALIASES.get(arg, arg)
            if want not in EVENT_TYPES:
                self.c.print(Text(f"Unknown event type '{arg}'. Types: {', '.join(EVENT_TYPES)}", style="red"))
                return
            i = self.pos + direction
            while 0 <= i < len(self.ids):
                if self.trace[self.ids[i]].type == want:
                    self.pos = i
                    self.show_current()
                    return
                i += direction
            self.c.print(Text(f"No {want} {'after' if direction > 0 else 'before'} #{self.cur}.", style="dim"))
            return
        n = int(arg) if arg else 1
        new = max(0, min(len(self.ids) - 1, self.pos + direction * n))
        if new == self.pos:
            self.c.print(Text("At the end of the trace." if direction > 0 else "At the start of the trace.", style="dim"))
            return
        self.pos = new
        self.show_current()

    @staticmethod
    def _flags(arg: str) -> tuple[list[str], set[str]]:
        try:
            parts = shlex.split(arg)
        except ValueError:
            parts = arg.split()
        return [p for p in parts if not p.startswith("--")], {p for p in parts if p.startswith("--")}

    # -- cmd plumbing

    def emptyline(self) -> bool:
        """Enter repeats a step, like a debugger."""
        self._move("", 1)
        return False

    def default(self, line: str) -> bool:
        if line.strip().lstrip("-").isdigit():
            self.do_goto(line)
            return False
        self.c.print(Text(f"Unknown command '{line}'. Type help.", style="red"))
        return False

    def do_EOF(self, arg: str) -> bool:
        self.c.print()
        return True

    # -- movement

    def do_step(self, arg: str) -> None:
        """step [N | TYPE]   Move forward N events, or to the next event of TYPE (e.g. step tool_call). Enter also steps."""
        self._move(arg, 1)

    def do_back(self, arg: str) -> None:
        """back [N | TYPE]   Move backward N events, or to the previous event of TYPE."""
        self._move(arg, -1)

    def do_goto(self, arg: str) -> None:
        """goto N   Jump to event N. Also: goto first, goto last, or just type the number."""
        if not arg.strip():
            self.c.print(Text("Usage: goto N", style="red"))
            return
        n = self._event_arg(arg.strip())
        if n is not None:
            self.pos = self.ids.index(n)
            self.show_current()

    def do_first(self, arg: str) -> None:
        """first   Jump to the first event."""
        self.pos = 0
        self.show_current()

    def do_last(self, arg: str) -> None:
        """last   Jump to the last event."""
        self.pos = len(self.ids) - 1
        self.show_current()

    # -- inspection

    def do_show(self, arg: str) -> None:
        """show [N] [--raw] [--full]   Show an event in detail (default: current). --raw prints the JSON line."""
        pos, flags = self._flags(arg)
        n = self._event_arg(pos[0]) if pos else self.cur
        if n is not None:
            self.c.print(
                render.show_event(self.trace, self.trace[n], raw="--raw" in flags, full="--full" in flags, highlight=self.term)
            )

    def do_context(self, arg: str) -> None:
        """context [N] [--full]   Everything the model had in its context window at event N (default: current)."""
        pos, flags = self._flags(arg)
        n = self._event_arg(pos[0]) if pos else self.cur
        if n is not None:
            self.c.print(render.show_context(self.trace, n, full="--full" in flags, highlight=self.term))

    def do_list(self, arg: str) -> None:
        """list [N | all]   Timeline of events around the current one (N either side, default 8), or the whole run."""
        arg = arg.strip()
        if arg == "all":
            evs = self.trace.events
        else:
            w = int(arg) if arg.isdigit() else 8
            evs = self.trace.events[max(0, self.pos - w) : self.pos + w + 1]
        self.c.print(render.timeline(self.trace, evs, cursor=self.cur))

    def do_grep(self, arg: str) -> None:
        """grep TERM   Find every event containing TERM. The first hit is where it entered the run.
        After a grep, the term is highlighted everywhere and next/prev jump between hits. grep with no term clears it."""
        term = arg.strip().strip('"').strip("'")
        if not term:
            self.term = None
            self.c.print(Text("Highlight cleared.", style="dim"))
            return
        self.term = term
        self.c.print(render.show_grep(self.trace, term))

    def _hit(self, direction: int) -> None:
        if not self.term:
            self.c.print(Text("No search yet. Use grep TERM first.", style="dim"))
            return
        ids = [h.event.id for h in self.trace.grep(self.term)]
        cand = [i for i in ids if (i > self.cur if direction > 0 else i < self.cur)]
        if not cand:
            self.c.print(Text(f'No more hits for "{self.term}" {"after" if direction > 0 else "before"} #{self.cur}.', style="dim"))
            return
        self.pos = self.ids.index(cand[0] if direction > 0 else cand[-1])
        self.show_current()

    def do_next(self, arg: str) -> None:
        """next   Jump to the next event matching the last grep."""
        self._hit(1)

    def do_prev(self, arg: str) -> None:
        """prev   Jump to the previous event matching the last grep."""
        self._hit(-1)

    def do_diff(self, arg: str) -> None:
        """diff [A] [B]   What changed between two events: events in between and how the model context changed.
        No args: from the previous model call to the current event. One arg: from A to the current event."""
        pos, _ = self._flags(arg)
        if len(pos) >= 2:
            a, b = self._event_arg(pos[0]), self._event_arg(pos[1])
        elif len(pos) == 1:
            a, b = self._event_arg(pos[0]), self.cur
        else:
            b = self.cur
            here = self.trace.context(b).request_id
            prev = [e.id for e in self.trace.of_type("llm_request") if here is not None and e.id < here]
            a = prev[-1] if prev else 0
        if a is not None and b is not None:
            self.c.print(render.show_diff(self.trace, a, b, highlight=self.term))

    def do_summary(self, arg: str) -> None:
        """summary   Totals for the run: model calls, tokens, tool calls, errors."""
        self.c.print(render.show_summary(self.trace))

    def do_errors(self, arg: str) -> None:
        """errors   List every error and failed tool call."""
        errs = [e for e in self.trace.events if e.type == "error" or (e.type == "tool_result" and "error" in e.payload)]
        if not errs:
            self.c.print(Text("No errors.", style="dim"))
        else:
            self.c.print(render.timeline(self.trace, errs, cursor=self.cur))

    def do_quit(self, arg: str) -> bool:
        """quit   Exit."""
        return True

    # short aliases
    do_s = do_step
    do_b = do_back
    do_g = do_goto
    do_p = do_show
    do_ctx = do_context
    do_l = do_list
    do_n = do_next
    do_q = do_quit
    do_exit = do_quit

    def get_names(self):
        # keep aliases out of the help listing
        hidden = {"do_s", "do_b", "do_g", "do_p", "do_ctx", "do_l", "do_n", "do_q", "do_exit", "do_EOF"}
        return [n for n in super().get_names() if n not in hidden]


# ------------------------------------------------------------------ main


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="runtape",
        description="Inspect and replay runtape agent traces. TRACE is a file path, part of a file name, "
        "or omitted/'last' for the newest trace in ./traces.",
    )
    p.add_argument("--version", action="version", version=f"runtape {__version__}")
    p.add_argument("--no-color", action="store_true", help="plain output")
    sub = p.add_subparsers(dest="cmd")

    s = sub.add_parser("replay", help="step through a trace interactively (default)")
    s.add_argument("trace", nargs="?")
    s.add_argument("--at", default="0", help="event to start at")

    s = sub.add_parser("ls", help="list recorded traces")
    s.add_argument("dir", nargs="?", default=DEFAULT_DIR)

    s = sub.add_parser("summary", help="totals for a run")
    s.add_argument("trace", nargs="?")

    s = sub.add_parser("timeline", help="one line per event")
    s.add_argument("trace", nargs="?")

    s = sub.add_parser("show", help="one event in detail")
    s.add_argument("trace")
    s.add_argument("event")
    s.add_argument("--raw", action="store_true")
    s.add_argument("--full", action="store_true")

    s = sub.add_parser("context", help="what the model saw at an event")
    s.add_argument("trace")
    s.add_argument("event")
    s.add_argument("--full", action="store_true")

    s = sub.add_parser("grep", help="find where a term appears and where it first entered")
    s.add_argument("trace")
    s.add_argument("term")

    s = sub.add_parser("diff", help="what changed between two events")
    s.add_argument("trace")
    s.add_argument("a")
    s.add_argument("b")
    return p


def main(argv: list[str] | None = None, console: Console | None = None) -> int:
    args = build_parser().parse_args(argv)
    c = console or Console(no_color=args.no_color, highlight=False)
    cmd_name = args.cmd or "replay"

    try:
        if cmd_name == "ls":
            traces = find_traces(args.dir)
            if not traces:
                c.print(Text(f"No traces in {args.dir}.", style="dim"))
                return 0
            for path in reversed(traces):
                try:
                    s = Trace.load(path).summary()
                    style = {"ok": "green", "error": "red", "crashed": "red"}.get(s["status"], "")
                    row = Text(f"{s['status']:<8}", style=style)
                    row.append(f"{s['events']:>5} events  {s['llm_calls']:>3} llm  ")
                    row.append(str(path), style="bold")
                except Exception as e:  # unreadable file shouldn't hide the rest
                    row = Text(f"{'broken':<8}{str(path)}  ({e})", style="red")
                c.print(row)
            return 0

        trace = Trace.load(resolve_trace(getattr(args, "trace", None)))
        if not trace.events:
            c.print(Text("Trace is empty.", style="red"))
            return 1

        if cmd_name == "summary":
            c.print(render.show_summary(trace))
        elif cmd_name == "timeline":
            c.print(render.timeline(trace, trace.events))
        elif cmd_name == "show":
            n = resolve_event(trace, args.event)
            c.print(render.show_event(trace, trace[n], raw=args.raw, full=args.full))
        elif cmd_name == "context":
            c.print(render.show_context(trace, resolve_event(trace, args.event), full=args.full))
        elif cmd_name == "grep":
            c.print(render.show_grep(trace, args.term))
        elif cmd_name == "diff":
            c.print(render.show_diff(trace, resolve_event(trace, args.a), resolve_event(trace, args.b)))
        elif cmd_name == "replay":
            start = resolve_event(trace, args.at) if hasattr(args, "at") else 0
            r = Replay(trace, c, start=start)
            c.print(render.show_summary(trace))
            c.print(Text("Enter = step.  back, goto N, context, grep TERM, diff, list, help, quit.", style="dim"))
            r.show_current()
            try:
                r.cmdloop()
            except KeyboardInterrupt:
                c.print()
    except SystemExit as e:
        if isinstance(e.code, str):
            c.print(Text(e.code, style="red"))
            return 1
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
