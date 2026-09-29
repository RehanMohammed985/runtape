"""runtape command line: inspect and replay trace files."""
from __future__ import annotations

import argparse
import cmd
import json
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


def resolve_decision(trace: Trace, arg: str | int) -> int:
    """Like resolve_event, but 'last' means the last model decision rather than run_end."""
    if isinstance(arg, str) and arg in ("last", "end", "$"):
        resp = trace.of_type("llm_response")
        if not resp:
            raise SystemExit("This trace has no model replies to explain.")
        return resp[-1].id
    return resolve_event(trace, arg)


def resolve_event(trace: Trace, arg: str | int) -> int:
    if isinstance(arg, str) and ":" in arg and arg.split(":", 1)[0] in ("tool", "error", "state"):
        # tool:NAME -> the last call of that tool; error: -> last error; state:KEY -> last state change
        kind, name = arg.split(":", 1)
        etype = {"tool": "tool_call", "error": "error", "state": "state"}[kind]
        hits = [e for e in trace.events if e.type == etype
                and (not name or e.payload.get("name") == name or e.payload.get("key") == name)]
        if not hits:
            raise SystemExit(f"No {etype} {name!r} in this trace.")
        return hits[-1].id
    try:
        n = int(arg)
    except (TypeError, ValueError):
        if arg in ("last", "end", "$"):
            return trace.events[-1].id
        if arg in ("first", "start"):
            return trace.events[0].id
        raise SystemExit(f"'{arg}' is not an event number.")
    if n < 0:
        if -n > len(trace.events):
            raise SystemExit(f"No event #{n}. Trace has {len(trace.events)} events.")
        n = trace.events[n].id
    if n not in {e.id for e in trace.events}:
        raise SystemExit(f"No event #{n}. Trace has events 0-{trace.events[-1].id}.")
    return n


# ------------------------------------------------------------ experiments


def _model(spec: str | None):
    if not spec:
        return None
    from .rerun import load_model_fn

    return load_model_fn(spec)


def _parse_replace(items: list[str] | None) -> dict[str, str]:
    out = {}
    for it in items or []:
        if "=>" not in it:
            raise SystemExit(f"--replace needs OLD=>NEW, got '{it}'")
        old, new = it.split("=>", 1)
        out[old] = new
    return out


def run_why(c: Console, trace: Trace, event: int, *, runs=5, tool=None, match=None, exact_args=False,
            model_fn=None, budget=300, cache=True, yes=False, show_all=False, json_out=None,
            max_pieces=40, dry=False) -> int:
    from .rerun import build_request, request_for
    from .why import estimate_calls, why

    if dry:
        from .why import suspects

        rows = suspects(trace, event)
        c.print(Text(f"Suspects for the decision behind #{event}, ranked by shared wording (no model calls):", style="bold"))
        if not rows:
            c.print(Text("  Nothing in the context shares wording with this decision.", style="dim"))
        for score, seg in rows:
            row = Text(f"  {score:5.2f}  ", style="dim")
            row.append(seg.where, style="bold")
            row.append("  " + render.compact(seg.text, 80))
            c.print(row)
        likely, _ = estimate_calls(trace, event, runs=runs, max_pieces=max_pieces)
        c.print(Text(f"This is a guess. Run without --dry to test them (about {likely} model calls).", style="dim"))
        return 0
    model = _model(model_fn)
    rid, _ = request_for(trace, event)
    likely, worst = estimate_calls(trace, event, runs=runs, max_pieces=max_pieces)
    live = model is None
    if live:
        req = build_request(trace, rid)
        resp = next((e for e in trace.children(rid) if e.type == "llm_response"), None)
        tin = ((resp.meta.get("tokens") or {}).get("input") if resp else None) or 0
        total = likely * tin
        tok = (f", roughly {total / 1e6:.1f}M input tokens" if total >= 1e6 else
               f", roughly {max(1, round(total / 1e3))}K input tokens" if total else "")
        c.print(Text(
            f"This reruns model call #{rid} ({req.get('model')}) about {likely} times "
            f"(at most {min(worst, budget)}){tok}. Results are cached, so repeats are free.",
            style="dim",
        ))
        if not yes and sys.stdin.isatty():
            if input("Continue? [Y/n] ").strip().lower() not in ("", "y", "yes"):
                return 1
    counter = {"n": 0}
    with c.status("starting") as status:
        def progress(msg: str) -> None:
            status.update(f"{msg}  ({counter['n']} model calls)")

        def on_call() -> None:
            counter["n"] += 1

        rep = why(trace, event, model=model, runs=runs, tool=tool, match=match, exact_args=exact_args,
                  budget=budget, cache_dir=".runtape/cache" if cache else None, max_pieces=max_pieces,
                  progress=progress, on_call=on_call)
    c.print(render.show_why(rep, show_all=show_all))
    if json_out:
        Path(json_out).write_text(json.dumps(rep.to_dict(), indent=2, ensure_ascii=False))
        c.print(Text(f"report written to {json_out}", style="dim"))
    return 0


def run_rerun(c: Console, trace: Trace, event: int, *, runs=5, drop=None, replace=None, system=None,
              system_file=None, model_name=None, model_fn=None, cache=True, title=None) -> int:
    from .rerun import rerun

    if system_file:
        system = Path(system_file).read_text()
    with c.status("rerunning"):
        dist = rerun(trace, event, runs=runs, drop=drop or [], replace=_parse_replace(replace), system=system,
                     model_name=model_name, model=_model(model_fn), cache_dir=".runtape/cache" if cache else None)
    what = "; ".join(dist.notes) if dist.notes else "unchanged context"
    c.print(render.show_distribution(
        dist, dist.recorded,
        title=title or f"rerun of decision #{dist.response_id} ({what}), {len(dist)} runs",
    ))
    return 0


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
        self.model_fn: str | None = None  # --model-fn for why/rerun/odds
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
        self.c.print(render.timeline(self.trace, evs, cursor=self.cur, width=self.row_width))

    @property
    def row_width(self) -> int:
        return max(30, self.c.width - 34)

    def overview(self) -> None:
        """What you see when the replay opens: one-line summary and the whole run."""
        self.c.print(render.summary_line(self.trace))
        evs = self.trace.events
        if len(evs) > 60:
            self.c.print(render.timeline(self.trace, evs[:25], width=self.row_width))
            self.c.print(Text(f"   ... {len(evs) - 50} more events (list all) ...", style="dim"))
            self.c.print(render.timeline(self.trace, evs[-25:], width=self.row_width))
        else:
            self.c.print(render.timeline(self.trace, evs, width=self.row_width))
        self.c.print(
            Text(
                "Type an event number to open it. Enter steps forward. "
                "grep TERM searches, context shows what the model saw, help lists everything.",
                style="dim",
            )
        )

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
            self.c.print(render.timeline(self.trace, errs, cursor=self.cur, width=self.row_width))

    def _exp_args(self, arg: str, prog: str):
        p = argparse.ArgumentParser(prog=prog, add_help=False, exit_on_error=False)
        p.add_argument("event", nargs="?")
        p.add_argument("--runs", type=int, default=5)
        p.add_argument("--tool")
        p.add_argument("--match")
        p.add_argument("--all", action="store_true")
        p.add_argument("--drop", action="append")
        p.add_argument("--replace", action="append")
        p.add_argument("--system-file")
        p.add_argument("--model")
        try:
            a = p.parse_args(shlex.split(arg))
        except (argparse.ArgumentError, SystemExit, ValueError) as e:
            self.c.print(Text(f"{prog}: {e}", style="red"))
            return None
        if a.event in ("last", "end", "$"):
            try:
                a.event = resolve_decision(self.trace, a.event)
            except SystemExit as e:
                self.c.print(Text(str(e), style="red"))
                return None
        else:
            a.event = self._event_arg(a.event) if a.event else self.cur
        return a if a.event is not None else None

    def _safely(self, fn) -> None:
        try:
            fn()
        except (ValueError, SystemExit) as e:
            self.c.print(Text(str(e), style="red"))
        except Exception as e:  # API errors, missing keys: report, don't kill the session
            self.c.print(Text(f"{type(e).__name__}: {e}", style="red"))

    def do_why(self, arg: str) -> None:
        """why [N] [--runs K] [--tool NAME] [--match REGEX] [--all]
        Find which part of the context caused the decision at N (default: current), by removing pieces and
        re-running that decision. Point at a tool call, a model reply, or a model call."""
        a = self._exp_args(arg, "why")
        if a:
            self._safely(lambda: run_why(self.c, self.trace, a.event, runs=a.runs, tool=a.tool, match=a.match,
                                         model_fn=self.model_fn, show_all=a.all, yes=False))

    def do_rerun(self, arg: str) -> None:
        """rerun [N] [--runs K] [--drop REF] [--replace OLD=>NEW] [--system-file F] [--model NAME]
        Re-run the decision at N as recorded or with edits and show what the model does.
        REF is an event number (9), or a part of one (9[1], 9[1].text)."""
        a = self._exp_args(arg, "rerun")
        if a:
            self._safely(lambda: run_rerun(self.c, self.trace, a.event, runs=a.runs, drop=a.drop, replace=a.replace,
                                           system_file=a.system_file, model_name=a.model, model_fn=self.model_fn))

    def do_odds(self, arg: str) -> None:
        """odds [N] [--runs K]   How often the model makes the same decision on the identical context."""
        a = self._exp_args(arg, "odds")
        if a:
            self._safely(lambda: run_rerun(self.c, self.trace, a.event, runs=a.runs, model_fn=self.model_fn,
                                           title=f"odds for decision at #{a.event}, identical context"))

    def do_quit(self, arg: str) -> bool:
        """quit   Exit."""
        return True

    # short aliases
    do_s = do_step
    do_b = do_back
    do_g = do_goto
    do_jump = do_goto
    do_p = do_show
    do_ctx = do_context
    do_l = do_list
    do_n = do_next
    do_q = do_quit
    do_exit = do_quit

    def get_names(self):
        # keep aliases out of the help listing
        hidden = {"do_s", "do_b", "do_g", "do_jump", "do_p", "do_ctx", "do_l", "do_n", "do_q", "do_exit", "do_EOF"}
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
    s.add_argument("--model-fn", help="model function for why/rerun/odds inside the replay")

    s = sub.add_parser("mcp", help="run an MCP server so coding agents can debug your runs")
    s.add_argument("--model-fn", help="model function for why/rerun instead of the live API")

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

    def experiment(s):
        s.add_argument("--runs", type=int, default=5, help="reruns per variant (default 5)")
        s.add_argument("--model-fn", help="use a Python function instead of the live API: module:function or file.py:function")
        s.add_argument("--no-cache", action="store_true", help="don't reuse cached model replies")

    s = sub.add_parser("why", help="find which part of the context caused a decision")
    s.add_argument("trace")
    s.add_argument("event", help="a tool call, model reply, or model call")
    experiment(s)
    s.add_argument("--tool", help="explain whether this tool gets called")
    s.add_argument("--match", help="explain whether the reply matches this regex")
    s.add_argument("--exact-args", action="store_true", help="count a rerun as the same only if tool arguments match too")
    s.add_argument("--budget", type=int, default=300, help="max model calls (default 300)")
    s.add_argument("--all", action="store_true", help="list every piece tested")
    s.add_argument("--max-pieces", type=int, default=40, help="test at most this many pieces, most suspicious first")
    s.add_argument("--json", dest="json_out", help="also write the report as JSON")
    s.add_argument("-y", "--yes", action="store_true", help="don't ask before making model calls")
    s.add_argument("--dry", action="store_true", help="just rank suspects by shared wording, no model calls")

    s = sub.add_parser("rerun", help="re-run a decision as recorded or with edits")
    s.add_argument("trace")
    s.add_argument("event")
    experiment(s)
    s.add_argument("--drop", action="append", help="remove context from an event: 9, 9[1], 9[1].text, system")
    s.add_argument("--replace", action="append", help="OLD=>NEW text replacement in the context")
    s.add_argument("--system-file", help="replace the system prompt with this file")
    s.add_argument("--model", dest="model_name", help="rerun on a different model")

    s = sub.add_parser("odds", help="how often the model repeats a decision on the same context")
    s.add_argument("trace")
    s.add_argument("event")
    experiment(s)

    s = sub.add_parser("diff", help="what changed between two events")
    s.add_argument("trace")
    s.add_argument("a")
    s.add_argument("b")
    return p


def main(argv: list[str] | None = None, console: Console | None = None) -> int:
    args = build_parser().parse_args(argv)
    c = console or Console(no_color=args.no_color, highlight=False)
    cmd_name = args.cmd or "replay"

    if cmd_name == "mcp":
        try:
            from .mcp_server import serve
        except ImportError:
            c.print(Text("The MCP server needs the mcp package: pip install 'runtape[mcp]'", style="red"))
            return 1
        serve(args.model_fn)
        return 0

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
            c.print(render.timeline(trace, trace.events, width=max(30, c.width - 34)))
        elif cmd_name == "show":
            n = resolve_event(trace, args.event)
            c.print(render.show_event(trace, trace[n], raw=args.raw, full=args.full))
        elif cmd_name == "context":
            c.print(render.show_context(trace, resolve_event(trace, args.event), full=args.full))
        elif cmd_name == "grep":
            c.print(render.show_grep(trace, args.term))
        elif cmd_name == "diff":
            c.print(render.show_diff(trace, resolve_event(trace, args.a), resolve_event(trace, args.b)))
        elif cmd_name == "why":
            return run_why(c, trace, resolve_decision(trace, args.event), runs=args.runs, tool=args.tool,
                           match=args.match, exact_args=args.exact_args, model_fn=args.model_fn,
                           budget=args.budget, cache=not args.no_cache, yes=args.yes, show_all=args.all,
                           json_out=args.json_out, max_pieces=args.max_pieces, dry=args.dry)
        elif cmd_name == "rerun":
            return run_rerun(c, trace, resolve_decision(trace, args.event), runs=args.runs, drop=args.drop,
                             replace=args.replace, system_file=args.system_file, model_name=args.model_name,
                             model_fn=args.model_fn, cache=not args.no_cache)
        elif cmd_name == "odds":
            ev = resolve_decision(trace, args.event)
            return run_rerun(c, trace, ev, runs=args.runs, model_fn=args.model_fn, cache=not args.no_cache,
                             title=f"odds for decision at #{ev}, identical context")
        elif cmd_name == "replay":
            start = resolve_event(trace, getattr(args, "at", "0"))
            r = Replay(trace, c, start=start)
            r.model_fn = getattr(args, "model_fn", None)
            r.overview()
            if start != trace.events[0].id:
                r.show_current()
            try:
                r.cmdloop()
            except KeyboardInterrupt:
                c.print()
    except ValueError as e:
        c.print(Text(str(e), style="red"))
        return 1
    except Exception as e:
        msg = str(e)
        if "api_key" in msg.lower() or "authentication" in msg.lower():
            msg = ("No API key for the model this decision was made with. Set ANTHROPIC_API_KEY or OPENAI_API_KEY, "
                   "or pass --model-fn to use a local function.")
        c.print(Text(f"{type(e).__name__}: {msg}", style="red"))
        return 1
    except SystemExit as e:
        if isinstance(e.code, str):
            c.print(Text(e.code, style="red"))
            return 1
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
