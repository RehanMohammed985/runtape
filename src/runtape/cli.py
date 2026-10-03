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


def _sdk_of(e: BaseException) -> tuple[str, str]:
    """(provider name, key variable) for an exception raised by the OpenAI or Anthropic SDK."""
    mod = type(e).__module__ or ""
    if mod.startswith("anthropic"):
        return "Anthropic", "ANTHROPIC_API_KEY"
    return "OpenAI", "OPENAI_API_KEY"


def friendly(e: BaseException) -> str:
    """One plain line for an error, instead of a traceback or an SDK's raw message."""
    name, msg = type(e).__name__, str(e)
    low = msg.lower()
    if isinstance(e, FileNotFoundError):
        return f"No such file: {e.filename}" if e.filename else msg
    if name == "AuthenticationError" or getattr(e, "status_code", None) == 401:
        who, var = _sdk_of(e)
        return (f"The {who} API rejected the key in {var} (401). Check the key, or pass --model-fn "
                "module:function to rerun with your own model function.")
    if name in ("APIConnectionError", "APITimeoutError"):
        url = getattr(getattr(e, "request", None), "url", None)
        where = f" at {url.scheme}://{url.netloc.decode() if isinstance(url.netloc, bytes) else url.netloc}" if url else ""
        return (f"Can't reach the model server{where}. Check your connection, or that the server is running "
                "(a local server like Ollama needs to be started first).")
    if "api_key" in low or "auth_token" in low:
        who, var = _sdk_of(e)
        return f"No {who} API key: set {var}, or pass --model-fn module:function to rerun with your own model function."
    return f"{name}: {msg}"


def need_key(req: dict) -> str | None:
    """Why reruns of this request can't start here, if they can't: the trace came from the examples' stand-in
    model, or the API it goes to needs a key that isn't set."""
    if req.get("model") in ("simulated", "claude-scripted"):
        return ("This trace was recorded with the examples' stand-in model, which has no API. Add the --model-fn "
                "the example printed (for example --model-fn examples/inbox_agent.py:simulated_model).")
    if req.get("endpoint"):  # a local or self-hosted server: usually no key, but it has to be running
        from urllib.parse import urlparse

        u = urlparse(req["endpoint"])
        if u.hostname in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
            import socket

            try:
                socket.create_connection((u.hostname, u.port or (443 if u.scheme == "https" else 80)), 2).close()
            except OSError:
                return (f"Nothing is answering at {req['endpoint']}, where reruns of this decision go. Start the "
                        "server (for Ollama: ollama serve, or open the app), or pass --model-fn module:function.")
        return None
    anthropic = req.get("api") == "messages" or req.get("provider") == "anthropic"
    var = "ANTHROPIC_API_KEY" if anthropic else "OPENAI_API_KEY"
    if req.get("api") in ("messages", "responses", "chat.completions", "langchain") and not os.environ.get(var):
        if anthropic and os.environ.get("ANTHROPIC_AUTH_TOKEN"):
            return None
        who = "Anthropic" if anthropic else "OpenAI"
        return (f"Reruns of this decision go to the {who} API ({req.get('model')}), and {var} isn't set. Set it, "
                "or pass --model-fn module:function to rerun with your own model function.")
    return None


def _check_regex(pattern: str | None, flag: str = "--match") -> None:
    if pattern is None:
        return
    import re

    try:
        re.compile(pattern)
    except re.error as e:
        raise SystemExit(f"{flag} '{pattern}' isn't a valid regular expression: {e}.")


def carry(**opts) -> str:
    """The options a user passed, as they'd type them again in a suggested next command."""
    out = []
    for flag, val in opts.items():
        if val in (None, False):
            continue
        out.append(f"--{flag.replace('_', '-')}" + ("" if val is True else " " + shlex.quote(str(val))))
    return (" " + " ".join(out)) if out else ""


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
            names = "\n  ".join(str(t) for t in reversed(matches[-5:]))
            more = f"\n  ... and {len(matches) - 5} older" if len(matches) > 5 else ""
            raise SystemExit(f"'{arg}' matches {len(matches)} traces; pass more of the name, or a path, or 'last' "
                             f"for the newest:\n  {names}{more}")
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
            there = sorted({e.payload.get("name") or e.payload.get("key") or "" for e in trace.events
                            if e.type == etype} - {""})
            listed = f" It has: {', '.join(there)}." if there else ""
            raise SystemExit(f"No {etype} {name!r} in {trace.path}.{listed}")
        return hits[-1].id
    try:
        n = int(arg)
    except (TypeError, ValueError):
        if arg in ("last", "end", "$"):
            return trace.events[-1].id
        if arg in ("first", "start"):
            return trace.events[0].id
        tools = sorted({e.payload.get("name") for e in trace.events if e.type == "tool_call"} - {None})
        hint = f" For the last call of a tool, use tool:NAME (tool:{arg})." if arg in tools else (
            " Use an event number, 'last', or tool:NAME" + (f" (tools called: {', '.join(tools)})." if tools else "."))
        raise SystemExit(f"'{arg}' is not an event number.{hint}")
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


def generic_tool_note(trace: Trace, event: int, tool=None, match=None) -> str | None:
    """When the decision is a call to a tool the agent calls many times with other arguments (run_command,
    search), the default question (does it call the tool at all?) is usually not the one meant."""
    if tool or match:
        return None
    from .why import make_target

    try:
        t = make_target(trace, event)
    except ValueError:
        return None
    if t.mode != "tool" or t.args is not None:
        return None
    this = next((tc.get("arguments") for tc in t.recorded.tool_calls if tc.get("name") == t.tool), None)
    calls = [e for e in trace.of_type("tool_call") if e.payload.get("name") == t.tool]
    others = [e.payload.get("arguments") for e in calls if e.payload.get("arguments") != this]
    if not others or not isinstance(this, dict):
        return None
    import re

    seen = json.dumps(others, ensure_ascii=False)
    val = next((v for v in this.values() if isinstance(v, str) and 2 < len(v) <= 60 and v not in seen), None)
    if val is None:
        return None
    pattern = val if all(ch.isalnum() or ch in " -_/:=@,'\"" for ch in val) else re.escape(val)
    return (f"{t.tool} is called {len(calls)} times in this run, with different arguments, and this counts any call "
            f"to it. To explain this call only, add --match {shlex.quote(pattern)}.")


def run_why(c: Console, trace: Trace, event: int, *, runs=5, tool=None, match=None, exact_args=False,
            model_fn=None, budget=400, cache=True, yes=False, show_all=False, json_out=None,
            max_pieces=80, dry=False, expand=6, fill=None, full=False, guess=True, event_arg=None) -> int:
    from .rerun import build_request, request_for
    from .why import estimate_calls, why

    _check_regex(match)
    if json_out:
        Path(json_out).parent.mkdir(parents=True, exist_ok=True)
    if dry:
        from .why import suspects

        rows = suspects(trace, event)
        from .rerun import request_for as _rf

        c.print(Text(f"Suspects for the decision at #{_rf(trace, event)[1]}, ranked by shared wording (no model calls):",
                     style="bold"))
        if not rows:
            c.print(Text("  Nothing in the context shares wording with this decision.", style="dim"))
        for score, seg in rows:
            row = Text(f"  {score:5.2f}  ", style="dim")
            row.append(seg.where, style="bold")
            row.append("  " + render.compact(seg.text, 80))
            c.print(row)
        likely, _ = estimate_calls(trace, event, runs=runs, max_pieces=max_pieces, full=full)
        c.print(Text(f"This is a guess. Run without --dry to test them (about {likely} model calls).", style="dim"))
        return 0
    model = _model(model_fn)
    rid, _ = request_for(trace, event)
    likely, worst = estimate_calls(trace, event, runs=runs, max_pieces=max_pieces, full=full)
    live = model is None
    note = generic_tool_note(trace, event, tool, match)
    if note:
        c.print(Text("! " + note, style="yellow"))
    if live:
        req = build_request(trace, rid)
        blocked = need_key(req)
        if blocked:
            raise SystemExit(blocked)
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
                  budget=budget, cache_dir=".runtape/cache" if cache else None, max_pieces=max_pieces, expand=expand,
                  fill=fill, progress=progress, on_call=on_call, depth="full" if full else "quick", guess=guess)
    c.print(render.show_why(rep, show_all=show_all))
    if json_out:
        Path(json_out).write_text(json.dumps(rep.to_dict(), indent=2, ensure_ascii=False))
        c.print(Text(f"report written to {json_out}", style="dim"))
    if any(x.kind == "decisive" for x in rep.causes) and not rep.stopped:
        opts = carry(tool=tool, match=match, model_fn=model_fn, fill=fill, full=full, no_guess=not guess)
        c.print(Text(f"Next: runtape fix {shlex.quote(str(trace.path))} {shlex.quote(str(event_arg or event))}{opts} "
                     "checks which fixes stop it (these reruns are reused), and --write-test keeps it fixed.",
                     style="dim"))
    return 0


def run_rerun(c: Console, trace: Trace, event: int, *, runs=5, drop=None, replace=None, system=None,
              system_file=None, model_name=None, model_fn=None, cache=True, title=None, fill=None) -> int:
    from .rerun import build_request, request_for, rerun

    if system_file:
        system = Path(system_file).read_text()
    if not model_fn and not model_name:
        blocked = need_key(build_request(trace, request_for(trace, event)[0]))
        if blocked:
            raise SystemExit(blocked)
    with c.status("rerunning"):
        dist = rerun(trace, event, runs=runs, drop=drop or [], replace=_parse_replace(replace), system=system,
                     model_name=model_name, model=_model(model_fn), cache_dir=".runtape/cache" if cache else None,
                     fill=fill)
    what = "; ".join(dist.notes) if dist.notes else "unchanged context"
    c.print(render.show_distribution(
        dist, dist.recorded,
        title=title or f"rerun of decision #{dist.response_id} ({what}), {len(dist)} runs",
    ))
    return 0


def run_fix(c: Console, trace: Trace, event: int, *, runs=10, tool=None, match=None, model_fn=None, budget=600,
            cache=True, yes=False, write_test=None, fill=None, full=False, guess=True, event_arg=None) -> int:
    from .fix import check_for, fix, test_for, unique_path
    from .rerun import build_request, request_for
    from .why import make_target

    _check_regex(match)
    if write_test:
        check_for(make_target(trace, event, tool=tool, match=match))  # fail now, before spending model calls
    model = _model(model_fn)
    note = generic_tool_note(trace, event, tool, match)
    if note:
        c.print(Text("! " + note, style="yellow"))
    if model is None:
        blocked = need_key(build_request(trace, request_for(trace, event)[0]))
        if blocked:
            raise SystemExit(blocked)
    if model is None and not yes and sys.stdin.isatty():
        from .why import estimate_calls

        likely, _ = estimate_calls(trace, event, full=full)
        c.print(Text(f"This finds the cause (about {likely} model calls), then checks up to 4 fixes with {runs} "
                     f"reruns each, {budget} calls at most. Replies are cached.", style="dim"))
        if input("Continue? [Y/n] ").strip().lower() not in ("", "y", "yes"):
            return 1
    counter = {"n": 0}
    with c.status("starting") as status:
        def progress(msg: str) -> None:
            status.update(f"{msg}  ({counter['n']} model calls)")

        def on_call() -> None:
            counter["n"] += 1

        fr = fix(trace, event, model=model, runs=runs, budget=budget, cache_dir=".runtape/cache" if cache else None,
                 progress=progress, tool=tool, match=match, fill=fill, on_call=on_call, depth="full" if full else "quick",
                 guess=guess)
    c.print(render.show_fix(fr))
    if write_test:
        path = test_for(fr, trace, event, unique_path(write_test), runs=runs, model_fn=model_fn)
        if path is None:
            c.print(Text("No fix passed, so no test was written.", style="yellow"))
            return 1
        c.print(Text(f"Regression test written to {path}. Run it with: pytest {path}", style="bold"))
    elif fr.best is not None:
        opts = carry(tool=tool, match=match, model_fn=model_fn, fill=fill, full=full, no_guess=not guess)
        c.print(Text(f"Turn it into a test: runtape fix {shlex.quote(str(trace.path))} "
                     f"{shlex.quote(str(event_arg or event))}{opts} --write-test "
                     f"tests/test_{_test_stem(fr.report.target)}.py", style="dim"))
    return 0


def _test_stem(target) -> str:
    from .fix import test_name

    return test_name(target)


def run_test(c: Console, trace: Trace, event: int, *, out=None, tool=None, match=None, add_system=None, runs=10,
             model_fn=None) -> int:
    from .fix import test_name, unique_path, write_test as _write
    from .why import make_target

    target = make_target(trace, event, tool=tool, match=match)
    out = unique_path(out or f"tests/test_{test_name(target)}.py")
    path = _write(trace.path, event, target, out, add_system=add_system, runs=runs, model_fn=model_fn)
    c.print(Text(f"Regression test written to {path}. Run it with: pytest {path}", style="bold"))
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
        class _Quiet(argparse.ArgumentParser):
            def error(self, message):  # report in the session instead of exiting with a usage dump
                raise ValueError(message)

        p = _Quiet(prog=prog, add_help=False)
        p.add_argument("event", nargs="?")
        p.add_argument("--model-fn")
        p.add_argument("--runs", type=int, default=5)
        p.add_argument("--tool")
        p.add_argument("--match")
        p.add_argument("--all", action="store_true")
        p.add_argument("--drop", action="append")
        p.add_argument("--replace", action="append")
        p.add_argument("--system-file")
        p.add_argument("--model")
        p.add_argument("--fill")
        try:
            a = p.parse_args(shlex.split(arg))
        except (argparse.ArgumentError, ValueError) as e:
            self.c.print(Text(f"{prog}: {e}", style="red"))
            return None
        if a.model_fn:  # kept for the rest of the session
            self.model_fn = a.model_fn
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
            self.c.print(Text(friendly(e), style="red"))

    def do_why(self, arg: str) -> None:
        """why [N] [--runs K] [--tool NAME] [--match REGEX] [--fill TEXT] [--all] [--model-fn FILE.py:FN]
        Find which part of the context caused the decision at N (default: current), by removing pieces and
        re-running that decision. Point at a tool call, a model reply, or a model call. --model-fn reruns with
        your own function instead of the live API, for the rest of the session."""
        a = self._exp_args(arg, "why")
        if a:
            self._safely(lambda: run_why(self.c, self.trace, a.event, runs=a.runs, tool=a.tool, match=a.match,
                                         model_fn=self.model_fn, show_all=a.all, yes=False, fill=a.fill))

    def do_rerun(self, arg: str) -> None:
        """rerun [N] [--runs K] [--drop REF] [--fill TEXT] [--replace OLD=>NEW] [--system-file F] [--model NAME] [--model-fn F]
        Re-run the decision at N as recorded or with edits and show what the model does.
        REF is an event number (9), or a part of one (9[1], 9[1].text)."""
        a = self._exp_args(arg, "rerun")
        if a:
            self._safely(lambda: run_rerun(self.c, self.trace, a.event, runs=a.runs, drop=a.drop, replace=a.replace,
                                           system_file=a.system_file, model_name=a.model, model_fn=self.model_fn,
                                           fill=a.fill))

    def do_odds(self, arg: str) -> None:
        """odds [N] [--runs K] [--model-fn F]   How often the model makes the same decision on the identical context."""
        a = self._exp_args(arg, "odds")
        if a:
            from .rerun import request_for as _rf

            self._safely(lambda: run_rerun(self.c, self.trace, a.event, runs=a.runs, model_fn=self.model_fn,
                                           title=f"odds for the decision at #{_rf(self.trace, a.event)[1]}, identical context"))

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

    s = sub.add_parser("import", help="import a trace from OpenTelemetry or Langfuse, to explain it with why")
    s.add_argument("source", help="an OpenTelemetry export (OTLP JSON or JSON lines), a Langfuse trace as JSON, "
                   "or langfuse:TRACE_ID to fetch it (LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST)")
    s.add_argument("-o", "--out", help="where to write the trace (default traces/<source name>.jsonl)")
    s.add_argument("--trace-id", help="import only this trace from an export that holds several")
    s.add_argument("--provider", choices=["openai", "anthropic"],
                   help="the API reruns go to (default: what the source says, else openai)")
    s.add_argument("--base-url", help="send reruns to this OpenAI-compatible server instead (vLLM, Ollama, a "
                   "hosted open model)")
    s.add_argument("--tools", help="a JSON file of tool definitions, when the source didn't record them")

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

    ref_help = ("event number, 'last' (the last model decision), or tool:NAME (the last call of a tool)")
    s = sub.add_parser("why", help="find which part of the context caused a decision")
    s.add_argument("trace", help="trace file, part of its name, or 'last' for the newest in ./traces")
    s.add_argument("event", help="the decision to explain: " + ref_help)
    experiment(s)
    s.add_argument("--tool", help="explain whether this tool gets called")
    s.add_argument("--match", help="explain whether the reply matches this regex")
    s.add_argument("--exact-args", action="store_true", help="count a rerun as the same only if tool arguments match too")
    s.add_argument("--budget", type=int, default=400, help="max model calls (default 400)")
    s.add_argument("--all", action="store_true", help="list every piece tested")
    s.add_argument("--max-pieces", type=int, default=80, help="test at most this many pieces, most suspicious first (default 80)")
    s.add_argument("--expand", type=int, default=6, help="look inside this many pieces for masked causes (default 6)")
    s.add_argument("--full", action="store_true", help="keep searching after the main cause is settled: causes hidden "
                   "inside other pieces, and combinations of pieces")
    s.add_argument("--no-guess", action="store_true", help="test every piece, instead of first asking the model which "
                   "piece made it decide and testing that one")
    s.add_argument("--json", dest="json_out", help="also write the report as JSON")
    s.add_argument("-y", "--yes", action="store_true", help="don't ask before making model calls")
    s.add_argument("--dry", action="store_true", help="just rank suspects by shared wording, no model calls")
    s.add_argument("--fill", help='text that replaces a removed message or tool result: "marker" ([content removed], default), "empty" ((empty)), or any text')

    s = sub.add_parser("rerun", help="re-run a decision as recorded or with edits")
    s.add_argument("trace", help="trace file, part of its name, or 'last'")
    s.add_argument("event", help=ref_help)
    experiment(s)
    s.add_argument("--drop", action="append", help="remove context from an event: 9, 9[1], 9[1].text, system")
    s.add_argument("--replace", action="append", help="OLD=>NEW text replacement in the context")
    s.add_argument("--system-file", help="replace the system prompt with this file")
    s.add_argument("--model", dest="model_name", help="rerun on a different model")
    s.add_argument("--fill", help='text that replaces a removed message or tool result: "marker" ([content removed], default), "empty" ((empty)), or any text')

    s = sub.add_parser("fix", help="find the cause, then check candidate fixes on the recorded context")
    s.add_argument("trace", help="trace file, part of its name, or 'last'")
    s.add_argument("event", help="the decision: " + ref_help)
    s.add_argument("--runs", type=int, default=10, help="reruns per fix (default 10)")
    s.add_argument("--model-fn", help="use a Python function instead of the live API: module:function or file.py:function")
    s.add_argument("--no-cache", action="store_true", help="don't reuse cached model replies")
    s.add_argument("--tool", help="the decision is whether this tool gets called")
    s.add_argument("--match", help="the decision is a reply matching this regex")
    s.add_argument("--budget", type=int, default=600, help="max model calls (default 600)")
    s.add_argument("--fill", help="replacement text for removed content (see runtape why --help)")
    s.add_argument("--write-test", metavar="PATH", help="write a pytest regression test using the best verified fix")
    s.add_argument("--full", action="store_true", help="run the full cause search (see runtape why --help)")
    s.add_argument("--no-guess", action="store_true", help="test every piece (see runtape why --help)")
    s.add_argument("-y", "--yes", action="store_true", help="don't ask before making model calls")

    s = sub.add_parser("test", help="write a pytest regression test for a recorded decision")
    s.add_argument("trace", help="trace file, part of its name, or 'last'")
    s.add_argument("event", help="the decision: " + ref_help)
    s.add_argument("--out", help="where to write the test (default tests/test_<decision>.py)")
    s.add_argument("--add-system", help="text to add to the system prompt (a fix) in the test")
    s.add_argument("--add-system-file", help="the same, read from a file")
    s.add_argument("--tool", help="the decision is whether this tool gets called")
    s.add_argument("--match", help="the decision is a reply matching this regex")
    s.add_argument("--runs", type=int, default=10, help="reruns in the test (default 10)")
    s.add_argument("--model-fn", help="a Python model function for the test to use instead of the live API")

    s = sub.add_parser("odds", help="how often the model repeats a decision on the same context")
    s.add_argument("trace", help="trace file, part of its name, or 'last'")
    s.add_argument("event", help=ref_help)
    experiment(s)

    s = sub.add_parser("diff", help="what changed between two events")
    s.add_argument("trace")
    s.add_argument("a")
    s.add_argument("b")
    return p


_COMMANDS = {"replay", "ls", "summary", "timeline", "show", "context", "grep", "diff", "why", "rerun", "odds", "mcp",
             "fix", "test", "import"}


def run_import(c: Console, args) -> int:
    from .importers import import_trace

    done = import_trace(args.source, args.out, trace_id=args.trace_id, provider=args.provider,
                        endpoint=args.base_url, tools_file=args.tools)
    for r in done:
        c.print(Text(f"Imported {r.calls} model calls and {r.tool_calls} tool calls into ", style="green")
                + Text(str(r.path), style="bold"))
        if r.missing_tools:
            c.print(Text(f"  {r.missing_tools} of the {r.calls} model calls use tools, but the source has no tool "
                         "definitions for them. A rerun can't call a tool it isn't given, so a decision to call one "
                         "won't repeat. Pass --tools tools.json (the tools as sent to the model), or turn on tool "
                         "capture in the instrumentation.", style="yellow"))
        for note in r.notes:
            c.print(Text("  " + note, style="yellow"))
    target = shlex.quote(str(done[0].path)) if len(done) == 1 else "<trace>"
    if args.base_url:
        reruns = f"reruns go to {args.base_url}"
    else:
        reruns = (f"reruns go to the {done[0].provider} API; to send them to another OpenAI-compatible server, "
                  "import again with --base-url")
    c.print(Text(f"Next: runtape why {target} last   ({reruns})", style="dim"))
    return 0


def main(argv: list[str] | None = None, console: Console | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    i = 0
    while i < len(argv) and argv[i] in ("--no-color",):
        i += 1  # global options come first
    if i < len(argv) and not argv[i].startswith("-") and argv[i] not in _COMMANDS and (
        argv[i].endswith(".jsonl") or argv[i] == "last" or Path(argv[i]).is_file()
        or any(argv[i] in t.name for t in find_traces())
    ):
        argv.insert(i, "replay")  # `runtape trace.jsonl`, `runtape last` or `runtape <part of a name>` opens it
    args = build_parser().parse_args(argv)
    c = console or Console(no_color=args.no_color, highlight=False)
    err = c if console is not None else Console(stderr=True, no_color=args.no_color, highlight=False)
    cmd_name = args.cmd or "replay"

    if cmd_name == "mcp":
        try:
            import mcp  # noqa: F401

            from .mcp_server import serve
        except ImportError:
            c.print(Text("The MCP server needs the mcp package: pip install 'runtape[mcp]'", style="red"))
            return 1
        serve(args.model_fn)
        return 0

    try:
        if cmd_name == "import":
            return run_import(c, args)
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
                           json_out=args.json_out, max_pieces=args.max_pieces, dry=args.dry, expand=args.expand,
                           fill=args.fill, full=args.full, guess=not args.no_guess, event_arg=args.event)
        elif cmd_name == "rerun":
            return run_rerun(c, trace, resolve_decision(trace, args.event), runs=args.runs, drop=args.drop,
                             replace=args.replace, system_file=args.system_file, model_name=args.model_name,
                             model_fn=args.model_fn, cache=not args.no_cache, fill=args.fill)
        elif cmd_name == "fix":
            return run_fix(c, trace, resolve_decision(trace, args.event), runs=args.runs, tool=args.tool,
                           match=args.match, model_fn=args.model_fn, budget=args.budget, cache=not args.no_cache,
                           yes=args.yes, write_test=args.write_test, fill=args.fill, full=args.full,
                           guess=not args.no_guess, event_arg=args.event)
        elif cmd_name == "test":
            add = Path(args.add_system_file).read_text() if args.add_system_file else args.add_system
            return run_test(c, trace, resolve_decision(trace, args.event), out=args.out, tool=args.tool,
                            match=args.match, add_system=add, runs=args.runs, model_fn=args.model_fn)
        elif cmd_name == "odds":
            ev = resolve_decision(trace, args.event)
            from .rerun import request_for as _rf

            return run_rerun(c, trace, ev, runs=args.runs, model_fn=args.model_fn, cache=not args.no_cache,
                             title=f"odds for the decision at #{_rf(trace, ev)[1]}, identical context")
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
        err.print(Text(str(e), style="red"))
        return 1
    except Exception as e:
        err.print(Text(friendly(e), style="red"))
        return 1
    except SystemExit as e:
        if isinstance(e.code, str):
            err.print(Text(e.code, style="red"))
            return 1
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
