"""Turn trace events into terminal output (rich renderables)."""
from __future__ import annotations

import difflib
import json
from typing import Any, Iterable

from rich.console import Group, RenderableType
from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from .trace import Event, Trace

TYPE_STYLE = {
    "run_start": "bold white",
    "run_end": "bold white",
    "llm_request": "cyan",
    "llm_response": "bold cyan",
    "tool_call": "yellow",
    "tool_result": "green",
    "error": "bold red",
    "state": "magenta",
    "log": "dim",
}

ROLE_STYLE = {
    "system": "bold white",
    "user": "bold blue",
    "assistant": "bold cyan",
    "tool": "bold green",
    "developer": "bold white",
}


# ------------------------------------------------------------- utilities


def compact(v: Any, limit: int = 80) -> str:
    """Short single-line form of any value."""
    if isinstance(v, str):
        s = v
    else:
        try:
            s = json.dumps(v, ensure_ascii=False, separators=(", ", ": "))
        except (TypeError, ValueError):
            s = repr(v)
    s = " ".join(s.split())
    return s if len(s) <= limit else s[: limit - 3] + "..."


def pretty(v: Any) -> str:
    if isinstance(v, str):
        return v
    try:
        return json.dumps(v, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return repr(v)


def clip(s: str, limit: int | None) -> str:
    if limit is None or len(s) <= limit:
        return s
    return s[:limit] + f"\n... [{len(s) - limit} more chars, use --full]"


def _text(s: str, style: str = "", highlight: str | None = None) -> Text:
    t = Text(s, style=style)
    if highlight:
        t.highlight_words([highlight], style="bold black on yellow", case_sensitive=False)
    return t


def offset(trace: Trace, e: Event) -> str:
    t0 = trace.events[0].ts if trace.events else e.ts
    return f"+{e.ts - t0:6.2f}s"


# ------------------------------------------------------ message handling


def message_parts(msg: Any) -> tuple[str, list[str]]:
    """(role, [text lines]) for OpenAI or Anthropic style messages."""
    if not isinstance(msg, dict):
        return "?", [compact(msg, 10_000)]
    role = msg.get("role") or msg.get("type") or "?"
    parts: list[str] = []
    content = msg.get("content")
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                parts.append(compact(block, 10_000))
                continue
            bt = block.get("type")
            if bt in ("text", "input_text", "output_text"):
                parts.append(block.get("text", ""))
            elif bt == "tool_use":
                parts.append(f"-> {block.get('name')}({compact(block.get('input'), 10_000)})")
            elif bt == "tool_result":
                inner = block.get("content")
                if isinstance(inner, list):
                    inner = " ".join(b.get("text", compact(b)) if isinstance(b, dict) else str(b) for b in inner)
                err = " [error]" if block.get("is_error") else ""
                parts.append(f"<- result{err}: {inner if isinstance(inner, str) else compact(inner, 10_000)}")
            elif bt in ("image", "image_url", "input_image"):
                parts.append("[image]")
            else:
                parts.append(compact(block, 10_000))
    elif content is not None:
        parts.append(compact(content, 10_000))
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", tc) if isinstance(tc, dict) else {}
        parts.append(f"-> {fn.get('name')}({fn.get('arguments')})")
    if msg.get("type") == "function_call":  # openai responses input item
        parts.append(f"-> {msg.get('name')}({msg.get('arguments')})")
    if msg.get("type") == "function_call_output":
        parts.append(f"<- result: {msg.get('output')}")
    if role == "tool" and msg.get("tool_call_id"):
        role = "tool"
    return str(role), parts


def render_message(i: int, msg: Any, limit: int | None, highlight: str | None) -> RenderableType:
    role, parts = message_parts(msg)
    head = Text(f"[{i}] ", style="dim")
    head.append(role, style=ROLE_STYLE.get(role, "bold"))
    body = clip("\n".join(p for p in parts if p), limit)
    return Group(head, _text(body or "(empty)", highlight=highlight))


# -------------------------------------------------------- one-line views


def oneline(e: Event, limit: int = 80) -> str:
    p = e.payload
    t = e.type
    if t == "run_start":
        return f"{p.get('name')}  run {p.get('run_id')}"
    if t == "run_end":
        return f"{p.get('status')}  {p.get('duration_ms')} ms"
    if t == "llm_request":
        if "messages" in p:
            new = len(p["messages"])
            what = f"{new} msgs"
        else:
            new = len(p.get("messages_append", []))
            what = f"+{new} msgs"
        extra = []
        if "system" in p:
            extra.append("system")
        if "tools" in p:
            extra.append("tools")
        chg = f"  (set {', '.join(extra)})" if extra else ""
        return f"{p.get('model')}  {what}{chg}"
    if t == "llm_response":
        bits = []
        if p.get("text"):
            bits.append(compact(p["text"], limit))
        for tc in p.get("tool_calls") or []:
            bits.append(f"-> {tc.get('name')}")
        return "  ".join(bits) or f"(no text) stop={p.get('stop_reason')}"
    if t == "tool_call":
        return f"{p.get('name')}({compact(p.get('arguments'), limit)})"
    if t == "tool_result":
        if "error" in p:
            err = p["error"]
            return f"{p.get('name')} FAILED {err.get('type')}: {compact(err.get('message'), limit)}"
        return f"{p.get('name')} -> {compact(p.get('result'), limit)}"
    if t == "error":
        return f"{p.get('type')}: {compact(p.get('message'), limit)}"
    if t == "state":
        return f"{p.get('key')} = {compact(p.get('value'), limit)}"
    if t == "log":
        return compact(p.get("message", p), limit)
    return compact(p, limit)


def timeline_row(trace: Trace, e: Event, cursor: int | None = None, width: int = 80) -> Text:
    mark = ">" if cursor == e.id else " "
    row = Text(f"{mark}{e.id:>4} ", style="bold" if mark == ">" else "dim", no_wrap=True, overflow="ellipsis")
    row.append(f"{e.type:<13}", style=TYPE_STYLE.get(e.type, ""))
    row.append(" " + oneline(e, width))
    lat = e.meta.get("latency_ms") if isinstance(e.meta, dict) else None
    if lat is not None:
        row.append(f"  {lat:.0f}ms", style="dim")
    return row


def timeline(trace: Trace, events: Iterable[Event], cursor: int | None = None, width: int = 80) -> RenderableType:
    return Group(*[timeline_row(trace, e, cursor, width) for e in events])


def summary_line(trace: Trace) -> Text:
    """One-line run overview for the top of the replay."""
    s = trace.summary()
    style = {"ok": "green", "error": "bold red", "crashed": "bold red"}.get(s["status"], "")
    t = Text(f"{s['name']}  ", style="bold")
    t.append(s["status"], style=style)
    ntools = sum(s["tool_calls"].values())
    errs = s["errors"] + s["tool_errors"]
    t.append(
        f"  |  {s['events']} events, {s['llm_calls']} model calls, {ntools} tool calls, "
        f"{errs} errors, {s['duration_s']}s",
        style="dim",
    )
    return t


# ------------------------------------------------------------ full views


def show_event(
    trace: Trace, e: Event, *, raw: bool = False, full: bool = False, highlight: str | None = None
) -> RenderableType:
    limit = None if full else 1500
    title = Text(f"#{e.id} ", style="bold")
    title.append(e.type, style=TYPE_STYLE.get(e.type, ""))
    title.append(f"   {offset(trace, e)}", style="dim")
    if e.parent is not None:
        title.append(f"   parent #{e.parent}", style="dim")

    if raw:
        return Panel(_text(pretty(e.raw), highlight=highlight), title=title, title_align="left")

    p = e.payload
    body: list[RenderableType] = []
    t = e.type

    if t == "llm_request":
        body.append(Text(f"model: {p.get('model')}   provider: {p.get('provider')}"))
        total = len(trace.messages(e.id))
        if "messages" in p:
            new, start = p["messages"], 0
            body.append(Text(f"{total} messages", style="dim"))
        else:
            new = p.get("messages_append", [])
            start = total - len(new)
            body.append(Text(f"{total} messages ({start} carried from #{p.get('base')}, {len(new)} new)", style="dim"))
        if "system" in p:
            body.append(Text("system (changed):", style="bold"))
            body.append(_text(clip(pretty(p["system"]), limit), highlight=highlight))
        if "tools" in p:
            names = [_tool_name(x) for x in p["tools"] or []]
            body.append(Text(f"tools (changed): {', '.join(names)}", style="bold"))
        for i, m in enumerate(new, start=start):
            body.append(render_message(i, m, limit, highlight))
        params = {k: v for k, v in (p.get("params") or {}).items() if v is not None}
        if params:
            body.append(Text(f"params: {compact(params, 200)}", style="dim"))
    elif t == "llm_response":
        if p.get("text"):
            body.append(_text(clip(p["text"], limit), highlight=highlight))
        for tc in p.get("tool_calls") or []:
            body.append(_text(f"-> {tc.get('name')}({pretty(tc.get('arguments'))})", "yellow", highlight))
        tok = e.meta.get("tokens") or {}
        body.append(
            Text(
                f"stop: {p.get('stop_reason')}   tokens in/out: {tok.get('input')}/{tok.get('output')}"
                f"   latency: {e.meta.get('latency_ms')} ms",
                style="dim",
            )
        )
    elif t == "tool_call":
        body.append(Text(p.get("name", "?"), style="bold yellow"))
        body.append(_text(clip(pretty(p.get("arguments")), limit), highlight=highlight))
        if e.meta.get("requested_by") is not None:
            body.append(Text(f"requested by model at #{e.meta['requested_by']} (call id {p.get('call_id')})", style="dim"))
    elif t == "tool_result":
        if "error" in p:
            err = p["error"]
            body.append(Text(f"{err.get('type')}: {err.get('message')}", style="bold red"))
            if full and err.get("traceback"):
                body.append(Text(err["traceback"], style="dim"))
        else:
            body.append(_text(clip(pretty(p.get("result")), limit), highlight=highlight))
        body.append(Text(f"latency: {e.meta.get('latency_ms')} ms", style="dim"))
    elif t == "error":
        body.append(Text(f"{p.get('type')}: {p.get('message')}", style="bold red"))
        if p.get("traceback"):
            body.append(Text(clip(p["traceback"], limit), style="dim"))
    elif t == "run_start":
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="dim")
        grid.add_column()
        grid.add_row("run", f"{p.get('name')} ({p.get('run_id')})")
        grid.add_row("started", str(p.get("started_at")))
        grid.add_row("command", " ".join(p.get("argv") or []))
        grid.add_row("python", str(p.get("python")))
        if p.get("tags"):
            grid.add_row("tags", compact(p["tags"], 200))
        body.append(grid)
    elif t == "run_end":
        body.append(Text(f"status: {p.get('status')}   duration: {p.get('duration_ms')} ms"))
    elif t == "state":
        body.append(Text(str(p.get("key")), style="bold magenta"))
        body.append(_text(clip(pretty(p.get("value")), limit), highlight=highlight))
    else:
        body.append(_text(clip(pretty(p), limit), highlight=highlight))

    return Panel(Group(*body), title=title, title_align="left")


def _tool_name(t: Any) -> str:
    if isinstance(t, dict):
        return t.get("name") or (t.get("function") or {}).get("name") or "?"
    return str(t)


def show_context(
    trace: Trace, at: int, *, full: bool = False, highlight: str | None = None
) -> RenderableType:
    ctx = trace.context(at)
    limit = None if full else 800
    out: list[RenderableType] = []
    if ctx.request_id is None:
        out.append(Text(f"No model call at or before #{at}. Nothing in context yet.", style="dim"))
        return Panel(Group(*out), title=f"context at #{at}", title_align="left")

    out.append(Text(f"From request #{ctx.request_id}: {len(ctx.messages)} messages", style="dim"))
    if ctx.system:
        out.append(Rule("system", style="dim"))
        out.append(_text(clip(pretty(ctx.system), limit), highlight=highlight))
    if ctx.tools:
        out.append(Text("tools: " + ", ".join(_tool_name(t) for t in ctx.tools), style="dim"))
    out.append(Rule("messages", style="dim"))
    for i, m in enumerate(ctx.messages):
        out.append(render_message(i, m, limit, highlight))
    if ctx.response is not None:
        out.append(Rule(f"model reply #{ctx.response.id}", style="cyan"))
        r = ctx.response.payload
        if r.get("text"):
            out.append(_text(clip(r["text"], limit), highlight=highlight))
        for tc in r.get("tool_calls") or []:
            out.append(_text(f"-> {tc.get('name')}({compact(tc.get('arguments'), 2000)})", "yellow", highlight))
    later = [e for e in ctx.since if ctx.response is None or e.id > ctx.response.id]
    if later:
        out.append(Rule("since then (not yet seen by the model)", style="dim"))
        out.append(timeline(trace, later))
    return Panel(Group(*out), title=f"context at #{at}", title_align="left")


def show_grep(trace: Trace, term: str) -> RenderableType:
    hits = trace.grep(term)
    if not hits:
        return Text(f'No events contain "{term}".', style="dim")
    rows: list[RenderableType] = [
        Text(f'{len(hits)} events contain "{term}". First appears at #{hits[0].event.id}.', style="bold")
    ]
    for h in hits:
        e = h.event
        row = Text(f"{e.id:>5} ", style="bold")
        row.append(f"{e.type:<13}", style=TYPE_STYLE.get(e.type, ""))
        label = e.payload.get("name") or e.payload.get("key") or e.payload.get("model") or ""
        row.append(f" {label}  " if label else " ")
        row.append(h.paths[0] if h.paths else "", style="dim")
        if h.first:
            row.append("   <- entered here", style="bold yellow")
        rows.append(row)
        rows.append(_text("      " + compact(h.snippets[0], 160), "", term))
    return Group(*rows)


def show_diff(trace: Trace, a: int, b: int, *, highlight: str | None = None) -> RenderableType:
    if a > b:
        a, b = b, a
    ca, cb = trace.context(a), trace.context(b)
    out: list[RenderableType] = []
    header = Text(f"#{a} -> #{b}", style="bold")
    out.append(header)

    between = [e for e in trace.events if a < e.id <= b]
    out.append(Rule(f"{len(between)} events between", style="dim"))
    out.append(timeline(trace, between[:40]))
    if len(between) > 40:
        out.append(Text(f"... {len(between) - 40} more", style="dim"))

    if ca.request_id == cb.request_id:
        out.append(Rule("context", style="dim"))
        out.append(Text("Model context did not change (no new model call in between).", style="dim"))
        return Group(*out)

    if json.dumps(ca.system, sort_keys=True) != json.dumps(cb.system, sort_keys=True):
        out.append(Rule("system prompt changed", style="yellow"))
        out.append(_udiff(pretty(ca.system or ""), pretty(cb.system or "")))
    ta = [_tool_name(t) for t in ca.tools or []]
    tb = [_tool_name(t) for t in cb.tools or []]
    if ta != tb:
        out.append(Rule("tools changed", style="yellow"))
        added = [t for t in tb if t not in ta]
        removed = [t for t in ta if t not in tb]
        if added:
            out.append(Text("+ " + ", ".join(added), style="green"))
        if removed:
            out.append(Text("- " + ", ".join(removed), style="red"))

    ma, mb = ca.messages, cb.messages
    ja = [json.dumps(m, sort_keys=True) for m in ma]
    jb = [json.dumps(m, sort_keys=True) for m in mb]
    n = len(ja)
    if n <= len(jb) and jb[:n] == ja:
        out.append(Rule(f"context: +{len(mb) - n} messages added, nothing removed", style="green"))
        for i in range(n, len(mb)):
            out.append(_prefixed(render_message(i, mb[i], 800, highlight), "+", "green"))
    else:
        common = 0
        while common < min(len(ja), len(jb)) and ja[common] == jb[common]:
            common += 1
        out.append(
            Rule(
                f"context rewritten: {common} messages kept, {len(ma) - common} dropped/changed, "
                f"{len(mb) - common} new",
                style="yellow",
            )
        )
        for i in range(common, len(ma)):
            out.append(_prefixed(render_message(i, ma[i], 300, None), "-", "red"))
        for i in range(common, len(mb)):
            out.append(_prefixed(render_message(i, mb[i], 800, highlight), "+", "green"))
    return Group(*out)


def _prefixed(r: RenderableType, mark: str, style: str) -> RenderableType:
    grid = Table.grid(padding=(0, 1))
    grid.add_column(width=1, style=style)
    grid.add_column(ratio=1)
    grid.add_row(mark, r)
    return grid


def _udiff(a: str, b: str) -> Text:
    t = Text()
    for line in difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm="", n=2):
        if line.startswith(("---", "+++")):
            continue
        style = "green" if line.startswith("+") else "red" if line.startswith("-") else "dim"
        t.append(line + "\n", style=style)
    return t


def show_summary(trace: Trace) -> RenderableType:
    s = trace.summary()
    tbl = Table.grid(padding=(0, 2))
    tbl.add_column(style="dim")
    tbl.add_column()
    status_style = {"ok": "green", "error": "bold red", "crashed": "bold red"}.get(s["status"], "")
    tbl.add_row("run", f"{s['name']}  ({s['run_id']})")
    tbl.add_row("file", escape(str(trace.path)) if trace.path else "-")
    tbl.add_row("status", Text(s["status"] + ("  (last line truncated)" if s["truncated"] else ""), style=status_style))
    tbl.add_row("events", str(s["events"]))
    tbl.add_row("duration", f"{s['duration_s']} s")
    tbl.add_row("model calls", f"{s['llm_calls']}   ({s['llm_ms']} ms total)")
    tbl.add_row("tokens", f"{s['tokens']['input']} in / {s['tokens']['output']} out")
    tools = ", ".join(f"{k} x{v}" for k, v in sorted(s["tool_calls"].items(), key=lambda kv: -kv[1])) or "none"
    tbl.add_row("tool calls", tools)
    tbl.add_row("tool errors", Text(str(s["tool_errors"]), style="red" if s["tool_errors"] else ""))
    tbl.add_row("errors", Text(str(s["errors"]), style="red" if s["errors"] else ""))
    errs = trace.of_type("error") + [e for e in trace.of_type("tool_result") if "error" in e.payload]
    parts: list[RenderableType] = [tbl]
    if errs:
        parts.append(Rule("errors", style="red"))
        parts.append(timeline(trace, sorted(errs, key=lambda e: e.id)))
    return Panel(Group(*parts), title="summary", title_align="left")


# ------------------------------------------------------------------- why


def _ratio(kept: int, n: int) -> str:
    return f"{kept}/{n}"


def _path(c) -> str:
    """Narrowing path of a cause, e.g. '#9 search_kb result > [1] > [1].text > sentence 2'."""
    parts = []
    for i, t in enumerate(c.chain):
        seg = t.removed[-1]
        label = seg.where if i == 0 else (seg.sub.split(" ")[-2] + " " + seg.sub.split(" ")[-1] if seg.span and " " in seg.sub else seg.sub)
        parts.append(f"{label} ({_ratio(t.kept, t.n)})")
    return " > ".join(parts)


def _p(p: float) -> str:
    return f"{p:.0e}".replace("e-0", "e-") if p < 0.001 else f"{p:.3f}"


def _evidence(rep, t) -> Text:
    b = rep.baseline
    if rep.deterministic:
        return Text(f"  Evidence: temperature 0, so each variant was rerun twice ({b.kept}/{b.n} with it, "
                    f"{t.kept}/{t.n} without).", style="dim")
    if t.p is None:
        return Text(f"  Evidence: {b.kept}/{b.n} reruns with it, {t.kept}/{t.n} without.", style="dim")
    return Text(
        f"  Evidence: {b.kept}/{b.n} reruns with it, {t.kept}/{t.n} without. p = {_p(t.p)}, significant after "
        f"correcting for all {t.family} variants tried.",
        style="dim",
    )


def _recheck(rep, c) -> Text | None:
    r = getattr(c, "recheck", None)
    if r is None:
        return None
    t = c.refined or c.finest
    if rep.deterministic:
        holds = r.kept == 0  # the same bar the cause itself had to clear
    else:
        holds = r.effect(rep.base) >= rep.threshold and r.p is not None and r.p <= rep.alpha
    if holds:
        return Text(f'  Same result with the removed text replaced by "{r.fill}" instead of "{rep.fill}": '
                    f"{_ratio(r.kept, r.n)}.", style="dim")
    return Text(f'  ! With "{r.fill}" in place of "{rep.fill}" the agent still {rep.target.describe()} in '
                f"{_ratio(r.kept, r.n)} reruns (vs {_ratio(t.kept, t.n)}). The replacement text itself may be "
                "steering the model. Try other replacement text with --fill.", style="yellow")


def show_why(rep, *, show_all: bool = False) -> RenderableType:
    t = rep.trace
    tgt = rep.target
    out: list[RenderableType] = []
    head = Text("Why does the agent ", style="bold")
    head.append(tgt.question(), style="bold yellow")
    head.append("?", style="bold")
    out.append(head)
    req = t[tgt.request_id]
    out.append(Text(
        f"decision at #{tgt.response_id} (the model reply), made on model call #{tgt.request_id} "
        f"({req.payload.get('model')}); {len(rep.trials)} pieces of its context tested",
        style="dim",
    ))
    b = rep.baseline
    stable = "green" if rep.base >= 0.8 else "yellow" if rep.base >= 0.6 else "red"
    line = Text("On the identical context it ")
    line.append(f"{tgt.describe()} in {_ratio(b.kept, b.n)} reruns", style=f"bold {stable}")
    if b.instead:
        alt, cnt = b.top_instead()
        line.append(f"   otherwise {alt} ({cnt})", style="dim")
    out.append(line)
    if rep.stopped:
        out.append(Text(f"! Stopped early ({rep.stopped}). The results below are incomplete: a cause may be "
                        "missing or not yet narrowed down. Raise --budget; finished reruns are cached.", style="bold yellow"))
    for w in rep.warnings:
        out.append(Text("! " + w, style="yellow"))

    decisive = [c for c in rep.causes if c.kind == "decisive"]
    prereq = [c for c in rep.causes if c.kind != "decisive"]

    for c in decisive:
        out.append(Rule(style="red"))
        if c.refined is not None:
            # the precise answer first: the same content appears more than once and only removing
            # every copy changes the decision
            j = c.refined
            h = Text("CAUSE  ", style="bold red")
            h.append(" + ".join(seg.where for seg in j.removed), style="bold")
            out.append(h)
            texts = []
            for seg in j.removed:
                t_ = " ".join(seg.text.split())
                if t_ not in texts:
                    texts.append(t_)
            for t_ in texts:
                out.append(Text("  " + clip('"' + t_ + '"', 300)))
            if len(texts) == 1:
                why_note = (f"The same content appears {len(j.removed)} times in the context, so removing any one "
                            "copy changes nothing. Removing them all does:")
            else:
                why_note = ("Each of these is enough on its own: removing one changes nothing, removing all of "
                            "them does:")
            out.append(Text("  " + why_note, style="magenta"))
            r = Text("  Without them the agent ")
            r.append(f"{tgt.describe()} in {_ratio(j.kept, j.n)} reruns", style="bold")
            if j.top_instead():
                alt, cnt = j.top_instead()
                r.append(" and instead ")
                r.append(f"{alt} ({cnt}/{j.n})", style="bold green")
            out.append(r)
            out.append(_evidence(rep, j))
            if _recheck(rep, c):
                out.append(_recheck(rep, c))
            continue
        seg = c.finest.removed[-1]
        h = Text("CAUSE  ", style="bold red")
        h.append(seg.where, style="bold")
        if c.masked:
            h.append("   masked", style="bold magenta")
        out.append(h)
        out.append(Text("  " + clip('"' + " ".join(seg.text.split()) + '"', 400)))
        r = Text("  Without it the agent ")
        r.append(f"{tgt.describe()} in {_ratio(c.finest.kept, c.finest.n)} reruns", style="bold")
        if c.finest.top_instead():
            alt, cnt = c.finest.top_instead()
            r.append(" and instead ")
            r.append(f"{alt} ({cnt}/{c.finest.n})", style="bold green")
        out.append(r)
        out.append(_evidence(rep, c.finest))
        if _recheck(rep, c):
            out.append(_recheck(rep, c))
        if len(c.chain) > 1:
            out.append(Text("  Narrowed down: " + _path(c), style="dim"))
        if c.masked:
            out.append(Text(
                "  Masked: removing all of " + c.top.removed[-1].where + " changes nothing, because other parts of it "
                "push the other way. Removing whole messages or tool results would never find this.",
                style="magenta",
            ))

    if rep.joint is not None and rep.joint.kind == "decisive":
        j = rep.joint.finest
        out.append(Rule(style="red"))
        out.append(Text("CAUSE (combined)  no single piece changes it; removing all of these together does:",
                        style="bold red"))
        for seg in j.removed:
            out.append(Text(f"  {seg.where}  " + compact(seg.text, 90)))
        r = Text("  Without all of them the agent ")
        r.append(f"{tgt.describe()} in {_ratio(j.kept, j.n)} reruns", style="bold")
        if j.top_instead():
            r.append(f" and instead {j.top_instead()[0]}", style="green")
        out.append(r)
        out.append(_evidence(rep, j))
        if _recheck(rep, rep.joint):
            out.append(_recheck(rep, rep.joint))

    if rep.joint is not None and rep.joint.kind != "decisive":
        prereq = prereq + [rep.joint]
    if prereq:
        out.append(Rule("also required (without these the agent stops or redoes a step)", style="dim"))
        for c in prereq:
            where = " + ".join(x.where for x in c.finest.removed)
            alt = c.finest.top_instead()
            row = Text(f"  {where}  ", style="bold")
            row.append(f"{_ratio(c.finest.kept, c.finest.n)}", style="dim")
            if alt:
                row.append(f"  instead: {compact(alt[0], 70)}", style="dim")
            out.append(row)

    headline = decisive or (rep.joint is not None and rep.joint.kind == "decisive")
    if not headline and not rep.stopped and rep.baseline.kept:
        out.append(Rule(style="yellow"))
        gaps = []
        if rep.untested:
            gaps.append(f"{len(rep.untested)} less suspicious pieces were not tested (--max-pieces)")
        if rep.unexpanded:
            gaps.append(f"{rep.unexpanded} pieces were not looked inside for masked causes (--expand)")
        if prereq:
            base_msg = ("Nothing in the context steers this decision: removing the pieces above only makes the "
                        "agent fetch them again or stop.")
        else:
            base_msg = "No piece of the context changes this decision when removed."
        if gaps:
            msg = base_msg + " Not fully checked: " + "; ".join(gaps) + "."
        else:
            msg = base_msg + " It comes from the model's own judgment given the task, not from something it read."
        out.append(Text(msg, style="yellow"))

    shown = {id(c.top) for c in rep.causes}
    weak = [x for x in rep.trials if rep.verdict(x) == "not significant" and id(x) not in shown]
    if weak and show_all:
        out.append(Rule("changed the decision in some reruns, but not significantly", style="dim"))
        for x in weak:
            out.append(Text(f"  {x.label}  {_ratio(x.kept, x.n)}" + (f"  p = {_p(x.p)}" if x.p is not None else ""),
                            style="yellow"))
    rest = [x for x in rep.trials if rep.verdict(x) == "none" and id(x) not in shown]
    if rest:
        if show_all:
            out.append(Rule("no effect when removed", style="dim"))
            for x in rest:
                out.append(Text(f"  {x.label}  {_ratio(x.kept, x.n)}   {compact(x.removed[0].text, 50)}", style="dim"))
        else:
            out.append(Text(f"No effect when removed: {len(rest)} other pieces (--all to list).", style="dim"))
    if rep.untested and (rep.causes or rep.joint):
        out.append(Text(f"Not tested: {len(rep.untested)} least suspicious pieces (raise --max-pieces to include them).", style="dim"))
    if getattr(rep, "depth", "full") == "quick" and (rep.causes or rep.joint):
        out.append(Text("Stopped at the main cause. --full also looks for causes hidden inside other pieces and for "
                        "combinations of pieces (reruns made so far are reused).", style="dim"))
    in_requests = f" in {rep.requests} requests" if 0 < getattr(rep, "requests", 0) < rep.calls else ""
    tail = f"{rep.calls} model calls{in_requests}, {rep.cache_hits} from cache."
    if rep.stopped:
        tail += f" Stopped early: {rep.stopped}. Raise --budget to finish."
    out.append(Text(tail, style="dim"))
    return Group(*out)


def show_distribution(dist, recorded, *, title: str) -> RenderableType:
    out: list[RenderableType] = [Text(title, style="bold")]
    if recorded is not None:
        out.append(Text("recorded:  " + recorded.describe(120), style="dim"))
    n = len(dist)
    for desc, cnt in dist.counts().most_common():
        same = recorded is not None and desc == recorded.describe(200)
        row = Text(f"  {cnt}/{n}  ", style="bold")
        row.append(compact(desc, 120), style="cyan" if same else "green")
        if same:
            row.append("  (same as recorded)", style="dim")
        out.append(row)
    return Group(*out)


def show_fix(fr) -> RenderableType:
    """The fixes runtape tried for a decision, and how each did on the recorded context."""
    rep = fr.report
    tgt = rep.target
    b = rep.baseline
    out: list[RenderableType] = []
    head = Text("Fixes for: the agent would ", style="bold")
    head.append(tgt.question(), style="bold yellow")
    out.append(head)
    out.append(Text(f"Without a fix it does this in {_ratio(b.kept, b.n)} reruns of the recorded context.", style="dim"))
    for w in rep.warnings:  # an unstable decision or a stopped search changes how to read everything below
        out.append(Text("! " + w, style="yellow"))
    if fr.cause is None and b.kept:
        out.append(Text("! No cause was found in the context, so only fixes that don't depend on one are tried.",
                        style="yellow"))
    if fr.cause is not None:
        t = fr.cause.refined or fr.cause.finest
        c = Text("Cause: ", style="bold")
        c.append(" + ".join(s.where for s in t.removed), style="bold red")
        out.append(c)
        out.append(Text("  " + clip('"' + " ".join(" ".join(s.text.split()) for s in t.removed) + '"', 300)))
    if b.kept == 0:
        out.append(Text("The model doesn't repeat this decision, so there is nothing to fix or measure.", style="yellow"))
        return Group(*out)
    if not fr.candidates:
        out.append(Text("No cause was found to base a fix on. Test a change directly with runtape rerun.", style="yellow"))
        return Group(*out)
    out.append(Rule(style="dim"))
    goal = fr.without_cause()
    if goal:
        out.append(Text(f"Without the cause, the agent instead {clip(goal, 150)}", style="dim"))
    for c in fr.candidates:
        ok = c.holds()
        if not c.complete:
            row = Text("  ----  ", style="bold yellow")
            row.append(f"{c.name:<18}", style="bold")
            row.append(f"not fully checked ({c.n} reruns before the budget ran out)", style="yellow")
            out.append(row)
            continue
        label, style = ("  PASS  ", "bold green") if ok else (("  PART  ", "bold yellow") if c.partial()
                                                             else ("  FAIL  ", "bold red"))
        row = Text(label, style=style)
        row.append(f"{c.name:<18}", style="bold")
        row.append(f"{tgt.describe()} in {_ratio(c.kept, c.n)} reruns")
        if c.p is not None and ok:
            row.append(f"  (p = {_p(c.p)})", style="dim")
        top = c.instead.most_common(1)
        if top and goal and top[0][0] == goal:
            row.append("  same as without the cause", style="dim")
        out.append(row)
        if top and not (goal and top[0][0] == goal):
            out.append(Text(f"        instead: {clip(top[0][0], 110)}", style="dim"))
    best = fr.best
    out.append(Rule(style="dim"))
    if best is None:
        out.append(Text("No fix removed the bad call in every rerun. PART means it became rarer but still "
                        "happened. Try your own with runtape rerun --system-file or --replace.", style="bold yellow"))
    elif best.add_system:
        out.append(Text("Recommended: add this to the system prompt", style="bold green"))
        for line in best.add_system.split("\n"):
            out.append(Text("  " + line))
    else:
        out.append(Text("Recommended: " + best.change, style="bold green"))
    if rep.stopped:
        out.append(Text(f"! Finding the cause stopped early ({rep.stopped}); the cause above may be incomplete.",
                        style="yellow"))
    if fr.stopped:
        out.append(Text(f"! Stopped early ({fr.stopped}); some fixes were not fully checked. Raise --budget; "
                        "finished reruns are cached.", style="yellow"))
    out.append(Text(f"{rep.calls + fr.calls} model calls ({rep.calls} finding the cause, {fr.calls} checking fixes), "
                    f"{rep.cache_hits + fr.cache_hits} from cache.", style="dim"))
    return Group(*out)
