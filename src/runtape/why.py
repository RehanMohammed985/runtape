"""Causal attribution for agent decisions.

runtape why answers "which part of the context made the agent do this?" by
experiment instead of by guessing: it removes pieces of the context the model
saw at a decision, re-runs that one decision several times per variant, and
reports which removals change what the agent does, down to the JSON item or
sentence, and what it does instead.

The agent itself is never re-run, so no tool is called and nothing is repeated.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

from .rerun import (
    BudgetExceeded,
    Model,
    Reply,
    Sampler,
    build_request,
    is_deterministic,
    model_for,
    request_for,
)
from .segments import Segment, ablate, extract, overlap
from .trace import Trace

# ------------------------------------------------------------------ targets


@dataclass
class Target:
    """The decision being explained, and how to tell if a re-run made the same one."""

    request_id: int
    response_id: int | None
    recorded: Reply
    mode: str  # tool | tools | match | judge
    tool: str | None = None
    args: dict | None = None  # when set, the tool must be called with these exact arguments
    pattern: re.Pattern | None = None
    judge: Callable[[Reply, Reply], bool] | None = None

    def matches(self, r: Reply) -> bool:
        if self.mode == "tool":
            for tc in r.tool_calls:
                if tc.get("name") == self.tool and (self.args is None or _canon(tc.get("arguments")) == _canon(self.args)):
                    return True
            return False
        if self.mode == "tools":
            return sorted(tc.get("name") for tc in r.tool_calls) == sorted(tc.get("name") for tc in self.recorded.tool_calls)
        if self.mode == "match":
            return bool(self.pattern.search(_reply_text(r)))
        if self.mode == "judge":
            return bool(self.judge(self.recorded, r))
        raise ValueError(self.mode)

    def describe(self) -> str:
        if self.mode == "tool":
            if self.args is not None:
                return f"calls {self.tool}({_fmt_args(self.args)})"
            return f"calls {self.tool}"
        if self.mode == "tools":
            return "calls " + " + ".join(sorted(tc.get("name") for tc in self.recorded.tool_calls))
        if self.mode == "match":
            return f"output matches /{self.pattern.pattern}/"
        return "gives the same answer: " + self.recorded.describe(70).removeprefix("replies: ")

    def decision_text(self) -> str:
        return _reply_text(self.recorded)


def _canon(v) -> str:
    return json.dumps(v, sort_keys=True, ensure_ascii=False)


def _fmt_args(a) -> str:
    if isinstance(a, dict):
        return ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in a.items())
    return json.dumps(a, ensure_ascii=False)


def _reply_text(r: Reply) -> str:
    parts = [r.text or ""]
    for tc in r.tool_calls:
        parts.append(f"{tc.get('name')} {_canon(tc.get('arguments'))}")
    return "\n".join(parts)


def make_target(
    trace: Trace,
    event_id: int,
    *,
    tool: str | None = None,
    match: str | None = None,
    exact_args: bool = False,
    judge: Callable[[Reply, Reply], bool] | None = None,
) -> Target:
    """Work out what 'the same decision' means for an event.

    - a tool_call event: does the model still call that tool
    - a model reply that called one tool: does it still call that tool
    - a model reply that called several: does it call the same set
    - a text-only reply: a judge decides if the answer is equivalent
    - match=REGEX overrides all of that
    """
    rid, resp_id = request_for(trace, event_id)
    if resp_id is None:
        raise ValueError(f"#{rid} has no recorded reply to explain")
    rp = trace[resp_id].payload
    recorded = Reply(rp.get("text"), list(rp.get("tool_calls") or []), rp.get("stop_reason"))
    ev = trace[event_id]
    if match:
        pat = re.compile(match, re.I)
        if not pat.search(_reply_text(recorded)):
            raise ValueError(f"the recorded reply at #{resp_id} doesn't match /{match}/ to begin with")
        return Target(rid, resp_id, recorded, "match", pattern=pat)
    if ev.type == "tool_call" and not tool:
        tool = ev.payload.get("name")
        args = ev.payload.get("arguments") if exact_args else None
        return Target(rid, resp_id, recorded, "tool", tool=tool, args=args)
    names = sorted({tc.get("name") for tc in recorded.tool_calls})
    if tool:
        if tool not in names:
            raise ValueError(f"the reply at #{resp_id} doesn't call {tool} (it calls: {', '.join(names) or 'nothing'})")
        args = next(tc.get("arguments") for tc in recorded.tool_calls if tc.get("name") == tool) if exact_args else None
        return Target(rid, resp_id, recorded, "tool", tool=tool, args=args)
    if len(names) == 1:
        args = recorded.tool_calls[0].get("arguments") if exact_args else None
        return Target(rid, resp_id, recorded, "tool", tool=names[0], args=args)
    if names:
        return Target(rid, resp_id, recorded, "tools")
    return Target(rid, resp_id, recorded, "judge", judge=judge)


def llm_judge(sampler: Sampler, like: dict) -> Callable[[Reply, Reply], bool]:
    """Judge text answers with the same model, in the same API format as the original call."""

    def judge(a: Reply, b: Reply) -> bool:
        prompt = (
            "Two answers from an AI agent to the same situation are below. Do they make the same decision "
            "(same action, same conclusion, same key facts), ignoring wording? Answer with one word: SAME or DIFFERENT.\n\n"
            f"ANSWER A:\n{a.text}\n\nANSWER B:\n{b.text}"
        )
        req = {
            "provider": like.get("provider"),
            "api": like.get("api"),
            "model": like.get("model"),
            "system": None,
            "tools": None,
            "messages": [{"role": "user", "content": prompt}],
            "params": {"max_tokens": 5, "temperature": 0},
        }
        out = sampler.one(req, 0)
        return "SAME" in (out.text or "").upper() and "DIFFERENT" not in (out.text or "").upper()

    return judge


# ------------------------------------------------------------------ results


@dataclass
class Trial:
    removed: list[Segment]
    kept: int = 0  # runs where the decision still happened
    n: int = 0
    instead: Counter = field(default_factory=Counter)  # what it did when the decision didn't happen

    @property
    def rate(self) -> float:
        return self.kept / self.n if self.n else 0.0

    def effect(self, base: float) -> float:
        return base - self.rate

    @property
    def label(self) -> str:
        return " + ".join(s.where for s in self.removed) or "(nothing removed)"

    def top_instead(self) -> tuple[str, int] | None:
        return self.instead.most_common(1)[0] if self.instead else None


@dataclass
class Cause:
    chain: list[Trial]  # from the top-level piece down to the most specific one that still matters
    # True when removing the whole top-level piece does NOT change the decision but a part of it does:
    # the piece also holds content pushing the other way, which hides the cause from coarse ablation.
    masked: bool = False

    @property
    def finest(self) -> Trial:
        return self.chain[-1]

    @property
    def kind(self) -> str:
        """decisive: without it the agent does something else.
        prerequisite: without it the agent can't act at all (asks, stalls, gives up)."""
        top = self.finest.top_instead()
        return "decisive" if top and top[0].startswith("calls ") else "prerequisite"

    @property
    def top(self) -> Trial:
        return self.chain[0]


@dataclass
class Report:
    trace: Trace
    target: Target
    baseline: Trial
    trials: list[Trial] = field(default_factory=list)  # one per top-level piece tested alone
    causes: list[Cause] = field(default_factory=list)
    joint: Cause | None = None  # set when only a combination of pieces flips the decision
    warnings: list[str] = field(default_factory=list)
    calls: int = 0
    cache_hits: int = 0
    k: int = 0
    stopped: str | None = None  # why the search ended early, if it did
    untested: list = field(default_factory=list)  # least suspicious pieces skipped by --max-pieces
    threshold: float = 0.5

    @property
    def base(self) -> float:
        return self.baseline.rate

    def verdict(self, t: Trial) -> str:
        e = t.effect(self.base)
        if e >= self.threshold:
            return "cause"
        if e >= 0.2:
            return "partial"
        return "none"

    def to_dict(self) -> dict:
        def seg(s: Segment) -> dict:
            return {
                "where": s.where, "origin": s.origin, "kind": s.kind, "tool": s.name,
                "part": s.sub or None, "text": s.text,
            }

        def trial(t: Trial) -> dict:
            return {
                "removed": [seg(s) for s in t.removed],
                "still_happens": t.kept, "runs": t.n, "effect": round(t.effect(self.base), 3),
                "verdict": self.verdict(t), "instead": dict(t.instead),
            }

        return {
            "decision": {"request": self.target.request_id, "response": self.target.response_id,
                         "outcome": self.target.describe()},
            "baseline": {"happens": self.baseline.kept, "runs": self.baseline.n},
            "causes": [{"kind": c.kind, "masked": c.masked, "chain": [trial(t) for t in c.chain]} for c in self.causes],
            "joint_cause": [trial(t) for t in self.joint.chain] if self.joint else None,
            "tested": [trial(t) for t in self.trials],
            "warnings": self.warnings, "model_calls": self.calls, "cache_hits": self.cache_hits,
            "stopped": self.stopped,
            "untested": [seg(x) for x in self.untested],
        }


# ------------------------------------------------------------------- engine


class Why:
    def __init__(
        self,
        trace: Trace,
        target: Target,
        sampler: Sampler,
        *,
        k: int = 5,
        screen: int = 2,
        threshold: float = 0.5,
        max_depth: int = 4,
        max_causes: int = 3,
        expand: int = 3,
        max_pieces: int | None = 40,
        joint_search: bool = True,
        progress: Callable[[str], None] | None = None,
    ):
        self.trace = trace
        self.target = target
        self.sampler = sampler
        self.req = build_request(trace, target.request_id)
        det = is_deterministic(self.req)
        self.k = 1 if det else max(1, k)
        self.screen = 1 if det else max(1, min(screen, self.k))
        self.threshold = threshold
        self.max_depth = max_depth
        self.max_causes = max_causes
        self.expand = expand
        self.max_pieces = max_pieces
        self.joint_search = joint_search
        self.progress = progress or (lambda msg: None)
        self._decision = target.decision_text()

    # -- running trials

    def _score(self, trial: Trial, replies: list[Reply]) -> None:
        for r in replies:
            trial.n += 1
            if self.target.matches(r):
                trial.kept += 1
            else:
                trial.instead[r.describe()] += 1

    def _trials(self, removals: list[list[Segment]]) -> list[Trial]:
        """Screen each removal with a few runs, then finish the ones that look like they matter."""
        trials = [Trial(list(rm)) for rm in removals]
        reqs = [ablate(self.req, rm) if rm else self.req for rm in removals]
        first = self.sampler.many((r, i) for r in reqs for i in range(self.screen))
        for n, t in enumerate(trials):
            self._score(t, first[n * self.screen : (n + 1) * self.screen])
        todo = [n for n, t in enumerate(trials) if t.kept < t.n and self.screen < self.k]
        if todo:
            extra = self.k - self.screen
            more = self.sampler.many((reqs[n], i) for n in todo for i in range(self.screen, self.k))
            for j, n in enumerate(todo):
                self._score(trials[n], more[j * extra : (j + 1) * extra])
        return trials

    def _full(self, removal: list[Segment]) -> Trial:
        t = Trial(list(removal))
        req = ablate(self.req, removal) if removal else self.req
        self._score(t, self.sampler.samples(req, self.k))
        return t

    def _is_cause(self, t: Trial, base: float) -> bool:
        return t.effect(base) >= self.threshold

    def _rank(self, segs: list[Segment]) -> list[Segment]:
        # most suspicious first: shares wording with the decision, then most recent
        n = len(segs)
        return [s for _, _, s in sorted(
            ((-overlap(s.text, self._decision), -(i / max(n, 1)), s) for i, s in enumerate(segs)),
            key=lambda x: (x[0], x[1]),
        )]

    # -- search

    def run(self) -> Report:
        self.progress("rerunning the recorded decision")
        baseline = self._full([])
        rep = Report(self.trace, self.target, baseline, k=self.k, threshold=self.threshold)
        try:
            self._search(rep)
        except BudgetExceeded as e:
            rep.stopped = str(e)
        rep.calls = self.sampler.calls
        rep.cache_hits = self.sampler.hits
        return rep

    def _search(self, rep: Report) -> None:
        base = rep.base
        if rep.baseline.kept == 0:
            rep.warnings.append(
                f"The model never repeated this decision in {rep.baseline.n} reruns of the exact same context, "
                "so there is nothing to attribute. The original was a rare outcome, or the model/settings changed."
            )
            return
        if base < 0.6:
            rep.warnings.append(
                f"Unstable decision: the model only repeats it in {rep.baseline.kept}/{rep.baseline.n} reruns of the "
                "same context. Results below are noisy; raise --runs for a clearer answer."
            )
        segs = self._rank(extract(self.req, self.trace, self.target.request_id))
        if not segs:
            rep.warnings.append("No removable context found in this request.")
            return
        if self.max_pieces and len(segs) > self.max_pieces:
            rep.untested = segs[self.max_pieces :]
            segs = segs[: self.max_pieces]

        self.progress(f"testing {len(segs)} pieces of context")
        rep.trials = self._trials([[s] for s in segs])
        singles = sorted((t for t in rep.trials if self._is_cause(t, base)), key=lambda t: -t.effect(base))

        for t in singles[: self.max_causes]:
            self.progress(f"narrowing down {t.removed[0].where}")
            rep.causes.append(Cause(self._drill(t, [], base)))

        # A piece can hold the cause and evidence against it at once (a search result with a stale
        # doc next to the real policy). Removing the whole piece then changes nothing, so look inside
        # the most suspicious pieces that didn't flip.
        others = [t for t in rep.trials if not self._is_cause(t, base) and t.removed[0].children()]
        for t in others[: self.expand]:
            self.progress(f"looking inside {t.removed[0].where}")
            chain = self._drill(t, [], base)
            if len(chain) > 1:
                rep.causes.append(Cause(chain, masked=True))

        rep.causes.sort(key=lambda c: (c.kind != "decisive", -c.finest.effect(base)))
        if not any(c.kind == "decisive" for c in rep.causes) and self.joint_search:
            # search combinations at the level of parts inside the most suspicious pieces too,
            # so two copies of the same bad doc in one result can be found together
            expanded = {id(t.removed[0]) for t in others[: self.expand]}
            # hold needed inputs fixed: the question is what makes the agent act differently,
            # not what makes it unable to act
            needed = {id(c.top.removed[0]) for c in rep.causes}
            cands: list[Segment] = []
            for sg in segs:
                if id(sg) in needed:
                    continue
                kids = sg.children() if id(sg) in expanded else []
                cands.extend(kids or [sg])
            if len(cands) > 1:
                self.progress("no single piece explains it; searching combinations")
                joint = self._joint(cands, base)
            else:
                joint = None
            if joint is not None:
                rep.joint = Cause([joint])

    def _drill(self, top: Trial, held: list[Segment], base: float) -> list[Trial]:
        """Follow a cause down into smaller and smaller pieces while they still flip the decision."""
        chain = [top]
        cur = top
        while len(chain) <= self.max_depth:
            seg = cur.removed[-1]
            kids = self._rank(seg.children())
            if not kids:
                break
            trials = self._trials([held + [k] for k in kids])
            hits = sorted((t for t in trials if self._is_cause(t, base)), key=lambda t: -t.effect(base))
            if not hits:
                break
            cur = hits[0]
            chain.append(cur)
        return chain

    def _joint(self, cands: list[Segment], base: float, top_m: int = 6) -> Trial | None:
        """Find a small set of pieces that only change the decision together.

        Removing everything is not guaranteed to flip the decision (it also removes evidence
        pointing the other way), so try the full set, then the most suspicious few, then pairs.
        """
        found = self._ddmin(cands, base)
        if found is not None:
            return found
        top = self._rank(cands)[:top_m]
        if len(top) > 1 and len(top) < len(cands):
            found = self._ddmin(top, base)
            if found is not None:
                return found
        pairs = [[a, b] for i, a in enumerate(top) for b in top[i + 1 :]]
        hits = [t for t in self._trials(pairs) if self._is_cause(t, base)]
        return max(hits, key=lambda t: t.effect(base)) if hits else None

    def _ddmin(self, segs: list[Segment], base: float) -> Trial | None:
        """Smallest set of pieces whose joint removal flips the decision (delta debugging)."""
        cache: dict[tuple, Trial] = {}

        def test(sub: list[Segment]) -> Trial:
            key = tuple(sorted(id(s) for s in sub))
            if key not in cache:
                cache[key] = self._full(sub)
            return cache[key]

        whole = test(segs)
        if not self._is_cause(whole, base):
            return None
        cur, n = list(segs), 2
        while len(cur) >= 2:
            size = max(1, len(cur) // n)
            chunks = [cur[i : i + size] for i in range(0, len(cur), size)]
            progressed = False
            for ch in chunks:
                if self._is_cause(test(ch), base):
                    cur, n, progressed = ch, 2, True
                    break
            if not progressed:
                for ch in chunks:
                    comp = [s for s in cur if s not in ch]
                    if comp and self._is_cause(test(comp), base):
                        cur, n, progressed = comp, max(n - 1, 2), True
                        break
            if not progressed:
                if n >= len(cur):
                    break
                n = min(len(cur), 2 * n)
        return test(cur)


# ----------------------------------------------------------------- facade


def why(
    trace: Trace,
    event_id: int,
    *,
    model: Model | None = None,
    runs: int = 5,
    screen: int = 2,
    tool: str | None = None,
    match: str | None = None,
    exact_args: bool = False,
    budget: int | None = 300,
    cache_dir: str | None = ".runtape/cache",
    workers: int = 8,
    threshold: float = 0.5,
    max_pieces: int | None = 40,
    progress: Callable[[str], None] | None = None,
    on_call: Callable[[], None] | None = None,
) -> Report:
    """Explain a decision in a trace. See module docstring."""
    rid, _ = request_for(trace, event_id)
    req = build_request(trace, rid)
    sampler = Sampler(model or model_for(req), cache_dir=cache_dir, budget=budget, workers=workers, on_call=on_call)
    target = make_target(trace, event_id, tool=tool, match=match, exact_args=exact_args)
    if target.mode == "judge":
        target.judge = llm_judge(sampler, req)
    return Why(trace, target, sampler, k=runs, screen=screen, threshold=threshold, max_pieces=max_pieces,
               progress=progress).run()


def estimate_calls(trace: Trace, event_id: int, runs: int = 5, screen: int = 2, max_pieces: int | None = 40) -> tuple[int, int]:
    """(likely, worst case) number of model calls a why run will make."""
    rid, _ = request_for(trace, event_id)
    req = build_request(trace, rid)
    n = len(extract(req, trace, rid))
    if max_pieces:
        n = min(n, max_pieces)
    if is_deterministic(req):
        return 1 + n + 6, 1 + n * 2 + 30
    return runs + n * screen + 3 * runs * 3, runs + n * runs + 60 * runs


def suspects(trace: Trace, event_id: int, top: int = 10) -> list[tuple[float, Segment]]:
    """Pieces of context ranked by how much of the decision's wording they share. No model calls.

    A fast first guess; `why` is what proves which of these actually matter.
    """
    tgt = make_target(trace, event_id, judge=lambda a, b: True)
    req = build_request(trace, tgt.request_id)
    decision = tgt.decision_text()
    out: list[tuple[float, Segment]] = []
    for seg in extract(req, trace, tgt.request_id):
        best = (overlap(seg.text, decision), seg)
        # descend to the most specific part that keeps most of the match
        for _ in range(4):
            kids = [(overlap(k.text, decision), k) for k in best[1].children()]
            if not kids:
                break
            top_kid = max(kids, key=lambda x: x[0])
            if top_kid[0] < 0.7 * best[0] or top_kid[0] == 0:
                break
            best = top_kid
        out.append(best)
    out.sort(key=lambda x: -x[0])
    return [x for x in out[:top] if x[0] > 0]
