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
from math import comb
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


def _judge_request(like: dict, prompt: str) -> dict:
    """A small request in the same API family as the original call, with no sampling settings
    (some models reject temperature or max_tokens)."""
    api = like.get("api")
    if api == "langchain":
        api = "messages" if like.get("provider") == "anthropic" else "chat.completions"
    params = {"messages": {"max_tokens": 16}, "responses": {"max_output_tokens": 16}}.get(api, {})
    return {
        "provider": like.get("provider"), "api": api, "model": like.get("model"), "system": None, "tools": None,
        "messages": [{"role": "user", "content": prompt}], "params": params,
    }


def llm_judge(sampler: Sampler, like: dict) -> Callable[[Reply, Reply], bool]:
    """Judge text answers with the same model the agent used."""
    seen: dict[tuple, bool] = {}

    def judge(a: Reply, b: Reply) -> bool:
        key = (a.text, b.text)
        if key not in seen:
            prompt = (
                "Two answers from an AI agent to the same situation are below. Do they make the same decision "
                "(same action, same conclusion, same key facts), ignoring wording? Answer with one word: SAME or DIFFERENT.\n\n"
                f"ANSWER A:\n{a.text}\n\nANSWER B:\n{b.text}"
            )
            out = (sampler.one(_judge_request(like, prompt), 0).text or "").upper()
            seen[key] = "SAME" in out and "DIFFERENT" not in out
        return seen[key]

    return judge


_WORDS = re.compile(r"[a-z0-9$.]+")


_NEG = {"not", "no", "never", "don't", "dont", "doesn't", "doesnt", "can't", "cant", "cannot", "won't", "wont",
        "shouldn't", "shouldnt", "isn't", "isnt", "aren't", "arent", "didn't", "didnt", "without", "nothing", "none"}


def text_judge(a: Reply, b: Reply) -> bool:
    """Wording-based comparison, used when there is no live model to judge with.
    Close wording counts as the same answer unless one of them negates it."""
    ta, tb = (a.text or "").lower(), (b.text or "").lower()
    wa, wb = set(_WORDS.findall(ta)), set(_WORDS.findall(tb))
    if not wa and not wb:
        return True
    na = {w for w in re.findall(r"[a-z']+", ta) if w in _NEG}
    nb = {w for w in re.findall(r"[a-z']+", tb) if w in _NEG}
    if na != nb:
        return False
    return len(wa & wb) / max(len(wa | wb), 1) >= 0.7


# ------------------------------------------------------------------ statistics


def fisher_less(kept_removed: int, n_removed: int, kept_base: int, n_base: int) -> float:
    """One-sided Fisher exact test: the chance of seeing this few repeats with the piece removed
    if removing it made no difference."""
    k_total, n_total = kept_removed + kept_base, n_removed + n_base
    denom = comb(n_total, n_removed)
    lo = max(0, k_total - n_base)
    p = sum(comb(k_total, x) * comb(n_total - k_total, n_removed - x) for x in range(lo, kept_removed + 1))
    return min(1.0, p / denom)


# ------------------------------------------------------------------ results


@dataclass
class Trial:
    removed: list[Segment]
    kept: int = 0  # runs where the decision still happened
    n: int = 0
    instead: Counter = field(default_factory=Counter)  # what it did when the decision didn't happen
    p: float | None = None  # significance, once confirmed
    family: int = 0  # how many removals it was compared against (for the correction)

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
    # decisive: it drives the decision (without it the agent acts differently)
    # prerequisite: it supplies the data the call is made with (without it the agent can't make this call)
    kind: str = "decisive"
    # when no single part of the piece is enough (the same text also appears elsewhere, e.g. a search
    # repeated later), the smallest set of parts, across pieces, whose removal together flips it
    refined: "Trial | None" = None

    @property
    def finest(self) -> Trial:
        return self.chain[-1]

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
    alpha: float = 0.05
    deterministic: bool = False

    @property
    def base(self) -> float:
        return self.baseline.rate

    def verdict(self, t: Trial) -> str:
        if any(t is c.finest or t in c.chain for c in self.causes) or (self.joint and t is self.joint.finest):
            return "cause"
        return "not significant" if t.effect(self.base) >= self.threshold else "none"

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
                "p": t.p, "compared_against": t.family or None,
                "verdict": self.verdict(t), "instead": dict(t.instead),
            }

        return {
            "decision": {"request": self.target.request_id, "response": self.target.response_id,
                         "outcome": self.target.describe()},
            "baseline": {"happens": self.baseline.kept, "runs": self.baseline.n},
            "causes": [{"kind": c.kind, "masked": c.masked, "chain": [trial(t) for t in c.chain],
                        "together": trial(c.refined) if c.refined else None} for c in self.causes],
            "joint_cause": [trial(t) for t in self.joint.chain] if self.joint else None,
            "tested": [trial(t) for t in self.trials],
            "warnings": self.warnings, "model_calls": self.calls, "cache_hits": self.cache_hits,
            "stopped": self.stopped, "deterministic": self.deterministic, "alpha": self.alpha,
            "untested": [seg(x) for x in self.untested],
        }


# ------------------------------------------------------------------- engine


class Why:
    """The search. See the module docstring for the idea; the steps are:

    1. rerun the recorded decision (baseline) to see how reliably the model makes it
    2. screen every piece of context: remove it, rerun a couple of times, more if anything changed
    3. confirm candidates: more reruns on both sides and a Fisher exact test, corrected for how many
       pieces were compared, so sampling noise isn't reported as a cause
    4. drill into each cause (and into suspicious pieces that didn't flip whole: masked causes)
    5. if nothing single explains it, look for a small combination that does
    """

    def __init__(
        self,
        trace: Trace,
        target: Target,
        sampler: Sampler,
        *,
        k: int = 5,
        screen: int = 2,
        confirm: int = 10,
        alpha: float = 0.05,
        threshold: float = 0.5,
        max_depth: int = 4,
        max_causes: int = 3,
        expand: int = 3,
        max_pieces: int | None = 40,
        joint_search: bool = True,
        progress: Callable[[str], None] | None = None,
    ):
        if k < 1:
            raise ValueError("runs must be at least 1")
        self.trace = trace
        self.target = target
        self.sampler = sampler
        self.req = build_request(trace, target.request_id)
        self.det = is_deterministic(self.req)
        # temperature 0 is close to deterministic, but not guaranteed: still rerun twice to confirm
        self.k = 1 if self.det else k
        self.screen = 1 if self.det else max(1, min(screen, self.k))
        self.confirm = 2 if self.det else max(confirm, self.k)
        self.alpha = alpha
        self.threshold = threshold
        self.max_depth = max_depth
        self.max_causes = max_causes
        self.expand = expand
        self.max_pieces = max_pieces
        self.joint_search = joint_search
        self.progress = progress or (lambda msg: None)
        self._decision = target.decision_text()
        self.base_trial = Trial([])
        self.tested = 0  # every removal compared so far; confirmations correct for all of them

    # -- running trials

    @property
    def base(self) -> float:
        return self.base_trial.rate

    def _score(self, trial: Trial, replies: list[Reply]) -> None:
        for r in replies:
            trial.n += 1
            if self.target.matches(r):
                trial.kept += 1
            else:
                trial.instead[r.describe()] += 1

    def _req(self, removed: list[Segment]) -> dict:
        return ablate(self.req, removed) if removed else self.req

    def _extend(self, trial: Trial, n: int) -> None:
        if trial.n < n:
            try:
                replies = self.sampler.many((self._req(trial.removed), i) for i in range(trial.n, n))
            except BudgetExceeded as e:
                self._score(trial, getattr(e, "partial", []))  # keep what was measured before stopping
                raise
            self._score(trial, replies)

    def _trials(self, removals: list[list[Segment]]) -> list[Trial]:
        """Screen each removal with a few runs, then finish the ones that look like they matter."""
        trials = [Trial(list(rm)) for rm in removals]
        self.tested += len(trials)
        reqs = [self._req(rm) for rm in removals]
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
        self.tested += 1
        self._extend(t, self.k)
        return t

    def _candidate(self, t: Trial) -> bool:
        return t.effect(self.base) >= self.threshold

    def _confirm(self, t: Trial, family: int | None = None) -> bool:
        """Is this removal's effect real, not sampling noise? More reruns on both sides, then a
        one-sided Fisher exact test, Bonferroni-corrected for every removal compared in this search."""
        if not self._candidate(t):
            return False
        t.family = max(1, self.tested, family or 0)
        self._extend(self.base_trial, self.confirm)
        self._extend(t, self.confirm)
        if self.det:
            return t.kept == 0 and self.base_trial.kept == self.base_trial.n
        level = self.alpha / t.family
        t.p = fisher_less(t.kept, t.n, self.base_trial.kept, self.base_trial.n)
        if t.p > level and t.p < self.alpha and self._candidate(t):
            # close call: double the evidence once before deciding
            self._extend(self.base_trial, 2 * self.confirm)
            self._extend(t, 2 * self.confirm)
            t.p = fisher_less(t.kept, t.n, self.base_trial.kept, self.base_trial.n)
        return t.p <= level and self._candidate(t)

    def _rank(self, segs: list[Segment]) -> list[Segment]:
        # most suspicious first: shares wording with the decision, then most recent
        n = len(segs)
        return [s for _, _, s in sorted(
            ((-overlap(s.text, self._decision), -(i / max(n, 1)), s) for i, s in enumerate(segs)),
            key=lambda x: (x[0], x[1]),
        )]

    def _kind(self, cause: Cause) -> str:
        """prerequisite when the piece supplies the values the call is made with and, without it,
        the agent doesn't make another call instead; otherwise the piece drives the decision."""
        if self.target.mode not in ("tool", "tools"):
            return "decisive"
        alt = cause.finest.top_instead()
        if alt and alt[0].startswith("calls "):
            return "decisive"
        text = cause.top.removed[0].text.lower()
        for v in _arg_values(self.target):
            if re.search(r"(?<![\w.])" + re.escape(v) + r"(?![\w])", text):
                return "prerequisite"
        return "decisive"

    # -- search

    def run(self) -> Report:
        rep = Report(self.trace, self.target, self.base_trial, k=self.k, threshold=self.threshold,
                     alpha=self.alpha, deterministic=self.det)
        try:
            self.progress("rerunning the recorded decision")
            self._extend(self.base_trial, self.k)
            self._search(rep)
        except BudgetExceeded as e:
            rep.stopped = str(e)
        rep.calls = self.sampler.calls
        rep.cache_hits = self.sampler.hits
        return rep

    def _search(self, rep: Report) -> None:
        from .rerun import server_side_context

        warn = server_side_context(self.req)
        if warn:
            rep.warnings.append(warn)
        if self.base_trial.kept == 0:
            rep.warnings.append(
                f"The model never repeated this decision in {self.base_trial.n} reruns of the exact same context, "
                "so there is nothing to attribute. The original was a rare outcome, or the model/settings changed."
            )
            return
        if self.base < 0.6:
            rep.warnings.append(
                f"Unstable decision: the model only repeats it in {self.base_trial.kept}/{self.base_trial.n} reruns of "
                "the same context. Causes need stronger evidence to show up; raise --runs for a clearer answer."
            )
        all_segs = extract(self.req, self.trace, self.target.request_id)
        if not all_segs:
            rep.warnings.append("No removable context found in this request.")
            return
        segs = self._rank(all_segs)
        if self.max_pieces and len(segs) > self.max_pieces:
            # always test the system prompt and the newest message, whatever their rank
            # always test the system prompt, the first user message (the task) and the newest message
            idx = [s.msg_index for s in all_segs if s.msg_index is not None]
            last = max(idx, default=None)
            first_user = min((s.msg_index for s in all_segs if s.kind == "user" and s.msg_index is not None), default=None)
            pinned = [s for s in segs if s.kind == "system" or s.msg_index in (last, first_user)]
            keep = [s for s in segs if s not in pinned][: max(0, self.max_pieces - len(pinned))]
            rep.untested = [s for s in segs if s not in keep and s not in pinned]
            segs = pinned + keep

        self.progress(f"testing {len(segs)} pieces of context")
        rep.trials = self._trials([[s] for s in segs])
        family = len(rep.trials)
        cands = sorted((t for t in rep.trials if self._candidate(t)), key=lambda t: -t.effect(self.base))
        confirmed = []
        for t in cands:
            self.progress(f"confirming {t.removed[0].where}")
            if self._confirm(t, family):
                confirmed.append(t)

        for t in confirmed[: self.max_causes]:
            self.progress(f"narrowing down {t.removed[0].where}")
            c = Cause(self._drill(t, []))
            c.kind = self._kind(c)
            rep.causes.append(c)

        # A piece can hold the cause and evidence against it at once (a search result with a stale
        # doc next to the real policy). Removing the whole piece then changes nothing, so look inside
        # the most suspicious pieces that didn't flip.
        others = [t for t in rep.trials if t not in confirmed and t.removed[0].children()]
        for t in others[: self.expand]:
            self.progress(f"looking inside {t.removed[0].where}")
            chain = self._drill(t, [])
            if len(chain) > 1:
                c = Cause(chain, masked=True)
                c.kind = self._kind(c)
                rep.causes.append(c)

        # A cause that couldn't be narrowed may be repeated elsewhere (a search run twice returns the
        # same bad doc twice): removing one copy then changes nothing. Look for the smallest set of
        # parts, in it and in the other suspicious pieces, that changes the decision together.
        for c in rep.causes:
            seg = c.top.removed[0]
            if c.kind != "decisive" or len(c.chain) > 1 or not seg.children():
                continue
            cands = list(seg.children())
            for t in others[: self.expand]:
                if t.removed[0] is not seg:
                    cands.extend(t.removed[0].children())
            if len(cands) > 1:
                self.progress(f"narrowing down {seg.where} across repeated content")
                j = self._joint(cands)
                if j is not None:
                    c.refined = j

        rep.causes.sort(key=lambda c: (c.kind != "decisive", -c.finest.effect(self.base)))
        if not any(c.kind == "decisive" for c in rep.causes) and self.joint_search:
            expanded = {id(t.removed[0]) for t in others[: self.expand]}
            # hold needed inputs fixed: the question is what makes the agent act differently
            needed = {id(c.top.removed[0]) for c in rep.causes}
            cands2: list[Segment] = []
            for sg in segs:
                if id(sg) in needed:
                    continue
                kids = sg.children() if id(sg) in expanded else []
                cands2.extend(kids or [sg])
            if len(cands2) > 1:
                self.progress("no single piece explains it; searching combinations")
                joint = self._joint(cands2)
                if joint is not None:
                    rep.joint = Cause([joint])

    def _drill(self, top: Trial, held: list[Segment]) -> list[Trial]:
        """Follow a cause down into smaller and smaller pieces while they still flip the decision."""
        chain = [top]
        cur = top
        while len(chain) <= self.max_depth:
            kids = self._rank(cur.removed[-1].children())
            if not kids:
                break
            trials = self._trials([held + [k] for k in kids])
            nxt = None
            for t in sorted((t for t in trials if self._candidate(t)), key=lambda t: -t.effect(self.base)):
                if self._confirm(t, len(kids)):
                    nxt = t
                    break
            if nxt is None:
                break
            cur = nxt
            chain.append(cur)
        return chain

    def _joint(self, cands: list[Segment], top_m: int = 6) -> Trial | None:
        """Find a small set of pieces that only change the decision together.

        Removing everything is not guaranteed to flip the decision (it also removes evidence
        pointing the other way), so try the full set, then the most suspicious few, then pairs.
        """
        tested = [0]
        found = self._ddmin(cands, tested)
        if found is None:
            top = self._rank(cands)[:top_m]
            if 1 < len(top) < len(cands):
                found = self._ddmin(top, tested)
            if found is None:
                pairs = [[a, b] for i, a in enumerate(top) for b in top[i + 1 :]]
                tested[0] += len(pairs)
                hits = sorted((t for t in self._trials(pairs) if self._candidate(t)), key=lambda t: -t.effect(self.base))
                found = hits[0] if hits else None
        if found is not None and self._confirm(found, max(tested[0], 1)):
            return found
        return None

    def _ddmin(self, segs: list[Segment], tested: list[int]) -> Trial | None:
        """Smallest set of pieces whose joint removal flips the decision (delta debugging)."""
        cache: dict[tuple, Trial] = {}

        def test(sub: list[Segment]) -> Trial:
            key = tuple(sorted(id(s) for s in sub))
            if key not in cache:
                tested[0] += 1
                cache[key] = self._full(sub)
            return cache[key]

        if not self._candidate(test(segs)):
            return None
        cur, n = list(segs), 2
        while len(cur) >= 2:
            size = max(1, len(cur) // n)
            chunks = [cur[i : i + size] for i in range(0, len(cur), size)]
            progressed = False
            for ch in chunks:
                if self._candidate(test(ch)):
                    cur, n, progressed = ch, 2, True
                    break
            if not progressed:
                for ch in chunks:
                    comp = [s for s in cur if s not in ch]
                    if comp and self._candidate(test(comp)):
                        cur, n, progressed = comp, max(n - 1, 2), True
                        break
            if not progressed:
                if n >= len(cur):
                    break
                n = min(len(cur), 2 * n)
        return test(cur)


def _arg_values(target: Target) -> list[str]:
    """Lowercased argument values of the recorded call(s), as they'd appear in text."""
    out: list[str] = []

    def add(v):
        if isinstance(v, bool) or v is None:
            return
        if isinstance(v, (int, float)):
            # short numbers ("1", "42") appear everywhere; only distinctive ones count
            for form in {repr(v), str(int(v)) if isinstance(v, float) and v.is_integer() else repr(v)}:
                if len(form) >= 3:
                    out.append(form)
        elif isinstance(v, str) and len(v.strip()) >= 3:
            out.append(v.lower())
        elif isinstance(v, dict):
            for x in v.values():
                add(x)
        elif isinstance(v, list):
            for x in v:
                add(x)

    for tc in target.recorded.tool_calls:
        if target.mode == "tools" or tc.get("name") == target.tool:
            add(tc.get("arguments"))
    return out


# ----------------------------------------------------------------- facade


def why(
    trace: "Trace | str",
    event_id: int,
    *,
    model: Model | None = None,
    runs: int = 5,
    screen: int = 2,
    confirm: int = 10,
    tool: str | None = None,
    match: str | None = None,
    exact_args: bool = False,
    judge: Callable[[Reply, Reply], bool] | None = None,
    budget: int | None = 300,
    cache_dir: str | None = ".runtape/cache",
    workers: int = 8,
    threshold: float = 0.5,
    alpha: float = 0.05,
    max_pieces: int | None = 40,
    progress: Callable[[str], None] | None = None,
    on_call: Callable[[], None] | None = None,
) -> Report:
    """Explain a decision in a trace. See module docstring."""
    from .rerun import FunctionModel

    if not isinstance(trace, Trace):
        trace = Trace.load(trace)
    rid, _ = request_for(trace, event_id)
    req = build_request(trace, rid)
    sampler = Sampler(model or model_for(req), cache_dir=cache_dir, budget=budget, workers=workers, on_call=on_call)
    target = make_target(trace, event_id, tool=tool, match=match, exact_args=exact_args)
    notes = []
    if target.mode == "judge":
        if judge is not None:
            target.judge = judge
        elif isinstance(sampler.model, FunctionModel):
            # a model function answers agent requests, not grading prompts: compare wording locally
            target.judge = text_judge
            notes.append("Text answers compared by wording (no live model to judge with). Pass --match REGEX "
                         "or judge= for a sharper test.")
        else:
            target.judge = llm_judge(sampler, req)
    rep = Why(trace, target, sampler, k=runs, screen=screen, confirm=confirm, threshold=threshold, alpha=alpha,
              max_pieces=max_pieces, progress=progress).run()
    rep.warnings[:0] = notes
    return rep


def estimate_calls(trace: Trace, event_id: int, runs: int = 5, screen: int = 2, max_pieces: int | None = 40) -> tuple[int, int]:
    """(likely, worst case) number of model calls a why run will make."""
    rid, _ = request_for(trace, event_id)
    req = build_request(trace, rid)
    n = len(extract(req, trace, rid))
    if max_pieces:
        n = min(n, max_pieces)
    if is_deterministic(req):
        return 2 + n + 12, 2 + n * 2 + 40
    return max(runs, 10) + n * screen + 4 * 10 + 3 * runs * 3, 20 + n * runs + 60 * runs


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
