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
    safe_workers,
    Model,
    Reply,
    Sampler,
    build_request,
    is_deterministic,
    model_for,
    request_for,
)
from .recorder import _loose
from .segments import FILLS, Segment, _tokens, ablate, extract, fill_text, overlap
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
                if tc.get("name") == self.tool and (self.args is None or _loose(tc.get("arguments")) == _loose(self.args)):
                    return True
            return False
        if self.mode == "tools":
            return sorted(tc.get("name") for tc in r.tool_calls) == sorted(tc.get("name") for tc in self.recorded.tool_calls)
        if self.mode == "match":
            if self._matched_call() is not None:
                # the pattern picked out a tool call: only another such call counts, not the model
                # mentioning the command in text (asking permission to run it is a different decision)
                return any(self.pattern.search(f"{tc.get('name')} {_canon(tc.get('arguments'))}")
                           for tc in r.tool_calls)
            return bool(self.pattern.search(_reply_text(r)))
        if self.mode == "judge":
            return bool(self.judge(self.recorded, r))
        raise ValueError(self.mode)

    def describe(self) -> str:
        """Present tense, for 'without it the agent ___ in 3/10 reruns'."""
        if self.mode == "tool":
            if self.args is not None:
                return f"calls {self.tool}({_fmt_args(self.args)})"
            return f"calls {self.tool}"
        if self.mode == "tools":
            return "calls " + " + ".join(sorted(tc.get("name") for tc in self.recorded.tool_calls))
        if self.mode == "match":
            if self._matched_call() is not None:
                return f"makes a call matching /{self.pattern.pattern}/"
            return f"produces output matching /{self.pattern.pattern}/"
        return "gives the same answer"

    def _matched_call(self) -> dict | None:
        """In match mode, the recorded tool call the pattern picks out, if it picks one."""
        for tc in self.recorded.tool_calls:
            if self.pattern.search(f"{tc.get('name')} {_canon(tc.get('arguments'))}"):
                return tc
        return None

    def question(self) -> str:
        """Base form, for 'Why does the agent ___?'."""
        if self.mode == "tool":
            call = next((tc for tc in self.recorded.tool_calls if tc.get("name") == self.tool), None)
            if call is not None:
                return "call " + Reply(None, [call]).describe(110).removeprefix("calls ")
            return f"call {self.tool}"
        if self.mode == "tools":
            return "call " + " + ".join(sorted(tc.get("name") for tc in self.recorded.tool_calls))
        if self.mode == "match":
            call = self._matched_call()
            if call is not None:
                return "call " + Reply(None, [call]).describe(110).removeprefix("calls ")
            return f"produce output matching /{self.pattern.pattern}/"
        return "answer: " + self.recorded.describe(90).removeprefix("replies: ")

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
    examples: dict = field(default_factory=dict, repr=False)  # one reply per alternative, by description
    p: float | None = None  # significance, once confirmed
    family: int = 0  # how many removals it was compared against (for the correction)
    fill: str | None = None  # replacement text for removed whole pieces; None = the run's default

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
    # the same removal rerun with a different replacement text, to check the marker isn't the cause
    recheck: "Trial | None" = None

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
    requests: int = 0  # API requests for those calls (fewer when the API returns several samples per request)
    cache_hits: int = 0
    k: int = 0
    stopped: str | None = None  # why the search ended early, if it did
    depth: str = "full"  # "quick" when the search stopped at the main cause (no search for hidden causes or combinations)
    untested: list = field(default_factory=list)  # least suspicious pieces skipped by --max-pieces
    threshold: float = 0.5
    alpha: float = 0.05
    deterministic: bool = False
    confirmed: list = field(default_factory=list)  # every trial that passed the significance test
    unexpanded: int = 0  # pieces with parts that were not looked inside for masked causes
    fill: str = FILLS["marker"]
    # the model makes the decision only some of the time: causes are pieces without which it happens at
    # most half as often, confirmed on more reruns (threshold is then that share of the rate)
    intermittent: bool = False

    @property
    def base(self) -> float:
        return self.baseline.rate

    def verdict(self, t: Trial) -> str:
        if any(t is x for x in self.confirmed) or any(t is c.finest or t in c.chain for c in self.causes) \
                or (self.joint and t is self.joint.finest):
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
            "baseline": {"happens": self.baseline.kept, "runs": self.baseline.n, "intermittent": self.intermittent},
            "causes": [{"kind": c.kind, "masked": c.masked, "chain": [trial(t) for t in c.chain],
                        "together": trial(c.refined) if c.refined else None,
                        "recheck": {"fill": c.recheck.fill, "still_happens": c.recheck.kept, "runs": c.recheck.n,
                                    "p": c.recheck.p} if c.recheck else None} for c in self.causes],
            "fill": self.fill,
            "joint_cause": [trial(t) for t in self.joint.chain] if self.joint else None,
            "joint_recheck": {"fill": self.joint.recheck.fill, "still_happens": self.joint.recheck.kept,
                              "runs": self.joint.recheck.n, "p": self.joint.recheck.p}
            if self.joint and self.joint.recheck else None,
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
        max_causes: int = 8,
        expand: int = 6,
        max_pieces: int | None = 80,
        joint_search: bool = True,
        fill: str | None = None,
        progress: Callable[[str], None] | None = None,
        early_share: float = 0.25,
        batch: int = 4,
        quick_screen: int = 1,
        depth: str = "quick",
        intermittent_confirm: int = 30,
        intermittent_min: float = 0.15,
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
        # Confirmation looks at the evidence twice: once with the reruns screening already made, and
        # again after more. Each look gets its own share of the significance level, so the overall
        # false-positive rate is unchanged (Bonferroni over the two looks).
        self.early_share = early_share
        # when narrowing, test this many sub-pieces at a time, most suspicious first
        self.batch = max(1, batch)
        # reruns per piece when looking for hidden causes and combinations
        self.quick_screen = quick_screen
        # "quick" stops after the confirmed causes are narrowed down; "full" also looks for causes hidden
        # inside pieces whose removal changed nothing, and for combinations
        if depth not in ("quick", "full"):
            raise ValueError("depth must be 'quick' or 'full'")
        self.depth = depth
        self.threshold = threshold
        # a decision made in under 60% of the reruns, but at least this often, is searched as an intermittent one
        self.intermittent_confirm = max(intermittent_confirm, confirm)
        self.intermittent_min = intermittent_min
        self.intermittent_below = 0.6  # where a decision counts as unstable
        self.intermittent = False
        self.max_depth = max_depth
        self.max_causes = max_causes
        self.expand = expand
        self.max_pieces = max_pieces
        self.joint_search = joint_search
        self.fill = fill_text(fill)
        # the other preset, to recheck causes that relied on replacement text
        self.alt_fill = FILLS["empty"] if self.fill == FILLS["marker"] else FILLS["marker"]
        self.progress = progress or (lambda msg: None)
        self._decision = target.decision_text()
        self.base_trial = Trial([])
        self._earlier = None
        self._values: list | None = None
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
                d = r.describe()
                trial.instead[d] += 1
                trial.examples.setdefault(d, r)

    def _req(self, removed: list[Segment], fill: str | None = None) -> dict:
        return ablate(self.req, removed, fill or self.fill) if removed else self.req

    def _extend(self, trial: Trial, n: int) -> None:
        if trial.n < n:
            try:
                replies = self.sampler.many((self._req(trial.removed, trial.fill), i) for i in range(trial.n, n))
            except BudgetExceeded as e:
                self._score(trial, getattr(e, "partial", []))  # keep what was measured before stopping
                raise
            self._score(trial, replies)

    def _trials(self, removals: list[list[Segment]], screen: int | None = None) -> list[Trial]:
        """Screen each removal with a few runs, then finish the ones that look like they matter."""
        screen = screen or self.screen
        trials = [Trial(list(rm)) for rm in removals]
        self.tested += len(trials)
        reqs = [self._req(rm) for rm in removals]
        try:
            first = self.sampler.many((r, i) for r in reqs for i in range(screen))
        except BudgetExceeded as e:  # keep whatever was measured, then stop
            got = getattr(e, "partial", [])
            for n, t in enumerate(trials):
                self._score(t, got[n * screen : (n + 1) * screen])
            e.trials = [t for t in trials if t.n]
            raise
        for n, t in enumerate(trials):
            self._score(t, first[n * screen : (n + 1) * screen])
        todo = [n for n, t in enumerate(trials) if t.kept < t.n and screen < self.k]
        if todo:
            extra = self.k - screen
            more = self.sampler.many((reqs[n], i) for n in todo for i in range(screen, self.k))
            for j, n in enumerate(todo):
                self._score(trials[n], more[j * extra : (j + 1) * extra])
        return trials

    def _full(self, removal: list[Segment]) -> Trial:
        """One removal in a combination search. These look for a set of pieces that together flip the
        decision, so one rerun that keeps the decision rules a set out; one that changes it gets the
        full number of reruns."""
        t = Trial(list(removal))
        self.tested += 1
        self._extend(t, self.quick_screen)
        if t.kept < t.n:
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
        if self.det:
            self._extend(t, self.confirm)
            return t.kept == 0 and self.base_trial.kept == self.base_trial.n
        level = self.alpha / t.family
        early = self.early_share * level
        if t.n < self.confirm and early > 0:
            # a clear effect is already significant on the reruns screening made: stop here
            t.p = fisher_less(t.kept, t.n, self.base_trial.kept, self.base_trial.n)
            if t.p <= early and self._candidate(t):
                return True
        level -= early
        self._extend(t, self.confirm)
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

    def _alt_class(self, t: Trial) -> int:
        """What the agent does without the removed content, ranked by how telling it is:
        0 a different action, 1 it stops/answers/asks, 2 it repeats a call it already made earlier
        (same tool and arguments: it is just fetching the removed data again)."""
        alt = t.top_instead()
        reply = t.examples.get(alt[0]) if alt else None
        if reply is None:
            return 1
        own = {c.get("name") for c in self.target.recorded.tool_calls}
        if self._is_value(t.removed) and not any(tc.get("name") in own for tc in reply.tool_calls):
            return 2  # the piece is a value the call uses (the amount, the path): without it the agent finds or asks for it
        if not reply.tool_calls:
            return 1
        if self._earlier is None:
            self._earlier = {(e.payload.get("name"), _canon(_loose(e.payload.get("arguments"))))
                             for e in self.trace.events if e.type == "tool_call" and e.id < self.target.request_id}
        if all((tc.get("name"), _canon(_loose(tc.get("arguments")))) in self._earlier for tc in reply.tool_calls):
            return 2
        if all(self._refetches(tc, t.removed) for tc in reply.tool_calls):
            return 2
        return 0

    def _is_value(self, removed: list) -> bool:
        """Is the removed content essentially one of the decision's argument values (an order total of
        "640.0", a listing line "120G /srv/backups"), as opposed to a sentence that mentions it?"""
        if self._values is None:
            self._values = _arg_values(self.target)
        for seg in removed:
            text = " ".join(seg.text.split()).lower()
            if text and any(v in text and len(v) >= 0.4 * len(text) for v in self._values):
                return True
        return False

    def _refetches(self, tc: dict, removed: list) -> bool:
        """Does this call fetch the removed content again? True when it repeats the tool call that produced
        a removed piece, with arguments drawn from that call's: without the `du` listing the agent runs `du`
        on one of the same folders. A different lookup (reading the folder with `ls` after a sentence about
        it was removed) is not a re-fetch: the agent is deciding differently, not recovering data."""
        for seg in removed:
            ev = self.trace[seg.origin] if seg.origin is not None and seg.origin < len(self.trace.events) else None
            if ev is not None and ev.type == "tool_result" and ev.parent is not None:
                ev = self.trace[ev.parent]
            if ev is None or ev.type != "tool_call" or ev.payload.get("name") != tc.get("name"):
                continue
            alt = set(_tokens(_canon(tc.get("arguments"))))
            orig = set(_tokens(_canon(ev.payload.get("arguments"))))
            if not alt or len(alt & orig) >= 0.6 * len(alt):
                return True
        return False

    def _rank_key(self, c: Cause) -> tuple:
        # 1. re-fetched input last
        # 2. content the agent read or wrote (tool results, its own earlier replies) before the user's request
        #    and the system prompt: those are expected to drive its actions, and they are listed too
        # 3. a cause narrowed to a part (a sentence, an item) before a whole message or tool result that
        #    couldn't be: all of a whole piece mattering is what data the action uses looks like, and a
        #    narrowed part is the sharper answer (the instruction, not the records it says to send)
        # 4. a different action before stopping or answering
        # 5. the piece that shares the most wording with what the agent did, then the larger effect
        t = c.refined or c.finest
        text = " ".join(s.text for s in t.removed)
        cls = self._alt_class(t)
        given = all(s.kind in ("user", "system") for s in t.removed)
        narrowed = c.refined is not None or len(c.chain) > 1
        return (cls == 2, given, not narrowed, cls, -overlap(text, self._decision), -t.effect(self.base))

    # -- search

    def run(self) -> Report:
        rep = Report(self.trace, self.target, self.base_trial, k=self.k, threshold=self.threshold,
                     alpha=self.alpha, deterministic=self.det, fill=self.fill)
        try:
            self.progress("rerunning the recorded decision")
            self._extend(self.base_trial, self.k)
            self._search(rep)
        except BudgetExceeded as e:
            rep.stopped = str(e)
        rep.calls = self.sampler.calls
        rep.requests = self.sampler.requests
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
                "so there is nothing to attribute. The original was a rare outcome, or the model/settings changed. "
                "To measure how rare, run: runtape odds <trace> <event> --runs 20. To test a suspect directly, "
                "compare that with: runtape rerun <trace> <event> --drop <event> --runs 20."
            )
            return
        if self.base < self.intermittent_below and not self.det:
            # Measure the rate once more first: a few unlucky reruns shouldn't change how the search runs.
            self._extend(self.base_trial, self.confirm)
            if self.base < self.intermittent_below:
                # A decision the model makes only some of the time (an attack that works one run in three is
                # still an attack): intermittent. A cause can't lower its rate by the usual margin, but it can make it stop
                # happening: measure the rate on more reruns, and count a piece as a cause when removing it
                # at least halves the rate, confirmed on as many reruns, with the same correction for every
                # piece compared.
                self._extend(self.base_trial, self.intermittent_confirm)
                if self.base < self.intermittent_min:
                    rep.warnings.append(
                        f"Unstable decision: the model only repeats it in {self.base_trial.kept}/"
                        f"{self.base_trial.n} reruns of the same context, too rarely for any piece of context to "
                        "be shown to cause it, so the search stopped here. To test a suspect directly, compare "
                        "runtape odds <trace> <event> --runs 50 with runtape rerun <trace> <event> --drop <event> "
                        "--runs 50."
                    )
                    return
            if self.base < self.intermittent_below:
                self.intermittent = rep.intermittent = True
                self.threshold = rep.threshold = self.threshold * self.base
                self.confirm = self.intermittent_confirm
                rep.warnings.append(
                    f"Intermittent decision: the model makes it in {self.base_trial.kept}/{self.base_trial.n} reruns "
                    "of the same context. The causes below are what it depends on: without each, it happens at most "
                    f"half as often, confirmed on {self.confirm} reruns."
                )
        self._warn_unstable(rep)
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
        try:
            rep.trials = self._trials([[s] for s in segs])
        except BudgetExceeded as e:
            rep.trials = getattr(e, "trials", [])
            raise
        family = len(rep.trials)
        cands = sorted((t for t in rep.trials if self._candidate(t)), key=lambda t: -t.effect(self.base))
        for t in cands:
            self.progress(f"confirming {t.removed[0].where}")
            if self._confirm(t, family):
                rep.confirmed.append(t)

        # every confirmed piece is a cause; narrow down the most telling ones (ranked like the report)
        rep.causes = sorted((Cause([t]) for t in rep.confirmed), key=self._rank_key)
        # Narrow every one down, including pieces the agent would just fetch again when removed:
        # the email it re-reads can still contain the sentence that hijacked it.
        drilled = 0
        for c in rep.causes:
            if drilled >= self.max_causes:
                break
            drilled += 1
            self.progress(f"narrowing down {c.top.removed[0].where}")
            c.chain = self._drill(c.top, [])
            if self.depth == "quick" and self._settled(c):
                break  # explained by one part the agent read: lower-ranked causes stay as found

        # A piece can hold the cause and evidence against it at once (a search result with a stale
        # doc next to the real policy). Removing the whole piece then changes nothing, so look inside
        # the most suspicious pieces that didn't flip.
        others = [t for t in rep.trials if not any(t is x for x in rep.confirmed) and t.removed[0].children()]
        if self.depth == "quick" and any(self._settled(c) for c in rep.causes):
            # The decision is already explained by one sentence or item the agent read, without which it does
            # something else: report that now. Otherwise (the cause only makes it stop or answer, or couldn't be
            # narrowed down) the hidden-cause and combination searches run as in a full search.
            rep.depth = "quick"
            self._finish(rep)
            return
        for t in others[: self.expand]:
            self.progress(f"looking inside {t.removed[0].where}")
            # a hidden cause is one strong enough to flip the decision on its own once its neighbor's
            # counter-evidence is gone, so one rerun per part is enough to spot it
            chain = self._drill(t, [], self.quick_screen)
            if len(chain) > 1:
                rep.causes.append(Cause(chain, masked=True))
        rep.unexpanded = max(0, len(others) - self.expand)

        # A cause that couldn't be narrowed may be one of several parts that are each enough on their
        # own (the same bad doc returned by two searches, two different lines saying the same thing).
        # Removing one then changes nothing. Look for the smallest set that changes the decision.
        refined = 0
        for c in sorted(rep.causes, key=self._rank_key):
            # the most specific part the cause was narrowed to; if it still has parts, none of them
            # alone was enough, so they may each be sufficient on their own
            seg = c.finest.removed[-1]
            if refined >= 3 or not seg.children() or self._alt_class(c.finest) == 2:
                continue
            refined += 1
            cands = list(seg.children())
            if len(c.chain) == 1:  # a whole piece: its content may also repeat in other pieces
                for t in others[: self.expand]:
                    if t.removed[0] is not seg:
                        cands.extend(t.removed[0].children())
            if len(cands) > 1:
                self.progress(f"narrowing down {seg.where} across parts that repeat each other")
                j = self._joint(cands)
                if j is not None and len(j.removed) == 1:
                    # a single part after all (its own test was unlucky): keep narrowing from it
                    c.chain.extend(self._drill(j, []))
                elif j is not None:
                    c.refined = j

        rep.causes.sort(key=self._rank_key)
        if not any(self._alt_class(c.refined or c.finest) == 0 for c in rep.causes) and self.joint_search:
            expanded = {id(t.removed[0]) for t in others[: self.expand]}
            # hold known causes fixed: the question is what else makes the agent act differently
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

        self._finish(rep)

    def _warn_unstable(self, rep: Report) -> None:
        if self.intermittent:
            return  # said already, with what the causes mean
        if self.base < 0.6 and not self.det and not any(w.startswith("Unstable decision") for w in rep.warnings):
            rep.warnings.append(
                f"Unstable decision: the model only repeats it in {self.base_trial.kept}/{self.base_trial.n} reruns of "
                "the same context. Causes need stronger evidence to show up; raise --runs, or test a suspect "
                "directly with runtape rerun <trace> <event> --drop <event> --runs 20."
            )

    def _explains(self, c: Cause) -> bool:
        """Could this cause be the headline on its own: content the agent read or wrote (not the user's
        request or the system prompt), whose removal changes the decision rather than just making the
        agent fetch it again."""
        t = c.refined or c.finest
        return self._alt_class(t) != 2 and not all(s.kind in ("user", "system") for s in t.removed)

    def _settled(self, c: Cause) -> bool:
        """A cause that answers the question on its own: one part of what the agent read, narrowed as far as
        it goes, without which the agent takes a different action."""
        return self._explains(c) and self._alt_class(c.finest) == 0 and not c.finest.removed[-1].children()

    def _finish(self, rep: Report) -> None:
        self._warn_unstable(rep)  # the baseline may have been measured more precisely during the search
        # The headline is the most telling cause. A piece whose removal only makes the agent fetch
        # the same data again is never the headline: it is input, not the reason for the choice.
        rep.causes.sort(key=self._rank_key)
        allc = rep.causes + ([rep.joint] if rep.joint else [])
        eligible = [c for c in allc if self._alt_class(c.refined or c.finest) != 2]
        best = min(eligible, key=self._rank_key) if eligible else None
        for c in allc:
            c.kind = "decisive" if (c is best or self._alt_class(c.refined or c.finest) == 0) else "prerequisite"
        self._warn_budget = False
        for c in [c for c in allc if c.kind == "decisive"][:3]:
            self._top_up(c)
            self._recheck(c)
        if self._warn_budget:
            rep.warnings.append("The budget ran out before every cause was rechecked with a different replacement "
                                "text. Raise --budget to finish; completed reruns are cached.")

    def _top_up(self, c: Cause) -> None:
        """A cause confirmed early, on few reruns, is reported with the full number, so the evidence
        shown for the main causes is as strong as before. Intermediate steps keep their early stop."""
        if self.det:
            return
        for t in {id(x): x for x in (c.finest, c.refined) if x is not None}.values():
            if t.n < self.confirm and t.p is not None:
                try:
                    self._extend(t, self.confirm)
                except BudgetExceeded:
                    self._warn_budget = True
                    return
                t.p = fisher_less(t.kept, t.n, self.base_trial.kept, self.base_trial.n)

    def _recheck(self, c: Cause) -> None:
        """Removing a whole message or tool result leaves replacement text in its place, and that text
        can itself sway a model. Rerun the cause with a different replacement; if the request is the
        same either way (the cause was cut out, not replaced), there is nothing to check."""
        t = c.refined or c.finest
        if self._req(t.removed) == self._req(t.removed, self.alt_fill):
            return
        self.progress(f"rechecking {t.label} with a different replacement")
        r = Trial(list(t.removed), fill=self.alt_fill)
        try:
            self._extend(r, self.confirm)
        except BudgetExceeded:
            self._warn_budget = True
            return
        r.p = fisher_less(r.kept, r.n, self.base_trial.kept, self.base_trial.n)
        r.family = 1
        c.recheck = r

    def _drill(self, top: Trial, held: list[Segment], screen: int | None = None) -> list[Trial]:
        """Follow a cause down into smaller and smaller pieces while they still flip the decision."""
        chain = [top]
        cur = top
        while len(chain) <= self.max_depth:
            kids = self._rank(cur.removed[-1].children())
            if not kids:
                break
            # Most suspicious parts first, a few at a time; stop at the first one confirmed. The search only
            # follows one part down, so testing the rest after a confirmed one would change nothing.
            nxt = None
            for start in range(0, len(kids), self.batch):
                trials = self._trials([held + [k] for k in kids[start : start + self.batch]], screen)
                for t in sorted((t for t in trials if self._candidate(t)), key=lambda t: -t.effect(self.base)):
                    if self._confirm(t, len(kids)):
                        nxt = t
                        break
                if nxt is not None:
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
        if target.mode in ("tools", "judge") or tc.get("name") == target.tool or \
                (target.mode == "match" and tc is target._matched_call()):
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
    budget: int | None = 400,
    cache_dir: str | None = ".runtape/cache",
    workers: int = 8,
    threshold: float = 0.5,
    alpha: float = 0.05,
    max_pieces: int | None = 80,
    expand: int = 6,
    fill: str | None = None,
    progress: Callable[[str], None] | None = None,
    on_call: Callable[[], None] | None = None,
    depth: str = "quick",
) -> Report:
    """Explain a decision in a trace. See module docstring."""
    from .rerun import FunctionModel

    if not isinstance(trace, Trace):
        trace = Trace.load(trace)
    rid, _ = request_for(trace, event_id)
    req = build_request(trace, rid)
    model = model or model_for(req)
    sampler = Sampler(model, cache_dir=cache_dir, budget=budget, workers=safe_workers(model, workers),
                      on_call=on_call)
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
              max_pieces=max_pieces, expand=expand, fill=fill, progress=progress, depth=depth).run()
    rep.warnings[:0] = notes
    return rep


def estimate_calls(trace: Trace, event_id: int, runs: int = 5, screen: int = 2, max_pieces: int | None = 80,
                   full: bool = False) -> tuple[int, int]:
    """(likely, worst case) number of model calls a why run will make."""
    rid, _ = request_for(trace, event_id)
    req = build_request(trace, rid)
    n = len(extract(req, trace, rid))
    if max_pieces:
        n = min(n, max_pieces)
    if is_deterministic(req):
        return 2 + n + 12 + 2, 2 + n * 2 + 40 + 6
    # baseline, screening each piece, confirming and narrowing down the main cause, and rechecking it; a full
    # search also looks inside pieces that changed nothing and tries combinations (measured on bench/)
    likely = max(runs, 10) + n * screen + 70 + (40 if full else 0)
    return likely, 20 + n * runs + 60 * runs + 30


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
