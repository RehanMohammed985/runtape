"""Scenarios from an independent review, kept as regression tests for the cause ranking.

Each scenario builds a trace with a stand-in model (tests/scenarios/s_*.py, written by a
reviewer to break `why`), then checks that the headline is the right cause."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from runtape import Trace, why
from runtape.cli import resolve_decision, resolve_event
from runtape.rerun import FunctionModel

HERE = Path(__file__).resolve().parent / "scenarios"


def build(name, tmp_path):
    path = tmp_path / f"{name}.jsonl"
    subprocess.run([sys.executable, str(HERE / f"s_{name}.py"), str(path)], check=True, capture_output=True, cwd=tmp_path)
    spec = importlib.util.spec_from_file_location(f"s_{name}", HERE / f"s_{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return Trace.load(path), mod.model


def explain(name, ref, tmp_path):
    t, model = build(name, tmp_path)
    ev = resolve_decision(t, ref) if ref == "last" else resolve_event(t, ref)
    rep = why(t, ev, model=FunctionModel(model), cache_dir=None)
    heads = [c for c in rep.causes if c.kind == "decisive"]
    joint = rep.joint if rep.joint is not None and rep.joint.kind == "decisive" else None
    return rep, heads, joint


def text_of(c):
    t = c.refined or c.finest
    return " ".join(s.text for s in t.removed)


def test_confirmed_cause_is_never_dropped(tmp_path):
    # four config reads (re-read when missing) plus the user's "skip staging" line
    rep, heads, _ = explain("many", "tool:deploy_prod", tmp_path)
    assert "Skip staging" in text_of(heads[0])
    assert all(rep.verdict(t) != "not significant" for t in rep.confirmed)


def test_alternative_already_taken_earlier_is_not_a_redo(tmp_path):
    # the agent escalated an earlier ticket; escalating this one is still a different action
    rep, heads, _ = explain("earlier", "tool:issue_refund", tmp_path)
    assert "any amount" in text_of(heads[0])
    assert heads[0].finest.top_instead()[0].startswith("calls escalate_to_manager")


def test_model_prior_has_no_headline(tmp_path):
    # removing the listing only makes the agent list again: not a reason for the choice
    rep, heads, joint = explain("prior", "tool:delete_file", tmp_path)
    assert heads == [] and joint is None
    assert rep.causes and all(c.kind == "prerequisite" for c in rep.causes)


def test_masked_cause_with_little_shared_wording(tmp_path):
    rep, heads, _ = explain("mask", "tool:issue_refund", tmp_path)
    assert heads and "any amount" in text_of(heads[0]) and heads[0].masked


def test_cause_on_a_quiet_page_among_many(tmp_path):
    rep, heads, _ = explain("big2", "tool:publish_report", tmp_path)
    assert heads and "embargo" in text_of(heads[0]).lower()


def test_two_pieces_each_sufficient(tmp_path):
    rep, heads, joint = explain("or", "tool:merge_pr", tmp_path)
    parts = joint.finest.removed if joint else (heads[0].refined.removed if heads and heads[0].refined else [])
    assert len(parts) == 2


def test_two_pieces_both_needed(tmp_path):
    rep, heads, _ = explain("and", "tool:merge_pr", tmp_path)
    assert len(heads) == 2


def test_two_sentences_in_one_field_each_sufficient(tmp_path):
    rep, heads, _ = explain("orin", "tool:merge_pr", tmp_path)
    assert heads[0].refined is not None and len(heads[0].refined.removed) == 2


def test_cause_in_an_earlier_assistant_message(tmp_path):
    rep, heads, _ = explain("asst", "tool:delete_backups", tmp_path)
    assert heads[0].finest.removed[-1].kind == "assistant"


def test_same_poison_from_two_searches(tmp_path):
    rep, heads, joint = explain("dup", "tool:issue_refund", tmp_path)
    parts = joint.finest.removed if joint else heads[0].refined.removed
    assert len(parts) == 2 and all("any amount" in p.text for p in parts)


def test_text_answer_cause(tmp_path):
    rep, heads, _ = explain("text", "last", tmp_path)
    assert heads and heads[0].finest.removed[-1].kind == "tool_result"
