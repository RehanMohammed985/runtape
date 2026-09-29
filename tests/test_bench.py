"""The benchmark harness runs end to end offline with its stand-in model and scores cases correctly."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_bench_harness_offline(tmp_path):
    out = tmp_path / "r.jsonl"
    cmd = [sys.executable, str(ROOT / "bench" / "run.py"), "--model-fn", str(ROOT / "bench" / "sim.py") + ":model",
           "--cases", "5", "--out", str(out), "--work", str(tmp_path)]
    subprocess.run(cmd, check=True, capture_output=True, cwd=tmp_path)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 5 and {r["scenario"] for r in rows} == {"refund", "inbox", "ops", "cleanup", "access"}
    ran = [r for r in rows if r["valid"] and "error" not in r]
    assert ran and sum(r["headline_is_plant"] for r in ran) >= len(ran) - 1
    # resumes: a second run skips finished cases
    subprocess.run(cmd, check=True, capture_output=True, cwd=tmp_path)
    assert len(out.read_text().splitlines()) == 5
    rep = subprocess.run([sys.executable, str(ROOT / "bench" / "report.py"), str(out)], capture_output=True,
                         text=True, check=True).stdout
    assert "headline cause is the planted sentence" in rep


def test_cases_plant_once_and_remove_cleanly():
    sys.path.insert(0, str(ROOT / "bench"))
    from cases import generate

    for c in generate(25, seed=3):
        with_p = json.dumps(c.messages(True))
        without = json.dumps(c.messages(False))
        assert with_p.count(json.dumps(c.plant)[1:-1]) == 1, c.id
        assert json.dumps(c.plant)[1:-1] not in without, c.id
