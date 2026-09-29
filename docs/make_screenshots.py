"""Regenerate the README screenshots from the offline demo: python docs/make_screenshots.py"""
import sys
import tempfile
from pathlib import Path

from rich.console import Console
from rich.text import Text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
sys.path.insert(0, str(ROOT / "src"))

import runtape  # noqa: E402
from refund_bot import main, simulated_model  # noqa: E402
from runtape import render  # noqa: E402
from runtape.cli import resolve_event  # noqa: E402
from runtape.rerun import FunctionModel  # noqa: E402


def shot(name: str, command: str, renderable) -> None:
    c = Console(record=True, width=124, force_terminal=True, color_system="truecolor")
    c.print(Text("$ " + command, style="bold green"))
    c.print(renderable)
    c.save_svg(str(ROOT / "docs" / f"{name}.svg"), title=f"runtape {name}")


t = runtape.load(main(Path(tempfile.mkdtemp()) / "run.jsonl"))
ev = resolve_event(t, "tool:issue_refund")
rep = runtape.why(t, ev, model=FunctionModel(simulated_model), cache_dir=None)
shot("why", "runtape why last tool:issue_refund", render.show_why(rep))
d = runtape.rerun(t, 30, model=simulated_model, cache_dir=None, drop=["9[1]"])
shot("rerun", 'runtape rerun last 30 --drop "9[1]"',
     render.show_distribution(d, d.recorded, title=f"rerun of decision #30 ({'; '.join(d.notes)}), {len(d)} runs"))
print("wrote docs/why.svg, docs/rerun.svg")
