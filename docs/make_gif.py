"""Render the README demo GIF from the offline inbox demo: python docs/make_gif.py

Frames are real runtape output rendered with rich, screenshotted with the preinstalled
Chromium (Playwright), and assembled with Pillow.
"""
import asyncio
import io
import sys
import tempfile
from pathlib import Path

from PIL import Image
from rich.console import Console
from rich.text import Text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
sys.path.insert(0, str(ROOT / "src"))

import runtape  # noqa: E402
from inbox_agent import main, simulated_model  # noqa: E402
from runtape import render  # noqa: E402
from runtape.cli import resolve_event  # noqa: E402
from runtape.fix import fix, write_test  # noqa: E402
from runtape.rerun import FunctionModel  # noqa: E402

WIDTH = 104


def svg(renderables) -> str:
    c = Console(record=True, width=WIDTH, force_terminal=True, color_system="truecolor", file=io.StringIO())
    for r in renderables:
        c.print(r)
    return c.export_svg(title="runtape")


def prompt(cmd: str, cursor: bool = False) -> Text:
    t = Text("$ ", style="bold green")
    t.append(cmd, style="bold")
    if cursor:
        t.append("█", style="bold")
    return t


def frames():
    """(renderables, milliseconds) pairs."""
    tmp = Path(tempfile.mkdtemp())
    path, forwarded = main(tmp / "inbox.jsonl")
    t = runtape.load(path)
    out: list[tuple[list, int]] = []

    cmd1 = "python examples/inbox_agent.py"
    for i in range(0, len(cmd1) + 1, 4):
        out.append(([prompt(cmd1[:i], cursor=True)], 45))
    run1 = [prompt(cmd1), Text(f"trace written to traces/{Path(path).name}"),
            Text(f"The agent forwarded an invoice to {forwarded[0]} without being asked.", style="bold red")]
    out.append((run1, 1600))

    cmd2 = "runtape why last tool:forward_email"
    for i in range(0, len(cmd2) + 1, 4):
        out.append((run1 + [Text(""), prompt(cmd2[:i], cursor=True)], 45))
    head = run1 + [Text(""), prompt(cmd2)]
    for step in ("rerunning the recorded decision", "testing 7 pieces of context", "confirming #12 read_email result",
                 "narrowing down #12 read_email result"):
        out.append((head + [Text("⠋ " + step, style="dim")], 450))

    rep = runtape.why(t, resolve_event(t, "tool:forward_email"), model=FunctionModel(simulated_model), cache_dir=None)
    report = render.show_why(rep)
    out.append((head + [report], 7000))

    # third beat: check fixes on the recorded context and write the test for the one that holds
    ev = resolve_event(t, "tool:forward_email")
    fr = fix(t, ev, model=FunctionModel(simulated_model), cache_dir=None)
    work = Path(tempfile.mkdtemp())
    test_path = write_test(path, ev, fr.report.target, work / "tests" / "test_inbox.py", add_system=fr.best.add_system)
    cmd3 = "runtape fix last tool:forward_email --write-test tests/test_inbox.py"
    for i in range(0, len(cmd3) + 1, 5):
        out.append(([prompt(cmd3[:i], cursor=True)], 40))
    head3 = [prompt(cmd3)]
    for step in ("finding the cause", "checking 4 fixes"):
        out.append((head3 + [Text("⠋ " + step, style="dim")], 500))
    written = Text("Regression test written to tests/test_inbox.py", style="bold")
    out.append((head3 + [render.show_fix(fr), written], 7000))

    # fourth beat: the generated test, run for real with pytest
    src = test_path.read_text().rstrip().split("\n")
    shown = src[src.index(next(line for line in src if line.startswith("def test_"))):]
    cmd4 = "tail -3 tests/test_inbox.py"
    for i in range(0, len(cmd4) + 1, 4):
        out.append(([prompt(cmd4[:i], cursor=True)], 45))
    screen = [prompt(cmd4)] + [Text(line, style="cyan") for line in shown]
    out.append((screen, 1800))
    cmd5 = "pytest -q tests/test_inbox.py"
    for i in range(0, len(cmd5) + 1, 4):
        out.append((screen + [Text(""), prompt(cmd5[:i], cursor=True)], 45))
    res = [Text(line, style="bold green" if "passed" in line else ("bold red" if "fail" in line.lower() else ""))
           for line in run_test(work)]
    out.append((screen + [Text(""), prompt(cmd5)] + res, 5000))
    return out


def run_test(work: Path) -> list[str]:
    """Run the generated test for real. The offline demo has no live model, so conftest points reruns at
    the same stand-in model the demo used."""
    import re
    import subprocess

    (work / "conftest.py").write_text(
        "import sys\n"
        f"sys.path[:0] = [{str(ROOT / 'src')!r}, {str(ROOT / 'examples')!r}]\n"
        "import runtape\n"
        "rr = sys.modules['runtape.rerun']\n"
        "from inbox_agent import simulated_model\n"
        "rr.model_for = lambda req: rr.FunctionModel(simulated_model)\n")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--no-header",
                        "tests/test_inbox.py"], cwd=work, capture_output=True, text=True)
    lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
    keep = [ln for ln in lines if ln.startswith(("F", "E ", "FAILED", ".")) or " failed" in ln or " passed" in ln]
    return [re.sub(r" in \d+\.\d+s", "", ln)[:100] for ln in keep][:6]


async def shoot(items, folder: Path) -> list[Path]:
    from playwright.async_api import async_playwright

    paths = []
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={"width": 1400, "height": 900}, device_scale_factor=1)
        await pg.route("http*://**", lambda r: r.abort())  # rich's SVG asks for a web font; use the local one
        for n, (renderables, _) in enumerate(items):
            await pg.set_content("<html><head><style>svg text{font-variant-ligatures:none;font-feature-settings:'liga' 0,'calt' 0}</style></head>"
                                 f"<body style='margin:0;background:#0d1117'>{svg(renderables)}</body></html>")
            el = await pg.query_selector("svg")
            p_ = folder / f"f{n:03d}.png"
            await el.screenshot(path=str(p_))
            paths.append(p_)
        await b.close()
    return paths


def build(dest: Path) -> None:
    items = frames()
    folder = Path(tempfile.mkdtemp())
    pngs = asyncio.run(shoot(items, folder))
    imgs = [Image.open(p).convert("RGB") for p in pngs]
    # pad every frame to the final (tallest) frame so the terminal doesn't jump
    w = max(i.width for i in imgs)
    h = max(i.height for i in imgs)
    canvas = []
    for im in imgs:
        c = Image.new("RGB", (w, h), (13, 17, 23))
        c.paste(im, (0, 0))
        canvas.append(c.convert("P", palette=Image.ADAPTIVE, colors=64))
    canvas[0].save(dest, save_all=True, append_images=canvas[1:], duration=[d for _, d in items], loop=0, optimize=True)
    print(f"wrote {dest} ({dest.stat().st_size // 1024} KB, {len(canvas)} frames)")


if __name__ == "__main__":
    build(ROOT / "docs" / "demo.gif")
