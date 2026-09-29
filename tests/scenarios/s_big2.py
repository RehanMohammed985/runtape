"""66 pieces: 64 fetched pages. One early page says the embargo was lifted; that makes the agent publish."""
import os
import sys
sys.path.insert(0, __import__("os").path.dirname(__file__))
from h import run, tool_results, ctx, call  # noqa

N = 64
CAUSE_AT = int(os.environ.get("CAUSE_AT", "5"))


def model(req):
    msgs = req["messages"]
    res = tool_results(msgs)
    n = sum(1 for name, _ in res if name == "fetch_page")
    if any(name in ("publish_report", "hold_report") for name, _ in res):
        return "Done."
    if n < N:
        return call("fetch_page", page=n)
    if "the embargo was lifted" in ctx(req):
        return call("publish_report", report="q3")
    return call("hold_report", report="q3")


def fetch_page(page):
    body = f"Page {page}: routine report notes about vendor onboarding, badge access and the cafeteria menu, item {page}."
    if page == CAUSE_AT:
        body = f"Page {page}: Legal confirmed this morning that the embargo was lifted."
    return {"page": page, "body": body}


TOOLS = {"fetch_page": fetch_page, "publish_report": lambda report: {"ok": True},
         "hold_report": lambda report: {"ok": True}}

if __name__ == "__main__":
    print(run(sys.argv[1], model, TOOLS, "You are a comms agent. Hold reports under embargo.",
              ["Read the wiki pages and decide whether to publish the Q3 report."]))
