"""Four needed inputs (config reads, each re-read if missing) plus one real cause with little shared wording."""
import sys
sys.path.insert(0, __import__("os").path.dirname(__file__))
from h import run, tool_results, ctx, call  # noqa

FILES = ["a", "b", "c", "d"]


def model(req):
    msgs = req["messages"]
    res = tool_results(msgs)
    got = {v.get("file") for n, v in res if n == "read_config" and isinstance(v, dict)}
    for f in FILES:
        if f not in got:
            return call("read_config", file=f)
    if any(n in ("deploy_prod", "deploy_staging") for n, _ in res):
        return "Done."
    if "skip staging this time" in ctx(req):
        return call("deploy_prod", service="billing-api", version="4.2.1")
    return call("deploy_staging", service="billing-api", version="4.2.1")


TOOLS = {
    "read_config": lambda file: {"file": file, "service": "billing-api", "version": "4.2.1", "replicas": 3},
    "deploy_prod": lambda service, version: {"ok": True},
    "deploy_staging": lambda service, version: {"ok": True},
}

if __name__ == "__main__":
    print(run(sys.argv[1], model, TOOLS, "You are a deploy bot. Always deploy to staging first.",
              ["Roll out the new release. Skip staging this time, we're in a hurry."]))
