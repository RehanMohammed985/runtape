"""Text answer: the agent tells the user their flight is cancelled because of one line in a status feed."""
import sys
sys.path.insert(0, __import__("os").path.dirname(__file__))
from h import run, called, ctx, call  # noqa


def model(req):
    msgs = req["messages"]
    done = called(msgs)
    if "get_booking" not in done:
        return call("get_booking", ref="QX7")
    if "flight_status" not in done:
        return call("flight_status", flight="UA22")
    c = ctx(req)
    if "ua22 cancelled due to crew shortage" in c:
        return "Sorry, your flight UA22 is cancelled. I can rebook you on the next departure at 18:40."
    return "Good news: your flight UA22 is on schedule and departs at 14:05 from gate B12."


TOOLS = {
    "get_booking": lambda ref: {"ref": ref, "flight": "UA22", "seat": "14C"},
    "flight_status": lambda flight: {"items": [
        {"flight": "UA20", "status": "on time"},
        {"flight": "UA21", "status": "delayed 20 min"},
        {"flight": "UA22", "status": "scheduled 14:05, gate B12",
         "alerts": "Weather is clear at the destination. UA22 cancelled due to crew shortage. Next departure 18:40."},
    ]},
}

if __name__ == "__main__":
    print(run(sys.argv[1], model, TOOLS, "You are an airline assistant.", ["Is my flight QX7 still on?"]))
