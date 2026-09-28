"""runtape: a flight recorder for AI agents."""
from .recorder import FORMAT_VERSION, Recorder, current
from .trace import Context, Event, Hit, Trace
from .rerun import Distribution, Reply, rerun
from .why import why
from .replay import ReplayDiverged, Replayer, replay

__version__ = "0.2.0"


def record(path=None, **kwargs) -> Recorder:
    """Start recording. Same arguments as Recorder."""
    return Recorder(path, **kwargs)


def load(path) -> Trace:
    return Trace.load(path)


__all__ = [
    "Recorder",
    "Trace",
    "Event",
    "Context",
    "Hit",
    "record",
    "load",
    "current",
    "FORMAT_VERSION",
    "rerun",
    "why",
    "Reply",
    "Distribution",
    "replay",
    "Replayer",
    "ReplayDiverged",
    "__version__",
]
