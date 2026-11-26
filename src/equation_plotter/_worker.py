"""Worker process entry point. Reads length-prefixed pickled jobs on stdin, writes results.

Only the job names in ``JOBS`` can be run; the parent never sends callables.
"""

from __future__ import annotations

import pickle
import struct
import sys


def _limit_resources():
    try:
        import resource
    except ImportError:  # Windows
        return
    import os

    # Address-space cap per worker. Hosts running several workers in one container should set
    # PLOTTER_WORKER_MAX_MEMORY_MB so that (max concurrent jobs x cap) fits the container.
    mib = 1024 ** 2
    cap = int(os.environ.get("PLOTTER_WORKER_MAX_MEMORY_MB", "3072")) * mib
    for lim, value in ((getattr(resource, "RLIMIT_AS", None), cap),
                       (getattr(resource, "RLIMIT_CPU", None), 300)):
        if lim is None:
            continue
        try:
            soft, hard = resource.getrlimit(lim)
            if hard == resource.RLIM_INFINITY or hard > value:
                resource.setrlimit(lim, (value, hard))
        except (ValueError, OSError):
            pass  # not enforceable on this platform (e.g. RLIMIT_AS on macOS)


def _read(stream, n):
    buf = b""
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def main():
    _limit_resources()
    from .core import _analysis_job, _render_job
    from .errors import PlotterError

    jobs = {"analyse": _analysis_job, "render": _render_job}
    inp, out = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr  # stray prints must not corrupt the protocol
    while True:
        head = _read(inp, 8)
        if head is None:
            return
        (size,) = struct.unpack("!Q", head)
        name, args = pickle.loads(_read(inp, size))
        try:
            result = ("ok", jobs[name](*args))
        except PlotterError as e:
            result = ("plotter_error", e.to_dict())
        except BaseException as e:  # noqa: BLE001 - report everything to the parent as data
            result = ("crash", f"{type(e).__name__}: {e}")
        data = pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
        out.write(struct.pack("!Q", len(data)) + data)
        out.flush()


if __name__ == "__main__":
    main()
