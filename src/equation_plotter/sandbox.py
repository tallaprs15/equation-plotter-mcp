"""Hard time limits.

Two layers:

* ``Worker`` runs jobs in a separate Python process and kills it if a job overruns. This is the
  hard guarantee — it works even when SymPy is stuck inside C code (big-integer arithmetic)
  where Python signal handlers never get a chance to run.
* ``Budget.step`` is a cooperative per-step deadline (SIGALRM) used *inside* the worker so a
  single slow SymPy call (say ``function_range``) can be abandoned and replaced by a numeric
  estimate without losing the rest of the analysis.
"""

from __future__ import annotations

import contextlib
import os
import pickle
import struct
import subprocess
import sys
import signal
import threading
import time

from .errors import PlotterError


class StepTimeout(BaseException):
    """Raised by the alarm handler. BaseException so SymPy's ``except Exception`` can't eat it."""


class HardTimeout(Exception):
    pass


_BOOT = ("import sys; sys.path.insert(0, {path!r}); "
         "from equation_plotter._worker import main; main()")


class Worker:
    """A child Python process that runs whitelisted jobs; killed if a job overruns.

    The child is started with ``python -I`` (isolated mode: no user site-packages, no
    PYTHONPATH, no current directory on sys.path) and never imports the caller's
    ``__main__``, so it is safe to use from unguarded scripts, REPLs and servers.
    """

    def __init__(self):
        self.proc = None

    def _start(self):
        pkg_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.proc = subprocess.Popen(
            [sys.executable, "-I", "-c", _BOOT.format(path=pkg_parent)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def call(self, name: str, args: tuple, timeout: float):
        if self.proc is None or self.proc.poll() is not None:
            self._start()
        payload = pickle.dumps((name, args), protocol=pickle.HIGHEST_PROTOCOL)
        box: dict = {}

        def exchange():
            try:
                self.proc.stdin.write(struct.pack("!Q", len(payload)) + payload)
                self.proc.stdin.flush()
                head = self.proc.stdout.read(8)
                if len(head) < 8:
                    box["eof"] = True
                    return
                (size,) = struct.unpack("!Q", head)
                box["result"] = pickle.loads(self.proc.stdout.read(size))
            except Exception as e:  # noqa: BLE001
                box["exc"] = e

        th = threading.Thread(target=exchange, daemon=True)
        th.start()
        th.join(timeout)
        if th.is_alive():
            self.close()
            th.join(2)
            raise HardTimeout()
        if "result" not in box:
            self.close()
            raise PlotterError("internal_error", "Analysis worker exited unexpectedly")
        status, value = box["result"]
        if status == "ok":
            return value
        if status == "plotter_error":
            raise PlotterError(value["code"], value["message"], equation=value.get("equation"),
                               position=value.get("position"))
        raise PlotterError("internal_error", f"Analysis worker failed ({value})")

    def close(self):
        if self.proc is None:
            return
        if self.proc.poll() is None:
            self.proc.kill()
        try:
            self.proc.wait(2)
        except subprocess.TimeoutExpired:
            pass
        for f in (self.proc.stdin, self.proc.stdout):
            try:
                f.close()
            except OSError:
                pass
        self.proc = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _alarm_usable() -> bool:
    return hasattr(signal, "setitimer") and threading.current_thread() is threading.main_thread()


class Budget:
    """Tracks a total time budget and hands out per-step deadlines."""

    def __init__(self, total: float, step: float, enabled: bool = True):
        self.deadline = time.monotonic() + total
        self.step_limit = step
        self.enabled = enabled
        self.alarm = _alarm_usable()

    def remaining(self) -> float:
        return self.deadline - time.monotonic()

    def exhausted(self) -> bool:
        return self.remaining() <= 0.05

    @contextlib.contextmanager
    def step(self, limit: float | None = None):
        seconds = min(limit or self.step_limit, max(self.remaining(), 0.01))
        if not self.alarm:
            yield seconds
            return

        def handler(signum, frame):
            raise StepTimeout()

        old = signal.signal(signal.SIGALRM, handler)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            yield seconds
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)
