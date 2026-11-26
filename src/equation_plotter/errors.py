"""Error type that always serialises to the documented ``{"error": {...}}`` shape."""

from __future__ import annotations


class PlotterError(Exception):
    """A user-facing error. ``code`` is a stable machine-readable string."""

    def __init__(self, code: str, message: str, *, equation: int | None = None,
                 position: int | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.equation = equation
        self.position = position

    def to_dict(self) -> dict:
        err = {"code": self.code, "message": self.message}
        if self.equation is not None:
            err["equation"] = self.equation
        if self.position is not None:
            err["position"] = self.position
        return err
