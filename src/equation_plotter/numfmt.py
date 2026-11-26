"""JSON formatting for numbers, points and sets: every number is ``{"exact", "float"}``."""

from __future__ import annotations

import math

import sympy as sp

MAX_EXACT_LEN = 160


def fnum(v) -> float | None:
    """Rounded float or None for non-finite / non-real values."""
    try:
        if isinstance(v, sp.Basic):
            v = complex(sp.N(v, 15))
            if abs(v.imag) > 1e-9 * max(1.0, abs(v.real)):
                return None
            v = v.real
        v = float(v)
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        return None
    if not math.isfinite(v):
        return None
    return float(f"{v:.12g}") + 0.0  # + 0.0 turns -0.0 into 0.0


def exact_str(v) -> str | None:
    if v is None:
        return None
    if v is sp.oo:
        return "oo"
    if v is sp.S.NegativeInfinity:
        return "-oo"
    s = sp.sstr(v)
    return s if len(s) <= MAX_EXACT_LEN else None


def num(v) -> dict:
    """Format a SymPy value (exact) or a Python float (numeric estimate)."""
    if isinstance(v, sp.Basic):
        return {"exact": exact_str(v), "float": fnum(v)}
    return {"exact": None, "float": fnum(v)}


def point(xv, yv, **extra) -> dict:
    d = {"x": num(xv), "y": num(yv)}
    d.update(extra)
    return d


def pretty(v) -> str:
    if isinstance(v, sp.Basic):
        if v is sp.oo:
            return "∞"
        if v is sp.S.NegativeInfinity:
            return "-∞"
        s = exact_str(v)
        if s is not None:
            return s
    f = fnum(v)
    return "?" if f is None else f"{f:.6g}"


def interval_json(lo, hi, left_open=True, right_open=True) -> dict:
    lb = "(" if left_open else "["
    rb = ")" if right_open else "]"
    return {"lower": num(lo), "upper": num(hi), "left_open": bool(left_open),
            "right_open": bool(right_open), "text": f"{lb}{pretty(lo)}, {pretty(hi)}{rb}"}


def set_intervals(s) -> list | None:
    """Turn a real SymPy set into a sorted list of (lo, hi, lopen, ropen); None if not simple."""
    if s is sp.S.EmptySet:
        return []
    if s is sp.S.Reals:
        return [(sp.S.NegativeInfinity, sp.oo, True, True)]
    if isinstance(s, sp.Interval):
        return [(s.start, s.end, bool(s.left_open), bool(s.right_open))]
    if isinstance(s, sp.FiniteSet):
        return [(p, p, False, False) for p in sorted(s.args, key=lambda q: fnum(q) or 0)]
    if isinstance(s, sp.Union):
        out = []
        for a in s.args:
            part = set_intervals(a)
            if part is None:
                return None
            out.extend(part)
        return sorted(out, key=lambda iv: (fnum(iv[0]) if iv[0] != -sp.oo else -math.inf))
    if isinstance(s, sp.Complement) and s.args[0] is sp.S.Reals and isinstance(s.args[1], sp.FiniteSet):
        pts = sorted(s.args[1].args, key=lambda q: fnum(q) or 0)
        bounds = [sp.S.NegativeInfinity, *pts, sp.oo]
        return [(bounds[i], bounds[i + 1], True, True) for i in range(len(bounds) - 1)]
    return None


def general_text(s) -> str:
    """Human-readable general solution, e.g. '{2*n*pi + pi/2 | n ∈ ℤ}'."""
    parts = s.args if isinstance(s, sp.Union) else (s,)
    out = []
    for p in parts:
        if isinstance(p, sp.ImageSet):
            n = p.lamda.variables[0]
            out.append("{" + sp.sstr(p.lamda.expr.subs(n, sp.Symbol("n"))) + " | n ∈ ℤ}")
        else:
            out.append(sp.sstr(p))
    return " ∪ ".join(out)


def set_json(s, method: str = "exact") -> dict:
    ivs = set_intervals(s)
    d: dict = {"method": method, "exact": exact_str(s)}
    if ivs is not None:
        d["intervals"] = [interval_json(*iv) for iv in ivs]
        d["text"] = " ∪ ".join(_iv_text(iv) for iv in ivs) if ivs else "∅"
    elif isinstance(s, sp.Complement) and s.args[0] is sp.S.Reals:
        d["text"] = "ℝ \\ (" + general_text(s.args[1]) + ")"
    else:
        d["text"] = exact_str(s) or "(complicated set)"
    return d


def _iv_text(iv):
    lo, hi, lop, rop = iv
    if lo == hi:
        return "{" + pretty(lo) + "}"
    return f"{'(' if lop else '['}{pretty(lo)}, {pretty(hi)}{')' if rop else ']'}"


def numeric_interval_json(lo: float, hi: float, note: str, lo_unbounded=False,
                          hi_unbounded=False) -> dict:
    lo_v = sp.S.NegativeInfinity if lo_unbounded else lo
    hi_v = sp.oo if hi_unbounded else hi
    iv = interval_json(lo_v, hi_v, lo_unbounded, hi_unbounded)
    if not lo_unbounded:
        iv["text"] = iv["text"].replace("(", "[", 1)
    if not hi_unbounded:
        iv["text"] = iv["text"][:-1] + "]"
    iv["left_open"], iv["right_open"] = lo_unbounded, hi_unbounded
    return {"method": "numeric_estimate", "exact": None, "intervals": [iv],
            "text": "≈ " + iv["text"], "note": note}
