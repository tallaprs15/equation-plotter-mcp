"""Vectorised numeric evaluation and root finding (used for fallbacks and rendering).

``lambdify`` generates Python source from the *already parsed and whitelisted* SymPy tree, so
it can only reference NumPy functions from our fixed function table — never user text.
"""

from __future__ import annotations

import warnings

import numpy as np
import sympy as sp


def _numeric_safe(expr):
    """Replace things NumPy can't evaluate (distributions, unevaluated derivatives)."""
    expr = expr.replace(sp.DiracDelta, lambda *a: sp.Integer(0))
    expr = expr.replace(lambda e: isinstance(e, (sp.Derivative, sp.Subs)), lambda e: sp.nan)
    return expr


def lambdify(expr, syms):
    try:
        fn = sp.lambdify(syms, _numeric_safe(sp.sympify(expr)), modules="numpy")
    except Exception:  # noqa: BLE001 - unprintable expression: behave as "undefined"
        def fn(*arrays):
            return np.full(np.shape(arrays[0]), np.nan)

    def wrapped(*arrays):
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                out = fn(*arrays)
            except Exception:  # noqa: BLE001
                return np.full(np.shape(arrays[0]), np.nan)
            out = np.asarray(out)
            if np.iscomplexobj(out):
                bad = np.abs(out.imag) > 1e-9 * np.maximum(1.0, np.abs(out.real))
                out = np.where(bad, np.nan, out.real)
            out = np.broadcast_to(out.astype(float), np.shape(arrays[0])).copy()
            out[~np.isfinite(out)] = np.nan
            return out

    return wrapped


def find_roots(fn, a: float, b: float, n: int = 4001, max_roots: int = 60) -> list[float]:
    """Sign-change + near-touch root finding on [a, b]. Skips sign changes across poles."""
    xs = np.linspace(a, b, n)
    ys = fn(xs)
    if not np.any(np.isfinite(ys)):
        return []
    if np.nanmax(np.abs(ys)) < 1e-10:
        return []  # identically zero (up to rounding): no isolated roots
    scale = np.nanmax(np.abs(ys))
    scale = scale if np.isfinite(scale) and scale > 0 else 1.0
    roots: list[float] = []
    zero = ys == 0
    run = zero & (np.r_[False, zero[:-1]] | np.r_[zero[1:], False])  # f == 0 on an interval
    exact = np.where(zero & ~run)[0]
    roots.extend(xs[exact].tolist())
    ys = np.where(run, np.nan, ys)
    s = np.sign(ys)
    idx = np.where((s[:-1] * s[1:]) < 0)[0]
    for i in idx[: max_roots * 3]:
        lo, hi = xs[i], xs[i + 1]
        flo = ys[i]
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            fm = fn(np.array([mid]))[0]
            if not np.isfinite(fm):
                break
            if np.sign(fm) == np.sign(flo):
                lo, flo = mid, fm
            else:
                hi = mid
        mid = 0.5 * (lo + hi)
        fm = fn(np.array([mid]))[0]
        if np.isfinite(fm) and abs(fm) < 1e-6 * max(1.0, min(scale, 1e6)):
            roots.append(mid)
    # touching roots (double roots like x^2): local minima of |f| that are ~0
    absy = np.abs(ys)
    for i in range(1, n - 1):
        if np.isfinite(absy[i]) and absy[i] <= absy[i - 1] and absy[i] <= absy[i + 1] \
                and absy[i] < 1e-3 * max(1.0, min(scale, 1e3)):
            lo, hi = xs[i - 1], xs[i + 1]
            for _ in range(80):
                m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
                f1, f2 = np.abs(fn(np.array([m1, m2])))
                if f1 < f2:
                    hi = m2
                else:
                    lo = m1
            m = 0.5 * (lo + hi)
            if abs(fn(np.array([m]))[0]) < 1e-9 * max(1.0, min(scale, 1e3)):
                roots.append(m)
    return dedupe(sorted(roots), tol=1e-6 * max(1.0, b - a))[:max_roots]


def dedupe(vals: list[float], tol: float) -> list[float]:
    out: list[float] = []
    for v in vals:
        if not out or abs(v - out[-1]) > tol:
            out.append(v)
    return out


def sign_at(fn, x0: float) -> int:
    v = fn(np.array([x0]))[0]
    if not np.isfinite(v):
        return 0
    if abs(v) < 1e-12:
        return 0
    return 1 if v > 0 else -1


def implicit_lines(Fn, xr, yr, n: int = 500) -> list[np.ndarray]:
    """Zero contour of F(x, y) on a grid, with spurious segments across poles removed.

    Linear interpolation puts a vertex where the grid values change sign. At a genuine zero
    |F(vertex)| is tiny compared with the surrounding grid values; across a pole (F jumps
    from +big to -big) it is not, so those vertices are dropped and the line is split there.
    """
    import contourpy

    xs = np.linspace(xr[0], xr[1], n)
    ys = np.linspace(yr[0], yr[1], n)
    X, Y = np.meshgrid(xs, ys)
    Z = Fn(X, Y)
    if not np.any(np.isfinite(Z)):
        return []
    Zm = np.ma.masked_invalid(Z)
    gen = contourpy.contour_generator(xs, ys, Zm, line_type="Separate")
    lines = gen.lines(0.0)
    dx = (xr[1] - xr[0]) / (n - 1)
    dy = (yr[1] - yr[0]) / (n - 1)
    absZ = np.abs(np.where(np.isfinite(Z), Z, 0.0))
    out = []
    for line in lines:
        if len(line) < 2:
            continue
        fv = np.abs(Fn(line[:, 0], line[:, 1]))
        ix = np.clip(((line[:, 0] - xr[0]) / dx).astype(int), 0, n - 2)
        iy = np.clip(((line[:, 1] - yr[0]) / dy).astype(int), 0, n - 2)
        local = np.maximum.reduce([absZ[iy, ix], absZ[iy + 1, ix], absZ[iy, ix + 1],
                                   absZ[iy + 1, ix + 1]])
        good = np.isfinite(fv) & ((fv <= 0.25 * local) | (fv < 1e-9))
        start = None
        for i, g in enumerate(good):
            if g and start is None:
                start = i
            if (not g or i == len(good) - 1) and start is not None:
                end = i + 1 if g else i
                if end - start >= 2:
                    out.append(line[start:end])
                start = None
    return out


def reflect_symmetries(X: np.ndarray, Y: np.ndarray) -> list[str]:
    """Set-based symmetry test for a sampled curve: is the reflected point cloud on the curve?"""
    ok = np.isfinite(X) & np.isfinite(Y)
    X, Y = X[ok], Y[ok]
    if X.size < 10:
        return []
    if X.size > 1500:
        idx = np.linspace(0, X.size - 1, 1500).astype(int)
        X, Y = X[idx], Y[idx]
    span = max(np.ptp(X), np.ptp(Y), 1e-9)
    step = np.median(np.hypot(np.diff(X), np.diff(Y)))
    tol = max(3 * step, 1e-3 * span)
    pts = np.stack([X, Y], axis=1)

    def on_curve(q):
        d2 = ((q[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
        return np.sqrt(d2.min(axis=1)).max() < tol

    found = []
    for name, q in (("x-axis", np.stack([X, -Y], 1)), ("y-axis", np.stack([-X, Y], 1)),
                    ("origin", np.stack([-X, -Y], 1)), ("y = x", np.stack([Y, X], 1))):
        if on_curve(q):
            found.append(name)
    return found
