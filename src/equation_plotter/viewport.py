"""Pick a viewport that shows the interesting features of every equation."""

from __future__ import annotations

import math

import numpy as np

DEFAULT_HALF = 10.0
MIN_SPAN = 4.0
# Data-area aspect (width / height) of the fixed axes rectangle used by the renderer.
AXES_RATIO = (0.86 * 9.0) / (0.86 * 6.75)


def _nice_step(span: float) -> float:
    if not (span > 0 and math.isfinite(span)):
        return 1.0
    raw = span / 10
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            return m * mag
    return 10 * mag


def _nice(lo: float, hi: float) -> tuple[float, float]:
    step = _nice_step(hi - lo) / 2
    return math.floor(lo / step) * step, math.ceil(hi / step) * step


def _pad(lo, hi, frac=0.18, min_span=MIN_SPAN):
    min_span = max(min_span, 1e-6 * max(abs(lo), abs(hi)))
    span = hi - lo
    if span < min_span:
        mid = 0.5 * (lo + hi)
        return mid - min_span / 2, mid + min_span / 2
    return lo - frac * span, hi + frac * span


def _include_origin(lo, hi):
    span = hi - lo
    if lo > 0 and lo < 0.75 * span:
        lo = -0.08 * span
    if hi < 0 and -hi < 0.75 * span:
        hi = 0.08 * span
    return lo, hi


def _robust_minmax(v: np.ndarray):
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    lo, hi = float(v.min()), float(v.max())
    p5, p95 = np.percentile(v, [4, 96])
    core = max(p95 - p5, 1e-9)
    if hi - p95 > 3 * core:
        hi = float(p95)
    if p5 - lo > 3 * core:
        lo = float(p5)
    return lo, hi


def choose(interests: list[dict], x_range=None, y_range=None) -> dict:
    pts, vx, hy, periods = [], [], [], []
    equal = any(i.get("equal_aspect") for i in interests)
    for it in interests:
        pts += [(px, py) for px, py in it.get("points", []) if abs(px) < 1e7 and abs(py) < 1e7]
        vx += [v for v in it.get("vlines", []) if v is not None and abs(v) < 1e7]
        hy += [v for v in it.get("hlines", []) if v is not None and abs(v) < 1e7]
        if it.get("bbox"):
            x0, x1, y0, y1 = it["bbox"]
            pts += [(x0, y0), (x1, y1)]
        if it.get("period"):
            periods.append(it["period"])

    open_curves = any(it.get("sampler") is not None or it.get("implicit") is not None
                      or it.get("open") for it in interests)
    # ---- x range
    if x_range is not None:
        a, b = x_range
    else:
        xs = [p[0] for p in pts] + vx
        min_span = (8.0 if vx else 6.0) if open_curves else 1.0
        frac = 0.25 if open_curves else 0.12
        if xs:
            a, b = _pad(min(xs), max(xs), frac=frac, min_span=min_span)
        else:
            a, b = -DEFAULT_HALF, DEFAULT_HALF
        if periods and max(periods) < 40:
            p = max(periods)
            a, b = min(a, -p), max(b, p)
        a, b = _include_origin(a, b)
        if not periods:
            a, b = _nice(a, b)

    # ---- y range
    if y_range is not None:
        ya, yb = y_range
    else:
        core = [py for px, py in pts if a <= px <= b] + hy
        samples = []
        grid = np.linspace(a, b, 2001)
        for it in interests:
            fn = it.get("sampler")
            if fn is not None:
                vals = fn(grid)
                for v in it.get("vlines", []):
                    if v is not None:
                        vals = np.where(np.abs(grid - v) < 0.02 * (b - a), np.nan, vals)
                mm = _robust_minmax(vals)
                if mm:
                    samples += list(mm)
            curve = it.get("curve")
            if curve is not None:
                X, Y = curve
                sel = (X >= a) & (X <= b)
                mm = _robust_minmax(Y[sel]) if sel.any() else None
                if mm:
                    # closed curves: always show all of it; open ones behave like functions
                    (samples if it.get("open") else core).extend(mm)
        core = [v for v in core if math.isfinite(v)]
        samples = [v for v in samples if math.isfinite(v)] + core
        if not samples:
            ya, yb = -DEFAULT_HALF, DEFAULT_HALF
        else:
            slo, shi = min(samples), max(samples)
            clo, chi = (min(core), max(core)) if core else (slo, slo)
            h_eq = (b - a) / AXES_RATIO
            if not open_curves:
                ya, yb = slo, shi
            elif shi - slo <= h_eq or not core:
                ya, yb = (slo, shi) if core or shi - slo <= h_eq else (slo, slo + h_eq)
            else:
                # the curve leaves any sensible window: keep every feature, then spend the
                # remaining height where the curve actually goes (at least 20% each way it goes)
                H = max(h_eq, (chi - clo) * 1.3)
                extra = H - (chi - clo)
                above, below = max(shi - chi, 0.0), max(clo - slo, 0.0)
                fa = above / (above + below) if above + below > 0 else 0.5
                if above > 0 and below > 0:
                    fa = min(max(fa, 0.2), 0.8)
                ya, yb = clo - extra * (1 - fa), chi + extra * fa
            ya, yb = _pad(ya, yb, frac=0.1 if open_curves else 0.08, min_span=2.0 if open_curves else 1.0)
        ya, yb = _include_origin(ya, yb)
        ya, yb = _nice(ya, yb)

    # ---- equal aspect for circles / polar / closed curves
    if equal and x_range is None and y_range is None:
        w, h = b - a, yb - ya
        if w / h < AXES_RATIO:
            extra = h * AXES_RATIO - w
            a, b = a - extra / 2, b + extra / 2
        else:
            extra = w / AXES_RATIO - h
            ya, yb = ya - extra / 2, yb + extra / 2
    elif equal:
        equal = False  # the user fixed the window; respect it exactly

    pi_ticks = bool(periods) and all(_is_pi_multiple(p) for p in periods) and \
        math.pi <= (b - a) <= 16 * math.pi
    return {"x": [round(float(a), 10), round(float(b), 10)],
            "y": [round(float(ya), 10), round(float(yb), 10)],
            "aspect": "equal" if equal else "auto", "auto": x_range is None and y_range is None,
            "pi_ticks": pi_ticks}


def _is_pi_multiple(p: float) -> bool:
    q = p / math.pi
    return abs(q * 12 - round(q * 12)) < 1e-9
