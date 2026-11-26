"""Headless Matplotlib rendering of the analysed equations."""

from __future__ import annotations

import base64
import io
import math

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.mathtext import MathTextParser  # noqa: E402
from matplotlib.ticker import FuncFormatter, MultipleLocator  # noqa: E402

from .numeric import implicit_lines, lambdify  # noqa: E402
from .parser import theta, t, x, y  # noqa: E402

FIG_W, FIG_H, DPI = 9.0, 6.75, 110
AXES_RECT = (0.085, 0.075, 0.86, 0.86)
PALETTE = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#db2777", "#0891b2", "#4b5563"]
INK = "#1f2937"
MUTED = "#6b7280"
GRID = "#e5e7eb"

MARKERS = {
    "x-intercept": dict(marker="o", s=34, zorder=6, label="intercept"),
    "y-intercept": dict(marker="o", s=34, zorder=6, label="intercept"),
    "turning point": dict(marker="s", s=36, zorder=7, label="turning point"),
    "stationary point": dict(marker="s", s=36, zorder=7, label="turning point"),
    "inflection": dict(marker="D", s=30, zorder=7, label="inflection point"),
    "centre": dict(marker="+", s=70, zorder=6, label="centre"),
    "focus": dict(marker="x", s=40, zorder=6, label="focus"),
    "vertex": dict(marker="o", s=26, zorder=6, label="vertex", hollow=True),
    "intersection": dict(marker="o", s=46, zorder=8, label="intersection"),
}
PRIORITY = {"intersection": 0, "turning point": 1, "stationary point": 1, "x-intercept": 2,
            "y-intercept": 2, "inflection": 3, "vertex": 4, "focus": 5, "centre": 6}
MAX_LABELS = 14

_mathtext = MathTextParser("path")


def _label(latex: str, plain: str) -> str:
    s = f"${latex}$"
    try:
        _mathtext.parse(s)
        return s if len(latex) < 120 else plain[:70]
    except Exception:  # noqa: BLE001 - mathtext can't render every SymPy LaTeX construct
        return plain if len(plain) < 70 else plain[:67] + "..."


def _fmt(v: float) -> str:
    if abs(v) < 5e-10:
        return "0"
    if abs(v - round(v)) < 1e-9:
        s = str(int(round(v)))
    elif abs(v) >= 100:
        s = f"{v:.0f}"
    else:
        s = f"{v:.3g}"
    return s.replace("-", "−")


def _break_jumps(xs, ys, vp):
    """Insert NaNs where the curve leaps across the view (poles) or has a true jump."""
    (ya, yb) = vp["y"]
    H = yb - ya
    hi = ys > yb + 0.5 * H
    lo = ys < ya - 0.5 * H
    cross = (hi[:-1] & lo[1:]) | (lo[:-1] & hi[1:])
    idx = list(np.where(cross)[0])
    return idx


def _segments_explicit(fn, vp, vlines):
    a, b = vp["x"]
    span = b - a
    xs = np.linspace(a, b, 5000)
    extra = []
    offs = span * np.geomspace(1e-7, 0.02, 60)
    for v in vlines:
        if a <= v <= b:
            extra += list(v - offs) + list(v + offs)
    xs = np.unique(np.concatenate([xs, np.array(extra)])) if extra else xs
    ys = fn(xs)
    for v in vlines:
        ys[np.abs(xs - v) < span * 5e-8] = np.nan
    H = vp["y"][1] - vp["y"][0]
    breaks = set(_break_jumps(xs, ys, vp))
    # genuine jump discontinuities (floor, sign): the gap does not shrink when refined
    dy = np.abs(np.diff(ys))
    cand = np.where(np.isfinite(dy) & (dy > 0.02 * H))[0][:400]
    for i in cand:
        x0, x1 = xs[i], xs[i + 1]
        gap = dy[i]
        for _ in range(4):
            xm = 0.5 * (x0 + x1)
            ym = fn(np.array([xm]))[0]
            if not np.isfinite(ym):
                break
            left, right = abs(ym - ys[i]), abs(ys[i + 1] - ym)
            if left >= right:
                x1 = xm
                gap = left
            else:
                x0 = xm
                gap = right
        if gap > 0.5 * dy[i]:
            breaks.add(i)
    if breaks:
        order = sorted(breaks)
        xs = np.insert(xs, [i + 1 for i in order], np.nan)
        ys = np.insert(ys, [i + 1 for i in order], np.nan)
    return xs, ys


def _break_curve(X, Y, vp):
    span = max(vp["x"][1] - vp["x"][0], vp["y"][1] - vp["y"][0])
    d = np.hypot(np.diff(X), np.diff(Y))
    idx = np.where(~np.isfinite(d) | (d > 0.25 * span))[0]
    if len(idx):
        X = np.insert(X, idx + 1, np.nan)
        Y = np.insert(Y, idx + 1, np.nan)
    return X, Y


def _pi_formatter(val, _pos):
    k = round(val / (math.pi / 2))
    if abs(val - k * math.pi / 2) > 1e-6:
        return ""
    if k == 0:
        return "0"
    sign = "−" if k < 0 else ""
    k = abs(k)
    if k % 2 == 0:
        n = k // 2
        return f"{sign}{'' if n == 1 else n}π"
    return f"{sign}{'' if k == 1 else k}π/2"


def render(plots: list[dict], vp: dict, fmt: str, intersections: list) -> dict:
    fig = Figure(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor="white")
    ax = fig.add_axes(AXES_RECT)
    (a, b), (ya, yb) = vp["x"], vp["y"]
    ax.set_xlim(a, b)
    ax.set_ylim(ya, yb)
    if vp.get("aspect") == "equal":
        ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("white")
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.minorticks_on()
    ax.grid(True, which="minor", color="#f3f4f6", linewidth=0.5, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#9ca3af")
    ax.tick_params(colors="#374151", labelsize=9, which="both")
    if ya < 0 < yb:
        ax.axhline(0, color=INK, linewidth=1.1, zorder=1)
    if a < 0 < b:
        ax.axvline(0, color=INK, linewidth=1.1, zorder=1)
    ax.set_xlabel("x", fontsize=11, color=INK)
    ax.set_ylabel("y", fontsize=11, color=INK, rotation=0, labelpad=12)
    if vp.get("pi_ticks"):
        base = math.pi / 2 if (b - a) <= 6 * math.pi else math.pi
        ax.xaxis.set_major_locator(MultipleLocator(base))
        ax.xaxis.set_minor_locator(MultipleLocator(base / 2))
        ax.xaxis.set_major_formatter(FuncFormatter(_pi_formatter))

    legend_curves, used_kinds = [], set()
    marks = []
    any_asymptote = False
    H, W = yb - ya, b - a

    for i, p in enumerate(plots):
        color = PALETTE[i % len(PALETTE)]
        label = _label(p["label"], p["plain"])
        style = dict(color=color, linewidth=2.2, zorder=4, solid_capstyle="round")
        dashed = p.get("dashed", False)
        if dashed:
            style["linestyle"] = (0, (6, 4))
        kind = p["kind"]

        if p.get("shade") is not None:
            sx = np.linspace(a, b, 500)
            sy = np.linspace(ya, yb, 500)
            SX, SY = np.meshgrid(sx, sy)
            Z = lambdify(p["shade"]["F"], (x, y))(SX, SY)
            rel = p["shade"]["relation"]
            with np.errstate(invalid="ignore"):
                mask = {"<": Z < 0, "<=": Z <= 0, ">": Z > 0, ">=": Z >= 0}[rel]
            mask = mask & np.isfinite(Z)
            ax.contourf(SX, SY, mask.astype(float), levels=[0.5, 1.5], colors=[color],
                        alpha=0.16, zorder=2)

        if kind == "explicit":
            fn = lambdify(p["expr"], x)
            xs, ys = _segments_explicit(fn, vp, p.get("vlines", []))
            ax.plot(xs, ys, **style)
            for v in p.get("vlines", []):
                if a <= v <= b:
                    any_asymptote = True
                    ax.axvline(v, color=MUTED, linewidth=1.1, linestyle=(0, (4, 4)), zorder=3)
                    ax.annotate(f"x = {_fmt(v)}", (v, yb), xytext=(4, -14), textcoords="offset points",
                                fontsize=8, color=MUTED)
            for h in p.get("hlines", []):
                if ya <= h <= yb:
                    any_asymptote = True
                    ax.axhline(h, color=MUTED, linewidth=1.1, linestyle=(0, (4, 4)), zorder=3)
                    ax.annotate(f"y = {_fmt(h)}", (b, h), xytext=(-48, 4), textcoords="offset points",
                                fontsize=8, color=MUTED)
            for m, c in p.get("oblique", []):
                if m is None or c is None:
                    continue
                any_asymptote = True
                ax.plot([a, b], [m * a + c, m * b + c], color=MUTED, linewidth=1.1,
                        linestyle=(0, (4, 4)), zorder=3)
            for hx, hy in p.get("holes", []):
                ax.scatter([hx], [hy], s=40, facecolors="white", edgecolors=color, linewidths=1.8,
                           zorder=9)
                used_kinds.add("hole")
        elif kind == "implicit":
            Fn = lambdify(p["expr"], (x, y))
            pad = 0.01
            lines = implicit_lines(Fn, (a - pad * W, b + pad * W), (ya - pad * H, yb + pad * H), n=700)
            for L in lines:
                ax.plot(L[:, 0], L[:, 1], **style)
            for m, q in p.get("lines_xy", []):
                any_asymptote = True
                if m is None:
                    ax.axvline(q, color=MUTED, linewidth=1.1, linestyle=(0, (4, 4)), zorder=3)
                else:
                    ax.plot([a, b], [m * a + q, m * b + q], color=MUTED, linewidth=1.1,
                            linestyle=(0, (4, 4)), zorder=3)
        elif kind in ("parametric", "polar"):
            ts = np.linspace(*p["trange"], 6000)
            if kind == "parametric":
                X, Y = lambdify(p["xt"], t)(ts), lambdify(p["yt"], t)(ts)
                curves = [(X, Y)]
            else:
                curves = []
                for br in p["branches"]:
                    R = lambdify(br, theta)(ts)
                    curves.append((R * np.cos(ts), R * np.sin(ts)))
            for X, Y in curves:
                X, Y = _break_curve(X, Y, vp)
                ax.plot(X, Y, **style)
            if kind == "parametric":
                X, Y = curves[0]
                ok = np.where(np.isfinite(X) & np.isfinite(Y))[0]
                if len(ok) > 20:
                    j = ok[len(ok) // 3]
                    k = min(j + 25, ok[-1])
                    if np.hypot(X[k] - X[j], Y[k] - Y[j]) > 1e-9:
                        ax.annotate("", xy=(X[k], Y[k]), xytext=(X[j], Y[j]),
                                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2,
                                                    mutation_scale=18), zorder=6)
                    sx, sy = p.get("start", (None, None))
                    if sx is not None and sy is not None and not p.get("closed"):
                        ax.scatter([sx], [sy], s=34, facecolors="white", edgecolors=color,
                                   linewidths=1.8, zorder=8)
        legend_curves.append(Line2D([], [], label=label, **{k: v for k, v in style.items()
                                                             if k != "zorder"}))
        for px, py, mk in p.get("points", []):
            marks.append((px, py, mk, color))

    for px, py in intersections:
        marks.append((px, py, "intersection", INK))

    # de-duplicate coincident points, keep the most important kind
    marks.sort(key=lambda m: PRIORITY.get(m[2], 9))
    placed = []
    px_per_x = ax.bbox.width / W
    px_per_y = ax.bbox.height / H
    for px, py, mk, color in marks:
        if not (a <= px <= b and ya <= py <= yb):
            continue
        if any(abs(px - q[0]) * px_per_x < 7 and abs(py - q[1]) * px_per_y < 7 for q in placed):
            continue
        placed.append((px, py, mk, color))
        spec = MARKERS.get(mk, MARKERS["x-intercept"])
        kw = dict(s=spec["s"], marker=spec["marker"], zorder=spec["zorder"])
        if spec.get("hollow"):
            ax.scatter([px], [py], facecolors="white", edgecolors=color, linewidths=1.5, **kw)
        elif spec["marker"] in ("+", "x"):
            ax.scatter([px], [py], color=color, linewidths=1.8, **kw)
        else:
            ax.scatter([px], [py], color=color, edgecolors="white", linewidths=1.0, **kw)
        used_kinds.add(spec["label"])

    _annotate(fig, ax, placed)

    handles = legend_curves[:]
    for name, marker in (("intercept", "o"), ("turning point", "s"), ("inflection point", "D"),
                         ("intersection", "o"), ("hole", "o"), ("vertex", "o"),
                         ("focus", "x"), ("centre", "+")):
        if name in used_kinds:
            hollow = name in ("hole", "vertex")
            handles.append(Line2D([], [], linestyle="none", marker=marker, markersize=6,
                                  color=MUTED, markerfacecolor="white" if hollow else MUTED,
                                  markeredgewidth=1.5 if marker in "x+" or hollow else 0.8,
                                  label=name))
    if any_asymptote:
        handles.append(Line2D([], [], color=MUTED, linestyle=(0, (4, 4)), label="asymptote"))
    leg = ax.legend(handles=handles, loc="best", fontsize=9, framealpha=0.92, edgecolor=GRID)
    leg.set_zorder(20)

    buf = io.BytesIO()
    if fmt == "svg":
        fig.savefig(buf, format="svg", facecolor="white")
        mime, w, h, units = "image/svg+xml", FIG_W * 72, FIG_H * 72, "pt"
    else:
        fig.savefig(buf, format="png", dpi=DPI, facecolor="white")
        mime, w, h, units = "image/png", round(FIG_W * DPI), round(FIG_H * DPI), "px"
    return {"format": fmt, "mime_type": mime, "width": w, "height": h, "units": units,
            "encoding": "base64", "data": base64.b64encode(buf.getvalue()).decode("ascii")}


def _annotate(fig, ax, placed):
    """Coordinate labels next to key points, skipping any that would overlap."""
    fig.canvas.draw() if hasattr(fig.canvas, "draw") else None
    trans = ax.transData
    boxes = []
    count = 0
    for px, py, mk, color in placed:
        if count >= MAX_LABELS or mk in ("centre",):
            continue
        text = f"({_fmt(px)}, {_fmt(py)})"
        w = 6.3 * len(text) + 6
        h = 15
        X, Y = trans.transform((px, py))
        for dx, dy in ((7, 7), (7, -7 - h), (-7 - w, 7), (-7 - w, -7 - h)):
            box = (X + dx, Y + dy, X + dx + w, Y + dy + h)
            bb = ax.bbox
            if box[0] < bb.x0 or box[2] > bb.x1 or box[1] < bb.y0 or box[3] > bb.y1:
                continue
            if any(not (box[2] < o[0] or box[0] > o[2] or box[3] < o[1] or box[1] > o[3])
                   for o in boxes):
                continue
            boxes.append(box)
            # also reserve the marker itself
            ax.annotate(text, (px, py), xytext=(dx + 3, dy + 3), textcoords="offset pixels",
                        fontsize=8, color="#111827", zorder=15,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
            count += 1
            break
