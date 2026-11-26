"""Public entry point: ``analyse_and_plot``."""

from __future__ import annotations

import math
import time

import numpy as np
import sympy as sp

from . import __version__
from .common import Ctx, tex, classify_solutions, realise
from .curves import ParametricAnalysis, PolarAnalysis
from .errors import PlotterError
from .explicit import ExplicitAnalysis
from .implicit import ImplicitAnalysis
from .numeric import find_roots, lambdify
from .numfmt import fnum, point
from .parser import ParsedEquation, parse_equation, parse_expression, x, y
from .sandbox import Budget, HardTimeout, Worker
from .viewport import choose

SCHEMA_VERSION = "1.0"
MAX_EQUATIONS = 8
MAX_COORD = 1e6
DEFAULT_SYMBOLIC_SECONDS = 20.0
STEP_SECONDS = 3.0
RENDER_SECONDS = 20.0
FORMATS = {"png", "svg", "none"}
_REL_LATEX = {"<": "<", ">": ">", "<=": r"\leq", ">=": r"\geq"}


# --------------------------------------------------------------------------- inequality wrapper

class InequalityAnalysis:
    kind = "inequality"

    def __init__(self, eq: ParsedEquation, ctx, search):
        self.eq = eq
        if eq.boundary_kind == "explicit":
            inner_eq = ParsedEquation(eq.input, "explicit", eq.name, f=eq.f, lhs=y, rhs=eq.f)
            self.inner = ExplicitAnalysis(inner_eq, ctx, search)
        else:
            inner_eq = ParsedEquation(eq.input, "implicit", lhs=eq.lhs, rhs=eq.rhs, F=eq.F)
            self.inner = ImplicitAnalysis(inner_eq, ctx, search)
        self.out = self.inner.out

    def run(self):
        self.inner.run()
        eq, o = self.eq, self.out
        rel = eq.relation
        if eq.boundary_kind == "explicit":
            o["normalised"] = f"y {rel} {sp.sstr(eq.f)}"
            o["latex"] = f"y {_REL_LATEX[rel]} {tex(eq.f)}"
            side = "above" if rel in (">", ">=") else "below"
            region = f"points {side} the curve y = {sp.sstr(eq.f)}"
        else:
            o["normalised"] = f"{sp.sstr(eq.lhs)} {rel} {sp.sstr(eq.rhs)}"
            o["latex"] = f"{tex(eq.lhs)} {_REL_LATEX[rel]} {tex(eq.rhs)}"
            region = f"points where {sp.sstr(eq.F)} {rel} 0"
        strict = rel in ("<", ">")
        o["inequality"] = {"relation": rel, "boundary_type": eq.boundary_kind,
                           "boundary_included": not strict, "region": region,
                           "boundary_style": "dashed" if strict else "solid"}

    def interest(self):
        it = self.inner.interest()
        it.pop("sampler", None) if self.eq.boundary_kind != "explicit" else None
        return it

    def finalise(self, vp):
        o, plot = self.inner.finalise(vp)
        plot["shade"] = {"F": self.eq.F, "relation": self.eq.relation}
        plot["dashed"] = self.eq.relation in ("<", ">")
        plot["label"], plot["plain"] = o["latex"], o["normalised"]
        return o, plot


def _make(eq, ctx, search, opts):
    if eq.kind == "explicit":
        return ExplicitAnalysis(eq, ctx, search)
    if eq.kind == "implicit":
        return ImplicitAnalysis(eq, ctx, search)
    if eq.kind == "parametric":
        return ParametricAnalysis(eq, ctx, search, opts.get("t_range"))
    if eq.kind == "polar":
        return PolarAnalysis(eq, ctx, search, opts.get("theta_range"))
    return InequalityAnalysis(eq, ctx, search)


# --------------------------------------------------------------------------- jobs (run isolated)

def _analysis_job(equations: list[str], opts: dict):
    warnings: list = []
    parsed = []
    for i, text in enumerate(equations):
        try:
            parsed.append(parse_equation(text))
        except PlotterError as e:
            e.equation = i
            raise
    n = len(parsed)
    total = opts["symbolic_seconds"]
    per_eq = max(total * 0.85 / n, 2.0)
    search = tuple(opts["x_range"]) if opts.get("x_range") else (-10.0, 10.0)
    analyses, ctxs = [], []
    for i, eq in enumerate(parsed):
        ctx = Ctx(i, Budget(per_eq, STEP_SECONDS), warnings, symbolic=opts["symbolic"])
        A = _make(eq, ctx, search, opts)
        A.run()
        analyses.append(A)
        ctxs.append(ctx)
    vp = choose([A.interest() for A in analyses], opts.get("x_range"), opts.get("y_range"))
    outs, plots = [], []
    for i, (A, ctx, eq) in enumerate(zip(analyses, ctxs, parsed)):
        o, plot = A.finalise(vp)
        entry = {"index": i, "input": eq.input, "type": eq.kind}
        entry.update(o)
        entry["methods"] = dict(sorted(ctx.methods.items()))
        outs.append(entry)
        plots.append(plot)
    ix_ctx = Ctx(-1, Budget(max(total * 0.15, 2.0), 2.0), warnings, symbolic=opts["symbolic"])
    inters = _intersections(parsed, vp, ix_ctx) if n > 1 else []
    for w in warnings:
        if w.get("equation") == -1:
            w["equation"] = None
    return {"equations": outs, "intersections": inters, "warnings": warnings}, plots, vp


def _boundary(eq):
    if eq.kind != "inequality":
        return eq
    if eq.boundary_kind == "explicit":
        return ParsedEquation(eq.input, "explicit", f=eq.f, lhs=y, rhs=eq.f)
    return ParsedEquation(eq.input, "implicit", lhs=eq.lhs, rhs=eq.rhs, F=eq.F)


def _intersections(parsed, vp, ctx):
    (a, b), (ya, yb) = vp["x"], vp["y"]
    parsed = [_boundary(e) for e in parsed]
    out = []
    for i in range(len(parsed)):
        for j in range(i + 1, len(parsed)):
            e1, e2 = parsed[i], parsed[j]
            kinds = {e1.kind, e2.kind}
            if not kinds <= {"explicit", "implicit"}:
                continue
            if e1.kind == "explicit" and e2.kind == "explicit":
                G = e1.f - e2.f
                yfun = e1.f
            else:
                ex, im = (e1, e2) if e1.kind == "explicit" else (e2, e1)
                if ex.kind == "explicit":
                    G = im.F.subs(y, ex.f)
                    yfun = ex.f
                else:
                    ok, sols = ctx.sym("intersections", lambda: sp.solve([e1.F, e2.F], [x, y],
                                                                         dict=True))
                    pts = []
                    for s in (sols or []) if ok else []:
                        if x in s and y in s and fnum(s[x]) is not None and fnum(s[y]) is not None:
                            pts.append(point(s[x], s[y]))
                    if not ok:
                        ctx.warn("intersections", "unavailable",
                                 f"intersections of equations {i} and {j} could not be computed")
                    out.append({"equations": [i, j], "points": pts[:40],
                                "method": "exact" if ok else "unavailable"})
                    continue
            ok, s = ctx.sym("intersections", lambda: sp.solveset(G, x, sp.S.Reals))
            kind, vals = classify_solutions(s if ok else None)
            method = "exact"
            if kind == "periodic":
                vals = realise(vals, a, b) or []
                method = "exact_in_viewport"
            elif kind != "finite":
                vals = find_roots(lambdify(G, x), a, b)
                method = "numeric_in_viewport"
            pts = []
            f1 = lambdify(yfun, x)
            check = [lambdify(e.f, x) if e.kind == "explicit" else None for e in (e1, e2)]
            for v in vals[:40]:
                fv = fnum(v)
                if fv is None or any(c is not None and not np.isfinite(c(np.array([fv]))[0])
                                     for c in check):
                    continue
                if isinstance(v, sp.Basic):
                    ok, yv = ctx.sym("intersections", lambda: sp.simplify(yfun.subs(x, v)),
                                     limit=0.8, quiet=True)
                    yv = yv if ok and fnum(yv) is not None else float(f1(np.array([fv]))[0])
                else:
                    yv = float(f1(np.array([fv]))[0])
                pts.append(point(v, yv))
            out.append({"equations": [i, j], "points": pts, "method": method})
    return out


def _render_job(plots, vp, fmt, inter_pts):
    from .render import render
    return render(plots, vp, fmt, inter_pts)


# --------------------------------------------------------------------------- public API

def _parse_range(r, name):
    if r is None:
        return None, None
    if isinstance(r, str):
        parts = r.strip().strip("[]()").split(",")
    else:
        parts = list(r)
    if len(parts) != 2:
        raise PlotterError("invalid_range", f"{name} must have exactly two values")
    vals = []
    for p in parts:
        if isinstance(p, (int, float)) and not isinstance(p, bool):
            v = sp.nsimplify(p) if float(p).is_integer() else sp.Float(p)
        elif isinstance(p, str):
            v = parse_expression(p)
        else:
            raise PlotterError("invalid_range", f"{name} values must be numbers or strings")
        if v.free_symbols:
            raise PlotterError("invalid_range", f"{name} values must be constants")
        fv = fnum(v)
        if fv is None or abs(fv) > MAX_COORD:
            raise PlotterError("invalid_range", f"{name} values must be finite and within ±{MAX_COORD:g}")
        vals.append(v)
    if not fnum(vals[0]) < fnum(vals[1]):
        raise PlotterError("invalid_range", f"{name}: the first value must be less than the second")
    return vals, (fnum(vals[0]), fnum(vals[1]))


def analyse_and_plot(equations: list[str] | str, x_range=None, y_range=None,
                     image_format: str = "png", *, t_range=None, theta_range=None,
                     timeout: float | None = None, isolate: bool = True) -> dict:
    """Analyse one or more equations and render them on one set of axes.

    Returns a JSON-serialisable dict. On failure returns ``{"version", "error": {...}}``;
    this function never raises for bad input.
    """
    started = time.monotonic()
    try:
        if isinstance(equations, str):
            equations = [equations]
        if not isinstance(equations, (list, tuple)) or not equations:
            raise PlotterError("invalid_input", "Provide at least one equation")
        if len(equations) > MAX_EQUATIONS:
            raise PlotterError("too_many_equations", f"At most {MAX_EQUATIONS} equations per request")
        for i, e in enumerate(equations):
            if not isinstance(e, str):
                raise PlotterError("invalid_input", "Each equation must be a string", equation=i)
        fmt = (image_format or "none").lower()
        if fmt not in FORMATS:
            raise PlotterError("invalid_format", "image_format must be 'png', 'svg' or 'none'")
        _, xr = _parse_range(x_range, "x_range")
        _, yr = _parse_range(y_range, "y_range")
        tr, _ = _parse_range(t_range, "t_range")
        thr, _ = _parse_range(theta_range, "theta_range")
        sym_s = float(timeout) if timeout else DEFAULT_SYMBOLIC_SECONDS
        if not (0.5 <= sym_s <= 120) or math.isnan(sym_s):
            raise PlotterError("invalid_timeout", "timeout must be between 0.5 and 120 seconds")
        opts = {"x_range": xr, "y_range": yr, "t_range": tr, "theta_range": thr,
                "symbolic": True, "symbolic_seconds": sym_s}
        equations = list(equations)

        hard_timeout = False
        worker = Worker() if isolate else None
        try:
            if worker:
                try:
                    data, plots, vp = worker.call("analyse", (equations, opts), sym_s + 10)
                except HardTimeout:
                    hard_timeout = True
                    opts["symbolic"] = False
                    data, plots, vp = worker.call("analyse", (equations, opts), 20)
            else:
                data, plots, vp = _analysis_job(equations, opts)
            warnings = data["warnings"]
            if hard_timeout:
                warnings.insert(0, {"equation": None, "field": "*", "code": "hard_timeout",
                                    "message": f"Symbolic analysis exceeded {sym_s + 10:.0f}s and "
                                               "was stopped; all results are numeric estimates"})
            t_an = time.monotonic()

            image = None
            if fmt != "none":
                inter_pts = [(p["x"]["float"], p["y"]["float"]) for it in data["intersections"]
                             for p in it["points"] if p["x"]["float"] is not None
                             and p["y"]["float"] is not None]
                try:
                    if worker:
                        image = worker.call("render", (plots, vp, fmt, inter_pts), RENDER_SECONDS)
                    else:
                        image = _render_job(plots, vp, fmt, inter_pts)
                except HardTimeout:
                    warnings.append({"equation": None, "field": "image", "code": "render_timeout",
                                     "message": f"Rendering exceeded {RENDER_SECONDS:.0f}s; no image"})
                except PlotterError as e:
                    warnings.append({"equation": None, "field": "image", "code": "render_failed",
                                     "message": e.message})
        finally:
            if worker:
                worker.close()
        done = time.monotonic()
        viewport = {"x": vp["x"], "y": vp["y"], "aspect": vp["aspect"], "auto": vp["auto"]}
        return {
            "version": SCHEMA_VERSION,
            "plotter_version": __version__,
            "equations": data["equations"],
            "intersections": data["intersections"],
            "viewport": viewport,
            "warnings": warnings,
            "timing_ms": {"analysis": round((t_an - started) * 1000),
                          "render": round((done - t_an) * 1000),
                          "total": round((done - started) * 1000)},
            "image": image,
        }
    except PlotterError as e:
        return {"version": SCHEMA_VERSION, "plotter_version": __version__, "error": e.to_dict()}
    except Exception as e:  # noqa: BLE001 - never leak a stack trace to callers
        return {"version": SCHEMA_VERSION, "plotter_version": __version__,
                "error": {"code": "internal_error", "message": f"{type(e).__name__}: {e}"}}
