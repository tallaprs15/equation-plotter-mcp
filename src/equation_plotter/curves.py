"""Analysis of parametric curves (x(t), y(t)) and polar curves r = f(theta)."""

from __future__ import annotations

import numpy as np
import sympy as sp
from sympy.calculus.util import function_range, periodicity

from .common import MAX_LISTED, Ctx, tex, classify_solutions, families, lcm_period
from .implicit import conic_info
from .numeric import lambdify, reflect_symmetries
from .numfmt import fnum, num, numeric_interval_json, point, pretty, set_json
from .parser import t, theta, x, y

TWO_PI = 2 * sp.pi
N_SAMPLES = 4000


def _period(expr, var, ctx: Ctx, field="periodicity"):
    if not expr.has(var):
        return sp.Integer(0)  # constant: compatible with any period
    ok, p = ctx.sym(field, lambda: periodicity(expr, var), limit=1.5)
    return p if ok and p is not None and p != 0 else None


def _combined_period(p1, p2):
    if p1 == 0:
        return p2
    if p2 == 0:
        return p1
    return lcm_period(p1, p2)


def _fit_curve(X, Y):
    """Fit a line or conic through sampled points. Returns (kind, coeffs) or None."""
    ok = np.isfinite(X) & np.isfinite(Y)
    X, Y = X[ok], Y[ok]
    if X.size < 12:
        return None
    scale = max(np.abs(X).max(), np.abs(Y).max(), 1e-12)
    xs, ys = X / scale, Y / scale
    if np.ptp(xs) < 1e-9 and np.ptp(ys) < 1e-9:
        return None
    for kind, cols in (("line", [xs, ys, np.ones_like(xs)]),
                       ("conic", [xs ** 2, xs * ys, ys ** 2, xs, ys, np.ones_like(xs)])):
        Dm = np.stack(cols, 1)
        _, s, vt = np.linalg.svd(Dm, full_matrices=False)
        if s[-1] / s[0] < 1e-9 and (len(s) < 2 or s[-2] / s[0] > 1e-7):
            v = vt[-1]
            if kind == "line":
                a, b, c = v
                return kind, (a / scale, b / scale, c)
            A, B, C, D, E, F0 = v
            return kind, (A / scale ** 2, B / scale ** 2, C / scale ** 2, D / scale, E / scale, F0)
    return None


def _rationalise(coeffs):
    m = max(abs(c) for c in coeffs)
    cs = [c / m for c in coeffs]
    # choose the normalisation that makes the first significant coefficient 1
    lead = next(c for c in cs if abs(c) > 1e-9)
    cs = [c / lead for c in cs]
    return [sp.nsimplify(c, [sp.pi, sp.sqrt(2), sp.sqrt(3)], tolerance=1e-7, rational=False)
            if abs(c) > 1e-10 else sp.Integer(0) for c in cs]


class _CurveBase:
    def _sample(self):
        lo, hi = self.prange
        ts = np.linspace(lo, hi, N_SAMPLES)
        return ts, self.Xn(ts), self.Yn(ts)

    def interest(self) -> dict:
        ts, X, Y = self._sample()
        ok = np.isfinite(X) & np.isfinite(Y)
        bbox = None
        open_curve = self.kind == "parametric" and self.period is None
        if ok.any():
            Xo, Yo = X[ok], Y[ok]
            for v in (Xo, Yo):
                p5, p95 = np.percentile(v, [5, 95])
                if v.max() - v.min() > 6 * max(p95 - p5, 1e-9):
                    open_curve = True  # e.g. x = tan t: the sweep runs off to infinity
            if not open_curve:
                bbox = (float(Xo.min()), float(Xo.max()), float(Yo.min()), float(Yo.max()))
        pts = self.key_points()
        if open_curve and not pts and ok.any():
            pts = [(float(np.median(X[ok])), float(np.median(Y[ok])))]
        return {"points": pts, "vlines": [], "hlines": [], "period": None, "bbox": bbox,
                "equal_aspect": self.equal_aspect, "curve": (X, Y), "open": open_curve}


class ParametricAnalysis(_CurveBase):
    kind = "parametric"

    def __init__(self, eq, ctx: Ctx, search, t_range=None):
        self.eq, self.ctx = eq, ctx
        self.xt, self.yt = eq.xt, eq.yt
        self.Xn, self.Yn = lambdify(self.xt, t), lambdify(self.yt, t)
        self.user_range = t_range
        self.out: dict = {}

    def run(self):
        ctx, o = self.ctx, self.out
        o["normalised"] = f"x = {sp.sstr(self.xt)}, y = {sp.sstr(self.yt)}"
        o["latex"] = rf"x = {tex(self.xt)},\ y = {tex(self.yt)}"
        px, py = _period(self.xt, t, ctx), _period(self.yt, t, ctx)
        P = _combined_period(px, py) if px is not None and py is not None else None
        self.period = P if P not in (None, 0) else None
        if self.user_range:
            lo, hi = self.user_range
            note = "user supplied"
        elif self.period is not None and fnum(self.period) <= 60:
            lo, hi = sp.Integer(0), self.period
            note = "one full period"
        else:
            lo, hi = sp.Integer(-5), sp.Integer(5)
            note = "default (curve is not periodic)"
        self.lo, self.hi = sp.nsimplify(lo), sp.nsimplify(hi)
        self.prange = (fnum(self.lo), fnum(self.hi))
        o["parameter"] = {"name": "t", "range": [num(self.lo), num(self.hi)], "chosen": note}
        o["periodicity"] = {"periodic": self.period is not None,
                            "period": num(self.period) if self.period is not None else None}

        ts, X, Y = self._sample()
        self.cartesian = None
        fit = _fit_curve(X, Y)
        cls = None
        if fit:
            kind, coeffs = fit
            cs = _rationalise(coeffs)
            if kind == "line":
                G = cs[0] * x + cs[1] * y + cs[2]
            else:
                G = cs[0] * x ** 2 + cs[1] * x * y + cs[2] * y ** 2 + cs[3] * x + cs[4] * y + cs[5]
            ok, exact = ctx.sym("cartesian_equation", lambda: sp.simplify(
                G.subs({x: self.xt, y: self.yt}, simultaneous=True)) == 0, limit=2.0, quiet=True)
            rng = np.random.default_rng(3)
            tt = rng.uniform(*self.prange, 15)
            res = lambdify(G, (x, y))(self.Xn(tt), self.Yn(tt))
            numeric_ok = np.all(np.abs(res[np.isfinite(res)]) < 1e-7)
            if (ok and exact) or numeric_ok:
                self.cartesian = G
                o["cartesian_equation"] = {"equation": f"{sp.sstr(sp.expand(G))} = 0",
                                           "method": "exact" if ok and exact else "numeric_fit"}
                if kind == "line":
                    cls = "line segment" if self.period is None else "line"
                else:
                    ok2, res2 = ctx.sym("classification", lambda: conic_info(sp.expand(G), ctx),
                                        limit=2.0)
                    if ok2:
                        cls, info, _ = res2
                        o["details"] = info
        tags = sorted(set(families(self.xt, t)) | set(families(self.yt, t)))
        if cls is None:
            closed = self.period is not None
            cls = "closed parametric curve" if closed else "parametric curve"
        o["classification"] = cls
        o["families"] = tags
        self.equal_aspect = cls in ("circle", "ellipse") or self.period is not None
        self._features()

    def _solve_t(self, expr, field):
        interval = sp.Interval(self.lo, self.hi)
        ok, s = self.ctx.sym(field, lambda: sp.solveset(expr, t, interval))
        kind, vals = classify_solutions(s if ok else None)
        if kind == "finite":
            self.ctx.methods[field] = "exact"
            return vals[:MAX_LISTED]
        self.ctx.methods[field] = "numeric"
        from .numeric import find_roots
        return find_roots(lambdify(expr, t), *self.prange)[:MAX_LISTED]

    def _pt(self, tv, **extra):
        if isinstance(tv, sp.Basic):
            ok, xv = self.ctx.sym("evaluate", lambda: sp.simplify(self.xt.subs(t, tv)), limit=0.8, quiet=True)
            ok2, yv = self.ctx.sym("evaluate", lambda: sp.simplify(self.yt.subs(t, tv)), limit=0.8, quiet=True)
            if ok and ok2:
                return point(xv, yv, t=num(tv), **extra)
        tf = fnum(tv)
        return point(float(self.Xn(np.array([tf]))[0]), float(self.Yn(np.array([tf]))[0]),
                     t=num(tv), **extra)

    def _features(self):
        o, ctx = self.out, self.ctx
        o["x_intercepts"] = _uniq_pts([self._pt(v) for v in self._solve_t(self.yt, "x_intercepts")])
        o["y_intercepts"] = _uniq_pts([self._pt(v) for v in self._solve_t(self.xt, "y_intercepts")])
        dx, dy = sp.diff(self.xt, t), sp.diff(self.yt, t)
        dxn = lambdify(dx, t)
        h = [v for v in self._solve_t(dy, "horizontal_tangents") if dy.has(t)]
        v = [w for w in self._solve_t(dx, "vertical_tangents") if dx.has(t)]
        o["horizontal_tangents"] = _uniq_pts([self._pt(tv) for tv in h
                                              if abs(dxn(np.array([fnum(tv)]))[0]) > 1e-12])
        o["vertical_tangents"] = _uniq_pts([self._pt(tv) for tv in v])
        o["stationary_points"] = o["horizontal_tangents"]
        o["start_point"] = self._pt(self.lo)
        o["end_point"] = self._pt(self.hi)
        sx, sy = o["start_point"], o["end_point"]
        o["closed"] = bool(sx["x"]["float"] is not None and sy["x"]["float"] is not None and
                           abs(sx["x"]["float"] - sy["x"]["float"]) < 1e-9 and
                           abs(sx["y"]["float"] - sy["y"]["float"]) < 1e-9)
        interval = sp.Interval(self.lo, self.hi)
        for key, expr in (("domain", self.xt), ("range", self.yt)):
            ok, s = ctx.sym(key, lambda: function_range(expr, t, interval), limit=2.5)
            if ok and s is not None and not s.has(sp.AccumBounds):
                o[key] = set_json(s)
                ctx.methods[key] = "exact"
            else:
                ts, X, Y = self._sample()
                vals = (X if key == "domain" else Y)
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    o[key] = numeric_interval_json(float(vals.min()), float(vals.max()),
                                                   "sampled over the parameter range")
                ctx.methods[key] = "numeric_estimate"

    def key_points(self):
        pts = []
        for key in ("x_intercepts", "y_intercepts", "horizontal_tangents", "vertical_tangents"):
            pts += [(p["x"]["float"], p["y"]["float"]) for p in self.out.get(key, [])]
        return [p for p in pts if None not in p]

    def finalise(self, vp):
        o = self.out
        ts, X, Y = self._sample()
        o["symmetry"] = {"axes": reflect_symmetries(X, Y)}
        self.ctx.methods["symmetry"] = "numeric"
        pts = [(p, "x-intercept") for p in o["x_intercepts"]] + \
              [(p, "y-intercept") for p in o["y_intercepts"]] + \
              [(p, "turning point") for p in o["horizontal_tangents"]]
        plot = {"kind": "parametric", "xt": self.xt, "yt": self.yt, "trange": self.prange,
                "label": o["latex"], "plain": o["normalised"],
                "start": (o["start_point"]["x"]["float"], o["start_point"]["y"]["float"]),
                "closed": o["closed"],
                "points": [(p["x"]["float"], p["y"]["float"], k) for p, k in pts
                           if p["x"]["float"] is not None and p["y"]["float"] is not None]}
        return o, plot


# --------------------------------------------------------------------------- polar

def _classify_polar(R, squared: bool):
    a, b, k = (sp.Wild(n, exclude=[theta]) for n in "abk")
    if squared:
        for trig in (sp.cos, sp.sin):
            m = R.match(a * trig(2 * theta))
            if m and m[a] != 0:
                return "lemniscate (of Bernoulli)", {"a": num(sp.sqrt(sp.Abs(m[a])))}
        return "polar curve (r² form)", {}
    if not R.has(theta):
        return "circle", {"centre": point(0, 0), "radius": num(sp.Abs(R))}
    for trig, axis in ((sp.cos, "x"), (sp.sin, "y")):
        m = R.match(a * trig(theta))
        if m and m[a] != 0 and not m[a].has(theta):
            c = point(m[a] / 2, 0) if axis == "x" else point(0, m[a] / 2)
            return "circle", {"centre": c, "radius": num(sp.Abs(m[a]) / 2), "passes_through_pole": True}
    for trig in (sp.cos, sp.sin):
        m = R.match(a + b * trig(theta))
        if m and m[a] != 0 and m[b] != 0:
            ratio = sp.Abs(m[a]) / sp.Abs(m[b])
            if ratio == 1:
                return "cardioid", {"a": num(m[a]), "b": num(m[b])}
            name = ("limaçon with inner loop" if ratio < 1 else
                    "dimpled limaçon" if ratio < 2 else "convex limaçon")
            return name, {"a": num(m[a]), "b": num(m[b]), "ratio_a_over_b": num(ratio)}
    for trig in (sp.cos, sp.sin):
        m = R.match(a * trig(k * theta))
        if m and m[a] != 0 and m[k] not in (0, 1):
            kk = sp.Abs(m[k])
            if kk.is_Integer:
                petals = kk if kk % 2 else 2 * kk
                return "rose", {"k": num(kk), "petals": int(petals), "petal_length": num(sp.Abs(m[a]))}
            return "rose (rational k)", {"k": num(kk)}
    m = R.match(a / (k + b * sp.cos(theta))) or R.match(a / (k + b * sp.sin(theta)))
    if m and m[a] != 0 and m[k] != 0 and m[b] != 0:
        e = sp.Abs(m[b] / m[k])
        name = "ellipse" if e < 1 else "parabola" if e == 1 else "hyperbola"
        return f"conic ({name}, focus at pole)", {"eccentricity": num(e)}
    m = R.match(a * theta)
    if m and m[a] != 0:
        return "Archimedean spiral", {"a": num(m[a])}
    m = R.match(a * sp.exp(b * theta))
    if m and m[a] != 0 and m[b] != 0:
        return "logarithmic spiral", {"a": num(m[a]), "b": num(m[b])}
    m = R.match(a / theta)
    if m and m[a] != 0:
        return "hyperbolic spiral", {"a": num(m[a])}
    return "polar curve", {}


class PolarAnalysis(_CurveBase):
    kind = "polar"
    equal_aspect = True

    def __init__(self, eq, ctx: Ctx, search, theta_range=None):
        self.eq, self.ctx = eq, ctx
        self.R = eq.rhs_polar
        self.branches = eq.r_exprs
        self.user_range = theta_range
        self.out: dict = {}
        rn = [lambdify(b, theta) for b in self.branches]
        self.rn = rn

    def Xn(self, ts):
        return np.concatenate([r(ts) * np.cos(ts) for r in self.rn]) if len(self.rn) > 1 \
            else self.rn[0](ts) * np.cos(ts)

    def Yn(self, ts):
        return np.concatenate([r(ts) * np.sin(ts) for r in self.rn]) if len(self.rn) > 1 \
            else self.rn[0](ts) * np.sin(ts)

    def run(self):
        ctx, o, eq = self.ctx, self.out, self.eq
        lhs = "r^2" if eq.r_squared else "r"
        o["normalised"] = f"{'r**2' if eq.r_squared else 'r'} = {sp.sstr(self.R)}"
        o["latex"] = f"{lhs} = {tex(self.R)}"
        p = _period(self.R, theta, ctx)
        self.period = p if p not in (None, 0) else None
        if self.user_range:
            lo, hi = self.user_range
            note = "user supplied"
        elif self.period is not None:
            full = lcm_period(self.period, TWO_PI)
            if full is not None and fnum(full) <= 12 * np.pi + 1e-9:
                lo, hi, note = sp.Integer(0), full, "smallest common period with 2π"
            else:
                lo, hi, note = sp.Integer(0), TWO_PI, "0 to 2π"
        elif not self.R.has(theta):
            lo, hi, note = sp.Integer(0), TWO_PI, "0 to 2π"
        else:
            lo, hi, note = sp.Integer(0), 4 * sp.pi, "default (not periodic)"
        self.lo, self.hi = sp.nsimplify(lo), sp.nsimplify(hi)
        self.prange = (fnum(self.lo), fnum(self.hi))
        o["parameter"] = {"name": "theta", "range": [num(self.lo), num(self.hi)], "chosen": note}
        o["periodicity"] = {"periodic": self.period is not None,
                            "period": num(self.period) if self.period is not None else None}
        ok, res = ctx.sym("classification", lambda: _classify_polar(self.R, eq.r_squared), limit=2.0)
        cls, details = res if ok else ("polar curve", {})
        o["classification"] = cls
        o["families"] = families(self.R, theta)
        if details:
            o["details"] = details
        self._features()

    def _features(self):
        o, ctx = self.out, self.ctx
        interval = sp.Interval(self.lo, self.hi)
        ok, s = ctx.sym("pole_crossings", lambda: sp.solveset(self.R, theta, interval))
        kind, vals = classify_solutions(s if ok else None)
        if kind == "finite":
            o["pole_crossings"] = [{"theta": num(v)} for v in vals[:MAX_LISTED]]
            ctx.methods["pole_crossings"] = "exact"
        else:
            from .numeric import find_roots
            o["pole_crossings"] = [{"theta": num(v)} for v in
                                   find_roots(lambdify(self.R, theta), *self.prange)]
            ctx.methods["pole_crossings"] = "numeric"
        key = "r_squared_range" if self.eq.r_squared else "r_range"
        ok, rr = ctx.sym(key, lambda: function_range(self.R, theta, interval), limit=2.5)
        if ok and rr is not None and not rr.has(sp.AccumBounds):
            o[key] = set_json(rr)
            ctx.methods[key] = "exact"
            try:
                o["max_distance_from_pole"] = num(sp.sqrt(sp.Max(sp.Abs(rr.inf), sp.Abs(rr.sup)))
                                                  if self.eq.r_squared else
                                                  sp.Max(sp.Abs(rr.inf), sp.Abs(rr.sup)))
            except Exception:  # noqa: BLE001
                pass
        else:
            ts = np.linspace(*self.prange, N_SAMPLES)
            rv = lambdify(self.R, theta)(ts)
            rv = rv[np.isfinite(rv)]
            if rv.size:
                o[key] = numeric_interval_json(float(rv.min()), float(rv.max()),
                                               "sampled over the theta range")
                o["max_distance_from_pole"] = num(float(np.abs(rv).max() ** (0.5 if self.eq.r_squared else 1)))
            ctx.methods[key] = "numeric_estimate"

        # axis crossings: theta = k*pi/2 inside the range (plus the pole)
        xs, ys = [], []
        k0 = int(np.ceil(self.prange[0] / (np.pi / 2) - 1e-9))
        k1 = int(np.floor(self.prange[1] / (np.pi / 2) + 1e-9))
        for kq in range(k0, min(k1, k0 + 24) + 1):
            th = kq * sp.pi / 2
            for br in self.branches:
                rv = sp.simplify(br.subs(theta, th))
                if fnum(rv) is None:
                    continue
                px, py = sp.simplify(rv * sp.cos(th)), sp.simplify(rv * sp.sin(th))
                (xs if kq % 2 == 0 else ys).append(point(px, py, theta=num(th)))
        if o["pole_crossings"]:
            xs.append(point(sp.Integer(0), sp.Integer(0), note="pole"))
        o["x_intercepts"] = _uniq_pts(xs)
        o["y_intercepts"] = _uniq_pts(ys)

    def key_points(self):
        pts = []
        for key in ("x_intercepts", "y_intercepts"):
            pts += [(p["x"]["float"], p["y"]["float"]) for p in self.out.get(key, [])]
        return [p for p in pts if None not in p]

    def finalise(self, vp):
        o = self.out
        ts, X, Y = self._sample()
        o["symmetry"] = {"axes": reflect_symmetries(X, Y)}
        o["domain"] = _extent(X, "x")
        o["range"] = _extent(Y, "y")
        for k in ("symmetry", "domain", "range"):
            self.ctx.methods[k] = "numeric_estimate" if k != "symmetry" else "numeric"
        pts = [(p, "x-intercept") for p in o["x_intercepts"]] + \
              [(p, "y-intercept") for p in o["y_intercepts"]]
        plot = {"kind": "polar", "branches": self.branches, "trange": self.prange,
                "label": o["latex"], "plain": o["normalised"],
                "points": [(p["x"]["float"], p["y"]["float"], k) for p, k in pts
                           if p["x"]["float"] is not None and p["y"]["float"] is not None]}
        return o, plot


def _extent(v, axis):
    v = v[np.isfinite(v)]
    if not v.size:
        return {"method": "numeric_estimate", "exact": None, "text": "unknown"}
    return numeric_interval_json(float(v.min()), float(v.max()),
                                 f"{axis}-extent of the sampled curve")


def _uniq_pts(pts):
    seen, out = set(), []
    for p in pts:
        if p["x"]["float"] is None or p["y"]["float"] is None:
            continue
        key = (round(p["x"]["float"], 9), round(p["y"]["float"], 9))
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out
