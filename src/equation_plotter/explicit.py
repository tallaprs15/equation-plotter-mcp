"""Analysis of explicit functions y = f(x)."""

from __future__ import annotations

import numpy as np
import sympy as sp
from sympy.calculus.util import continuous_domain, function_range, periodicity

from .common import (MAX_LISTED, Ctx, tex, classify_function, classify_solutions, general_text,
                     realise, strip_steps, tidy)
from .numeric import find_roots, lambdify, sign_at
from .numfmt import (fnum, interval_json, num, numeric_interval_json, point, pretty,
                     set_intervals, set_json)
from .parser import x

NEG_OO = sp.S.NegativeInfinity


class Feature:
    """A set of x-values: exact finite list, a periodic general solution, or numeric."""

    def __init__(self, kind: str, vals=None, general=None):
        self.kind = kind          # finite | periodic | numeric | none
        self.vals = vals or []
        self.general = general

    def in_window(self, a, b):
        if self.kind == "periodic":
            got = realise(self.general, a, b)
            return got if got is not None else []
        return [v for v in self.vals if (fv := fnum(v)) is not None and a <= fv <= b]


class ExplicitAnalysis:
    kind = "explicit"

    def __init__(self, eq, ctx: Ctx, search: tuple[float, float]):
        self.eq = eq
        self.f = eq.f
        self.ctx = ctx
        self.search = search
        self.fn = lambdify(self.f, x)
        self.d1 = sp.diff(self.f, x)
        self.d2 = sp.diff(self.d1, x)
        self.f1n = lambdify(self.d1, x)
        self.f2n = lambdify(self.d2, x)
        self.out: dict = {}
        self._raw: dict = {}

    # ------------------------------------------------------------------ helpers
    def in_domain(self, v) -> bool:
        fv = fnum(v)
        if fv is None:
            return False
        if self.dom is not None:
            try:
                c = self.dom.contains(v)
                if c is sp.true:
                    return True
                if c is sp.false:
                    return False
            except Exception:  # noqa: BLE001
                pass
        return bool(np.isfinite(self.fn(np.array([fv]))[0]))

    def yval(self, xv):
        """Exact y if cheap, else float."""
        ok, v = self.ctx.sym("evaluate", lambda: sp.simplify(self.f.subs(x, xv)), limit=0.8,
                             quiet=True)
        if ok and fnum(v) is not None and len(sp.sstr(v)) < 120:
            return v
        fv = fnum(xv)
        return float(self.fn(np.array([fv]))[0]) if fv is not None else None

    def solve(self, expr, field):
        ok, s = self.ctx.sym(field, lambda: sp.solveset(expr, x, sp.S.Reals))
        kind, vals = classify_solutions(s if ok else None)
        if kind == "finite":
            vals = [v for v in vals if self.in_domain(v)]
            if len(vals) > MAX_LISTED:
                self.ctx.warn(field, "truncated", f"{field}: only the first {MAX_LISTED} listed")
                vals = vals[:MAX_LISTED]
            return Feature("finite", vals)
        if kind == "periodic":
            return Feature("periodic", general=vals)
        if kind == "all":
            return Feature("none")
        return Feature("numeric")

    # ------------------------------------------------------------------ main
    def run(self):
        f, ctx = self.f, self.ctx
        o = self.out
        o["normalised"] = f"{self.eq.name}(x) = {sp.sstr(f)}" if self.eq.name != "y" \
            else f"y = {sp.sstr(f)}"
        o["latex"] = (f"{self.eq.name}(x) = " if self.eq.name != "y" else "y = ") + tex(f)
        cls, tags = classify_function(f, x)
        o["classification"] = cls
        o["families"] = tags

        ok, dom = ctx.sym("domain", lambda: continuous_domain(strip_steps(f), x, sp.S.Reals))
        self.dom = dom if ok else None
        ok, d1 = ctx.sym("derivative", lambda: tidy(self.d1, ctx), limit=1.5, quiet=True)
        if ok:
            self.d1 = d1
        ok, d2 = ctx.sym("derivative", lambda: tidy(self.d2, ctx), limit=1.5, quiet=True)
        if ok:
            self.d2 = d2

        self.roots = self.solve(f, "x_intercepts")
        self.stat = self.solve(self.d1, "stationary_points") if f.has(x) else Feature("none")
        self.infl_c = self.solve(self.d2, "points_of_inflection") if f.has(x) else Feature("none")
        self._asymptotes()
        self._symmetry()
        self.period = None
        if f.has(x) and not f.is_polynomial(x):
            ok, p = ctx.sym("periodicity", lambda: periodicity(f, x))
            self.period = p if ok and p is not None and p != 0 else None
        a, b = self.search
        if self.roots.kind == "numeric":
            self.roots.vals = find_roots(self.fn, a, b)
        if self.stat.kind == "numeric":
            self.stat.vals = find_roots(self.f1n, a, b)
        if self.infl_c.kind == "numeric":
            self.infl_c.vals = find_roots(self.f2n, a, b)

    def _asymptotes(self):
        f, ctx = self.f, self.ctx
        cands: list = []
        self.sing_periodic = None
        ok, sing = ctx.sym("asymptotes", lambda: sp.singularities(f, x, sp.S.Reals))
        kind, vals = classify_solutions(sing if ok else None)
        if kind == "finite":
            cands.extend(vals)
        elif kind == "periodic":
            self.sing_periodic = vals
        if self.dom is not None and self.dom is not sp.S.Reals:
            ok, bd = ctx.sym("domain", lambda: self.dom.boundary, quiet=True)
            if ok and isinstance(bd, sp.FiniteSet):
                cands.extend(bd.args)
        uniq = {}
        for c in cands:
            if fnum(c) is not None:
                uniq.setdefault(round(fnum(c), 10), c)
        self.vert, self.holes, self.jumps = [], [], []
        for c in [uniq[k] for k in sorted(uniq)][:MAX_LISTED]:
            self._classify_point(c)
        self.vert_numeric = ok is False and kind == "unknown" and self.ctx.symbolic is False

        # horizontal / oblique / end behaviour
        self.horiz, self.oblique, self.ends = [], [], {}
        for label, target in (("x→+∞", sp.oo), ("x→-∞", NEG_OO)):
            probe = 1e6 if target is sp.oo else -1e6
            if not np.isfinite(self.fn(np.array([probe]))[0]) and \
                    not np.isfinite(self.fn(np.array([probe * 1e-3]))[0]):
                self.ends[label] = "outside domain"
                continue
            ok, L = ctx.sym("asymptotes", lambda: sp.limit(f, x, target), limit=2.0)
            if not ok:
                v = self.fn(np.array([probe]))[0]
                self.ends[label] = f"≈ {v:.6g} (numeric)" if np.isfinite(v) else "unknown"
                continue
            if isinstance(L, sp.AccumBounds):
                self.ends[label] = f"oscillates between {pretty(L.min)} and {pretty(L.max)}"
                continue
            if L in (sp.oo, NEG_OO):
                self.ends[label] = "y→+∞" if L is sp.oo else "y→-∞"
                if not f.is_polynomial(x):
                    ok, m = ctx.sym("asymptotes", lambda: sp.limit(f / x, x, target), limit=1.5)
                    if ok and m.is_finite and m != 0:
                        ok, c = ctx.sym("asymptotes", lambda: sp.limit(f - m * x, x, target),
                                        limit=1.5)
                        far = np.array([probe * 1e-3, probe * 1e-2])
                        gap = self.fn(far) - (fnum(m) or 0) * far - (fnum(c) or 0)
                        if ok and np.all(np.abs(gap) < 1e-9 * np.abs(far)):
                            ok = False  # the curve *is* this line out there (e.g. |x - 1| - 2)
                        if ok and c.is_finite and c.is_real:
                            key = (sp.sstr(m), sp.sstr(c))
                            if key not in [(sp.sstr(a), sp.sstr(b)) for a, b, _ in self.oblique]:
                                self.oblique.append((m, c, label))
                            else:
                                self.oblique = [(a, b, "x→±∞") for a, b, _ in self.oblique]
                continue
            if L.is_real and L.is_finite:
                self.ends[label] = f"y→{pretty(L)}"
                if f.has(x):
                    if any(sp.sstr(L) == sp.sstr(h) for h, _ in self.horiz):
                        self.horiz = [(h, "x→±∞") for h, _ in self.horiz]
                    else:
                        self.horiz.append((L, label))
            else:
                self.ends[label] = "undefined"

    def _classify_point(self, c):
        f, ctx = self.f, self.ctx
        fc = fnum(c)
        eps = 1e-7 * (1 + abs(fc))
        sides = {}
        for d, sgn in (("-", -1), ("+", 1)):
            if not np.isfinite(self.fn(np.array([fc + sgn * eps]))[0]) and \
                    not np.isfinite(self.fn(np.array([fc + sgn * 1e3 * eps]))[0]):
                continue  # side lies outside the domain
            ok, L = ctx.sym("asymptotes", lambda: sp.limit(f, x, c, d), limit=1.5)
            if not ok or isinstance(L, sp.AccumBounds) or L.is_real is False:
                v = self.fn(np.array([fc + sgn * eps]))[0]
                L = (sp.oo if v > 0 else NEG_OO) if abs(v) > 1e6 else (v if np.isfinite(v) else None)
            sides[d] = L
        if not sides or any(v is None for v in sides.values()):
            return
        if any(v in (sp.oo, NEG_OO) for v in sides.values()):
            self.vert.append((c, sides))
            return
        vals = list(sides.values())
        defined = np.isfinite(self.fn(np.array([fc]))[0]) if self.dom is None else self.in_domain(c)
        if len(vals) == 2 and abs((fnum(vals[0]) or 0) - (fnum(vals[1]) or 0)) < 1e-9:
            if not defined:
                self.holes.append((c, vals[0]))
        elif len(vals) == 2:
            self.jumps.append((c, sides))

    def _symmetry(self):
        f, ctx = self.f, self.ctx
        a, b = self.search
        rng = np.random.default_rng(7)
        xs = rng.uniform(0.05, max(abs(a), abs(b)), 25)
        fp, fm = self.fn(xs), self.fn(-xs)
        mask = np.isfinite(fp) & np.isfinite(fm)
        parity = "neither"
        if mask.sum() >= 5:
            scale = 1e-7 * (1 + np.abs(fp[mask]))
            if np.all(np.abs(fp[mask] - fm[mask]) < scale):
                parity = "even"
            elif np.all(np.abs(fp[mask] + fm[mask]) < scale):
                parity = "odd"
        # a counterexample at sampled points *proves* "neither"; even/odd needs confirmation
        method = "exact" if parity == "neither" and mask.sum() >= 5 else "numeric"
        if parity != "neither":
            sgn = -1 if parity == "even" else 1
            ok, z = ctx.sym("symmetry",
                            lambda: sp.simplify(f.subs(x, -x) + sgn * f) == 0, limit=1.5, quiet=True)
            if ok and z:
                method = "exact"
        elif not f.has(x):
            parity = "even"
            method = "exact"
        self.parity = parity
        ctx.methods["symmetry"] = method

    # ------------------------------------------------------------------ viewport interface
    def interest(self) -> dict:
        a, b = self.search
        pts = []
        for xv in self.roots.in_window(-1e6, 1e6) if self.roots.kind != "periodic" else []:
            pts.append((fnum(xv), 0.0))
        if self.roots.kind == "periodic" and self.period is None:
            pts += [(fnum(v), 0.0) for v in self.roots.in_window(-5, 5)]
        for feat in (self.stat, self.infl_c):
            vals = feat.vals if feat.kind in ("finite", "numeric") else \
                feat.in_window(-5, 5) if feat.kind == "periodic" and self.period is None else []
            if vals:
                for xv in vals[:MAX_LISTED]:
                    fv = fnum(xv)
                    yv = self.fn(np.array([fv]))[0]
                    if np.isfinite(yv):
                        pts.append((fv, float(yv)))
        y0 = self.fn(np.array([0.0]))[0]
        if np.isfinite(y0):
            pts.append((0.0, float(y0)))
        pts += [(fnum(c), fnum(v)) for c, v in self.holes]
        return {
            "points": [p for p in pts if p[0] is not None and p[1] is not None],
            "vlines": [fnum(c) for c, _ in self.vert] + [fnum(c) for c, _ in self.jumps],
            "hlines": [fnum(h) for h, _ in self.horiz if fnum(h) is not None],
            "period": fnum(self.period) if self.period is not None else None,
            "sampler": self.fn,
            "equal_aspect": False,
        }

    # ------------------------------------------------------------------ final JSON
    def finalise(self, vp) -> tuple[dict, dict]:
        f, ctx, o = self.f, self.ctx, self.out
        (a, b), (ya, yb) = vp["x"], vp["y"]
        plot_pts = []

        # domain & range
        if self.dom is not None:
            o["domain"] = set_json(self.dom)
            ctx.methods["domain"] = "exact"
        else:
            o["domain"] = self._numeric_domain(a, b)
            ctx.methods["domain"] = "numeric_estimate"
        if self.dom is not None and f.has(x):
            ok, rg = ctx.sym("range", lambda: function_range(f, x, self.dom), limit=3.0)
        else:
            ok, rg = (True, sp.FiniteSet(f)) if not f.has(x) else (False, None)
        if ok and rg is not None and not rg.has(sp.AccumBounds):
            o["range"] = set_json(rg)
            ctx.methods["range"] = "exact"
        else:
            o["range"] = self._numeric_range(a, b)
            ctx.methods["range"] = "numeric_estimate"

        # intercepts
        o["x_intercepts"] = self._points(self.roots, a, b, "x_intercepts", zero_y=True)
        plot_pts += [(p, "x-intercept") for p in o["x_intercepts"]]
        o["y_intercept"] = None
        if self.in_domain(sp.Integer(0)):
            yv = self.yval(sp.Integer(0))
            if yv is not None and fnum(yv) is not None:
                o["y_intercept"] = point(sp.Integer(0), yv)
                plot_pts.append((o["y_intercept"], "y-intercept"))

        # stationary / inflection
        stats = []
        for p in self._points(self.stat, a, b, "stationary_points"):
            p["nature"] = self._nature(p["x"]["float"])
            stats.append(p)
        o["stationary_points"] = stats
        plot_pts += [(p, "turning point" if "local" in p["nature"] else "stationary point")
                     for p in stats]
        infl = []
        for p in self._points(self.infl_c, a, b, "points_of_inflection"):
            c = p["x"]["float"]
            h = 1e-4 * (1 + abs(c))
            s1, s2 = sign_at(self.f2n, c - h), sign_at(self.f2n, c + h)
            if s1 * s2 < 0:
                infl.append(p)
        o["points_of_inflection"] = infl
        plot_pts += [(p, "inflection") for p in infl if not any(
            abs(p["x"]["float"] - q["x"]["float"]) < 1e-9 for q in stats)]

        # asymptotes / holes
        verts = [(c, sides) for c, sides in self.vert]
        if self.sing_periodic is not None:
            realised = realise(self.sing_periodic, a, b) or []
            for c in realised:
                self._classify_point(c)
            verts = list(self.vert)
            ctx.methods["asymptotes"] = "exact_in_viewport"
            o["asymptotes_general"] = {"vertical": "x ∈ " + general_text(self.sing_periodic)}
        if not ctx.symbolic or ("asymptotes", "symbolic_failed") in ctx._warned \
                or ("asymptotes", "timeout") in ctx._warned:
            for c in self._numeric_vertical(a, b):
                if not any(abs(c - fnum(v)) < 1e-4 * (b - a) for v, _ in verts):
                    verts.append((c, {}))
            ctx.methods.setdefault("asymptotes", "numeric_estimate")
        ctx.methods.setdefault("asymptotes", "exact")
        o["asymptotes"] = {
            "vertical": [{"x": num(c), "equation": f"x = {pretty(c)}",
                          **({"left_limit": pretty(s["-"])} if "-" in s else {}),
                          **({"right_limit": pretty(s["+"])} if "+" in s else {})}
                         for c, s in verts],
            "horizontal": [{"y": num(h), "equation": f"y = {pretty(h)}", "as": side}
                           for h, side in self.horiz],
            "oblique": [{"slope": num(m), "intercept": num(c),
                         "equation": f"y = {sp.sstr(m * x + c)}", "as": side}
                        for m, c, side in self.oblique],
        }
        o["holes"] = [point(c, v) for c, v in self.holes]
        if self.jumps:
            o["jump_discontinuities"] = [{"x": num(c), "left_limit": pretty(s.get("-")),
                                          "right_limit": pretty(s.get("+"))} for c, s in self.jumps]

        o["symmetry"] = {"parity": self.parity,
                         "description": {"even": "symmetric about the y-axis",
                                         "odd": "symmetric about the origin",
                                         "neither": "no even/odd symmetry"}[self.parity]}
        o["periodicity"] = {"periodic": self.period is not None,
                            "period": num(self.period) if self.period is not None else None}
        o["end_behaviour"] = self.ends
        self._monotonic(o, a, b, [c for c, _ in verts])
        details = self._details()
        if details:
            o["details"] = details

        plot = {
            "kind": "explicit", "expr": f, "label": o["latex"], "plain": o["normalised"],
            "vlines": [fnum(c) for c, _ in verts if fnum(c) is not None],
            "hlines": [fnum(h) for h, _ in self.horiz],
            "oblique": [(fnum(m), fnum(c)) for m, c, _ in self.oblique],
            "holes": [(p["x"]["float"], p["y"]["float"]) for p in o["holes"]],
            "points": [(p["x"]["float"], p["y"]["float"], kind) for p, kind in plot_pts
                       if p["x"]["float"] is not None and p["y"]["float"] is not None],
        }
        return o, plot

    def _points(self, feat: Feature, a, b, field, zero_y=False):
        out = []
        if feat.kind == "periodic":
            vals = feat.in_window(a, b)
            self.out.setdefault("general_solutions", {})[field] = "x ∈ " + general_text(feat.general)
            self.ctx.methods[field] = "exact_in_viewport"
        elif feat.kind == "numeric":
            vals = find_roots({"x_intercepts": self.fn, "stationary_points": self.f1n,
                               "points_of_inflection": self.f2n}[field], a, b)
            self.ctx.methods[field] = "numeric_in_viewport"
        else:
            vals = feat.vals
            self.ctx.methods[field] = "exact"
        self._raw[field] = list(vals[:MAX_LISTED])
        for v in vals[:MAX_LISTED]:
            if zero_y:
                out.append(point(v, sp.Integer(0)))
            else:
                yv = self.yval(v) if isinstance(v, sp.Basic) else float(self.fn(np.array([v]))[0])
                if yv is not None and fnum(yv) is not None:
                    out.append(point(v, yv))
        return out

    def _nature(self, c: float) -> str:
        v2 = self.f2n(np.array([c]))[0]
        if np.isfinite(v2) and abs(v2) > 1e-9:
            return "local minimum" if v2 > 0 else "local maximum"
        h = 1e-4 * (1 + abs(c))
        s1, s2 = sign_at(self.f1n, c - h), sign_at(self.f1n, c + h)
        if s1 < 0 < s2:
            return "local minimum"
        if s1 > 0 > s2:
            return "local maximum"
        if s1 == s2 and s1 != 0:
            return "stationary inflection"
        return "stationary"

    def _numeric_domain(self, a, b):
        xs = np.linspace(a, b, 4001)
        ok = np.isfinite(self.fn(xs))
        if ok.all():
            return numeric_interval_json(a, b, f"defined everywhere on the sampled window [{a:.6g}, {b:.6g}]",
                                         True, True) | {"text": "≈ ℝ (no gaps found when sampling)"}
        runs, start = [], None
        for i, good in enumerate(ok):
            if good and start is None:
                start = xs[i]
            if (not good or i == len(ok) - 1) and start is not None:
                runs.append([start, xs[i] if good else xs[i - 1]])
                start = None
        ivs = [interval_json(float(lo), float(hi), False, False) for lo, hi in runs[:20]]
        return {"method": "numeric_estimate", "exact": None, "intervals": ivs,
                "text": "≈ " + " ∪ ".join(iv["text"] for iv in ivs),
                "note": f"sampled on [{a:.6g}, {b:.6g}]"}

    def _numeric_range(self, a, b):
        xs = np.linspace(a, b, 8001)
        ys = self.fn(xs)
        ys = ys[np.isfinite(ys)]
        if ys.size == 0:
            return {"method": "numeric_estimate", "exact": None, "text": "unknown"}
        lo, hi = float(ys.min()), float(ys.max())
        lo_unb = lo < -1e6 or any(s.get("-") is NEG_OO or s.get("+") is NEG_OO for _, s in self.vert)
        hi_unb = hi > 1e6 or any(s.get("-") is sp.oo or s.get("+") is sp.oo for _, s in self.vert)
        return numeric_interval_json(lo, hi, f"sampled on x ∈ [{a:.6g}, {b:.6g}]", lo_unb, hi_unb)

    def _numeric_vertical(self, a, b) -> list[float]:
        xs = np.linspace(a, b, 8001)
        ys = self.fn(xs)
        out = []
        finite = ys[np.isfinite(ys)]
        if finite.size == 0:
            return out
        thresh = 50 * (np.median(np.abs(finite)) + 1)
        for i in range(len(xs) - 1):
            y1, y2 = ys[i], ys[i + 1]
            if not (np.isfinite(y1) and np.isfinite(y2)):
                continue
            if max(abs(y1), abs(y2)) > thresh and (np.sign(y1) != np.sign(y2) or abs(y2 - y1) > thresh):
                lo, hi = xs[i], xs[i + 1]
                for _ in range(50):
                    m = 0.5 * (lo + hi)
                    ym = self.fn(np.array([m]))[0]
                    if np.isfinite(ym) and np.sign(ym) == np.sign(y1) and abs(ym) >= abs(y1) * 0.5:
                        lo = m
                    else:
                        hi = m
                m = 0.5 * (lo + hi)
                probe = np.abs(self.fn(np.array([m - 1e-9, m + 1e-9])))
                if np.nanmax(probe) > 1e6:
                    out.append(float(m))
        return out

    def _monotonic(self, o, a, b, vert_x):
        if not self.f.has(x):
            o["increasing_intervals"], o["decreasing_intervals"] = [], []
            o["constant_intervals"] = [interval_json(NEG_OO, sp.oo)]
            return
        ivs = set_intervals(self.dom) if self.dom is not None else None
        exact_stat = self.stat.kind == "finite"
        scope = "global"
        if ivs is None or not exact_stat or self.sing_periodic is not None:
            scope = "viewport"
            ivs = [(a, b, True, True)]
        ivs = [iv for iv in ivs if iv[0] != iv[1]]
        if scope == "global":
            splits = list(self.stat.vals)
        else:
            splits = list(self._raw.get("stationary_points", [])) + list(vert_x)
        inc, dec, const = [], [], []
        for lo, hi, _, _ in ivs:
            flo = fnum(lo) if lo is not NEG_OO else -np.inf
            fhi = fnum(hi) if hi is not sp.oo else np.inf
            inner = sorted({s for s in splits if flo < (fnum(s) if fnum(s) is not None else np.nan) < fhi},
                           key=lambda s: fnum(s))
            edges = [lo, *inner, hi]
            segs = []
            for p, q in zip(edges, edges[1:]):
                fp = -np.inf if p is NEG_OO else fnum(p)
                fq = np.inf if q is sp.oo else fnum(q)
                if np.isinf(fp) and np.isinf(fq):
                    x0 = 0.0
                elif np.isinf(fp):
                    x0 = fq - 1
                elif np.isinf(fq):
                    x0 = fp + 1
                else:
                    x0 = 0.5 * (fp + fq)
                segs.append([p, q, sign_at(self.f1n, x0)])
            merged = []
            for seg in segs:
                p = seg[0]
                mergeable = merged and merged[-1][2] == seg[2] and not any(
                    abs(fnum(p) - fnum(v)) < 1e-12 for v in vert_x if fnum(p) is not None)
                if mergeable:
                    merged[-1][1] = seg[1]
                else:
                    merged.append(seg)
            for p, q, s in merged:
                (inc if s > 0 else dec if s < 0 else const).append(interval_json(p, q))
        o["increasing_intervals"], o["decreasing_intervals"] = inc, dec
        if const:
            o["constant_intervals"] = const
        o["monotonicity_scope"] = scope
        self.ctx.methods["monotonicity"] = "exact" if scope == "global" else "viewport"

    def _details(self):
        f = self.f
        if not f.is_polynomial(x) or not f.has(x):
            return None
        poly = sp.Poly(f, x)
        if poly.degree() == 1:
            m, c = poly.all_coeffs()
            return {"slope": num(m), "y_intercept": num(c)}
        if poly.degree() == 2:
            A, B, C = poly.all_coeffs()
            h = -B / (2 * A)
            k = sp.simplify(f.subs(x, h))
            p = 1 / (4 * A)
            return {"vertex": point(h, k), "axis_of_symmetry": f"x = {pretty(h)}",
                    "discriminant": num(B ** 2 - 4 * A * C),
                    "opens": "up" if A > 0 else "down",
                    "focus": point(h, sp.simplify(k + p)),
                    "directrix": f"y = {pretty(sp.simplify(k - p))}",
                    "vertex_form": f"y = {sp.sstr(A * (x - h) ** 2 + k)}"}
        return {"degree": poly.degree(), "leading_coefficient": num(poly.LC())}
