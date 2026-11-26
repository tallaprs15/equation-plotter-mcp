"""Analysis of implicit curves F(x, y) = 0: lines, conics, and general curves."""

from __future__ import annotations

import numpy as np
import sympy as sp

from .common import MAX_LISTED, Ctx, tex, classify_solutions, families, general_text, realise
from .numeric import find_roots, implicit_lines, lambdify
from .numfmt import fnum, num, numeric_interval_json, point, pretty, set_json
from .parser import x, y


def _sym_eq(F, G, ctx: Ctx, field: str) -> bool:
    ok, z = ctx.sym(field, lambda: sp.expand(F - G) == 0 or sp.simplify(F - G) == 0,
                    limit=1.0, quiet=True)
    return bool(ok and z)


def solve_for_extent(F, var, other, ctx: Ctx, field: str):
    """Set of ``other`` values for which F(var, other) = 0 has a real solution in ``var``.

    Exact when F is polynomial of degree <= 2 in ``var``; None otherwise.
    """
    def work():
        P = sp.Poly(F, var)
        deg = P.degree()
        if deg == 2:
            a2, a1, a0 = P.all_coeffs()
            disc = sp.expand(a1 ** 2 - 4 * a2 * a0)
            if a2.has(other):
                return None
            if a2.is_number and a2 != 0:
                if not disc.has(other):
                    return sp.S.Reals if disc >= 0 else sp.S.EmptySet
                return sp.solve_univariate_inequality(disc >= 0, other, relational=False)
            return None
        if deg == 1:
            a1, a0 = P.all_coeffs()
            if not a1.has(other):
                return sp.S.Reals
            bad = sp.solveset(a1, other, sp.S.Reals)
            if isinstance(bad, sp.FiniteSet):
                keep = [b for b in bad.args if sp.simplify(a0.subs(other, b)) == 0]
                drop = [b for b in bad.args if b not in keep]
                return sp.Complement(sp.S.Reals, sp.FiniteSet(*drop)) if drop else sp.S.Reals
            return None
        if deg == 0:
            return sp.solveset(F, other, sp.S.Reals)
        return None
    ok, s = ctx.sym(field, work, limit=2.0)
    if ok and s is not None and isinstance(s, sp.Complement):
        s = s.simplify() if hasattr(s, "simplify") else s
    return s if ok else None


def conic_info(F, ctx: Ctx):
    """Classify a degree-2 polynomial F(x, y) and compute centre/axes/foci where cheap."""
    P = sp.Poly(sp.expand(F), x, y)
    A, B, C = P.coeff_monomial(x ** 2), P.coeff_monomial(x * y), P.coeff_monomial(y ** 2)
    D, E, F0 = P.coeff_monomial(x), P.coeff_monomial(y), P.coeff_monomial(1)
    M = sp.Matrix([[A, B / 2, D / 2], [B / 2, C, E / 2], [D / 2, E / 2, F0]])
    det = sp.simplify(M.det())
    disc = sp.simplify(B ** 2 - 4 * A * C)
    info: dict = {"coefficients": {k: num(v) for k, v in
                                   zip(("A", "B", "C", "D", "E", "F"), (A, B, C, D, E, F0))},
                  "general_form": "A*x**2 + B*x*y + C*y**2 + D*x + E*y + F = 0",
                  "discriminant": num(disc)}
    geo: dict = {"centre": None, "foci": [], "vertices": [], "asymptote_lines": []}
    if det == 0:
        name = ("degenerate conic (two intersecting lines)" if disc > 0 else
                "degenerate conic (parallel lines, a line or no points)" if disc == 0 else
                "degenerate conic (single point)")
        return name, info, geo
    if disc != 0:
        h, k = sp.symbols("h k")
        sol = sp.solve([2 * A * h + B * k + D, B * h + 2 * C * k + E], [h, k], dict=True)[0]
        h, k = sp.simplify(sol[h]), sp.simplify(sol[k])
        Fc = sp.simplify(F.subs({x: h, y: k}))
        info["centre"] = point(h, k)
        geo["centre"] = (h, k)
        Q = sp.Matrix([[A, B / 2], [B / 2, C]])
        if disc < 0:
            if (Fc * A).is_positive:
                return "ellipse (no real points)", info, geo
            if B == 0 and A == C:
                rad = sp.sqrt(sp.simplify(-Fc / A))
                info.update({"radius": num(rad),
                             "standard_form": f"{sp.sstr((x - h) ** 2 + (y - k) ** 2)} = {sp.sstr(sp.simplify(rad ** 2))}"})
                return "circle", info, geo
        eig = []
        for val, mult, vecs in Q.eigenvects():
            for v in vecs:
                v = sp.simplify(v / v.norm())
                eig.append((sp.simplify(val), v))
        if len(eig) != 2:
            return ("ellipse" if disc < 0 else "hyperbola"), info, geo
        if B != 0:
            info["rotated"] = True
        if disc < 0:
            axes = sorted(((sp.sqrt(sp.simplify(-Fc / lam)), v) for lam, v in eig),
                          key=lambda av: -fnum(av[0]))
            (a, ua), (b, _) = axes
            c = sp.sqrt(sp.simplify(a ** 2 - b ** 2))
            info.update({"semi_major_axis": num(a), "semi_minor_axis": num(b),
                         "eccentricity": num(sp.simplify(c / a)),
                         "major_axis_direction": [num(ua[0]), num(ua[1])]})
            if B == 0:
                ax2, by2 = sp.simplify(-Fc / A), sp.simplify(-Fc / C)
                info["standard_form"] = f"{sp.sstr((x - h) ** 2 / ax2 + (y - k) ** 2 / by2)} = 1"
            name = "ellipse"
        else:
            trans = [(lam, v) for lam, v in eig if (-Fc / lam).is_positive]
            conj = [(lam, v) for lam, v in eig if (Fc / lam).is_positive]
            if not trans or not conj:
                return "hyperbola", info, geo
            (lt, ua), (lc, ub) = trans[0], conj[0]
            a, b = sp.sqrt(sp.simplify(-Fc / lt)), sp.sqrt(sp.simplify(Fc / lc))
            c = sp.sqrt(sp.simplify(a ** 2 + b ** 2))
            asy = []
            for sgn in (1, -1):
                d = a * ua + sgn * b * ub
                if sp.simplify(d[0]) == 0:
                    asy.append((None, h))
                else:
                    m = sp.simplify(d[1] / d[0])
                    asy.append((m, sp.simplify(k - m * h)))
            geo["asymptote_lines"] = asy
            info.update({"semi_transverse_axis": num(a), "semi_conjugate_axis": num(b),
                         "eccentricity": num(sp.simplify(c / a)),
                         "transverse_axis_direction": [num(ua[0]), num(ua[1])],
                         "asymptotes": [f"x = {pretty(q)}" if m is None else
                                        f"y = {sp.sstr(m * x + q)}" for m, q in asy]})
            if B == 0:
                ax2, by2 = sp.simplify(-Fc / A), sp.simplify(-Fc / C)
                info["standard_form"] = f"{sp.sstr((x - h) ** 2 / ax2 + (y - k) ** 2 / by2)} = 1"
            name = "hyperbola"
        foci = [(sp.simplify(h + s * c * ua[0]), sp.simplify(k + s * c * ua[1])) for s in (1, -1)]
        verts = [(sp.simplify(h + s * a * ua[0]), sp.simplify(k + s * a * ua[1])) for s in (1, -1)]
        info["foci"] = [point(*p) for p in foci]
        info["vertices"] = [point(*p) for p in verts]
        geo["foci"], geo["vertices"] = foci, verts
        return name, info, geo
    # parabola
    if B != 0:
        info["rotated"] = True
        return "parabola", info, geo
    if C == 0:
        h = -D / (2 * A)
        k = sp.simplify(-(A * h ** 2 + D * h + F0) / E)
        p = sp.simplify(-E / (4 * A))
        info.update({"vertex": point(h, k), "focus": point(h, k + p),
                     "directrix": f"y = {pretty(sp.simplify(k - p))}", "axis": f"x = {pretty(h)}",
                     "focal_length": num(sp.Abs(p)), "opens": "up" if p > 0 else "down",
                     "standard_form": f"{sp.sstr((x - h) ** 2)} = {sp.sstr(sp.simplify(4 * p) * (y - k))}"})
        geo["vertices"], geo["foci"] = [(h, k)], [(h, sp.simplify(k + p))]
    else:
        k = -E / (2 * C)
        h = sp.simplify(-(C * k ** 2 + E * k + F0) / D)
        p = sp.simplify(-D / (4 * C))
        info.update({"vertex": point(h, k), "focus": point(h + p, k),
                     "directrix": f"x = {pretty(sp.simplify(h - p))}", "axis": f"y = {pretty(k)}",
                     "focal_length": num(sp.Abs(p)), "opens": "right" if p > 0 else "left",
                     "standard_form": f"{sp.sstr((y - k) ** 2)} = {sp.sstr(sp.simplify(4 * p) * (x - h))}"})
        geo["vertices"], geo["foci"] = [(h, k)], [(sp.simplify(h + p), k)]
    return "parabola", info, geo


class ImplicitAnalysis:
    kind = "implicit"

    def __init__(self, eq, ctx: Ctx, search):
        self.eq = eq
        self.F = eq.F
        self.ctx = ctx
        self.search = search
        self.Fn = lambdify(self.F, (x, y))
        self.out: dict = {}
        self.geo = {"centre": None, "foci": [], "vertices": [], "asymptote_lines": []}

    def run(self):
        F, ctx, o, eq = self.F, self.ctx, self.out, self.eq
        o["normalised"] = f"{sp.sstr(eq.lhs)} = {sp.sstr(eq.rhs)}"
        o["latex"] = f"{tex(eq.lhs)} = {tex(eq.rhs)}"
        ok, Fe = ctx.sym("canonical", lambda: sp.expand(F) if F.is_polynomial(x, y)
                         else sp.simplify(F), limit=1.5, quiet=True)
        o["canonical"] = f"{sp.sstr(Fe if ok else F)} = 0"
        self.poly_deg = None
        if F.is_polynomial(x, y):
            ok, P = ctx.sym("classification", lambda: sp.Poly(F, x, y), limit=1.0)
            if ok:
                self.poly_deg = P.total_degree()
        self.cls = "implicit curve"
        self.details = None
        if self.poly_deg == 1:
            self._line()
        elif self.poly_deg == 2:
            ok, res = ctx.sym("classification", lambda: conic_info(F, ctx), limit=3.0)
            if ok:
                self.cls, self.details, self.geo = res
                self.cls = self.cls if self.cls != "circle" else "circle"
            else:
                self.cls = "conic"
        elif self.poly_deg is not None:
            self.cls = f"algebraic curve (degree {self.poly_deg})"
        else:
            tags = families(F, x) or families(F, y)
            self.cls = "implicit curve" + (f" ({', '.join(tags)})" if tags else "")
        o["classification"] = self.cls
        if self.cls in ("circle", "ellipse", "hyperbola", "parabola") or self.poly_deg == 1:
            o["families"] = ["conic"] if self.poly_deg == 2 else ["linear"]

        self.xint = self._solve(F.subs(y, 0), x, "x_intercepts")
        self.yint = self._solve(F.subs(x, 0), y, "y_intercepts")
        self.dom = solve_for_extent(F, y, x, ctx, "domain") if self.poly_deg else None
        self.rng = solve_for_extent(F, x, y, ctx, "range") if self.poly_deg else None
        self.htan, self.vtan = self._tangents()

    def _line(self):
        P = sp.Poly(self.F, x, y)
        a, b, c = P.coeff_monomial(x), P.coeff_monomial(y), P.coeff_monomial(1)
        self.cls = "linear"
        if b == 0:
            self.details = {"vertical": True, "equation": f"x = {pretty(-c / a)}"}
        else:
            m = sp.simplify(-a / b)
            q = sp.simplify(-c / b)
            self.details = {"vertical": False, "slope": num(m), "y_intercept": num(q),
                            "slope_intercept_form": f"y = {sp.sstr(m * x + q)}"}

    def _solve(self, expr, var, field):
        if not expr.has(var):
            return ("finite", []) if expr != 0 else ("all", None)
        ok, s = self.ctx.sym(field, lambda: sp.solveset(expr, var, sp.S.Reals))
        kind, vals = classify_solutions(s if ok else None)
        if kind == "finite" and len(vals) > MAX_LISTED:
            vals = vals[:MAX_LISTED]
        return kind, vals

    def _tangents(self):
        F, ctx = self.F, self.ctx
        Fx, Fy = sp.diff(F, x), sp.diff(F, y)
        res = []
        for field, G in (("stationary_points", Fx), ("vertical_tangents", Fy)):
            if G == 0 or (self.poly_deg is None and not ctx.symbolic):
                res.append(None)
                continue
            ok, sols = ctx.sym(field, lambda: sp.solve([F, G], [x, y], dict=True), limit=2.5)
            if not ok:
                res.append(None)
                continue
            pts = []
            for s in sols:
                if x not in s or y not in s:
                    continue
                px, py = s[x], s[y]
                if fnum(px) is None or fnum(py) is None:
                    continue
                pts.append((px, py))
            res.append(pts[:MAX_LISTED])
        self.Fx, self.Fy = Fx, Fy
        return res

    def _bbox_numeric(self, xr, yr):
        lines = implicit_lines(self.Fn, xr, yr, n=300)
        if not lines:
            return None, lines
        pts = np.concatenate(lines)
        return (float(pts[:, 0].min()), float(pts[:, 0].max()),
                float(pts[:, 1].min()), float(pts[:, 1].max())), lines

    def interest(self) -> dict:
        pts = []
        for kind, vals, mk in ((self.xint[0], self.xint[1], lambda v: (fnum(v), 0.0)),
                               (self.yint[0], self.yint[1], lambda v: (0.0, fnum(v)))):
            if kind == "finite":
                pts += [mk(v) for v in vals]
        for group in (self.htan, self.vtan):
            pts += [(fnum(a), fnum(b)) for a, b in group or []]
        for key in ("foci", "vertices"):
            pts += [(fnum(a), fnum(b)) for a, b in self.geo.get(key, [])]
        if self.geo.get("centre"):
            pts.append(tuple(fnum(v) for v in self.geo["centre"]))
        bbox = None
        if self.cls in ("circle", "ellipse") and self.dom is not None and self.rng is not None:
            try:
                bbox = (fnum(self.dom.inf), fnum(self.dom.sup), fnum(self.rng.inf), fnum(self.rng.sup))
            except Exception:  # noqa: BLE001
                bbox = None
        if not pts and bbox is None:
            a, b = self.search
            bbox, _ = self._bbox_numeric((a, b), (a, b))
        return {"points": [p for p in pts if None not in p], "vlines": [], "hlines": [],
                "period": None, "bbox": bbox, "equal_aspect": self.poly_deg == 2,
                "implicit": self.Fn}

    def finalise(self, vp):
        o, ctx = self.out, self.ctx
        (a, b), (ya, yb) = vp["x"], vp["y"]
        plot_pts = []
        lines_cache = None

        def numeric_lines():
            nonlocal lines_cache
            if lines_cache is None:
                lines_cache = implicit_lines(self.Fn, (a, b), (ya, yb), n=500)
            return lines_cache

        for key, (kind, vals), mk, lo, hi, pname in (
                ("x_intercepts", self.xint, lambda v: point(v, sp.Integer(0)), a, b, "x-intercept"),
                ("y_intercepts", self.yint, lambda v: point(sp.Integer(0), v), ya, yb, "y-intercept")):
            if kind == "finite":
                o[key] = [mk(v) for v in vals]
                ctx.methods[key] = "exact"
            elif kind == "periodic":
                o[key] = [mk(v) for v in (realise(vals, lo, hi) or [])]
                o.setdefault("general_solutions", {})[key] = general_text(vals)
                ctx.methods[key] = "exact_in_viewport"
            elif kind == "all":
                o[key] = []
                o.setdefault("notes", []).append(f"{key}: the curve lies along the axis")
                ctx.methods[key] = "exact"
            else:
                o[key] = self._numeric_axis_cross(numeric_lines(), key)
                ctx.methods[key] = "numeric_in_viewport"
            plot_pts += [(p, pname) for p in o[key]]

        if self.dom is not None:
            o["domain"] = set_json(self.dom)
            ctx.methods["domain"] = "exact"
        if self.rng is not None:
            o["range"] = set_json(self.rng)
            ctx.methods["range"] = "exact"
        if self.dom is None or self.rng is None:
            ls = numeric_lines()
            if ls:
                P = np.concatenate(ls)
                note = "extent of the curve within the viewport"
                ex = (P[:, 0].min() <= a + 1e-6 * (b - a), P[:, 0].max() >= b - 1e-6 * (b - a),
                      P[:, 1].min() <= ya + 1e-6 * (yb - ya), P[:, 1].max() >= yb - 1e-6 * (yb - ya))
                if self.dom is None:
                    o["domain"] = numeric_interval_json(float(P[:, 0].min()), float(P[:, 0].max()),
                                                        note, ex[0], ex[1])
                    ctx.methods["domain"] = "numeric_estimate"
                if self.rng is None:
                    o["range"] = numeric_interval_json(float(P[:, 1].min()), float(P[:, 1].max()),
                                                       note, ex[2], ex[3])
                    ctx.methods["range"] = "numeric_estimate"

        stats = []
        if self.htan is not None:
            Fxx = sp.diff(self.F, x, 2)
            for px, py in self.htan:
                fy = fnum(self.Fy.subs({x: px, y: py}))
                fxx = fnum(Fxx.subs({x: px, y: py}))
                if fy is None or abs(fy) < 1e-12:
                    nature = "singular point"
                elif fxx is None or abs(fxx) < 1e-12:
                    nature = "horizontal tangent"
                else:
                    ypp = -fxx / fy
                    nature = "local maximum" if ypp < 0 else "local minimum"
                stats.append(point(px, py, nature=nature))
            ctx.methods["stationary_points"] = "exact"
        else:
            ctx.methods["stationary_points"] = "unavailable"
        o["stationary_points"] = stats
        plot_pts += [(p, "turning point") for p in stats]
        o["vertical_tangents"] = [point(px, py) for px, py in (self.vtan or [])]
        ctx.methods["vertical_tangents"] = "exact" if self.vtan is not None else "unavailable"

        o["symmetry"] = {"axes": self._symmetry(numeric_lines)}
        if self.details:
            o["details"] = self.details
        asy = self.geo.get("asymptote_lines") or []
        if asy:
            o["asymptotes"] = {"oblique_or_axis_aligned": [
                f"x = {pretty(q)}" if m is None else f"y = {sp.sstr(m * x + q)}" for m, q in asy]}
        for key, label in (("foci", "focus"), ("vertices", "vertex")):
            for p in self.geo.get(key, []):
                if label == "vertex" and self.cls in ("circle",):
                    continue
                plot_pts.append((point(*p), label))
        if self.geo.get("centre") and self.cls in ("circle", "ellipse", "hyperbola"):
            plot_pts.append((point(*self.geo["centre"]), "centre"))

        plot = {
            "kind": "implicit", "expr": self.F, "label": o["latex"], "plain": o["normalised"],
            "lines_xy": [(fnum(m), fnum(q)) if m is not None else (None, fnum(q)) for m, q in asy],
            "points": _plot_points(plot_pts),
        }
        return o, plot

    def _numeric_axis_cross(self, lines, key):
        out = []
        idx = 1 if key == "x_intercepts" else 0
        for L in lines:
            v = L[:, idx]
            s = np.sign(v)
            for i in np.where(s[:-1] * s[1:] < 0)[0]:
                t = v[i] / (v[i] - v[i + 1])
                p = L[i] + t * (L[i + 1] - L[i])
                est = float(p[1 - idx])
                d = 2 * float(np.hypot(*(L[i + 1] - L[i]))) + 1e-9
                g = (lambda s: self.Fn(s, 0 * s)) if idx == 1 else (lambda s: self.Fn(0 * s, s))
                ref = find_roots(g, est - d, est + d, n=41)
                out.append(min(ref, key=lambda r: abs(r - est)) if ref else est)
        out = sorted(set(round(v, 9) for v in out))[:MAX_LISTED]
        return [point(v, 0.0) if key == "x_intercepts" else point(0.0, v) for v in out]

    def _symmetry(self, numeric_lines):
        F = self.F
        tests = {"y-axis": F.subs(x, -x), "x-axis": F.subs(y, -y),
                 "origin": F.subs({x: -x, y: -y}, simultaneous=True),
                 "y = x": F.subs({x: y, y: x}, simultaneous=True)}
        rng = np.random.default_rng(11)
        X, Y = rng.uniform(-7, 7, 30), rng.uniform(-7, 7, 30)
        base = self.Fn(X, Y)
        found, method = [], "exact"
        for name, G in tests.items():
            Gn = lambdify(G, (x, y))(X, Y)
            mask = np.isfinite(base) & np.isfinite(Gn)
            if mask.sum() < 5:
                continue
            tol = 1e-7 * (1 + np.abs(base[mask]))
            plus = np.all(np.abs(Gn[mask] - base[mask]) < tol)
            minus = np.all(np.abs(Gn[mask] + base[mask]) < tol)
            if plus or minus:
                if not (_sym_eq(G, F, self.ctx, "symmetry") or _sym_eq(G, -F, self.ctx, "symmetry")):
                    method = "numeric"
                found.append(name)
        self.ctx.methods["symmetry"] = method
        return found


def _plot_points(plot_pts):
    out = []
    for p, kind in plot_pts:
        px, py = p["x"]["float"], p["y"]["float"]
        if px is not None and py is not None:
            out.append((px, py, kind))
    return out
