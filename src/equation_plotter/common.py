"""Shared analysis plumbing: the per-equation context, solution-set handling, classification."""

from __future__ import annotations

import math

import sympy as sp

from .errors import PlotterError
from .numfmt import fnum, general_text  # noqa: F401 (re-export)
from .sandbox import Budget, StepTimeout

MAX_LISTED = 40


class Ctx:
    """Per-equation analysis context: budget, warnings, and which method produced each field."""

    def __init__(self, index: int, budget: Budget, warnings: list, symbolic: bool = True):
        self.index = index
        self.budget = budget
        self.warnings = warnings
        self.symbolic = symbolic
        self.methods: dict[str, str] = {}
        self._warned: set = set()

    def warn(self, field: str, code: str, message: str):
        key = (field, code)
        if key in self._warned:
            return
        self._warned.add(key)
        self.warnings.append({"equation": self.index, "field": field, "code": code,
                              "message": message})

    def sym(self, field: str, fn, limit: float | None = None, quiet: bool = False):
        """Run a symbolic step under a deadline. Returns (ok, value)."""
        if not self.symbolic:
            return False, None
        if self.budget.exhausted():
            if not quiet:
                self.warn(field, "budget_exhausted",
                          f"{field}: symbolic time budget used up; numeric estimate used where possible")
            return False, None
        seconds = 0.0
        try:
            with self.budget.step(limit) as seconds:
                return True, fn()
        except StepTimeout:
            if not quiet:
                self.warn(field, "timeout",
                          f"{field}: symbolic computation timed out after {seconds:.1f}s; "
                          "numeric estimate used where possible")
        except PlotterError:
            raise
        except RecursionError:
            if not quiet:
                self.warn(field, "symbolic_failed", f"{field}: expression too deep for symbolic work")
        except Exception as e:  # noqa: BLE001 - SymPy raises many types for "can't do this"
            if not quiet:
                self.warn(field, "symbolic_failed",
                          f"{field}: symbolic computation not possible ({type(e).__name__}); "
                          "numeric estimate used where possible")
        return False, None


# --------------------------------------------------------------------------- solution sets

def classify_solutions(s):
    """('finite', [vals]) | ('periodic', set) | ('all', set) | ('unknown', None)."""
    if s is None:
        return "unknown", None
    if s is sp.S.EmptySet:
        return "finite", []
    if isinstance(s, sp.FiniteSet):
        vals = [v for v in s.args if v.is_real is not False]
        return "finite", sorted(vals, key=lambda v: fnum(v) if fnum(v) is not None else 0)
    if isinstance(s, (sp.ConditionSet, sp.Complement, sp.Intersection)):
        return "unknown", None
    if isinstance(s, (sp.Interval,)) or (isinstance(s, sp.Union) and any(
            isinstance(a, sp.Interval) for a in s.args)):
        return "all", s
    if isinstance(s, sp.ImageSet) or (isinstance(s, sp.Union) and all(
            isinstance(a, (sp.ImageSet, sp.FiniteSet)) for a in s.args)):
        return "periodic", s
    return "unknown", None


def realise(s, a: float, b: float, limit: int = 200):
    """Concrete members of a FiniteSet/ImageSet union inside [a, b]; None if unsupported."""
    parts = s.args if isinstance(s, sp.Union) else (s,)
    out = []
    for part in parts:
        if isinstance(part, sp.FiniteSet):
            out.extend(v for v in part.args if (fv := fnum(v)) is not None and a <= fv <= b)
            continue
        if not isinstance(part, sp.ImageSet) or len(part.lamda.variables) != 1 \
                or part.base_sets[0] is not sp.S.Integers:
            return None
        n = part.lamda.variables[0]
        e = part.lamda.expr
        v0, v1 = fnum(e.subs(n, 0)), fnum(e.subs(n, 1))
        if v0 is None or v1 is None:
            return None
        d = v1 - v0
        if d == 0:
            ks = [0]
        else:
            k_lo, k_hi = sorted(((a - v0) / d, (b - v0) / d))
            k_lo, k_hi = math.floor(k_lo) - 1, math.ceil(k_hi) + 1
            if k_hi - k_lo > limit:
                return None
            ks = range(k_lo, k_hi + 1)
        for k in ks:
            v = e.subs(n, k)
            fv = fnum(v)
            if fv is not None and a - 1e-12 <= fv <= b + 1e-12:
                out.append(v)
    uniq = {}
    for v in out:
        uniq.setdefault(round(fnum(v), 10), v)
    return [uniq[k] for k in sorted(uniq)]


# --------------------------------------------------------------------------- classification

TRIG = (sp.sin, sp.cos, sp.tan, sp.sec, sp.csc, sp.cot)
INV_TRIG = (sp.asin, sp.acos, sp.atan)
HYP = (sp.sinh, sp.cosh, sp.tanh, sp.sech, sp.csch, sp.coth, sp.asinh, sp.acosh, sp.atanh)
POLY_NAMES = {0: "constant", 1: "linear", 2: "quadratic", 3: "cubic", 4: "quartic", 5: "quintic"}


def families(expr, var) -> list[str]:
    tags = []
    if expr.has(*TRIG):
        tags.append("trigonometric")
    if expr.has(*INV_TRIG):
        tags.append("inverse trigonometric")
    if expr.has(*HYP):
        tags.append("hyperbolic")
    if expr.has(sp.exp) or any(p.exp.has(var) for p in expr.atoms(sp.Pow)):
        tags.append("exponential")
    if expr.has(sp.log):
        tags.append("logarithmic")
    if expr.has(sp.Abs):
        tags.append("absolute value")
    if expr.has(sp.floor, sp.ceiling):
        tags.append("step")
    if expr.has(sp.sign) and not expr.has(sp.Abs):
        tags.append("piecewise")
    if any(p.base.has(var) and p.exp.is_Rational and not p.exp.is_Integer
           for p in expr.atoms(sp.Pow)):
        tags.append("radical")
    if expr.has(sp.sign) and "absolute value" in tags and "radical" in tags:
        tags = [t for t in tags if t != "absolute value"]  # real cube root: sign(x)*|x|^(1/3)
    return tags


def classify_function(f, var) -> tuple[str, list[str]]:
    if f.is_polynomial(var):
        deg = sp.degree(f, var) if f.has(var) else 0
        name = POLY_NAMES.get(deg, f"polynomial (degree {deg})")
        if deg >= 3:
            name = f"{name} (polynomial, degree {deg})"
        return name, ["polynomial"]
    tags = families(f, var)
    if f.is_rational_function(var):
        return "rational", ["rational"]
    if not tags:
        return "algebraic", []
    if len(tags) == 1:
        return tags[0], tags
    return "composite (" + ", ".join(tags) + ")", tags


def tex(e) -> str:
    return sp.latex(e, ln_notation=True)


def strip_steps(f):
    """Replace floor/ceiling/sign by their arguments for domain purposes (they're total)."""
    return f.replace(sp.floor, lambda a: a).replace(sp.ceiling, lambda a: a) \
        .replace(sp.sign, lambda a: sp.Integer(1))


def tidy(expr, ctx: Ctx, limit: float = 1.0):
    ok, v = ctx.sym("simplify", lambda: sp.simplify(expr), limit=limit, quiet=True)
    if ok and sp.count_ops(v) <= sp.count_ops(expr):
        return v
    return expr


def lcm_period(p1, p2):
    if p1 is None or p2 is None:
        return None
    ratio = sp.nsimplify(p1 / p2)
    if not ratio.is_Rational:
        return None
    return sp.simplify(p1 * ratio.q)
