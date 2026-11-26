"""Hostile-input-safe equation parser.

No ``eval``/``exec``/``sympify``/``parse_expr`` is ever applied to user text. The input is
normalised, checked against a character whitelist, tokenised with a fixed regex, and parsed
by a small recursive-descent parser that only ever constructs SymPy objects from a fixed
table of functions, constants and variables. Limits bound length, token count, nesting
depth, literal size and exponent size so a single request cannot burn unbounded CPU.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import sympy as sp

from .errors import PlotterError
from .latex_input import latex_to_plain, looks_like_latex

MAX_INPUT_LEN = 300
MAX_TOKENS = 250
MAX_DEPTH = 40
MAX_LITERAL_DIGITS = 30
MAX_EXPONENT = 1000
MAX_RESULT_DIGITS = 1000

x, y, t, r, theta = sp.symbols("x y t r theta", real=True)

VARIABLES = {"x": x, "y": y, "t": t, "r": r, "theta": theta}
CONSTANTS = {"pi": sp.pi, "e": sp.E}


def _log(a, base=None):
    return sp.log(a) / sp.log(10) if base is None else sp.log(a, base)


FUNCTIONS = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "sec": sp.sec, "csc": sp.csc,
    "cosec": sp.csc, "cot": sp.cot,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
    "arcsin": sp.asin, "arccos": sp.acos, "arctan": sp.atan,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh, "sech": sp.sech, "csch": sp.csch,
    "coth": sp.coth, "asinh": sp.asinh, "acosh": sp.acosh, "atanh": sp.atanh,
    "arsinh": sp.asinh, "arcosh": sp.acosh, "artanh": sp.atanh,
    "exp": sp.exp, "ln": sp.log, "log": _log,
    "log10": lambda a: sp.log(a, 10), "log2": lambda a: sp.log(a, 2),
    "sqrt": sp.sqrt, "cbrt": lambda a: sp.real_root(a, 3),
    "abs": sp.Abs, "sgn": sp.sign, "sign": sp.sign, "floor": sp.floor,
    "ceil": sp.ceiling, "ceiling": sp.ceiling,
}
TWO_ARG = {"log"}
INVERSE = {"sin": sp.asin, "cos": sp.acos, "tan": sp.atan, "sinh": sp.asinh,
           "cosh": sp.acosh, "tanh": sp.atanh}

_KNOWN_NAMES = sorted([*VARIABLES, *CONSTANTS, *FUNCTIONS], key=len, reverse=True)

_UNICODE = {
    "−": "-", "–": "-", "×": "*", "·": "*", "⋅": "*", "÷": "/",
    "²": "^2", "³": "^3", "√": " sqrt ", "≤": "<=", "≥": ">=",
    "π": " pi ", "θ": " theta ", "′": "'",
}
_ALLOWED = re.compile(r"[A-Za-z0-9\s+\-*/^().,=<>|\[\]]*")
_TOKEN = re.compile(
    r"\s*(?:(?P<num>\d+\.\d*|\.\d+|\d+)|(?P<name>[A-Za-z][A-Za-z0-9]*)"
    r"|(?P<op><=|>=|\^|[-+*/(),=<>|\[\]]))")
_FUNC_LHS = re.compile(r"^\s*([fgh])\s*\(\s*x\s*\)\s*=(?!=)")
_RELATIONS = {"=", "<", ">", "<=", ">="}


@dataclass
class Tok:
    kind: str          # num | var | const | func | op
    value: object
    pos: int


@dataclass
class ParsedEquation:
    """Result of parsing one input string. ``kind`` drives the analysis pipeline."""
    input: str
    kind: str                       # explicit | implicit | parametric | polar | inequality
    name: str = "y"                 # f/g/h for f(x)= notation
    f: sp.Expr | None = None        # explicit: y = f(x)
    lhs: sp.Expr | None = None
    rhs: sp.Expr | None = None
    F: sp.Expr | None = None        # implicit / inequality: F(x, y) (rel) 0, F = lhs - rhs
    relation: str = "="
    boundary_kind: str | None = None  # inequality: explicit | implicit
    xt: sp.Expr | None = None       # parametric
    yt: sp.Expr | None = None
    r_exprs: list = field(default_factory=list)  # polar branches r(theta)
    r_squared: bool = False
    rhs_polar: sp.Expr | None = None


# --------------------------------------------------------------------------- input hygiene

def normalise_text(text: str) -> str:
    if not isinstance(text, str):
        raise PlotterError("invalid_input", "Equation must be a string")
    if len(text) > MAX_INPUT_LEN:
        raise PlotterError("input_too_long",
                           f"Equation is {len(text)} characters; the limit is {MAX_INPUT_LEN}")
    s = text
    for k, v in _UNICODE.items():
        s = s.replace(k, v)
    if looks_like_latex(s):
        s = latex_to_plain(s)
    s = s.replace("**", "^")
    if not s.strip():
        raise PlotterError("empty_input", "Equation is empty")
    if len(s) > 4 * MAX_INPUT_LEN:
        raise PlotterError("input_too_long", "Equation expands to an excessively long form")
    m = _ALLOWED.fullmatch(s)
    if not m:
        bad = next(i for i, ch in enumerate(s) if not _ALLOWED.fullmatch(ch))
        raise PlotterError("invalid_character", f"Character {s[bad]!r} is not allowed",
                           position=bad)
    return s


def _split_word(word: str, pos: int) -> list[Tok]:
    """Split a run of letters (``xsinx``) into known names, longest match first."""
    toks: list[Tok] = []
    i = 0
    while i < len(word):
        for name in _KNOWN_NAMES:
            if word.startswith(name, i):
                break
        else:
            raise PlotterError(
                "unknown_name",
                f"Unknown name {word[i:]!r}. Use x, y, t, r, theta, pi, e and functions like "
                "sin, cos, tan, exp, ln, log, sqrt, abs", position=pos + i)
        if name in VARIABLES:
            toks.append(Tok("var", VARIABLES[name], pos + i))
        elif name in CONSTANTS:
            toks.append(Tok("const", CONSTANTS[name], pos + i))
        else:
            toks.append(Tok("func", name, pos + i))
        i += len(name)
    return toks


def tokenize(s: str) -> list[Tok]:
    toks: list[Tok] = []
    i = 0
    while i < len(s):
        if s[i].isspace():
            i += 1
            continue
        m = _TOKEN.match(s, i)
        if not m or m.end() == i:
            raise PlotterError("syntax_error", f"Unexpected {s[i]!r}", position=i)
        start = m.start(m.lastgroup)
        if m.group("num") is not None:
            lit = m.group("num")
            if len(lit) > MAX_LITERAL_DIGITS:
                raise PlotterError("number_too_large", "Numeric literal is too long",
                                   position=start)
            toks.append(Tok("num", sp.Rational(lit), start))
        elif m.group("name") is not None:
            word = m.group("name").lower()
            # Allow log10 / log2 as names; otherwise digits inside a word are implicit mult.
            parts = re.findall(r"log10|log2|[A-Za-z]+|\d+", word)
            off = start
            for part in parts:
                if part.isdigit():
                    toks.append(Tok("num", sp.Integer(part), off))
                elif part in ("log10", "log2"):
                    toks.append(Tok("func", part, off))
                else:
                    toks.extend(_split_word(part, off))
                off += len(part)
        else:
            toks.append(Tok("op", m.group("op"), start))
        i = m.end()
        if len(toks) > MAX_TOKENS:
            raise PlotterError("too_complex", f"Equation has more than {MAX_TOKENS} tokens")
    return toks


# --------------------------------------------------------------------------- expression parser

_OPEN = {"(": ")", "[": "]"}


def _check_number(v):
    if v.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
        raise PlotterError("undefined_value",
                           "Expression contains an undefined value (e.g. division by zero)")
    return v


def make_pow(base, exp):
    if exp.is_Number and abs(exp) > MAX_EXPONENT:
        raise PlotterError("number_too_large", f"Exponents are limited to ±{MAX_EXPONENT}")
    if base.is_Number and exp.is_Number and base != 0:
        mag = abs(float(exp)) * abs(math.log10(abs(float(base)))) if abs(base) != 1 else 0
        if mag > MAX_RESULT_DIGITS:
            raise PlotterError("number_too_large", "Numeric result would be too large")
    return sp.Pow(base, exp)


class _Parser:
    def __init__(self, toks: list[Tok]):
        self.toks = toks
        self.i = 0
        self.depth = 0
        self.abs_depth = 0

    # helpers
    def peek(self) -> Tok | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def is_op(self, *ops) -> bool:
        tk = self.peek()
        return tk is not None and tk.kind == "op" and tk.value in ops

    def take(self) -> Tok:
        tk = self.peek()
        if tk is None:
            raise PlotterError("syntax_error", "Unexpected end of equation")
        self.i += 1
        return tk

    def expect(self, op: str):
        tk = self.peek()
        if tk is None or tk.kind != "op" or tk.value != op:
            where = tk.pos if tk else None
            raise PlotterError("syntax_error", f"Expected {op!r}", position=where)
        self.i += 1

    def enter(self):
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise PlotterError("too_complex", "Equation is nested too deeply")

    def starts_operand(self) -> bool:
        tk = self.peek()
        if tk is None:
            return False
        if tk.kind in ("num", "var", "const", "func"):
            return True
        if tk.kind == "op" and tk.value in _OPEN:
            return True
        return tk.kind == "op" and tk.value == "|" and self.abs_depth == 0

    # grammar
    def expr(self):
        self.enter()
        left = self.term()
        while self.is_op("+", "-"):
            op = self.take().value
            right = self.term()
            left = left + right if op == "+" else left - right
        self.depth -= 1
        return left

    def term(self):
        left = self.unary()
        while True:
            if self.is_op("*", "/"):
                op = self.take().value
                right = self.unary()
                if op == "/" and right == 0:
                    raise PlotterError("undefined_value", "Division by zero")
                left = left * right if op == "*" else left / right
            elif self.starts_operand():
                prev = self.peek()
                if self.toks[self.i - 1].kind == "num" and prev.kind == "num":
                    raise PlotterError("syntax_error", "Two numbers in a row", position=prev.pos)
                left = left * self.power()
            else:
                return left

    def unary(self):
        if self.is_op("-"):
            self.take()
            self.enter()
            v = -self.unary()
            self.depth -= 1
            return v
        if self.is_op("+"):
            self.take()
            return self.unary()
        return self.power()

    def power(self):
        base = self.primary()
        if self.is_op("^"):
            self.take()
            self.enter()
            exp = self.unary()
            self.depth -= 1
            return make_pow(base, exp)
        return base

    def primary(self):
        tk = self.take()
        if tk.kind in ("num", "const", "var"):
            return tk.value
        if tk.kind == "func":
            return self.call(tk)
        if tk.kind == "op" and tk.value in _OPEN:
            v = self.expr()
            self.expect(_OPEN[tk.value])
            return v
        if tk.kind == "op" and tk.value == "|":
            self.abs_depth += 1
            v = self.expr()
            self.expect("|")
            self.abs_depth -= 1
            return sp.Abs(v)
        raise PlotterError("syntax_error", f"Unexpected {tk.value!r}", position=tk.pos)

    def call(self, tk: Tok):
        name = tk.value
        fn = FUNCTIONS[name]
        post_power = None
        if self.is_op("^"):                       # sin^2(x), sin^-1(x)
            self.take()
            p = self.unary()
            if p == -1 and name in INVERSE:
                fn = INVERSE[name]
            else:
                post_power = p
        self.enter()
        if self.is_op("(", "["):
            closer = _OPEN[self.take().value]
            args = [self.expr()]
            while self.is_op(","):
                self.take()
                args.append(self.expr())
            self.expect(closer)
        else:                                     # sin x, sin 2x
            if not self.starts_operand():
                raise PlotterError("syntax_error", f"{name} needs an argument", position=tk.pos)
            arg = self.power()
            while self.starts_operand() and self.peek().kind != "func" \
                    and not self.is_op("(", "[", "|"):
                arg = arg * self.power()
            args = [arg]
        self.depth -= 1
        if len(args) > (2 if name in TWO_ARG else 1):
            raise PlotterError("syntax_error", f"{name} takes {'1 or 2 arguments' if name in TWO_ARG else 'one argument'}",
                               position=tk.pos)
        v = fn(*args)
        return make_pow(v, post_power) if post_power is not None else v


def parse_tokens(toks: list[Tok]) -> sp.Expr:
    if not toks:
        raise PlotterError("syntax_error", "Missing expression")
    p = _Parser(toks)
    v = p.expr()
    if p.peek() is not None:
        tk = p.peek()
        raise PlotterError("syntax_error", f"Unexpected {tk.value if tk.kind != 'num' else 'number'!r}",
                           position=tk.pos)
    return _check_number(sp.sympify(v))


def parse_expression(text: str) -> sp.Expr:
    """Parse a standalone expression (used for viewport bounds like ``-2pi``)."""
    return parse_tokens(tokenize(normalise_text(text)))


# --------------------------------------------------------------------------- equation level

def _split_top(toks: list[Tok], sep: str) -> list[list[Tok]]:
    parts, cur, depth = [], [], 0
    for tk in toks:
        if tk.kind == "op" and tk.value in ("(", "["):
            depth += 1
        elif tk.kind == "op" and tk.value in (")", "]"):
            depth -= 1
        if depth == 0 and tk.kind == "op" and tk.value == sep:
            parts.append(cur)
            cur = []
        else:
            cur.append(tk)
    parts.append(cur)
    return parts


def _relation_split(toks: list[Tok]):
    depth, idx = 0, []
    for k, tk in enumerate(toks):
        if tk.kind == "op" and tk.value in ("(", "["):
            depth += 1
        elif tk.kind == "op" and tk.value in (")", "]"):
            depth -= 1
        elif depth == 0 and tk.kind == "op" and tk.value in _RELATIONS:
            idx.append(k)
    if len(idx) > 1:
        raise PlotterError("syntax_error", "Only one '=' or inequality sign is allowed",
                           position=toks[idx[1]].pos)
    if not idx:
        return None, toks, None
    k = idx[0]
    return toks[k].value, toks[:k], toks[k + 1:]


def _free(e) -> set:
    return set(e.free_symbols)


def _require_vars(e, allowed: set, context: str):
    extra = _free(e) - allowed
    if extra:
        names = ", ".join(sorted(str(s) for s in extra))
        allowed_s = ", ".join(sorted(str(s) for s in allowed)) or "no variables"
        raise PlotterError("unsupported_variable",
                           f"{context} may only use {allowed_s}; found {names}")


def _flip(rel: str) -> str:
    return {"<": ">", ">": "<", "<=": ">=", ">=": "<="}.get(rel, rel)


def parse_equation(text: str) -> ParsedEquation:
    s = normalise_text(text)
    name = "y"
    m = _FUNC_LHS.match(s)
    if m:
        name = m.group(1)
        s = "y =" + s[m.end():]
    toks = tokenize(s)
    if not toks:
        raise PlotterError("empty_input", "Equation is empty")

    parts = _split_top(toks, ",")
    if len(parts) == 1 and toks[0].kind == "op" and toks[0].value == "(" \
            and toks[-1].kind == "op" and toks[-1].value == ")":
        inner = _split_top(toks[1:-1], ",")
        if len(inner) == 2:
            return _parametric(text, parse_tokens(inner[0]), parse_tokens(inner[1]))
    if len(parts) == 2:
        comps = {}
        for part in parts:
            rel, lt, rt = _relation_split(part)
            if rel != "=" or len(lt) != 1 or lt[0].kind != "var" or lt[0].value not in (x, y):
                raise PlotterError("syntax_error",
                                   "Parametric form is 'x = f(t), y = g(t)' or '(f(t), g(t))'")
            comps[lt[0].value] = parse_tokens(rt)
        if set(comps) != {x, y}:
            raise PlotterError("syntax_error", "Parametric form needs both x = ... and y = ...")
        return _parametric(text, comps[x], comps[y])
    if len(parts) > 2:
        raise PlotterError("syntax_error", "Too many commas; use separate equations instead")

    rel, lt, rt = _relation_split(toks)
    if rel is None:
        e = parse_tokens(toks)
        if not _free(e) - {x}:
            return ParsedEquation(text, "explicit", name, f=e, lhs=y, rhs=e)
        if _free(e) <= {x, y}:
            return ParsedEquation(text, "implicit", lhs=e, rhs=sp.Integer(0), F=e)
        raise PlotterError("unsupported_variable",
                           "A bare expression must use x only (or x and y for '= 0'). "
                           "For polar curves write r = f(theta); for parametric x = f(t), y = g(t)")
    if not lt or not rt:
        raise PlotterError("syntax_error", f"Missing expression on one side of {rel!r}")
    lhs, rhs = parse_tokens(lt), parse_tokens(rt)

    if rel == "=" and (lhs in (r, r ** 2) or rhs in (r, r ** 2)):
        if lhs not in (r, r ** 2):
            lhs, rhs = rhs, lhs
        _require_vars(rhs, {theta}, "A polar equation r = f(theta)")
        if lhs == r:
            return ParsedEquation(text, "polar", lhs=lhs, rhs=rhs, r_exprs=[rhs], rhs_polar=rhs)
        return ParsedEquation(text, "polar", lhs=lhs, rhs=rhs, r_squared=True, rhs_polar=rhs,
                              r_exprs=[sp.sqrt(rhs), -sp.sqrt(rhs)])

    _require_vars(lhs, {x, y}, "A Cartesian equation")
    _require_vars(rhs, {x, y}, "A Cartesian equation")
    if not (_free(lhs) | _free(rhs)):
        raise PlotterError("no_variables", "Equation has no variables to plot")

    explicit_f = None
    rel_y = rel
    if lhs == y and y not in _free(rhs):
        explicit_f = rhs
    elif rhs == y and y not in _free(lhs):
        explicit_f, rel_y = lhs, _flip(rel)

    if rel == "=":
        if explicit_f is not None:
            return ParsedEquation(text, "explicit", name, f=explicit_f, lhs=y, rhs=explicit_f)
        F = lhs - rhs
        if F == 0 or not _free(F):
            raise PlotterError("degenerate_equation",
                               "Equation simplifies to a constant statement with nothing to plot")
        return ParsedEquation(text, "implicit", lhs=lhs, rhs=rhs, F=F)

    # inequalities: normalise to F rel 0 with F = lhs - rhs
    if explicit_f is not None:
        return ParsedEquation(text, "inequality", name, f=explicit_f, lhs=y, rhs=explicit_f,
                              F=y - explicit_f, relation=rel_y, boundary_kind="explicit")
    F = lhs - rhs
    if not _free(F):
        raise PlotterError("degenerate_equation", "Inequality has no variables to plot")
    return ParsedEquation(text, "inequality", lhs=lhs, rhs=rhs, F=F, relation=rel,
                          boundary_kind="implicit")


def _parametric(text: str, xt, yt) -> ParsedEquation:
    _require_vars(xt, {t}, "A parametric x(t)")
    _require_vars(yt, {t}, "A parametric y(t)")
    if t not in _free(xt) | _free(yt):
        raise PlotterError("no_variables", "Parametric equations must depend on t")
    return ParsedEquation(text, "parametric", xt=xt, yt=yt)
