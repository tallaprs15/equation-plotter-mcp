"""Convert a small, safe subset of LaTeX into the plain-text syntax the parser understands.

The output is *not* trusted: it goes through the same whitelisted tokenizer/parser as any
other input. Unknown commands are rejected rather than passed through.
"""

from __future__ import annotations

import re

from .errors import PlotterError

_FUNCS = {
    "sin", "cos", "tan", "sec", "csc", "cot", "arcsin", "arccos", "arctan",
    "sinh", "cosh", "tanh", "coth", "sech", "csch", "exp", "ln", "log",
}
_SYMBOLS = {"pi": " pi ", "theta": " theta ", "infty": None}
_RELATIONS = {"le": "<=", "leq": "<=", "leqslant": "<=", "ge": ">=", "geq": ">=",
              "geqslant": ">=", "lt": "<", "gt": ">"}
_SPACING = {",", ";", ":", " ", "!", "quad", "qquad", "displaystyle", "textstyle"}
_SIZING = {"left", "right", "big", "Big", "bigg", "Bigg", "bigl", "bigr", "Bigl", "Bigr"}
_CMD_RE = re.compile(r"\\([A-Za-z]+|.)")

MAX_NEST = 30


def looks_like_latex(s: str) -> bool:
    return "\\" in s or "{" in s


def latex_to_plain(s: str) -> str:
    out, i = _convert(s, 0, depth=0, closing=None)
    if i != len(s):
        raise PlotterError("latex_error", "Unbalanced braces in LaTeX input", position=i)
    return out


def _convert(s: str, i: int, depth: int, closing: str | None) -> tuple[str, int]:
    if depth > MAX_NEST:
        raise PlotterError("too_complex", "LaTeX input is nested too deeply")
    out: list[str] = []
    while i < len(s):
        c = s[i]
        if closing is not None and c == closing:
            return "".join(out), i
        if c == "}":
            raise PlotterError("latex_error", "Unexpected '}' in LaTeX input", position=i)
        if c == "{":
            inner, i = _group(s, i, depth)
            out.append(f"({inner})")
            continue
        if c != "\\":
            out.append(c)
            i += 1
            continue
        m = _CMD_RE.match(s, i)
        if not m:
            raise PlotterError("latex_error", "Dangling backslash in LaTeX input", position=i)
        name = m.group(1)
        i = m.end()
        if name in ("frac", "dfrac", "tfrac"):
            num, i = _group(s, i, depth)
            den, i = _group(s, i, depth)
            out.append(f"(({num})/({den}))")
        elif name == "sqrt":
            index = None
            j = _skip_ws(s, i)
            if j < len(s) and s[j] == "[":
                index, i = _convert(s, j + 1, depth + 1, closing="]")
                if i >= len(s):
                    raise PlotterError("latex_error", "Unclosed '[' after \\sqrt")
                i += 1
            arg, i = _group(s, i, depth)
            out.append(f"(({arg})^(1/({index})))" if index else f"sqrt({arg})")
        elif name in _FUNCS:
            out.append(f" {name} ")
        elif name in _SYMBOLS:
            if _SYMBOLS[name] is None:
                raise PlotterError("latex_error", "Infinity is not supported in equations")
            out.append(_SYMBOLS[name])
        elif name in _RELATIONS:
            out.append(f" {_RELATIONS[name]} ")
        elif name in ("cdot", "times"):
            out.append("*")
        elif name == "div":
            out.append("/")
        elif name in ("vert", "lvert", "rvert", "|", "mid"):
            out.append("|")
        elif name in _SPACING:
            out.append(" ")
        elif name in _SIZING:
            j = _skip_ws(s, i)
            if j < len(s) and s[j] == ".":  # \left. / \right. are invisible delimiters
                i = j + 1
        elif name in ("operatorname", "mathrm", "text", "mathit"):
            inner, i = _group(s, i, depth)
            out.append(f" {inner} ")
        elif name in ("{", "}"):
            out.append("(" if name == "{" else ")")
        else:
            raise PlotterError("latex_error", f"Unsupported LaTeX command \\{name}",
                               position=m.start())
    if closing is not None:
        raise PlotterError("latex_error", f"Missing '{closing}' in LaTeX input")
    return "".join(out), i


def _skip_ws(s: str, i: int) -> int:
    while i < len(s) and s[i].isspace():
        i += 1
    return i


def _group(s: str, i: int, depth: int) -> tuple[str, int]:
    """Read a ``{...}`` group, or a single character/command as LaTeX allows (``\\frac12``)."""
    i = _skip_ws(s, i)
    if i >= len(s):
        raise PlotterError("latex_error", "Missing argument in LaTeX input")
    if s[i] == "{":
        inner, j = _convert(s, i + 1, depth + 1, closing="}")
        return inner, j + 1
    if s[i] == "\\":
        m = _CMD_RE.match(s, i)
        inner, _ = _convert(m.group(0), 0, depth + 1, closing=None)
        return inner, m.end()
    return s[i], i + 1
