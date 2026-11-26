"""Regenerate examples/: one JSON + one PNG per representative equation set.

Usage: python scripts/make_examples.py [name ...]
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

from equation_plotter import analyse_and_plot

EXAMPLES = {
    "01_quadratic": (["y = x^2 - 3x + 2"], {}),
    "02_rational_asymptote": (["f(x) = 1/(x-2)"], {}),
    "03_cubic_bare": (["x^3 - x"], {}),
    "04_rational_hole_oblique": (["y = (x^2 - 1)/(x - 1) + 1/x"], {}),
    "05_circle": (["x^2 + y^2 = 25"], {}),
    "06_hyperbola": (["x^2/9 - y^2/4 = 1"], {}),
    "07_parametric_circle": (["x = cos(t), y = sin(t)"], {}),
    "08_polar_cardioid": (["r = 1 + cos(theta)"], {}),
    "09_sin_cos": (["y = sin(x)", "y = cos(x)"], {}),
    "10_tan": (["y = tan(x)"], {}),
    "11_exp_log": (["y = e^x", "y = ln(x)", "y = x"], {}),
    "12_latex_rational": ([r"y = \frac{x^2 - 4}{x^2 - 1}"], {}),
    "13_inequality": (["y > x^2 - 4", "x^2 + y^2 <= 9"], {}),
    "14_rose": (["r = 2cos(3theta)"], {}),
    "15_lissajous": (["x = sin(3t), y = sin(2t)"], {}),
    "16_line_and_parabola": (["2x + 3y = 6", "y = 2sin(x) + x/2"], {}),
}


def main(names: list[str]) -> None:
    out = Path(__file__).resolve().parent.parent / "examples"
    out.mkdir(exist_ok=True)
    for name, (eqs, kw) in EXAMPLES.items():
        if names and name not in names:
            continue
        res = analyse_and_plot(eqs, **kw)
        if "error" in res:
            print(name, "ERROR", res["error"])
            continue
        (out / f"{name}.png").write_bytes(base64.b64decode(res["image"]["data"]))
        res["image"]["data"] = res["image"]["data"][:60] + "...(truncated; see PNG)"
        (out / f"{name}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
        print(f"{name}: {res['timing_ms']['total']} ms, {len(res['warnings'])} warnings, "
              f"viewport {res['viewport']['x']} x {res['viewport']['y']}")
        for w in res["warnings"]:
            print("   !", w["message"])


if __name__ == "__main__":
    main(sys.argv[1:])
