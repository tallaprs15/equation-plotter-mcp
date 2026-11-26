"""Lean smoke tests: every input type end to end, key facts, hostile input, timeouts, CLI."""

import base64
import json

import pytest

from equation_plotter import analyse_and_plot
from equation_plotter import core
from equation_plotter.cli import main as cli_main
from equation_plotter.sandbox import HardTimeout, Worker

CASES = [
    (["y = x^2 - 3x + 2"], "explicit", "quadratic"),
    (["f(x) = 1/(x-2)"], "explicit", "rational"),
    (["x^3 - x"], "explicit", "cubic (polynomial, degree 3)"),
    (["x^2 + y^2 = 25"], "implicit", "circle"),
    (["x^2/9 - y^2/4 = 1"], "implicit", "hyperbola"),
    (["x = cos(t), y = sin(t)"], "parametric", "circle"),
    (["r = 1 + cos(theta)"], "polar", "cardioid"),
    ([r"y = \frac{1}{x-2}"], "explicit", "rational"),
    (["y > x^2"], "inequality", "quadratic"),
    (["y = sin(x)", "y = cos(x)"], "explicit", "trigonometric"),
]


def _get(eq, key):
    return [(p["x"]["float"], p["y"]["float"]) for p in eq[key]]


@pytest.mark.parametrize("eqs,kind,cls", CASES)
def test_end_to_end(eqs, kind, cls):
    res = analyse_and_plot(eqs)
    assert "error" not in res, res.get("error")
    json.dumps(res)  # strictly JSON-serialisable
    assert res["version"] and res["viewport"]["x"][0] < res["viewport"]["x"][1]
    assert base64.b64decode(res["image"]["data"])[:8] == b"\x89PNG\r\n\x1a\n"
    eq = res["equations"][0]
    assert eq["type"] == kind and eq["classification"] == cls
    assert len(res["equations"]) == len(eqs)


def test_key_facts():
    r = analyse_and_plot(["y = x^2 - 3x + 2", "y = (x^2 - 1)/(x - 1) + 1/x", "x^2 + y^2 = 25"],
                         image_format="none", isolate=False)
    quad, hole, circle = r["equations"]
    assert _get(quad, "x_intercepts") == [(1.0, 0.0), (2.0, 0.0)]
    assert quad["stationary_points"][0]["x"]["exact"] == "3/2"
    assert quad["stationary_points"][0]["nature"] == "local minimum"
    assert _get(hole, "holes") == [(1.0, 3.0)]
    assert hole["asymptotes"]["vertical"][0]["x"]["exact"] == "0"
    assert hole["asymptotes"]["oblique"][0]["equation"] == "y = x + 1"
    assert circle["details"]["radius"]["exact"] == "5"
    assert r["image"] is None


def test_svg_and_periodic():
    r = analyse_and_plot("y = tan(x)", image_format="svg")
    assert r["image"]["mime_type"] == "image/svg+xml"
    assert b"<svg" in base64.b64decode(r["image"]["data"])[:400]
    eq = r["equations"][0]
    assert eq["periodicity"]["period"]["exact"] == "pi"
    assert {a["x"]["exact"] for a in eq["asymptotes"]["vertical"]} >= {"-pi/2", "pi/2"}


@pytest.mark.parametrize("bad,code", [
    ("__import__('os').system('id')", "invalid_character"),
    ("y = x.__class__", "invalid_character"),
    ("y = eval(x)", "unknown_name"),
    ("lambda: 1", "invalid_character"),
    ("y = 9^9^9^9", "number_too_large"),
    ("y = " + "(" * 60 + "x" + ")" * 60, "too_complex"),
    ("y = " + "x+" * 200 + "1", "input_too_long"),
    (r"y = \input{/etc/passwd}", "latex_error"),
    ("y = (x", "syntax_error"),
])
def test_hostile_input_is_rejected_as_json(bad, code):
    r = analyse_and_plot(bad)
    assert r["error"]["code"] == code
    json.dumps(r)


def test_bad_arguments_are_json_errors():
    assert analyse_and_plot([])["error"]["code"] == "invalid_input"
    assert analyse_and_plot("y=x", x_range=(5, 1))["error"]["code"] == "invalid_range"
    assert analyse_and_plot("y=x", image_format="gif")["error"]["code"] == "invalid_format"


def test_worker_hard_timeout_kills_job():
    with Worker() as w:
        with pytest.raises(HardTimeout):
            w.call("analyse", (["y = x^2"], {"symbolic": True, "symbolic_seconds": 5}), 0.001)


def test_symbolic_hard_timeout_falls_back_to_numeric(monkeypatch):
    real = Worker.call
    calls = []

    def flaky(self, name, args, timeout):
        calls.append(args[1]["symbolic"] if name == "analyse" else name)
        if name == "analyse" and args[1]["symbolic"]:
            raise HardTimeout()
        return real(self, name, args, timeout)

    monkeypatch.setattr(core.Worker, "call", flaky)
    r = analyse_and_plot("y = x^3 - x", image_format="none")
    assert calls == [True, False]
    assert r["warnings"][0]["code"] == "hard_timeout"
    xs = [round(v, 6) for v, _ in _get(r["equations"][0], "x_intercepts")]
    assert xs == [-1.0, 0.0, 1.0]


def test_cli(tmp_path, capsys):
    out, img = tmp_path / "r.json", tmp_path / "g.svg"
    assert cli_main(["x^2 + y^2 = 4", "--out", str(out), "--save-image", str(img)]) == 0
    data = json.loads(out.read_text())
    assert data["image"]["format"] == "svg" and img.read_bytes().lstrip().startswith(b"<")
    assert cli_main(["y = foo(x)"]) == 2
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "unknown_name"
