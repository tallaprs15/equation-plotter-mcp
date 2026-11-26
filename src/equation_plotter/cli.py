"""Command-line interface: ``equation-plotter "y = x^2" [--format svg] [--out r.json] [--save-image g.png]``."""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys

EXAMPLES = """examples:
  equation-plotter "y = x^2 - 3x + 2"
  equation-plotter "x^2 + y^2 = 25" --format svg --out result.json
  equation-plotter "y = sin(x)" "y = cos(x)" --save-image graph.png
  equation-plotter "r = 1 + cos(theta)" --x-range=-1,3
  equation-plotter "x = cos(3t), y = sin(2t)" --t-range "[0, 2pi]"

ranges are 'a,b' (use --x-range=-5,5 or --x-range "[-5,5]" for negatives); values may be
expressions like 2pi or -pi/2."""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="equation-plotter", epilog=EXAMPLES,
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description="Analyse and plot equations; prints one JSON document.")
    ap.add_argument("equations", nargs="+", help="one or more equations (quote each one)")
    ap.add_argument("--x-range", help="x window, e.g. -5,5")
    ap.add_argument("--y-range", help="y window, e.g. -2,2")
    ap.add_argument("--t-range", help="parameter range for parametric curves")
    ap.add_argument("--theta-range", help="theta range for polar curves")
    ap.add_argument("--format", choices=["png", "svg", "none"], default=None,
                    help="image format embedded in the JSON (default png, or from --save-image)")
    ap.add_argument("--out", help="write the JSON here instead of stdout")
    ap.add_argument("--save-image", help="also write the decoded image to this file")
    ap.add_argument("--timeout", type=float, default=None,
                    help="symbolic analysis budget in seconds (default 20)")
    ap.add_argument("--compact", action="store_true", help="compact JSON (no indentation)")
    ap.add_argument("--no-image-data", action="store_true",
                    help="drop the base64 data from the JSON (useful with --save-image)")
    ap.add_argument("--version", action="version", version=_version())
    args = ap.parse_args(argv)

    fmt = args.format
    if fmt is None:
        ext = os.path.splitext(args.save_image or "")[1].lower()
        fmt = "svg" if ext == ".svg" else "png"
    if args.save_image and fmt == "none":
        ap.error("--save-image needs --format png or svg")

    from .core import analyse_and_plot

    result = analyse_and_plot(args.equations, x_range=args.x_range, y_range=args.y_range,
                              image_format=fmt, t_range=args.t_range,
                              theta_range=args.theta_range, timeout=args.timeout)
    code = 0 if "error" not in result else 2
    if args.save_image and result.get("image"):
        try:
            with open(args.save_image, "wb") as fh:
                fh.write(base64.b64decode(result["image"]["data"]))
        except OSError as e:
            result.setdefault("warnings", []).append(
                {"equation": None, "field": "image", "code": "save_failed", "message": str(e)})
    if args.no_image_data and result.get("image"):
        result["image"]["data"] = None
    text = json.dumps(result, indent=None if args.compact else 2, ensure_ascii=False)
    if args.out:
        try:
            with open(args.out, "w", encoding="utf-8") as fh:
                fh.write(text + "\n")
        except OSError as e:
            print(json.dumps({"error": {"code": "write_failed", "message": str(e)}}))
            return 2
    else:
        sys.stdout.write(text + "\n")
    return code


def _version() -> str:
    from . import __version__
    return f"equation-plotter {__version__}"


if __name__ == "__main__":
    raise SystemExit(main())
