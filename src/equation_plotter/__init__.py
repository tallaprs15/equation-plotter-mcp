"""Plotter: equation in, one JSON document (analysis + rendered image) out."""

__version__ = "1.0.0"


def analyse_and_plot(*args, **kwargs):
    """See :func:`equation_plotter.core.analyse_and_plot`."""
    from .core import analyse_and_plot as _impl
    return _impl(*args, **kwargs)


__all__ = ["analyse_and_plot", "__version__"]
