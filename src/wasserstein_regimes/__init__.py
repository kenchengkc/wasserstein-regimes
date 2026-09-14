"""Exact univariate empirical-distribution reference implementation."""

from .core import FitResult, barycenter, fit, rolling_windows, wasserstein

__all__ = ["FitResult", "barycenter", "fit", "rolling_windows", "wasserstein"]
