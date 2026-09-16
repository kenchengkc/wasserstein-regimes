"""Exact univariate empirical-distribution clustering."""

from .core import FitResult, barycenter, fit, rolling_windows, wasserstein
from .clustering import WassersteinKMeans
from .transport import pairwise_distance, standardize_windows, w2_decomposition

__all__ = ["FitResult", "barycenter", "fit", "rolling_windows", "wasserstein",
           "WassersteinKMeans", "pairwise_distance", "standardize_windows", "w2_decomposition"]
