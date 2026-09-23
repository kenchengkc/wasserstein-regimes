"""Empirical-distribution clustering with exact 1D and finite-slice geometry."""

from .sliced import SlicedWassersteinKMedoids, sliced_w2
from .core import FitResult, barycenter, fit, rolling_windows, wasserstein
from .clustering import WassersteinKMeans
from .transport import pairwise_distance, standardize_windows, w2_decomposition

__all__ = ["SlicedWassersteinKMedoids", "sliced_w2", "FitResult", "barycenter", "fit", "rolling_windows", "wasserstein",
           "WassersteinKMeans", "pairwise_distance", "standardize_windows", "w2_decomposition"]
