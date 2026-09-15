# SPDX-License-Identifier: GPL-3.0-only
"""Exact transport geometry for equally weighted one-dimensional samples."""

from numbers import Integral

import numpy as np


_METRICS = {"w1": 1, "w2": 2}


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _windows(values, name):
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a nonempty finite 2D array") from error
    if array.ndim != 2 or 0 in array.shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a nonempty finite 2D array")
    return array


def _paired_quantiles(samples, centers):
    samples = _windows(samples, "samples")
    centers = _windows(centers, "centers")
    if samples.shape[1] != centers.shape[1]:
        raise ValueError("samples and centers must have the same number of atoms")
    return np.sort(samples, axis=1), np.sort(centers, axis=1)


def pairwise_distance(samples, centers, metric="w2", chunk_size=256):
    """Return all true W1 or W2 distances between equal-size samples.

    Work is split across sample rows and center rows, so the largest temporary
    is two-dimensional rather than an ``M x K x L`` broadcast.
    """
    if metric not in _METRICS:
        raise ValueError("metric must be 'w1' or 'w2'")
    chunk_size = _positive_int(chunk_size, "chunk_size")
    quantiles, center_quantiles = _paired_quantiles(samples, centers)
    power = _METRICS[metric]
    costs = _pairwise_sorted_costs(quantiles, center_quantiles, power, chunk_size)
    return costs if power == 1 else np.sqrt(costs)


def _pairwise_sorted_costs(quantiles, center_quantiles, power, chunk_size):
    """Objective costs for sorted, already validated equal-length quantiles."""
    distances = np.empty((len(quantiles), len(center_quantiles)), dtype=np.float64)
    for start in range(0, len(quantiles), chunk_size):
        stop = min(start + chunk_size, len(quantiles))
        block = quantiles[start:stop]
        for cluster, center in enumerate(center_quantiles):
            distances[start:stop, cluster] = np.mean(np.abs(block - center) ** power, axis=1)
    return distances


def w2_decomposition(samples, centers, chunk_size=256):
    """Decompose squared W2 into nonnegative location, scale, and shape terms.

    Means and population standard deviations (``ddof=0``) are computed in
    each distribution's own coordinates. Constant distributions have zero
    standardized shape and therefore contribute no shape term.
    """
    chunk_size = _positive_int(chunk_size, "chunk_size")
    quantiles, center_quantiles = _paired_quantiles(samples, centers)
    sample_means = quantiles.mean(axis=1)
    center_means = center_quantiles.mean(axis=1)
    sample_scales = quantiles.std(axis=1, ddof=0)
    center_scales = center_quantiles.std(axis=1, ddof=0)

    center_shapes = np.zeros_like(center_quantiles)
    nonconstant_centers = center_scales > 0
    center_shapes[nonconstant_centers] = (
        center_quantiles[nonconstant_centers]
        - center_means[nonconstant_centers, None]
    ) / center_scales[nonconstant_centers, None]

    output_shape = (len(quantiles), len(center_quantiles))
    location = np.empty(output_shape, dtype=np.float64)
    scale = np.empty(output_shape, dtype=np.float64)
    shape = np.empty(output_shape, dtype=np.float64)
    for start in range(0, len(quantiles), chunk_size):
        stop = min(start + chunk_size, len(quantiles))
        means = sample_means[start:stop]
        scales = sample_scales[start:stop]
        block = quantiles[start:stop]
        block_shapes = np.zeros_like(block)
        nonconstant = scales > 0
        block_shapes[nonconstant] = (
            block[nonconstant] - means[nonconstant, None]
        ) / scales[nonconstant, None]

        location[start:stop] = (means[:, None] - center_means[None, :]) ** 2
        scale[start:stop] = (scales[:, None] - center_scales[None, :]) ** 2
        correlations = block_shapes @ center_shapes.T / quantiles.shape[1]
        correlations = np.clip(correlations, -1.0, 1.0)
        shape[start:stop] = (
            2.0
            * scales[:, None]
            * center_scales[None, :]
            * (1.0 - correlations)
        )

    np.maximum(location, 0.0, out=location)
    np.maximum(scale, 0.0, out=scale)
    np.maximum(shape, 0.0, out=shape)
    total = location + scale + shape
    return {"location": location, "scale": scale, "shape": shape, "total": total}


def standardize_windows(samples):
    """Standardize each window in temporal order using population scale.

    A constant window has no scale information and is mapped explicitly to a
    row of zeros. The observations are never sorted by this function.
    """
    windows = _windows(samples, "samples")
    means = windows.mean(axis=1, keepdims=True)
    scales = windows.std(axis=1, ddof=0, keepdims=True)
    standardized = np.zeros_like(windows)
    nonconstant = scales[:, 0] > 0
    standardized[nonconstant] = (
        windows[nonconstant] - means[nonconstant]
    ) / scales[nonconstant]
    return standardized
