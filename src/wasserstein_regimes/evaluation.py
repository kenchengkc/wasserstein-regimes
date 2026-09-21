# SPDX-License-Identifier: GPL-3.0-only
"""Descriptive diagnostics with explicit temporal and sampling contracts."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def mmd2(x, y, *, bandwidth):
    """Biased Gaussian-kernel squared MMD on scalar observations, not indices."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or not x.size or not y.size:
        raise ValueError('MMD samples must be nonempty vectors')
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError('MMD requires finite samples and positive bandwidth')
    def kernel(a, b):
        return np.exp(-.5 * ((a[:, None] - b[None, :]) / bandwidth) ** 2).mean()
    return float(max(0., kernel(x, x) + kernel(y, y) - 2 * kernel(x, y)))


def future_outcomes(returns, endpoints, *, horizon=21):
    """Evaluation only: horizon starts strictly after each assignment endpoint."""
    r, endpoints = np.asarray(returns, float), np.asarray(endpoints, int)
    if horizon < 1 or r.ndim != 1 or endpoints.ndim != 1:
        raise ValueError('Invalid horizon or array shape')
    keys = ('forward_log_return', 'forward_volatility', 'forward_drawdown', 'forward_downside')
    result = {key: np.full(len(endpoints), np.nan) for key in keys}
    for row, end in enumerate(endpoints):
        if end < 0 or end + horizon >= len(r):
            continue
        sample = r[end + 1:end + horizon + 1]
        if not np.isfinite(sample).all():
            continue
        wealth = np.r_[0., np.cumsum(sample)]
        result['forward_log_return'][row] = sample.sum()
        result['forward_volatility'][row] = sample.std(ddof=0) * np.sqrt(252)
        result['forward_drawdown'][row] = -np.expm1(np.min(wealth - np.maximum.accumulate(wealth)))
        result['forward_downside'][row] = np.sqrt(np.mean(np.minimum(sample, 0) ** 2)) * np.sqrt(252)
    return result


def novelty(distances, calibration, *, threshold=.99):
    d, cal = np.asarray(distances, float), np.sort(np.asarray(calibration, float))
    if d.ndim != 2 or d.shape[1] < 1 or not len(cal) or cal.ndim != 1:
        raise ValueError('Novelty requires at least one center and calibration distances')
    if not np.isfinite(d).all() or not np.isfinite(cal).all() or (d < 0).any() or not 0 < threshold <= 1:
        raise ValueError('Invalid distances or novelty threshold')
    ordered = np.sort(d, axis=1)
    d1 = ordered[:, 0]
    d2 = ordered[:, 1] if d.shape[1] > 1 else np.full(len(d1), np.nan)
    percentile = np.searchsorted(cal, d1, side='right') / len(cal)
    margin = np.divide(d2 - d1, d2, out=np.zeros(len(d2)), where=d2 > 0)
    if d.shape[1] == 1:
        margin[:] = np.nan
    return dict(nearest_distance=d1, second_distance=d2, margin=margin,
                novelty_percentile=percentile, ood=percentile >= threshold)


def temporal_summary(labels, positions=None, *, k=None):
    labels = np.asarray(labels, int)
    positions = np.arange(len(labels)) if positions is None else np.asarray(positions, int)
    if labels.ndim != 1 or positions.shape != labels.shape or not len(labels):
        raise ValueError('Expected nonempty aligned label and position vectors')
    k = int(labels.max()) + 1 if k is None else k
    matrix = np.zeros((k, k), int)
    adjacent = np.diff(positions) == 1
    for a, b in zip(labels[:-1][adjacent], labels[1:][adjacent]):
        matrix[a, b] += 1
    breaks = np.flatnonzero((np.diff(labels) != 0) | ~adjacent) + 1
    durations = np.diff(np.r_[0, breaks, len(labels)])
    n_adjacent = int(adjacent.sum())
    switches = int(((np.diff(labels) != 0) & adjacent).sum())
    return dict(transitions=matrix.tolist(), switching_frequency=switches / n_adjacent if n_adjacent else None,
                mean_dwell=float(durations.mean()), median_dwell=float(np.median(durations)),
                dwell_times=durations.tolist(), adjacent_pairs=n_adjacent)


def label_mapping(truth, predicted, *, k):
    """Map cluster IDs to known synthetic states using training labels only."""
    counts = np.zeros((k, k), int)
    for a, b in zip(truth, predicted):
        counts[int(b), int(a)] += 1
    row, col = linear_sum_assignment(-counts)
    mapping = np.arange(k)
    mapping[row] = col
    return mapping


def moving_block_indices(n, *, block_length, rng):
    if not isinstance(block_length, (int, np.integer)) or not 1 <= block_length <= n:
        raise ValueError('block_length must be an integer within the sample')
    starts = rng.integers(0, n - block_length + 1, size=int(np.ceil(n / block_length)))
    return np.concatenate([np.arange(start, start + block_length) for start in starts])[:n]


def align_centroids(reference, current):
    """Return current-ID -> reference-ID minimum total W2-distance matching."""
    from .transport import pairwise_distance
    if len(reference) != len(current):
        raise ValueError('Matching requires equal state counts')
    rows, cols = linear_sum_assignment(pairwise_distance(current, reference))
    mapping = np.empty(len(current), int)
    mapping[rows] = cols
    return mapping


def component_fractions(parts):
    sums = {key: float(np.sum(parts[key])) for key in ('location', 'scale', 'shape')}
    total = sum(sums.values())
    return {key: value / total if total else 0. for key, value in sums.items()}
