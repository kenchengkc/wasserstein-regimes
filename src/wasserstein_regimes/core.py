# SPDX-License-Identifier: GPL-3.0-only
"""Small CPU reference, not the full timestamp-aware research pipeline.

All distributions have the same number of equally weighted atoms. Inputs may
be unsorted. Objective = sum of W_p ** p for p in {1, 2}. No network access.
"""

from dataclasses import dataclass
from numbers import Integral

import numpy as np


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _validate_p(p):
    if isinstance(p, bool) or p not in (1, 2):
        raise ValueError("Only p=1 (W1) and p=2 (squared W2 objective) are supported")


def _atoms(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or 0 in values.shape or not np.isfinite(values).all():
        raise ValueError("Expected a nonempty finite 2D array of equal-size samples")
    return np.sort(values, axis=1)


def wasserstein(x, y, *, p=2):
    """Return the true W_p distance, not its p-th power."""
    _validate_p(p)
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or x.size == 0:
        raise ValueError("Expected two nonempty 1D samples of equal length")
    q = _atoms(np.stack([x, y]))
    return float(np.mean(np.abs(q[0] - q[1]) ** p) ** (1 / p))


def barycenter(samples, *, p=2):
    """Minimize sum_i W_p(sample_i, center)**p with equal window weights."""
    _validate_p(p)
    q = _atoms(samples)
    return np.median(q, axis=0) if p == 1 else np.mean(q, axis=0)


def rolling_windows(returns, *, length, stride=1):
    """Return full trailing samples and zero-based inclusive endpoint indices.

    Reject missing/nonfinite inputs. Calendar-aware gap handling belongs to the
    planned data layer. Empty output is returned when history is too short.
    """
    _positive_int(length, "length")
    _positive_int(stride, "stride")
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("returns must be a finite 1D array")
    if len(values) < length:
        return np.empty((0, length)), np.empty(0, dtype=int)
    windows = np.lib.stride_tricks.sliding_window_view(values, length)[::stride].copy()
    ends = np.arange(length - 1, len(values), stride)
    return windows, ends


def _costs(q, centers, p):
    # O(M*k) output and O(M*w) temporary, not a full M*k*w allocation.
    return np.column_stack([
        np.mean(np.abs(q - center) ** p, axis=1) for center in centers
    ])


@dataclass(frozen=True)
class FitResult:
    centers: np.ndarray
    labels: np.ndarray
    objective: float
    objective_history: tuple[float, ...]
    n_iter: int
    converged: bool
    p: int

    def transform(self, samples):
        """Distances to all centers (true W_p, rather than objective costs)."""
        q = _atoms(samples)
        if q.shape[1] != self.centers.shape[1]:
            raise ValueError("Atom count must match fitted centers")
        return _costs(q, self.centers, self.p) ** (1 / self.p)

    def predict(self, samples):
        return self.transform(samples).argmin(axis=1)


def fit(samples, *, n_clusters=2, p=2, n_init=10, max_iter=300, seed=42):
    """Lloyd reference with random distinct-distribution starts.

    Select the lowest objective among converged starts, or among all starts
    if none converged. In the latter case the result is marked unconverged.
    No tolerance stopping: convergence requires unchanged nearest assignments
    and all requested clusters occupied. Ties choose the lowest center index.
    """
    _validate_p(p)
    for name, value in (("n_clusters", n_clusters), ("n_init", n_init), ("max_iter", max_iter)):
        _positive_int(value, name)
    q = _atoms(samples)
    unique = np.unique(q, axis=0)
    if len(unique) < n_clusters:
        raise ValueError("n_clusters exceeds the number of distinct distributions")
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(n_init):
        centers = unique[rng.choice(len(unique), n_clusters, replace=False)].copy()
        history = []
        converged = False
        for iteration in range(1, max_iter + 1):
            costs = _costs(q, centers, p)
            labels = costs.argmin(axis=1)
            if not history:
                history.append(float(costs[np.arange(len(q)), labels].sum()))
            new_centers = centers.copy()
            empty = []
            for cluster in range(n_clusters):
                members = q[labels == cluster]
                if len(members):
                    new_centers[cluster] = (
                        np.median(members, axis=0) if p == 1 else members.mean(axis=0)
                    )
                else:
                    empty.append(cluster)
            if empty:
                # Add farthest representatives sequentially to occupied centers.
                occupied = [c for c in range(n_clusters) if c not in empty]
                for cluster in empty:
                    residual = _costs(q, new_centers[occupied], p).min(axis=1)
                    new_centers[cluster] = q[int(residual.argmax())]
                    occupied.append(cluster)
            centers = new_centers
            final_costs = _costs(q, centers, p)
            final_labels = final_costs.argmin(axis=1)
            objective = float(final_costs[np.arange(len(q)), final_labels].sum())
            history.append(objective)
            if np.array_equal(labels, final_labels) and len(np.unique(final_labels)) == n_clusters:
                converged = True
                break
        centers.setflags(write=False)
        final_labels.setflags(write=False)
        results.append(FitResult(
            centers, final_labels, objective, tuple(history), iteration, converged, p,
        ))
    candidates = [result for result in results if result.converged] or results
    return min(candidates, key=lambda result: result.objective)
