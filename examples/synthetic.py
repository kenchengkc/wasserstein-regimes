# SPDX-License-Identifier: GPL-3.0-only
"""Offline retrospective smoke example, not a paper replication."""

import numpy as np

from wasserstein_regimes import fit, rolling_windows

rng = np.random.default_rng(42)
sigma = np.repeat([0.007, 0.025, 0.007, 0.025], 1000)
returns = rng.normal(0, sigma)
windows, endpoints = rolling_windows(returns, length=63, stride=5)
for p in (1, 2):
    result = fit(windows, n_clusters=2, p=p, seed=42)
    print({
        "metric": f"W{p}",
        "mode": "retrospective synthetic smoke test",
        "windows": len(endpoints),
        "converged": result.converged,
        "objective": result.objective,
        "cluster_sizes": np.bincount(result.labels, minlength=2).tolist(),
        "centroid_std": result.centers.std(axis=1).tolist(),
    })
