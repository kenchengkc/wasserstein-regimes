# SPDX-License-Identifier: GPL-3.0-only
"""Reproducible dependence-only controls and a bounded-candidate scaling study."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
import tracemalloc
from pathlib import Path

import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

from wasserstein_regimes.benchmark import _machine
from wasserstein_regimes.sliced import SlicedWassersteinKMedoids, sliced_w2


def correlated_windows(seed, n=120, length=63):
    """Independent windows: standard normal marginals, correlation +/−0.8."""
    rng = np.random.default_rng(seed)
    labels = np.arange(n) % 2
    rng.shuffle(labels)
    noise = rng.normal(size=(n, length, 2))
    first = noise[:, :, 0]
    rho = np.where(labels == 0, -.8, .8)
    second = rho[:, None] * first + .6 * noise[:, :, 1]
    return np.stack([first, second], axis=2), labels


def marginal_features(x):
    # Sorting is within each asset, not across the concatenated vector.
    return np.sort(x, axis=1).reshape(len(x), -1) / np.sqrt(x.shape[1]*x.shape[2])


def covariance_features(x):
    centered = x - x.mean(axis=1, keepdims=True)
    return np.stack([(centered[:,:,0]**2).mean(axis=1),
                     (centered[:,:,1]**2).mean(axis=1),
                     (centered[:,:,0]*centered[:,:,1]).mean(axis=1)], axis=1)


def run():
    settings = dict(seeds=[17, 42, 83], projection_counts=[8, 32, 128],
                    n_train=120, n_test=120, atoms=63, candidate_size=64,
                    n_init=3, n_clusters=2, max_iter=100)
    a = np.array([[-1., -1.], [1., 1.]])
    b = np.array([[-1., 1.], [1., -1.]])
    directions = np.array([[1.,1.], [1.,-1.]])/np.sqrt(2)
    control = dict(marginal_embedding_distance=float(np.linalg.norm(
                       marginal_features(a[None])-marginal_features(b[None]))),
                   joint_sliced_distance=float(sliced_w2(a[None], b[None], directions)[0,0]),
                   projections=directions.tolist())
    rows = []
    for seed in settings["seeds"]:
        train, _ = correlated_windows(seed)
        test, truth = correlated_windows(seed+10000)
        baselines = {}
        for name, features in (("marginal", marginal_features), ("covariance", covariance_features)):
            model = KMeans(n_clusters=2, n_init=20, random_state=seed).fit(features(train))
            baselines[name+"_ari"] = float(adjusted_rand_score(truth, model.predict(features(test))))
        for count in settings["projection_counts"]:
            model = SlicedWassersteinKMedoids(n_clusters=2, n_projections=count, candidate_size=64,
                                             n_init=3, random_state=seed).fit(train)
            prediction = model.predict(test)
            rows.append(dict(seed=seed, projections=count,
                             sliced_ari=float(adjusted_rand_score(truth, prediction)),
                             occupied_states=int(len(np.unique(prediction))),
                             training_objective=model.inertia_, **baselines))
    benchmark = []
    for n in [1000, 10000]:
        x = np.random.default_rng(91).normal(size=(n, 63, 5))
        params = dict(n_clusters=3, n_projections=32, candidate_size=64, n_init=2,
                      random_state=42, chunk_size=32)
        # One warmup; timings are observations, not an optimized speed claim.
        SlicedWassersteinKMedoids(**params).fit(x[:64])
        start = time.perf_counter()
        model = SlicedWassersteinKMedoids(**params).fit(x)
        fit_seconds = time.perf_counter()-start
        start = time.perf_counter()
        model.predict(x[:1000])
        score_seconds = time.perf_counter()-start
        tracemalloc.start()
        SlicedWassersteinKMedoids(**params).fit(x)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        benchmark.append(dict(n=n, atoms=63, assets=5, parameters=params,
                              fit_seconds=fit_seconds, score_1000_seconds=score_seconds,
                              peak_traced_fit_bytes=peak, input_bytes=x.nbytes))
    source = Path(__file__)
    return dict(schema_version=1, settings=settings, deterministic_control=control,
                synthetic_rows=rows, benchmark=benchmark,
                limitations=[
                    "Synthetic distribution classification, not time-series change-point delay or market prediction.",
                    "A covariance baseline can solve this control; no incremental advantage is asserted.",
                    "Data seeds and projection seeds vary together; a crossed sensitivity study is future work.",
                    "Only three seeds; descriptive results, no confidence interval or power claim.",
                    "Memory is peak traced allocations during fit, excluding preallocated input; not process RSS.",
                    "Timing excludes input generation, is single-run after warmup, with no tracemalloc active.",
                    "Finite-projection sliced W2 differs from full multivariate W2; medoid optimization is sampled."
                ],
                provenance=dict(git_commit=subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip(),
                                example_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                                numpy=np.__version__, sklearn=sklearn.__version__, python=platform.python_version(),
                                machine=_machine(), blas_threads_requested=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/joint-synthetic.json")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        result = run()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(destination)
