# SPDX-License-Identifier: GPL-3.0-only
"""Independent, seeded synthetic checks of distributional regime recovery."""

from __future__ import annotations

import math
import time

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .baselines import FeatureBaseline
from .clustering import WassersteinKMeans
from .evaluation import label_mapping, temporal_summary
from .transport import pairwise_distance, standardize_windows

_METHODS = ("w2", "w1", "shape_w2", "volatility", "mean_vol", "moments", "rich", "gmm")
_DWELL = 756
_SUPPORT = np.arange(6, dtype=np.float64)
_DIFFERENCE = np.array([1, -5, 10, -10, 5, -1])
_COUNTS = np.stack([50 + 3 * _DIFFERENCE, 50 - 3 * _DIFFERENCE])


def exact_moment_blocks(*, seed: int = 42, n_blocks: int = 6):
    """Alternating, independently permuted 300-atom blocks of two laws.

    The signed fifth finite difference makes powers zero through four agree.
    """
    if n_blocks < 2:
        raise ValueError("n_blocks must be at least two")
    rng = np.random.default_rng(seed)
    truth = np.arange(n_blocks) % 2
    blocks = np.stack([rng.permutation(np.repeat(_SUPPORT, _COUNTS[state])) for state in truth])
    return blocks, truth


def _sample_stream(suite, rng):
    states = np.tile([int(rng.integers(2)), 0], 2)
    states[1::2] = 1 - states[0]
    truth = np.repeat(states, _DWELL)
    values = np.empty(len(truth), dtype=np.float64)
    for state in (0, 1):
        indices = np.flatnonzero(truth == state)
        n = len(indices)
        if suite == "variance":
            sample = rng.normal(0, .01 if state == 0 else .03, n)
        elif suite == "normal_t5":
            sample = (rng.normal(size=n) if state == 0 else rng.standard_t(5, size=n) * math.sqrt(3 / 5)) * .01
        elif suite == "signed_exponential":
            sample = (rng.exponential(size=n) - 1) * (.01 if state == 0 else -.01)
        elif suite == "equal_four_moments":
            sample = rng.choice(_SUPPORT, size=n, p=_COUNTS[state] / 300) * .01
        else:
            raise ValueError(f"unknown synthetic suite {suite!r}")
        values[indices] = sample
    return values, truth


def _window_rows(values, truth, length, stride):
    ends = np.arange(length - 1, len(values), stride)
    starts = ends - length + 1
    windows = np.lib.stride_tricks.sliding_window_view(values, length)[starts].copy()
    changes = np.r_[0, np.cumsum(np.diff(truth) != 0)]
    pure = changes[starts] == changes[ends]
    return windows, truth[ends], pure, ends, starts


def _fit(method, train, *, n_init, seed):
    if method in ("w2", "w1", "shape_w2"):
        metric = "w1" if method == "w1" else "w2"
        fit_samples = standardize_windows(train) if method == "shape_w2" else train
        model = WassersteinKMeans(metric=metric, n_clusters=2, n_init=n_init,
                                  random_state=seed, max_iter=100).fit(fit_samples)
        return model, fit_samples
    model = FeatureBaseline(kind=method, n_clusters=2, n_init=n_init, random_state=seed).fit(train)
    return model, train


def _predict(model, method, windows):
    samples = standardize_windows(windows) if method == "shape_w2" else windows
    return model.predict(samples), model.transform(samples)


def _novelty_distances(model, method, windows):
    """Use the same empirical transport geometry for every model's novelty."""
    samples = standardize_windows(windows) if method == "shape_w2" else windows
    return pairwise_distance(samples, model.centers_, metric="w1" if method == "w1" else "w2")


def _metric_record(truth, labels):
    if not len(truth):
        return {"ari": None, "nmi": None, "balanced_accuracy": None, "windows": 0}
    recalls = [np.mean(labels[truth == state] == state) for state in np.unique(truth)]
    has_two_states = len(np.unique(truth)) > 1
    return {"ari": float(adjusted_rand_score(truth, labels)) if has_two_states else None,
            "nmi": float(normalized_mutual_info_score(truth, labels)) if has_two_states else None,
            "balanced_accuracy": float(np.mean(recalls)),
            "windows": int(len(truth))}


def _delays(ends, truth, mapped, *, test_start, stride):
    # The train/test boundary is purged; only the internal test switch can
    # measure first detection without that imposed gap.
    switch_points = [3 * _DWELL]
    delays = []
    censored = 0
    for switch in switch_points:
        if switch < test_start or switch >= 4 * _DWELL:
            continue
        eligible = np.flatnonzero((ends >= switch) & (ends < min(switch + _DWELL, 4 * _DWELL)))
        correct = eligible[mapped[eligible] == truth[eligible]]
        if len(correct):
            delays.append(int(ends[correct[0]] - switch))
        else:
            censored += 1
    return {"events": len(delays) + censored, "detected": len(delays),
            "censored": censored, "observed_delays": delays,
            "resolution_observations": stride}


def _summarize(values):
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    n = len(finite)
    if not n:
        return {"n": 0, "mean": None, "range": None, "standard_error": None}
    return {"n": n, "mean": float(np.mean(finite)),
            "range": [float(min(finite)), float(max(finite))],
            "standard_error": float(np.std(finite, ddof=1) / math.sqrt(n)) if n > 1 else None}


def _aggregate(records):
    output = {}
    for method in _METHODS:
        rows = [record[method] for record in records]
        output[method] = {
            section: {key: _summarize([row[section][key] for row in rows])
                      for key in ("ari", "nmi", "balanced_accuracy", "windows")}
            for section in ("pure", "mixed")
        }
        delays = [row["detection_delay"] for row in rows]
        observed = [value for row in delays for value in row["observed_delays"]]
        events = sum(row["events"] for row in delays)
        censored = sum(row["censored"] for row in delays)
        output[method]["detection_delay"] = {
            "events": events, "detected": events - censored, "censored": censored,
            "failure_rate": censored / events if events else None,
            "detected_only_delay": _summarize(observed),
            "resolution_observations": delays[0]["resolution_observations"],
        }
    return output


def _recover_once(suite, length, stride, n_init, seed):
    rng = np.random.default_rng(seed)
    values, truth = _sample_stream(suite, rng)
    windows, window_truth, pure, ends, starts = _window_rows(values, truth, length, stride)
    train = ends < 2 * _DWELL
    test = starts >= 2 * _DWELL
    if set(window_truth[train & pure]) != {0, 1}:
        raise RuntimeError("both states must appear in pure training windows")
    result = {}
    for method in _METHODS:
        model, _ = _fit(method, windows[train][::5], n_init=n_init, seed=seed)
        train_pred, _ = _predict(model, method, windows[train])
        mapping = label_mapping(window_truth[train], train_pred, k=2)
        test_pred, _ = _predict(model, method, windows[test])
        mapped = mapping[test_pred]
        test_truth, test_pure, test_ends = window_truth[test], pure[test], ends[test]
        result[method] = {
            "pure": _metric_record(test_truth[test_pure], mapped[test_pure]),
            "mixed": _metric_record(test_truth[~test_pure], mapped[~test_pure]),
            "detection_delay": _delays(test_ends, test_truth, mapped,
                                       test_start=2 * _DWELL, stride=stride),
        }
    return result


def _exact_result(seed, n_init):
    blocks, truth = exact_moment_blocks(seed=seed)
    moments = np.array([[np.mean(row ** power) for power in range(1, 5)] for row in blocks])
    gap = float(np.max(np.abs(moments - moments[0])))
    between = float(pairwise_distance(blocks[:1], blocks[1:2])[0, 0])
    methods = {}
    for method in ("w2", "w1", "volatility", "moments"):
        model, _ = _fit(method, blocks[:4], n_init=n_init, seed=seed)
        training_pred, _ = _predict(model, method, blocks[:4])
        mapping = label_mapping(truth[:4], training_pred, k=2)
        test_pred, _ = _predict(model, method, blocks[4:])
        mapped = mapping[test_pred]
        methods[method] = {"test_ari": float(adjusted_rand_score(truth[4:], mapped)),
                           "test_balanced_accuracy": float(np.mean(mapped == truth[4:])),
                           "effective_clusters": int(getattr(model, "n_effective_clusters_", 2))}
    return {"block_length": 300, "count_vectors": _COUNTS.tolist(),
            "max_first_four_moment_gap": gap, "w2_between_laws": between,
            "w2_test_ari": methods["w2"]["test_ari"],
            "moments_effective_clusters": methods["moments"]["effective_clusters"],
            "methods": methods,
            "scope": "exact disjoint blocks; rolling iid windows only match population moments"}


def _stationary_once(length, stride, n_init, seed):
    rng = np.random.default_rng(seed)
    values = rng.normal(0, .01, 4 * _DWELL)
    windows, _, _, ends, starts = _window_rows(values, np.zeros(len(values), int), length, stride)
    train = ends < 2 * _DWELL
    validation = (starts >= 2 * _DWELL) & (ends < 3 * _DWELL)
    test = starts >= 3 * _DWELL
    result = {}
    for method in _METHODS:
        model, _ = _fit(method, windows[train][::5], n_init=n_init, seed=seed)
        train_labels, _ = _predict(model, method, windows[train])
        calibration = _novelty_distances(model, method, windows[validation])
        labels, _ = _predict(model, method, windows[test])
        distances = _novelty_distances(model, method, windows[test])
        alternate, _ = _fit(method, windows[train][::5], n_init=n_init, seed=seed + 10000)
        alternate_labels, _ = _predict(alternate, method, windows[test])
        nearest_train = np.min(calibration, axis=1)
        nearest_test = np.min(distances, axis=1)
        threshold = float(np.quantile(nearest_train, .99))
        temporal = temporal_summary(labels, k=2)
        result[method] = {"forced_k": 2,
                          "seed_ari": float(adjusted_rand_score(labels, alternate_labels)),
                          "switch_frequency": temporal["switching_frequency"],
                          "mean_dwell_scored_windows": temporal["mean_dwell"],
                          "mean_dwell_observations_approx": temporal["mean_dwell"] * stride,
                          "calibration_segment": "heldout_validation",
                          "novelty_distance_metric": "empirical_w1" if method == "w1" else "empirical_w2",
                          "validation_windows": int(validation.sum()),
                          "novelty_false_flag_rate": float(np.mean(nearest_test > threshold)),
                          "centroid_w2_distance": float(pairwise_distance(model.centers_[:1], model.centers_[1:2])[0, 0]),
                          "train_occupancy": np.bincount(train_labels, minlength=2).tolist()}
    return result


def _stationary_aggregate(records):
    output = {}
    for method in _METHODS:
        rows = [row[method] for row in records]
        output[method] = {"forced_k": 2,
                          "seed_ari": _summarize([row["seed_ari"] for row in rows]),
                          **{key: _summarize([row[key] for row in rows]) for key in
                             ("switch_frequency", "mean_dwell_scored_windows",
                              "mean_dwell_observations_approx", "novelty_false_flag_rate",
                              "centroid_w2_distance")},
                          "calibration_segment": "heldout_validation",
                          "novelty_distance_metric": rows[0]["novelty_distance_metric"],
                          "validation_windows": rows[0]["validation_windows"]}
    return output


def run_synthetic(config) -> dict:
    """Run all frozen K=2 recovery and stationary-null suites; return plain JSON data."""
    repeats = int(config.get("synthetic_repeats", 10))
    lengths = [int(value) for value in config.get("window_lengths", [21, 63, 126, 252])]
    seed = int(config.get("seed", 42))
    n_init = int(config.get("synthetic_n_init", config.get("n_init", 5)))
    if repeats < 1 or not lengths or any(length < 2 or length > _DWELL for length in lengths) or n_init < 1:
        raise ValueError("invalid synthetic repeats, window lengths, or n_init")
    started = time.perf_counter()
    recovery = {}
    stationary = {}
    suites = ("variance", "normal_t5", "signed_exponential", "equal_four_moments")
    for suite in suites:
        recovery[suite] = {}
        for length in lengths:
            stride = max(1, length // 5)
            rows = [_recover_once(suite, length, stride, n_init, seed + 100000 * i + 1000 * suites.index(suite) + length)
                    for i in range(repeats)]
            recovery[suite][str(length)] = {"repetitions": repeats, "score_stride": stride,
                                            "fit_subsample_every_scored_window": 5,
                                            "dwell_observations": _DWELL, "methods": _aggregate(rows)}
    for length in lengths:
        stride = max(1, length // 5)
        rows = [_stationary_once(length, stride, n_init, seed + 100000 * i + length)
                for i in range(repeats)]
        stationary[str(length)] = {"repetitions": repeats, "score_stride": stride,
                                   "fit_subsample_every_scored_window": 5,
                                   "methods": _stationary_aggregate(rows)}
    return {"config": {"synthetic_repeats": repeats, "window_lengths": lengths,
                        "seed": seed, "synthetic_n_init": n_init,
                        "general_n_init_reference": int(config.get("n_init", 20)), "forced_k": 2},
            "recovery": recovery, "exact_moments": _exact_result(seed, n_init),
            "stationary": stationary,
            "elapsed_seconds": time.perf_counter() - started,
            "inference_note": "repetitions are independent streams; overlapping windows are not independent observations"}
