# SPDX-License-Identifier: GPL-3.0-only
"""Return-level stress controls. No fitting or market-data access in this module."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import adjusted_rand_score

from .evaluation import label_mapping, temporal_summary
from .transport import _positive_int


def _vectors(values):
    if np.iscomplexobj(values):
        raise ValueError('vectors must be real')
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or 0 in x.shape or not np.isfinite(x).all():
        raise ValueError('expected finite nonempty daily vectors')
    return x


def return_windows(vectors, length, stride=1):
    """Windows wholly inside the supplied segment; shape (windows, returns, assets)."""
    x = _vectors(vectors)
    _positive_int(length, 'length')
    _positive_int(stride, 'stride')
    if len(x) < length:
        raise ValueError('segment shorter than window')
    return np.lib.stride_tricks.sliding_window_view(x, length, axis=0).transpose(0, 2, 1)[::stride]


def stationary_indices(n, mean_block, rng):
    """Politis–Romano circular stationary bootstrap; seams include wrap/restarts."""
    _positive_int(n, 'n')
    if isinstance(mean_block, bool) or not np.isfinite(mean_block) or mean_block < 1:
        raise ValueError('mean block length must be finite and at least one')
    indices = np.empty(n, int)
    seams = np.zeros(n, bool)
    indices[0], seams[0] = rng.integers(n), True
    for i in range(1, n):
        restart = rng.random() < 1 / mean_block
        indices[i] = rng.integers(n) if restart else (indices[i-1] + 1) % n
        seams[i] = restart or indices[i] == 0
    return indices, seams


def contaminate(vectors, fraction, magnitude, rng):
    x = _vectors(vectors)
    if (isinstance(fraction, bool) or not np.isfinite(fraction) or not 0 < fraction <= 1
            or isinstance(magnitude, bool) or not np.isfinite(magnitude) or magnitude <= 0):
        raise ValueError('invalid contamination fraction or magnitude')
    count = max(1, int(np.ceil(len(x) * fraction)))
    indices = rng.choice(len(x), count, replace=False)
    out = x.copy()
    out[indices] += magnitude * x.std(axis=0) * rng.choice([-1., 1.], size=(count, x.shape[1]))
    return out, count


def control_stream(kind, seed):
    """Fixed 3,000-vector, five-asset protocol; rho is population innovation rho."""
    rng = np.random.default_rng(seed)
    n, d = 3000, 5
    if kind in ('gaussian_null', 'student_null'):
        rho = np.full(n, .45)
        innovations = np.sqrt(.45)*rng.normal(size=(n+512, 1)) + np.sqrt(.55)*rng.normal(size=(n+512, d))
        if kind == 'student_null':
            # Shared radial mixing gives multivariate t5, with unit marginal variance.
            innovations *= np.sqrt(3 / rng.chisquare(5, size=(n+512, 1)))
        values = np.empty_like(innovations)
        state = np.zeros(d)
        for i, innovation in enumerate(innovations):
            state = .25 * state + np.sqrt(1 - .25**2) * innovation
            values[i] = state
        return values[512:], rho
    rho = np.full(n, .2)
    if kind == 'rare':
        for start in (950, 1800, 2550):
            rho[start:start+126] = .8
    elif kind == 'gradual':
        for start, end in ((0, 1500), (1500, 2250), (2250, 3000)):
            mid = (start+end)//2
            rho[mid-125:mid+125] = np.linspace(.2, .8, 250)
            rho[mid+125:end] = .8
    else:
        raise ValueError('unknown synthetic control')
    values = np.sqrt(rho[:, None])*rng.normal(size=(n, 1)) + np.sqrt(1-rho[:, None])*rng.normal(size=(n, d))
    return values, rho


def pure_truth(rho, length):
    windows = np.lib.stride_tricks.sliding_window_view(rho, length)
    return np.where(np.all(windows == .2, axis=1), 0,
                    np.where(np.all(windows == .8, axis=1), 1, -1))


def partition_diagnostics(labels, k, reference=None):
    labels = np.asarray(labels)
    if (labels.ndim != 1 or not len(labels) or not np.issubdtype(labels.dtype, np.integer)
            or np.any(labels < 0) or np.any(labels >= k)):
        raise ValueError('invalid partition labels')
    counts = np.bincount(labels, minlength=k)
    occupied = int(np.count_nonzero(counts))
    result = dict(n=len(labels), counts=counts.tolist(), occupancy=(counts/len(labels)).tolist(),
                  occupied=occupied, single_state=occupied == 1, temporal=temporal_summary(labels, k=k))
    if reference is not None:
        other = partition_diagnostics(reference, k)
        if len(reference) != len(labels):
            raise ValueError('reference must use identical anchors')
        result.update(ari=float(adjusted_rand_score(reference, labels)), reference_counts=other['counts'],
                      both_single_state=other['single_state'] and result['single_state'])
    return result


def recovery_metrics(train_truth, train_labels, test_truth, test_labels):
    pure_train, pure_test = train_truth >= 0, test_truth >= 0
    if set(train_truth[pure_train]) != {0, 1} or set(test_truth[pure_test]) != {0, 1}:
        raise ValueError('both pure states required for recovery metrics')
    mapping = label_mapping(train_truth[pure_train], train_labels[pure_train], k=2)
    mapped = mapping[test_labels]
    recall = [float(np.mean(mapped[test_truth == state] == state)) for state in (0, 1)]
    return dict(mapping=mapping.tolist(), recall=recall, balanced_accuracy=float(np.mean(recall)),
                pure_ari=float(adjusted_rand_score(test_truth[pure_test], test_labels[pure_test])),
                pure_counts=np.bincount(test_truth[pure_test], minlength=2).tolist(),
                mixed_windows=int((~pure_test).sum()))


def rare_delay(endpoints, mapped):
    """First sustained high-state run; explicit censoring at exclusive episode end."""
    start, end = 2550, 2676
    selected = (endpoints >= start) & (endpoints < end)
    dates, high = endpoints[selected], np.asarray(mapped)[selected] == 1
    for i in range(max(0, len(dates)-4)):
        if high[i:i+5].all() and np.all(np.diff(dates[i:i+5]) == 1):
            return dict(censored=False, delay=int(dates[i]-start), confirmation_delay=int(dates[i+4]-start),
                        episode_start=start, episode_end=end)
    return dict(censored=True, delay=None, confirmation_delay=None, episode_start=start, episode_end=end)
