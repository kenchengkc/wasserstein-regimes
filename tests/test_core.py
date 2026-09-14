# SPDX-License-Identifier: GPL-3.0-only
from itertools import permutations

import numpy as np
import pytest
from scipy.stats import wasserstein_distance

from wasserstein_regimes import barycenter, fit, rolling_windows, wasserstein


def test_w1_matches_independent_scipy_oracle():
    rng = np.random.default_rng(7)
    for _ in range(20):
        x, y = rng.normal(size=(2, 31))
        assert wasserstein(x, y, p=1) == pytest.approx(wasserstein_distance(x, y))


def test_w2_matches_brute_force_optimal_assignment():
    x, y = np.array([5, -2, 0, 1]), np.array([7, 3, -3, 1])
    optimal_cost = min(np.mean((x - permutation) ** 2) for permutation in permutations(y))
    assert wasserstein(x, y, p=2) ** 2 == pytest.approx(optimal_cost)


@pytest.mark.parametrize("p", [1, 2])
def test_metric_geometry(p):
    x, y = np.array([-3., 0, 2]), np.array([-1., 1, 4])
    assert wasserstein(x, x[::-1], p=p) == 0
    assert wasserstein(x + 10, y + 10, p=p) == pytest.approx(wasserstein(x, y, p=p))
    assert wasserstein(-3 * x, -3 * y, p=p) == pytest.approx(3 * wasserstein(x, y, p=p))


def test_centroids_use_the_correct_objective():
    samples = [[0, 1], [0, 1], [9, 10]]
    np.testing.assert_array_equal(barycenter(samples, p=1), [0, 1])
    np.testing.assert_array_equal(barycenter(samples, p=2), [3, 4])


def test_window_length_stride_and_endpoints():
    windows, ends = rolling_windows(np.arange(49), length=35, stride=7)
    assert windows.shape == (3, 35)
    np.testing.assert_array_equal(ends, [34, 41, 48])
    for window, end in zip(windows, ends):
        np.testing.assert_array_equal(window, np.arange(end - 34, end + 1))
    assert rolling_windows([1, 2], length=3)[0].shape == (0, 3)


@pytest.mark.parametrize("p", [1, 2])
def test_fit_is_reproducible_consistent_and_descending(p):
    rng = np.random.default_rng(10)
    samples = np.concatenate([rng.normal(-2, .1, (30, 15)), rng.normal(2, .1, (30, 15))])
    result = fit(samples, p=p, seed=9)
    again = fit(samples, p=p, seed=9)
    assert result.converged
    assert np.all(np.diff(result.objective_history) <= 1e-12)
    assert np.all(np.diff(result.centers, axis=1) >= 0)
    np.testing.assert_array_equal(result.labels, again.labels)
    np.testing.assert_array_equal(result.labels, result.predict(samples))
    assert result.objective == pytest.approx(np.sum(result.transform(samples).min(axis=1) ** p))
    assert len(np.unique(result.labels[:30])) == len(np.unique(result.labels[30:])) == 1
    assert result.labels[0] != result.labels[-1]
    with pytest.raises(ValueError):
        result.predict([[1, 2]])


def test_full_distributions_distinguish_equal_first_four_moments():
    support = np.arange(6, dtype=float)
    perturbation = np.array([1, -5, 10, -10, 5, -1])
    a = np.repeat(support, 50 + 3 * perturbation)
    b = np.repeat(support, 50 - 3 * perturbation)
    for order in range(1, 5):
        assert np.mean(a ** order) == pytest.approx(np.mean(b ** order))
    assert wasserstein(a, b, p=1) > 0
    assert wasserstein(a, b, p=2) > 0


@pytest.mark.parametrize("p", [1, 2])
def test_single_cluster_and_duplicates(p):
    samples = np.array([[3, 1, 2], [1, 2, 3]])
    result = fit(samples, n_clusters=1, p=p)
    assert result.converged and result.objective == 0
    with pytest.raises(ValueError):
        fit(samples, n_clusters=2, p=p)


@pytest.mark.parametrize("bad", [[], [[np.nan]], [[np.inf]], [1, 2]])
def test_invalid_atoms_rejected(bad):
    with pytest.raises(ValueError):
        fit(bad)


@pytest.mark.parametrize("kwargs", [{"p": 3}, {"n_clusters": 0}, {"n_init": 0}, {"max_iter": 0}, {"n_clusters": 1.5}])
def test_invalid_fit_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        fit([[1, 2], [2, 3]], **kwargs)


def test_invalid_distance_and_window_inputs():
    with pytest.raises(ValueError):
        wasserstein([1], [1, 2])
    with pytest.raises(ValueError):
        rolling_windows([1, np.nan], length=2)
    with pytest.raises(ValueError):
        rolling_windows([1, 2], length=2, stride=0)
