# SPDX-License-Identifier: GPL-3.0-only
import numpy as np
import pytest


def test_stationary_indices_reproduce_pairs_and_circular_continuation():
    from wasserstein_regimes.robustness import stationary_indices
    idx, seams = stationary_indices(1000, 50, np.random.default_rng(7))
    again, _ = stationary_indices(1000, 50, np.random.default_rng(7))
    np.testing.assert_array_equal(idx, again)
    assert len(idx) == 1000 and idx.min() >= 0 and idx.max() < 1000
    assert seams[0] and 5 < seams.sum() < 60
    assert np.all(np.diff(idx)[~seams[1:]] == 1)
    x = np.column_stack([np.arange(1000), -np.arange(1000)])
    assert np.all(x[idx].sum(axis=1) == 0)
    _, iid_seams = stationary_indices(100, 1, np.random.default_rng(3))
    assert iid_seams.all()
    for n, length in [(0, 3), (10, 0), (10, np.nan), (True, 3)]:
        with pytest.raises(ValueError): stationary_indices(n, length, np.random.default_rng(0))


def test_windows_preserve_joint_rows_and_split_segments():
    from wasserstein_regimes.robustness import return_windows
    x = np.arange(40).reshape(20, 2)
    np.testing.assert_array_equal(return_windows(x, 4, 2), np.stack([x[i:i+4] for i in range(0, 17, 2)]))
    train, test = return_windows(x[:10], 4), return_windows(x[10:], 4)
    assert train.max() < test.min()
    for bad in [np.ones((2, 2)), np.full((5, 2), np.nan), np.ones(10)]:
        with pytest.raises(ValueError): return_windows(bad, 4)


def test_contamination_does_not_mutate_input_and_uses_training_scale():
    from wasserstein_regimes.robustness import contaminate
    x = np.random.default_rng(4).normal(size=(100, 3))
    original = x.copy()
    y, count = contaminate(x, .02, 10, np.random.default_rng(1))
    np.testing.assert_array_equal(x, original)
    assert count == 2
    changed = np.any(y != x, axis=1)
    assert changed.sum() == 2
    np.testing.assert_allclose(np.abs(y[changed] - x[changed]), np.tile(10*x.std(axis=0), (2, 1)))
    for fraction in [-.1, 1.1, np.nan, True]:
        with pytest.raises(ValueError): contaminate(x, fraction, 10, np.random.default_rng(1))


@pytest.mark.parametrize('kind', ['gaussian_null', 'student_null', 'rare', 'gradual'])
def test_controls_fixed_parameters_reproducible_and_pure_truth(kind):
    from wasserstein_regimes.robustness import control_stream, pure_truth
    x, rho = control_stream(kind, 17)
    xx, rr = control_stream(kind, 17)
    np.testing.assert_array_equal(x, xx)
    np.testing.assert_array_equal(rho, rr)
    assert x.shape == (3000, 5) and np.isfinite(x).all()
    if kind.endswith('null'):
        assert np.all(rho == .45)
    elif kind == 'rare':
        assert np.count_nonzero(rho == .8) == 3*126
        truth = pure_truth(rho[2250:], 63)
        assert (truth == 1).sum() == 64
        assert (truth == -1).sum() == 124
    else:
        truth = pure_truth(rho[2250:], 63)
        assert set(truth) == {-1, 0, 1}
        assert (truth == -1).sum() > 250


def test_diagnostics_flag_collapsed_agreement():
    from wasserstein_regimes.robustness import partition_diagnostics
    result = partition_diagnostics(np.ones(20, int), 3, np.zeros(20, int))
    assert result['counts'] == [0, 20, 0]
    assert result['ari'] == 1. and result['both_single_state']
    assert result['reference_counts'] == [20, 0, 0]
    assert result['single_state'] and result['temporal']['switching_frequency'] == 0


def test_recovery_maps_only_from_training_and_censors_rare_misses():
    from wasserstein_regimes.robustness import recovery_metrics, rare_delay
    result = recovery_metrics(np.array([0, 0, 1, 1]), np.array([1, 1, 0, 0]),
                              np.array([0, 1, -1]), np.array([0, 1, 0]))
    assert result['recall'] == [0., 0.]
    assert result['balanced_accuracy'] == 0 and result['mixed_windows'] == 1
    assert result['mapping'] == [1, 0]
    # Test truth must not reverse the deliberately incorrect predictions.
    endpoints = np.arange(2500, 2700)
    assert rare_delay(endpoints, np.zeros(len(endpoints), int))['censored']
    labels = ((endpoints >= 2560) & (endpoints < 2676)).astype(int)
    result = rare_delay(endpoints, labels)
    assert result == dict(censored=False, delay=10, confirmation_delay=14, episode_start=2550, episode_end=2676)
