import importlib
import importlib.util

import numpy as np
import pytest


def evaluation():
    assert importlib.util.find_spec('wasserstein_regimes.evaluation') is not None
    return importlib.import_module('wasserstein_regimes.evaluation')


def test_mmd_is_scalar_sample_permutation_invariant_and_matches_dirac_formula():
    e = evaluation()
    assert e.mmd2([0.], [1.], bandwidth=1.) == pytest.approx(2 * (1 - np.exp(-.5)))
    assert e.mmd2([0, 1, 2], [2, 0, 1], bandwidth=.4) == pytest.approx(0., abs=1e-14)


def test_future_outcomes_exclude_the_assignment_return_and_missing_horizons():
    e = evaluation()
    result = e.future_outcomes(np.array([9., .1, -.2, .3]), np.array([0, 1, 2]), horizon=2)
    np.testing.assert_allclose(result['forward_log_return'][:2], [-.1, .1])
    assert result['forward_drawdown'][0] == pytest.approx(1 - np.exp(-.2))
    assert np.isnan(result['forward_log_return'][2])


def test_novelty_uses_only_fixed_calibration_and_margin_is_bounded():
    e = evaluation()
    result = e.novelty(np.array([[1.5, 3], [9, 10]]), np.array([1., 2., 3.]), threshold=.95)
    np.testing.assert_allclose(result['novelty_percentile'], [1/3, 1])
    np.testing.assert_array_equal(result['ood'], [False, True])
    np.testing.assert_allclose(result['margin'], [.5, .1])


def test_temporal_summary_does_not_connect_disjoint_folds():
    e = evaluation()
    result = e.temporal_summary(np.array([0, 0, 1, 1]), np.array([1, 2, 10, 11]), k=2)
    assert result['switching_frequency'] == 0
    assert result['mean_dwell'] == 2
    assert result['transitions'] == [[1, 0], [0, 1]]


def test_training_mapping_does_not_use_test_labels():
    e = evaluation()
    mapping = e.label_mapping([0, 0, 1, 1], [1, 1, 0, 0], k=2)
    np.testing.assert_array_equal(mapping, [1, 0])


def test_moving_block_indices_preserve_adjacent_runs_and_are_reproducible():
    e = evaluation()
    a = e.moving_block_indices(20, block_length=4, rng=np.random.default_rng(2))
    b = e.moving_block_indices(20, block_length=4, rng=np.random.default_rng(2))
    np.testing.assert_array_equal(a, b)
    assert len(a) == 20 and a.min() >= 0 and a.max() < 20
    for start in range(0, 20, 4):
        np.testing.assert_array_equal(np.diff(a[start:start + 4]), np.ones(3))


def test_centroid_alignment_is_minimum_cost_bijection():
    e = evaluation()
    import numpy as np
    centers = np.array([[0.,0.], [2.,2.], [5.,5.]])
    assert e.align_centroids(centers, centers[[2,0,1]]).tolist() == [2,0,1]


def test_component_fractions_are_ratio_of_sums_not_mean_of_ratios():
    e = evaluation()
    result = e.component_fractions({'location':np.array([1.,0.]), 'scale':np.array([0.,9.]), 'shape':np.array([0.,0.])})
    assert result == {'location':.1, 'scale':.9, 'shape':0.}
