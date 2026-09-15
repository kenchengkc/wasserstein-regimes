# SPDX-License-Identifier: GPL-3.0-only
import json
import warnings

import numpy as np
import pytest

from wasserstein_regimes.baselines import CausalGaussianHMM, FeatureBaseline, feature_matrix


def test_moment_features_are_finite_for_constant_windows():
    features = feature_matrix([[2.0, 2.0, 2.0], [0.0, 1.0, 2.0]], kind="moments")

    np.testing.assert_allclose(features[0], [2.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(features[1], [1.0, np.sqrt(2.0 / 3.0), 0.0, -1.5])
    assert np.isfinite(features).all()


def test_feature_kinds_have_the_documented_columns():
    samples = np.array([[-2.0, -1.0, 1.0, 3.0], [0.0, 0.0, 1.0, 2.0]])

    np.testing.assert_allclose(feature_matrix(samples, "volatility")[:, 0], samples.std(axis=1))
    np.testing.assert_allclose(feature_matrix(samples, "mean_vol"), np.column_stack([
        samples.mean(axis=1), samples.std(axis=1),
    ]))
    rich = feature_matrix(samples, "rich")
    assert rich.shape == (2, 15)
    quantiles = np.quantile(samples, [.01, .05, .1, .25, .5, .75, .9, .95, .99], axis=1).T
    np.testing.assert_allclose(rich[:, 4:13], quantiles)
    np.testing.assert_allclose(rich[:, 13], np.sqrt(np.mean(np.minimum(samples, 0.0) ** 2, axis=1)))


def test_rich_drawdown_preserves_temporal_ordering():
    alternating = [0.1, -0.15, 0.1, -0.15]
    gains_then_losses = [0.1, 0.1, -0.15, -0.15]

    features = feature_matrix([alternating, gains_then_losses], "rich")

    np.testing.assert_allclose(features[0, :-1], features[1, :-1], atol=1e-14)
    assert features[0, -1] == pytest.approx(1.0 - np.exp(-0.2))
    assert features[1, -1] == pytest.approx(1.0 - np.exp(-0.3))


def test_feature_baseline_uses_training_scaler_and_distributional_prototypes():
    training = np.array([
        [-2.0, -1.0, 0.0, 1.0],
        [-1.8, -0.8, 0.2, 1.2],
        [3.0, 4.0, 5.0, 6.0],
        [3.2, 4.2, 5.2, 6.2],
    ])
    future = np.full((2, 4), 1e9)
    model = FeatureBaseline(kind="mean_vol", n_clusters=2, random_state=7, n_init=5).fit(training)
    scaler_mean = model.scaler_.mean_.copy()
    scaler_scale = model.scaler_.scale_.copy()
    predictions = model.predict(training)

    model.predict(future)

    np.testing.assert_array_equal(model.predict(training), predictions)
    np.testing.assert_array_equal(model.labels_, predictions)
    np.testing.assert_array_equal(model.scaler_.mean_, scaler_mean)
    np.testing.assert_array_equal(model.scaler_.scale_, scaler_scale)
    assert model.centers_.shape == (2, training.shape[1])
    for cluster in range(model.n_effective_clusters_):
        expected = np.sort(training[model.labels_ == cluster], axis=1).mean(axis=0)
        np.testing.assert_allclose(model.centers_[cluster], expected)


def test_equal_moment_distributions_collapse_without_fake_regimes_or_warnings():
    support = np.arange(6, dtype=float)
    perturbation = np.array([1, -5, 10, -10, 5, -1])
    a = np.repeat(support, 50 + 3 * perturbation)
    b = np.repeat(support, 50 - 3 * perturbation)
    samples = np.stack([a, b, a[::-1], b[::-1]])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = FeatureBaseline(kind="moments", n_clusters=2).fit(samples)

    assert model.n_effective_clusters_ == 1
    np.testing.assert_array_equal(model.labels_, np.zeros(4, dtype=int))
    assert model.centers_.shape == (1, len(a))
    assert caught == []


@pytest.mark.parametrize("kind", ["volatility", "mean_vol", "moments", "rich", "gmm"])
def test_feature_baseline_rejects_invalid_or_too_short_samples(kind):
    with pytest.raises(ValueError):
        FeatureBaseline(kind=kind, n_clusters=2).fit([[1.0, np.nan], [2.0, 3.0]])
    with pytest.raises(ValueError):
        FeatureBaseline(kind=kind, n_clusters=2).fit([[1.0, 2.0]])


def test_feature_configuration_and_unfitted_use_are_rejected():
    with pytest.raises(ValueError):
        feature_matrix([[1.0, 2.0]], "unknown")
    with pytest.raises(ValueError):
        FeatureBaseline(kind="unknown")
    with pytest.raises(ValueError):
        FeatureBaseline(n_clusters=0)
    with pytest.raises(RuntimeError):
        FeatureBaseline().predict([[1.0, 2.0]])


def _known_hmm():
    model = CausalGaussianHMM(n_clusters=2)
    model.train_mean_ = 0.0
    model.train_std_ = 1.0
    model.startprob_ = np.array([0.6, 0.4])
    model.transmat_ = np.array([[0.7, 0.3], [0.2, 0.8]])
    model.means_ = np.array([-1.0, 1.0])
    model.covars_ = np.array([1.0, 1.0])
    model.converged_ = True
    model.n_iter_ = 3
    model.reached_iteration_limit_ = False
    model.training_loglikelihood_ = -4.0
    model.fit_diagnostics_ = []
    return model


def test_hmm_forward_filter_matches_hand_calculation():
    model = _known_hmm()

    probabilities = model.filter_proba([0.0, 1.0])

    np.testing.assert_allclose(probabilities[0], [0.6, 0.4])
    np.testing.assert_allclose(probabilities[1], [0.11920292202211755, 0.8807970779778824])
    np.testing.assert_array_equal(model.predict_filtered([0.0, 1.0]), [0, 1])


def test_hmm_explicit_initial_continues_after_previous_posterior():
    model = _known_hmm()

    probabilities = model.filter_proba([0.0], initial=[0.25, 0.75])

    np.testing.assert_allclose(probabilities[0], [0.325, 0.675])


def test_hmm_filtering_is_prefix_invariant_and_continuable():
    rng = np.random.default_rng(12)
    training = np.r_[rng.normal(-1.0, .25, 80), rng.normal(1.0, .25, 80)]
    evaluation = np.array([-1.2, -0.8, 0.2, 1.1, 0.9])
    model = CausalGaussianHMM(n_clusters=2, random_state=9, n_init=3, max_iter=100).fit(training)

    full = model.filter_proba(evaluation)
    prefix = model.filter_proba(evaluation[:3])
    continuation = model.filter_proba(evaluation[3:], initial=prefix[-1])

    np.testing.assert_allclose(prefix, full[:3])
    np.testing.assert_allclose(continuation, full[3:])
    np.testing.assert_allclose(model.filter_proba(evaluation), full)


def test_hmm_fit_records_training_state_without_future_mutation():
    rng = np.random.default_rng(21)
    training = np.r_[rng.normal(-.5, .2, 60), rng.normal(.7, .3, 60)]
    model = CausalGaussianHMM(n_clusters=2, random_state=4, n_init=2, max_iter=100).fit(training)
    state = json.dumps(model.to_dict(), sort_keys=True)
    training_probabilities = model.filter_proba(training)

    model.filter_proba([1e9, -1e9], initial=model.terminal_proba_)

    assert json.dumps(model.to_dict(), sort_keys=True) == state
    np.testing.assert_allclose(model.filter_proba(training), training_probabilities)
    np.testing.assert_allclose(model.terminal_proba_, training_probabilities[-1])
    assert np.isfinite(model.training_loglikelihood_)
    assert model.n_iter_ <= model.max_iter
    assert model.reached_iteration_limit_ == (model.n_iter_ >= model.max_iter)


def test_hmm_to_dict_contains_json_safe_learned_parameters():
    model = _known_hmm()
    model.terminal_proba_ = model.filter_proba([0.0])[-1]

    artifact = model.to_dict()

    assert {"mu", "cov", "trans", "start", "train_mean", "train_std"} <= artifact.keys()
    assert json.loads(json.dumps(artifact))["trans"] == [[0.7, 0.3], [0.2, 0.8]]


@pytest.mark.parametrize("bad", [[], [1.0], [1.0, np.nan], [[1.0, 2.0]]])
def test_hmm_fit_rejects_invalid_or_too_short_returns(bad):
    with pytest.raises(ValueError):
        CausalGaussianHMM(n_clusters=2).fit(bad)


def test_hmm_filter_rejects_invalid_values_and_initial_state():
    model = _known_hmm()
    with pytest.raises(ValueError):
        model.filter_proba([np.inf])
    with pytest.raises(ValueError):
        model.filter_proba([0.0], initial=[1.0])
    with pytest.raises(ValueError):
        model.filter_proba([0.0], initial=[-1.0, 2.0])
    with pytest.raises(RuntimeError):
        CausalGaussianHMM(n_clusters=2).filter_proba([0.0])


def test_hmm_configuration_is_validated():
    with pytest.raises(ValueError):
        CausalGaussianHMM(n_clusters=0)
    with pytest.raises(ValueError):
        CausalGaussianHMM(n_init=0)
    with pytest.raises(ValueError):
        CausalGaussianHMM(max_iter=0)


def test_hmm_rejects_constant_training_returns_without_inventing_states():
    with pytest.raises(ValueError):
        CausalGaussianHMM(n_clusters=2).fit(np.zeros(20))


def test_hmm_rejects_sequences_too_short_to_fit_free_parameters():
    with pytest.raises(ValueError):
        CausalGaussianHMM(n_clusters=2).fit(np.arange(6, dtype=float))


def test_hmm_iteration_limit_and_fit_history_are_reported():
    rng = np.random.default_rng(99)
    training = np.r_[rng.normal(-1.0, .3, 60), rng.normal(1.0, .3, 60)]
    model = CausalGaussianHMM(n_clusters=2, n_init=2, max_iter=1).fit(training)

    assert not model.converged_
    assert model.reached_iteration_limit_
    assert model.n_iter_ == 1
    assert len(model.fit_diagnostics_) == 2
    for diagnostic in model.fit_diagnostics_:
        assert len(diagnostic["loglikelihood_history"]) == diagnostic["n_iter"]
