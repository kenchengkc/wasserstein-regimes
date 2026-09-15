# SPDX-License-Identifier: GPL-3.0-only
import json

import numpy as np
import pytest
from sklearn.cluster import KMeans

from wasserstein_regimes.clustering import WassersteinKMeans
from wasserstein_regimes.transport import pairwise_distance


def separated_windows(seed=7):
    rng = np.random.default_rng(seed)
    return np.vstack(
        [rng.normal(-3, 0.2, (25, 8)), rng.normal(2, 0.3, (25, 8))]
    )


@pytest.mark.parametrize("metric,power", [("w1", 1), ("w2", 2)])
def test_fit_is_deterministic_consistent_and_uses_metric_objective(metric, power):
    samples = separated_windows()
    first = WassersteinKMeans(metric=metric, n_clusters=2, n_init=7, random_state=13).fit(samples)
    second = WassersteinKMeans(metric=metric, n_clusters=2, n_init=7, random_state=13).fit(samples)
    assert first.converged_
    assert first.distance_metric == metric
    assert first.objective_power == power
    np.testing.assert_array_equal(first.labels_, second.labels_)
    np.testing.assert_allclose(first.centers_, second.centers_)
    np.testing.assert_array_equal(first.labels_, first.predict(samples))
    assert first.inertia_ == pytest.approx(np.sum(first.transform(samples).min(axis=1) ** power))
    assert np.all(np.diff(first.centers_, axis=1) >= 0)
    assert first.labels_[0] != first.labels_[-1]
    assert set(first.labels_[:25]) == {first.labels_[0]}
    assert set(first.labels_[25:]) == {first.labels_[-1]}
    assert np.all(np.diff(first.objective_history_) <= 1e-10)


def test_w2_explicit_initialization_matches_sklearn_quantile_oracle():
    samples = separated_windows(21)
    quantiles = np.sort(samples, axis=1)
    initial = quantiles[[2, 37]].copy()
    expected = KMeans(
        n_clusters=2,
        init=initial,
        n_init=1,
        max_iter=100,
        tol=1e-12,
        algorithm="lloyd",
        random_state=0,
    ).fit(quantiles)
    actual = WassersteinKMeans(
        metric="w2",
        n_clusters=2,
        initialization=initial,
        n_init=99,
        max_iter=100,
        tol=1e-12,
        random_state=0,
    ).fit(samples)
    np.testing.assert_array_equal(actual.labels_, expected.labels_)
    np.testing.assert_allclose(actual.centers_, expected.cluster_centers_, atol=1e-12)
    assert actual.inertia_ == pytest.approx(expected.inertia_ / samples.shape[1])


@pytest.mark.parametrize("metric", ["w1", "w2"])
def test_single_cluster_barycenter_is_locally_optimal(metric):
    samples = np.array([[0, 2, 5], [1, 4, 8], [2, 6, 11]], dtype=float)
    model = WassersteinKMeans(metric=metric, n_clusters=1, n_init=1).fit(samples)
    center = model.centers_[0]
    perturbation = np.array([-0.01, 0.01, 0.01])
    alternate_cost = np.sum(
        pairwise_distance(samples, (center + perturbation)[None], metric=metric)[:, 0]
        ** model.objective_power
    )
    assert model.inertia_ <= alternate_cost + 1e-14
    expected = np.median(np.sort(samples, axis=1), axis=0) if metric == "w1" else np.mean(np.sort(samples, axis=1), axis=0)
    np.testing.assert_allclose(center, expected)


def test_fit_predict_and_transform_sort_inputs_without_standardizing():
    samples = np.array([[3, 1, 2], [2, 3, 1], [11, 10, 12], [12, 11, 10]], dtype=float)
    model = WassersteinKMeans(metric="w2", n_clusters=2, random_state=3).fit(samples)
    np.testing.assert_array_equal(model.fit_predict(samples), model.labels_)
    np.testing.assert_array_equal(model.predict(samples[:, ::-1]), model.labels_)
    assert model.transform(samples).shape == (4, 2)
    np.testing.assert_allclose(np.sort(model.centers_.mean(axis=1)), [2, 11])


def test_model_copies_explicit_initialization_and_freezes_results():
    samples = separated_windows()
    initial = np.sort(samples[[0, -1]], axis=1)
    model = WassersteinKMeans(n_clusters=2, initialization=initial)
    initial[:] = 999
    model.fit(samples)
    assert np.max(np.abs(model.centers_)) < 10
    assert not model.centers_.flags.writeable
    assert not model.labels_.flags.writeable


def test_save_load_round_trip_and_rejects_tampering(tmp_path):
    samples = separated_windows()
    model = WassersteinKMeans(metric="w2", n_clusters=2, random_state=8).fit(samples)
    base = tmp_path / "regime_model"
    model.save(base)
    loaded = WassersteinKMeans.load(base)
    np.testing.assert_array_equal(loaded.predict(samples), model.predict(samples))
    np.testing.assert_allclose(loaded.transform(samples), model.transform(samples))
    assert loaded.inertia_ == model.inertia_
    assert not loaded.centers_.flags.writeable

    metadata_path = base.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    metadata["distance_metric"] = "unknown"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="metadata"):
        WassersteinKMeans.load(base)


def test_load_rejects_unsorted_or_nonfinite_centers(tmp_path):
    samples = separated_windows()
    base = tmp_path / "regime_model"
    WassersteinKMeans(n_clusters=2).fit(samples).save(base)
    arrays_path = base.with_suffix(".npz")
    with np.load(arrays_path, allow_pickle=False) as archive:
        labels = archive["labels"].copy()
        history = archive["objective_history"].copy()
    np.savez(arrays_path, centers=np.array([[2.0, 1.0], [3.0, np.nan]]), labels=labels, objective_history=history)
    with pytest.raises(ValueError, match="centers"):
        WassersteinKMeans.load(base)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"metric": "bad"},
        {"n_clusters": 0},
        {"n_clusters": True},
        {"n_init": 0},
        {"max_iter": 0},
        {"tol": -1},
        {"chunk_size": 0},
        {"initialization": "bad"},
        {"random_state": 1.2},
    ],
)
def test_constructor_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        WassersteinKMeans(**kwargs)


def test_fit_rejects_bad_data_and_duplicate_shortfall():
    with pytest.raises(ValueError):
        WassersteinKMeans(n_clusters=2).fit([[1, 2], [2, 1]])
    with pytest.raises(ValueError):
        WassersteinKMeans(n_clusters=2).fit([[1, np.nan], [2, 3]])
    with pytest.raises(ValueError):
        WassersteinKMeans(n_clusters=2, initialization=np.ones((2, 3))).fit([[1, 2], [2, 3]])
    with pytest.raises(ValueError, match="distinct"):
        WassersteinKMeans(n_clusters=2, initialization=[[2, 1], [1, 2]])


def test_empty_initial_clusters_are_reseeded_from_observed_distributions():
    samples = np.array([[0.0, 1.0], [0.1, 1.1], [10, 11], [10.1, 11.1], [20, 21], [20.1, 21.1]])
    model = WassersteinKMeans(
        metric="w2",
        n_clusters=3,
        initialization=[[0, 1], [100, 101], [200, 201]],
        max_iter=30,
    ).fit(samples)
    assert model.converged_
    assert set(model.labels_) == {0, 1, 2}
    np.testing.assert_array_equal(model.labels_, model.predict(samples))
    assert model.inertia_ < 1


def test_negative_seed_rejected_at_construction():
    with pytest.raises(ValueError, match="random_state"):
        WassersteinKMeans(random_state=-1)


def test_corrupted_archive_is_rejected_as_model_error(tmp_path):
    base = tmp_path / "regime_model"
    WassersteinKMeans(n_clusters=1).fit([[1, 2], [2, 3]]).save(base)
    base.with_suffix(".npz").write_bytes(b"not a zip archive")
    with pytest.raises(ValueError, match="model"):
        WassersteinKMeans.load(base)


def test_load_rejects_inconsistent_metadata_and_labels(tmp_path):
    base = tmp_path / "regime_model"
    samples = separated_windows()
    WassersteinKMeans(n_clusters=2).fit(samples).save(base)
    metadata_path = base.with_suffix(".json")
    original = json.loads(metadata_path.read_text())
    metadata = dict(original)
    metadata["objective_power"] = 1
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="metadata"):
        WassersteinKMeans.load(base)

    metadata_path.write_text(json.dumps(original))
    arrays_path = base.with_suffix(".npz")
    with np.load(arrays_path, allow_pickle=False) as archive:
        centers = archive["centers"].copy()
        history = archive["objective_history"].copy()
        labels = archive["labels"].copy()
    labels[0] = 2
    np.savez(arrays_path, centers=centers, labels=labels, objective_history=history)
    with pytest.raises(ValueError, match="labels"):
        WassersteinKMeans.load(base)


def test_unfitted_and_wrong_dimension_calls_fail():
    model = WassersteinKMeans(n_clusters=1)
    with pytest.raises(RuntimeError):
        model.predict([[1, 2]])
    model.fit([[1, 2], [2, 3]])
    with pytest.raises(ValueError):
        model.transform([[1, 2, 3]])
