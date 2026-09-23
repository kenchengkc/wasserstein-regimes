# SPDX-License-Identifier: GPL-3.0-only
import json

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from wasserstein_regimes.sliced import SlicedWassersteinKMedoids, sliced_w2
from wasserstein_regimes.transport import pairwise_distance


def control():
    a = np.array([[-1., -1.], [1., 1.]])
    b = np.array([[-1., 1.], [1., -1.]])
    return np.stack([a, b, a[::-1], b[::-1]])


def test_1d_exact_reduction():
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=(7, 9)), rng.normal(size=(3, 9))
    np.testing.assert_allclose(sliced_w2(x[..., None], y[..., None], [[1.], [-1.]]),
                               pairwise_distance(x, y), atol=1e-14)


def test_projection_assignment_oracle_and_dependence():
    x = control()
    directions = np.array([[1., 1.], [1., -1.]]) / np.sqrt(2)
    assert all(pairwise_distance(x[:1, :, d], x[1:2, :, d])[0, 0] == 0 for d in range(2))
    cost = 0.
    for theta in directions:
        a, b = x[0] @ theta, x[1] @ theta
        matrix = (a[:, None] - b[None, :]) ** 2
        i, j = linear_sum_assignment(matrix)
        cost += matrix[i, j].mean() / len(directions)
    assert sliced_w2(x[:1], x[1:2], directions)[0, 0] == pytest.approx(np.sqrt(cost))
    assert cost > 0
    np.testing.assert_allclose(sliced_w2(x, x, directions), sliced_w2(x[:, ::-1], x, directions))


def test_fit_observed_medoids_roundtrip_and_batching(tmp_path):
    x = control()
    model = SlicedWassersteinKMedoids(n_clusters=2, candidate_size=4, n_init=2, chunk_size=1).fit(x)
    assert model.inertia_ == pytest.approx(0)
    assert model.labels_[0] == model.labels_[2] != model.labels_[1]
    np.testing.assert_array_equal(model.medoids_, x[model.medoid_indices_])
    np.testing.assert_allclose(model.transform(x),
                               sliced_w2(x, model.medoids_, model.projections_, scales=model.scales_, chunk_size=4))
    second = SlicedWassersteinKMedoids(n_clusters=2, candidate_size=4, n_init=2, chunk_size=4).fit(x)
    np.testing.assert_array_equal(model.labels_, second.labels_)
    model.save(tmp_path/'model.npz')
    loaded = SlicedWassersteinKMedoids.load(tmp_path/'model.npz')
    np.testing.assert_array_equal(loaded.predict(x), model.predict(x))
    np.testing.assert_allclose(loaded.transform(x), model.transform(x))


def test_training_state_frozen_and_failed_refit_preserves_model():
    x = control()
    scales = np.array([1., 2.])
    model = SlicedWassersteinKMedoids(scales=scales, candidate_size=4).fit(x)
    before = model.transform(x)
    scales[:] = 100
    np.testing.assert_allclose(model.transform(x), before)
    with pytest.raises(ValueError, match='distinct'):
        model.fit(np.zeros_like(x))
    np.testing.assert_allclose(model.transform(x), before)
    for array in [model.projections_, model.scales_, model.medoids_]:
        assert not array.flags.writeable


@pytest.mark.parametrize('kwargs', [{'candidate_size':1}, {'n_clusters':0}, {'n_init':True},
                                    {'chunk_size':0}, {'random_state':-1}])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        SlicedWassersteinKMedoids(**kwargs)


@pytest.mark.parametrize('directions', [[[0., 0.]], [[2., 0.]], [[np.nan, 0.]], [[1., 0., 0.]]])
def test_invalid_projection(directions):
    with pytest.raises(ValueError):
        sliced_w2(control(), control(), directions)


def test_nonfinite_scaling_and_shape_errors():
    for scale in [[0, 1], [1], [1, np.inf]]:
        with pytest.raises(ValueError):
            SlicedWassersteinKMedoids(scales=scale).fit(control())
    m = SlicedWassersteinKMedoids().fit(control())
    with pytest.raises(ValueError):
        m.predict(np.ones((3, 3, 2)))
    with pytest.raises(ValueError):
        m.predict(np.full((1, 2, 2), np.nan))
    with pytest.raises(ValueError, match='fitted'):
        SlicedWassersteinKMedoids().predict(control())


def test_corrupt_archive_rejected(tmp_path):
    path = tmp_path/'model.npz'
    SlicedWassersteinKMedoids().fit(control()).save(path)
    with np.load(path, allow_pickle=False) as f:
        contents = {key:f[key].copy() for key in f.files}
    contents['projections'][0] = 0
    np.savez(path, **contents)
    with pytest.raises(ValueError):
        SlicedWassersteinKMedoids.load(path)


def test_candidate_memory_budget(monkeypatch):
    import wasserstein_regimes.sliced as module
    original = module._costs
    shapes = []
    def observed(x, y, chunk_size):
        shapes.append((len(x), len(y)))
        return original(x, y, chunk_size)
    monkeypatch.setattr(module, '_costs', observed)
    x = np.random.default_rng(7).normal(size=(200, 9, 2))
    m = SlicedWassersteinKMedoids(candidate_size=16, n_init=2, n_projections=8, chunk_size=7).fit(x)
    assert len(m.labels_) == 200
    assert all(a <= 16 and b <= 16 for a,b in shapes)
    assert m.inertia_ == pytest.approx(np.sum(m.transform(x).min(axis=1)**2))
    assert all(np.all(np.diff(run['objective_history']) <= 1e-12) for run in m.candidate_runs_)
