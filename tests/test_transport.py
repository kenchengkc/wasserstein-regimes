# SPDX-License-Identifier: GPL-3.0-only
from itertools import permutations

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from wasserstein_regimes.transport import (
    pairwise_distance,
    standardize_windows,
    w2_decomposition,
)


@st.composite
def equal_length_samples(draw):
    length = draw(st.integers(min_value=1, max_value=7))
    values = st.lists(
        st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        min_size=length,
        max_size=length,
    )
    return np.array(draw(values)), np.array(draw(values))


@settings(max_examples=50, deadline=None)
@given(
    sample_pair=equal_length_samples(),
    shift=st.floats(min_value=-10, max_value=10),
    scale=st.floats(min_value=0, max_value=10),
    metric=st.sampled_from(["w1", "w2"]),
)
def test_metric_invariances_and_scaling(sample_pair, shift, scale, metric):
    x, y = sample_pair
    xy = np.stack([x, y])
    reverse = np.stack([x[::-1], y[::-1]])
    distance = pairwise_distance(xy[:1], xy[1:], metric=metric)[0, 0]
    assert pairwise_distance(xy[:1], xy[:1], metric=metric)[0, 0] == pytest.approx(0, abs=1e-12)
    assert pairwise_distance(xy[1:], xy[:1], metric=metric)[0, 0] == pytest.approx(distance)
    assert pairwise_distance(reverse[:1], reverse[1:], metric=metric)[0, 0] == pytest.approx(distance)
    assert pairwise_distance(xy[:1] + shift, xy[1:] + shift, metric=metric)[0, 0] == pytest.approx(distance)
    assert pairwise_distance(xy[:1] * scale, xy[1:] * scale, metric=metric)[0, 0] == pytest.approx(scale * distance)


@settings(max_examples=50, deadline=None)
@given(
    windows=st.lists(
        st.lists(
            st.floats(min_value=-50, max_value=50, allow_nan=False, allow_infinity=False),
            min_size=4,
            max_size=4,
        ),
        min_size=3,
        max_size=3,
    ),
    metric=st.sampled_from(["w1", "w2"]),
)
def test_triangle_inequality(windows, metric):
    x, y, z = np.asarray(windows, dtype=float)
    d_xy = pairwise_distance(x[None], y[None], metric=metric)[0, 0]
    d_yz = pairwise_distance(y[None], z[None], metric=metric)[0, 0]
    d_xz = pairwise_distance(x[None], z[None], metric=metric)[0, 0]
    assert d_xz <= d_xy + d_yz + 1e-10


@settings(max_examples=30, deadline=None)
@given(equal_length_samples())
def test_w2_matches_brute_force_assignment(sample_pair):
    x, y = sample_pair
    expected_squared = min(
        np.mean((np.asarray(x) - permutation) ** 2)
        for permutation in permutations(y)
    )
    actual = pairwise_distance([x], [y], metric="w2")[0, 0]
    assert actual**2 == pytest.approx(expected_squared, rel=1e-10, abs=1e-10)


def test_pairwise_known_values_and_chunk_equivalence():
    samples = np.array([[0, 2], [4, 0], [-1, 3]], dtype=float)
    centers = np.array([[1, 3], [2, 0]], dtype=float)
    np.testing.assert_allclose(
        pairwise_distance(samples, centers, metric="w1", chunk_size=1),
        [[1, 0], [1, 1], [1, 1]],
    )
    expected_w2 = np.array([[1, 0], [1, np.sqrt(2)], [np.sqrt(2), 1]])
    np.testing.assert_allclose(
        pairwise_distance(samples, centers, metric="w2", chunk_size=2),
        expected_w2,
    )
    np.testing.assert_allclose(
        pairwise_distance(samples, centers, metric="w2", chunk_size=99),
        expected_w2,
    )


def test_w2_decomposition_has_exact_interpretable_components():
    shift = w2_decomposition([[0, 2]], [[2, 4]])
    assert {name: value[0, 0] for name, value in shift.items()} == pytest.approx(
        {"location": 4, "scale": 0, "shape": 0, "total": 4}
    )

    scale = w2_decomposition([[-1, 1]], [[-2, 2]])
    assert {name: value[0, 0] for name, value in scale.items()} == pytest.approx(
        {"location": 0, "scale": 1, "shape": 0, "total": 1}
    )

    shape = w2_decomposition(
        [[-1, -1, 1, 1]],
        [[-np.sqrt(2), 0, 0, np.sqrt(2)]],
    )
    assert shape["location"][0, 0] == pytest.approx(0, abs=1e-14)
    assert shape["scale"][0, 0] == pytest.approx(0, abs=1e-14)
    assert shape["shape"][0, 0] > 0
    assert shape["total"][0, 0] == pytest.approx(shape["shape"][0, 0])


def test_decomposition_identity_nonnegative_and_constants():
    rng = np.random.default_rng(99)
    samples = np.vstack([rng.normal(size=(9, 17)), np.full((1, 17), 3.0)])
    centers = np.vstack([rng.normal(size=(4, 17)), np.full((1, 17), -2.0)])
    parts = w2_decomposition(samples, centers, chunk_size=3)
    component_sum = parts["location"] + parts["scale"] + parts["shape"]
    np.testing.assert_allclose(parts["total"], component_sum, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        parts["total"],
        pairwise_distance(samples, centers, metric="w2", chunk_size=2) ** 2,
        rtol=1e-12,
        atol=1e-12,
    )
    assert all(np.all(value >= 0) for value in parts.values())
    assert parts["shape"][-1, -1] == 0
    assert parts["total"][-1, -1] == pytest.approx(25)


def test_standardization_preserves_order_and_maps_constants_to_zero():
    windows = np.array([[3, 1, 2], [7, 7, 7]], dtype=float)
    standardized = standardize_windows(windows)
    np.testing.assert_allclose(standardized[0], [np.sqrt(1.5), -np.sqrt(1.5), 0])
    np.testing.assert_array_equal(standardized[1], [0, 0, 0])
    np.testing.assert_allclose(standardized.mean(axis=1), 0, atol=1e-15)
    np.testing.assert_allclose(standardized.std(axis=1), [1, 0])


@pytest.mark.parametrize(
    "call",
    [
        lambda: pairwise_distance([], [[1]], metric="w2"),
        lambda: pairwise_distance([[1, 2]], [[1]], metric="w2"),
        lambda: pairwise_distance([[1]], [[np.nan]], metric="w2"),
        lambda: pairwise_distance([[1]], [[1]], metric="bad"),
        lambda: pairwise_distance([[1]], [[1]], chunk_size=0),
        lambda: w2_decomposition([[1]], [[1]], chunk_size=True),
        lambda: standardize_windows([1, 2]),
    ],
)
def test_transport_rejects_malformed_inputs(call):
    with pytest.raises(ValueError):
        call()
