# SPDX-License-Identifier: GPL-3.0-only

import numpy as np
import pandas as pd
import pytest

from wasserstein_regimes.data import ReturnSeries, load_csv
from wasserstein_regimes.windows import WindowBatch, make_windows


def _series(values):
    count = len(values)
    prices = pd.bdate_range("2024-01-02", periods=count + 1)
    dates = prices[1:]
    available = dates.tz_localize("UTC") + pd.Timedelta(hours=21, minutes=1)
    return ReturnSeries(values, dates, available, prices[:-1], dates)


def test_complete_windows_have_inclusive_endpoints_and_raw_price_intervals():
    series = _series(np.arange(1, 9, dtype=float))

    batch = make_windows(series, length=3, stride=2)

    np.testing.assert_array_equal(batch.samples, [[1, 2, 3], [3, 4, 5], [5, 6, 7]])
    np.testing.assert_array_equal(batch.start_indices, [0, 2, 4])
    np.testing.assert_array_equal(batch.endpoints, [2, 4, 6])
    assert batch.dates.equals(series.dates[[2, 4, 6]])
    assert batch.available_at.equals(series.available_at[[2, 4, 6]])
    assert batch.price_start.equals(series.price_start[[0, 2, 4]])
    assert batch.price_end.equals(series.price_end[[2, 4, 6]])


def test_fit_stride_uses_same_endpoint_anchor_as_daily_score_stride():
    series = _series(np.arange(1, 11, dtype=float))

    scores = make_windows(series, length=3, stride=1)
    fits = make_windows(series, length=3, stride=3)

    np.testing.assert_array_equal(scores.endpoints, np.arange(2, 10))
    np.testing.assert_array_equal(fits.endpoints, [2, 5, 8])
    assert all(endpoint in scores.endpoints for endpoint in fits.endpoints)


def test_gap_excludes_all_windows_crossing_invalid_returns(tmp_path):
    path = tmp_path / "gap.csv"
    path.write_text(
        "Date,Adj Close\n"
        "2024-07-01,100\n"
        "2024-07-03,101\n"
        "2024-07-05,102\n"
        "2024-07-08,103\n"
        "2024-07-09,104\n",
        encoding="utf-8",
    )
    series = load_csv(path)

    batch = make_windows(series, length=2)

    np.testing.assert_array_equal(batch.endpoints, [3, 4])
    assert batch.dates.equals(pd.DatetimeIndex(["2024-07-08", "2024-07-09"]))
    assert np.isfinite(batch.samples).all()
    assert batch.price_start[0] == pd.Timestamp("2024-07-03")


def test_incomplete_history_returns_empty_batch_with_requested_width():
    batch = make_windows(_series([0.01, 0.02]), length=3)

    assert batch.samples.shape == (0, 3)
    assert batch.endpoints.shape == batch.start_indices.shape == (0,)
    assert len(batch.dates) == len(batch.available_at) == 0


def test_subset_keeps_metadata_aligned_and_copies_immutable_boundaries():
    series = _series(np.arange(1, 7, dtype=float))
    batch = make_windows(series, length=2)
    subset = batch.subset(np.array([True, False, True, False, True]))

    np.testing.assert_array_equal(subset.endpoints, [1, 3, 5])
    assert subset.price_start.equals(series.price_start[[0, 2, 4]])
    with pytest.raises(ValueError):
        batch.samples[0, 0] = 99
    with pytest.raises(ValueError):
        subset.endpoints[0] = 99
    with pytest.raises(ValueError, match="mask"):
        batch.subset([True])


def test_strict_partition_purge_uses_preceding_price_not_just_return_dates():
    batch = make_windows(_series(np.arange(1, 9, dtype=float)), length=2)
    train = batch.subset(batch.endpoints <= 2)
    prior_price_end = train.price_end.max()
    test = batch.subset(batch.price_start > prior_price_end)

    assert train.price_end.max() == pd.Timestamp("2024-01-05")
    assert test.price_start.min() == pd.Timestamp("2024-01-08")
    np.testing.assert_array_equal(test.endpoints, [5, 6, 7])
    assert set(train.price_start).isdisjoint(test.price_start)


def test_future_return_perturbation_does_not_change_prior_window_or_availability():
    original = _series(np.arange(1, 9, dtype=float))
    revised = _series(np.array([1, 2, 3, 4, 50, 60, 70, 80], dtype=float))
    before = make_windows(original, length=3)
    after = make_windows(revised, length=3)
    cutoff = pd.Timestamp("2024-01-05")
    before_prior = before.subset(before.dates <= cutoff)
    after_prior = after.subset(after.dates <= cutoff)

    np.testing.assert_array_equal(before_prior.samples, after_prior.samples)
    assert before_prior.available_at.equals(after_prior.available_at)


@pytest.mark.parametrize("kwargs", [{"length": 0}, {"stride": 0}, {"length": 1.5}])
def test_invalid_window_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        make_windows(_series([0.01, 0.02]), **kwargs)
