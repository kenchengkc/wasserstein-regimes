# SPDX-License-Identifier: GPL-3.0-only
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from wasserstein_regimes.data import ReturnSeries
from wasserstein_regimes.joint import joint_windows


def series(values):
    dates = pd.bdate_range('2024-01-02', periods=len(values)+1)
    return ReturnSeries(values, dates[1:], dates[1:].tz_localize('UTC') + pd.Timedelta(hours=21),
                        dates[:-1], dates[1:])


def test_joint_alignment_gaps_and_availability():
    a = series(np.arange(8.))
    b = replace(a, returns=np.array([0., 1., 2., np.nan, 4., 5., 6., 7.]),
                available_at=a.available_at + pd.Timedelta(minutes=10))
    result = joint_windows({'B':b, 'A':a}, length=3)
    assert result.symbols == ('B', 'A')
    assert result.samples.shape == (3, 3, 2)
    assert result.dates.equals(a.dates[[2, 6, 7]])
    assert result.available_at.equals(b.available_at[[2, 6, 7]])
    assert result.price_start.equals(a.price_start[[0, 4, 5]])
    assert not result.samples.flags.writeable


def test_different_input_intervals_are_rejected():
    a = series(np.arange(6.))
    b = replace(a, price_start=a.price_start - pd.Timedelta(days=1))
    with pytest.raises(ValueError, match='interval'):
        joint_windows({'A':a, 'B':b}, length=3)


def test_calendar_and_basis_mismatch_are_rejected():
    a = series(np.arange(6.))
    for b in [series(np.arange(5.)), replace(a, return_basis='other')]:
        with pytest.raises(ValueError):
            joint_windows({'A':a, 'B':b})


def test_future_changes_do_not_affect_previous_windows():
    a = series(np.arange(8.))
    b = replace(a, returns=np.array([0., 1., 2., 3., 4., 100., 200., 300.]))
    first = joint_windows({'A':a, 'B':a}, length=3)
    later = joint_windows({'A':a, 'B':b}, length=3)
    np.testing.assert_array_equal(first.samples[:3], later.samples[:3])
    assert first.available_at[:3].equals(later.available_at[:3])
    assert first.price_start[:3].equals(later.price_start[:3])


def test_empty_and_invalid_inputs():
    with pytest.raises(ValueError):
        joint_windows({})
    a = series(np.arange(2.))
    result = joint_windows({'A':a}, length=3)
    assert result.samples.shape == (0, 3, 1)
    with pytest.raises(ValueError):
        joint_windows({'A':a}, stride=0)


def test_late_earlier_observation_controls_window_availability():
    a = series(np.arange(4.))
    times = list(a.available_at)
    times[0] = times[-1] + pd.Timedelta(days=3)
    b = replace(a, available_at=pd.DatetimeIndex(times))
    result = joint_windows({'A':a, 'B':b}, length=3)
    assert result.available_at[0] == times[0]
    assert result.available_at[1] == times[-1]


def test_shared_broken_chain_excludes_window():
    a = series(np.arange(5.))
    starts = list(a.price_start)
    starts[2] = starts[2] - pd.Timedelta(days=1)
    a = replace(a, price_start=pd.DatetimeIndex(starts))
    result = joint_windows({'A':a}, length=3)
    assert result.dates.equals(a.dates[[4]])


def test_batch_constructor_validates_and_freezes_metadata():
    from wasserstein_regimes.joint import JointWindowBatch
    good = joint_windows({'A':series(np.arange(5.))}, length=3)
    symbols = ['A']
    copy = replace(good, symbols=symbols)
    symbols.append('B')
    assert copy.symbols == ('A',)
    with pytest.raises(ValueError):
        replace(good, available_at=pd.DatetimeIndex([pd.NaT]*3))
    with pytest.raises(ValueError):
        replace(good, price_start=good.price_end)
    with pytest.raises(ValueError):
        replace(good, dates=pd.DatetimeIndex([good.dates[0]]*3))
