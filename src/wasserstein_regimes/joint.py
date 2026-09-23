# SPDX-License-Identifier: GPL-3.0-only
"""Strictly aligned joint return windows; never fill or join implicitly."""
from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np
import pandas as pd

from .data import ReturnSeries, _session_index, _utc_index
from .windows import _positive_integer


@dataclass(frozen=True)
class JointWindowBatch:
    samples: np.ndarray
    symbols: tuple[str, ...]
    dates: pd.DatetimeIndex
    available_at: pd.DatetimeIndex
    price_start: pd.DatetimeIndex
    price_end: pd.DatetimeIndex

    def __post_init__(self):
        if np.iscomplexobj(self.samples):
            raise ValueError("joint samples must be real")
        x = np.array(self.samples, dtype=np.float64, copy=True)
        symbols = tuple(self.symbols)
        if (x.ndim != 3 or x.shape[1] < 1 or x.shape[2] != len(symbols)
                or not symbols or any(not isinstance(s, str) or not s for s in symbols)
                or len(set(symbols)) != len(symbols) or not np.isfinite(x).all()):
            raise ValueError("invalid joint samples or symbols")
        dates = _session_index(self.dates, "dates").copy(deep=True)
        start = _session_index(self.price_start, "price_start").copy(deep=True)
        end = _session_index(self.price_end, "price_end").copy(deep=True)
        ready = _utc_index(self.available_at, "available_at").copy(deep=True)
        if any(len(field) != len(x) for field in (dates, ready, start, end)):
            raise ValueError("joint metadata length mismatch")
        if not dates.equals(end) or dates.has_duplicates or not dates.is_monotonic_increasing:
            raise ValueError("joint dates must be unique, increasing and equal price_end")
        if len(x) and (not np.all(start < end) or not np.all(ready >= end.tz_localize("UTC"))):
            raise ValueError("invalid price intervals or observation availability")
        x.setflags(write=False)
        for name, value in dict(samples=x, symbols=symbols, dates=dates, price_start=start,
                                price_end=end, available_at=ready).items():
            object.__setattr__(self, name, value)


def joint_windows(series: Mapping[str, ReturnSeries], length=63, stride=1):
    """Create (window, atom, asset) samples in mapping insertion order.

    Inputs must already share exact sessions, price intervals and return basis.
    Any NaN component or discontinuous price chain excludes its entire window.
    Availability includes all atoms, even if an earlier observation arrived late.
    """
    length = _positive_integer(length, "length")
    stride = _positive_integer(stride, "stride")
    if not isinstance(series, Mapping) or not series:
        raise ValueError("series must be a nonempty symbol mapping")
    if any(not isinstance(symbol, str) or not symbol for symbol in series):
        raise ValueError("symbols must be nonempty strings")
    if any(not isinstance(value, ReturnSeries) for value in series.values()):
        raise ValueError("values must be ReturnSeries")
    first = next(iter(series.values()))
    for symbol, value in series.items():
        if not value.dates.equals(first.dates):
            raise ValueError(f"{symbol}: sessions must match exactly; align explicitly")
        if not value.price_start.equals(first.price_start) or not value.price_end.equals(first.price_end):
            raise ValueError(f"{symbol}: input-price intervals differ")
        if value.return_basis != first.return_basis:
            raise ValueError(f"{symbol}: return basis differs")
    values = np.column_stack([value.returns for value in series.values()])
    availability = np.column_stack([value.available_at.as_unit("ns").asi8 for value in series.values()]).max(axis=1)
    endpoints, starts, windows, ready = [], [], [], []
    for end in range(length - 1, len(values), stride):
        start = end - length + 1
        x = values[start:end+1]
        if not np.isfinite(x).all():
            continue
        if not first.price_start[start+1:end+1].equals(first.price_end[start:end]):
            continue
        windows.append(x)
        endpoints.append(end)
        starts.append(start)
        ready.append(availability[start:end+1].max())
    ends = np.asarray(endpoints, dtype=int)
    begins = np.asarray(starts, dtype=int)
    samples = np.asarray(windows) if windows else np.empty((0, length, len(series)))
    return JointWindowBatch(samples, tuple(series), first.dates[ends],
                            pd.to_datetime(ready, unit="ns", utc=True),
                            first.price_start[begins], first.price_end[ends])
