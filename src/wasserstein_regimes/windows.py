# SPDX-License-Identifier: GPL-3.0-only
"""Trailing return windows with exact input-price and availability metadata."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd

from .data import ReturnSeries, _session_index, _utc_index


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


@dataclass(frozen=True)
class WindowBatch:
    """Complete trailing windows and zero-based inclusive return endpoints.

    Every row uses return indices ``start_indices[i]`` through
    ``endpoints[i]``. Its input-price interval begins at the price preceding
    the first return and ends at the final return's ending price.
    """

    samples: np.ndarray
    endpoints: np.ndarray
    start_indices: np.ndarray
    dates: pd.DatetimeIndex
    available_at: pd.DatetimeIndex
    price_start: pd.DatetimeIndex
    price_end: pd.DatetimeIndex

    def __post_init__(self) -> None:
        samples = np.array(self.samples, dtype=np.float64, copy=True)
        endpoints = np.array(self.endpoints, dtype=np.int64, copy=True)
        start_indices = np.array(self.start_indices, dtype=np.int64, copy=True)
        if samples.ndim != 2 or samples.shape[1] < 1 or not np.isfinite(samples).all():
            raise ValueError("samples must be a finite two-dimensional array with positive width")
        if endpoints.ndim != 1 or start_indices.ndim != 1:
            raise ValueError("endpoints and start_indices must be one-dimensional")
        dates = _session_index(self.dates, "dates")
        available_at = _utc_index(self.available_at, "available_at")
        price_start = _session_index(self.price_start, "price_start")
        price_end = _session_index(self.price_end, "price_end")
        count = samples.shape[0]
        if any(len(field) != count for field in (
            endpoints, start_indices, dates, available_at, price_start, price_end,
        )):
            raise ValueError("WindowBatch rows and metadata must have equal length")
        if count and (
            np.any(start_indices < 0)
            or np.any(endpoints - start_indices + 1 != samples.shape[1])
            or np.any(np.diff(endpoints) <= 0)
        ):
            raise ValueError("window indices must be nonnegative, aligned and increasing")
        if not dates.equals(price_end):
            raise ValueError("dates must equal price_end session labels")
        if count and not np.all(price_start < price_end):
            raise ValueError("every price_start must precede its price_end")
        samples.setflags(write=False)
        endpoints.setflags(write=False)
        start_indices.setflags(write=False)
        for name, value in (
            ("samples", samples),
            ("endpoints", endpoints),
            ("start_indices", start_indices),
            ("dates", dates),
            ("available_at", available_at),
            ("price_start", price_start),
            ("price_end", price_end),
        ):
            object.__setattr__(self, name, value)

    def subset(self, mask: Any) -> WindowBatch:
        """Return copied rows selected by a same-length boolean mask."""
        selection = np.asarray(mask)
        if selection.ndim != 1 or selection.dtype != np.bool_ or len(selection) != len(self.samples):
            raise ValueError("mask must be a same-length one-dimensional boolean array")
        return WindowBatch(
            self.samples[selection],
            self.endpoints[selection],
            self.start_indices[selection],
            self.dates[selection],
            self.available_at[selection],
            self.price_start[selection],
            self.price_end[selection],
        )


def make_windows(series: ReturnSeries, length: int = 63, stride: int = 1) -> WindowBatch:
    """Build complete, finite trailing windows without crossing NaN gaps.

    Endpoint stride is anchored at the first endpoint with ``length`` returns
    (return index ``length - 1``), before invalid candidates are excluded.
    Rebuilding the same series at stride one yields the daily score set and
    preserves identical endpoint alignment with a coarser fit stride.
    """
    length = _positive_integer(length, "length")
    stride = _positive_integer(stride, "stride")
    count = len(series.returns)
    if count < length:
        empty = np.empty(0, dtype=np.int64)
        return WindowBatch(
            np.empty((0, length), dtype=np.float64),
            empty,
            empty,
            series.dates[:0],
            series.available_at[:0],
            series.price_start[:0],
            series.price_end[:0],
        )

    candidates = np.arange(length - 1, count, stride, dtype=np.int64)
    starts = candidates - length + 1
    all_samples = np.lib.stride_tricks.sliding_window_view(series.returns, length)
    selected_samples = all_samples[starts]
    valid = np.isfinite(selected_samples).all(axis=1)
    endpoints = candidates[valid]
    start_indices = starts[valid]
    return WindowBatch(
        selected_samples[valid],
        endpoints,
        start_indices,
        series.dates[endpoints],
        series.available_at[endpoints],
        series.price_start[start_indices],
        series.price_end[endpoints],
    )


__all__ = ["WindowBatch", "make_windows"]
