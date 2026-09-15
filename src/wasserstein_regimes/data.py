# SPDX-License-Identifier: GPL-3.0-only
"""Calendar-aware ingestion for daily adjusted equity prices.

The module performs no network access. Daily source dates are interpreted as
exchange session labels and returns become available one minute after the
actual scheduled close of their ending session.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import exchange_calendars as xcals
import numpy as np
import pandas as pd


def _session_index(values: Any, name: str) -> pd.DatetimeIndex:
    """Coerce date-like values to copied, UTC-naive session labels."""
    try:
        index = pd.DatetimeIndex(pd.to_datetime(values, errors="raise", utc=True))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Could not parse {name} as dates") from exc
    if index.hasnans:
        raise ValueError(f"Could not parse {name} as dates")
    return index.tz_convert("UTC").tz_localize(None).normalize().copy()


def _utc_index(values: Any, name: str) -> pd.DatetimeIndex:
    try:
        index = pd.DatetimeIndex(pd.to_datetime(values, errors="raise", utc=True))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Could not parse {name} as UTC datetimes") from exc
    if index.hasnans:
        raise ValueError(f"Could not parse {name} as UTC datetimes")
    return index.copy()


@dataclass(frozen=True)
class ReturnSeries:
    """A one-dimensional return series with its complete temporal metadata.

    ``returns``, ``dates``, ``available_at``, ``price_start`` and ``price_end``
    are required and must have equal length. NaN returns represent unavailable
    intervals; infinities are rejected. Provenance defaults support concise
    synthetic fixtures while production ingestion supplies all three values.
    """

    returns: np.ndarray
    dates: pd.DatetimeIndex
    available_at: pd.DatetimeIndex
    price_start: pd.DatetimeIndex
    price_end: pd.DatetimeIndex
    dataset_sha256: str = ""
    provider: str = "synthetic"
    return_basis: str = "provided_returns"
    quality: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        returns = np.array(self.returns, dtype=np.float64, copy=True)
        if returns.ndim != 1:
            raise ValueError("returns must be a one-dimensional array")
        if np.isinf(returns).any():
            raise ValueError("returns may contain NaN gaps but not infinity")

        dates = _session_index(self.dates, "dates")
        available_at = _utc_index(self.available_at, "available_at")
        price_start = _session_index(self.price_start, "price_start")
        price_end = _session_index(self.price_end, "price_end")
        lengths = {len(returns), len(dates), len(available_at), len(price_start), len(price_end)}
        if len(lengths) != 1:
            raise ValueError("ReturnSeries arrays and indexes must have equal length")
        if dates.has_duplicates or not dates.is_monotonic_increasing:
            raise ValueError("dates must be unique and chronologically sorted")
        if not dates.equals(price_end):
            raise ValueError("dates must equal price_end session labels")
        if len(returns) and not np.all(price_start < price_end):
            raise ValueError("every price_start must precede its price_end")
        if not isinstance(self.dataset_sha256, str):
            raise ValueError("dataset_sha256 must be a string")
        if not isinstance(self.provider, str) or not self.provider:
            raise ValueError("provider must be a nonempty string")
        if not isinstance(self.return_basis, str) or not self.return_basis:
            raise ValueError("return_basis must be a nonempty string")

        returns.setflags(write=False)
        quality = MappingProxyType(deepcopy(dict(self.quality)))
        object.__setattr__(self, "returns", returns)
        object.__setattr__(self, "dates", dates)
        object.__setattr__(self, "available_at", available_at)
        object.__setattr__(self, "price_start", price_start)
        object.__setattr__(self, "price_end", price_end)
        object.__setattr__(self, "quality", quality)


@lru_cache(maxsize=None)
def _calendar(name: str, start_year: int, end_year: int):
    try:
        return xcals.get_calendar(
            name,
            start=f"{start_year:04d}-01-01",
            end=f"{end_year:04d}-12-31",
        )
    except Exception as exc:
        raise ValueError(f"Unknown or unsupported exchange calendar {name!r}") from exc


def _frame_sha256(frame: pd.DataFrame, date_column: str, price_column: str) -> str:
    canonical = frame.loc[:, [date_column, price_column]].to_csv(index=False).encode("utf-8")
    return sha256(canonical).hexdigest()


def from_prices(
    frame: pd.DataFrame,
    *,
    date_column: str = "Date",
    price_column: str = "Adj Close",
    provider: str = "csv",
    return_basis: str = "adjusted_close",
    calendar: str = "XNYS",
    dataset_sha256: str | None = None,
) -> ReturnSeries:
    """Validate daily adjusted prices and construct close-to-close log returns."""
    if return_basis != "adjusted_close":
        raise ValueError("return_basis must be 'adjusted_close' for daily equity prices")
    missing_columns = [name for name in (date_column, price_column) if name not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    if frame.empty:
        raise ValueError("Price data must contain at least one row")

    dates = _session_index(frame[date_column], date_column)
    try:
        prices = pd.to_numeric(frame[price_column], errors="raise").to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{price_column} prices must be finite and strictly positive") from exc
    if not np.isfinite(prices).all() or np.any(prices <= 0):
        raise ValueError(f"{price_column} prices must be finite and strictly positive")

    duplicate_dates = dates[dates.duplicated(keep=False)].unique()
    if len(duplicate_dates):
        conflicting = any(
            not np.all(prices[dates == date] == prices[dates == date][0])
            for date in duplicate_dates
        )
        kind = "conflicting duplicate" if conflicting else "identical duplicate"
        labels = ", ".join(date.strftime("%Y-%m-%d") for date in duplicate_dates)
        raise ValueError(f"Found {kind} session date(s): {labels}")

    observed = pd.Series(prices, index=dates, dtype=np.float64).sort_index()
    start_year = min(1993, int(observed.index[0].year))
    end_year = max(2026, int(observed.index[-1].year))
    exchange = _calendar(calendar, start_year, end_year)
    sessions = exchange.sessions_in_range(observed.index[0], observed.index[-1])
    non_sessions = observed.index.difference(sessions)
    if len(non_sessions):
        labels = ", ".join(date.strftime("%Y-%m-%d") for date in non_sessions)
        raise ValueError(f"Date(s) are not an {calendar} session: {labels}")

    aligned_prices = observed.reindex(sessions).to_numpy(dtype=np.float64)
    returns = np.full(max(len(sessions) - 1, 0), np.nan, dtype=np.float64)
    valid = np.isfinite(aligned_prices[:-1]) & np.isfinite(aligned_prices[1:])
    returns[valid] = (
        np.log(aligned_prices[1:][valid]) - np.log(aligned_prices[:-1][valid])
    )

    closes = pd.DatetimeIndex(exchange.schedule.loc[sessions, "close"])
    available_at = closes[1:].tz_convert("UTC") + pd.Timedelta(minutes=1)
    quality = {
        "calendar": calendar,
        "observed_price_count": int(len(observed)),
        "expected_session_count": int(len(sessions)),
        "missing_session_count": int(np.isnan(aligned_prices).sum()),
        "invalid_return_count": int(np.isnan(returns).sum()),
        "assumed_availability_lag_minutes": 1,
        "point_in_time_verified": False,
    }
    fingerprint = dataset_sha256 or _frame_sha256(frame, date_column, price_column)
    return ReturnSeries(
        returns=returns,
        dates=sessions[1:],
        available_at=available_at,
        price_start=sessions[:-1],
        price_end=sessions[1:],
        dataset_sha256=fingerprint,
        provider=provider,
        return_basis=return_basis,
        quality=quality,
    )


def load_csv(
    path: str | Path,
    *,
    date_column: str = "Date",
    price_column: str = "Adj Close",
    provider: str = "csv",
    return_basis: str = "adjusted_close",
    calendar: str = "XNYS",
) -> ReturnSeries:
    """Load a local daily-price CSV without changing or acquiring source data."""
    payload = Path(path).read_bytes()
    frame = pd.read_csv(BytesIO(payload))
    return from_prices(
        frame,
        date_column=date_column,
        price_column=price_column,
        provider=provider,
        return_basis=return_basis,
        calendar=calendar,
        dataset_sha256=sha256(payload).hexdigest(),
    )


__all__ = ["ReturnSeries", "from_prices", "load_csv"]
