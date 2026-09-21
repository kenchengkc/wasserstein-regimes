# SPDX-License-Identifier: GPL-3.0-only

import hashlib

import numpy as np
import pandas as pd
import pytest

from wasserstein_regimes.data import ReturnSeries, from_prices, load_csv


def test_load_csv_uses_adjusted_prices_across_split_and_dividend_and_sorts_dates(tmp_path):
    csv_path = tmp_path / "spy.csv"
    csv_path.write_text(
        "Date,Close,Adj Close\n"
        "2024-01-05,51,51\n"
        "2024-01-02,100,50\n"
        "2024-01-03,50,50\n"
        "2024-01-04,49,50\n",
        encoding="utf-8",
    )

    series = load_csv(csv_path)

    np.testing.assert_allclose(series.returns, [0.0, 0.0, np.log(51 / 50)])
    assert series.dates.equals(pd.DatetimeIndex(["2024-01-03", "2024-01-04", "2024-01-05"]))
    assert series.price_start.equals(pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"]))
    assert series.price_end.equals(series.dates)
    assert series.provider == "csv"
    assert series.return_basis == "adjusted_close"
    assert series.dataset_sha256 == hashlib.sha256(csv_path.read_bytes()).hexdigest()


def test_missing_session_invalidates_both_adjacent_returns_but_holiday_does_not(tmp_path):
    csv_path = tmp_path / "gap.csv"
    csv_path.write_text(
        "Date,Adj Close\n"
        "2024-07-01,100\n"
        "2024-07-03,121\n"
        "2024-07-05,133.1\n",
        encoding="utf-8",
    )

    series = load_csv(csv_path)

    assert series.dates.equals(
        pd.DatetimeIndex(["2024-07-02", "2024-07-03", "2024-07-05"])
    )
    assert np.isnan(series.returns[:2]).all()
    assert series.returns[2] == pytest.approx(np.log(1.1))
    assert series.quality["missing_session_count"] == 1
    assert series.quality["invalid_return_count"] == 2


def test_availability_uses_actual_utc_closes_across_early_close_and_dst(tmp_path):
    csv_path = tmp_path / "calendar.csv"
    csv_path.write_text(
        "Date,Adj Close\n"
        "2024-03-08,100\n"
        "2024-03-11,101\n"
        "2024-07-02,102\n"
        "2024-07-03,103\n",
        encoding="utf-8",
    )

    series = load_csv(csv_path)

    by_date = dict(zip(series.dates, series.available_at))
    assert by_date[pd.Timestamp("2024-03-11")] == pd.Timestamp("2024-03-11 20:01Z")
    assert by_date[pd.Timestamp("2024-07-03")] == pd.Timestamp("2024-07-03 17:01Z")
    assert str(series.available_at.tz) == "UTC"


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (["2024-01-02,100", "2024-01-02,100"], "identical duplicate"),
        (["2024-01-02,100", "2024-01-02,101"], "conflicting duplicate"),
    ],
)
def test_duplicate_session_dates_are_rejected_clearly(tmp_path, rows, message):
    csv_path = tmp_path / "duplicate.csv"
    csv_path.write_text("Date,Adj Close\n" + "\n".join(rows) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_csv(csv_path)


@pytest.mark.parametrize("price", ["0", "-1", "nan", "inf"])
def test_nonpositive_or_nonfinite_prices_are_rejected(tmp_path, price):
    csv_path = tmp_path / "bad-price.csv"
    csv_path.write_text(
        f"Date,Adj Close\n2024-01-02,100\n2024-01-03,{price}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="finite and strictly positive"):
        load_csv(csv_path)


def test_missing_columns_bad_dates_non_sessions_and_wrong_basis_are_rejected(tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("when,value\n2024-07-04,100\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Missing required columns"):
        load_csv(csv_path)
    with pytest.raises(ValueError, match="not an XNYS session"):
        load_csv(csv_path, date_column="when", price_column="value")
    with pytest.raises(ValueError, match="adjusted_close"):
        load_csv(
            csv_path,
            date_column="when",
            price_column="value",
            return_basis="raw_close",
        )

    csv_path.write_text("Date,Adj Close\nnot-a-date,100\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Could not parse"):
        load_csv(csv_path)


def test_return_series_constructor_copies_synthetic_arrays_and_metadata():
    values = np.array([0.01, np.nan, -0.02])
    dates = ["2024-01-03", "2024-01-04", "2024-01-05"]
    available = ["2024-01-03 21:01Z", "2024-01-04 21:01Z", "2024-01-05 21:01Z"]
    starts = ["2024-01-02", "2024-01-03", "2024-01-04"]
    quality = {"fixture": True}

    series = ReturnSeries(values, dates, available, starts, dates, quality=quality)
    values[0] = 99
    quality["fixture"] = False

    assert series.returns[0] == 0.01
    assert series.quality["fixture"] is True
    assert series.dates.tz is None
    assert str(series.available_at.tz) == "UTC"
    with pytest.raises(ValueError):
        series.returns[0] = 1
    with pytest.raises(TypeError):
        series.quality["new"] = "value"


def test_future_price_perturbation_leaves_returns_through_cutoff_unchanged():
    dates = pd.bdate_range("2024-01-02", "2024-01-12")
    original = pd.DataFrame({"Date": dates, "Adj Close": np.arange(100, 109, dtype=float)})
    perturbed = original.copy()
    perturbed.loc[perturbed["Date"] > pd.Timestamp("2024-01-05"), "Adj Close"] *= 3

    before = from_prices(original)
    after = from_prices(perturbed)
    through_cutoff = before.dates <= pd.Timestamp("2024-01-05")

    np.testing.assert_array_equal(before.returns[through_cutoff], after.returns[through_cutoff])
    assert before.available_at[through_cutoff].equals(after.available_at[through_cutoff])


def test_finite_adjusted_prices_with_extreme_ratio_produce_finite_log_difference():
    frame = pd.DataFrame({
        "Date": ["2024-01-02", "2024-01-03"],
        "Adj Close": [1e-200, 1e200],
    })

    series = from_prices(frame)

    assert np.isfinite(series.returns[0])
    assert series.returns[0] == pytest.approx(np.log(1e200) - np.log(1e-200))
