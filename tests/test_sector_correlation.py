"""Tests for data/sector_correlation.py."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


def _make_price_df(n: int = 60, start_price: float = 100.0) -> pd.DataFrame:
    """Return a minimal OHLCV DataFrame with n rows of daily close data."""
    idx = pd.bdate_range("2024-01-01", periods=n)
    prices = start_price + np.arange(n, dtype=float) * 0.1
    return pd.DataFrame(
        {
            "Open": prices,
            "High": prices + 0.5,
            "Low": prices - 0.5,
            "Close": prices,
            "Volume": [1_000_000] * n,
        },
        index=idx,
    )


class TestComputeStockSectorCorr(unittest.TestCase):
    def test_happy_path_returns_float_in_range(self):
        from data.sector_correlation import compute_stock_sector_corr

        price_data = {
            "AAPL": _make_price_df(60, 100.0),
            "XLK": _make_price_df(60, 50.0),
        }
        result = compute_stock_sector_corr("AAPL", "XLK", price_data=price_data)
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result, -1.0)
        self.assertLessEqual(result, 1.0)

    def test_insufficient_data_returns_none(self):
        from data.sector_correlation import compute_stock_sector_corr

        # Only 10 rows — window+2 = 22 rows needed
        price_data = {
            "AAPL": _make_price_df(10),
            "XLK": _make_price_df(10),
        }
        result = compute_stock_sector_corr("AAPL", "XLK", price_data=price_data)
        self.assertIsNone(result)

    def test_stock_not_in_preloaded_falls_back_to_yfinance(self):
        """ETF in price_data but stock missing → downloads stock via yfinance."""
        from data.sector_correlation import compute_stock_sector_corr

        price_data = {"XLK": _make_price_df(60)}
        stock_df = _make_price_df(60, 120.0)

        with patch("data.sector_correlation.yf.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = stock_df
            result = compute_stock_sector_corr("AAPL", "XLK", price_data=price_data)

        self.assertIsNotNone(result)

    def test_etf_not_in_preloaded_falls_back_to_yfinance(self):
        """Stock in price_data but ETF missing → downloads ETF via yfinance."""
        from data.sector_correlation import compute_stock_sector_corr

        price_data = {"AAPL": _make_price_df(60)}
        etf_df = _make_price_df(60, 50.0)

        with patch("data.sector_correlation.yf.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = etf_df
            result = compute_stock_sector_corr("AAPL", "XLK", price_data=price_data)

        self.assertIsNotNone(result)

    def test_yfinance_download_fails_returns_none(self):
        """When yfinance raises for a missing ticker, result is None."""
        from data.sector_correlation import compute_stock_sector_corr

        with patch("data.sector_correlation.yf.Ticker") as mock_ticker:
            mock_ticker.return_value.history.side_effect = RuntimeError("network error")
            result = compute_stock_sector_corr("AAPL", "XLK", price_data={})

        self.assertIsNone(result)

    def test_all_nan_close_returns_none(self):
        """When Close is all NaN after dropna, result is None."""
        from data.sector_correlation import compute_stock_sector_corr

        nan_df = pd.DataFrame({"Close": [float("nan")] * 60})
        price_data = {"AAPL": nan_df, "XLK": nan_df}
        result = compute_stock_sector_corr("AAPL", "XLK", price_data=price_data)
        self.assertIsNone(result)

    def test_result_is_rounded_to_4dp(self):
        from data.sector_correlation import compute_stock_sector_corr

        price_data = {
            "AAPL": _make_price_df(60, 100.0),
            "XLK": _make_price_df(60, 50.0),
        }
        result = compute_stock_sector_corr("AAPL", "XLK", price_data=price_data)
        self.assertIsNotNone(result)
        # Should be rounded to 4 decimal places
        self.assertEqual(result, round(result, 4))

    def test_last_corr_is_nan_returns_none(self):
        """If rolling corr produces NaN on the final row (e.g. identical series → std=0
        after a single pct_change value), result is None."""
        from data.sector_correlation import compute_stock_sector_corr

        # Flat price series → pct_change = 0 for all rows → corr is NaN (0/0)
        flat = _make_price_df(30)
        flat["Close"] = 100.0
        price_data = {"AAPL": flat, "XLK": flat.copy()}
        result = compute_stock_sector_corr("AAPL", "XLK", price_data=price_data)
        self.assertIsNone(result)


class TestGetDf(unittest.TestCase):
    def test_returns_from_preloaded_when_present(self):
        from data.sector_correlation import _get_df

        df = _make_price_df(30)
        result = _get_df("AAPL", {"AAPL": df})
        self.assertIs(result, df)

    def test_downloads_when_not_in_preloaded(self):
        from data.sector_correlation import _get_df

        df = _make_price_df(30)
        with patch("data.sector_correlation.yf.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = df
            result = _get_df("AAPL", {})

        self.assertIs(result, df)

    def test_returns_none_when_yfinance_empty(self):
        from data.sector_correlation import _get_df

        with patch("data.sector_correlation.yf.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = pd.DataFrame()
            result = _get_df("AAPL", {})

        self.assertIsNone(result)

    def test_returns_none_when_yfinance_raises(self):
        from data.sector_correlation import _get_df

        with patch("data.sector_correlation.yf.Ticker") as mock_ticker:
            mock_ticker.return_value.history.side_effect = ConnectionError("timeout")
            result = _get_df("AAPL", {})

        self.assertIsNone(result)

    def test_returns_none_when_price_data_is_none_and_yfinance_raises(self):
        from data.sector_correlation import _get_df

        with patch("data.sector_correlation.yf.Ticker") as mock_ticker:
            mock_ticker.return_value.history.side_effect = OSError("dns failure")
            result = _get_df("AAPL", None)

        self.assertIsNone(result)
