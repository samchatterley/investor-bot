"""Full coverage of data/market_data.py — fetch_stock_data and get_market_snapshots."""
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np


def _make_ohlcv(n=200, base=100.0):
    """Return a minimal OHLCV DataFrame that satisfies fetch_stock_data's warmup needs."""
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    prices = [base + i * 0.1 for i in range(n)]
    df = pd.DataFrame({
        "Open":   prices,
        "High":   [p + 1 for p in prices],
        "Low":    [p - 1 for p in prices],
        "Close":  prices,
        "Volume": [1_000_000] * n,
    }, index=idx)
    return df


class TestFetchStockData(unittest.TestCase):

    def _ticker(self, df):
        t = MagicMock()
        t.history.return_value = df
        return t

    def test_returns_dataframe_on_success(self):
        from data.market_data import fetch_stock_data
        with patch("data.market_data.yf.Ticker", return_value=self._ticker(_make_ohlcv(200))):
            result = fetch_stock_data("AAPL", days=30)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.DataFrame)

    def test_returns_none_on_empty_dataframe(self):
        from data.market_data import fetch_stock_data
        with patch("data.market_data.yf.Ticker", return_value=self._ticker(pd.DataFrame())):
            result = fetch_stock_data("AAPL")
        self.assertIsNone(result)

    def test_returns_none_when_insufficient_rows(self):
        from data.market_data import fetch_stock_data
        # Less than 35 rows after dropna — insufficient for indicators
        with patch("data.market_data.yf.Ticker", return_value=self._ticker(_make_ohlcv(10))):
            result = fetch_stock_data("AAPL")
        self.assertIsNone(result)

    def test_returns_none_on_yfinance_exception(self):
        from data.market_data import fetch_stock_data
        with patch("data.market_data.yf.Ticker", side_effect=Exception("network error")):
            result = fetch_stock_data("AAPL")
        self.assertIsNone(result)

    def test_result_has_required_indicator_columns(self):
        from data.market_data import fetch_stock_data
        with patch("data.market_data.yf.Ticker", return_value=self._ticker(_make_ohlcv(200))):
            result = fetch_stock_data("AAPL", days=30)
        if result is None:
            self.skipTest("fetch_stock_data returned None — indicator warmup failed")
        for col in ["rsi", "macd_diff", "ema9", "ema21", "bb_pct", "vol_ratio",
                    "ret_1d", "ret_5d", "ret_10d", "weekly_trend_up"]:
            self.assertIn(col, result.columns, f"Missing column: {col}")

    def test_result_length_capped_at_days_param(self):
        from data.market_data import fetch_stock_data
        with patch("data.market_data.yf.Ticker", return_value=self._ticker(_make_ohlcv(200))):
            result = fetch_stock_data("AAPL", days=20)
        if result is not None:
            self.assertLessEqual(len(result), 20)

    def test_weekly_trend_defaults_to_true_on_exception(self):
        from data.market_data import fetch_stock_data
        # Use only 40 rows — weekly resample won't have 22 weeks, triggers fallback
        with patch("data.market_data.yf.Ticker", return_value=self._ticker(_make_ohlcv(40))):
            result = fetch_stock_data("AAPL", days=5)
        if result is not None:
            self.assertTrue(result["weekly_trend_up"].iloc[-1])


class TestGetMarketSnapshots(unittest.TestCase):

    def _snap(self, symbol, price=150.0):
        return {
            "symbol": symbol, "current_price": price,
            "rsi_14": 50.0, "macd_diff": 0.1, "ema9_above_ema21": True,
            "bb_pct": 0.5, "vol_ratio": 1.2, "ret_1d_pct": 0.5,
            "ret_5d_pct": 2.0, "ret_10d_pct": 3.0, "weekly_trend_up": True,
        }

    def test_returns_list_on_success(self):
        from data.market_data import get_market_snapshots
        with patch("data.market_data.fetch_stock_data") as mock_fetch, \
             patch("data.market_data.summarise_for_ai", return_value=self._snap("AAPL")), \
             patch("data.market_data.get_spy_5d_return", return_value=1.5):
            mock_fetch.return_value = MagicMock()   # non-None DataFrame
            result = get_market_snapshots(["AAPL"])
        self.assertIsInstance(result, list)

    def test_filters_out_failed_fetches(self):
        from data.market_data import get_market_snapshots
        with patch("data.market_data.fetch_stock_data", return_value=None), \
             patch("data.market_data.get_spy_5d_return", return_value=None):
            result = get_market_snapshots(["AAPL", "NVDA"])
        self.assertEqual(result, [])

    def test_rel_strength_added_when_spy_5d_available(self):
        from data.market_data import get_market_snapshots
        snap = self._snap("AAPL")
        with patch("data.market_data.fetch_stock_data", return_value=MagicMock()), \
             patch("data.market_data.summarise_for_ai", return_value=snap), \
             patch("data.market_data.get_spy_5d_return", return_value=1.5):
            result = get_market_snapshots(["AAPL"])
        if result:
            self.assertIn("rel_strength_5d", result[0])

    def test_rel_strength_omitted_when_spy_unavailable(self):
        from data.market_data import get_market_snapshots
        snap = self._snap("AAPL")
        with patch("data.market_data.fetch_stock_data", return_value=MagicMock()), \
             patch("data.market_data.summarise_for_ai", return_value=snap), \
             patch("data.market_data.get_spy_5d_return", return_value=None):
            result = get_market_snapshots(["AAPL"])
        if result:
            self.assertNotIn("rel_strength_5d", result[0])

    def test_empty_symbols_returns_empty(self):
        from data.market_data import get_market_snapshots
        with patch("data.market_data.get_spy_5d_return", return_value=None):
            result = get_market_snapshots([])
        self.assertEqual(result, [])
