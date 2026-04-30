"""Tests for data/market_data.py — summarise_for_ai and yfinance helpers."""
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np


def _make_df(rows=3, close_vals=None):
    """Build a minimal DataFrame matching the shape summarise_for_ai expects."""
    n = rows
    closes = close_vals or [100.0 + i for i in range(n)]
    df = pd.DataFrame({
        "Close":          closes,
        "High":           [c + 1 for c in closes],
        "Low":            [c - 1 for c in closes],
        "Volume":         [1_000_000] * n,
        "rsi":            [50.0] * n,
        "macd":           [0.1] * n,
        "macd_signal":    [0.05] * n,
        "macd_diff":      [0.05, -0.01, 0.05][: n],   # prev negative → crossed up on last
        "ema9":           [c * 0.99 for c in closes],  # ema9 below close
        "ema21":          [c * 0.98 for c in closes],  # ema21 below ema9
        "bb_upper":       [c + 5 for c in closes],
        "bb_lower":       [c - 5 for c in closes],
        "bb_pct":         [0.5] * n,
        "vol_ratio":      [1.2] * n,
        "avg_volume_20":  [1_000_000] * n,
        "ret_1d":         [0.5] * n,
        "ret_5d":         [2.0] * n,
        "ret_10d":        [4.0] * n,
        "weekly_trend_up": [True] * n,
        "weekly_rsi":     [55.0] * n,
    })
    return df


class TestSummariseForAI(unittest.TestCase):

    def test_returns_dict_with_required_keys(self):
        from data.market_data import summarise_for_ai
        result = summarise_for_ai("AAPL", _make_df())
        for key in ["symbol", "current_price", "rsi_14", "macd_diff",
                    "ema9_above_ema21", "bb_pct", "vol_ratio",
                    "ret_1d_pct", "ret_5d_pct", "ret_10d_pct",
                    "macd_crossed_up", "macd_crossed_down",
                    "price_vs_ema9_pct", "weekly_trend_up", "weekly_rsi"]:
            self.assertIn(key, result, f"Missing key: {key}")

    def test_symbol_passed_through(self):
        from data.market_data import summarise_for_ai
        result = summarise_for_ai("NVDA", _make_df())
        self.assertEqual(result["symbol"], "NVDA")

    def test_current_price_is_last_close(self):
        from data.market_data import summarise_for_ai
        df = _make_df(close_vals=[100.0, 110.0, 125.5])
        result = summarise_for_ai("AAPL", df)
        self.assertAlmostEqual(result["current_price"], 125.5, places=2)

    def test_macd_crossed_up_detected(self):
        from data.market_data import summarise_for_ai
        # prev macd_diff negative, latest positive → crossed up
        df = _make_df()
        df["macd_diff"] = [-0.1, -0.05, 0.02]
        result = summarise_for_ai("AAPL", df)
        self.assertTrue(result["macd_crossed_up"])
        self.assertFalse(result["macd_crossed_down"])

    def test_macd_crossed_down_detected(self):
        from data.market_data import summarise_for_ai
        df = _make_df()
        df["macd_diff"] = [0.1, 0.05, -0.02]
        result = summarise_for_ai("AAPL", df)
        self.assertTrue(result["macd_crossed_down"])
        self.assertFalse(result["macd_crossed_up"])

    def test_ema9_above_ema21_true(self):
        from data.market_data import summarise_for_ai
        df = _make_df()
        df["ema9"] = [101.0, 102.0, 103.0]
        df["ema21"] = [99.0, 100.0, 101.0]
        result = summarise_for_ai("AAPL", df)
        self.assertTrue(result["ema9_above_ema21"])

    def test_ema9_above_ema21_false(self):
        from data.market_data import summarise_for_ai
        df = _make_df()
        df["ema9"] = [98.0, 99.0, 100.0]
        df["ema21"] = [101.0, 102.0, 103.0]
        result = summarise_for_ai("AAPL", df)
        self.assertFalse(result["ema9_above_ema21"])

    def test_price_vs_ema9_pct_positive_when_above(self):
        from data.market_data import summarise_for_ai
        df = _make_df(close_vals=[100.0, 105.0, 110.0])
        df["ema9"] = [100.0, 104.0, 105.0]   # close above ema9
        result = summarise_for_ai("AAPL", df)
        self.assertGreater(result["price_vs_ema9_pct"], 0)

    def test_weekly_trend_up_passed_through(self):
        from data.market_data import summarise_for_ai
        df = _make_df()
        df["weekly_trend_up"] = [False, False, False]
        result = summarise_for_ai("AAPL", df)
        self.assertFalse(result["weekly_trend_up"])

    def test_values_are_rounded(self):
        from data.market_data import summarise_for_ai
        result = summarise_for_ai("AAPL", _make_df())
        # Spot-check that no value has excessive decimal places
        self.assertEqual(result["rsi_14"], round(result["rsi_14"], 1))
        self.assertEqual(result["bb_pct"], round(result["bb_pct"], 2))


class TestGetVix(unittest.TestCase):

    def test_returns_float_on_success(self):
        from data.market_data import get_vix
        hist = pd.DataFrame({"Close": [18.5, 19.2, 20.1]})
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = hist
        with patch("data.market_data.yf.Ticker", return_value=mock_ticker):
            result = get_vix()
        self.assertIsInstance(result, float)
        self.assertAlmostEqual(result, 20.1, places=1)

    def test_returns_none_on_empty_history(self):
        from data.market_data import get_vix
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        with patch("data.market_data.yf.Ticker", return_value=mock_ticker):
            result = get_vix()
        self.assertIsNone(result)

    def test_returns_none_on_exception(self):
        from data.market_data import get_vix
        with patch("data.market_data.yf.Ticker", side_effect=Exception("network")):
            result = get_vix()
        self.assertIsNone(result)


class TestGetSpy5dReturn(unittest.TestCase):

    def test_returns_correct_pct(self):
        from data.market_data import get_spy_5d_return
        # 6 rows: [-5] = 100, [-1] = 110 → 10% return
        hist = pd.DataFrame({"Close": [100.0, 103.0, 105.0, 107.0, 109.0, 110.0]})
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = hist
        with patch("data.market_data.yf.Ticker", return_value=mock_ticker):
            result = get_spy_5d_return()
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 10.0, places=1)

    def test_returns_none_when_insufficient_data(self):
        from data.market_data import get_spy_5d_return
        hist = pd.DataFrame({"Close": [100.0, 105.0]})  # Only 2 rows
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = hist
        with patch("data.market_data.yf.Ticker", return_value=mock_ticker):
            result = get_spy_5d_return()
        self.assertIsNone(result)

    def test_returns_none_on_exception(self):
        from data.market_data import get_spy_5d_return
        with patch("data.market_data.yf.Ticker", side_effect=Exception("timeout")):
            result = get_spy_5d_return()
        self.assertIsNone(result)


class TestFetchStockData(unittest.TestCase):

    def _make_ticker(self, rows=60, empty=False):
        mock_ticker = MagicMock()
        if empty:
            mock_ticker.history.return_value = pd.DataFrame()
        else:
            closes = [100.0 + i * 0.5 for i in range(rows)]
            # Use recent dates so the stale-data guard (> 3 days) doesn't reject the fixture.
            df = pd.DataFrame({
                "Open":   closes,
                "High":   [c + 1 for c in closes],
                "Low":    [c - 1 for c in closes],
                "Close":  closes,
                "Volume": [1_000_000] * rows,
            }, index=pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=rows))
            mock_ticker.history.return_value = df
        return mock_ticker

    def test_returns_dataframe_on_success(self):
        from data.market_data import fetch_stock_data
        with patch("data.market_data.yf.Ticker", return_value=self._make_ticker(60)):
            result = fetch_stock_data("AAPL", days=30)
        self.assertIsNotNone(result)
        self.assertIn("rsi", result.columns)
        self.assertIn("macd", result.columns)

    def test_returns_none_on_empty_dataframe(self):
        from data.market_data import fetch_stock_data
        with patch("data.market_data.yf.Ticker", return_value=self._make_ticker(empty=True)):
            result = fetch_stock_data("AAPL", days=30)
        self.assertIsNone(result)

    def test_returns_none_when_insufficient_rows(self):
        from data.market_data import fetch_stock_data
        with patch("data.market_data.yf.Ticker", return_value=self._make_ticker(rows=10)):
            result = fetch_stock_data("AAPL", days=30)
        self.assertIsNone(result)

    def test_returns_none_on_exception(self):
        from data.market_data import fetch_stock_data
        with patch("data.market_data.yf.Ticker", side_effect=Exception("network")):
            result = fetch_stock_data("AAPL", days=30)
        self.assertIsNone(result)

    def test_result_limited_to_requested_days(self):
        from data.market_data import fetch_stock_data
        with patch("data.market_data.yf.Ticker", return_value=self._make_ticker(200)):
            result = fetch_stock_data("AAPL", days=20)
        self.assertIsNotNone(result)
        self.assertLessEqual(len(result), 20)


class TestGetMarketSnapshots(unittest.TestCase):

    def _make_snap(self, symbol):
        return {
            "symbol": symbol, "current_price": 100.0,
            "ret_1d_pct": 0.5, "ret_5d_pct": 2.0, "ret_10d_pct": 4.0,
            "rsi_14": 55.0, "macd_diff": 0.1,
            "macd_crossed_up": False, "macd_crossed_down": False,
            "ema9_above_ema21": True, "bb_pct": 0.5,
            "vol_ratio": 1.2, "price_vs_ema9_pct": 1.0,
            "weekly_trend_up": True, "weekly_rsi": 55.0,
        }

    def test_returns_list_of_snapshots(self):
        from data.market_data import get_market_snapshots
        snap = self._make_snap("AAPL")
        with patch("data.market_data.fetch_stock_data", return_value=MagicMock()), \
             patch("data.market_data.summarise_for_ai", return_value=snap), \
             patch("data.market_data.get_spy_5d_return", return_value=1.0):
            result = get_market_snapshots(["AAPL"])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["symbol"], "AAPL")

    def test_skips_symbols_with_no_data(self):
        from data.market_data import get_market_snapshots
        with patch("data.market_data.fetch_stock_data", return_value=None), \
             patch("data.market_data.get_spy_5d_return", return_value=None):
            result = get_market_snapshots(["AAPL", "NVDA"])
        self.assertEqual(result, [])

    def test_adds_relative_strength_when_spy_available(self):
        from data.market_data import get_market_snapshots
        snap = self._make_snap("AAPL")
        with patch("data.market_data.fetch_stock_data", return_value=MagicMock()), \
             patch("data.market_data.summarise_for_ai", return_value=snap), \
             patch("data.market_data.get_spy_5d_return", return_value=1.5):
            result = get_market_snapshots(["AAPL"])
        self.assertIn("rel_strength_5d", result[0])
        self.assertAlmostEqual(result[0]["rel_strength_5d"], snap["ret_5d_pct"] - 1.5, places=2)

    def test_skips_relative_strength_when_spy_unavailable(self):
        from data.market_data import get_market_snapshots
        snap = self._make_snap("AAPL")
        with patch("data.market_data.fetch_stock_data", return_value=MagicMock()), \
             patch("data.market_data.summarise_for_ai", return_value=snap), \
             patch("data.market_data.get_spy_5d_return", return_value=None):
            result = get_market_snapshots(["AAPL"])
        self.assertNotIn("rel_strength_5d", result[0])

    def test_returns_empty_list_for_no_symbols(self):
        from data.market_data import get_market_snapshots
        with patch("data.market_data.get_spy_5d_return", return_value=None):
            result = get_market_snapshots([])
        self.assertEqual(result, [])
