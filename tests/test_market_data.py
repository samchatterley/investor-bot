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
