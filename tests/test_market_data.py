"""Tests for data/market_data.py — summarise_for_ai and yfinance helpers."""

import os
import pickle
import tempfile
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd

from data.sentiment_client import AAIISentiment

# A fixed AAII reading for deterministic, offline market-data injection tests.
_AAII_FIXTURE = AAIISentiment(
    bullish_pct=0.35,
    bearish_pct=0.30,
    neutral_pct=0.35,
    bull_bear_spread=0.05,
    extreme_bearish=False,
    extreme_bullish=False,
)


def _make_df(rows=3, close_vals=None):
    """Build a minimal DataFrame matching the shape summarise_for_ai expects."""
    n = rows
    closes = close_vals or [100.0 + i for i in range(n)]
    df = pd.DataFrame(
        {
            "Close": closes,
            "High": [c + 1 for c in closes],
            "Low": [c - 1 for c in closes],
            "Volume": [1_000_000] * n,
            "rsi": [50.0] * n,
            "macd": [0.1] * n,
            "macd_signal": [0.05] * n,
            "macd_diff": [0.05, -0.01, 0.05][:n],  # prev negative → crossed up on last
            "ema9": [c * 0.99 for c in closes],  # ema9 below close
            "ema21": [c * 0.98 for c in closes],  # ema21 below ema9
            "bb_upper": [c + 5 for c in closes],
            "bb_lower": [c - 5 for c in closes],
            "bb_pct": [0.5] * n,
            "vol_ratio": [1.2] * n,
            "avg_volume_20": [1_000_000] * n,
            "ret_1d": [0.5] * n,
            "ret_5d": [2.0] * n,
            "ret_10d": [4.0] * n,
            "weekly_trend_up": [True] * n,
            "weekly_rsi": [55.0] * n,
            # New fields
            "bb_squeeze": [False] * n,
            "high_52w": [closes[-1] * 1.05] * n,  # 5% above current price
            "is_inside_day": [False] * n,
            "rsi_divergence": [False] * n,
        }
    )
    return df


class TestSummariseForAI(unittest.TestCase):
    def test_returns_dict_with_required_keys(self):
        from data.market_data import summarise_for_ai

        result = summarise_for_ai("AAPL", _make_df())
        for key in [
            "symbol",
            "current_price",
            "rsi_14",
            "macd_diff",
            "ema9_above_ema21",
            "bb_pct",
            "vol_ratio",
            "ret_1d_pct",
            "ret_5d_pct",
            "ret_10d_pct",
            "macd_crossed_up",
            "macd_crossed_down",
            "price_vs_ema9_pct",
            "price_vs_ema21_pct",
            "weekly_trend_up",
            "weekly_rsi",
            "bb_squeeze",
            "is_inside_day",
            "price_vs_52w_high_pct",
            "bar_date",
            "bar_is_final",
            "data_source",
            "rsi_divergence",
        ]:
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
        df["ema9"] = [100.0, 104.0, 105.0]  # close above ema9
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

    def test_price_vs_ema21_pct_positive_when_above(self):
        from data.market_data import summarise_for_ai

        df = _make_df(close_vals=[100.0, 105.0, 110.0])
        df["ema21"] = [100.0, 102.0, 104.0]  # close > ema21
        result = summarise_for_ai("AAPL", df)
        self.assertGreater(result["price_vs_ema21_pct"], 0)

    def test_price_vs_ema21_pct_negative_when_below(self):
        from data.market_data import summarise_for_ai

        df = _make_df(close_vals=[100.0, 105.0, 110.0])
        df["ema21"] = [102.0, 108.0, 120.0]  # close < ema21
        result = summarise_for_ai("AAPL", df)
        self.assertLess(result["price_vs_ema21_pct"], 0)

    def test_bb_squeeze_true_when_set(self):
        from data.market_data import summarise_for_ai

        df = _make_df()
        df["bb_squeeze"] = [False, False, True]
        result = summarise_for_ai("AAPL", df)
        self.assertTrue(result["bb_squeeze"])

    def test_bb_squeeze_false_when_not_set(self):
        from data.market_data import summarise_for_ai

        df = _make_df()
        df["bb_squeeze"] = [False, False, False]
        result = summarise_for_ai("AAPL", df)
        self.assertFalse(result["bb_squeeze"])

    def test_is_inside_day_passed_through(self):
        from data.market_data import summarise_for_ai

        df = _make_df()
        df["is_inside_day"] = [False, False, True]
        result = summarise_for_ai("AAPL", df)
        self.assertTrue(result["is_inside_day"])

    def test_price_vs_52w_high_pct_negative_when_below_high(self):
        from data.market_data import summarise_for_ai

        df = _make_df(close_vals=[100.0, 102.0, 104.0])
        df["high_52w"] = [110.0, 110.0, 110.0]
        result = summarise_for_ai("AAPL", df)
        # 104 / 110 - 1 ≈ -5.45%
        self.assertAlmostEqual(
            result["price_vs_52w_high_pct"], round((104 / 110 - 1) * 100, 2), places=2
        )

    def test_price_vs_52w_high_pct_zero_when_high_is_nan(self):
        from data.market_data import summarise_for_ai

        df = _make_df()
        df["high_52w"] = [float("nan"), float("nan"), float("nan")]
        result = summarise_for_ai("AAPL", df)
        self.assertEqual(result["price_vs_52w_high_pct"], 0.0)

    def test_data_source_live_by_default(self):
        from data.market_data import summarise_for_ai

        result = summarise_for_ai("AAPL", _make_df())
        self.assertEqual(result["data_source"], "live")

    def test_data_source_preloaded_when_flag_set(self):
        from data.market_data import summarise_for_ai

        result = summarise_for_ai("AAPL", _make_df(), is_preloaded=True)
        self.assertEqual(result["data_source"], "preloaded")

    def test_bar_date_none_for_integer_index(self):
        from data.market_data import summarise_for_ai

        # _make_df() uses a RangeIndex — bar_date should be None
        result = summarise_for_ai("AAPL", _make_df())
        self.assertIsNone(result["bar_date"])

    def test_bar_date_string_for_datetime_index(self):
        import pandas as pd

        from data.market_data import summarise_for_ai

        df = _make_df()
        df.index = pd.bdate_range("2026-04-01", periods=len(df))
        result = summarise_for_ai("AAPL", df)
        self.assertEqual(result["bar_date"], "2026-04-03")

    def test_bar_is_final_true_for_past_date(self):
        import pandas as pd

        from data.market_data import summarise_for_ai

        df = _make_df()
        df.index = pd.bdate_range("2020-01-01", periods=len(df))
        result = summarise_for_ai("AAPL", df)
        self.assertTrue(result["bar_is_final"])

    def test_bar_is_final_none_for_integer_index(self):
        from data.market_data import summarise_for_ai

        result = summarise_for_ai("AAPL", _make_df())
        self.assertIsNone(result["bar_is_final"])

    def test_rsi_divergence_true_when_set(self):
        from data.market_data import summarise_for_ai

        df = _make_df()
        df["rsi_divergence"] = [False, False, True]
        result = summarise_for_ai("AAPL", df)
        self.assertTrue(result["rsi_divergence"])

    def test_rsi_divergence_false_by_default(self):
        from data.market_data import summarise_for_ai

        result = summarise_for_ai("AAPL", _make_df())
        self.assertFalse(result["rsi_divergence"])


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


class TestGetIndexPrice(unittest.TestCase):
    def test_returns_latest_close(self):
        from data.market_data import get_index_price

        hist = pd.DataFrame({"Close": [398.0, 399.0, 401.5]})
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = hist
        with patch("data.market_data.yf.Ticker", return_value=mock_ticker):
            result = get_index_price("SPY")
        self.assertEqual(result, 401.5)

    def test_returns_none_when_empty(self):
        from data.market_data import get_index_price

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame({"Close": []})
        with patch("data.market_data.yf.Ticker", return_value=mock_ticker):
            result = get_index_price("SPY")
        self.assertIsNone(result)

    def test_returns_none_on_exception(self):
        from data.market_data import get_index_price

        with patch("data.market_data.yf.Ticker", side_effect=Exception("timeout")):
            result = get_index_price("QQQ")
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


class TestGetSpy10dReturn(unittest.TestCase):
    def test_returns_correct_pct(self):
        from data.market_data import get_spy_10d_return

        # 11 rows: [-10] = 100, [-1] = 115 → 15% return
        hist = pd.DataFrame(
            {"Close": [100.0, 103.0, 105.0, 107.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0]}
        )
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = hist
        with patch("data.market_data.yf.Ticker", return_value=mock_ticker):
            result = get_spy_10d_return()
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 15.0, places=1)

    def test_returns_none_when_insufficient_data(self):
        from data.market_data import get_spy_10d_return

        hist = pd.DataFrame({"Close": [100.0, 105.0]})
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = hist
        with patch("data.market_data.yf.Ticker", return_value=mock_ticker):
            result = get_spy_10d_return()
        self.assertIsNone(result)

    def test_returns_none_on_exception(self):
        from data.market_data import get_spy_10d_return

        with patch("data.market_data.yf.Ticker", side_effect=Exception("timeout")):
            result = get_spy_10d_return()
        self.assertIsNone(result)


class TestFetchStockData(unittest.TestCase):
    def _make_ticker(self, rows=60, empty=False):
        mock_ticker = MagicMock()
        if empty:
            mock_ticker.history.return_value = pd.DataFrame()
        else:
            idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=rows)
            actual_rows = len(idx)
            closes = [100.0 + i * 0.5 for i in range(actual_rows)]
            # Use recent dates so the stale-data guard (> 3 days) doesn't reject the fixture.
            df = pd.DataFrame(
                {
                    "Open": closes,
                    "High": [c + 1 for c in closes],
                    "Low": [c - 1 for c in closes],
                    "Close": closes,
                    "Volume": [1_000_000] * actual_rows,
                },
                index=idx,
            )
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

    def _make_preloaded_df(self, rows=60):
        """Return a valid DataFrame suitable for preloaded path (Close + Volume required)."""
        idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=rows)
        n = len(idx)
        closes = [100.0 + i * 0.5 for i in range(n)]
        return pd.DataFrame(
            {
                "Open": closes,
                "High": [c + 1 for c in closes],
                "Low": [c - 1 for c in closes],
                "Close": closes,
                "Volume": [1_000_000] * n,
            },
            index=idx,
        )

    def test_preloaded_data_no_as_of_returns_dataframe(self):
        """Lines 40,44: preloaded symbol with no as_of → df.copy() path."""
        from data.market_data import fetch_stock_data

        df = self._make_preloaded_df(rows=60)
        result = fetch_stock_data("AAPL", days=30, preloaded={"AAPL": df})
        self.assertIsNotNone(result)
        self.assertIn("rsi", result.columns)

    def test_preloaded_data_with_as_of_slices_dataframe(self):
        """Lines 40,42: preloaded symbol with as_of → slice and copy path."""
        from data.market_data import fetch_stock_data

        df = self._make_preloaded_df(rows=60)
        as_of = df.index[-5].strftime("%Y-%m-%d")
        result = fetch_stock_data("AAPL", days=30, preloaded={"AAPL": df}, as_of=as_of)
        self.assertIsNotNone(result)

    def test_preloaded_data_too_few_rows_returns_none(self):
        """Lines 45-47: preloaded df with < 35 rows → warning logged, returns None."""
        from data.market_data import fetch_stock_data

        df = self._make_preloaded_df(rows=10)
        result = fetch_stock_data("AAPL", days=30, preloaded={"AAPL": df})
        self.assertIsNone(result)

    def test_stale_ticker_data_returns_none(self):
        """Lines 64-67: ticker last date > 4 days old → stale feed, returns None."""
        from data.market_data import fetch_stock_data

        # Use dates that are definitely stale (10 days ago and earlier)
        idx = pd.bdate_range(
            end=pd.Timestamp.today().normalize() - pd.Timedelta(days=10), periods=60
        )
        n = len(idx)
        closes = [100.0 + i * 0.5 for i in range(n)]
        df = pd.DataFrame(
            {
                "Open": closes,
                "High": [c + 1 for c in closes],
                "Low": [c - 1 for c in closes],
                "Close": closes,
                "Volume": [1_000_000] * n,
            },
            index=idx,
        )
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = df
        with patch("data.market_data.yf.Ticker", return_value=mock_ticker):
            result = fetch_stock_data("AAPL", days=30)
        self.assertIsNone(result)

    def test_result_limited_to_requested_days(self):
        from data.market_data import fetch_stock_data

        with patch("data.market_data.yf.Ticker", return_value=self._make_ticker(200)):
            result = fetch_stock_data("AAPL", days=20)
        self.assertIsNotNone(result)
        self.assertLessEqual(len(result), 20)

    def test_weekly_trend_fallback_when_insufficient_weekly_data(self):
        # Lines 79-81: fewer than 22 weekly bars → weekly_trend_up=True, weekly_rsi=50.0
        from data.market_data import fetch_stock_data

        # Provide only 40 daily bars — resampled to weekly gives < 22 weekly candles
        idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=40)
        closes = [100.0 + i * 0.5 for i in range(len(idx))]
        df = pd.DataFrame(
            {
                "Open": closes,
                "High": [c + 1 for c in closes],
                "Low": [c - 1 for c in closes],
                "Close": closes,
                "Volume": [1_000_000] * len(idx),
            },
            index=idx,
        )
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = df
        with patch("data.market_data.yf.Ticker", return_value=mock_ticker):
            result = fetch_stock_data("AAPL", days=10)
        self.assertIsNotNone(result)
        # Fallback values must be present
        self.assertTrue(result["weekly_trend_up"].iloc[-1])
        self.assertAlmostEqual(result["weekly_rsi"].iloc[-1], 50.0)

    def test_weekly_trend_exception_fallback(self):
        # Lines 82-84: exception inside the weekly try block → weekly_trend_up=True, weekly_rsi=50.0
        from data.market_data import fetch_stock_data

        # Use enough bars that all daily indicators succeed, but patch RSIIndicator
        # inside the weekly block to raise (called with weekly_close series argument)
        idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=200)
        closes = [100.0 + i * 0.5 for i in range(len(idx))]
        df = pd.DataFrame(
            {
                "Open": closes,
                "High": [c + 1 for c in closes],
                "Low": [c - 1 for c in closes],
                "Close": closes,
                "Volume": [1_000_000] * len(idx),
            },
            index=idx,
        )
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = df

        # Patch RSIIndicator to raise only when called with a small (weekly) series
        real_rsi = __import__("ta.momentum", fromlist=["RSIIndicator"]).RSIIndicator

        call_count = [0]

        def patched_rsi(close, window):
            call_count[0] += 1
            if call_count[0] >= 2:  # second call = weekly RSI
                raise Exception("weekly rsi failed")
            return real_rsi(close=close, window=window)

        with (
            patch("data.market_data.yf.Ticker", return_value=mock_ticker),
            patch("data.market_data.RSIIndicator", side_effect=patched_rsi),
        ):
            result = fetch_stock_data("AAPL", days=10)
        self.assertIsNotNone(result)
        self.assertTrue(result["weekly_trend_up"].iloc[-1])
        self.assertAlmostEqual(result["weekly_rsi"].iloc[-1], 50.0)


class TestGetMarketSnapshots(unittest.TestCase):
    def _make_snap(self, symbol):
        return {
            "symbol": symbol,
            "current_price": 100.0,
            "ret_1d_pct": 0.5,
            "ret_5d_pct": 2.0,
            "ret_10d_pct": 4.0,
            "rsi_14": 55.0,
            "macd_diff": 0.1,
            "macd_crossed_up": False,
            "macd_crossed_down": False,
            "ema9_above_ema21": True,
            "bb_pct": 0.5,
            "vol_ratio": 1.2,
            "price_vs_ema9_pct": 1.0,
            "weekly_trend_up": True,
            "weekly_rsi": 55.0,
        }

    def test_returns_list_of_snapshots(self):
        from data.market_data import get_market_snapshots

        snap = self._make_snap("AAPL")
        with (
            patch("data.market_data._bulk_download", return_value={}),
            patch("data.market_data.get_fundamentals", return_value={}),
            patch("data.market_data.fetch_stock_data", return_value=MagicMock()),
            patch("data.market_data.summarise_for_ai", return_value=snap),
            patch("data.market_data.get_spy_5d_return", return_value=1.0),
            patch("data.market_data.get_spy_10d_return", return_value=None),
        ):
            result = get_market_snapshots(["AAPL"])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["symbol"], "AAPL")

    def test_skips_symbols_with_no_data(self):
        from data.market_data import get_market_snapshots

        with (
            patch("data.market_data._bulk_download", return_value={}),
            patch("data.market_data.get_fundamentals", return_value={}),
            patch("data.market_data.fetch_stock_data", return_value=None),
            patch("data.market_data.get_spy_5d_return", return_value=None),
            patch("data.market_data.get_spy_10d_return", return_value=None),
        ):
            result = get_market_snapshots(["AAPL", "NVDA"])
        self.assertEqual(result, [])

    def test_adds_relative_strength_when_spy_available(self):
        from data.market_data import get_market_snapshots

        snap = self._make_snap("AAPL")
        with (
            patch("data.market_data._bulk_download", return_value={}),
            patch("data.market_data.fetch_stock_data", return_value=MagicMock()),
            patch("data.market_data.summarise_for_ai", return_value=snap),
            patch("data.market_data.get_spy_5d_return", return_value=1.5),
            patch("data.market_data.get_spy_10d_return", return_value=None),
        ):
            result = get_market_snapshots(["AAPL"])
        self.assertIn("rel_strength_5d", result[0])
        self.assertAlmostEqual(result[0]["rel_strength_5d"], snap["ret_5d_pct"] - 1.5, places=2)

    def test_skips_relative_strength_when_spy_unavailable(self):
        from data.market_data import get_market_snapshots

        snap = self._make_snap("AAPL")
        with (
            patch("data.market_data._bulk_download", return_value={}),
            patch("data.market_data.fetch_stock_data", return_value=MagicMock()),
            patch("data.market_data.summarise_for_ai", return_value=snap),
            patch("data.market_data.get_spy_5d_return", return_value=None),
            patch("data.market_data.get_spy_10d_return", return_value=None),
        ):
            result = get_market_snapshots(["AAPL"])
        self.assertNotIn("rel_strength_5d", result[0])

    def test_adds_rel_strength_10d_when_spy_available(self):
        from data.market_data import get_market_snapshots

        snap = self._make_snap("AAPL")
        with (
            patch("data.market_data._bulk_download", return_value={}),
            patch("data.market_data.fetch_stock_data", return_value=MagicMock()),
            patch("data.market_data.summarise_for_ai", return_value=snap),
            patch("data.market_data.get_spy_5d_return", return_value=1.0),
            patch("data.market_data.get_spy_10d_return", return_value=2.0),
        ):
            result = get_market_snapshots(["AAPL"])
        self.assertIn("rel_strength_10d", result[0])
        self.assertAlmostEqual(result[0]["rel_strength_10d"], snap["ret_10d_pct"] - 2.0, places=2)

    def test_skips_rel_strength_10d_when_spy_unavailable(self):
        from data.market_data import get_market_snapshots

        snap = self._make_snap("AAPL")
        with (
            patch("data.market_data._bulk_download", return_value={}),
            patch("data.market_data.fetch_stock_data", return_value=MagicMock()),
            patch("data.market_data.summarise_for_ai", return_value=snap),
            patch("data.market_data.get_spy_5d_return", return_value=None),
            patch("data.market_data.get_spy_10d_return", return_value=None),
        ):
            result = get_market_snapshots(["AAPL"])
        self.assertNotIn("rel_strength_10d", result[0])

    def test_returns_empty_list_for_no_symbols(self):
        from data.market_data import get_market_snapshots

        with (
            patch("data.market_data._bulk_download", return_value={}),
            patch("data.market_data.get_fundamentals", return_value={}),
            patch("data.market_data.get_spy_5d_return", return_value=None),
            patch("data.market_data.get_spy_10d_return", return_value=None),
        ):
            result = get_market_snapshots([])
        self.assertEqual(result, [])

    def test_rs_rank_pct_computed_when_four_or_more_symbols(self):
        """Lines 374-377: ≥4 snapshots with rel_strength_20d → rs_rank_pct added."""
        from data.market_data import get_market_snapshots

        symbols = ["AAPL", "MSFT", "GOOGL", "NVDA"]
        snaps = {
            sym: dict(self._make_snap(sym), ret_20d_pct=float(i + 1))
            for i, sym in enumerate(symbols)
        }

        def _fake_summarise(sym, df, **kw):
            return snaps[sym]

        patches = [
            patch("data.market_data._bulk_download", return_value={}),
            patch("data.market_data.get_fundamentals", return_value={}),
            patch("data.market_data.fetch_stock_data", return_value=MagicMock()),
            patch("data.market_data.summarise_for_ai", side_effect=_fake_summarise),
            patch("data.market_data.get_spy_5d_return", return_value=None),
            patch("data.market_data.get_spy_10d_return", return_value=None),
            patch("data.market_data.get_spy_20d_return", return_value=1.0),
        ]
        for p in patches:
            p.start()
        try:
            result = get_market_snapshots(symbols)
        finally:
            for p in patches:
                p.stop()

        self.assertEqual(len(result), 4)
        self.assertTrue(all("rs_rank_pct" in s for s in result))

    def test_fundamentals_merged_into_snapshot_when_present(self):
        """Line 357: snap.update(fundamentals[sym]) executed when sym in fundamentals."""
        from data.market_data import get_market_snapshots

        snap = self._make_snap("AAPL")
        with (
            patch("data.market_data._bulk_download", return_value={}),
            patch("data.market_data.get_fundamentals", return_value={"AAPL": {"pe_ratio": 22.5}}),
            patch("data.market_data.fetch_stock_data", return_value=MagicMock()),
            patch("data.market_data.summarise_for_ai", return_value=snap),
            patch("data.market_data.get_spy_5d_return", return_value=None),
            patch("data.market_data.get_spy_10d_return", return_value=None),
        ):
            result = get_market_snapshots(["AAPL"])
        self.assertEqual(result[0]["pe_ratio"], 22.5)

    def test_amihud_cross_sectional_ranking_with_ten_plus_symbols(self):
        """10+ non-zero amihud values trigger cross-sectional ranking (lines 535-537)."""
        from data.market_data import get_market_snapshots

        symbols = [f"SYM{i:02d}" for i in range(12)]
        snaps = {sym: self._make_snap(sym) for sym in symbols}

        patches = [
            patch("data.market_data._bulk_download", return_value={}),
            patch("data.market_data.get_fundamentals", return_value={}),
            patch("data.market_data.fetch_stock_data", return_value=MagicMock()),
            patch(
                "data.market_data.summarise_for_ai",
                side_effect=lambda sym, df, **kw: dict(snaps[sym]),
            ),
            patch("data.market_data.get_spy_5d_return", return_value=None),
            patch("data.market_data.get_spy_10d_return", return_value=None),
            patch("data.market_data.get_spy_20d_return", return_value=None),
            patch("data.market_data.compute_amihud_illiquidity", return_value=1e-7),
        ]
        for p in patches:
            p.start()
        try:
            result = get_market_snapshots(symbols)
        finally:
            for p in patches:
                p.stop()

        self.assertEqual(len(result), 12)
        # All snapshots should have amihud_illiquid set (the top-10% threshold path ran)
        self.assertTrue(all("amihud_illiquid" in s for s in result))

    def test_compute_amihud_illiquidity_exception_returns_zero(self):
        """Exception inside compute_amihud_illiquidity → return 0.0 (lines 269-270)."""
        from data.market_data import compute_amihud_illiquidity

        result = compute_amihud_illiquidity(None)  # type: ignore[arg-type]
        self.assertEqual(result, 0.0)

    def test_preloaded_as_of_uses_preloaded_spy_returns(self):
        """Lines 472-474: preloaded+as_of → _spy_return_from_preloaded called instead of live."""
        from data.market_data import get_market_snapshots

        snap = self._make_snap("AAPL")
        idx = pd.bdate_range("2024-01-01", periods=30)
        spy_df = pd.DataFrame({"Close": [400.0 + i for i in range(30)]}, index=idx)
        preloaded = {"AAPL": MagicMock(), "SPY": spy_df}
        as_of = "2024-02-15"
        with (
            patch("data.market_data.fetch_stock_data", return_value=MagicMock()),
            patch("data.market_data.summarise_for_ai", return_value=snap),
        ):
            result = get_market_snapshots(["AAPL"], preloaded=preloaded, as_of=as_of)
        self.assertEqual(len(result), 1)

    def test_live_bulk_download_log_when_data_returned(self):
        """Line 483: logger.info fires when _bulk_download returns non-empty dict."""
        from data.market_data import get_market_snapshots

        snap = self._make_snap("AAPL")
        fake_bulk = {"AAPL": MagicMock()}
        with (
            patch("data.market_data._bulk_download", return_value=fake_bulk),
            patch("data.market_data.get_fundamentals", return_value={}),
            patch("data.market_data.fetch_stock_data", return_value=MagicMock()),
            patch("data.market_data.summarise_for_ai", return_value=snap),
            patch("data.market_data.get_spy_5d_return", return_value=None),
            patch("data.market_data.get_spy_10d_return", return_value=None),
            patch("data.market_data.get_spy_20d_return", return_value=None),
        ):
            result = get_market_snapshots(["AAPL"])
        self.assertEqual(len(result), 1)


class TestGetSpy20dReturn(unittest.TestCase):
    def test_exception_returns_none(self):
        """Lines 256-258: yf.Ticker raises → exception caught, returns None."""
        from data.market_data import get_spy_20d_return

        with patch("data.market_data.yf.Ticker", side_effect=Exception("network error")):
            result = get_spy_20d_return()
        self.assertIsNone(result)

    def test_insufficient_rows_returns_none(self):
        """Line 252->258: hist has fewer than 21 rows → falls through to return None."""
        from data.market_data import get_spy_20d_return

        short_hist = pd.DataFrame({"Close": [400.0] * 10})
        ticker_mock = MagicMock()
        ticker_mock.history.return_value = short_hist
        with patch("data.market_data.yf.Ticker", return_value=ticker_mock):
            result = get_spy_20d_return()
        self.assertIsNone(result)


class TestBulkDownload(unittest.TestCase):
    """_bulk_download: single API call returns per-symbol DataFrames."""

    def setUp(self):
        self._p_load = patch("data.market_data._load_bulk_cache", return_value={})
        self._p_save = patch("data.market_data._save_bulk_cache")
        self._p_load.start()
        self._p_save.start()

    def tearDown(self):
        self._p_load.stop()
        self._p_save.stop()

    def _make_multi_df(self, symbols: list[str], periods: int = 10) -> pd.DataFrame:
        idx = pd.bdate_range("2025-01-01", periods=periods)
        data = {
            (field, sym): [float(i) for i in range(periods)]
            for field in ("Open", "Close", "Volume")
            for sym in symbols
        }
        return pd.DataFrame(data, index=idx)

    def test_returns_empty_on_download_exception(self):
        from data.market_data import _bulk_download

        with patch("data.market_data.yf.download", side_effect=Exception("network error")):
            result = _bulk_download(["AAPL"], 200)
        self.assertEqual(result, {})

    def test_returns_empty_on_empty_raw(self):
        from data.market_data import _bulk_download

        with patch("data.market_data.yf.download", return_value=pd.DataFrame()):
            result = _bulk_download(["AAPL"], 200)
        self.assertEqual(result, {})

    def test_multi_symbol_splits_by_ticker(self):
        from data.market_data import _bulk_download

        raw = self._make_multi_df(["AAPL", "NVDA"])
        with patch("data.market_data.yf.download", return_value=raw):
            result = _bulk_download(["AAPL", "NVDA"], 200)
        self.assertIn("AAPL", result)
        self.assertIn("NVDA", result)
        self.assertIsInstance(result["AAPL"], pd.DataFrame)

    def test_single_symbol_flat_columns(self):
        from data.market_data import _bulk_download

        idx = pd.bdate_range("2025-01-01", periods=5)
        raw = pd.DataFrame({"Open": [1.0] * 5, "Close": [2.0] * 5}, index=idx)
        with patch("data.market_data.yf.download", return_value=raw):
            result = _bulk_download(["AAPL"], 200)
        self.assertIn("AAPL", result)

    def test_missing_symbol_excluded(self):
        from data.market_data import _bulk_download

        raw = self._make_multi_df(["AAPL"])
        with patch("data.market_data.yf.download", return_value=raw):
            result = _bulk_download(["AAPL", "MISSING"], 200)
        self.assertIn("AAPL", result)
        self.assertNotIn("MISSING", result)


class TestSpyReturnFromPreloadedException(unittest.TestCase):
    """Lines 221-222: exception in _spy_return_from_preloaded returns None."""

    def test_exception_returns_none(self):
        from data.market_data import _spy_return_from_preloaded

        idx = pd.bdate_range("2025-01-01", periods=20)
        spy_df = pd.DataFrame({"Close": [100.0 + i for i in range(20)]}, index=idx)
        # "not-a-date" causes pd.Timestamp to raise ValueError → except branch fires
        result = _spy_return_from_preloaded({"SPY": spy_df}, "not-a-date", 5)
        self.assertIsNone(result)

    def test_spy_not_in_preloaded_returns_none(self):
        """Line 327: spy_df is None when SPY not in preloaded dict."""
        from data.market_data import _spy_return_from_preloaded

        result = _spy_return_from_preloaded({}, "2024-06-01", 5)
        self.assertIsNone(result)

    def test_returns_float_with_valid_spy_data(self):
        """Lines 330-332: sliced data has enough rows → returns rounded float."""
        from data.market_data import _spy_return_from_preloaded

        idx = pd.bdate_range("2024-01-01", periods=30)
        closes = [100.0 + i for i in range(30)]
        spy_df = pd.DataFrame({"Close": closes}, index=idx)
        result = _spy_return_from_preloaded({"SPY": spy_df}, "2024-02-15", 5)
        self.assertIsInstance(result, float)

    def test_returns_none_when_too_few_sliced_rows(self):
        """Line 331: sliced df has fewer rows than lookback + 1 → returns None."""
        from data.market_data import _spy_return_from_preloaded

        idx = pd.bdate_range("2024-01-01", periods=3)
        spy_df = pd.DataFrame({"Close": [100.0, 101.0, 102.0]}, index=idx)
        # lookback=5 requires 6 rows; we have 3 → returns None
        result = _spy_return_from_preloaded({"SPY": spy_df}, "2024-01-05", 5)
        self.assertIsNone(result)


class TestBulkDownloadKeyError(unittest.TestCase):
    """Lines 290-291: KeyError in xs() is swallowed silently."""

    def setUp(self):
        self._p_load = patch("data.market_data._load_bulk_cache", return_value={})
        self._p_save = patch("data.market_data._save_bulk_cache")
        self._p_load.start()
        self._p_save.start()

    def tearDown(self):
        self._p_load.stop()
        self._p_save.stop()

    def test_xs_keyerror_is_ignored(self):
        from data.market_data import _bulk_download

        # Build a MultiIndex DataFrame that lists AAPL in the level-1 index but
        # whose xs() raises KeyError (we simulate this by patching xs on the
        # returned raw object).
        idx = pd.bdate_range("2025-01-01", periods=5)
        # Create real MultiIndex columns so isinstance(raw.columns, pd.MultiIndex)
        # is True and "AAPL" appears in available.
        cols = pd.MultiIndex.from_tuples([("Open", "AAPL"), ("Close", "AAPL")], names=[None, None])
        raw = pd.DataFrame(
            [[1.0, 2.0]] * 5,
            index=idx,
            columns=cols,
        )

        original_xs = raw.xs

        def _xs_raises(key, **kwargs):
            if key == "AAPL":
                raise KeyError("AAPL")
            return original_xs(key, **kwargs)  # pragma: no cover

        raw.xs = _xs_raises  # type: ignore[method-assign]

        with patch("data.market_data.yf.download", return_value=raw):
            result = _bulk_download(["AAPL"], 200)

        # AAPL should be absent (KeyError silently skipped), not raise
        self.assertNotIn("AAPL", result)


class TestGetIntradayData(unittest.TestCase):
    """get_intraday_data coverage gaps."""

    def test_empty_symbols_returns_empty(self):
        """Line 365: empty symbol list → immediate {}."""
        from data.market_data import get_intraday_data

        result = get_intraday_data([])
        self.assertEqual(result, {})

    def test_before_market_open_returns_empty(self):
        """Line 374: current time before 09:30 ET → {}."""

        from data.market_data import get_intraday_data

        _ET = __import__("zoneinfo").ZoneInfo("America/New_York")
        # 08:00 ET is before market open
        fake_now = datetime(2025, 6, 10, 8, 0, 0, tzinfo=_ET)

        with patch("data.market_data.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            # Pass through other datetime usages
            mock_dt.strptime = datetime.strptime
            result = get_intraday_data(["AAPL"])

        self.assertEqual(result, {})

    def _make_alpaca_bar(self, ts_et, open_=100.0, high=101.0, low=99.0, close=100.5, vol=50_000):
        """Build a minimal mock Alpaca bar object."""
        bar = MagicMock()
        bar.timestamp = ts_et
        bar.open = open_
        bar.high = high
        bar.low = low
        bar.close = close
        bar.volume = vol
        return bar

    def _patch_alpaca(self, bars_by_sym: dict):
        """Return a context manager that patches StockHistoricalDataClient."""
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.data = bars_by_sym
        mock_client.get_stock_bars.return_value = mock_resp

        return patch("data.market_data.StockHistoricalDataClient", return_value=mock_client)

    def _fake_now_et(self, hour=10, minute=0):
        """A datetime that is after market open (09:30 ET)."""
        _ET = __import__("zoneinfo").ZoneInfo("America/New_York")
        return datetime(2025, 6, 10, hour, minute, 0, tzinfo=_ET)

    def test_df_empty_after_sort_skips_symbol(self):
        """Line 420: df is empty after sort_values/reset_index → continue.

        Provide a valid bar so the DataFrame is constructed successfully, then
        patch ``sort_values`` to return an empty DataFrame so ``df.empty``
        is True and the continue on line 420 is executed.
        """
        from data.market_data import get_intraday_data

        _ET = __import__("zoneinfo").ZoneInfo("America/New_York")
        fake_now = self._fake_now_et(hour=10)
        bar_ts = datetime(2025, 6, 10, 9, 35, 0, tzinfo=_ET)
        bar = self._make_alpaca_bar(bar_ts)

        original_sort = pd.DataFrame.sort_values
        call_count = [0]

        def _patched_sort(self_df, *args, **kwargs):
            call_count[0] += 1
            # First call is inside the per-symbol intraday loop
            if call_count[0] == 1:
                return pd.DataFrame(columns=self_df.columns)
            return original_sort(self_df, *args, **kwargs)  # pragma: no cover

        with (
            self._patch_alpaca({"AAPL": [bar]}),
            patch("data.market_data.datetime") as mock_dt,
            patch.object(pd.DataFrame, "sort_values", _patched_sort),
        ):
            mock_dt.now.return_value = fake_now
            mock_dt.strptime = datetime.strptime
            result = get_intraday_data(["AAPL"])

        self.assertNotIn("AAPL", result)

    def test_all_bars_before_market_open_skips_symbol(self):
        """Line 427: today_bars is empty (all timestamps < market_open) → continue."""
        from data.market_data import get_intraday_data

        _ET = __import__("zoneinfo").ZoneInfo("America/New_York")
        fake_now = self._fake_now_et(hour=10)
        # A bar at 09:15 ET — before 09:30 open, so today_bars will be empty
        pre_open_ts = datetime(2025, 6, 10, 9, 15, 0, tzinfo=_ET)
        bar = self._make_alpaca_bar(pre_open_ts)

        with (
            self._patch_alpaca({"AAPL": [bar]}),
            patch("data.market_data.datetime") as mock_dt,
        ):
            mock_dt.now.return_value = fake_now
            mock_dt.strptime = datetime.strptime
            result = get_intraday_data(["AAPL"])

        self.assertNotIn("AAPL", result)

    def test_metrics_exception_is_swallowed(self):
        """Lines 499-501: exception during metrics calculation → continue, symbol absent.

        Supply a bar with open=0 so intraday_change_pct = (close/0 - 1)*100
        raises ZeroDivisionError inside the per-symbol try block.
        """
        from data.market_data import get_intraday_data

        _ET = __import__("zoneinfo").ZoneInfo("America/New_York")
        fake_now = self._fake_now_et(hour=10, minute=30)
        # Bar at 09:35 ET (after market open) with open=0 to trigger ZeroDivisionError
        bar_ts = datetime(2025, 6, 10, 9, 35, 0, tzinfo=_ET)
        bar = self._make_alpaca_bar(bar_ts, open_=0.0, high=1.0, low=0.0, close=1.0, vol=50_000)

        with (
            self._patch_alpaca({"AAPL": [bar]}),
            patch("data.market_data.datetime") as mock_dt,
        ):
            mock_dt.now.return_value = fake_now
            mock_dt.strptime = datetime.strptime
            result = get_intraday_data(["AAPL"])

        self.assertNotIn("AAPL", result)


class TestBulkDownloadBranchGaps(unittest.TestCase):
    """Branches 288->284 and 292->294 in _bulk_download."""

    def setUp(self):
        self._p_load = patch("data.market_data._load_bulk_cache", return_value={})
        self._p_save = patch("data.market_data._save_bulk_cache")
        self._p_load.start()
        self._p_save.start()

    def tearDown(self):
        self._p_load.stop()
        self._p_save.stop()

    def test_xs_returns_empty_df_symbol_excluded(self):
        """Line 288->284: sym_df.empty after xs() → symbol not added to result."""
        from data.market_data import _bulk_download

        idx = pd.bdate_range("2025-01-01", periods=5)
        cols = pd.MultiIndex.from_tuples([("Open", "AAPL"), ("Close", "AAPL")], names=[None, None])
        raw = pd.DataFrame([[float("nan")] * 2] * 5, index=idx, columns=cols)

        with patch("data.market_data.yf.download", return_value=raw):
            result = _bulk_download(["AAPL"], 200)

        self.assertNotIn("AAPL", result)

    def test_multi_symbol_non_multiindex_returns_empty(self):
        """Line 292->294: not MultiIndex and len(symbols)>1 → elif False → return {}."""
        from data.market_data import _bulk_download

        idx = pd.bdate_range("2025-01-01", periods=5)
        raw = pd.DataFrame({"Open": [1.0] * 5, "Close": [2.0] * 5}, index=idx)

        with patch("data.market_data.yf.download", return_value=raw):
            result = _bulk_download(["AAPL", "NVDA"], 200)

        self.assertEqual(result, {})


class TestGetIntradayDataOrbBranches(unittest.TestCase):
    """Branches 459->475, 463->475, 477->485 in get_intraday_data."""

    def _make_alpaca_bar(self, ts_et, open_=100.0, high=101.0, low=99.0, close=100.5, vol=50_000):
        bar = MagicMock()
        bar.timestamp = ts_et
        bar.open = open_
        bar.high = high
        bar.low = low
        bar.close = close
        bar.volume = vol
        return bar

    def _patch_alpaca(self, bars_by_sym: dict):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.data = bars_by_sym
        mock_client.get_stock_bars.return_value = mock_resp
        return patch("data.market_data.StockHistoricalDataClient", return_value=mock_client)

    def _fake_now_et(self, hour=11, minute=0):
        _ET = __import__("zoneinfo").ZoneInfo("America/New_York")
        return datetime(2025, 6, 10, hour, minute, 0, tzinfo=_ET)

    def test_all_bars_after_orb_end_gives_empty_orb_bars(self):
        """Line 459->475: orb_bars.empty (all bars after ORB end at 10:00) → skip orb block."""
        from data.market_data import get_intraday_data

        _ET = __import__("zoneinfo").ZoneInfo("America/New_York")
        fake_now = self._fake_now_et(hour=11)
        bar_ts = datetime(2025, 6, 10, 10, 30, 0, tzinfo=_ET)
        bar = self._make_alpaca_bar(bar_ts)

        with (
            self._patch_alpaca({"AAPL": [bar]}),
            patch("data.market_data.datetime") as mock_dt,
        ):
            mock_dt.now.return_value = fake_now
            mock_dt.strptime = datetime.strptime
            result = get_intraday_data(["AAPL"])

        self.assertIn("AAPL", result)
        self.assertIsNone(result["AAPL"]["orb_high"])
        self.assertIsNone(result["AAPL"]["orb_low"])

    def test_no_post_orb_bars_skips_breakout_check(self):
        """Line 463->475: post_orb.empty (all bars within ORB) → skip breakout calculation."""
        from data.market_data import get_intraday_data

        _ET = __import__("zoneinfo").ZoneInfo("America/New_York")
        fake_now = self._fake_now_et(hour=11)
        bar_ts = datetime(2025, 6, 10, 9, 35, 0, tzinfo=_ET)
        bar = self._make_alpaca_bar(bar_ts)

        with (
            self._patch_alpaca({"AAPL": [bar]}),
            patch("data.market_data.datetime") as mock_dt,
        ):
            mock_dt.now.return_value = fake_now
            mock_dt.strptime = datetime.strptime
            result = get_intraday_data(["AAPL"])

        self.assertIn("AAPL", result)
        self.assertFalse(result["AAPL"]["orb_breakout_up"])
        self.assertFalse(result["AAPL"]["orb_breakout_down"])

    def test_fewer_than_15_five_min_bars_skips_rsi(self):
        """Line 477->485: len(five_min) < 15 → intraday_rsi stays None."""
        from data.market_data import get_intraday_data

        _ET = __import__("zoneinfo").ZoneInfo("America/New_York")
        fake_now = self._fake_now_et(hour=10, minute=5)
        bar_ts = datetime(2025, 6, 10, 9, 35, 0, tzinfo=_ET)
        bar = self._make_alpaca_bar(bar_ts)

        with (
            self._patch_alpaca({"AAPL": [bar]}),
            patch("data.market_data.datetime") as mock_dt,
        ):
            mock_dt.now.return_value = fake_now
            mock_dt.strptime = datetime.strptime
            result = get_intraday_data(["AAPL"])

        self.assertIn("AAPL", result)
        self.assertIsNone(result["AAPL"]["intraday_rsi"])


# ── Same-day bulk cache ───────────────────────────────────────────────────────


def _make_sym_df(periods: int = 5) -> pd.DataFrame:
    idx = pd.bdate_range("2025-01-01", periods=periods)
    return pd.DataFrame(
        {"Close": [100.0 + i for i in range(periods)], "Volume": [1e6] * periods}, index=idx
    )


class TestBulkCachePath(unittest.TestCase):
    def test_path_contains_todays_et_date(self):
        from zoneinfo import ZoneInfo

        from data.market_data import _bulk_cache_path

        expected_date = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
        path = _bulk_cache_path()
        self.assertIn(expected_date, path)
        self.assertTrue(path.endswith(".pkl"))


class TestLoadBulkCache(unittest.TestCase):
    def test_returns_empty_when_file_missing(self):
        from data.market_data import _load_bulk_cache

        with patch("data.market_data._bulk_cache_path", return_value="/nonexistent/path.pkl"):
            result = _load_bulk_cache()
        self.assertEqual(result, {})

    def test_returns_data_when_valid_cache_exists(self):
        from data.market_data import _load_bulk_cache

        data = {"AAPL": _make_sym_df()}
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp = f.name
        try:
            with patch("data.market_data._bulk_cache_path", return_value=tmp):
                result = _load_bulk_cache()
            self.assertIn("AAPL", result)
        finally:
            os.unlink(tmp)

    def test_returns_empty_on_corrupt_file(self):
        from data.market_data import _load_bulk_cache

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            f.write(b"not-a-pickle")
            tmp = f.name
        try:
            with patch("data.market_data._bulk_cache_path", return_value=tmp):
                result = _load_bulk_cache()
            self.assertEqual(result, {})
        finally:
            os.unlink(tmp)

    def test_returns_empty_when_pickle_not_a_dict(self):
        from data.market_data import _load_bulk_cache

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(["not", "a", "dict"], f)
            tmp = f.name
        try:
            with patch("data.market_data._bulk_cache_path", return_value=tmp):
                result = _load_bulk_cache()
            self.assertEqual(result, {})
        finally:
            os.unlink(tmp)


class TestSaveBulkCache(unittest.TestCase):
    def test_writes_pickle_file(self):
        from data.market_data import _save_bulk_cache

        data = {"AAPL": _make_sym_df()}
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = os.path.join(tmpdir, "cache.pkl")
            with (
                patch("data.market_data._bulk_cache_path", return_value=tmp_path),
                patch("data.market_data.LOG_DIR", tmpdir),
            ):
                _save_bulk_cache(data)
            self.assertTrue(os.path.exists(tmp_path))
            with open(tmp_path, "rb") as f:
                loaded = pickle.load(f)
            self.assertIn("AAPL", loaded)

    def test_handles_write_error_without_raising(self):
        from data.market_data import _save_bulk_cache

        with (
            patch("data.market_data._bulk_cache_path", return_value="/no/such/dir/cache.pkl"),
            patch("data.market_data.os.makedirs", side_effect=OSError("no perms")),
        ):
            _save_bulk_cache({"AAPL": _make_sym_df()})  # must not raise


class TestBulkDownloadCacheBehavior(unittest.TestCase):
    """_bulk_download cache-aware routing."""

    def test_cache_hit_skips_download(self):
        from data.market_data import _bulk_download

        warm = {"AAPL": _make_sym_df()}
        with (
            patch("data.market_data._load_bulk_cache", return_value=warm),
            patch("data.market_data._download_symbols") as mock_dl,
        ):
            result = _bulk_download(["AAPL"], 200)
        mock_dl.assert_not_called()
        self.assertIn("AAPL", result)

    def test_cache_miss_downloads_and_saves(self):
        from data.market_data import _bulk_download

        with (
            patch("data.market_data._load_bulk_cache", return_value={}),
            patch(
                "data.market_data._download_symbols", return_value={"AAPL": _make_sym_df()}
            ) as mock_dl,
            patch("data.market_data._save_bulk_cache") as mock_save,
        ):
            result = _bulk_download(["AAPL"], 200)
        mock_dl.assert_called_once_with(["AAPL"], 200)
        mock_save.assert_called_once()
        self.assertIn("AAPL", result)

    def test_partial_cache_downloads_only_missing(self):
        from data.market_data import _bulk_download

        warm = {"AAPL": _make_sym_df()}
        with (
            patch("data.market_data._load_bulk_cache", return_value=warm),
            patch(
                "data.market_data._download_symbols", return_value={"NVDA": _make_sym_df()}
            ) as mock_dl,
            patch("data.market_data._save_bulk_cache"),
        ):
            result = _bulk_download(["AAPL", "NVDA"], 200)
        mock_dl.assert_called_once_with(["NVDA"], 200)
        self.assertIn("AAPL", result)
        self.assertIn("NVDA", result)

    def test_download_failure_with_empty_cache_returns_empty(self):
        from data.market_data import _bulk_download

        with (
            patch("data.market_data._load_bulk_cache", return_value={}),
            patch("data.market_data._download_symbols", return_value={}),
            patch("data.market_data._save_bulk_cache"),
        ):
            result = _bulk_download(["AAPL"], 200)
        self.assertEqual(result, {})

    def test_download_failure_with_partial_cache_returns_cached_symbols(self):
        # fresh=empty, cache has AAPL → elif not cache: is False → return partial cache silently
        from data.market_data import _bulk_download

        warm = {"AAPL": _make_sym_df()}
        with (
            patch("data.market_data._load_bulk_cache", return_value=warm),
            patch("data.market_data._download_symbols", return_value={}),
            patch("data.market_data._save_bulk_cache"),
        ):
            result = _bulk_download(["AAPL", "NVDA"], 200)
        self.assertIn("AAPL", result)
        self.assertNotIn("NVDA", result)


class TestPrefetchMarketData(unittest.TestCase):
    def test_downloads_when_cache_empty(self):
        from data.market_data import prefetch_market_data

        with (
            patch("data.market_data._load_bulk_cache", return_value={}),
            patch(
                "data.market_data._download_symbols", return_value={"AAPL": _make_sym_df()}
            ) as mock_dl,
            patch("data.market_data._save_bulk_cache"),
        ):
            prefetch_market_data(["AAPL"])
        mock_dl.assert_called_once()

    def test_noop_when_all_symbols_cached(self):
        from data.market_data import prefetch_market_data

        warm = {"AAPL": _make_sym_df(), "MSFT": _make_sym_df()}
        with (
            patch("data.market_data._load_bulk_cache", return_value=warm),
            patch("data.market_data._download_symbols") as mock_dl,
        ):
            prefetch_market_data(["AAPL", "MSFT"])
        mock_dl.assert_not_called()

    def test_handles_download_failure_without_raising(self):
        from data.market_data import prefetch_market_data

        with (
            patch("data.market_data._load_bulk_cache", return_value={}),
            patch("data.market_data._download_symbols", return_value={}),
            patch("data.market_data._save_bulk_cache"),
        ):
            prefetch_market_data(["AAPL"])  # must not raise


class TestComputeAmihudIlliquidity(unittest.TestCase):
    def _make_df(self, n: int = 25, volume: float = 1_000_000, price: float = 100.0):

        idx = pd.bdate_range("2024-01-01", periods=n)
        closes = pd.Series([price + i * 0.1 for i in range(n)], index=idx)
        df = pd.DataFrame({"Close": closes, "Volume": volume}, index=idx)
        return df

    def test_returns_positive_float_for_normal_data(self):
        from data.market_data import compute_amihud_illiquidity

        df = self._make_df()
        result = compute_amihud_illiquidity(df)
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0.0)

    def test_returns_zero_for_empty_df(self):
        from data.market_data import compute_amihud_illiquidity

        df = pd.DataFrame({"Close": [], "Volume": []})
        result = compute_amihud_illiquidity(df)
        self.assertEqual(result, 0.0)

    def test_returns_zero_for_zero_volume(self):
        from data.market_data import compute_amihud_illiquidity

        df = self._make_df(volume=0)
        result = compute_amihud_illiquidity(df)
        self.assertEqual(result, 0.0)

    def test_higher_volume_yields_lower_illiquidity(self):
        from data.market_data import compute_amihud_illiquidity

        low_vol = self._make_df(volume=10_000)
        high_vol = self._make_df(volume=10_000_000)
        self.assertGreater(
            compute_amihud_illiquidity(low_vol), compute_amihud_illiquidity(high_vol)
        )

    def test_returns_zero_for_insufficient_bars(self):
        from data.market_data import compute_amihud_illiquidity

        df = self._make_df(n=3)
        result = compute_amihud_illiquidity(df)
        self.assertEqual(result, 0.0)


# ── Batch 1 summarise_for_ai fields ──────────────────────────────────────────


class TestSummariseForAIBatch1Fields(unittest.TestCase):
    """summarise_for_ai returns Batch 1 OHLCV signal fields (v1.94)."""

    def _make_df_with_batch1(self):
        """Minimal DataFrame with Batch 1 indicator columns pre-computed."""
        closes = [100.0, 101.0, 102.0]
        df = _make_df(close_vals=closes)
        df["golden_cross"] = [False, False, True]
        df["death_cross"] = [False, False, False]
        df["obv_divergence_bull"] = [False, False, True]
        df["obv_divergence_bear"] = [False, False, False]
        df["obv_accelerating_up"] = [False, True, True]
        df["obv_accelerating_down"] = [False, False, False]
        df["near_20d_low"] = [False, False, True]
        df["near_20d_high"] = [False, False, False]
        df["hammer"] = [False, False, True]
        df["bullish_engulf"] = [False, False, False]
        df["shooting_star"] = [False, False, False]
        df["bearish_engulf"] = [False, False, False]
        df["high_vol_streak"] = [0, 0, 3]
        return df

    def test_batch1_keys_present_in_output(self):
        from data.market_data import summarise_for_ai

        result = summarise_for_ai("AAPL", self._make_df_with_batch1())
        for key in (
            "golden_cross",
            "death_cross",
            "obv_divergence_bull",
            "obv_divergence_bear",
            "obv_accelerating_up",
            "obv_accelerating_down",
            "near_20d_low",
            "near_20d_high",
            "hammer",
            "bullish_engulf",
            "shooting_star",
            "bearish_engulf",
            "high_vol_streak",
        ):
            self.assertIn(key, result, f"Missing key: {key}")

    def test_batch1_values_extracted_correctly(self):
        from data.market_data import summarise_for_ai

        result = summarise_for_ai("AAPL", self._make_df_with_batch1())
        self.assertTrue(result["golden_cross"])
        self.assertFalse(result["death_cross"])
        self.assertTrue(result["obv_divergence_bull"])
        self.assertTrue(result["near_20d_low"])
        self.assertTrue(result["hammer"])
        self.assertEqual(result["high_vol_streak"], 3)

    def test_batch1_defaults_to_false_when_columns_absent(self):
        from data.market_data import summarise_for_ai

        result = summarise_for_ai("AAPL", _make_df())
        for key in ("golden_cross", "death_cross", "hammer", "near_20d_low"):
            self.assertFalse(result[key], f"{key} should default to False")
        self.assertEqual(result["high_vol_streak"], 0)


# ── Batch 2 summarise_for_ai fields ──────────────────────────────────────────


class TestSummariseForAIBatch2Fields(unittest.TestCase):
    """summarise_for_ai returns Batch 2 OHLCV signal fields (v1.95)."""

    def _make_df_with_batch2(self):
        df = _make_df()
        df["spread_proxy_20d"] = 0.006
        return df

    def test_spread_proxy_20d_key_present(self):
        from data.market_data import summarise_for_ai

        result = summarise_for_ai("AAPL", self._make_df_with_batch2())
        self.assertIn("spread_proxy_20d", result)

    def test_spread_proxy_20d_value_extracted(self):
        from data.market_data import summarise_for_ai

        result = summarise_for_ai("AAPL", self._make_df_with_batch2())
        self.assertAlmostEqual(result["spread_proxy_20d"], 0.006, places=4)

    def test_spread_proxy_20d_defaults_to_zero_when_absent(self):
        from data.market_data import summarise_for_ai

        result = summarise_for_ai("AAPL", _make_df())
        self.assertAlmostEqual(result["spread_proxy_20d"], 0.0)

    def test_spread_proxy_20d_is_float(self):
        from data.market_data import summarise_for_ai

        result = summarise_for_ai("AAPL", self._make_df_with_batch2())
        self.assertIsInstance(result["spread_proxy_20d"], float)

    def test_breadth_thrust_injection_exception_falls_back(self):
        """Lines 678-682: get_breadth_snapshot raising → setdefault fallback applied."""
        from data.market_data import get_market_snapshots

        snap = {
            "symbol": "AAPL",
            "current_price": 100.0,
            "ret_1d_pct": 0.5,
            "ret_5d_pct": 2.0,
            "ret_10d_pct": 4.0,
            "rsi_14": 55.0,
            "macd_diff": 0.1,
            "macd_crossed_up": False,
            "macd_crossed_down": False,
            "ema9_above_ema21": True,
            "bb_pct": 0.5,
            "vol_ratio": 1.2,
            "price_vs_ema9_pct": 1.0,
            "weekly_trend_up": True,
            "weekly_rsi": 55.0,
        }
        with (
            patch("data.market_data._bulk_download", return_value={"AAPL": MagicMock()}),
            patch("data.market_data.get_fundamentals", return_value={}),
            patch("data.market_data.fetch_stock_data", return_value=MagicMock()),
            patch("data.market_data.summarise_for_ai", return_value=snap),
            patch("data.market_data.get_spy_5d_return", return_value=None),
            patch("data.market_data.get_spy_10d_return", return_value=None),
            patch("data.market_data.get_spy_20d_return", return_value=None),
            patch("data.breadth.get_breadth_snapshot", side_effect=RuntimeError("feed down")),
        ):
            result = get_market_snapshots(["AAPL"])
        self.assertEqual(len(result), 1)
        self.assertFalse(result[0].get("breadth_thrust", True))
        self.assertEqual(result[0].get("breadth_symbols_counted", -1), 0)
        self.assertAlmostEqual(result[0].get("nhl_ratio", -1.0), 1.0)


class TestBreadthNHLInjection(unittest.TestCase):
    """NHL ratio is injected alongside breadth_thrust in get_market_snapshots()."""

    def _base_snap(self):
        return {
            "symbol": "AAPL",
            "current_price": 100.0,
            "ret_1d_pct": 0.5,
            "ret_5d_pct": 2.0,
            "ret_10d_pct": 4.0,
            "rsi_14": 55.0,
            "macd_diff": 0.1,
            "macd_crossed_up": False,
            "macd_crossed_down": False,
            "ema9_above_ema21": True,
            "bb_pct": 0.5,
            "vol_ratio": 1.2,
            "price_vs_ema9_pct": 1.0,
            "weekly_trend_up": True,
            "weekly_rsi": 55.0,
        }

    def test_nhl_ratio_injected_from_breadth_snapshot(self):
        from unittest.mock import MagicMock

        from data.market_data import get_market_snapshots

        mock_breadth = MagicMock()
        mock_breadth.breadth_thrust = True
        mock_breadth.symbols_counted = 80
        mock_breadth.nh_nl_ratio = 2.5

        with (
            patch("data.market_data._bulk_download", return_value={"AAPL": MagicMock()}),
            patch("data.market_data.get_fundamentals", return_value={}),
            patch("data.market_data.fetch_stock_data", return_value=MagicMock()),
            patch("data.market_data.summarise_for_ai", return_value=self._base_snap()),
            patch("data.market_data.get_spy_5d_return", return_value=None),
            patch("data.market_data.get_spy_10d_return", return_value=None),
            patch("data.market_data.get_spy_20d_return", return_value=None),
            patch("data.breadth.get_breadth_snapshot", return_value=mock_breadth),
            patch("data.sector_correlation.compute_stock_sector_corr", return_value=None),
            patch("data.sector_data.get_sector_etf", return_value=None),
        ):
            result = get_market_snapshots(["AAPL"])

        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0].get("nhl_ratio"), 2.5)
        self.assertTrue(result[0].get("breadth_thrust"))


class TestSectorCorrelationInjection(unittest.TestCase):
    """sector_correlation_20d is injected per-symbol in get_market_snapshots()."""

    def _base_snap(self, symbol="AAPL"):
        return {
            "symbol": symbol,
            "current_price": 100.0,
            "ret_1d_pct": 0.5,
            "ret_5d_pct": 2.0,
            "ret_10d_pct": 4.0,
            "rsi_14": 55.0,
            "macd_diff": 0.1,
            "macd_crossed_up": False,
            "macd_crossed_down": False,
            "ema9_above_ema21": True,
            "bb_pct": 0.5,
            "vol_ratio": 1.2,
            "price_vs_ema9_pct": 1.0,
            "weekly_trend_up": True,
            "weekly_rsi": 55.0,
        }

    def _std_patches(self, snap):
        from unittest.mock import MagicMock

        mock_breadth = MagicMock()
        mock_breadth.breadth_thrust = False
        mock_breadth.symbols_counted = 0
        mock_breadth.nh_nl_ratio = 1.0
        return [
            patch("data.market_data._bulk_download", return_value={"AAPL": MagicMock()}),
            patch("data.market_data.get_fundamentals", return_value={}),
            patch("data.market_data.fetch_stock_data", return_value=MagicMock()),
            patch("data.market_data.summarise_for_ai", return_value=snap),
            patch("data.market_data.get_spy_5d_return", return_value=None),
            patch("data.market_data.get_spy_10d_return", return_value=None),
            patch("data.market_data.get_spy_20d_return", return_value=None),
            patch("data.breadth.get_breadth_snapshot", return_value=mock_breadth),
            patch("data.sentiment_client.get_aaii_sentiment", return_value=_AAII_FIXTURE),
        ]

    def test_correlation_injected_when_etf_found_and_compute_returns_float(self):
        from data.market_data import get_market_snapshots

        patches = self._std_patches(self._base_snap())
        patches += [
            patch("data.sector_data.get_sector_etf", return_value="XLK"),
            patch("data.sector_correlation.compute_stock_sector_corr", return_value=0.82),
        ]
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
            patches[8],
            patches[9],
        ):
            result = get_market_snapshots(["AAPL"])

        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0].get("sector_correlation_20d"), 0.82)

    def test_no_injection_when_get_sector_etf_returns_none(self):
        from data.market_data import get_market_snapshots

        patches = self._std_patches(self._base_snap())
        patches += [
            patch("data.sector_data.get_sector_etf", return_value=None),
            patch("data.sector_correlation.compute_stock_sector_corr", return_value=0.70),
        ]
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
            patches[8],
            patches[9],
        ):
            result = get_market_snapshots(["AAPL"])

        self.assertEqual(len(result), 1)
        self.assertNotIn("sector_correlation_20d", result[0])

    def test_no_injection_when_compute_returns_none(self):
        from data.market_data import get_market_snapshots

        patches = self._std_patches(self._base_snap())
        patches += [
            patch("data.sector_data.get_sector_etf", return_value="XLK"),
            patch("data.sector_correlation.compute_stock_sector_corr", return_value=None),
        ]
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
            patches[8],
            patches[9],
        ):
            result = get_market_snapshots(["AAPL"])

        self.assertEqual(len(result), 1)
        self.assertNotIn("sector_correlation_20d", result[0])

    def test_exception_in_correlation_block_swallowed(self):
        from data.market_data import get_market_snapshots

        patches = self._std_patches(self._base_snap())
        patches += [
            patch("data.sector_data.get_sector_etf", side_effect=RuntimeError("sector down")),
        ]
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
            patches[8],
        ):
            result = get_market_snapshots(["AAPL"])

        self.assertEqual(len(result), 1)
        self.assertNotIn("sector_correlation_20d", result[0])


class TestPremarketGapRetrace(unittest.TestCase):
    """premarket_gap_retrace field in get_intraday_data() result."""

    def _make_alpaca_bar(self, ts_et, open_=100.0, high=101.0, low=99.0, close=100.5, vol=50_000):
        bar = MagicMock()
        bar.timestamp = ts_et
        bar.open = open_
        bar.high = high
        bar.low = low
        bar.close = close
        bar.volume = vol
        return bar

    def _patch_alpaca(self, bars_by_sym: dict):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.data = bars_by_sym
        mock_client.get_stock_bars.return_value = mock_resp
        return patch("data.market_data.StockHistoricalDataClient", return_value=mock_client)

    def _fake_now_et(self, hour=10, minute=30):
        _ET = __import__("zoneinfo").ZoneInfo("America/New_York")
        return datetime(2025, 6, 10, hour, minute, 0, tzinfo=_ET)

    def _build_bars(self, prev_close, day_open, close_at_935):
        """Build a pre-market bar + 5 regular-session bars for a gap scenario."""
        _ET = __import__("zoneinfo").ZoneInfo("America/New_York")
        # Pre-market bar at 09:00
        bars = [
            self._make_alpaca_bar(
                datetime(2025, 6, 10, 9, 0, 0, tzinfo=_ET),
                open_=prev_close,
                high=prev_close,
                low=prev_close,
                close=prev_close,
            )
        ]
        # 5 regular-session minute bars starting at 09:30
        for m in range(5):
            ts = datetime(2025, 6, 10, 9, 30 + m, 0, tzinfo=_ET)
            close = close_at_935 if m == 4 else day_open
            bars.append(
                self._make_alpaca_bar(
                    ts, open_=day_open, high=day_open + 0.5, low=day_open - 0.5, close=close
                )
            )
        return bars

    def test_gap_retrace_true_when_more_than_50pct_filled(self):
        """Gap-up of $2, price at 09:35 has fallen $1.10 (55% retrace) → True."""
        from data.market_data import get_intraday_data

        _ET = __import__("zoneinfo").ZoneInfo("America/New_York")
        fake_now = self._fake_now_et()
        prev_close = 100.0
        day_open = 102.0  # +2% gap
        close_935 = 100.9  # fell $1.10 of $2.00 gap (55% retrace)
        bars = self._build_bars(prev_close, day_open, close_935)

        with (
            self._patch_alpaca({"AAPL": bars}),
            patch("data.market_data.datetime") as mock_dt,
        ):
            mock_dt.now.return_value = fake_now
            mock_dt.strptime = datetime.strptime
            result = get_intraday_data(["AAPL"])

        self.assertIn("AAPL", result)
        self.assertTrue(result["AAPL"]["premarket_gap_retrace"])

    def test_gap_retrace_false_when_gap_holds(self):
        """Gap-up of $2, price at 09:35 only fell $0.50 (25% retrace) → False."""
        from data.market_data import get_intraday_data

        _ET = __import__("zoneinfo").ZoneInfo("America/New_York")
        fake_now = self._fake_now_et()
        prev_close = 100.0
        day_open = 102.0
        close_935 = 101.5  # fell $0.50 (25% retrace)
        bars = self._build_bars(prev_close, day_open, close_935)

        with (
            self._patch_alpaca({"AAPL": bars}),
            patch("data.market_data.datetime") as mock_dt,
        ):
            mock_dt.now.return_value = fake_now
            mock_dt.strptime = datetime.strptime
            result = get_intraday_data(["AAPL"])

        self.assertIn("AAPL", result)
        self.assertFalse(result["AAPL"]["premarket_gap_retrace"])

    def test_no_significant_gap_sets_retrace_false(self):
        """Gap < 2% (0.5%) → premarket_gap_retrace stays False."""
        from data.market_data import get_intraday_data

        _ET = __import__("zoneinfo").ZoneInfo("America/New_York")
        fake_now = self._fake_now_et()
        prev_close = 100.0
        day_open = 100.5  # only 0.5% gap — below the 2% threshold
        close_935 = 99.0  # would be a retrace but gap_pct < 2 so gate doesn't fire
        bars = self._build_bars(prev_close, day_open, close_935)

        with (
            self._patch_alpaca({"AAPL": bars}),
            patch("data.market_data.datetime") as mock_dt,
        ):
            mock_dt.now.return_value = fake_now
            mock_dt.strptime = datetime.strptime
            result = get_intraday_data(["AAPL"])

        self.assertIn("AAPL", result)
        self.assertFalse(result["AAPL"]["premarket_gap_retrace"])

    def test_fewer_than_5_bars_sets_retrace_false(self):
        """Only 3 session bars at call time → fewer than 5, retrace check skipped → False."""
        from data.market_data import get_intraday_data

        _ET = __import__("zoneinfo").ZoneInfo("America/New_York")
        fake_now = self._fake_now_et(hour=9, minute=33)
        prev_close = 100.0
        day_open = 104.0  # 4% gap-up
        # Only 3 session bars (09:30, 09:31, 09:32)
        bars = [
            self._make_alpaca_bar(
                datetime(2025, 6, 10, 9, 0, 0, tzinfo=_ET),
                open_=prev_close,
                high=prev_close,
                low=prev_close,
                close=prev_close,
            )
        ]
        for m in range(3):
            ts = datetime(2025, 6, 10, 9, 30 + m, 0, tzinfo=_ET)
            bars.append(
                self._make_alpaca_bar(
                    ts,
                    open_=day_open,
                    high=day_open + 0.5,
                    low=day_open - 0.5,
                    close=day_open - 2.5,
                )
            )

        with (
            self._patch_alpaca({"AAPL": bars}),
            patch("data.market_data.datetime") as mock_dt,
        ):
            mock_dt.now.return_value = fake_now
            mock_dt.strptime = datetime.strptime
            result = get_intraday_data(["AAPL"])

        self.assertIn("AAPL", result)
        self.assertFalse(result["AAPL"]["premarket_gap_retrace"])


class TestNewInjectionBlocks(unittest.TestCase):
    """Tests for the fundamental_cache, AAII, analyst_revisions, and lockup injection blocks."""

    def _base_snap(self):
        return {
            "symbol": "AAPL",
            "current_price": 100.0,
            "ret_1d_pct": 0.5,
            "ret_5d_pct": 2.0,
            "ret_10d_pct": 4.0,
            "rsi_14": 55.0,
            "macd_diff": 0.1,
            "macd_crossed_up": False,
            "macd_crossed_down": False,
            "ema9_above_ema21": True,
            "bb_pct": 0.5,
            "vol_ratio": 1.2,
            "price_vs_ema9_pct": 1.0,
            "weekly_trend_up": True,
            "weekly_rsi": 55.0,
        }

    def _std_patches(self, snap):
        mock_breadth = MagicMock()
        mock_breadth.breadth_thrust = False
        mock_breadth.symbols_counted = 0
        mock_breadth.nh_nl_ratio = 1.0
        return [
            patch("data.market_data._bulk_download", return_value={"AAPL": MagicMock()}),
            patch("data.market_data.get_fundamentals", return_value={}),
            patch("data.market_data.fetch_stock_data", return_value=MagicMock()),
            patch("data.market_data.summarise_for_ai", return_value=snap),
            patch("data.market_data.get_spy_5d_return", return_value=None),
            patch("data.market_data.get_spy_10d_return", return_value=None),
            patch("data.market_data.get_spy_20d_return", return_value=None),
            patch("data.breadth.get_breadth_snapshot", return_value=mock_breadth),
            patch("data.sentiment_client.get_aaii_sentiment", return_value=_AAII_FIXTURE),
        ]

    def test_fundamental_cache_exception_is_swallowed(self):
        """Line 648-649: fundamental_cache injection failure → warning, snapshots still returned."""
        from data.market_data import get_market_snapshots

        patches = self._std_patches(self._base_snap()) + [
            patch(
                "data.fundamental_cache.get_altman_z",
                side_effect=RuntimeError("cache down"),
            ),
        ]
        for p in patches:
            p.__enter__()
        try:
            result = get_market_snapshots(["AAPL"])
        finally:
            for p in reversed(patches):
                p.__exit__(None, None, None)
        self.assertEqual(len(result), 1)

    def test_aaii_10y_exception_is_swallowed(self):
        """Lines 657-658: AAII/10y injection failure → warning, snapshots still returned."""
        from data.market_data import get_market_snapshots

        patches = self._std_patches(self._base_snap()) + [
            patch(
                "data.sentiment_client.get_aaii_sentiment",
                side_effect=RuntimeError("AAII down"),
            ),
        ]
        for p in patches:
            p.__enter__()
        try:
            result = get_market_snapshots(["AAPL"])
        finally:
            for p in reversed(patches):
                p.__exit__(None, None, None)
        self.assertEqual(len(result), 1)

    def test_analyst_revisions_exception_is_swallowed(self):
        """Lines 665-666: analyst_revisions injection failure → warning, snapshots still returned."""
        from data.market_data import get_market_snapshots

        patches = self._std_patches(self._base_snap()) + [
            patch(
                "data.analyst_revisions.get_analyst_revisions",
                side_effect=RuntimeError("revisions down"),
            ),
        ]
        for p in patches:
            p.__enter__()
        try:
            result = get_market_snapshots(["AAPL"])
        finally:
            for p in reversed(patches):
                p.__exit__(None, None, None)
        self.assertEqual(len(result), 1)

    def test_lockup_exception_is_swallowed(self):
        """Lines 673-674: lockup_calendar injection failure → warning, snapshots still returned."""
        from data.market_data import get_market_snapshots

        patches = self._std_patches(self._base_snap()) + [
            patch(
                "data.lockup_calendar.get_lockup_expiry_flags",
                side_effect=RuntimeError("lockup down"),
            ),
        ]
        for p in patches:
            p.__enter__()
        try:
            result = get_market_snapshots(["AAPL"])
        finally:
            for p in reversed(patches):
                p.__exit__(None, None, None)
        self.assertEqual(len(result), 1)

    def test_macro_10y_and_aaii_injected_into_snapshot(self):
        """macro_10y_yield (FRED) + AAII fields (sentiment_client) injected into snapshots."""
        from data.market_data import get_market_snapshots

        # AAII comes from the _std_patches sentiment_client fixture (bearish 0.30, both bools False).
        patches = self._std_patches(self._base_snap()) + [
            patch("data.fred_client.get_10y_yield", return_value=4.25),
        ]
        for p in patches:
            p.__enter__()
        try:
            result = get_market_snapshots(["AAPL"])
        finally:
            for p in reversed(patches):
                p.__exit__(None, None, None)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0].get("macro_10y_yield"), 4.25)
        # AAII mapped from sentiment_client: bearish_pct 0.30 -> 30.0; bools pass through
        self.assertAlmostEqual(result[0].get("aaii_bears_pct"), 30.0)
        self.assertFalse(result[0].get("aaii_extreme_fear"))
        self.assertFalse(result[0].get("aaii_excessive_bulls"))

    def test_lockup_flags_injected_into_snapshot(self):
        """Line 704: lockup_expiry_soon injected when get_lockup_expiry_flags returns entry."""
        from data.market_data import get_market_snapshots

        lockup_data = {
            "AAPL": {"lockup_expiry_soon": True, "days_to_lockup": 7, "ipo_date": "2025-12-15"}
        }
        patches = self._std_patches(self._base_snap()) + [
            patch("data.lockup_calendar.get_lockup_expiry_flags", return_value=lockup_data),
        ]
        for p in patches:
            p.__enter__()
        try:
            result = get_market_snapshots(["AAPL"])
        finally:
            for p in reversed(patches):
                p.__exit__(None, None, None)
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0].get("lockup_expiry_soon"))
        self.assertEqual(result[0].get("days_to_lockup"), 7)
