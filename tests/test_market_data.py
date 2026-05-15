"""Tests for data/market_data.py — summarise_for_ai and yfinance helpers."""

import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd


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


class TestBulkDownload(unittest.TestCase):
    """_bulk_download: single API call returns per-symbol DataFrames."""

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


class TestBulkDownloadKeyError(unittest.TestCase):
    """Lines 290-291: KeyError in xs() is swallowed silently."""

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
            return original_xs(key, **kwargs)

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
            return original_sort(self_df, *args, **kwargs)

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
