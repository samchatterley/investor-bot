"""Tests for execution/stock_scanner.py — get_market_regime, prefilter_candidates, get_top_movers."""
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from execution.stock_scanner import get_market_regime, prefilter_candidates


def _spy_history(prices: list[float]) -> pd.DataFrame:
    """Build a minimal history DataFrame from a list of close prices."""
    return pd.DataFrame({"Close": prices, "Volume": [1_000_000] * len(prices)})


def _snap(**kwargs):
    defaults = {
        "symbol": "TEST",
        "rsi_14": 50, "bb_pct": 0.5, "vol_ratio": 1.0,
        "ema9_above_ema21": False, "macd_diff": 0, "macd_crossed_up": False,
        "weekly_trend_up": True, "ret_5d_pct": 0,
        "avg_volume": 1_000_000,  # above MIN_VOLUME = 500_000
    }
    defaults.update(kwargs)
    return defaults


class TestGetMarketRegime(unittest.TestCase):

    def _mock_spy(self, prices):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = _spy_history(prices)
        return patch("execution.stock_scanner.yf.Ticker", return_value=mock_ticker)

    def test_bull_trending_regime(self):
        # 5-day gain > 2%, today positive
        prices = [100, 101, 102, 103, 104, 106]
        with self._mock_spy(prices):
            result = get_market_regime(threshold_pct=-1.5)
        self.assertEqual(result["regime"], "BULL_TRENDING")
        self.assertFalse(result["is_bearish"])

    def test_bear_day_regime(self):
        # Today drops > 1.5%
        prices = [100, 100, 100, 100, 100, 98]
        with self._mock_spy(prices):
            result = get_market_regime(threshold_pct=-1.5)
        self.assertEqual(result["regime"], "BEAR_DAY")
        self.assertTrue(result["is_bearish"])

    def test_high_vol_regime(self):
        # VIX > 25, 5-day return < -3%
        prices = [100, 100, 100, 100, 100, 96]
        # spy_1d = -4% → triggers BEAR_DAY first; we need 1d flat but 5d down
        prices = [100, 96, 96, 96, 96, 96]
        with self._mock_spy(prices):
            result = get_market_regime(threshold_pct=-1.5, vix=30.0)
        # 1d = 0%, 5d ≈ -4%, VIX 30 → HIGH_VOL
        self.assertEqual(result["regime"], "HIGH_VOL")

    def test_choppy_regime(self):
        prices = [100, 100, 100, 100, 100, 100]
        with self._mock_spy(prices):
            result = get_market_regime(threshold_pct=-1.5)
        self.assertEqual(result["regime"], "CHOPPY")

    def test_insufficient_history_returns_unknown(self):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = _spy_history([100, 101])
        with patch("execution.stock_scanner.yf.Ticker", return_value=mock_ticker):
            result = get_market_regime()
        self.assertEqual(result["regime"], "UNKNOWN")

    def test_exception_returns_unknown(self):
        with patch("execution.stock_scanner.yf.Ticker", side_effect=Exception("network")):
            result = get_market_regime()
        self.assertEqual(result["regime"], "UNKNOWN")
        self.assertFalse(result["is_bearish"])

    def test_result_has_required_keys(self):
        prices = [100] * 6
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = _spy_history(prices)
        with patch("execution.stock_scanner.yf.Ticker", return_value=mock_ticker):
            result = get_market_regime()
        for key in ("is_bearish", "spy_change_pct", "spy_5d_pct", "regime"):
            self.assertIn(key, result)


class TestPrefilterCandidates(unittest.TestCase):

    def test_momentum_signal_passes(self):
        snap = _snap(ema9_above_ema21=True, macd_diff=0.5, ret_5d_pct=2.0, vol_ratio=1.5)
        result = prefilter_candidates([snap])
        self.assertEqual(len(result), 1)

    def test_mean_reversion_signal_passes(self):
        snap = _snap(rsi_14=30, bb_pct=0.20, vol_ratio=1.2)
        result = prefilter_candidates([snap])
        self.assertEqual(len(result), 1)

    def test_macd_crossover_passes(self):
        snap = _snap(macd_crossed_up=True, vol_ratio=1.5)
        result = prefilter_candidates([snap])
        self.assertEqual(len(result), 1)

    def test_no_signal_filtered_out(self):
        snap = _snap()  # all defaults — no signal
        result = prefilter_candidates([snap])
        self.assertEqual(len(result), 0)

    def test_against_weekly_trend_filtered(self):
        # Momentum setup but weekly trend is down
        snap = _snap(ema9_above_ema21=True, macd_diff=0.5, ret_5d_pct=2.0,
                     vol_ratio=1.5, weekly_trend_up=False)
        result = prefilter_candidates([snap])
        self.assertEqual(len(result), 0)

    def test_deeply_oversold_bypasses_weekly_trend(self):
        # RSI < 30, BB < 0.15 → allowed even against weekly trend
        snap = _snap(rsi_14=25, bb_pct=0.10, vol_ratio=1.5, weekly_trend_up=False)
        result = prefilter_candidates([snap])
        self.assertEqual(len(result), 1)

    def test_illiquid_stock_filtered_by_min_volume(self):
        snap = _snap(rsi_14=30, bb_pct=0.20, vol_ratio=1.2, avg_volume=100_000)
        result = prefilter_candidates([snap])
        self.assertEqual(len(result), 0)

    def test_empty_input_returns_empty(self):
        self.assertEqual(prefilter_candidates([]), [])

    def test_mixed_batch_filters_correctly(self):
        good = _snap(symbol="GOOD", rsi_14=30, bb_pct=0.20, vol_ratio=1.2)
        bad = _snap(symbol="BAD")
        result = prefilter_candidates([good, bad])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["symbol"], "GOOD")


class TestGetTopMovers(unittest.TestCase):

    def _make_data(self, symbols, n_rows=5):
        idx = pd.date_range("2026-01-01", periods=n_rows, freq="B")
        closes = pd.DataFrame(
            {sym: [100 + i for i in range(n_rows)] for sym in symbols}, index=idx
        )
        volumes = pd.DataFrame(
            {sym: [1_000_000] * n_rows for sym in symbols}, index=idx
        )
        mock = MagicMock()
        mock.empty = False
        mock.__len__ = MagicMock(return_value=n_rows)
        mock.__getitem__ = lambda self, key: closes if key == "Close" else volumes
        return mock

    def test_returns_list_on_success(self):
        from execution.stock_scanner import get_top_movers
        syms = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN"]
        with patch("execution.stock_scanner.yf.download", return_value=self._make_data(syms)):
            result = get_top_movers(n=3)
        self.assertIsInstance(result, list)

    def test_returns_empty_on_exception(self):
        from execution.stock_scanner import get_top_movers
        with patch("execution.stock_scanner.yf.download", side_effect=Exception("network error")):
            result = get_top_movers()
        self.assertEqual(result, [])

    def test_returns_empty_on_empty_data(self):
        from execution.stock_scanner import get_top_movers
        mock = MagicMock()
        mock.empty = True
        with patch("execution.stock_scanner.yf.download", return_value=mock):
            result = get_top_movers()
        self.assertEqual(result, [])

    def test_returns_empty_when_insufficient_rows(self):
        from execution.stock_scanner import get_top_movers
        mock = MagicMock()
        mock.empty = False
        mock.__len__ = MagicMock(return_value=1)
        with patch("execution.stock_scanner.yf.download", return_value=mock):
            result = get_top_movers()
        self.assertEqual(result, [])
