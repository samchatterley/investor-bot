import unittest
from unittest.mock import patch

import pandas as pd

from risk.exit_optimiser import compute_atr_pct, compute_exit_levels


class TestComputeExitLevels(unittest.TestCase):
    def _levels(self, stop=0.07, tp=0.20, atr=None, days=1, max_hold=5):
        return compute_exit_levels(stop, tp, atr, days, max_hold)

    # ── partial_pct ───────────────────────────────────────────────────────────

    def test_partial_pct_is_2x_stop_when_no_atr(self):
        levels = self._levels(stop=0.07, atr=None)
        self.assertAlmostEqual(levels["partial_pct"], 14.0)

    def test_partial_pct_uses_atr_when_larger(self):
        levels = self._levels(stop=0.05, atr=15.0)  # 2×5=10 < 15
        self.assertAlmostEqual(levels["partial_pct"], 15.0)

    def test_partial_pct_uses_2x_stop_when_larger_than_atr(self):
        levels = self._levels(stop=0.10, atr=5.0)  # 2×10=20 > 5
        self.assertAlmostEqual(levels["partial_pct"], 20.0)

    # ── full_target_pct ───────────────────────────────────────────────────────

    def test_full_target_is_config_tp_when_no_atr(self):
        levels = self._levels(tp=0.20, atr=None)
        self.assertAlmostEqual(levels["full_target_pct"], 20.0)

    def test_full_target_uses_2x_atr_when_larger(self):
        levels = self._levels(tp=0.10, atr=8.0)  # 2×8=16 > 10
        self.assertAlmostEqual(levels["full_target_pct"], 16.0)

    def test_full_target_uses_config_tp_when_larger_than_2x_atr(self):
        levels = self._levels(tp=0.25, atr=5.0)  # 25 > 2×5=10
        self.assertAlmostEqual(levels["full_target_pct"], 25.0)

    # ── stop_pct ──────────────────────────────────────────────────────────────

    def test_stop_pct_is_negative_stop_loss(self):
        levels = self._levels(stop=0.07)
        self.assertAlmostEqual(levels["stop_pct"], -7.0)

    # ── timedecay_stop_pct ────────────────────────────────────────────────────

    def test_timedecay_stop_pct_is_zero(self):
        levels = self._levels()
        self.assertEqual(levels["timedecay_stop_pct"], 0.0)

    # ── apply_timedecay ───────────────────────────────────────────────────────

    def test_apply_timedecay_false_on_day_1_of_5(self):
        levels = self._levels(days=1, max_hold=5)
        self.assertFalse(levels["apply_timedecay"])

    def test_apply_timedecay_true_on_penultimate_day(self):
        levels = self._levels(days=4, max_hold=5)
        self.assertTrue(levels["apply_timedecay"])

    def test_apply_timedecay_true_on_last_day(self):
        levels = self._levels(days=5, max_hold=5)
        self.assertTrue(levels["apply_timedecay"])

    def test_apply_timedecay_true_when_days_exceeds_max(self):
        levels = self._levels(days=10, max_hold=5)
        self.assertTrue(levels["apply_timedecay"])

    def test_apply_timedecay_max_hold_1_triggers_on_day_1(self):
        # max(1, 1-1)=max(1,0)=1 → triggers from day 1
        levels = self._levels(days=1, max_hold=1)
        self.assertTrue(levels["apply_timedecay"])

    # ── rounding ──────────────────────────────────────────────────────────────

    def test_values_are_rounded_to_2dp(self):
        levels = compute_exit_levels(0.073, 0.213, 3.145, 1, 5)
        for key in ("partial_pct", "full_target_pct", "stop_pct"):
            val = levels[key]
            self.assertEqual(val, round(val, 2))


class TestComputeAtrPct(unittest.TestCase):
    def _make_df(self, n=20):
        """Create a minimal DataFrame that mimics yfinance output."""

        dates = pd.date_range("2026-01-01", periods=n)
        high = pd.Series([105.0] * n, index=dates, name="High")
        low = pd.Series([95.0] * n, index=dates, name="Low")
        close = pd.Series([100.0] * n, index=dates, name="Close")
        return pd.DataFrame({"High": high, "Low": low, "Close": close})

    def test_returns_float_for_valid_data(self):
        df = self._make_df(20)
        with patch("yfinance.download", return_value=df):
            result = compute_atr_pct("AAPL")
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)

    def test_returns_none_when_too_few_rows(self):
        df = self._make_df(10)  # < period+1=15 rows
        with patch("yfinance.download", return_value=df):
            result = compute_atr_pct("AAPL")
        self.assertIsNone(result)

    def test_returns_none_on_download_exception(self):
        with patch("yfinance.download", side_effect=RuntimeError("network error")):
            result = compute_atr_pct("AAPL")
        self.assertIsNone(result)

    def test_returns_none_for_empty_dataframe(self):
        with patch("yfinance.download", return_value=pd.DataFrame()):
            result = compute_atr_pct("AAPL")
        self.assertIsNone(result)

    def test_handles_multiindex_columns(self):
        df = self._make_df(20)
        # Wrap in MultiIndex as yfinance sometimes does for single tickers
        df.columns = pd.MultiIndex.from_tuples([(c, "AAPL") for c in df.columns])
        with patch("yfinance.download", return_value=df):
            result = compute_atr_pct("AAPL")
        self.assertIsInstance(result, float)

    def test_returns_none_when_download_returns_none(self):
        with patch("yfinance.download", return_value=None):
            result = compute_atr_pct("AAPL")
        self.assertIsNone(result)

    def test_atr_pct_value_is_reasonable(self):
        # H=105, L=95 → TR=10 always → ATR=10, close=100 → 10%
        df = self._make_df(20)
        with patch("yfinance.download", return_value=df):
            result = compute_atr_pct("AAPL")
        self.assertAlmostEqual(result, 10.0, places=1)

    def test_returns_none_when_price_is_zero(self):

        n = 20
        dates = pd.date_range("2026-01-01", periods=n)
        df = pd.DataFrame(
            {
                "High": pd.Series([1.0] * n, index=dates),
                "Low": pd.Series([0.5] * n, index=dates),
                "Close": pd.Series([0.0] * n, index=dates),  # price = 0 → division guard
            }
        )
        with patch("yfinance.download", return_value=df):
            result = compute_atr_pct("AAPL")
        self.assertIsNone(result)
