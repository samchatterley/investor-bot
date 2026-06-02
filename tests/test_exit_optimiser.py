import unittest
from unittest.mock import patch

import pandas as pd

from risk.exit_optimiser import (
    adverse_volume_triggered,
    compute_atr_pct,
    compute_exit_levels,
    profit_acceleration_triggered,
    rs_decay_triggered,
)


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


class TestRsDecayTriggered(unittest.TestCase):
    def test_no_decay_returns_false(self):
        # entry=70, current=65 → drop of 5, below default threshold of 25
        self.assertFalse(rs_decay_triggered(65.0, 70.0))

    def test_decay_equals_threshold_returns_true(self):
        # entry=75, current=50 → drop of exactly 25
        self.assertTrue(rs_decay_triggered(50.0, 75.0))

    def test_decay_exceeds_threshold_returns_true(self):
        # entry=80, current=50 → drop of 30 > 25
        self.assertTrue(rs_decay_triggered(50.0, 80.0))

    def test_entry_80_current_50_default_threshold(self):
        self.assertTrue(rs_decay_triggered(50.0, 80.0))

    def test_custom_threshold_not_triggered(self):
        # entry=80, current=60 → drop of 20; custom threshold=30 → no trigger
        self.assertFalse(rs_decay_triggered(60.0, 80.0, decay_threshold=30.0))

    def test_custom_threshold_triggered(self):
        # entry=80, current=60 → drop of 20; custom threshold=10 → triggers
        self.assertTrue(rs_decay_triggered(60.0, 80.0, decay_threshold=10.0))


class TestAdverseVolumeTriggered(unittest.TestCase):
    def test_both_days_qualify_returns_true(self):
        self.assertTrue(
            adverse_volume_triggered(
                vol_ratio_today=3.0,
                day_return_today=-2.0,
                vol_ratio_yesterday=2.8,
                day_return_yesterday=-1.8,
            )
        )

    def test_only_today_qualifies_returns_false(self):
        self.assertFalse(
            adverse_volume_triggered(
                vol_ratio_today=3.0,
                day_return_today=-2.0,
                vol_ratio_yesterday=2.8,
                day_return_yesterday=0.5,  # positive return — not adverse
            )
        )

    def test_only_yesterday_qualifies_returns_false(self):
        self.assertFalse(
            adverse_volume_triggered(
                vol_ratio_today=3.0,
                day_return_today=0.3,  # positive return — not adverse
                vol_ratio_yesterday=2.8,
                day_return_yesterday=-2.0,
            )
        )

    def test_neither_qualifies_returns_false(self):
        self.assertFalse(
            adverse_volume_triggered(
                vol_ratio_today=1.2,
                day_return_today=0.5,
                vol_ratio_yesterday=1.0,
                day_return_yesterday=0.1,
            )
        )

    def test_vol_below_threshold_today_returns_false(self):
        # return is adverse but vol ratio is below threshold
        self.assertFalse(
            adverse_volume_triggered(
                vol_ratio_today=1.5,  # below 2.5
                day_return_today=-2.0,
                vol_ratio_yesterday=3.0,
                day_return_yesterday=-2.0,
            )
        )


class TestProfitAccelerationTriggered(unittest.TestCase):
    def test_non_fast_signal_returns_hold(self):
        result = profit_acceleration_triggered(10.0, 1, "momentum")
        self.assertEqual(result, "hold")

    def test_fast_signal_day_2_nine_percent_gain_returns_full_exit(self):
        result = profit_acceleration_triggered(9.0, 2, "mean_reversion")
        self.assertEqual(result, "full_exit")

    def test_fast_signal_day_1_six_percent_gain_returns_partial_exit(self):
        result = profit_acceleration_triggered(6.0, 1, "mean_reversion")
        self.assertEqual(result, "partial_exit")

    def test_fast_signal_day_3_ten_percent_gain_returns_hold(self):
        # days_held=3 > 2, so full_exit rule doesn't fire; days_held=3 > 1, partial_exit doesn't fire
        result = profit_acceleration_triggered(10.0, 3, "mean_reversion")
        self.assertEqual(result, "hold")

    def test_fast_signal_small_gain_returns_hold(self):
        result = profit_acceleration_triggered(3.0, 1, "range_reversion")
        self.assertEqual(result, "hold")

    def test_range_reversion_day_2_eight_percent_returns_full_exit(self):
        result = profit_acceleration_triggered(8.0, 2, "range_reversion")
        self.assertEqual(result, "full_exit")

    def test_custom_fast_exit_signals_respected(self):
        custom = frozenset({"gap_and_go"})
        # gap_and_go is in custom set → should evaluate normally
        result = profit_acceleration_triggered(9.0, 1, "gap_and_go", fast_exit_signals=custom)
        self.assertEqual(result, "full_exit")

    def test_signal_not_in_custom_set_returns_hold(self):
        custom = frozenset({"gap_and_go"})
        result = profit_acceleration_triggered(9.0, 1, "mean_reversion", fast_exit_signals=custom)
        self.assertEqual(result, "hold")
