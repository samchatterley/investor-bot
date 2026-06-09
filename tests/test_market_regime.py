"""Tests for data/market_regime.py — 5-state market regime classifier."""

import json
import os
import pickle
import tempfile
import unittest
from datetime import date, timedelta
from unittest.mock import patch

import pandas as pd

from data.market_regime import (
    MarketRegime,
    MarketRegimeSnapshot,
    PreviousRegimeState,
    RegimeFeatures,
    RegimeThresholds,
    _load_cache,
    _save_cache,
    apply_regime_hysteresis,
    compute_regime_features,
    compute_regime_series,
    fetch_spy_vix_history,
    get_market_regime,
    load_regime_state,
    resolve_regime,
    save_regime_state,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_spy_df(closes: list[float], start: str = "2024-01-01") -> pd.DataFrame:
    """Build a minimal SPY DataFrame from a list of closing prices."""
    idx = pd.date_range(start=start, periods=len(closes), freq="B")
    return pd.DataFrame({"Close": closes}, index=idx)


def _make_vix_df(closes: list[float], start: str = "2024-01-01") -> pd.DataFrame:
    idx = pd.date_range(start=start, periods=len(closes), freq="B")
    return pd.DataFrame({"Close": closes}, index=idx)


def _spy_df_bull(n: int = 250) -> pd.DataFrame:
    """SPY trending up strongly — should produce BULL_TREND after warmup."""
    base = 400.0
    closes = [base * (1 + i * 0.002) for i in range(n)]
    return _make_spy_df(closes)


def _spy_df_flat(n: int = 250) -> pd.DataFrame:
    """SPY flat — should produce NEUTRAL_CHOP."""
    return _make_spy_df([400.0] * n)


def _features_bull() -> RegimeFeatures:
    return RegimeFeatures(
        spy_ret_1d=0.5,
        spy_ret_5d=3.0,
        spy_ret_20d=5.0,
        spy_above_ma200=True,
        spy_drawdown_pct=-1.0,
        vix=15.0,
        vix_ma20=14.0,
        vix_vs_ma=15.0 / 14.0,
        vix_5d_change=-5.0,
        vix9d=None,
        data_quality="full",
    )


def _features_choppy() -> RegimeFeatures:
    return RegimeFeatures(
        spy_ret_1d=0.1,
        spy_ret_5d=0.5,
        spy_ret_20d=1.0,
        spy_above_ma200=True,
        spy_drawdown_pct=-2.0,
        vix=18.0,
        vix_ma20=17.0,
        vix_vs_ma=18.0 / 17.0,
        vix_5d_change=2.0,
        vix9d=None,
        data_quality="full",
    )


def _features_defensive() -> RegimeFeatures:
    return RegimeFeatures(
        spy_ret_1d=-0.5,
        spy_ret_5d=-2.0,
        spy_ret_20d=-3.0,
        spy_above_ma200=True,
        spy_drawdown_pct=-4.0,
        vix=22.0,
        vix_ma20=18.0,
        vix_vs_ma=22.0 / 18.0,
        vix_5d_change=10.0,
        vix9d=None,
        data_quality="full",
    )


def _features_high_vol() -> RegimeFeatures:
    return RegimeFeatures(
        spy_ret_1d=-0.8,
        spy_ret_5d=-4.0,
        spy_ret_20d=-6.0,
        spy_above_ma200=False,
        spy_drawdown_pct=-6.0,
        vix=27.0,
        vix_ma20=20.0,
        vix_vs_ma=27.0 / 20.0,
        vix_5d_change=15.0,
        vix9d=None,
        data_quality="full",
    )


def _features_stress_a() -> RegimeFeatures:
    """Trigger A: single-day shock + 5d weakness."""
    return RegimeFeatures(
        spy_ret_1d=-2.0,
        spy_ret_5d=-6.0,
        spy_ret_20d=-8.0,
        spy_above_ma200=False,
        spy_drawdown_pct=-10.0,
        vix=35.0,
        vix_ma20=20.0,
        vix_vs_ma=1.75,
        vix_5d_change=40.0,
        vix9d=None,
        data_quality="full",
    )


# ── MarketRegime enum ─────────────────────────────────────────────────────────


class TestMarketRegimeEnum(unittest.TestCase):
    def test_all_values_are_strings(self):
        for r in MarketRegime:
            self.assertIsInstance(r.value, str)

    def test_str_enum_equality(self):
        self.assertEqual(MarketRegime.BULL_TREND, "BULL_TREND")
        self.assertEqual(MarketRegime.STRESS_RISK_OFF, "STRESS_RISK_OFF")

    def test_nine_states_defined(self):
        self.assertEqual(len(MarketRegime), 9)

    def test_new_v2_states_are_members(self):
        self.assertIn(MarketRegime.CREDIT_STRESS, list(MarketRegime))
        self.assertIn(MarketRegime.LATE_CYCLE_BULL, list(MarketRegime))
        self.assertIn(MarketRegime.RECOVERY, list(MarketRegime))


# ── RegimeThresholds defaults ─────────────────────────────────────────────────


class TestRegimeThresholds(unittest.TestCase):
    def test_default_spy_bear_1d_is_negative(self):
        t = RegimeThresholds()
        self.assertLess(t.spy_bear_1d, 0)

    def test_require_above_ma200_default_true(self):
        self.assertTrue(RegimeThresholds().require_above_ma200)

    def test_can_override(self):
        t = RegimeThresholds(spy_bear_1d=-2.5, require_above_ma200=False)
        self.assertEqual(t.spy_bear_1d, -2.5)
        self.assertFalse(t.require_above_ma200)


# ── compute_regime_features ───────────────────────────────────────────────────


class TestComputeRegimeFeatures(unittest.TestCase):
    def test_insufficient_returns_zero_features(self):
        spy = _make_spy_df([400.0, 401.0, 402.0])  # only 3 bars
        f = compute_regime_features(spy, None)
        self.assertEqual(f.data_quality, "insufficient")
        self.assertEqual(f.spy_ret_1d, 0.0)
        self.assertIsNone(f.vix)

    def test_minimal_data_quality_with_6_bars(self):
        spy = _make_spy_df([400.0] * 10)
        f = compute_regime_features(spy, None)
        self.assertEqual(f.data_quality, "minimal")

    def test_partial_data_quality_with_25_bars(self):
        spy = _make_spy_df([400.0] * 25)
        f = compute_regime_features(spy, None)
        self.assertEqual(f.data_quality, "partial")
        self.assertIsNone(f.spy_above_ma200)

    def test_full_data_quality_with_200_bars(self):
        spy = _make_spy_df([400.0] * 210)
        f = compute_regime_features(spy, None)
        self.assertEqual(f.data_quality, "full")
        self.assertIsNotNone(f.spy_above_ma200)

    def test_spy_ret_1d_computed_correctly(self):
        spy = _make_spy_df([100.0, 102.0] + [102.0] * 8)  # 1d return = +2%
        f = compute_regime_features(spy, None)
        self.assertAlmostEqual(f.spy_ret_1d, 0.0, places=2)  # last two same

    def test_spy_ret_1d_positive_on_up_day(self):
        closes = [100.0] * 9 + [105.0]
        spy = _make_spy_df(closes)
        f = compute_regime_features(spy, None)
        self.assertAlmostEqual(f.spy_ret_1d, 5.0, places=2)

    def test_spy_ret_5d_computed(self):
        closes = [100.0] * 5 + [105.0] * 5
        spy = _make_spy_df(closes)
        f = compute_regime_features(spy, None)
        self.assertAlmostEqual(f.spy_ret_5d, 5.0, places=2)

    def test_drawdown_negative_when_below_high(self):
        closes = list(range(100, 350)) + [200.0]  # peak 349, then 200
        spy = _make_spy_df(closes)
        f = compute_regime_features(spy, None)
        self.assertLess(f.spy_drawdown_pct, -30)

    def test_drawdown_zero_at_all_time_high(self):
        closes = list(range(100, 320))
        spy = _make_spy_df(closes)
        f = compute_regime_features(spy, None)
        self.assertAlmostEqual(f.spy_drawdown_pct, 0.0, places=1)

    def test_vix_none_without_vix_df(self):
        spy = _make_spy_df([400.0] * 10)
        f = compute_regime_features(spy, None)
        self.assertIsNone(f.vix)

    def test_vix_populated_when_provided(self):
        spy = _make_spy_df([400.0] * 10)
        vix = _make_vix_df([20.0] * 10)
        f = compute_regime_features(spy, vix)
        self.assertAlmostEqual(f.vix, 20.0)

    def test_vix_ma20_none_with_fewer_than_20_bars(self):
        spy = _make_spy_df([400.0] * 10)
        vix = _make_vix_df([20.0] * 10)
        f = compute_regime_features(spy, vix)
        self.assertIsNone(f.vix_ma20)

    def test_vix_ma20_computed_with_20_bars(self):
        spy = _make_spy_df([400.0] * 25)
        vix = _make_vix_df([20.0] * 25)
        f = compute_regime_features(spy, vix)
        self.assertIsNotNone(f.vix_ma20)
        self.assertAlmostEqual(f.vix_ma20, 20.0, places=1)

    def test_vix_vs_ma_ratio_correct(self):
        spy = _make_spy_df([400.0] * 25)
        vix = _make_vix_df([20.0] * 20 + [30.0] * 5)
        f = compute_regime_features(spy, vix)
        self.assertIsNotNone(f.vix_vs_ma)
        self.assertGreater(f.vix_vs_ma, 1.0)  # current 30 > MA (~20-22)

    def test_vix_5d_change_computed(self):
        spy = _make_spy_df([400.0] * 10)
        vix_closes = [10.0] * 5 + [15.0] * 5
        vix = _make_vix_df(vix_closes)
        f = compute_regime_features(spy, vix)
        self.assertIsNotNone(f.vix_5d_change)
        self.assertAlmostEqual(f.vix_5d_change, 50.0, places=1)  # 15/10 - 1 = +50%

    def test_as_of_filters_data(self):
        closes = [100.0] * 10 + [200.0] * 10  # price doubles on day 11
        spy = _make_spy_df(closes)
        # as_of = day 10 — should see the pre-doubling regime
        as_of = spy.index[9]
        f = compute_regime_features(spy, None, as_of=as_of)
        self.assertAlmostEqual(f.spy_ret_1d, 0.0, places=1)  # all 100 before cutoff

    def test_as_of_also_filters_vix_df(self):
        """Line 147: vix_df sliced when as_of and vix_df both provided."""
        spy = _make_spy_df([100.0] * 15)
        vix = _make_vix_df([20.0] * 15)
        as_of = spy.index[9]
        f = compute_regime_features(spy, vix, as_of=as_of)
        self.assertAlmostEqual(f.vix, 20.0)

    def test_empty_vix_after_dropna_returns_none_vix(self):
        """Lines 188->198: vix_df provided but Close column all NaN."""
        spy = _make_spy_df([100.0] * 10)
        idx = pd.date_range("2024-01-01", periods=10, freq="B")
        vix = pd.DataFrame({"Close": [float("nan")] * 10}, index=idx)
        f = compute_regime_features(spy, vix)
        self.assertIsNone(f.vix)
        self.assertIsNone(f.vix_5d_change)

    def test_vix_5d_change_none_with_fewer_than_6_vix_bars(self):
        """Lines 195->198: vix_df has <6 bars — vix_5d_change stays None."""
        spy = _make_spy_df([100.0] * 10)
        vix = _make_vix_df([20.0] * 5)
        f = compute_regime_features(spy, vix)
        self.assertIsNotNone(f.vix)
        self.assertIsNone(f.vix_5d_change)

    def test_spy_above_ma200_true_in_uptrend(self):
        closes = list(range(200, 450))  # 250 bars trending up
        spy = _make_spy_df(closes)
        f = compute_regime_features(spy, None)
        self.assertTrue(f.spy_above_ma200)

    def test_spy_above_ma200_false_in_downtrend(self):
        closes = list(range(450, 200, -1))  # 250 bars trending down
        spy = _make_spy_df(closes)
        f = compute_regime_features(spy, None)
        self.assertFalse(f.spy_above_ma200)

    def test_vix9d_empty_after_dropna_leaves_vix9d_none(self):
        """Branch 223->226: vix9d_df provided but Close all NaN → vix9d stays None."""
        import numpy as np

        spy = _make_spy_df([400.0] * 10)
        vix9d_df = pd.DataFrame(
            {"Close": [np.nan] * 5},
            index=pd.date_range("2024-01-01", periods=5, freq="B"),
        )
        f = compute_regime_features(spy, None, vix9d_df=vix9d_df)
        self.assertIsNone(f.vix9d)


# ── resolve_regime ────────────────────────────────────────────────────────────


class TestResolveRegime(unittest.TestCase):
    def test_insufficient_data_returns_unknown(self):
        f = RegimeFeatures(
            spy_ret_1d=0.0,
            spy_ret_5d=0.0,
            spy_ret_20d=0.0,
            spy_above_ma200=None,
            spy_drawdown_pct=0.0,
            vix=None,
            vix_ma20=None,
            vix_vs_ma=None,
            vix_5d_change=None,
            vix9d=None,
            data_quality="insufficient",
        )
        regime, reasons = resolve_regime(f)
        self.assertEqual(regime, MarketRegime.UNKNOWN)
        self.assertTrue(len(reasons) > 0)

    def test_stress_trigger_a_single_day_plus_5d(self):
        f = RegimeFeatures(
            spy_ret_1d=-2.0,
            spy_ret_5d=-6.0,
            spy_ret_20d=-8.0,
            spy_above_ma200=False,
            spy_drawdown_pct=-5.0,
            vix=20.0,
            vix_ma20=18.0,
            vix_vs_ma=1.1,
            vix_5d_change=5.0,
            vix9d=None,
            data_quality="full",
        )
        regime, reasons = resolve_regime(f)
        self.assertEqual(regime, MarketRegime.STRESS_RISK_OFF)
        self.assertTrue(any("STRESS-A" in r for r in reasons))

    def test_single_bad_day_without_5d_weakness_is_not_stress(self):
        """A -2% day with only -1% over 5 days must NOT trigger STRESS."""
        f = RegimeFeatures(
            spy_ret_1d=-2.0,
            spy_ret_5d=-1.0,
            spy_ret_20d=-2.0,
            spy_above_ma200=True,
            spy_drawdown_pct=-3.0,
            vix=20.0,
            vix_ma20=18.0,
            vix_vs_ma=1.1,
            vix_5d_change=5.0,
            vix9d=None,
            data_quality="full",
        )
        regime, _ = resolve_regime(f)
        self.assertNotEqual(regime, MarketRegime.STRESS_RISK_OFF)

    def test_stress_trigger_b_drawdown_plus_vix(self):
        f = RegimeFeatures(
            spy_ret_1d=-2.0,
            spy_ret_5d=-3.0,
            spy_ret_20d=-8.0,
            spy_above_ma200=False,
            spy_drawdown_pct=-9.0,
            vix=26.0,
            vix_ma20=18.0,
            vix_vs_ma=1.4,
            vix_5d_change=5.0,
            vix9d=None,
            data_quality="full",
        )
        regime, reasons = resolve_regime(f)
        self.assertEqual(regime, MarketRegime.STRESS_RISK_OFF)
        self.assertTrue(any("STRESS-B" in r for r in reasons))

    def test_stress_trigger_c_absolute_vix(self):
        f = RegimeFeatures(
            spy_ret_1d=-2.0,
            spy_ret_5d=-3.0,
            spy_ret_20d=-5.0,
            spy_above_ma200=False,
            spy_drawdown_pct=-5.0,
            vix=31.0,
            vix_ma20=20.0,
            vix_vs_ma=1.55,
            vix_5d_change=5.0,
            vix9d=None,
            data_quality="full",
        )
        regime, reasons = resolve_regime(f)
        self.assertEqual(regime, MarketRegime.STRESS_RISK_OFF)
        self.assertTrue(any("STRESS-C" in r for r in reasons))

    def test_stress_trigger_d_vix_surge(self):
        f = RegimeFeatures(
            spy_ret_1d=-2.0,
            spy_ret_5d=-3.0,
            spy_ret_20d=-5.0,
            spy_above_ma200=False,
            spy_drawdown_pct=-5.0,
            vix=25.0,
            vix_ma20=20.0,
            vix_vs_ma=1.25,
            vix_5d_change=35.0,
            vix9d=None,
            data_quality="full",
        )
        regime, reasons = resolve_regime(f)
        self.assertEqual(regime, MarketRegime.STRESS_RISK_OFF)
        self.assertTrue(any("STRESS-D" in r for r in reasons))

    def test_stress_trigger_e_no_single_day_shock(self):
        """5d sustained weakness + absolute VIX triggers STRESS without single-day shock."""
        f = RegimeFeatures(
            spy_ret_1d=-0.5,
            spy_ret_5d=-6.0,
            spy_ret_20d=-8.0,
            spy_above_ma200=False,
            spy_drawdown_pct=-10.0,
            vix=32.0,
            vix_ma20=20.0,
            vix_vs_ma=1.6,
            vix_5d_change=20.0,
            vix9d=None,
            data_quality="full",
        )
        regime, reasons = resolve_regime(f)
        self.assertEqual(regime, MarketRegime.STRESS_RISK_OFF)
        self.assertTrue(any("STRESS-E" in r for r in reasons))

    def test_stress_trigger_f_vix_spike_above_ma(self):
        f = RegimeFeatures(
            spy_ret_1d=-0.5,
            spy_ret_5d=-6.0,
            spy_ret_20d=-8.0,
            spy_above_ma200=False,
            spy_drawdown_pct=-5.0,
            vix=28.0,
            vix_ma20=18.0,
            vix_vs_ma=1.56,
            vix_5d_change=10.0,
            vix9d=None,
            data_quality="full",
        )
        regime, reasons = resolve_regime(f)
        self.assertEqual(regime, MarketRegime.STRESS_RISK_OFF)
        self.assertTrue(any("STRESS-F" in r for r in reasons))

    def test_high_vol_downtrend(self):
        regime, reasons = resolve_regime(_features_high_vol())
        self.assertEqual(regime, MarketRegime.HIGH_VOL_DOWNTREND)
        self.assertTrue(len(reasons) > 0)

    def test_high_vol_no_match_without_5d_weakness(self):
        f = RegimeFeatures(
            spy_ret_1d=0.2,
            spy_ret_5d=1.0,
            spy_ret_20d=2.0,
            spy_above_ma200=True,
            spy_drawdown_pct=-2.0,
            vix=27.0,
            vix_ma20=20.0,
            vix_vs_ma=1.35,
            vix_5d_change=5.0,
            vix9d=None,
            data_quality="full",
        )
        regime, _ = resolve_regime(f)
        self.assertNotEqual(regime, MarketRegime.HIGH_VOL_DOWNTREND)

    def test_defensive_downtrend(self):
        regime, reasons = resolve_regime(_features_defensive())
        self.assertEqual(regime, MarketRegime.DEFENSIVE_DOWNTREND)
        self.assertTrue(len(reasons) > 0)

    def test_bull_trend_with_ma200(self):
        regime, reasons = resolve_regime(_features_bull())
        self.assertEqual(regime, MarketRegime.BULL_TREND)
        self.assertTrue(len(reasons) > 0)

    def test_bull_trend_requires_ma200_by_default(self):
        f = RegimeFeatures(
            spy_ret_1d=0.5,
            spy_ret_5d=3.0,
            spy_ret_20d=5.0,
            spy_above_ma200=False,
            spy_drawdown_pct=-1.0,  # below MA200
            vix=15.0,
            vix_ma20=14.0,
            vix_vs_ma=1.07,
            vix_5d_change=-5.0,
            vix9d=None,
            data_quality="full",
        )
        regime, _ = resolve_regime(f)
        self.assertNotEqual(regime, MarketRegime.BULL_TREND)

    def test_bull_trend_without_ma200_when_require_false(self):
        f = RegimeFeatures(
            spy_ret_1d=0.5,
            spy_ret_5d=3.0,
            spy_ret_20d=5.0,
            spy_above_ma200=False,
            spy_drawdown_pct=-1.0,
            vix=15.0,
            vix_ma20=14.0,
            vix_vs_ma=1.07,
            vix_5d_change=-5.0,
            vix9d=None,
            data_quality="full",
        )
        t = RegimeThresholds(require_above_ma200=False)
        regime, _ = resolve_regime(f, t)
        self.assertEqual(regime, MarketRegime.BULL_TREND)

    def test_neutral_chop_is_fallback(self):
        regime, reasons = resolve_regime(_features_choppy())
        self.assertEqual(regime, MarketRegime.NEUTRAL_CHOP)
        self.assertTrue(len(reasons) > 0)

    def test_bull_trend_requires_positive_1d(self):
        f = RegimeFeatures(
            spy_ret_1d=-0.1,
            spy_ret_5d=3.0,
            spy_ret_20d=5.0,
            spy_above_ma200=True,
            spy_drawdown_pct=-1.0,
            vix=15.0,
            vix_ma20=14.0,
            vix_vs_ma=1.07,
            vix_5d_change=-5.0,
            vix9d=None,
            data_quality="full",
        )
        regime, _ = resolve_regime(f)
        self.assertNotEqual(regime, MarketRegime.BULL_TREND)

    def test_returns_list_of_reason_strings(self):
        _, reasons = resolve_regime(_features_stress_a())
        self.assertIsInstance(reasons, list)
        for r in reasons:
            self.assertIsInstance(r, str)


# ── apply_regime_hysteresis ───────────────────────────────────────────────────


class TestApplyRegimeHysteresis(unittest.TestCase):
    def _snap(self, candidate: MarketRegime, previous: PreviousRegimeState | None):
        return apply_regime_hysteresis(
            candidate, previous, [f"reason for {candidate}"], _features_choppy()
        )

    def test_first_run_confirms_immediately(self):
        snap = self._snap(MarketRegime.NEUTRAL_CHOP, None)
        self.assertEqual(snap.regime, MarketRegime.NEUTRAL_CHOP)
        self.assertIsNone(snap.pending_candidate)

    def test_stress_confirms_immediately_even_on_first_bar(self):
        previous = PreviousRegimeState(regime=MarketRegime.BULL_TREND)
        snap = self._snap(MarketRegime.STRESS_RISK_OFF, previous)
        self.assertEqual(snap.regime, MarketRegime.STRESS_RISK_OFF)

    def test_stress_confirms_immediately_from_any_regime(self):
        for r in [
            MarketRegime.NEUTRAL_CHOP,
            MarketRegime.HIGH_VOL_DOWNTREND,
            MarketRegime.BULL_TREND,
        ]:
            previous = PreviousRegimeState(regime=r)
            snap = self._snap(MarketRegime.STRESS_RISK_OFF, previous)
            self.assertEqual(snap.regime, MarketRegime.STRESS_RISK_OFF)

    def test_non_stress_change_held_on_first_bar(self):
        previous = PreviousRegimeState(regime=MarketRegime.BULL_TREND)
        snap = self._snap(MarketRegime.NEUTRAL_CHOP, previous)
        self.assertEqual(snap.regime, MarketRegime.BULL_TREND)  # still BULL
        self.assertEqual(snap.pending_candidate, MarketRegime.NEUTRAL_CHOP)
        self.assertEqual(snap.pending_count, 1)

    def test_non_stress_change_confirmed_on_second_bar(self):
        previous = PreviousRegimeState(
            regime=MarketRegime.BULL_TREND,
            pending_candidate=MarketRegime.NEUTRAL_CHOP,
            pending_count=1,
        )
        snap = self._snap(MarketRegime.NEUTRAL_CHOP, previous)
        self.assertEqual(snap.regime, MarketRegime.NEUTRAL_CHOP)
        self.assertIsNone(snap.pending_candidate)

    def test_pending_reset_when_candidate_changes(self):
        previous = PreviousRegimeState(
            regime=MarketRegime.BULL_TREND,
            pending_candidate=MarketRegime.NEUTRAL_CHOP,
            pending_count=1,
        )
        # Candidate changes from NEUTRAL_CHOP to DEFENSIVE_DOWNTREND
        snap = self._snap(MarketRegime.DEFENSIVE_DOWNTREND, previous)
        self.assertEqual(snap.regime, MarketRegime.BULL_TREND)
        self.assertEqual(snap.pending_candidate, MarketRegime.DEFENSIVE_DOWNTREND)
        self.assertEqual(snap.pending_count, 1)

    def test_no_change_clears_pending(self):
        previous = PreviousRegimeState(
            regime=MarketRegime.BULL_TREND,
            pending_candidate=MarketRegime.NEUTRAL_CHOP,
            pending_count=1,
        )
        snap = self._snap(MarketRegime.BULL_TREND, previous)
        self.assertEqual(snap.regime, MarketRegime.BULL_TREND)
        self.assertIsNone(snap.pending_candidate)

    def test_reasons_in_snapshot_include_hysteresis_note(self):
        previous = PreviousRegimeState(regime=MarketRegime.BULL_TREND)
        snap = self._snap(MarketRegime.NEUTRAL_CHOP, previous)
        self.assertTrue(any("hysteresis" in r for r in snap.reasons))

    def test_unknown_confirms_immediately(self):
        previous = PreviousRegimeState(regime=MarketRegime.BULL_TREND)
        snap = self._snap(MarketRegime.UNKNOWN, previous)
        self.assertEqual(snap.regime, MarketRegime.UNKNOWN)


# ── MarketRegimeSnapshot.to_dict() ───────────────────────────────────────────


class TestMarketRegimeSnapshotToDict(unittest.TestCase):
    def _snapshot(self, regime: MarketRegime) -> MarketRegimeSnapshot:
        return MarketRegimeSnapshot(
            regime=regime,
            reasons=(f"reason for {regime}",),
            features=_features_choppy(),
        )

    def test_is_bearish_true_for_stress(self):
        d = self._snapshot(MarketRegime.STRESS_RISK_OFF).to_dict()
        self.assertTrue(d["is_bearish"])

    def test_is_bearish_true_for_unknown(self):
        d = self._snapshot(MarketRegime.UNKNOWN).to_dict()
        self.assertTrue(d["is_bearish"])

    def test_is_bearish_false_for_bull(self):
        d = self._snapshot(MarketRegime.BULL_TREND).to_dict()
        self.assertFalse(d["is_bearish"])

    def test_is_bearish_false_for_neutral_chop(self):
        d = self._snapshot(MarketRegime.NEUTRAL_CHOP).to_dict()
        self.assertFalse(d["is_bearish"])

    def test_regime_value_is_string(self):
        d = self._snapshot(MarketRegime.BULL_TREND).to_dict()
        self.assertIsInstance(d["regime"], str)
        self.assertEqual(d["regime"], "BULL_TREND")

    def test_backward_compat_keys_present(self):
        d = self._snapshot(MarketRegime.NEUTRAL_CHOP).to_dict()
        for key in ("is_bearish", "spy_change_pct", "spy_5d_pct", "regime", "vix"):
            self.assertIn(key, d)

    def test_spy_change_pct_is_float(self):
        d = self._snapshot(MarketRegime.BULL_TREND).to_dict()
        self.assertIsInstance(d["spy_change_pct"], float)

    def test_reasons_is_list(self):
        d = self._snapshot(MarketRegime.BULL_TREND).to_dict()
        self.assertIsInstance(d["reasons"], list)


# ── get_market_regime (integration) ──────────────────────────────────────────


class TestGetMarketRegime(unittest.TestCase):
    def test_returns_snapshot(self):
        spy = _spy_df_bull()
        snap = get_market_regime(spy, None)
        self.assertIsInstance(snap, MarketRegimeSnapshot)

    def test_bull_trend_from_uptrending_spy(self):
        spy = _spy_df_bull(n=250)
        snap = get_market_regime(spy, None)
        self.assertIn(snap.regime, (MarketRegime.BULL_TREND, MarketRegime.NEUTRAL_CHOP))

    def test_neutral_chop_from_flat_spy(self):
        spy = _spy_df_flat(n=250)
        snap = get_market_regime(spy, None)
        self.assertEqual(snap.regime, MarketRegime.NEUTRAL_CHOP)

    def test_previous_state_threads_hysteresis(self):
        spy = _spy_df_flat(n=250)
        snap1 = get_market_regime(spy, None)
        previous = snap1.to_previous()
        # Flat SPY → NEUTRAL_CHOP both runs → confirmed
        snap2 = get_market_regime(spy, None, previous=previous)
        self.assertEqual(snap2.regime, MarketRegime.NEUTRAL_CHOP)
        self.assertIsNone(snap2.pending_candidate)

    def test_as_of_limits_data(self):
        closes = [400.0] * 200 + list(range(300, 500, 2))
        spy = _make_spy_df(closes)
        as_of = spy.index[199]
        snap = get_market_regime(spy, None, as_of=as_of)
        self.assertIsInstance(snap, MarketRegimeSnapshot)

    def test_insufficient_data_returns_unknown(self):
        spy = _make_spy_df([400.0, 401.0])  # < 6 bars
        snap = get_market_regime(spy, None)
        self.assertEqual(snap.regime, MarketRegime.UNKNOWN)


# ── compute_regime_series ─────────────────────────────────────────────────────


class TestComputeRegimeSeries(unittest.TestCase):
    def test_returns_dict_of_string_to_string(self):
        spy = _spy_df_flat(n=250)
        dates = [ts.strftime("%Y-%m-%d") for ts in spy.index[-10:]]
        result = compute_regime_series(spy, None, dates)
        self.assertIsInstance(result, dict)
        for k, v in result.items():
            self.assertIsInstance(k, str)
            self.assertIsInstance(v, str)

    def test_all_input_dates_present_in_output(self):
        spy = _spy_df_flat(n=250)
        dates = [ts.strftime("%Y-%m-%d") for ts in spy.index[-5:]]
        result = compute_regime_series(spy, None, dates)
        for d in dates:
            self.assertIn(d, result)

    def test_hysteresis_applied_across_dates(self):
        """After first date the regime should be confirmed after 2 identical bars."""
        spy = _spy_df_flat(n=250)
        dates = [ts.strftime("%Y-%m-%d") for ts in spy.index[-5:]]
        result = compute_regime_series(spy, None, dates)
        # All flat dates → eventually NEUTRAL_CHOP
        self.assertTrue(all(v == "NEUTRAL_CHOP" for v in result.values()))

    def test_empty_dates_returns_empty_dict(self):
        spy = _spy_df_flat(n=250)
        result = compute_regime_series(spy, None, [])
        self.assertEqual(result, {})

    def test_regime_values_are_valid_enum_strings(self):
        spy = _spy_df_bull(n=250)
        dates = [ts.strftime("%Y-%m-%d") for ts in spy.index[-10:]]
        result = compute_regime_series(spy, None, dates)
        valid = {r.value for r in MarketRegime}
        for v in result.values():
            self.assertIn(v, valid)


# ── State persistence ─────────────────────────────────────────────────────────


class TestRegimeStatePersistence(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._state_path_patcher = patch(
            "data.market_regime._STATE_PATH",
            os.path.join(self._tmp.name, "regime_state.json"),
        )
        self._state_path_patcher.start()

    def tearDown(self):
        self._state_path_patcher.stop()
        self._tmp.cleanup()

    def _make_snapshot(self, regime: MarketRegime) -> MarketRegimeSnapshot:
        return MarketRegimeSnapshot(
            regime=regime,
            reasons=("test",),
            features=_features_choppy(),
            pending_candidate=None,
            pending_count=0,
        )

    def test_save_and_load_roundtrip(self):
        snap = self._make_snapshot(MarketRegime.BULL_TREND)
        save_regime_state(snap)
        loaded = load_regime_state()
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.regime, MarketRegime.BULL_TREND)

    def test_load_returns_none_when_file_missing(self):
        result = load_regime_state()
        self.assertIsNone(result)

    def test_save_pending_candidate(self):
        snap = MarketRegimeSnapshot(
            regime=MarketRegime.BULL_TREND,
            reasons=("test",),
            features=_features_choppy(),
            pending_candidate=MarketRegime.NEUTRAL_CHOP,
            pending_count=1,
        )
        save_regime_state(snap)
        loaded = load_regime_state()
        self.assertEqual(loaded.pending_candidate, MarketRegime.NEUTRAL_CHOP)
        self.assertEqual(loaded.pending_count, 1)

    def test_load_returns_none_on_corrupt_json(self):
        path = os.path.join(self._tmp.name, "regime_state.json")
        with open(path, "w") as f:
            f.write("{not valid json")
        result = load_regime_state()
        self.assertIsNone(result)

    def test_load_returns_none_on_invalid_regime_value(self):
        path = os.path.join(self._tmp.name, "regime_state.json")
        with open(path, "w") as f:
            json.dump({"regime": "INVALID_REGIME_XYZ"}, f)
        result = load_regime_state()
        self.assertIsNone(result)

    def test_load_returns_none_on_unexpected_exception(self):
        """Line 433-435: generic Exception path in load_regime_state."""
        with patch("builtins.open", side_effect=PermissionError("denied")):
            result = load_regime_state()
        self.assertIsNone(result)

    def test_save_failure_does_not_raise(self):
        """Lines 452-453: Exception in save_regime_state is silently logged."""
        snap = self._make_snapshot(MarketRegime.BULL_TREND)
        with patch("builtins.open", side_effect=PermissionError("read-only")):
            save_regime_state(snap)  # must not raise

    def test_all_regimes_survive_roundtrip(self):
        for regime in MarketRegime:
            snap = self._make_snapshot(regime)
            save_regime_state(snap)
            loaded = load_regime_state()
            self.assertEqual(loaded.regime, regime)


# ── Data caching ──────────────────────────────────────────────────────────────


class TestSpyVixCache(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._cache_path_patcher = patch(
            "data.market_regime._CACHE_PATH",
            os.path.join(self._tmp.name, "spy_vix_cache.pkl"),
        )
        self._cache_path_patcher.start()

    def tearDown(self):
        self._cache_path_patcher.stop()
        self._tmp.cleanup()

    def test_load_cache_returns_nones_when_missing(self):
        spy, vix, vix9d, d = _load_cache()
        self.assertIsNone(spy)
        self.assertIsNone(vix)
        self.assertIsNone(vix9d)
        self.assertIsNone(d)

    def test_save_and_load_cache_today(self):
        spy = _spy_df_flat(n=10)
        vix = _make_vix_df([20.0] * 10)
        _save_cache(spy, vix)
        spy2, vix2, vix9d2, d = _load_cache()
        self.assertIsNotNone(spy2)
        self.assertIsNone(vix9d2)
        self.assertEqual(d, date.today())

    def test_cache_miss_when_stale(self):
        spy = _spy_df_flat(n=10)
        cache_path = os.path.join(self._tmp.name, "spy_vix_cache.pkl")
        with open(cache_path, "wb") as f:
            pickle.dump({"spy": spy, "vix": None, "date": date.today() - timedelta(days=1)}, f)
        _, _, _vix9d, d = _load_cache()
        self.assertEqual(d, date.today() - timedelta(days=1))  # stale date returned


class TestFetchSpyVixHistory(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        patcher = patch(
            "data.market_regime._CACHE_PATH",
            os.path.join(self._tmp.name, "spy_vix_cache.pkl"),
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._tmp.cleanup)

    def _mock_yf_download(self, spy_prices: list[float], vix_prices: list[float]):
        spy_close = pd.Series(
            spy_prices,
            index=pd.date_range("2024-01-01", periods=len(spy_prices), freq="B"),
        )
        vix_close = pd.Series(
            vix_prices,
            index=pd.date_range("2024-01-01", periods=len(vix_prices), freq="B"),
        )
        calls = [0]

        def side_effect(ticker, **kwargs):
            calls[0] += 1
            if ticker == "SPY":
                return pd.DataFrame({"Close": spy_close})
            elif ticker == "^VIX":
                return pd.DataFrame({"Close": vix_close})
            return pd.DataFrame()  # ^VIX9D and any other tickers return empty

        return side_effect

    def test_returns_spy_and_vix_dataframes(self):
        with patch(
            "data.market_regime.yf.download",
            side_effect=self._mock_yf_download([400.0] * 10, [20.0] * 10),
        ):
            spy, vix = fetch_spy_vix_history()
        self.assertFalse(spy.empty)
        self.assertIsNotNone(vix)
        self.assertFalse(vix.empty)

    def test_cache_hit_skips_yfinance(self):
        cached_spy = _spy_df_flat(n=10)
        cached_vix = _make_vix_df([20.0] * 10)
        _save_cache(cached_spy, cached_vix)
        with patch("data.market_regime.yf.download") as mock_dl:
            spy, vix = fetch_spy_vix_history()
            mock_dl.assert_not_called()
        self.assertIsNotNone(spy)

    def test_returns_empty_df_on_spy_failure_no_cache(self):
        with patch("data.market_regime.yf.download", side_effect=Exception("network error")):
            spy, vix = fetch_spy_vix_history()
        self.assertTrue(spy.empty)
        self.assertIsNone(vix)

    def test_returns_stale_cache_on_spy_failure(self):
        cached_spy = _spy_df_flat(n=10)
        stale_path = self._tmp.name + "/spy_vix_cache.pkl"
        with open(stale_path, "wb") as f:
            pickle.dump(
                {"spy": cached_spy, "vix": None, "date": date.today() - timedelta(days=1)}, f
            )
        with patch("data.market_regime.yf.download", side_effect=Exception("network error")):
            spy, vix = fetch_spy_vix_history()
        self.assertFalse(spy.empty)

    def test_vix_df_is_none_on_vix_failure(self):
        spy_close = pd.DataFrame(
            {"Close": [400.0] * 10},
            index=pd.date_range("2024-01-01", periods=10, freq="B"),
        )

        def mock_dl(ticker, **kwargs):
            if ticker == "SPY":
                return spy_close
            raise Exception("VIX network error")  # covers ^VIX and ^VIX9D

        with patch("data.market_regime.yf.download", side_effect=mock_dl):
            spy, vix = fetch_spy_vix_history()
        self.assertFalse(spy.empty)
        self.assertIsNone(vix)

    def test_empty_spy_raises_and_returns_empty(self):
        """Line 499: yfinance returns empty SPY DataFrame → ValueError raised internally."""
        with patch("data.market_regime.yf.download", return_value=pd.DataFrame()):
            spy, vix = fetch_spy_vix_history()
        self.assertTrue(spy.empty)
        self.assertIsNone(vix)

    def test_spy_multiindex_close_extracted(self):
        """Lines 501-502: SPY raw has MultiIndex columns → take first column."""
        idx = pd.date_range("2024-01-01", periods=10, freq="B")
        mi = pd.MultiIndex.from_tuples([("Close", "SPY"), ("Volume", "SPY")])
        raw_spy = pd.DataFrame([[400.0, 1e6]] * 10, index=idx, columns=mi)

        def mock_dl(ticker, **kwargs):
            if ticker == "SPY":
                return raw_spy
            return pd.DataFrame()

        with patch("data.market_regime.yf.download", side_effect=mock_dl):
            spy, _ = fetch_spy_vix_history()
        self.assertFalse(spy.empty)
        self.assertIn("Close", spy.columns)

    def test_vix_multiindex_close_extracted(self):
        """Lines 517-518: VIX raw has MultiIndex columns → take first column."""
        idx = pd.date_range("2024-01-01", periods=10, freq="B")
        spy_raw = pd.DataFrame({"Close": [400.0] * 10}, index=idx)
        mi = pd.MultiIndex.from_tuples([("Close", "^VIX"), ("Volume", "^VIX")])
        raw_vix = pd.DataFrame([[20.0, 1e6]] * 10, index=idx, columns=mi)

        call_count = [0]

        def mock_dl(ticker, **kwargs):
            call_count[0] += 1
            if ticker == "SPY":
                return spy_raw
            return raw_vix

        with patch("data.market_regime.yf.download", side_effect=mock_dl):
            spy, vix = fetch_spy_vix_history()
        self.assertIsNotNone(vix)
        self.assertFalse(vix.empty)

    def test_vix9d_series_format_is_handled(self):
        """Branch 553->555: VIX9D raw has simple Close Series → isinstance is False, skip iloc."""
        idx = pd.date_range("2024-01-01", periods=10, freq="B")
        spy_raw = pd.DataFrame({"Close": [400.0] * 10}, index=idx)
        vix_raw = pd.DataFrame({"Close": [20.0] * 10}, index=idx)
        vix9d_raw = pd.DataFrame({"Close": [18.0] * 10}, index=idx)

        def mock_dl(ticker, **kwargs):
            if ticker == "SPY":
                return spy_raw
            if ticker == "^VIX":
                return vix_raw
            return vix9d_raw  # ^VIX9D returns simple single-column DataFrame

        with patch("data.market_regime.yf.download", side_effect=mock_dl):
            spy, vix = fetch_spy_vix_history()
        self.assertFalse(spy.empty)

    def test_vix9d_multiindex_close_extracted(self):
        """Branch 553->554: VIX9D raw has MultiIndex columns → take first column."""
        idx = pd.date_range("2024-01-01", periods=10, freq="B")
        spy_raw = pd.DataFrame({"Close": [400.0] * 10}, index=idx)
        vix_raw = pd.DataFrame({"Close": [20.0] * 10}, index=idx)
        mi = pd.MultiIndex.from_tuples([("Close", "^VIX9D"), ("Volume", "^VIX9D")])
        raw_vix9d = pd.DataFrame([[18.0, 1e6]] * 10, index=idx, columns=mi)

        def mock_dl(ticker, **kwargs):
            if ticker == "SPY":
                return spy_raw
            if ticker == "^VIX":
                return vix_raw
            return raw_vix9d  # ^VIX9D returns MultiIndex DataFrame

        with patch("data.market_regime.yf.download", side_effect=mock_dl):
            spy, vix = fetch_spy_vix_history()
        self.assertFalse(spy.empty)

    def test_save_cache_failure_does_not_raise(self):
        """Lines 473-474: Exception in _save_cache is silently logged."""
        spy = _spy_df_flat(n=10)
        with patch("builtins.open", side_effect=PermissionError("read-only")):
            _save_cache(spy, None)  # must not raise

    def test_fetch_vix9d_history_cold_cache_triggers_refresh(self):
        """fetch_vix9d_history cold-cache path: calls fetch_spy_vix_history then re-reads cache."""
        from data.market_regime import fetch_vix9d_history

        # First _load_cache call (cold): returns stale date so cache_date != date.today()
        # Second _load_cache call (after refresh): returns a real vix9d DataFrame
        vix9d_df = pd.DataFrame(
            {"Close": [18.0] * 5},
            index=pd.date_range("2024-01-01", periods=5, freq="B"),
        )
        cold = (None, None, None, date(2020, 1, 1))
        warm = (None, None, vix9d_df, date.today())
        load_returns = iter([cold, warm])

        with (
            patch("data.market_regime._load_cache", side_effect=lambda: next(load_returns)),
            patch("data.market_regime.fetch_spy_vix_history") as mock_fetch,
        ):
            result = fetch_vix9d_history()

        mock_fetch.assert_called_once()
        self.assertIsNotNone(result)

    def test_fetch_vix9d_history_warm_cache_returns_immediately(self):
        """fetch_vix9d_history warm-cache path: cache_date == today → return without fetching."""
        from data.market_regime import fetch_vix9d_history

        vix9d_df = pd.DataFrame(
            {"Close": [18.0] * 5},
            index=pd.date_range("2024-01-01", periods=5, freq="B"),
        )
        warm = (None, None, vix9d_df, date.today())

        with (
            patch("data.market_regime._load_cache", return_value=warm),
            patch("data.market_regime.fetch_spy_vix_history") as mock_fetch,
        ):
            result = fetch_vix9d_history()

        mock_fetch.assert_not_called()
        self.assertIsNotNone(result)


# ── PreviousRegimeState.to_previous() ────────────────────────────────────────


class TestToPrevious(unittest.TestCase):
    def test_to_previous_preserves_regime(self):
        snap = MarketRegimeSnapshot(
            regime=MarketRegime.BULL_TREND,
            reasons=("ok",),
            features=_features_bull(),
            pending_candidate=MarketRegime.NEUTRAL_CHOP,
            pending_count=1,
        )
        prev = snap.to_previous()
        self.assertEqual(prev.regime, MarketRegime.BULL_TREND)
        self.assertEqual(prev.pending_candidate, MarketRegime.NEUTRAL_CHOP)
        self.assertEqual(prev.pending_count, 1)


# ── Regime v2: RegimeFeatures new fields ─────────────────────────────────────


class TestRegimeFeaturesV2Fields(unittest.TestCase):
    def test_new_fields_default_to_none(self):
        f = _features_bull()
        self.assertIsNone(f.credit_spread_roc)
        self.assertIsNone(f.breadth_pct_above_sma50)
        self.assertIsNone(f.t10y2y)

    def test_new_fields_can_be_set(self):
        f = RegimeFeatures(
            spy_ret_1d=0.5,
            spy_ret_5d=3.0,
            spy_ret_20d=5.0,
            spy_above_ma200=True,
            spy_drawdown_pct=-1.0,
            vix=15.0,
            vix_ma20=14.0,
            vix_vs_ma=1.07,
            vix_5d_change=-5.0,
            vix9d=None,
            data_quality="full",
            credit_spread_roc=-3.0,
            breadth_pct_above_sma50=0.45,
            t10y2y=-0.25,
        )
        self.assertEqual(f.credit_spread_roc, -3.0)
        self.assertEqual(f.breadth_pct_above_sma50, 0.45)
        self.assertEqual(f.t10y2y, -0.25)


# ── Regime v2: RegimeThresholds new fields ────────────────────────────────────


class TestRegimeThresholdsV2(unittest.TestCase):
    def test_credit_stress_roc_min_default(self):
        self.assertEqual(RegimeThresholds().credit_stress_roc_min, -2.0)

    def test_t10y2y_inversion_threshold_default(self):
        self.assertEqual(RegimeThresholds().t10y2y_inversion_threshold, 0.0)

    def test_breadth_divergence_max_default(self):
        self.assertEqual(RegimeThresholds().breadth_divergence_max, 0.50)

    def test_recovery_spy_5d_min_default(self):
        self.assertEqual(RegimeThresholds().recovery_spy_5d_min, 0.5)

    def test_recovery_drawdown_max_default(self):
        self.assertEqual(RegimeThresholds().recovery_drawdown_max, -5.0)

    def test_can_override_new_thresholds(self):
        t = RegimeThresholds(credit_stress_roc_min=-3.0, breadth_divergence_max=0.40)
        self.assertEqual(t.credit_stress_roc_min, -3.0)
        self.assertEqual(t.breadth_divergence_max, 0.40)


# ── Regime v2: compute_regime_features new series inputs ─────────────────────


def _make_series(values: list[float], start: str = "2024-01-01") -> pd.Series:
    idx = pd.date_range(start=start, periods=len(values), freq="B")
    return pd.Series(values, index=idx)


class TestComputeRegimeFeaturesV2(unittest.TestCase):
    def _spy(self, n: int = 250) -> pd.DataFrame:
        return _spy_df_bull(n)

    def test_credit_spread_roc_none_when_series_absent(self):
        f = compute_regime_features(self._spy(), None)
        self.assertIsNone(f.credit_spread_roc)

    def test_credit_spread_roc_computed_from_series(self):
        # 10d ROC: 105 / 100 - 1 = +5%
        vals = [100.0] * 10 + [105.0]
        hyg_lqd = _make_series(vals)
        f = compute_regime_features(self._spy(), None, hyg_lqd_series=hyg_lqd)
        self.assertAlmostEqual(f.credit_spread_roc, 5.0, places=4)

    def test_credit_spread_roc_negative_on_decline(self):
        vals = [100.0] * 10 + [97.0]
        hyg_lqd = _make_series(vals)
        f = compute_regime_features(self._spy(), None, hyg_lqd_series=hyg_lqd)
        self.assertAlmostEqual(f.credit_spread_roc, -3.0, places=4)

    def test_credit_spread_roc_none_when_too_few_bars(self):
        # Only 5 bars — need ≥11 for 10d ROC
        hyg_lqd = _make_series([100.0] * 5)
        f = compute_regime_features(self._spy(), None, hyg_lqd_series=hyg_lqd)
        self.assertIsNone(f.credit_spread_roc)

    def test_breadth_none_when_series_absent(self):
        f = compute_regime_features(self._spy(), None)
        self.assertIsNone(f.breadth_pct_above_sma50)

    def test_breadth_latest_value_used(self):
        breadth = _make_series([0.60, 0.55, 0.48])
        f = compute_regime_features(self._spy(), None, breadth_series=breadth)
        self.assertAlmostEqual(f.breadth_pct_above_sma50, 0.48, places=4)

    def test_t10y2y_none_when_series_absent(self):
        f = compute_regime_features(self._spy(), None)
        self.assertIsNone(f.t10y2y)

    def test_t10y2y_latest_value_used(self):
        t10y2y = _make_series([0.5, 0.2, -0.1])
        f = compute_regime_features(self._spy(), None, t10y2y_series=t10y2y)
        self.assertAlmostEqual(f.t10y2y, -0.1, places=4)

    def test_as_of_filters_all_new_series(self):
        # Series runs 2024-01-01 through 2024-01-15 (11 business days).
        # as_of="2024-01-10" should exclude the later bars.
        vals_hyg_lqd = [1.0] * 5 + [0.97] * 5 + [0.90]  # 11 bars total; bar 11 is excluded
        hyg_lqd = _make_series(vals_hyg_lqd)
        as_of_date = hyg_lqd.index[9]  # 10th bar
        f = compute_regime_features(self._spy(), None, as_of=as_of_date, hyg_lqd_series=hyg_lqd)
        # Only 10 bars visible at as_of — too few for 10d ROC (need ≥11), so None
        self.assertIsNone(f.credit_spread_roc)


# ── Regime v2: resolve_regime new state branches ─────────────────────────────


def _features_credit_stress() -> RegimeFeatures:
    """SPY flat but credit spreads deteriorating sharply."""
    return RegimeFeatures(
        spy_ret_1d=0.1,
        spy_ret_5d=0.5,
        spy_ret_20d=1.0,
        spy_above_ma200=True,
        spy_drawdown_pct=-3.0,
        vix=18.0,
        vix_ma20=17.0,
        vix_vs_ma=1.06,
        vix_5d_change=5.0,
        vix9d=None,
        data_quality="full",
        credit_spread_roc=-3.0,  # below -2.0 threshold
    )


def _features_late_cycle_inverted() -> RegimeFeatures:
    """Bull price conditions met but yield curve inverted."""
    return RegimeFeatures(
        spy_ret_1d=0.5,
        spy_ret_5d=3.0,
        spy_ret_20d=5.0,
        spy_above_ma200=True,
        spy_drawdown_pct=-1.0,
        vix=15.0,
        vix_ma20=14.0,
        vix_vs_ma=1.07,
        vix_5d_change=-5.0,
        vix9d=None,
        data_quality="full",
        t10y2y=-0.3,  # inverted
    )


def _features_late_cycle_breadth() -> RegimeFeatures:
    """Bull price conditions met but narrow breadth."""
    return RegimeFeatures(
        spy_ret_1d=0.5,
        spy_ret_5d=3.0,
        spy_ret_20d=5.0,
        spy_above_ma200=True,
        spy_drawdown_pct=-1.0,
        vix=15.0,
        vix_ma20=14.0,
        vix_vs_ma=1.07,
        vix_5d_change=-5.0,
        vix9d=None,
        data_quality="full",
        breadth_pct_above_sma50=0.40,  # below 0.50 threshold
    )


def _features_recovery() -> RegimeFeatures:
    """SPY bouncing positively but still in drawdown."""
    return RegimeFeatures(
        spy_ret_1d=0.3,
        spy_ret_5d=1.5,
        spy_ret_20d=-2.0,
        spy_above_ma200=False,
        spy_drawdown_pct=-8.0,  # still in drawdown
        vix=20.0,
        vix_ma20=22.0,
        vix_vs_ma=0.91,
        vix_5d_change=-10.0,
        vix9d=None,
        data_quality="full",
    )


class TestResolveRegimeV2(unittest.TestCase):
    # ── CREDIT_STRESS ─────────────────────────────────────────────────────────

    def test_credit_stress_fires_on_roc_below_threshold(self):
        regime, reasons = resolve_regime(_features_credit_stress())
        self.assertEqual(regime, MarketRegime.CREDIT_STRESS)

    def test_credit_stress_reason_mentions_roc(self):
        _, reasons = resolve_regime(_features_credit_stress())
        self.assertTrue(any("CREDIT_STRESS" in r for r in reasons))

    def test_credit_stress_not_triggered_when_roc_above_threshold(self):
        f = RegimeFeatures(**{**_features_credit_stress().__dict__, "credit_spread_roc": -1.0})
        regime, _ = resolve_regime(f)
        self.assertNotEqual(regime, MarketRegime.CREDIT_STRESS)

    def test_credit_stress_not_triggered_when_roc_none(self):
        f = RegimeFeatures(**{**_features_credit_stress().__dict__, "credit_spread_roc": None})
        regime, _ = resolve_regime(f)
        self.assertNotEqual(regime, MarketRegime.CREDIT_STRESS)

    def test_credit_stress_not_triggered_when_spy_already_defensive(self):
        # Defensive_downtrend has higher priority than credit_stress
        f = RegimeFeatures(**{**_features_credit_stress().__dict__, "spy_ret_5d": -2.0})
        regime, _ = resolve_regime(f)
        self.assertEqual(regime, MarketRegime.DEFENSIVE_DOWNTREND)

    def test_credit_stress_custom_threshold(self):
        t = RegimeThresholds(credit_stress_roc_min=-3.5)
        # credit_spread_roc=-3.0 > -3.5 threshold → should NOT trigger
        regime, _ = resolve_regime(_features_credit_stress(), t)
        self.assertNotEqual(regime, MarketRegime.CREDIT_STRESS)

    # ── LATE_CYCLE_BULL ───────────────────────────────────────────────────────

    def test_late_cycle_bull_fires_on_inverted_curve(self):
        regime, _ = resolve_regime(_features_late_cycle_inverted())
        self.assertEqual(regime, MarketRegime.LATE_CYCLE_BULL)

    def test_late_cycle_bull_fires_on_narrow_breadth(self):
        regime, _ = resolve_regime(_features_late_cycle_breadth())
        self.assertEqual(regime, MarketRegime.LATE_CYCLE_BULL)

    def test_late_cycle_bull_reason_mentions_inverted_curve(self):
        _, reasons = resolve_regime(_features_late_cycle_inverted())
        self.assertTrue(any("inverted" in r.lower() for r in reasons))

    def test_late_cycle_bull_reason_mentions_narrow_leadership(self):
        _, reasons = resolve_regime(_features_late_cycle_breadth())
        self.assertTrue(any("narrow leadership" in r for r in reasons))

    def test_bull_trend_fires_when_no_macro_warning(self):
        # Standard bull features with no T10Y2Y or breadth — pure BULL_TREND
        regime, _ = resolve_regime(_features_bull())
        self.assertEqual(regime, MarketRegime.BULL_TREND)

    def test_bull_trend_not_downgraded_when_curve_positive(self):
        # T10Y2Y positive — not inverted, no breadth warning
        f = RegimeFeatures(**{**_features_bull().__dict__, "t10y2y": 0.5})
        regime, _ = resolve_regime(f)
        self.assertEqual(regime, MarketRegime.BULL_TREND)

    def test_bull_trend_not_downgraded_when_breadth_above_threshold(self):
        f = RegimeFeatures(**{**_features_bull().__dict__, "breadth_pct_above_sma50": 0.65})
        regime, _ = resolve_regime(f)
        self.assertEqual(regime, MarketRegime.BULL_TREND)

    def test_late_cycle_bull_fires_with_both_warnings(self):
        f = RegimeFeatures(
            **{**_features_bull().__dict__, "t10y2y": -0.5, "breadth_pct_above_sma50": 0.35}
        )
        regime, _ = resolve_regime(f)
        self.assertEqual(regime, MarketRegime.LATE_CYCLE_BULL)

    def test_late_cycle_bull_not_triggered_without_bull_price_conditions(self):
        # Choppy price action + inverted curve → not LATE_CYCLE_BULL
        f = RegimeFeatures(**{**_features_choppy().__dict__, "t10y2y": -0.3})
        regime, _ = resolve_regime(f)
        self.assertNotEqual(regime, MarketRegime.LATE_CYCLE_BULL)

    # ── RECOVERY ──────────────────────────────────────────────────────────────

    def test_recovery_fires_when_positive_5d_and_in_drawdown(self):
        regime, _ = resolve_regime(_features_recovery())
        self.assertEqual(regime, MarketRegime.RECOVERY)

    def test_recovery_reason_mentions_drawdown(self):
        _, reasons = resolve_regime(_features_recovery())
        self.assertTrue(any("RECOVERY" in r for r in reasons))

    def test_recovery_not_triggered_when_5d_below_min(self):
        f = RegimeFeatures(**{**_features_recovery().__dict__, "spy_ret_5d": 0.2})
        regime, _ = resolve_regime(f)
        self.assertNotEqual(regime, MarketRegime.RECOVERY)

    def test_recovery_not_triggered_when_drawdown_shallow(self):
        # drawdown only -3%, above -5% threshold
        f = RegimeFeatures(**{**_features_recovery().__dict__, "spy_drawdown_pct": -3.0})
        regime, _ = resolve_regime(f)
        self.assertNotEqual(regime, MarketRegime.RECOVERY)

    def test_recovery_takes_priority_over_neutral_chop(self):
        # Recovery criteria met → should be RECOVERY not NEUTRAL_CHOP
        regime, _ = resolve_regime(_features_recovery())
        self.assertNotEqual(regime, MarketRegime.NEUTRAL_CHOP)

    def test_bull_trend_takes_priority_over_recovery(self):
        # Full bull conditions (spy_5d=3%, above_ma200) → BULL_TREND even if in drawdown
        f = RegimeFeatures(
            **{
                **_features_recovery().__dict__,
                "spy_ret_5d": 3.0,
                "spy_ret_1d": 0.5,
                "spy_above_ma200": True,
            }
        )
        regime, _ = resolve_regime(f)
        self.assertEqual(regime, MarketRegime.BULL_TREND)

    def test_recovery_custom_thresholds(self):
        t = RegimeThresholds(recovery_spy_5d_min=2.0, recovery_drawdown_max=-3.0)
        f = RegimeFeatures(**{**_features_recovery().__dict__, "spy_ret_5d": 1.5})
        # 1.5% < 2.0% recovery_spy_5d_min → not RECOVERY
        regime, _ = resolve_regime(f, t)
        self.assertNotEqual(regime, MarketRegime.RECOVERY)

    # ── Priority ordering between new states ──────────────────────────────────

    def test_credit_stress_does_not_fire_in_stress_risk_off(self):
        f = RegimeFeatures(**{**_features_stress_a().__dict__, "credit_spread_roc": -5.0})
        regime, _ = resolve_regime(f)
        self.assertEqual(regime, MarketRegime.STRESS_RISK_OFF)

    def test_credit_stress_does_not_fire_in_high_vol_downtrend(self):
        f = RegimeFeatures(**{**_features_high_vol().__dict__, "credit_spread_roc": -5.0})
        regime, _ = resolve_regime(f)
        self.assertEqual(regime, MarketRegime.HIGH_VOL_DOWNTREND)


# ── Regime v2: to_dict includes new fields ────────────────────────────────────


class TestMarketRegimeSnapshotToDictV2(unittest.TestCase):
    def test_to_dict_includes_credit_spread_roc(self):
        f = RegimeFeatures(**{**_features_bull().__dict__, "credit_spread_roc": -1.5})
        snap = MarketRegimeSnapshot(regime=MarketRegime.BULL_TREND, reasons=("ok",), features=f)
        d = snap.to_dict()
        self.assertIn("credit_spread_roc", d)
        self.assertEqual(d["credit_spread_roc"], -1.5)

    def test_to_dict_includes_breadth_pct_above_sma50(self):
        f = RegimeFeatures(**{**_features_bull().__dict__, "breadth_pct_above_sma50": 0.62})
        snap = MarketRegimeSnapshot(regime=MarketRegime.BULL_TREND, reasons=("ok",), features=f)
        d = snap.to_dict()
        self.assertIn("breadth_pct_above_sma50", d)
        self.assertEqual(d["breadth_pct_above_sma50"], 0.62)

    def test_to_dict_includes_t10y2y(self):
        f = RegimeFeatures(**{**_features_bull().__dict__, "t10y2y": -0.2})
        snap = MarketRegimeSnapshot(
            regime=MarketRegime.LATE_CYCLE_BULL, reasons=("macro",), features=f
        )
        d = snap.to_dict()
        self.assertIn("t10y2y", d)
        self.assertEqual(d["t10y2y"], -0.2)

    def test_to_dict_new_fields_none_when_not_set(self):
        snap = MarketRegimeSnapshot(
            regime=MarketRegime.BULL_TREND, reasons=("ok",), features=_features_bull()
        )
        d = snap.to_dict()
        self.assertIsNone(d["credit_spread_roc"])
        self.assertIsNone(d["breadth_pct_above_sma50"])
        self.assertIsNone(d["t10y2y"])


# ── Regime v2: compute_regime_series passes new series through ────────────────


class TestComputeRegimeSeriesV2(unittest.TestCase):
    def test_credit_stress_appears_in_series_when_data_provided(self):
        spy = _spy_df_flat(250)
        # Gradual 10% decline over last 30 bars ensures ROC ≈ -3.5% on all 5 test dates.
        # First test date has prev=None → CREDIT_STRESS confirms immediately (no hysteresis).
        idx = spy.index
        n = len(idx)
        vals = [1.0] * (n - 30) + [1.0 - i * (0.1 / 29) for i in range(30)]
        hyg_lqd = pd.Series(vals, index=idx)
        dates = [d.strftime("%Y-%m-%d") for d in spy.index[-5:]]
        result = compute_regime_series(spy, None, dates, hyg_lqd_series=hyg_lqd)
        self.assertIn("CREDIT_STRESS", result.values())

    def test_regime_series_still_works_without_new_series(self):
        spy = _spy_df_flat(250)
        dates = [d.strftime("%Y-%m-%d") for d in spy.index[-3:]]
        result = compute_regime_series(spy, None, dates)
        self.assertEqual(len(result), 3)
        for v in result.values():
            self.assertIn(v, [r.value for r in MarketRegime])
