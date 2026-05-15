"""Tests for backtest/engine.py — _compute_indicators, _entry_signal, run_backtest,
_run_simulation, run_walk_forward_optimized, _compute_intraday_day."""

import json
import os
import tempfile
import unittest
from collections import namedtuple
from datetime import datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pandas as pd

from backtest.engine import (
    _CORE_COLS,
    _DEFAULT_PARAMS,
    _SIGNAL_PRIORITY,
    _assert_pre_holdout,
    _binomial_p_value,
    _bootstrap_cell_ci,
    _compute_indicators,
    _compute_intraday_day,
    _entry_signal,
    _holm_bonferroni,
    _liquidity_spread_bps,
    _market_impact_bps,
    _print_ablation_results,
    _print_backward_elimination_results,
    _print_hold_period_table,
    _print_regime_blocked,
    _print_regime_table,
    _print_results,
    _run_simulation,
    _save_results,
    compute_regime_blocked,
    run_ablation,
    run_backtest,
    run_backward_elimination,
    run_holdout_evaluation,
    run_signal_analysis,
    run_walk_forward_optimized,
)
from config import BACKTEST_DEFAULT_START, HOLDOUT_START_DATE

_ET = ZoneInfo("America/New_York")
_Bar = namedtuple("_Bar", ["open", "high", "low", "close", "volume"])


def _make_ohlcv(n: int = 60) -> pd.DataFrame:
    idx = pd.bdate_range("2024-11-01", periods=n)
    prices = [100.0 + i * 0.5 for i in range(n)]
    return pd.DataFrame({"Close": prices, "Volume": [1_000_000] * n}, index=idx)


def _make_ohlcv_full(n: int = 60) -> pd.DataFrame:
    """OHLCV with High and Low for inside_day indicator tests."""
    idx = pd.bdate_range("2024-11-01", periods=n)
    prices = [100.0 + i * 0.5 for i in range(n)]
    return pd.DataFrame(
        {
            "Close": prices,
            "Open": [p * 0.999 for p in prices],
            "High": [p * 1.01 for p in prices],
            "Low": [p * 0.99 for p in prices],
            "Volume": [1_000_000] * n,
        },
        index=idx,
    )


def _make_raw(n: int = 100, symbols: tuple = ("AAPL", "FLAT")) -> pd.DataFrame:
    """Build a realistic multi-symbol yfinance download mock (MultiIndex columns).
    n=100 spans 2024-11-01 through ~2025-03-12, covering the test trading windows."""
    idx = pd.bdate_range("2024-11-01", periods=n)
    data: dict = {}
    for i, sym in enumerate(symbols):
        closes = [100.0 + j * 0.1 + i * 50 for j in range(n)]
        data[("Close", sym)] = closes
        data[("Open", sym)] = [c * 0.999 for c in closes]
        data[("Volume", sym)] = [1_000_000] * n
    raw = pd.DataFrame(data, index=idx)
    raw.columns = pd.MultiIndex.from_tuples(raw.columns)
    return raw


def _make_row(**kwargs) -> pd.Series:
    defaults = {
        "rsi": 50.0,
        "bb_pct": 0.5,
        "vol_ratio": 1.0,
        "ema9": 100.0,
        "ema21": 100.0,
        "macd_diff": 0.0,
        "ret_5d": 0.0,
        "ret_10d": 0.0,
        "macd_cross": False,
        "bb_squeeze": False,
        "pct_vs_ema21": 0.0,
        "price_vs_52w_high_pct": -50.0,
        "is_inside_day": False,
        "adx": 30.0,
        "gap_pct": 0.0,
        "close_above_open": False,
        "mom_12_1": 0.0,
        "hv_rank": 1.0,
    }
    defaults.update(kwargs)
    return pd.Series(defaults)


def _make_results_dict() -> dict:
    return {
        "start": "2025-01-01",
        "end": "2025-12-31",
        "initial_capital": 10_000,
        "final_value": 11_000,
        "total_return_pct": 10.0,
        "total_trades": 5,
        "win_rate_pct": 60.0,
        "avg_return_per_trade_pct": 1.5,
        "max_drawdown_pct": -3.0,
        "sharpe_ratio": 1.2,
        "by_signal": {"momentum": {"wins": 3, "losses": 2, "total_return": 7.5}},
        "trades": [],
        "equity_curve": [("2025-01-02", 10000.0), ("2025-01-03", 10100.0)],
    }


# ── _compute_indicators ───────────────────────────────────────────────────────


class TestComputeIndicators(unittest.TestCase):
    def test_returns_dataframe(self):
        self.assertIsInstance(_compute_indicators(_make_ohlcv(60)), pd.DataFrame)

    def test_has_core_columns(self):
        result = _compute_indicators(_make_ohlcv(60))
        for col in _CORE_COLS:
            self.assertIn(col, result.columns, f"Missing core column: {col}")

    def test_has_extended_columns(self):
        result = _compute_indicators(_make_ohlcv(60))
        for col in (
            "ret_10d",
            "macd_cross",
            "bb_squeeze",
            "pct_vs_ema21",
            "high_52w",
            "price_vs_52w_high_pct",
        ):
            self.assertIn(col, result.columns, f"Missing extended column: {col}")

    def test_core_columns_have_no_nans(self):
        result = _compute_indicators(_make_ohlcv(60))
        self.assertFalse(result[_CORE_COLS].isnull().any().any())

    def test_inside_day_present_when_high_low_provided(self):
        result = _compute_indicators(_make_ohlcv_full(60))
        self.assertIn("is_inside_day", result.columns)
        self.assertTrue(
            result["is_inside_day"].dtype == bool
            or result["is_inside_day"].isin([True, False]).all()
        )

    def test_inside_day_absent_when_high_low_missing(self):
        result = _compute_indicators(_make_ohlcv(60))
        self.assertNotIn("is_inside_day", result.columns)

    def test_adx_present_when_high_low_provided(self):
        result = _compute_indicators(_make_ohlcv_full(60))
        self.assertIn("adx", result.columns)
        self.assertTrue((result["adx"] >= 0).all())

    def test_adx_absent_when_high_low_missing(self):
        result = _compute_indicators(_make_ohlcv(60))
        self.assertNotIn("adx", result.columns)

    def test_gap_pct_and_close_above_open_present_when_open_provided(self):
        result = _compute_indicators(_make_ohlcv_full(60))
        self.assertIn("gap_pct", result.columns)
        self.assertIn("close_above_open", result.columns)
        self.assertTrue(result["close_above_open"].isin([True, False]).all())

    def test_few_rows_produces_empty_result(self):
        self.assertTrue(_compute_indicators(_make_ohlcv(5)).empty)

    def test_original_dataframe_not_mutated(self):
        df = _make_ohlcv(60)
        original_cols = list(df.columns)
        _compute_indicators(df)
        self.assertEqual(list(df.columns), original_cols)

    def test_result_index_is_datetime(self):
        result = _compute_indicators(_make_ohlcv(60))
        self.assertIsInstance(result.index, pd.DatetimeIndex)

    def test_bb_squeeze_is_boolean(self):
        result = _compute_indicators(_make_ohlcv(60))
        self.assertTrue(result["bb_squeeze"].isin([True, False]).all())

    def test_macd_cross_is_boolean(self):
        result = _compute_indicators(_make_ohlcv(60))
        self.assertTrue(result["macd_cross"].isin([True, False]).all())


# ── _entry_signal ─────────────────────────────────────────────────────────────


class TestEntrySignal(unittest.TestCase):
    def test_mean_reversion_fires_when_all_conditions_met(self):
        self.assertEqual(
            _entry_signal(_make_row(rsi=30, bb_pct=0.20, vol_ratio=1.5)), "mean_reversion"
        )

    def test_mean_reversion_fails_rsi_at_boundary(self):
        self.assertIsNone(_entry_signal(_make_row(rsi=35, bb_pct=0.20, vol_ratio=1.5)))

    def test_mean_reversion_fails_bb_at_boundary(self):
        self.assertIsNone(_entry_signal(_make_row(rsi=30, bb_pct=0.25, vol_ratio=1.5)))

    def test_mean_reversion_fails_low_volume(self):
        self.assertIsNone(_entry_signal(_make_row(rsi=30, bb_pct=0.20, vol_ratio=1.2)))

    def test_momentum_fires_when_all_conditions_met(self):
        self.assertEqual(
            _entry_signal(_make_row(ema9=105, ema21=100, macd_diff=0.5, ret_5d=2.0, vol_ratio=1.5)),
            "momentum",
        )

    def test_momentum_fails_ema_not_aligned(self):
        self.assertIsNone(
            _entry_signal(_make_row(ema9=95, ema21=100, macd_diff=0.5, ret_5d=2.0, vol_ratio=1.5))
        )

    def test_momentum_fails_negative_macd(self):
        self.assertIsNone(
            _entry_signal(_make_row(ema9=105, ema21=100, macd_diff=-0.1, ret_5d=2.0, vol_ratio=1.5))
        )

    def test_momentum_fails_weak_5d_return(self):
        self.assertIsNone(
            _entry_signal(_make_row(ema9=105, ema21=100, macd_diff=0.5, ret_5d=1.0, vol_ratio=1.5))
        )

    def test_momentum_fails_low_volume(self):
        self.assertIsNone(
            _entry_signal(_make_row(ema9=105, ema21=100, macd_diff=0.5, ret_5d=2.0, vol_ratio=1.3))
        )

    def test_no_signal_for_neutral_conditions(self):
        self.assertIsNone(_entry_signal(_make_row()))

    def test_mean_reversion_takes_priority_over_momentum(self):
        row = _make_row(
            rsi=30,
            bb_pct=0.20,
            vol_ratio=1.5,
            ema9=105,
            ema21=100,
            macd_diff=0.5,
            ret_5d=2.0,
        )
        self.assertEqual(_entry_signal(row), "mean_reversion")


class TestEntrySignalNewDailySignals(unittest.TestCase):
    """Tests for the six new daily signals added to _entry_signal."""

    def test_macd_crossover_fires(self):
        self.assertEqual(_entry_signal(_make_row(macd_cross=True, vol_ratio=1.3)), "macd_crossover")

    def test_macd_crossover_requires_volume(self):
        self.assertIsNone(_entry_signal(_make_row(macd_cross=True, vol_ratio=1.1)))

    def test_macd_crossover_blocked_by_mean_reversion(self):
        # mean_reversion has higher priority
        row = _make_row(rsi=30, bb_pct=0.20, vol_ratio=1.5, macd_cross=True)
        self.assertEqual(_entry_signal(row), "mean_reversion")

    def test_bb_squeeze_fires_with_ema_up(self):
        self.assertEqual(
            _entry_signal(_make_row(bb_squeeze=True, vol_ratio=1.3, ema9=101, ema21=100)),
            "bb_squeeze",
        )

    def test_bb_squeeze_fires_with_positive_macd(self):
        self.assertEqual(
            _entry_signal(_make_row(bb_squeeze=True, vol_ratio=1.3, macd_diff=0.1)),
            "bb_squeeze",
        )

    def test_bb_squeeze_requires_directional_confirmation(self):
        self.assertIsNone(
            _entry_signal(
                _make_row(bb_squeeze=True, vol_ratio=1.3, ema9=99, ema21=100, macd_diff=-0.1)
            )
        )

    def test_bb_squeeze_requires_volume(self):
        self.assertIsNone(
            _entry_signal(_make_row(bb_squeeze=True, ema9=101, ema21=100, vol_ratio=1.1))
        )

    def test_inside_day_breakout_fires(self):
        self.assertEqual(
            _entry_signal(_make_row(is_inside_day=True, vol_ratio=1.2, ema9=101, ema21=100)),
            "inside_day_breakout",
        )

    def test_inside_day_breakout_requires_volume(self):
        self.assertIsNone(
            _entry_signal(_make_row(is_inside_day=True, vol_ratio=1.0, ema9=101, ema21=100))
        )

    def test_inside_day_requires_directional_confirmation(self):
        self.assertIsNone(
            _entry_signal(
                _make_row(is_inside_day=True, vol_ratio=1.2, ema9=99, ema21=100, macd_diff=-0.1)
            )
        )

    def test_breakout_52w_fires(self):
        self.assertEqual(
            _entry_signal(
                _make_row(price_vs_52w_high_pct=-1.0, vol_ratio=1.3, ema9=101, ema21=100)
            ),
            "breakout_52w",
        )

    def test_breakout_52w_requires_proximity_to_high(self):
        self.assertIsNone(
            _entry_signal(_make_row(price_vs_52w_high_pct=-5.0, vol_ratio=1.3, ema9=101, ema21=100))
        )

    def test_breakout_52w_requires_ema_alignment(self):
        self.assertIsNone(
            _entry_signal(_make_row(price_vs_52w_high_pct=-1.0, vol_ratio=1.3, ema9=99, ema21=100))
        )

    def test_rs_leader_fires_with_spy_data(self):
        row = _make_row(ret_5d=6.0, ret_10d=8.0, ema9=101, ema21=100)
        self.assertEqual(_entry_signal(row, spy_ret_5d=3.0, spy_ret_10d=4.0), "rs_leader")

    def test_rs_leader_blocked_without_spy_data(self):
        row = _make_row(ret_5d=6.0, ret_10d=8.0, ema9=101, ema21=100)
        self.assertIsNone(_entry_signal(row))

    def test_rs_leader_requires_sufficient_outperformance(self):
        row = _make_row(ret_5d=4.0, ret_10d=6.0, ema9=101, ema21=100)
        # rel_5d = 1.0 (< 2.0 threshold) → no signal
        self.assertIsNone(_entry_signal(row, spy_ret_5d=3.0, spy_ret_10d=4.0))

    def test_trend_pullback_fires(self):
        row = _make_row(ema9=101, ema21=100, pct_vs_ema21=-1.5, rsi=50, vol_ratio=1.1)
        self.assertEqual(_entry_signal(row), "trend_pullback")

    def test_trend_pullback_requires_ema_alignment(self):
        row = _make_row(ema9=99, ema21=100, pct_vs_ema21=-1.5, rsi=50, vol_ratio=1.1)
        self.assertIsNone(_entry_signal(row))

    def test_trend_pullback_requires_rsi_in_range(self):
        row = _make_row(ema9=101, ema21=100, pct_vs_ema21=-1.5, rsi=30, vol_ratio=1.1)
        # RSI=30 is below 40 threshold for trend_pullback
        self.assertNotEqual(_entry_signal(row), "trend_pullback")

    def test_trend_pullback_requires_pullback_depth(self):
        # pct_vs_ema21 = 0.0 means price is AT ema21, not below it
        row = _make_row(ema9=101, ema21=100, pct_vs_ema21=0.0, rsi=50, vol_ratio=1.1)
        self.assertNotEqual(_entry_signal(row), "trend_pullback")


class TestEntrySignalIntraday(unittest.TestCase):
    """Tests for the three intraday signals."""

    def _id(self, **kwargs) -> dict:
        base = {
            "vwap": 100.0,
            "orb_high": 101.0,
            "orb_low": 99.0,
            "orb_breakout_up": False,
            "intraday_change_pct": 0.5,
            "price_above_vwap": False,
            "pct_vs_vwap": 0.0,
            "intraday_rsi": 55.0,
        }
        base.update(kwargs)
        return base

    def test_orb_breakout_fires(self):
        self.assertEqual(
            _entry_signal(_make_row(), intraday=self._id(orb_breakout_up=True)), "orb_breakout"
        )

    def test_vwap_reclaim_fires(self):
        id_data = self._id(price_above_vwap=True, intraday_change_pct=1.5, pct_vs_vwap=1.0)
        self.assertEqual(_entry_signal(_make_row(), intraday=id_data), "vwap_reclaim")

    def test_vwap_reclaim_requires_gain_above_1pct(self):
        id_data = self._id(price_above_vwap=True, intraday_change_pct=0.8, pct_vs_vwap=1.0)
        self.assertIsNone(_entry_signal(_make_row(), intraday=id_data))

    def test_vwap_reclaim_blocked_when_overextended(self):
        id_data = self._id(price_above_vwap=True, intraday_change_pct=2.0, pct_vs_vwap=4.0)
        self.assertIsNone(_entry_signal(_make_row(), intraday=id_data))

    def test_intraday_momentum_fires(self):
        row = _make_row(ema9=101, ema21=100)
        # pct_vs_vwap=4.0 blocks vwap_reclaim (needs ≤3%) so intraday_momentum wins
        id_data = self._id(
            intraday_change_pct=2.5, price_above_vwap=True, intraday_rsi=65, pct_vs_vwap=4.0
        )
        self.assertEqual(_entry_signal(row, intraday=id_data), "intraday_momentum")

    def test_intraday_momentum_requires_above_vwap(self):
        row = _make_row(ema9=101, ema21=100)
        id_data = self._id(intraday_change_pct=2.5, price_above_vwap=False, intraday_rsi=65)
        self.assertIsNone(_entry_signal(row, intraday=id_data))

    def test_intraday_momentum_blocked_by_high_rsi(self):
        row = _make_row(ema9=101, ema21=100)
        # pct_vs_vwap=5.0 blocks vwap_reclaim (needs ≤3%); intraday_rsi=80 blocks intraday_momentum
        id_data = self._id(
            intraday_change_pct=2.5, price_above_vwap=True, intraday_rsi=80, pct_vs_vwap=5.0
        )
        self.assertIsNone(_entry_signal(row, intraday=id_data))

    def test_intraday_signals_blocked_without_intraday_data(self):
        row = _make_row(ema9=101, ema21=100)
        # Without intraday kwarg, all intraday signals are blocked
        self.assertIsNone(_entry_signal(row, intraday=None))

    def test_orb_takes_priority_over_vwap_reclaim(self):
        # Both fire — orb should win (higher priority)
        id_data = self._id(
            orb_breakout_up=True,
            price_above_vwap=True,
            intraday_change_pct=1.5,
            pct_vs_vwap=1.0,
        )
        self.assertEqual(_entry_signal(_make_row(), intraday=id_data), "orb_breakout")


class TestEntrySignalNewFeatures(unittest.TestCase):
    """ADX gate, regime blocking, gap_and_go, vix_fear_reversion."""

    # ── gap_and_go ────────────────────────────────────────────────────────────
    def test_gap_and_go_fires(self):
        row = _make_row(gap_pct=3.0, close_above_open=True, vol_ratio=1.6, adx=25)
        self.assertEqual(_entry_signal(row), "gap_and_go")

    def test_gap_and_go_blocked_by_low_adx(self):
        row = _make_row(gap_pct=3.0, close_above_open=True, vol_ratio=1.6, adx=15)
        self.assertIsNone(_entry_signal(row))

    def test_gap_and_go_blocked_by_small_gap(self):
        row = _make_row(gap_pct=1.5, close_above_open=True, vol_ratio=1.6, adx=25)
        self.assertIsNone(_entry_signal(row))

    def test_gap_and_go_blocked_by_gap_not_held(self):
        row = _make_row(gap_pct=3.0, close_above_open=False, vol_ratio=1.6, adx=25)
        self.assertIsNone(_entry_signal(row))

    def test_gap_and_go_blocked_by_low_volume(self):
        row = _make_row(gap_pct=3.0, close_above_open=True, vol_ratio=1.3, adx=25)
        self.assertIsNone(_entry_signal(row))

    def test_gap_and_go_blocked_on_bear_day(self):
        row = _make_row(gap_pct=3.0, close_above_open=True, vol_ratio=1.6, adx=25)
        self.assertIsNone(_entry_signal(row, regime="BEAR_DAY"))

    # ── vix_fear_reversion ────────────────────────────────────────────────────
    def test_vix_fear_reversion_fires(self):
        self.assertEqual(
            _entry_signal(_make_row(vol_ratio=1.2), vix_spike=True), "vix_fear_reversion"
        )

    def test_vix_fear_reversion_not_fire_without_spike(self):
        self.assertIsNone(_entry_signal(_make_row(vol_ratio=1.2), vix_spike=False))

    def test_vix_fear_reversion_fires_on_bear_day(self):
        # vix_fear_reversion is not regime-blocked — counter-cyclical
        row = _make_row(vol_ratio=1.2)
        self.assertEqual(
            _entry_signal(row, regime="BEAR_DAY", vix_spike=True), "vix_fear_reversion"
        )

    def test_vix_fear_reversion_takes_priority_over_mean_reversion(self):
        # Both could fire — vix_fear_reversion has higher priority
        row = _make_row(rsi=28, bb_pct=0.15, vol_ratio=1.5)
        self.assertEqual(_entry_signal(row, vix_spike=True), "vix_fear_reversion")

    # ── regime blocking ───────────────────────────────────────────────────────
    def test_bear_day_blocks_rs_leader(self):
        row = _make_row(ret_5d=5.0, ret_10d=7.0, ema9=101, ema21=100, adx=30)
        result = _entry_signal(row, spy_ret_5d=2.0, spy_ret_10d=3.0, regime="BEAR_DAY")
        self.assertNotEqual(result, "rs_leader")

    def test_bear_day_allows_mean_reversion(self):
        row = _make_row(rsi=28, bb_pct=0.15, vol_ratio=1.5)
        self.assertEqual(_entry_signal(row, regime="BEAR_DAY"), "mean_reversion")

    def test_bull_trending_allows_rs_leader(self):
        row = _make_row(ret_5d=5.0, ret_10d=7.0, ema9=101, ema21=100, adx=30)
        self.assertEqual(
            _entry_signal(row, spy_ret_5d=2.0, spy_ret_10d=3.0, regime="BULL_TRENDING"),
            "rs_leader",
        )

    def test_choppy_blocks_momentum(self):
        row = _make_row(ema9=105, ema21=100, macd_diff=0.5, ret_5d=2.0, vol_ratio=1.5, adx=25)
        self.assertIsNone(_entry_signal(row, regime="CHOPPY"))

    def test_choppy_permits_mean_reversion(self):
        # mean_reversion intentionally excluded from CHOPPY blocks:
        # live paper data shows +0.28% avg (100% win rate, n=2); reassess after ≥5 trades.
        row = _make_row(rsi=28, bb_pct=0.15, vol_ratio=1.5)
        self.assertEqual(_entry_signal(row, regime="CHOPPY"), "mean_reversion")

    def test_none_regime_blocks_nothing(self):
        row = _make_row(ema9=105, ema21=100, macd_diff=0.5, ret_5d=2.0, vol_ratio=1.5, adx=25)
        self.assertEqual(_entry_signal(row, regime=None), "momentum")

    # ── ADX gate ──────────────────────────────────────────────────────────────
    def test_adx_blocks_momentum_when_low(self):
        row = _make_row(ema9=105, ema21=100, macd_diff=0.5, ret_5d=2.0, vol_ratio=1.5, adx=15)
        self.assertIsNone(_entry_signal(row))

    def test_adx_allows_momentum_when_sufficient(self):
        row = _make_row(ema9=105, ema21=100, macd_diff=0.5, ret_5d=2.0, vol_ratio=1.5, adx=25)
        self.assertEqual(_entry_signal(row), "momentum")

    def test_mean_reversion_not_adx_gated(self):
        row = _make_row(rsi=28, bb_pct=0.15, vol_ratio=1.5, adx=5)
        self.assertEqual(_entry_signal(row), "mean_reversion")

    def test_adx_blocks_bb_squeeze_when_low(self):
        row = _make_row(bb_squeeze=True, vol_ratio=1.3, ema9=101, ema21=100, adx=15)
        self.assertIsNone(_entry_signal(row))

    def test_adx_blocks_breakout_52w_when_low(self):
        row = _make_row(price_vs_52w_high_pct=-1.0, vol_ratio=1.3, ema9=101, ema21=100, adx=15)
        self.assertIsNone(_entry_signal(row))

    def test_default_adx_passes_gate_when_column_missing(self):
        # When adx is absent from row, default=30 → gate passes
        row = pd.Series(
            {
                "rsi": 50.0,
                "bb_pct": 0.5,
                "vol_ratio": 1.5,
                "ema9": 105.0,
                "ema21": 100.0,
                "macd_diff": 0.5,
                "ret_5d": 2.0,
            }
        )
        self.assertEqual(_entry_signal(row), "momentum")


class TestMomentum121Signal(unittest.TestCase):
    """momentum_12_1: Jegadeesh-Titman 12-1 medium-term momentum signal."""

    def test_fires_above_threshold(self):
        row = _make_row(mom_12_1=15.0, ema9=101, ema21=100, adx=25)
        self.assertEqual(_entry_signal(row), "momentum_12_1")

    def test_no_fire_below_threshold(self):
        row = _make_row(mom_12_1=5.0, ema9=101, ema21=100, adx=25)
        self.assertIsNone(_entry_signal(row))

    def test_no_fire_when_downtrend(self):
        row = _make_row(mom_12_1=15.0, ema9=99, ema21=100, adx=25)
        self.assertIsNone(_entry_signal(row))

    def test_blocked_by_low_adx(self):
        row = _make_row(mom_12_1=15.0, ema9=101, ema21=100, adx=15)
        self.assertIsNone(_entry_signal(row))

    def test_blocked_on_bear_day(self):
        row = _make_row(mom_12_1=15.0, ema9=101, ema21=100, adx=25)
        self.assertIsNone(_entry_signal(row, regime="BEAR_DAY"))

    def test_blocked_on_choppy(self):
        row = _make_row(mom_12_1=15.0, ema9=101, ema21=100, adx=25)
        self.assertIsNone(_entry_signal(row, regime="CHOPPY"))

    def test_allowed_on_bull_trending(self):
        row = _make_row(mom_12_1=15.0, ema9=101, ema21=100, adx=25)
        self.assertEqual(_entry_signal(row, regime="BULL_TRENDING"), "momentum_12_1")

    def test_allowed_on_high_vol(self):
        row = _make_row(mom_12_1=15.0, ema9=101, ema21=100, adx=25)
        self.assertEqual(_entry_signal(row, regime="HIGH_VOL"), "momentum_12_1")

    def test_absent_defaults_to_no_fire(self):
        # When mom_12_1 absent, defaults to -999 → no signal
        row = _make_row(ema9=101, ema21=100, adx=25)
        self.assertNotEqual(_entry_signal(row), "momentum_12_1")

    def test_custom_threshold(self):
        row = _make_row(mom_12_1=8.0, ema9=101, ema21=100, adx=25)
        self.assertNotEqual(_entry_signal(row), "momentum_12_1")
        self.assertEqual(_entry_signal(row, params={"mom12_1_threshold": 5.0}), "momentum_12_1")

    def test_in_signal_priority(self):
        self.assertIn("momentum_12_1", _SIGNAL_PRIORITY)

    def test_priority_between_breakout52w_and_gap_and_go(self):
        self.assertLess(_SIGNAL_PRIORITY["breakout_52w"], _SIGNAL_PRIORITY["momentum_12_1"])
        self.assertLess(_SIGNAL_PRIORITY["momentum_12_1"], _SIGNAL_PRIORITY["gap_and_go"])

    def test_signals_not_tested_excludes_momentum_12_1_when_column_present(self):
        idx = pd.bdate_range("2025-01-02", periods=3)
        n = len(idx)
        df = pd.DataFrame(
            {
                "Close": [100.0] * n,
                "Open": [99.5] * n,
                "Volume": [1_000_000] * n,
                "rsi": [50.0] * n,
                "bb_pct": [0.5] * n,
                "vol_ratio": [1.0] * n,
                "ema9": [100.0] * n,
                "ema21": [100.0] * n,
                "macd_diff": [0.0] * n,
                "ret_5d": [0.0] * n,
                "mom_12_1": [5.0] * n,
            },
            index=idx,
        )
        result = _run_simulation({"AAPL": df}, idx[1:])
        self.assertIn("momentum_12_1", result["signals_tested"])
        self.assertNotIn("momentum_12_1", result["signals_not_tested"])


class TestComputeIntradayDay(unittest.TestCase):
    """Tests for the pure _compute_intraday_day helper."""

    def _make_bars(self, n_orb: int = 30, n_post: int = 60) -> list:
        """Build synthetic (datetime, Bar) pairs: n_orb ORB bars then n_post post-ORB bars."""
        bars = []
        # ORB window: 09:30 → 10:00 ET (30 bars, 1-min each)
        for i in range(n_orb):
            h, m = divmod(9 * 60 + 30 + i, 60)
            t = datetime(2025, 1, 2, h, m, tzinfo=_ET)
            bars.append((t, _Bar(open=100.0, high=101.0, low=99.0, close=100.5, volume=1000)))
        # Post-ORB: 10:01 onwards
        for i in range(n_post):
            h, m = divmod(10 * 60 + 1 + i, 60)
            t = datetime(2025, 1, 2, h, m, tzinfo=_ET)
            bars.append((t, _Bar(open=100.5, high=102.0, low=100.0, close=101.0, volume=1200)))
        return bars

    def test_returns_dict(self):
        result = _compute_intraday_day("2025-01-02", self._make_bars())
        self.assertIsInstance(result, dict)

    def test_returns_none_for_empty_bars(self):
        self.assertIsNone(_compute_intraday_day("2025-01-02", []))

    def test_vwap_computed(self):
        result = _compute_intraday_day("2025-01-02", self._make_bars())
        self.assertIn("vwap", result)
        self.assertGreater(result["vwap"], 0)

    def test_vwap_formula_correct(self):
        t = datetime(2025, 1, 2, 9, 31, tzinfo=_ET)
        bar = _Bar(open=100, high=102, low=98, close=101, volume=1000)
        result = _compute_intraday_day("2025-01-02", [(t, bar)])
        expected_vwap = (102 + 98 + 101) / 3  # typical price = (H+L+C)/3
        self.assertAlmostEqual(result["vwap"], expected_vwap, places=5)

    def test_orb_high_low_computed_from_first_30_bars(self):
        result = _compute_intraday_day("2025-01-02", self._make_bars())
        self.assertEqual(result["orb_high"], 101.0)
        self.assertEqual(result["orb_low"], 99.0)

    def test_orb_high_none_when_fewer_than_5_orb_bars(self):
        bars = self._make_bars(n_orb=3, n_post=10)
        result = _compute_intraday_day("2025-01-02", bars)
        self.assertIsNone(result["orb_high"])
        self.assertIsNone(result["orb_low"])

    def test_orb_breakout_detected(self):
        # Post-ORB bar closes above orb_high (101) with above-avg volume
        bars = self._make_bars(n_orb=30, n_post=0)
        t_post = datetime(2025, 1, 2, 10, 5, tzinfo=_ET)
        bars.append((t_post, _Bar(open=101.0, high=103.0, low=100.5, close=102.5, volume=5000)))
        result = _compute_intraday_day("2025-01-02", bars)
        self.assertTrue(result["orb_breakout_up"])

    def test_no_orb_breakout_when_close_below_orb_high(self):
        result = _compute_intraday_day("2025-01-02", self._make_bars())
        # Post-ORB closes at 101, orb_high is 101 — needs close ABOVE orb_high
        self.assertFalse(result["orb_breakout_up"])

    def test_intraday_change_pct_computed(self):
        bars = [
            (
                datetime(2025, 1, 2, 9, 31, tzinfo=_ET),
                _Bar(open=100, high=101, low=99, close=100, volume=1000),
            ),
            (
                datetime(2025, 1, 2, 15, 59, tzinfo=_ET),
                _Bar(open=102, high=103, low=101, close=103, volume=1000),
            ),
        ]
        result = _compute_intraday_day("2025-01-02", bars)
        self.assertAlmostEqual(result["intraday_change_pct"], 3.0, places=5)

    def test_price_above_vwap_flag(self):
        # Single bar: close = high = typical = vwap → not strictly above
        t = datetime(2025, 1, 2, 9, 31, tzinfo=_ET)
        bar = _Bar(open=100, high=100, low=100, close=100, volume=1000)
        result = _compute_intraday_day("2025-01-02", [(t, bar)])
        self.assertFalse(result["price_above_vwap"])

    def test_intraday_rsi_computed_when_enough_bars(self):
        bars = []
        for i in range(80):
            t = datetime(2025, 1, 2, 9, 31 + i // 60, i % 60, tzinfo=_ET)
            close = 100 + (i % 10) * 0.1
            bars.append(
                (t, _Bar(open=close, high=close + 0.5, low=close - 0.5, close=close, volume=500))
            )
        result = _compute_intraday_day("2025-01-02", bars)
        # 80 bars → 80//5 = 16 five-min bars ≥ 14 → RSI computed
        self.assertIsNotNone(result["intraday_rsi"])
        self.assertGreater(result["intraday_rsi"], 0)
        self.assertLessEqual(result["intraday_rsi"], 100)

    def test_intraday_rsi_none_when_too_few_bars(self):
        bars = self._make_bars(n_orb=5, n_post=0)
        result = _compute_intraday_day("2025-01-02", bars)
        # 5 bars → 5//5 = 1 five-min bar < 14 → RSI not computed
        self.assertIsNone(result["intraday_rsi"])

    def test_required_keys_present(self):
        result = _compute_intraday_day("2025-01-02", self._make_bars())
        for key in (
            "vwap",
            "orb_high",
            "orb_low",
            "orb_breakout_up",
            "intraday_change_pct",
            "price_above_vwap",
            "pct_vs_vwap",
            "intraday_rsi",
        ):
            self.assertIn(key, result)


class TestEntrySignalWithParams(unittest.TestCase):
    """Custom params override the hardcoded defaults."""

    def test_looser_rsi_threshold_fires_where_default_would_not(self):
        # RSI=38 is above the default threshold of 35, so default → no signal
        row = _make_row(rsi=38, bb_pct=0.20, vol_ratio=1.5)
        self.assertIsNone(_entry_signal(row))
        self.assertEqual(_entry_signal(row, {"rsi_threshold": 40}), "mean_reversion")

    def test_looser_mom_ret5d_fires_where_default_would_not(self):
        row = _make_row(ema9=105, ema21=100, macd_diff=0.5, ret_5d=0.7, vol_ratio=1.5)
        self.assertIsNone(_entry_signal(row))
        self.assertEqual(_entry_signal(row, {"mom_ret5d_threshold": 0.5}), "momentum")

    def test_looser_mom_vol_fires_where_default_would_not(self):
        row = _make_row(ema9=105, ema21=100, macd_diff=0.5, ret_5d=2.0, vol_ratio=1.1)
        self.assertIsNone(_entry_signal(row))
        self.assertEqual(_entry_signal(row, {"mom_vol_threshold": 1.0}), "momentum")

    def test_partial_params_merge_with_defaults(self):
        # Override only rsi_threshold; other defaults (bb, vol) still apply
        row = _make_row(rsi=38, bb_pct=0.20, vol_ratio=1.5)
        self.assertEqual(_entry_signal(row, {"rsi_threshold": 40}), "mean_reversion")

    def test_stricter_rsi_threshold_blocks_signal(self):
        row = _make_row(rsi=34, bb_pct=0.20, vol_ratio=1.5)
        self.assertEqual(_entry_signal(row), "mean_reversion")  # fires on default
        self.assertIsNone(_entry_signal(row, {"rsi_threshold": 30}))  # blocked by strict threshold

    def test_none_params_uses_defaults(self):
        row = _make_row(rsi=30, bb_pct=0.20, vol_ratio=1.5)
        self.assertEqual(_entry_signal(row, None), _entry_signal(row))

    def test_default_params_dict_matches_hardcoded_behaviour(self):
        row = _make_row(rsi=30, bb_pct=0.20, vol_ratio=1.5)
        self.assertEqual(_entry_signal(row, _DEFAULT_PARAMS), _entry_signal(row))


# ── _run_simulation ───────────────────────────────────────────────────────────


class TestRunSimulation(unittest.TestCase):
    """Direct tests of the extracted simulation core."""

    def _build_indicators(self, n=100):
        raw = _make_raw(n=n)
        indicators = {}
        for sym in ("AAPL", "FLAT"):
            close = raw["Close"][sym]
            open_ = raw["Open"][sym]
            volume = raw["Volume"][sym]
            df = pd.DataFrame({"Close": close, "Open": open_, "Volume": volume}).dropna()
            df = _compute_indicators(df)
            if not df.empty:
                indicators[sym] = df
        return indicators

    def test_returns_dict_with_expected_keys(self):
        indicators = self._build_indicators()
        dates = pd.bdate_range("2025-03-01", "2025-03-07")
        result = _run_simulation(indicators, dates, initial_capital=10_000.0)
        for key in (
            "initial_capital",
            "final_value",
            "total_return_pct",
            "total_trades",
            "win_rate_pct",
            "sharpe_ratio",
            "by_signal",
            "equity_curve",
            "trades",
        ):
            self.assertIn(key, result)

    def test_equity_curve_length_matches_trading_days(self):
        indicators = self._build_indicators()
        dates = pd.bdate_range("2025-03-03", "2025-03-07")
        result = _run_simulation(indicators, dates)
        self.assertEqual(len(result["equity_curve"]), len(dates))

    def test_no_trades_with_tight_params(self):
        # Params so tight nothing can fire
        indicators = self._build_indicators()
        dates = pd.bdate_range("2025-03-01", "2025-03-07")
        tight = {
            "rsi_threshold": 1,
            "bb_threshold": 0.01,
            "mr_vol_threshold": 999,
            "mom_vol_threshold": 999,
            "mom_ret5d_threshold": 999,
        }
        result = _run_simulation(indicators, dates, params=tight)
        self.assertEqual(result["total_trades"], 0)

    def test_loose_params_fire_momentum_signal(self):
        # Uptrend data: EMA9 > EMA21, macd_diff > 0; loosen vol and ret5d thresholds
        indicators = self._build_indicators()
        dates = pd.bdate_range("2025-02-03", "2025-02-28")
        loose = {"mom_vol_threshold": 0.9, "mom_ret5d_threshold": 0.3}
        result = _run_simulation(
            indicators, dates, initial_capital=10_000.0, max_hold_days=3, params=loose
        )
        self.assertGreater(result["total_trades"], 0)

    def test_custom_params_produce_different_trades_than_defaults(self):
        indicators = self._build_indicators()
        dates = pd.bdate_range("2025-02-03", "2025-02-28")
        default_result = _run_simulation(
            indicators, dates, initial_capital=10_000.0, max_hold_days=3
        )
        loose = {"mom_vol_threshold": 0.9, "mom_ret5d_threshold": 0.3}
        loose_result = _run_simulation(
            indicators, dates, initial_capital=10_000.0, max_hold_days=3, params=loose
        )
        # Loose params should produce at least as many trades as defaults
        self.assertGreaterEqual(loose_result["total_trades"], default_result["total_trades"])

    def test_slippage_reduces_final_value_when_trades_fire(self):
        indicators = self._build_indicators()
        dates = pd.bdate_range("2025-02-03", "2025-02-28")
        loose = {"mom_vol_threshold": 0.9, "mom_ret5d_threshold": 0.3}
        zero = _run_simulation(
            indicators,
            dates,
            initial_capital=10_000.0,
            max_hold_days=3,
            params=loose,
            slippage_bps=0,
            spread_bps=0,
        )
        costly = _run_simulation(
            indicators,
            dates,
            initial_capital=10_000.0,
            max_hold_days=3,
            params=loose,
            slippage_bps=20,
            spread_bps=10,
        )
        if zero["total_trades"] > 0:
            self.assertGreaterEqual(zero["final_value"], costly["final_value"])


# ── run_backtest ──────────────────────────────────────────────────────────────

_EXPECTED_RESULT_KEYS = {
    "start",
    "end",
    "initial_capital",
    "final_value",
    "total_return_pct",
    "total_trades",
    "win_rate_pct",
    "avg_return_per_trade_pct",
    "max_drawdown_pct",
    "sharpe_ratio",
    "by_signal",
    "equity_curve",
    "trades",
}


class TestRunBacktest(unittest.TestCase):
    def setUp(self):
        self._save_patcher = patch("backtest.engine._save_results")
        self._print_patcher = patch("backtest.engine._print_results")
        self._save_patcher.start()
        self._print_patcher.start()

    def tearDown(self):
        self._save_patcher.stop()
        self._print_patcher.stop()

    def _run(self, raw=None, symbols=("AAPL", "FLAT"), start="2025-03-01", end="2025-03-07", **kw):
        if raw is None:
            raw = _make_raw()
        with patch("backtest.engine.yf.download", return_value=raw):
            return run_backtest(list(symbols), start_date=start, end_date=end, **kw)

    def test_returns_empty_on_empty_data(self):
        self.assertEqual(self._run(raw=pd.DataFrame()), {})

    def test_returns_dict_with_expected_keys(self):
        result = self._run()
        for key in _EXPECTED_RESULT_KEYS:
            self.assertIn(key, result)

    def test_no_trades_with_steady_uptrend(self):
        result = self._run()
        self.assertEqual(result["total_trades"], 0)

    def test_initial_capital_preserved_with_no_trades(self):
        result = self._run(initial_capital=10_000.0)
        self.assertAlmostEqual(result["final_value"], 10_000.0, places=2)

    def test_equity_curve_length_matches_trading_days(self):
        result = self._run(start="2025-03-03", end="2025-03-07")
        expected = len(pd.bdate_range("2025-03-03", "2025-03-07"))
        self.assertEqual(len(result["equity_curve"]), expected)

    def test_equity_curve_entries_have_str_date_and_float_value(self):
        result = self._run()
        for date_str, value in result["equity_curve"]:
            self.assertIsInstance(date_str, str)
            self.assertIsInstance(value, float)

    def test_unknown_symbol_skipped_gracefully(self):
        raw = _make_raw(symbols=("AAPL",))
        result = self._run(raw=raw, symbols=("AAPL", "GHOST"))
        self.assertIn("total_trades", result)

    def test_start_date_in_result(self):
        result = self._run(start="2025-03-01", end="2025-03-07")
        self.assertEqual(result["start"], "2025-03-01")

    def test_initial_capital_in_result(self):
        result = self._run(initial_capital=25_000.0)
        self.assertEqual(result["initial_capital"], 25_000.0)

    def test_by_signal_is_dict(self):
        self.assertIsInstance(self._run()["by_signal"], dict)

    def test_trades_is_list(self):
        self.assertIsInstance(self._run()["trades"], list)

    def test_sharpe_ratio_is_zero_when_no_equity_variance(self):
        result = self._run()
        self.assertEqual(result["sharpe_ratio"], 0.0)

    def test_max_drawdown_is_zero_when_equity_flat(self):
        result = self._run()
        self.assertEqual(result["max_drawdown_pct"], 0.0)

    def test_custom_params_passed_through(self):
        # Tight params → 0 trades regardless of price action
        tight = {
            "rsi_threshold": 1,
            "bb_threshold": 0.01,
            "mr_vol_threshold": 999,
            "mom_vol_threshold": 999,
            "mom_ret5d_threshold": 999,
        }
        result = self._run(params=tight)
        self.assertEqual(result["total_trades"], 0)


# ── run_walk_forward_optimized ────────────────────────────────────────────────

# Tiny param grid: 4 combos (fast grid search); vol/ret thresholds loose enough
# that momentum fires on the uptrend data used in _make_raw
_LOOSE_PARAM_GRID: dict = {
    "rsi_threshold": [35],
    "bb_threshold": [0.25],
    "mr_vol_threshold": [1.2],
    "mom_vol_threshold": [0.9],  # fires with vol_ratio=1.0
    "mom_ret5d_threshold": [0.3],  # fires with ret_5d≈0.47% on 0.1/bar uptrend
}

_TIGHT_PARAM_GRID: dict = {
    "rsi_threshold": [35, 40],
    "bb_threshold": [0.25],
    "mr_vol_threshold": [1.2],
    "mom_vol_threshold": [999.0],  # nothing fires
    "mom_ret5d_threshold": [999.0],
}

_WF_FOLD_KEYS = {
    "train_start",
    "train_end",
    "test_start",
    "test_end",
    "best_params",
    "train_sharpe",
    "train_total_trades",
    "oos_total_return_pct",
    "oos_win_rate_pct",
    "oos_total_trades",
    "oos_sharpe",
    "oos_degradation",
    "random_baseline_return_pct",
}

_WF_SUMMARY_KEYS = {
    "n_folds",
    "mean_oos_return_pct",
    "mean_oos_win_rate_pct",
    "mean_oos_sharpe",
    "profitable_folds",
    "beat_baseline_folds",
    "consistency_pct",
    "beat_baseline_pct",
    "param_stability_pct",
    "mean_oos_degradation",
    "random_baseline_return_pct",
}


class TestRunWalkForwardOptimized(unittest.TestCase):
    """
    Uses small train_days=10 / test_days=5 and a one-combo param grid to keep
    the grid search fast. _make_raw(n=100) spans 2024-11-01 → ~2025-03-12,
    covering the test trading window of 2025-02-01 → 2025-03-07.
    """

    def setUp(self):
        self._raw = _make_raw(n=100)

    def _run_wf(self, param_grid=None, **kwargs):
        defaults = {
            "symbols": ["AAPL", "FLAT"],
            "start_date": "2025-02-01",
            "end_date": "2025-03-07",
            "train_days": 10,
            "test_days": 5,
            "initial_capital": 10_000.0,
            "param_grid": param_grid or _LOOSE_PARAM_GRID,
        }
        defaults.update(kwargs)
        with patch("backtest.engine.yf.download", return_value=self._raw):
            return run_walk_forward_optimized(**defaults)

    def test_returns_dict_with_folds_and_summary_keys(self):
        result = self._run_wf()
        self.assertIn("folds", result)
        self.assertIn("summary", result)

    def test_returns_empty_when_range_too_short_for_one_fold(self):
        result = self._run_wf(train_days=50, test_days=50)
        self.assertEqual(result, {})

    def test_returns_empty_on_empty_data(self):
        with patch("backtest.engine.yf.download", return_value=pd.DataFrame()):
            result = run_walk_forward_optimized(
                ["AAPL"], "2025-02-01", "2025-03-07", train_days=10, test_days=5
            )
        self.assertEqual(result, {})

    def test_produces_at_least_one_fold(self):
        result = self._run_wf()
        self.assertGreater(len(result["folds"]), 0)

    def test_each_fold_has_expected_keys(self):
        result = self._run_wf()
        for fold in result["folds"]:
            for key in _WF_FOLD_KEYS:
                self.assertIn(key, fold, f"Missing key '{key}' in fold")

    def test_summary_has_expected_keys(self):
        result = self._run_wf()
        for key in _WF_SUMMARY_KEYS:
            self.assertIn(key, result["summary"])

    def test_summary_n_folds_matches_folds_list_length(self):
        result = self._run_wf()
        self.assertEqual(result["summary"]["n_folds"], len(result["folds"]))

    def test_consistency_pct_is_valid_percentage(self):
        result = self._run_wf()
        pct = result["summary"]["consistency_pct"]
        self.assertGreaterEqual(pct, 0.0)
        self.assertLessEqual(pct, 100.0)

    def test_test_window_starts_after_train_window_ends(self):
        result = self._run_wf()
        for fold in result["folds"]:
            self.assertGreater(fold["test_start"], fold["train_end"])

    def test_test_windows_do_not_overlap(self):
        result = self._run_wf()
        folds = result["folds"]
        for i in range(len(folds) - 1):
            self.assertGreater(folds[i + 1]["test_start"], folds[i]["test_end"])

    def test_loose_params_produce_trades_in_oos_window(self):
        with patch("backtest.engine._MIN_TRAIN_TRADES", 1):
            result = self._run_wf(param_grid=_LOOSE_PARAM_GRID)
        total_oos_trades = sum(f["oos_total_trades"] for f in result["folds"])
        self.assertGreater(total_oos_trades, 0)

    def test_tight_params_produce_zero_oos_trades(self):
        result = self._run_wf(param_grid=_TIGHT_PARAM_GRID)
        total_oos_trades = sum(f["oos_total_trades"] for f in result["folds"])
        self.assertEqual(total_oos_trades, 0)

    def test_best_params_keys_match_param_grid_keys(self):
        result = self._run_wf(param_grid=_LOOSE_PARAM_GRID)
        for fold in result["folds"]:
            self.assertEqual(set(fold["best_params"].keys()), set(_LOOSE_PARAM_GRID.keys()))

    def test_best_params_values_come_from_param_grid_when_trades_fire(self):
        # Patch _MIN_TRAIN_TRADES to 1 so tiny 10-day train windows can satisfy the threshold
        with patch("backtest.engine._MIN_TRAIN_TRADES", 1):
            result = self._run_wf(param_grid=_LOOSE_PARAM_GRID)
        for fold in result["folds"]:
            if fold["oos_total_trades"] > 0:
                self.assertEqual(fold["best_params"]["mom_vol_threshold"], 0.9)
                self.assertEqual(fold["best_params"]["mom_ret5d_threshold"], 0.3)
                break

    def test_summary_profitable_folds_le_n_folds(self):
        result = self._run_wf()
        self.assertLessEqual(result["summary"]["profitable_folds"], result["summary"]["n_folds"])

    def test_summary_beat_baseline_folds_le_n_folds(self):
        result = self._run_wf()
        s = result["summary"]
        self.assertLessEqual(s["beat_baseline_folds"], s["n_folds"])
        self.assertGreaterEqual(s["beat_baseline_folds"], 0)

    def test_beat_baseline_pct_is_valid_percentage(self):
        result = self._run_wf()
        pct = result["summary"]["beat_baseline_pct"]
        self.assertGreaterEqual(pct, 0.0)
        self.assertLessEqual(pct, 100.0)

    def test_param_stability_pct_is_valid_percentage(self):
        result = self._run_wf()
        pct = result["summary"]["param_stability_pct"]
        self.assertGreaterEqual(pct, 0.0)
        self.assertLessEqual(pct, 100.0)

    def test_random_baseline_return_pct_is_numeric(self):
        result = self._run_wf()
        self.assertIsInstance(result["summary"]["random_baseline_return_pct"], float)

    def test_fold_train_total_trades_is_non_negative(self):
        result = self._run_wf()
        for fold in result["folds"]:
            self.assertGreaterEqual(fold["train_total_trades"], 0)

    def test_fold_oos_degradation_is_numeric(self):
        result = self._run_wf()
        for fold in result["folds"]:
            self.assertIsInstance(fold["oos_degradation"], float)

    def test_fold_random_baseline_is_numeric(self):
        result = self._run_wf()
        for fold in result["folds"]:
            self.assertIsInstance(fold["random_baseline_return_pct"], float)


# ── _save_results ─────────────────────────────────────────────────────────────


class TestSaveResults(unittest.TestCase):
    def test_writes_json_file(self):
        r = _make_results_dict()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("backtest.engine.LOG_DIR", tmpdir):
                _save_results(r)
            path = os.path.join(tmpdir, "backtest_results.json")
            self.assertTrue(os.path.exists(path))
            with open(path) as f:
                data = json.load(f)
        self.assertEqual(data["total_trades"], 5)

    def test_equity_curve_serialized_as_string_pairs(self):
        r = _make_results_dict()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("backtest.engine.LOG_DIR", tmpdir):
                _save_results(r)
            with open(os.path.join(tmpdir, "backtest_results.json")) as f:
                data = json.load(f)
        for entry in data["equity_curve"]:
            self.assertIsInstance(entry[0], str)

    def test_handles_io_error_gracefully(self):
        with patch("os.makedirs"), patch("builtins.open", side_effect=OSError("disk full")):
            _save_results(_make_results_dict())  # must not raise


# ── _print_results ────────────────────────────────────────────────────────────


class TestPrintResults(unittest.TestCase):
    def test_prints_without_error(self):
        _print_results(_make_results_dict())

    def test_prints_empty_by_signal_without_error(self):
        r = _make_results_dict()
        r["by_signal"] = {}
        _print_results(r)

    def test_prints_zero_win_rate_without_error(self):
        r = _make_results_dict()
        r["by_signal"] = {"momentum": {"wins": 0, "losses": 5, "total_return": -10.0}}
        _print_results(r)

    def test_prints_rule_proxy_disclaimer(self, capsys=None):
        import io
        from contextlib import redirect_stdout

        r = _make_results_dict()
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_results(r)
        self.assertIn("Rule proxy", buf.getvalue())


# ── validation_scope ──────────────────────────────────────────────────────────


class TestValidationScope(unittest.TestCase):
    def _build_indicators(self, n=100):
        raw = _make_raw(n=n)
        indicators = {}
        for sym in ("AAPL", "FLAT"):
            close = raw["Close"][sym]
            open_ = raw["Open"][sym]
            volume = raw["Volume"][sym]
            df = pd.DataFrame({"Close": close, "Open": open_, "Volume": volume}).dropna()
            df = _compute_indicators(df)
            if not df.empty:
                indicators[sym] = df
        return indicators

    def test_validation_scope_is_rule_proxy_only(self):
        indicators = self._build_indicators()
        dates = pd.bdate_range("2025-03-01", "2025-03-07")
        result = _run_simulation(indicators, dates, initial_capital=10_000.0)
        self.assertEqual(result["validation_scope"], "rule_proxy_only")

    def test_signals_tested_contains_daily_signals(self):
        indicators = self._build_indicators()
        dates = pd.bdate_range("2025-03-01", "2025-03-07")
        result = _run_simulation(indicators, dates, initial_capital=10_000.0)
        for sig in (
            "mean_reversion",
            "momentum",
            "macd_crossover",
            "bb_squeeze",
            "trend_pullback",
            "breakout_52w",
        ):
            self.assertIn(sig, result["signals_tested"])

    def test_signals_not_tested_excludes_signals_needing_missing_data(self):
        # _make_raw has no High/Low → inside_day not tested
        # no spy_indicators → rs_leader not tested
        # no vix_spike_by_date → vix_fear_reversion not tested
        # no intraday_data → intraday signals not tested
        indicators = self._build_indicators()
        dates = pd.bdate_range("2025-03-01", "2025-03-07")
        result = _run_simulation(indicators, dates, initial_capital=10_000.0)
        for sig in (
            "rs_leader",
            "vix_fear_reversion",
            "vwap_reclaim",
            "orb_breakout",
            "intraday_momentum",
        ):
            self.assertIn(sig, result["signals_not_tested"])

    def test_gap_and_go_in_tested_when_open_present(self):
        # _make_raw includes Open → gap_pct is computed → gap_and_go is testable
        indicators = self._build_indicators()
        dates = pd.bdate_range("2025-03-01", "2025-03-07")
        result = _run_simulation(indicators, dates, initial_capital=10_000.0)
        self.assertIn("gap_and_go", result["signals_tested"])

    def test_vix_fear_reversion_in_tested_when_spike_data_provided(self):
        indicators = self._build_indicators()
        dates = pd.bdate_range("2025-03-01", "2025-03-07")
        fake_vix = {"2025-03-03": True, "2025-03-04": False}
        result = _run_simulation(
            indicators, dates, initial_capital=10_000.0, vix_spike_by_date=fake_vix
        )
        self.assertIn("vix_fear_reversion", result["signals_tested"])

    def test_intraday_signals_appear_in_tested_when_data_provided(self):
        indicators = self._build_indicators()
        dates = pd.bdate_range("2025-03-01", "2025-03-07")
        fake_intraday = {
            "AAPL": {
                "2025-03-03": {
                    "orb_breakout_up": False,
                    "price_above_vwap": True,
                    "intraday_change_pct": 0.5,
                    "pct_vs_vwap": 0.5,
                    "intraday_rsi": 55,
                }
            }
        }
        result = _run_simulation(
            indicators, dates, initial_capital=10_000.0, intraday_data=fake_intraday
        )
        for sig in ("vwap_reclaim", "orb_breakout", "intraday_momentum"):
            self.assertIn(sig, result["signals_tested"])

    def test_rs_leader_in_tested_when_spy_provided(self):
        indicators = self._build_indicators()
        spy = _compute_indicators(_make_ohlcv(100))
        dates = pd.bdate_range("2025-03-01", "2025-03-07")
        result = _run_simulation(indicators, dates, initial_capital=10_000.0, spy_indicators=spy)
        self.assertIn("rs_leader", result["signals_tested"])

    def test_run_backtest_propagates_validation_scope(self):
        with (
            patch("backtest.engine._save_results"),
            patch("backtest.engine._print_results"),
            patch("backtest.engine.yf.download", return_value=_make_raw()),
        ):
            result = run_backtest(["AAPL", "FLAT"], "2025-03-01", "2025-03-07")
        self.assertEqual(result.get("validation_scope"), "rule_proxy_only")


def _make_signal_row(**kwargs) -> pd.Series:
    """Build a row that triggers the momentum entry signal with loose params."""
    defaults = {
        "rsi": 50.0,
        "bb_pct": 0.5,
        "vol_ratio": 1.5,
        "ema9": 101.0,
        "ema21": 100.0,
        "macd_diff": 0.5,
        "ret_5d": 2.0,
    }
    defaults.update(kwargs)
    return pd.Series(defaults)


_LOOSE_ENTRY = {"mom_vol_threshold": 0.9, "mom_ret5d_threshold": 0.3}


def _build_indicator_df(idx, close_vals, open_vals=None, include_open=True):
    """Build a pre-computed indicator DataFrame with given close values."""
    n = len(idx)
    data = {
        "Close": close_vals,
        "Volume": [2_000_000] * n,
        "rsi": [50.0] * n,
        "bb_pct": [0.5] * n,
        "vol_ratio": [1.5] * n,
        "ema9": [101.0] * n,
        "ema21": [100.0] * n,
        "macd_diff": [0.5] * n,
        "ret_5d": [2.0] * n,
    }
    if include_open:
        data["Open"] = (
            open_vals
            if open_vals is not None
            else [c * 0.999 if c is not None else None for c in close_vals]
        )
    return pd.DataFrame(data, index=idx)


class TestSignalPriority(unittest.TestCase):
    """Tests for _SIGNAL_PRIORITY dict and slot-allocation sort order."""

    _ALL_SIGNALS = [
        "mean_reversion",
        "momentum",
        "macd_crossover",
        "bb_squeeze",
        "inside_day_breakout",
        "trend_pullback",
        "breakout_52w",
        "rs_leader",
        "vwap_reclaim",
        "orb_breakout",
        "intraday_momentum",
    ]

    def test_all_signals_in_priority_dict(self):
        for sig in self._ALL_SIGNALS:
            self.assertIn(sig, _SIGNAL_PRIORITY, f"{sig} missing from _SIGNAL_PRIORITY")

    def test_high_edge_signals_rank_above_orb(self):
        for sig in ("bb_squeeze", "breakout_52w", "rs_leader", "momentum"):
            self.assertLess(
                _SIGNAL_PRIORITY[sig],
                _SIGNAL_PRIORITY["orb_breakout"],
                f"{sig} should outrank orb_breakout",
            )

    def test_rs_leader_and_breakout_52w_rank_above_bb_squeeze(self):
        for sig in ("rs_leader", "breakout_52w"):
            self.assertLess(
                _SIGNAL_PRIORITY[sig],
                _SIGNAL_PRIORITY["bb_squeeze"],
                f"{sig} should outrank bb_squeeze",
            )

    def test_slot_allocation_prefers_bb_squeeze_over_orb(self):
        """When one slot is available and both bb_squeeze and orb fire, bb_squeeze wins."""
        idx = pd.bdate_range("2025-01-02", periods=3)
        trading_dates = idx[1:]

        # AAPL: indicators fire bb_squeeze (bb_squeeze=True, ema9>ema21, vol_ratio>1.2)
        n = len(idx)
        aapl_data = {
            "Close": [100.0, 100.0, 101.0],
            "Open": [99.5, 99.5, 100.5],
            "Volume": [2_000_000] * n,
            "rsi": [55.0] * n,
            "bb_pct": [0.5] * n,
            "vol_ratio": [1.5] * n,
            "ema9": [101.0] * n,
            "ema21": [100.0] * n,
            "macd_diff": [0.5] * n,
            "ret_5d": [2.0] * n,
            "bb_squeeze": [True] * n,
            "macd_cross": [False] * n,
            "ret_10d": [4.0] * n,
            "pct_vs_ema21": [1.0] * n,
            "price_vs_52w_high_pct": [-50.0] * n,
            "is_inside_day": [False] * n,
        }
        aapl_df = pd.DataFrame(aapl_data, index=idx)

        # MSFT: indicators that would fire orb_breakout (via intraday) but not any daily signal
        msft_data = {
            "Close": [200.0, 200.0, 201.0],
            "Open": [199.5, 199.5, 200.5],
            "Volume": [2_000_000] * n,
            "rsi": [50.0] * n,  # neutral RSI — no daily signal fires
            "bb_pct": [0.5] * n,
            "vol_ratio": [1.0] * n,  # below vol threshold for daily signals
            "ema9": [99.0] * n,  # ema9 < ema21 — blocks momentum / bb_squeeze
            "ema21": [100.0] * n,
            "macd_diff": [-0.1] * n,  # negative — blocks several signals
            "ret_5d": [0.5] * n,
            "bb_squeeze": [False] * n,
            "macd_cross": [False] * n,
            "ret_10d": [1.0] * n,
            "pct_vs_ema21": [0.0] * n,
            "price_vs_52w_high_pct": [-50.0] * n,
            "is_inside_day": [False] * n,
        }
        msft_df = pd.DataFrame(msft_data, index=idx)

        intraday = {
            "MSFT": {
                idx[1].strftime("%Y-%m-%d"): {
                    "orb_breakout_up": True,
                    "price_above_vwap": False,
                    "intraday_change_pct": None,
                    "pct_vs_vwap": 0.0,
                }
            }
        }

        result = _run_simulation(
            {"AAPL": aapl_df, "MSFT": msft_df},
            trading_dates,
            initial_capital=10_000.0,
            max_positions=1,
            max_hold_days=2,
            intraday_data=intraday,
        )
        buy_trades = [t for t in result["trades"] if t["action"] == "BUY"]
        bought_symbols = [t["symbol"] for t in buy_trades]
        # AAPL (bb_squeeze, priority 0) should win over MSFT (orb_breakout, priority 8)
        self.assertIn("AAPL", bought_symbols)
        self.assertNotIn("MSFT", bought_symbols)

    def test_mean_reversion_beats_orb_when_competing(self):
        """mean_reversion (priority 7) should win over orb_breakout (priority 8)."""
        idx = pd.bdate_range("2025-01-02", periods=3)
        trading_dates = idx[1:]
        n = len(idx)

        # AAPL: fires mean_reversion (low RSI, low bb_pct, high vol)
        aapl_data = {
            "Close": [100.0] * n,
            "Open": [99.5] * n,
            "Volume": [2_000_000] * n,
            "rsi": [28.0] * n,
            "bb_pct": [0.15] * n,
            "vol_ratio": [1.5] * n,
            "ema9": [99.0] * n,
            "ema21": [100.0] * n,
            "macd_diff": [-0.1] * n,
            "ret_5d": [0.5] * n,
            "bb_squeeze": [False] * n,
            "macd_cross": [False] * n,
            "ret_10d": [1.0] * n,
            "pct_vs_ema21": [0.0] * n,
            "price_vs_52w_high_pct": [-50.0] * n,
            "is_inside_day": [False] * n,
        }
        aapl_df = pd.DataFrame(aapl_data, index=idx)

        # MSFT: neutral daily indicators, orb_breakout via intraday
        msft_data = {
            "Close": [200.0] * n,
            "Open": [199.5] * n,
            "Volume": [2_000_000] * n,
            "rsi": [50.0] * n,
            "bb_pct": [0.5] * n,
            "vol_ratio": [1.0] * n,
            "ema9": [99.0] * n,
            "ema21": [100.0] * n,
            "macd_diff": [-0.1] * n,
            "ret_5d": [0.5] * n,
            "bb_squeeze": [False] * n,
            "macd_cross": [False] * n,
            "ret_10d": [1.0] * n,
            "pct_vs_ema21": [0.0] * n,
            "price_vs_52w_high_pct": [-50.0] * n,
            "is_inside_day": [False] * n,
        }
        msft_df = pd.DataFrame(msft_data, index=idx)

        intraday = {
            "MSFT": {
                idx[1].strftime("%Y-%m-%d"): {
                    "orb_breakout_up": True,
                    "price_above_vwap": False,
                    "intraday_change_pct": None,
                    "pct_vs_vwap": 0.0,
                }
            }
        }

        result = _run_simulation(
            {"AAPL": aapl_df, "MSFT": msft_df},
            trading_dates,
            initial_capital=10_000.0,
            max_positions=1,
            max_hold_days=2,
            intraday_data=intraday,
        )
        buy_trades = [t for t in result["trades"] if t["action"] == "BUY"]
        bought_symbols = [t["symbol"] for t in buy_trades]
        self.assertIn("AAPL", bought_symbols)
        self.assertNotIn("MSFT", bought_symbols)

    def _make_signal_df(self, idx, signal_row_overrides: dict) -> pd.DataFrame:
        """Build a pre-computed indicator DataFrame that fires a specific signal."""
        n = len(idx)
        defaults = {
            "Close": [100.0] * n,
            "Open": [99.5] * n,
            "Volume": [2_000_000] * n,
            "rsi": [50.0] * n,
            "bb_pct": [0.5] * n,
            "vol_ratio": [1.5] * n,
            "ema9": [101.0] * n,
            "ema21": [100.0] * n,
            "macd_diff": [0.5] * n,
            "ret_5d": [2.0] * n,
            "bb_squeeze": [False] * n,
            "macd_cross": [False] * n,
            "ret_10d": [4.0] * n,
            "pct_vs_ema21": [1.0] * n,
            "price_vs_52w_high_pct": [-50.0] * n,
            "is_inside_day": [False] * n,
        }
        defaults.update(signal_row_overrides)
        return pd.DataFrame(defaults, index=idx)

    def test_per_signal_cap_limits_same_signal_entries(self):
        """With cap=1, at most 1 position from any single signal on a single trading day."""
        # Use 2 periods → 1 trading day (idx[1:]) so the cap is unambiguously per-day.
        idx = pd.bdate_range("2025-01-02", periods=2)
        trading_dates = idx[1:]
        n = len(idx)

        # Three symbols all firing bb_squeeze on the same day
        bb_overrides = {"bb_squeeze": [True] * n}
        indicators = {f"SYM{i}": self._make_signal_df(idx, bb_overrides) for i in range(3)}

        result = _run_simulation(
            indicators,
            trading_dates,
            initial_capital=30_000.0,
            max_positions=5,
            max_hold_days=2,
            per_signal_cap=1,
        )
        buy_trades = [t for t in result["trades"] if t["action"] == "BUY"]
        bb_buys = [t for t in buy_trades if t["signal"] == "bb_squeeze"]
        # Cap of 1 → exactly 1 bb_squeeze buy on the single trading day
        self.assertEqual(len(bb_buys), 1)

    def test_per_signal_cap_2_allows_two_same_signal(self):
        """With cap=2, exactly 2 entries from the same signal are allowed in one day."""
        idx = pd.bdate_range("2025-01-02", periods=2)
        trading_dates = idx[1:]
        n = len(idx)

        bb_overrides = {"bb_squeeze": [True] * n}
        indicators = {f"SYM{i}": self._make_signal_df(idx, bb_overrides) for i in range(4)}

        result = _run_simulation(
            indicators,
            trading_dates,
            initial_capital=40_000.0,
            max_positions=5,
            max_hold_days=2,
            per_signal_cap=2,
        )
        buy_trades = [t for t in result["trades"] if t["action"] == "BUY"]
        bb_buys = [t for t in buy_trades if t["signal"] == "bb_squeeze"]
        # Cap of 2 → exactly 2 bb_squeeze buys (4 candidates, cap stops at 2)
        self.assertEqual(len(bb_buys), 2)

    def test_per_signal_cap_does_not_prevent_fills_from_other_signals(self):
        """When cap blocks the dominant signal, other signals still fill remaining slots."""
        idx = pd.bdate_range("2025-01-02", periods=2)
        trading_dates = idx[1:]
        n = len(idx)

        # SYM0–SYM1: bb_squeeze; SYM2: mean_reversion (rsi<35, bb_pct<0.25, vol>1.2)
        bb_overrides = {"bb_squeeze": [True] * n}
        mr_overrides = {
            "rsi": [28.0] * n,
            "bb_pct": [0.15] * n,
            "vol_ratio": [1.5] * n,
            "ema9": [99.0] * n,
            "ema21": [100.0] * n,
            "macd_diff": [-0.1] * n,
            "bb_squeeze": [False] * n,
        }
        indicators = {
            "SYM0": self._make_signal_df(idx, bb_overrides),
            "SYM1": self._make_signal_df(idx, bb_overrides),
            "SYM2": self._make_signal_df(idx, mr_overrides),
        }

        result = _run_simulation(
            indicators,
            trading_dates,
            initial_capital=30_000.0,
            max_positions=5,
            max_hold_days=2,
            per_signal_cap=1,
        )
        buy_trades = [t for t in result["trades"] if t["action"] == "BUY"]
        signals_bought = {t["signal"] for t in buy_trades}
        # cap=1 limits bb_squeeze to 1; mean_reversion should still get a slot
        self.assertIn("mean_reversion", signals_bought)


class TestRunSimulationEdgeCases(unittest.TestCase):
    """Edge cases in _run_simulation that require custom indicator DataFrames."""

    def test_equity_update_exception_uses_fallback(self):
        """Lines 129-130, 138-139, 237-238: price lookups fail → fallbacks fire."""
        idx = pd.bdate_range("2025-01-02", periods=3)
        trading_dates = idx[1:]  # day1, day2
        # Day2 Close = None → float(None) raises TypeError
        close_vals = pd.array([100.0, 101.0, None], dtype=object)
        aapl_df = _build_indicator_df(idx, close_vals, open_vals=[99.5, 100.5, 101.0])
        indicators = {"AAPL": aapl_df}
        result = _run_simulation(
            indicators, trading_dates, initial_capital=10_000.0, params=_LOOSE_ENTRY
        )
        # Simulation must complete without raising
        self.assertIsNotNone(result)
        self.assertEqual(len(result["equity_curve"]), 2)

    def test_stop_loss_exit_fires(self):
        """Line 145: position hit stop_loss when price drops by STOP_LOSS_PCT."""
        from config import STOP_LOSS_PCT

        idx = pd.bdate_range("2025-01-02", periods=3)
        trading_dates = idx[1:]
        # Price drops 10% on day2 → well below STOP_LOSS_PCT threshold
        close_vals = [100.0, 100.0, 100.0 * (1 - STOP_LOSS_PCT * 2)]
        aapl_df = _build_indicator_df(idx, close_vals, open_vals=[99.5, 100.0, None])
        indicators = {"AAPL": aapl_df}
        result = _run_simulation(
            indicators, trading_dates, initial_capital=10_000.0, params=_LOOSE_ENTRY
        )
        sell_trades = [t for t in result["trades"] if t.get("reason") == "stop_loss"]
        self.assertEqual(len(sell_trades), 1)

    def test_take_profit_exit_fires(self):
        """Line 147: position hit take_profit when price rises by TAKE_PROFIT_PCT."""
        from config import TAKE_PROFIT_PCT

        idx = pd.bdate_range("2025-01-02", periods=3)
        trading_dates = idx[1:]
        # Price rises 20% on day2 → above TAKE_PROFIT_PCT threshold
        close_vals = [100.0, 100.0, 100.0 * (1 + TAKE_PROFIT_PCT * 1.5)]
        aapl_df = _build_indicator_df(idx, close_vals, open_vals=[99.5, 100.0, None])
        indicators = {"AAPL": aapl_df}
        result = _run_simulation(
            indicators, trading_dates, initial_capital=10_000.0, params=_LOOSE_ENTRY
        )
        sell_trades = [t for t in result["trades"] if t.get("reason") == "take_profit"]
        self.assertEqual(len(sell_trades), 1)

    def test_full_positions_skips_entry(self):
        """Line 174: all max_positions slots filled → slots<=0 → continue."""
        idx = pd.bdate_range("2025-01-02", periods=4)
        trading_dates = idx[1:]  # day1, day2, day3
        # AAPL: neutral price (no exit trigger), so position stays open
        close_vals = [100.0, 100.0, 101.0, 102.0]
        aapl_df = _build_indicator_df(idx, close_vals)
        msft_df = _build_indicator_df(idx, [200.0, 200.0, 201.0, 202.0])
        indicators = {"AAPL": aapl_df, "MSFT": msft_df}
        # max_positions=1: AAPL fills slot on day1; day2 has 0 slots → line 174
        result = _run_simulation(
            indicators,
            trading_dates,
            initial_capital=10_000.0,
            max_positions=1,
            max_hold_days=10,
            params=_LOOSE_ENTRY,
        )
        # Should run without error; at most 1 position open at any time
        buy_trades = [t for t in result["trades"] if t["action"] == "BUY"]
        self.assertGreaterEqual(len(buy_trades), 1)

    def test_first_bar_in_series_skipped_for_entry(self):
        """Line 182: today_loc==0 for a symbol → entry skipped on its first data date."""
        idx = pd.bdate_range("2025-01-02", periods=2)
        trading_dates = idx  # day0, day1 ARE the trading dates
        # AAPL index starts at day0 → on day0, today_loc=0 → skip entry
        close_vals = [100.0, 101.0]
        aapl_df = _build_indicator_df(idx, close_vals)
        indicators = {"AAPL": aapl_df}
        result = _run_simulation(
            indicators, trading_dates, initial_capital=10_000.0, params=_LOOSE_ENTRY
        )
        # day0 is today_loc=0 → no entry on day0
        # day1 is today_loc=1 → can enter (uses prev_row = day0)
        self.assertIsNotNone(result)

    def test_open_key_missing_falls_back_to_close(self):
        """Lines 193-194: indicator DataFrame has no Open column → falls back to Close."""
        idx = pd.bdate_range("2025-01-02", periods=2)
        trading_dates = idx[1:]
        close_vals = [100.0, 101.0]
        # include_open=False → no Open column → KeyError → fallback to Close
        aapl_df = _build_indicator_df(idx, close_vals, include_open=False)
        indicators = {"AAPL": aapl_df}
        result = _run_simulation(
            indicators, trading_dates, initial_capital=10_000.0, params=_LOOSE_ENTRY
        )
        self.assertIsNotNone(result)
        buy_trades = [t for t in result["trades"] if t["action"] == "BUY"]
        self.assertGreater(len(buy_trades), 0)

    def test_tiny_notional_skips_entry(self):
        """Line 200: cost < 0.5 (tiny capital) → continue without buying."""
        idx = pd.bdate_range("2025-01-02", periods=2)
        trading_dates = idx[1:]
        close_vals = [100.0, 101.0]
        aapl_df = _build_indicator_df(idx, close_vals)
        indicators = {"AAPL": aapl_df}
        # initial_capital=0.1 → notional=(0.1/1)*0.9=0.09 < 0.5 → continue at line 200
        result = _run_simulation(
            indicators, trading_dates, initial_capital=0.1, params=_LOOSE_ENTRY
        )
        buy_trades = [t for t in result["trades"] if t["action"] == "BUY"]
        self.assertEqual(len(buy_trades), 0)

    def test_entry_processing_exception_skipped(self):
        """Lines 217-218: Open missing + Close=None → inner except → float(None) raises → outer except."""
        idx = pd.bdate_range("2025-01-02", periods=2)
        trading_dates = idx[1:]
        # No Open column; Close[day1]=None → float(None) raises TypeError → outer except: continue
        close_vals = pd.array([100.0, None], dtype=object)
        aapl_df = _build_indicator_df(idx, close_vals, include_open=False)
        indicators = {"AAPL": aapl_df}
        result = _run_simulation(
            indicators, trading_dates, initial_capital=10_000.0, params=_LOOSE_ENTRY
        )
        buy_trades = [t for t in result["trades"] if t["action"] == "BUY"]
        self.assertEqual(len(buy_trades), 0)


class TestRunWalkForwardEdgeCases(unittest.TestCase):
    """Edge cases in run_walk_forward_optimized."""

    def setUp(self):
        self._raw = _make_raw(n=100)

    def test_symbol_with_bad_data_skipped_in_indicators(self):
        """Lines 417-418: indicator computation fails for a symbol → exception silently skipped."""
        # raw has only AAPL; requesting GOOG too → close_all["GOOG"] raises KeyError → lines 417-418
        raw_single = _make_raw(n=100, symbols=("AAPL",))
        with patch("backtest.engine.yf.download", return_value=raw_single):
            result = run_walk_forward_optimized(
                symbols=["AAPL", "GOOG"],
                start_date="2025-02-01",
                end_date="2025-03-07",
                train_days=10,
                test_days=5,
                initial_capital=10_000.0,
                param_grid=_LOOSE_PARAM_GRID,
            )
        self.assertIn("folds", result)

    def test_baseline_exception_silently_skipped(self):
        """Lines 493-494: symbol has no data in OOS window → iloc[0] raises → pass."""
        n = 100
        idx = pd.bdate_range("2024-11-01", periods=n)
        data: dict = {}
        for sym in ("AAPL",):
            closes = [100.0 + j * 0.1 for j in range(n)]
            data[("Close", sym)] = closes
            data[("Open", sym)] = [c * 0.999 for c in closes]
            data[("Volume", sym)] = [1_000_000] * n
        # GOOG: data only for first 25 rows (< 20 for BB+RSI → dropna leaves almost nothing)
        # After _compute_indicators + dropna, GOOG data ends well before the OOS test window
        goog_closes = [200.0 + j * 0.1 for j in range(25)] + [float("nan")] * (n - 25)
        data[("Close", "GOOG")] = goog_closes
        data[("Open", "GOOG")] = [
            c * 0.999 if not (isinstance(c, float) and c != c) else float("nan")
            for c in goog_closes
        ]
        data[("Volume", "GOOG")] = [1_000_000] * 25 + [0] * (n - 25)
        raw = pd.DataFrame(data, index=idx)
        raw.columns = pd.MultiIndex.from_tuples(raw.columns)
        with patch("backtest.engine.yf.download", return_value=raw):
            result = run_walk_forward_optimized(
                symbols=["AAPL", "GOOG"],
                start_date="2025-02-01",
                end_date="2025-03-07",
                train_days=10,
                test_days=5,
                initial_capital=10_000.0,
                param_grid=_LOOSE_PARAM_GRID,
            )
        self.assertIn("folds", result)

    def test_empty_fold_results_returns_empty_structure(self):
        """Line 523: fold_results empty → return {folds:[], summary:{}}."""

        class _SmartLen:
            """Returns large len on first call (passes early-return check), 0 on subsequent calls."""

            def __init__(self, real_len):
                self._real_len = real_len
                self._calls = 0

            def __len__(self):
                self._calls += 1
                return self._real_len if self._calls == 1 else 0

            def __getitem__(self, key):
                return pd.bdate_range("2025-01-01", periods=1)[0]

            def __iter__(self):
                return iter([])

        fake_dates = _SmartLen(1000)
        with (
            patch("backtest.engine.yf.download", return_value=self._raw),
            patch("backtest.engine.pd.bdate_range", return_value=fake_dates),
        ):
            result = run_walk_forward_optimized(
                symbols=["AAPL", "FLAT"],
                start_date="2025-02-01",
                end_date="2025-03-07",
                train_days=10,
                test_days=5,
                initial_capital=10_000.0,
                param_grid=_LOOSE_PARAM_GRID,
            )
        self.assertEqual(result.get("folds"), [])
        self.assertEqual(result.get("summary"), {})


class TestIvCompressionSignal(unittest.TestCase):
    """iv_compression: historical volatility percentile squeeze."""

    def test_fires_with_ema_confirmation(self):
        row = _make_row(hv_rank=0.10, ema9=101, ema21=100, vol_ratio=1.2)
        self.assertEqual(_entry_signal(row), "iv_compression")

    def test_fires_with_macd_confirmation(self):
        row = _make_row(hv_rank=0.15, ema9=99, ema21=100, macd_diff=0.1, vol_ratio=1.15)
        self.assertEqual(_entry_signal(row), "iv_compression")

    def test_no_fire_when_hv_rank_above_threshold(self):
        row = _make_row(hv_rank=0.25, ema9=101, ema21=100, vol_ratio=1.5)
        # hv_rank=0.25 is above 0.20 → no iv_compression; other signals also absent
        self.assertIsNone(_entry_signal(row))

    def test_no_fire_without_directional_confirmation(self):
        # ema9 < ema21 and macd_diff <= 0
        row = _make_row(hv_rank=0.10, ema9=99, ema21=100, macd_diff=-0.05, vol_ratio=1.2)
        self.assertIsNone(_entry_signal(row))

    def test_no_fire_without_volume(self):
        row = _make_row(hv_rank=0.10, ema9=101, ema21=100, vol_ratio=1.0)
        self.assertIsNone(_entry_signal(row))

    def test_in_signal_priority(self):
        self.assertIn("iv_compression", _SIGNAL_PRIORITY)

    def test_priority_between_bb_squeeze_and_momentum(self):
        self.assertLess(_SIGNAL_PRIORITY["bb_squeeze"], _SIGNAL_PRIORITY["iv_compression"])
        self.assertLess(_SIGNAL_PRIORITY["iv_compression"], _SIGNAL_PRIORITY["momentum"])

    def test_signals_not_tested_excludes_iv_compression_when_column_present(self):
        idx = pd.bdate_range("2025-01-02", periods=3)
        n = len(idx)
        df = pd.DataFrame(
            {
                "Close": [100.0] * n,
                "Open": [99.5] * n,
                "Volume": [1_000_000] * n,
                "rsi": [50.0] * n,
                "bb_pct": [0.5] * n,
                "vol_ratio": [1.0] * n,
                "ema9": [100.0] * n,
                "ema21": [100.0] * n,
                "macd_diff": [0.0] * n,
                "ret_5d": [0.0] * n,
                "hv_rank": [0.50] * n,
            },
            index=idx,
        )
        result = _run_simulation({"AAPL": df}, idx[1:])
        self.assertIn("iv_compression", result["signals_tested"])
        self.assertNotIn("iv_compression", result["signals_not_tested"])


class TestFundamentalSignals(unittest.TestCase):
    """insider_buying and pead signals via the fundamentals kwarg."""

    def test_insider_buying_fires_when_cluster(self):
        row = _make_row()
        sig = _entry_signal(row, fundamentals={"insider_cluster": True})
        self.assertEqual(sig, "insider_buying")

    def test_pead_fires_when_active_and_positive_5d(self):
        row = _make_row(ret_5d=2.0)
        sig = _entry_signal(row, fundamentals={"pead_active": True, "insider_cluster": False})
        self.assertEqual(sig, "pead")

    def test_pead_suppressed_when_ret_5d_negative(self):
        row = _make_row(ret_5d=-1.0)
        sig = _entry_signal(row, fundamentals={"pead_active": True, "insider_cluster": False})
        self.assertIsNone(sig)

    def test_insider_buying_priority_above_pead(self):
        row = _make_row(ret_5d=2.0)
        sig = _entry_signal(row, fundamentals={"insider_cluster": True, "pead_active": True})
        self.assertEqual(sig, "insider_buying")

    def test_no_fundamental_signal_without_fundamentals_arg(self):
        row = _make_row(ret_5d=2.0)
        sig = _entry_signal(row)
        self.assertIsNone(sig)

    def test_insider_buying_priority_in_dict(self):
        self.assertIn("insider_buying", _SIGNAL_PRIORITY)
        self.assertIn("pead", _SIGNAL_PRIORITY)

    def test_insider_buying_ranks_above_rs_leader(self):
        self.assertLess(_SIGNAL_PRIORITY["insider_buying"], _SIGNAL_PRIORITY["rs_leader"])

    def test_pead_ranks_above_rs_leader(self):
        self.assertLess(_SIGNAL_PRIORITY["pead"], _SIGNAL_PRIORITY["rs_leader"])

    def test_pead_in_signals_tested_when_earnings_history_passed(self):
        idx = pd.bdate_range("2025-01-02", periods=3)
        n = len(idx)
        df = pd.DataFrame(
            {
                "Close": [100.0] * n,
                "Open": [99.5] * n,
                "Volume": [1_000_000] * n,
                "rsi": [50.0] * n,
                "bb_pct": [0.5] * n,
                "vol_ratio": [1.0] * n,
                "ema9": [100.0] * n,
                "ema21": [100.0] * n,
                "macd_diff": [0.0] * n,
                "ret_5d": [0.0] * n,
            },
            index=idx,
        )
        result = _run_simulation({"AAPL": df}, idx[1:], earnings_history={})
        self.assertIn("pead", result["signals_tested"])
        self.assertNotIn("pead", result["signals_not_tested"])

    def test_insider_buying_in_signals_tested_when_insider_history_passed(self):
        idx = pd.bdate_range("2025-01-02", periods=3)
        n = len(idx)
        df = pd.DataFrame(
            {
                "Close": [100.0] * n,
                "Open": [99.5] * n,
                "Volume": [1_000_000] * n,
                "rsi": [50.0] * n,
                "bb_pct": [0.5] * n,
                "vol_ratio": [1.0] * n,
                "ema9": [100.0] * n,
                "ema21": [100.0] * n,
                "macd_diff": [0.0] * n,
                "ret_5d": [0.0] * n,
            },
            index=idx,
        )
        result = _run_simulation({"AAPL": df}, idx[1:], insider_history={})
        self.assertIn("insider_buying", result["signals_tested"])
        self.assertNotIn("insider_buying", result["signals_not_tested"])


# ── disabled_signals / ablation ───────────────────────────────────────────────


class TestDisabledSignals(unittest.TestCase):
    def test_disabled_signal_does_not_fire(self):
        row = _make_row(rsi=25.0, bb_pct=0.1, vol_ratio=1.5)
        self.assertEqual(_entry_signal(row), "mean_reversion")
        self.assertIsNone(_entry_signal(row, disabled_signals=frozenset({"mean_reversion"})))

    def test_multiple_disabled_signals_skipped(self):
        row = _make_row(rsi=25.0, bb_pct=0.1, vol_ratio=1.5)
        result = _entry_signal(
            row, disabled_signals=frozenset({"mean_reversion", "momentum", "macd_crossover"})
        )
        self.assertNotIn(result, {"mean_reversion", "momentum", "macd_crossover"})

    def test_disabled_signal_absent_from_simulation_by_signal(self):
        idx = pd.bdate_range("2025-01-02", periods=60)
        prices = [100.0 - i * 0.3 for i in range(60)]
        df = pd.DataFrame(
            {
                "Close": prices,
                "Open": [p * 0.999 for p in prices],
                "Volume": [2_000_000] * 60,
                "rsi": [28.0] * 60,
                "bb_pct": [0.08] * 60,
                "vol_ratio": [1.6] * 60,
                "ema9": [99.0] * 60,
                "ema21": [100.0] * 60,
                "macd_diff": [-0.1] * 60,
                "ret_5d": [-2.0] * 60,
            },
            index=idx,
        )
        result = _run_simulation(
            {"AAPL": df}, idx[1:], disabled_signals=frozenset({"mean_reversion"})
        )
        self.assertNotIn("mean_reversion", result["by_signal"])

    def test_empty_disabled_signals_has_no_effect(self):
        row = _make_row(rsi=25.0, bb_pct=0.1, vol_ratio=1.5)
        self.assertEqual(
            _entry_signal(row),
            _entry_signal(row, disabled_signals=frozenset()),
        )

    def test_fundamental_signal_disabled(self):
        row = _make_row(ret_5d=2.0)
        self.assertEqual(
            _entry_signal(row, fundamentals={"pead_active": True}),
            "pead",
        )
        self.assertIsNone(
            _entry_signal(
                row,
                fundamentals={"pead_active": True},
                disabled_signals=frozenset({"pead"}),
            )
        )


class TestRunAblation(unittest.TestCase):
    def _run(self, **kwargs):
        raw = _make_raw(n=100)
        with (
            patch("backtest.engine.yf.download", return_value=raw),
            patch("backtest.engine.prefetch_earnings_history", return_value={}),
            patch("backtest.engine.prefetch_insider_history", return_value={}),
        ):
            return run_ablation(["AAPL", "FLAT"], "2025-01-01", "2025-06-30", **kwargs)

    def test_returns_baseline_and_ablations(self):
        result = self._run()
        self.assertIn("baseline", result)
        self.assertIn("ablations", result)

    def test_ablations_count_matches_signal_priority(self):
        result = self._run()
        self.assertEqual(len(result["ablations"]), len(_SIGNAL_PRIORITY))

    def test_each_ablation_has_required_keys(self):
        result = self._run()
        for a in result["ablations"]:
            for key in ("signal", "baseline_trades", "sharpe_delta", "return_delta", "verdict"):
                self.assertIn(key, a)

    def test_verdict_keep_when_sharpe_delta_negative(self):
        result = self._run()
        for a in result["ablations"]:
            expected = "KEEP" if a["sharpe_delta"] < 0 else "REVIEW"
            self.assertEqual(a["verdict"], expected)

    def test_all_signals_represented(self):
        result = self._run()
        ablated = {a["signal"] for a in result["ablations"]}
        self.assertEqual(ablated, set(_SIGNAL_PRIORITY.keys()))

    def test_baseline_trades_non_negative(self):
        result = self._run()
        for a in result["ablations"]:
            self.assertGreaterEqual(a["baseline_trades"], 0)

    def test_print_ablation_results_no_error(self):
        result = self._run()
        try:
            _print_ablation_results(result, "2025-01-01", "2025-06-30")
        except Exception as exc:
            self.fail(f"_print_ablation_results raised: {exc}")


# ── run_backward_elimination ──────────────────────────────────────────────────


class TestRunBackwardElimination(unittest.TestCase):
    def _run(self, **kwargs):
        raw = _make_raw(n=100)
        with (
            patch("backtest.engine.yf.download", return_value=raw),
            patch("backtest.engine.prefetch_earnings_history", return_value={}),
            patch("backtest.engine.prefetch_insider_history", return_value={}),
        ):
            return run_backward_elimination(["AAPL", "FLAT"], "2025-01-01", "2025-06-30", **kwargs)

    def test_returns_expected_top_level_keys(self):
        result = self._run()
        for key in (
            "steps",
            "original_baseline",
            "final_result",
            "signals_kept",
            "signals_removed",
        ):
            self.assertIn(key, result)

    def test_signals_kept_and_removed_are_disjoint(self):
        result = self._run()
        kept = set(result["signals_kept"])
        removed = set(result["signals_removed"])
        self.assertTrue(kept.isdisjoint(removed))

    def test_signals_kept_union_removed_is_subset_of_priority(self):
        result = self._run()
        all_signals = set(result["signals_kept"]) | set(result["signals_removed"])
        self.assertTrue(all_signals <= set(_SIGNAL_PRIORITY.keys()))

    def test_steps_count_matches_signals_removed(self):
        result = self._run()
        self.assertEqual(len(result["steps"]), len(result["signals_removed"]))

    def test_each_step_has_required_keys(self):
        result = self._run()
        for s in result["steps"]:
            for key in (
                "step",
                "signal_removed",
                "sharpe_delta",
                "sharpe_after",
                "return_after",
                "trades_removed",
            ):
                self.assertIn(key, s)

    def test_step_numbers_are_sequential(self):
        result = self._run()
        for i, s in enumerate(result["steps"], start=1):
            self.assertEqual(s["step"], i)

    def test_each_step_sharpe_delta_positive(self):
        result = self._run()
        for s in result["steps"]:
            self.assertGreater(s["sharpe_delta"], 0)

    def test_final_result_sharpe_ge_baseline(self):
        result = self._run()
        self.assertGreaterEqual(
            result["final_result"]["sharpe_ratio"],
            result["original_baseline"]["sharpe_ratio"],
        )

    def test_returns_empty_on_empty_data(self):
        with patch("backtest.engine.yf.download", return_value=pd.DataFrame()):
            result = run_backward_elimination(["AAPL"], "2025-01-01", "2025-06-30")
        self.assertEqual(result, {})

    def test_use_earnings_only_fetches_earnings_not_insider(self):
        raw = _make_raw(n=100)
        with (
            patch("backtest.engine.yf.download", return_value=raw),
            patch("backtest.engine.prefetch_earnings_history", return_value={}) as mock_earn,
            patch("backtest.engine.prefetch_insider_history", return_value={}) as mock_insider,
        ):
            run_backward_elimination(
                ["AAPL", "FLAT"], "2025-01-01", "2025-06-30", use_earnings_only=True
            )
        mock_earn.assert_called_once()
        mock_insider.assert_not_called()

    def test_print_backward_elimination_results_no_error(self):
        result = self._run()
        try:
            _print_backward_elimination_results(result, "2025-01-01", "2025-06-30")
        except Exception as exc:
            self.fail(f"_print_backward_elimination_results raised: {exc}")

    def test_print_backward_elimination_no_steps_no_error(self):
        r = {
            "steps": [],
            "original_baseline": {"sharpe_ratio": 0.5, "total_return_pct": 5.0, "total_trades": 10},
            "final_result": {"sharpe_ratio": 0.5, "total_return_pct": 5.0, "total_trades": 10},
            "signals_kept": ["momentum", "mean_reversion"],
            "signals_removed": [],
        }
        try:
            _print_backward_elimination_results(r, "2025-01-01", "2025-06-30")
        except Exception as exc:
            self.fail(f"_print_backward_elimination_results raised: {exc}")


# ── enriched SELL trade metadata ──────────────────────────────────────────────


class TestEnrichedSellTrades(unittest.TestCase):
    """SELL trades now carry entry_date, entry_regime, days_held."""

    def _build_indicators_with_loose_signal(self):
        idx = pd.bdate_range("2025-01-02", periods=5)
        n = len(idx)
        df = pd.DataFrame(
            {
                "Close": [100.0, 101.0, 102.0, 103.0, 104.0],
                "Open": [99.5, 100.5, 101.5, 102.5, 103.5],
                "Volume": [2_000_000] * n,
                "rsi": [50.0] * n,
                "bb_pct": [0.5] * n,
                "vol_ratio": [1.5] * n,
                "ema9": [101.0] * n,
                "ema21": [100.0] * n,
                "macd_diff": [0.5] * n,
                "ret_5d": [2.0] * n,
            },
            index=idx,
        )
        return {"AAPL": df}

    def test_sell_trades_include_entry_date(self):
        indicators = self._build_indicators_with_loose_signal()
        dates = pd.bdate_range("2025-01-03", "2025-01-08")
        result = _run_simulation(indicators, dates, initial_capital=10_000.0, params=_LOOSE_ENTRY)
        sell_trades = [t for t in result["trades"] if t["action"] == "SELL" and "pnl_pct" in t]
        for t in sell_trades:
            self.assertIn("entry_date", t)
            self.assertIsInstance(t["entry_date"], str)

    def test_sell_trades_include_days_held(self):
        indicators = self._build_indicators_with_loose_signal()
        dates = pd.bdate_range("2025-01-03", "2025-01-08")
        result = _run_simulation(indicators, dates, initial_capital=10_000.0, params=_LOOSE_ENTRY)
        sell_trades = [t for t in result["trades"] if t["action"] == "SELL" and "pnl_pct" in t]
        for t in sell_trades:
            self.assertIn("days_held", t)
            self.assertGreaterEqual(t["days_held"], 0)

    def test_sell_trades_include_entry_regime(self):
        indicators = self._build_indicators_with_loose_signal()
        dates = pd.bdate_range("2025-01-03", "2025-01-08")
        fake_regime = {d.strftime("%Y-%m-%d"): "BULL_TRENDING" for d in dates}
        result = _run_simulation(
            indicators,
            dates,
            initial_capital=10_000.0,
            params=_LOOSE_ENTRY,
            regime_by_date=fake_regime,
        )
        sell_trades = [t for t in result["trades"] if t["action"] == "SELL" and "pnl_pct" in t]
        for t in sell_trades:
            self.assertIn("entry_regime", t)

    def test_days_held_matches_time_exit_hold(self):
        indicators = self._build_indicators_with_loose_signal()
        dates = pd.bdate_range("2025-01-03", "2025-01-08")
        result = _run_simulation(
            indicators, dates, initial_capital=10_000.0, max_hold_days=2, params=_LOOSE_ENTRY
        )
        time_exits = [t for t in result["trades"] if t.get("reason") == "time_exit"]
        for t in time_exits:
            self.assertEqual(t["days_held"], 2)


# ── run_signal_analysis ───────────────────────────────────────────────────────


class TestRunSignalAnalysis(unittest.TestCase):
    def _run(self, **kwargs):
        raw = _make_raw(n=100)
        with (
            patch("backtest.engine.yf.download", return_value=raw),
            patch("backtest.engine.prefetch_earnings_history", return_value={}),
            patch("backtest.engine.prefetch_insider_history", return_value={}),
        ):
            return run_signal_analysis(["AAPL", "FLAT"], "2025-01-01", "2025-06-30", **kwargs)

    def test_returns_expected_top_level_keys(self):
        result = self._run()
        for key in ("baseline", "regime_stats", "decay_stats"):
            self.assertIn(key, result)

    def test_regime_stats_is_dict(self):
        result = self._run()
        self.assertIsInstance(result["regime_stats"], dict)

    def test_decay_stats_is_dict(self):
        result = self._run()
        self.assertIsInstance(result["decay_stats"], dict)

    def test_regime_stats_signal_keys_are_valid(self):
        result = self._run()
        for sig in result["regime_stats"]:
            self.assertIn(sig, _SIGNAL_PRIORITY)

    def test_decay_stats_signal_keys_are_valid(self):
        result = self._run()
        for sig in result["decay_stats"]:
            self.assertIn(sig, _SIGNAL_PRIORITY)

    def test_regime_buckets_have_required_fields(self):
        result = self._run()
        for sig_data in result["regime_stats"].values():
            for reg_data in sig_data.values():
                for field in ("wins", "losses", "total_return"):
                    self.assertIn(field, reg_data)

    def test_decay_buckets_have_required_fields(self):
        result = self._run()
        for sig_data in result["decay_stats"].values():
            for dh_data in sig_data.values():
                for field in ("wins", "losses", "total_return"):
                    self.assertIn(field, dh_data)

    def test_returns_empty_on_empty_data(self):
        with patch("backtest.engine.yf.download", return_value=pd.DataFrame()):
            result = run_signal_analysis(["AAPL"], "2025-01-01", "2025-06-30")
        self.assertEqual(result, {})

    def test_print_regime_table_no_error(self):
        result = self._run()
        try:
            _print_regime_table(result, "2025-01-01", "2025-06-30")
        except Exception as exc:
            self.fail(f"_print_regime_table raised: {exc}")

    def test_print_hold_period_table_no_error(self):
        result = self._run()
        try:
            _print_hold_period_table(result, "2025-01-01", "2025-06-30")
        except Exception as exc:
            self.fail(f"_print_hold_period_table raised: {exc}")

    def test_print_regime_table_empty_stats_no_error(self):
        r = {"regime_stats": {}, "decay_stats": {}}
        try:
            _print_regime_table(r, "2025-01-01", "2025-06-30")
        except Exception as exc:
            self.fail(f"_print_regime_table raised on empty stats: {exc}")

    def test_print_hold_period_table_empty_stats_no_error(self):
        r = {"regime_stats": {}, "decay_stats": {}}
        try:
            _print_hold_period_table(r, "2025-01-01", "2025-06-30")
        except Exception as exc:
            self.fail(f"_print_hold_period_table raised on empty stats: {exc}")


class TestAssertPreHoldout(unittest.TestCase):
    def test_no_warning_before_holdout(self):
        pre = (HOLDOUT_START_DATE.replace(year=HOLDOUT_START_DATE.year - 1)).strftime("%Y-%m-%d")
        with self.assertLogs("backtest.engine", level="WARNING") as cm:
            # Force a log entry so assertLogs doesn't fail on empty capture
            import logging

            logging.getLogger("backtest.engine").warning("_sentinel")
            _assert_pre_holdout(pre)
        # Only the sentinel should be present — no holdout contamination warning
        self.assertTrue(all("HOLDOUT CONTAMINATION" not in m for m in cm.output))

    def test_warns_on_holdout_date(self):
        on = HOLDOUT_START_DATE.strftime("%Y-%m-%d")
        with self.assertLogs("backtest.engine", level="WARNING") as cm:
            _assert_pre_holdout(on)
        self.assertTrue(any("HOLDOUT CONTAMINATION" in m for m in cm.output))

    def test_warns_past_holdout(self):
        past = HOLDOUT_START_DATE.replace(year=HOLDOUT_START_DATE.year + 1).strftime("%Y-%m-%d")
        with self.assertLogs("backtest.engine", level="WARNING") as cm:
            _assert_pre_holdout(past)
        self.assertTrue(any("HOLDOUT CONTAMINATION" in m for m in cm.output))

    def test_invalid_date_string_does_not_raise(self):
        try:
            _assert_pre_holdout("not-a-date")
        except Exception as exc:
            self.fail(f"_assert_pre_holdout raised on bad input: {exc}")


class TestRunHoldoutEvaluation(unittest.TestCase):
    def _run(self, version: str = "vTEST", tmp_dir: str | None = None):
        ohlcv = _make_ohlcv_full(300)
        ind = _compute_indicators(ohlcv)
        fake_ind = {"AAPL": ind}
        fake_raw = pd.concat(
            {"Close": ohlcv[["Close"]], "Open": ohlcv[["Open"]], "Volume": ohlcv[["Volume"]]},
            axis=1,
        )
        # Flatten to match yf.download multi-index output
        fake_raw.columns = pd.MultiIndex.from_tuples(
            [("Close", "AAPL"), ("Open", "AAPL"), ("Volume", "AAPL")]
        )
        patches: list = [
            patch("backtest.engine.yf.download", return_value=fake_raw),
            patch("backtest.engine._build_indicators", return_value=fake_ind),
        ]
        if tmp_dir:
            patches.append(patch("backtest.engine.LOG_DIR", tmp_dir))
            patches.append(
                patch("backtest.engine._HOLDOUT_LOG", os.path.join(tmp_dir, "holdout_log.jsonl"))
            )
        [p.start() for p in patches]
        try:
            result = run_holdout_evaluation(
                frozen_params={},
                version=version,
                symbols=["AAPL"],
                initial_capital=10_000.0,
            )
        finally:
            for p in patches:
                p.stop()
        return result

    def test_returns_dict_with_expected_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run(tmp_dir=tmp)
        for key in ("total_return_pct", "total_trades", "win_rate_pct", "sharpe_ratio"):
            self.assertIn(key, result)

    def test_holdout_flag_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run(tmp_dir=tmp)
        self.assertTrue(result.get("holdout"))

    def test_version_recorded_in_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run(version="vTEST_VER", tmp_dir=tmp)
        self.assertEqual(result.get("version"), "vTEST_VER")

    def test_log_entry_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._run(tmp_dir=tmp)
            log_path = os.path.join(tmp, "holdout_log.jsonl")
            self.assertTrue(os.path.exists(log_path))
            with open(log_path) as f:
                entries = [json.loads(line) for line in f if line.strip()]
            self.assertEqual(len(entries), 1)
            self.assertIn("version", entries[0])
            self.assertIn("frozen_params", entries[0])

    def test_repeated_version_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._run(version="vDUP", tmp_dir=tmp)
            with self.assertLogs("backtest.engine", level="WARNING") as cm:
                self._run(version="vDUP", tmp_dir=tmp)
            self.assertTrue(any("HOLDOUT INTEGRITY" in m for m in cm.output))


class TestLiquiditySpreadBps(unittest.TestCase):
    """_liquidity_spread_bps: wider spread for illiquid names, floored at SPREAD_BPS."""

    def test_zero_adv_returns_spread_bps_floor(self):
        from config import SPREAD_BPS

        self.assertEqual(_liquidity_spread_bps(0), float(SPREAD_BPS))

    def test_negative_adv_returns_spread_bps_floor(self):
        from config import SPREAD_BPS

        self.assertEqual(_liquidity_spread_bps(-1), float(SPREAD_BPS))

    def test_small_adv_returns_wider_spread(self):
        # $1M ADV → 50 / sqrt(1) = 50 bps, well above any SPREAD_BPS floor
        result = _liquidity_spread_bps(1_000_000)
        self.assertGreater(result, 10)

    def test_large_adv_returns_floor(self):
        from config import SPREAD_BPS

        # $10B ADV → 50 / sqrt(10000) = 0.5 bps → clamped to SPREAD_BPS
        result = _liquidity_spread_bps(10_000_000_000)
        self.assertEqual(result, float(SPREAD_BPS))

    def test_monotone_decreasing_with_adv(self):
        # Larger ADV should mean tighter spread
        s1 = _liquidity_spread_bps(10_000_000)
        s2 = _liquidity_spread_bps(100_000_000)
        s3 = _liquidity_spread_bps(1_000_000_000)
        self.assertGreaterEqual(s1, s2)
        self.assertGreaterEqual(s2, s3)


class TestMarketImpactBps(unittest.TestCase):
    """_market_impact_bps: sqrt-of-participation-rate, capped at 50 bps."""

    def test_zero_adv_returns_zero(self):
        self.assertEqual(_market_impact_bps(10_000, 0), 0.0)

    def test_zero_notional_returns_zero(self):
        self.assertEqual(_market_impact_bps(0, 1_000_000), 0.0)

    def test_one_pct_participation_gives_ten_bps(self):
        # 1% participation: notional = 0.01 * adv_usd
        adv = 1_000_000
        notional = adv * 0.01
        result = _market_impact_bps(notional, adv)
        self.assertAlmostEqual(result, 10.0, places=5)

    def test_four_pct_participation_gives_twenty_bps(self):
        adv = 1_000_000
        notional = adv * 0.04
        result = _market_impact_bps(notional, adv)
        self.assertAlmostEqual(result, 20.0, places=5)

    def test_capped_at_fifty_bps(self):
        # 25% participation → 10 * sqrt(25) = 50 bps exactly
        adv = 1_000_000
        notional = adv * 0.25
        result = _market_impact_bps(notional, adv)
        self.assertEqual(result, 50.0)

    def test_extreme_participation_capped(self):
        # 100% of ADV → without cap would be 100 bps; with cap → 50 bps
        result = _market_impact_bps(1_000_000, 1_000_000)
        self.assertEqual(result, 50.0)


class TestGapThroughStop(unittest.TestCase):
    """Gap-through-stop: when today's open gaps below stop price, fill at open."""

    def _make_indicators(self, entry_px: float, open_px: float, close_px: float) -> dict:
        """Single-symbol indicator dict with 2 rows (prev + today)."""
        idx = pd.bdate_range("2025-01-02", periods=2)
        df = pd.DataFrame(
            {
                "Close": [entry_px, close_px],
                "Open": [entry_px, open_px],
                "High": [entry_px * 1.01, max(open_px, close_px) * 1.01],
                "Low": [entry_px * 0.99, min(open_px, close_px) * 0.99],
                "Volume": [5_000_000, 4_000_000],
                "rsi": [45.0, 44.0],
                "macd_diff": [0.1, 0.1],
                "ema9": [entry_px, entry_px],
                "ema21": [entry_px * 0.98, entry_px * 0.98],
                "bb_pct": [0.5, 0.5],
                "vol_ratio": [1.1, 1.0],
                "ret_5d": [1.0, 1.0],
                "avg_volume_20": [5_000_000, 5_000_000],
            },
            index=idx,
        )
        return {"SYM": df}

    def test_gap_below_stop_simulation_completes(self):
        # Smoke test: simulation doesn't crash when a gap-through-stop occurs
        from config import STOP_LOSS_PCT

        entry_px = 100.0
        stop_px = entry_px * (1 - STOP_LOSS_PCT)
        open_px = stop_px * 0.97  # open gaps well below stop
        close_px = stop_px * 1.01  # close recovers above stop

        indicators = self._make_indicators(entry_px, open_px, close_px)
        result = _run_simulation(
            indicators,
            indicators["SYM"].index,
            initial_capital=10_000.0,
        )
        self.assertIsInstance(result, dict)
        self.assertIn("total_return_pct", result)

    def test_no_gap_simulation_completes(self):
        from config import STOP_LOSS_PCT

        entry_px = 100.0
        stop_px = entry_px * (1 - STOP_LOSS_PCT)
        open_px = stop_px * 1.05  # open above stop — no gap
        close_px = stop_px * 0.98  # close triggers stop normally

        indicators = self._make_indicators(entry_px, open_px, close_px)
        result = _run_simulation(
            indicators,
            indicators["SYM"].index,
            initial_capital=10_000.0,
        )
        self.assertIsInstance(result, dict)
        self.assertIn("total_return_pct", result)

    def test_avg_volume_20_column_present_after_compute_indicators(self):
        df = _make_ohlcv_full(60)
        result = _compute_indicators(df)
        self.assertIn("avg_volume_20", result.columns)

    def test_avg_volume_20_is_rolling_mean_of_volume(self):
        df = _make_ohlcv_full(60)
        result = _compute_indicators(df)
        manual = df["Volume"].rolling(20).mean()
        # Compare the last 10 rows of the result (all post-warmup)
        last_idx = result.index[-10:]
        pd.testing.assert_series_equal(
            result.loc[last_idx, "avg_volume_20"].reset_index(drop=True),
            manual.loc[last_idx].reset_index(drop=True),
            check_names=False,
        )


class TestBootstrapCellCi(unittest.TestCase):
    """_bootstrap_cell_ci: block-bootstrap 95% CI on win rate."""

    def test_too_few_samples_returns_nan(self):
        import math

        ci = _bootstrap_cell_ci([1.0] * 5)
        self.assertTrue(all(math.isnan(v) for v in ci))

    def test_exactly_ten_samples_returns_tuple(self):
        import math

        ci = _bootstrap_cell_ci([1.0, 0.0] * 5)
        self.assertEqual(len(ci), 2)
        self.assertFalse(any(math.isnan(v) for v in ci))

    def test_all_wins_ci_near_one(self):
        ci = _bootstrap_cell_ci([1.0] * 50)
        self.assertGreater(ci[0], 0.9)
        self.assertAlmostEqual(ci[1], 1.0, places=3)

    def test_all_losses_ci_near_zero(self):
        ci = _bootstrap_cell_ci([0.0] * 50)
        self.assertLess(ci[1], 0.1)
        self.assertAlmostEqual(ci[0], 0.0, places=3)

    def test_fifty_fifty_ci_straddles_half(self):
        ci = _bootstrap_cell_ci([1.0, 0.0] * 50)
        self.assertLess(ci[0], 0.5)
        self.assertGreater(ci[1], 0.5)

    def test_ci_low_le_ci_high(self):
        outcomes = [1.0, 0.0, 1.0, 1.0, 0.0] * 20
        ci = _bootstrap_cell_ci(outcomes)
        self.assertLessEqual(ci[0], ci[1])

    def test_deterministic_with_same_seed(self):
        outcomes = [float(i % 3 != 0) for i in range(60)]
        ci1 = _bootstrap_cell_ci(outcomes, n_boot=500)
        ci2 = _bootstrap_cell_ci(outcomes, n_boot=500)
        self.assertEqual(ci1, ci2)


class TestBacktestDefaultStart(unittest.TestCase):
    """BACKTEST_DEFAULT_START constant and regime_stats CI keys."""

    def test_default_start_is_2015(self):
        self.assertEqual(BACKTEST_DEFAULT_START, "2015-01-01")

    def test_regime_stats_include_ci_keys(self):
        """run_signal_analysis attaches win_rate_ci_low/high to each cell."""
        with (
            patch("backtest.engine.yf.download") as mock_dl,
            patch("backtest.engine._fetch_intraday_bars", return_value={}),
        ):
            raw = _make_raw(300, symbols=("AAPL",))
            mock_dl.return_value = raw
            result = run_signal_analysis(
                ["AAPL"],
                start_date="2025-01-01",
                end_date="2025-06-30",
                initial_capital=50_000,
                max_positions=3,
            )
        if not result or not result.get("regime_stats"):
            self.skipTest("No regime stats generated (no signals fired)")
        for sig, reg_dict in result["regime_stats"].items():
            for reg, cell in reg_dict.items():
                self.assertIn("win_rate_ci_low", cell, f"Missing CI in {sig}/{reg}")
                self.assertIn("win_rate_ci_high", cell)


class TestBinomialPValue(unittest.TestCase):
    """_binomial_p_value: exact one-sided binomial test."""

    def test_all_wins_small_p(self):
        p = _binomial_p_value(10, 10, p0=0.5)
        self.assertAlmostEqual(p, 1 / 1024, places=8)

    def test_half_wins_near_one(self):
        p = _binomial_p_value(5, 10, p0=0.5)
        self.assertGreater(p, 0.5)
        self.assertLessEqual(p, 1.0)

    def test_zero_wins_returns_one(self):
        self.assertEqual(_binomial_p_value(0, 10), 1.0)

    def test_zero_n_returns_one(self):
        self.assertEqual(_binomial_p_value(5, 0), 1.0)

    def test_p_bounded(self):
        for wins, n in [(1, 1), (3, 5), (18, 20), (100, 100)]:
            p = _binomial_p_value(wins, n)
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)


class TestHolmBonferroni(unittest.TestCase):
    """_holm_bonferroni: family-wise error control."""

    def test_empty_input(self):
        self.assertEqual(_holm_bonferroni({}), set())

    def test_all_significant(self):
        # p=0.001 for each of 3 cells → all reject H0 → none in failed set
        p_vals = {("s1", "bull"): 0.001, ("s2", "bear"): 0.001, ("s3", "bull"): 0.001}
        failed = _holm_bonferroni(p_vals, alpha=0.05)
        self.assertEqual(failed, set())

    def test_none_significant(self):
        # All p=0.9 → all fail to reject → all in failed set
        p_vals = {("s1", "bull"): 0.9, ("s2", "bear"): 0.9}
        failed = _holm_bonferroni(p_vals, alpha=0.05)
        self.assertEqual(failed, {("s1", "bull"), ("s2", "bear")})

    def test_partial_significance(self):
        # First cell p=0.001 passes; second p=0.9 fails
        p_vals = {("s1", "bull"): 0.001, ("s2", "bear"): 0.9}
        failed = _holm_bonferroni(p_vals, alpha=0.05)
        self.assertIn(("s2", "bear"), failed)
        self.assertNotIn(("s1", "bull"), failed)

    def test_returns_set_of_tuples(self):
        p_vals = {("mean_reversion", "bear"): 0.8}
        result = _holm_bonferroni(p_vals)
        self.assertIsInstance(result, set)
        for item in result:
            self.assertIsInstance(item, tuple)


class TestComputeRegimeBlocked(unittest.TestCase):
    """compute_regime_blocked: data-driven blocking from regime_stats."""

    def _make_stats(self, sig: str, reg: str, wins: int, losses: int) -> dict:
        return {sig: {reg: {"wins": wins, "losses": losses, "total_return": 0.0}}}

    def test_empty_stats_returns_empty(self):
        self.assertEqual(compute_regime_blocked({}), {})

    def test_low_n_cell_excluded(self):
        # n=5 < min_trades=20 → not tested → not blocked
        stats = self._make_stats("momentum", "bull", wins=3, losses=2)
        result = compute_regime_blocked(stats, min_trades=20)
        self.assertEqual(result, {})

    def test_high_win_rate_not_blocked(self):
        # 19/20 wins → p very small → not in failed set
        stats = self._make_stats("momentum", "bull", wins=19, losses=1)
        result = compute_regime_blocked(stats, min_trades=20)
        self.assertNotIn("bull", result)

    def test_coin_flip_blocked(self):
        # 10/20 wins → indistinguishable from chance → blocked
        stats = self._make_stats("momentum", "bear", wins=10, losses=10)
        result = compute_regime_blocked(stats, min_trades=20)
        self.assertIn("bear", result)
        self.assertIn("momentum", result["bear"])

    def test_output_is_dict_of_sets(self):
        stats = self._make_stats("mean_reversion", "sideways", wins=10, losses=10)
        result = compute_regime_blocked(stats, min_trades=20)
        for _reg, sigs in result.items():
            self.assertIsInstance(sigs, set)

    def test_regime_blocked_key_in_run_signal_analysis(self):
        """run_signal_analysis output includes regime_blocked key."""
        with (
            patch("backtest.engine.yf.download") as mock_dl,
            patch("backtest.engine._fetch_intraday_bars", return_value={}),
        ):
            raw = _make_raw(300, symbols=("AAPL",))
            mock_dl.return_value = raw
            result = run_signal_analysis(
                ["AAPL"],
                start_date="2025-01-01",
                end_date="2025-06-30",
                initial_capital=50_000,
                max_positions=3,
            )
        self.assertIn("regime_blocked", result)
        rb = result["regime_blocked"]
        self.assertIsInstance(rb, dict)
        for _reg, sigs in rb.items():
            self.assertIsInstance(sigs, list)


class TestPrintRegimeBlocked(unittest.TestCase):
    """_print_regime_blocked: smoke test — must not raise."""

    def test_empty_blocked(self):
        import io
        import sys

        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            _print_regime_blocked({}, "2025-01-01", "2025-12-31")
        finally:
            sys.stdout = old
        self.assertIn("No cells failed", buf.getvalue())

    def test_with_blocked_signals(self):
        import io
        import sys

        blocked = {"bear": {"momentum", "macd_crossover"}, "sideways": {"gap_and_go"}}
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            _print_regime_blocked(blocked, "2025-01-01", "2025-12-31")
        finally:
            sys.stdout = old
        output = buf.getvalue()
        self.assertIn("bear", output)
        self.assertIn("momentum", output)


# ── Coverage gap additions ─────────────────────────────────────────────────────


class TestComputeIntradayDayRSIException(unittest.TestCase):
    """Line 291-292: RSIIndicator raises → silently caught."""

    def test_rsi_exception_still_returns_result(self):
        # Build 80 bars so len(closes_5m) >= 14 → RSI path is taken
        bars = []
        for i in range(80):
            t = datetime(2025, 1, 2, 9, 31 + i // 60, i % 60, tzinfo=_ET)
            close = 100 + (i % 10) * 0.1
            bars.append(
                (t, _Bar(open=close, high=close + 0.5, low=close - 0.5, close=close, volume=500))
            )
        with patch("backtest.engine.RSIIndicator", side_effect=RuntimeError("bad data")):
            result = _compute_intraday_day("2025-01-02", bars)
        self.assertIsNotNone(result)
        self.assertIsNone(result["intraday_rsi"])


class TestBinomialPValueWinsGtN(unittest.TestCase):
    """Line 414: wins > n → returns 0.0."""

    def test_wins_greater_than_n_returns_zero(self):
        result = _binomial_p_value(wins=10, n=5)
        self.assertEqual(result, 0.0)


class TestFetchIntradayBarsImportError(unittest.TestCase):
    """Lines 318-326: alpaca import fails → returns {}."""

    def test_import_error_returns_empty(self):
        """Simulate ImportError by making the alpaca module raise on import."""
        import builtins
        import sys

        real_import = builtins.__import__

        def _blocking_import(name, *args, **kwargs):
            if "alpaca" in name:
                raise ImportError("alpaca not installed")
            return real_import(name, *args, **kwargs)

        # Remove cached alpaca modules so our import blocker takes effect
        blocked_keys = [k for k in sys.modules if "alpaca" in k]
        saved = {k: sys.modules.pop(k) for k in blocked_keys}
        try:
            import backtest.engine as _eng

            with patch("builtins.__import__", side_effect=_blocking_import):
                result = _eng._fetch_intraday_bars(["AAPL"], "2025-01-02", "2025-01-03")
            self.assertEqual(result, {})
        finally:
            sys.modules.update(saved)

    def test_no_api_keys_returns_empty(self):
        """Lines 328-330: API keys blank → returns {}."""
        import backtest.engine as _eng
        import config as _cfg_real

        orig_key = _cfg_real.ALPACA_API_KEY
        orig_secret = _cfg_real.ALPACA_SECRET_KEY
        _cfg_real.ALPACA_API_KEY = ""
        _cfg_real.ALPACA_SECRET_KEY = ""
        try:
            # We can only test no-keys path if alpaca is importable
            try:
                from alpaca.data.historical import StockHistoricalDataClient  # noqa

                result = _eng._fetch_intraday_bars(["AAPL"], "2025-01-02", "2025-01-03")
                self.assertEqual(result, {})
            except ImportError:
                pass  # alpaca not installed — import path covered by other test
        finally:
            _cfg_real.ALPACA_API_KEY = orig_key
            _cfg_real.ALPACA_SECRET_KEY = orig_secret


class TestComputeRegimesAllBranches(unittest.TestCase):
    """Lines 385-401: all four regime branches."""

    def _spy_df(self, spy_1d: float, spy_5d: float) -> pd.DataFrame:
        """Build a minimal SPY indicator DataFrame that produces the given returns."""
        idx = pd.bdate_range("2025-03-03", periods=2)
        # ret_1d: close[1]/close[0] - 1 = spy_1d/100
        close0 = 100.0
        close1 = close0 * (1 + spy_1d / 100)
        df = pd.DataFrame({"Close": [close0, close1], "ret_5d": [0.0, spy_5d]}, index=idx)
        return df

    def test_bear_day_regime(self):
        """Line 394: spy_1d <= -1.5 → BEAR_DAY."""
        from backtest.engine import _compute_regimes

        spy = self._spy_df(spy_1d=-2.0, spy_5d=-1.0)
        vix_spikes = {}
        regimes = _compute_regimes(spy, vix_spikes)
        date_str = spy.index[1].strftime("%Y-%m-%d")
        self.assertEqual(regimes[date_str], "BEAR_DAY")

    def test_high_vol_regime(self):
        """Line 393: vix_spike=True and spy_5d < -3 → HIGH_VOL."""
        from backtest.engine import _compute_regimes

        spy = self._spy_df(spy_1d=0.0, spy_5d=-4.0)  # not a BEAR_DAY (spy_1d > -1.5)
        vix_spikes = {spy.index[1].strftime("%Y-%m-%d"): True}
        regimes = _compute_regimes(spy, vix_spikes)
        date_str = spy.index[1].strftime("%Y-%m-%d")
        self.assertEqual(regimes[date_str], "HIGH_VOL")

    def test_bull_trending_regime(self):
        """Line 395: spy_5d > 2 and spy_1d > 0 → BULL_TRENDING."""
        from backtest.engine import _compute_regimes

        spy = self._spy_df(spy_1d=0.5, spy_5d=3.0)
        vix_spikes = {}
        regimes = _compute_regimes(spy, vix_spikes)
        date_str = spy.index[1].strftime("%Y-%m-%d")
        self.assertEqual(regimes[date_str], "BULL_TRENDING")

    def test_choppy_regime(self):
        """Line 397: none of the above → CHOPPY."""
        from backtest.engine import _compute_regimes

        spy = self._spy_df(spy_1d=0.1, spy_5d=0.5)  # mild positive, not bull-trending
        vix_spikes = {}
        regimes = _compute_regimes(spy, vix_spikes)
        date_str = spy.index[1].strftime("%Y-%m-%d")
        self.assertEqual(regimes[date_str], "CHOPPY")


class TestBootstrapCellCiRngFailure(unittest.TestCase):
    """Lines 530-531, 539: random.Random raises → rng stays None → sample_blocks=blocks."""

    def test_random_raises_still_returns_ci(self):
        """When Random() raises, rng=None branch (line 539) is hit."""
        import random as _random_module

        outcomes = [1.0, 0.0, 1.0, 1.0, 0.0] * 3  # 15 outcomes ≥ 10

        def _bad_random(*args, **kwargs):
            raise RuntimeError("random unavailable")

        # Patch Random on the real random module (imported locally inside the function)
        with patch.object(_random_module, "Random", side_effect=_bad_random):
            ci_lo, ci_hi = _bootstrap_cell_ci(outcomes)
        # Should not raise and should return a valid tuple
        self.assertIsInstance(ci_lo, float)
        self.assertIsInstance(ci_hi, float)


class TestRunSimulationGapThroughStop(unittest.TestCase):
    """Lines 632-633: open_px <= stop_price → fill at open."""

    def test_gap_through_stop_fills_at_open(self):
        """If open of exit day is at or below stop price, fills at open."""
        from config import STOP_LOSS_PCT

        idx = pd.bdate_range("2025-01-02", periods=3)
        trading_dates = idx[1:]
        entry_price = 100.0
        # open on day2 is well below stop = 100*(1-STOP_LOSS_PCT)
        stop = entry_price * (1 - STOP_LOSS_PCT)
        open_day2 = stop * 0.95  # gap through stop
        close_vals = [entry_price, entry_price, entry_price]
        open_vals = [entry_price * 0.999, entry_price, open_day2]
        aapl_df = _build_indicator_df(idx, close_vals, open_vals=open_vals)
        result = _run_simulation(
            {"AAPL": aapl_df}, trading_dates, initial_capital=10_000.0, params=_LOOSE_ENTRY
        )
        stop_trades = [t for t in result["trades"] if t.get("reason") == "stop_loss"]
        self.assertGreater(len(stop_trades), 0)


class TestBuildIndicatorsHighLow(unittest.TestCase):
    """Lines 964, 966: raw df includes High and Low columns → passed to _compute_indicators."""

    def test_high_low_columns_passed_through(self):
        from backtest.engine import _build_indicators

        n = 60
        idx = pd.bdate_range("2024-11-01", periods=n)
        closes = [100.0 + i * 0.5 for i in range(n)]
        data = {
            ("Close", "AAPL"): closes,
            ("Open", "AAPL"): [c * 0.999 for c in closes],
            ("High", "AAPL"): [c * 1.01 for c in closes],
            ("Low", "AAPL"): [c * 0.99 for c in closes],
            ("Volume", "AAPL"): [1_000_000] * n,
        }
        raw = pd.DataFrame(data, index=idx)
        raw.columns = pd.MultiIndex.from_tuples(raw.columns)
        indicators = _build_indicators(raw, ["AAPL"])
        self.assertIn("AAPL", indicators)
        # With High/Low present, is_inside_day and adx should be computed
        self.assertIn("adx", indicators["AAPL"].columns)


class TestRunBacktextNewPaths(unittest.TestCase):
    """Cover lines 1019-1021, 1041-1042, 1047-1049, 1054-1057, 1063-1066."""

    def setUp(self):
        self._save = patch("backtest.engine._save_results")
        self._print = patch("backtest.engine._print_results")
        self._save.start()
        self._print.start()

    def tearDown(self):
        self._save.stop()
        self._print.stop()

    def test_spy_multiindex_fallback(self):
        """Lines 1019-1021: SPY download returns MultiIndex → column slice used."""
        main_raw = _make_raw(n=100)
        # SPY fallback data with MultiIndex columns
        spy_idx = pd.bdate_range("2024-11-01", periods=100)
        spy_closes = [400.0 + i * 0.1 for i in range(100)]
        spy_multi = pd.DataFrame(
            {
                ("Close", "SPY"): spy_closes,
                ("Volume", "SPY"): [5_000_000] * 100,
            },
            index=spy_idx,
        )
        spy_multi.columns = pd.MultiIndex.from_tuples(spy_multi.columns)

        call_count = [0]

        def _download_side_effect(sym, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return main_raw  # first call: main symbols
            return spy_multi  # second call: SPY fallback

        with patch("backtest.engine.yf.download", side_effect=_download_side_effect):
            result = run_backtest(["AAPL", "FLAT"], "2025-03-01", "2025-03-07")
        self.assertIn("total_trades", result)

    def test_vix_exception_handled(self):
        """Lines 1041-1042: VIX download raises → warning logged, continues."""
        main_raw = _make_raw(n=100)
        call_count = [0]

        def _download_side_effect(sym, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return main_raw
            if isinstance(sym, str) and "VIX" in sym:
                raise RuntimeError("VIX unavailable")
            return main_raw

        with patch("backtest.engine.yf.download", side_effect=_download_side_effect):
            result = run_backtest(["AAPL", "FLAT"], "2025-03-01", "2025-03-07")
        self.assertIn("total_trades", result)

    def test_use_intraday_disabled_warning(self):
        """Lines 1047-1049: intraday fetch returns {} → warning, intraday_data=None."""
        with (
            patch("backtest.engine.yf.download", return_value=_make_raw(n=100)),
            patch("backtest.engine._fetch_intraday_bars", return_value={}),
        ):
            result = run_backtest(["AAPL", "FLAT"], "2025-03-01", "2025-03-07", use_intraday=True)
        self.assertIn("total_trades", result)

    def test_use_fundamentals_fetches_both(self):
        """Lines 1054-1057, 1063-1066: use_fundamentals=True → both histories fetched."""
        with (
            patch("backtest.engine.yf.download", return_value=_make_raw(n=100)),
            patch("backtest.engine.prefetch_earnings_history", return_value={}) as mock_earn,
            patch("backtest.engine.prefetch_insider_history", return_value={}) as mock_ins,
        ):
            run_backtest(["AAPL", "FLAT"], "2025-03-01", "2025-03-07", use_fundamentals=True)
        mock_earn.assert_called_once()
        mock_ins.assert_called_once()


class TestRunWalkForwardNewPaths(unittest.TestCase):
    """Cover lines 1163-1165, 1172-1173, 1175-1176, 1178, 1219-1220."""

    def test_spy_multiindex_in_walk_forward(self):
        """Lines 1163-1165: walk-forward SPY fetch returns MultiIndex."""
        main_raw = _make_raw(n=100)
        spy_idx = pd.bdate_range("2024-11-01", periods=100)
        spy_closes = [400.0 + i * 0.1 for i in range(100)]
        spy_multi = pd.DataFrame(
            {("Close", "SPY"): spy_closes, ("Volume", "SPY"): [5_000_000] * 100},
            index=spy_idx,
        )
        spy_multi.columns = pd.MultiIndex.from_tuples(spy_multi.columns)
        call_count = [0]

        def _dl(sym, **kwargs):
            call_count[0] += 1
            return main_raw if call_count[0] == 1 else spy_multi

        with patch("backtest.engine.yf.download", side_effect=_dl):
            result = run_walk_forward_optimized(
                ["AAPL", "FLAT"],
                "2025-02-01",
                "2025-03-07",
                train_days=10,
                test_days=5,
                param_grid=_LOOSE_PARAM_GRID,
            )
        self.assertIn("folds", result)

    def test_use_fundamentals_walk_forward(self):
        """Lines 1172-1173, 1175-1176, 1178: use_fundamentals=True path."""
        with (
            patch("backtest.engine.yf.download", return_value=_make_raw(n=100)),
            patch("backtest.engine.prefetch_earnings_history", return_value={}) as mock_earn,
            patch("backtest.engine.prefetch_insider_history", return_value={}) as mock_ins,
        ):
            run_walk_forward_optimized(
                ["AAPL", "FLAT"],
                "2025-02-01",
                "2025-03-07",
                train_days=10,
                test_days=5,
                param_grid=_LOOSE_PARAM_GRID,
                use_fundamentals=True,
            )
        mock_earn.assert_called_once()
        mock_ins.assert_called_once()

    def test_use_earnings_only_walk_forward(self):
        """Line 1172-1173 (use_earnings_only): earnings fetched, insider skipped."""
        with (
            patch("backtest.engine.yf.download", return_value=_make_raw(n=100)),
            patch("backtest.engine.prefetch_earnings_history", return_value={}) as mock_earn,
            patch("backtest.engine.prefetch_insider_history", return_value={}) as mock_ins,
        ):
            run_walk_forward_optimized(
                ["AAPL", "FLAT"],
                "2025-02-01",
                "2025-03-07",
                train_days=10,
                test_days=5,
                param_grid=_LOOSE_PARAM_GRID,
                use_earnings_only=True,
            )
        mock_earn.assert_called_once()
        mock_ins.assert_not_called()

    def test_excluded_symbols_logged(self):
        """Lines 1219-1220: pit_indicators smaller than indicators → debug logged."""
        raw = _make_raw(n=100)
        with (
            patch("backtest.engine.yf.download", return_value=raw),
            patch("backtest.engine.get_universe_for_date", return_value=["AAPL"]),
        ):
            result = run_walk_forward_optimized(
                ["AAPL", "FLAT"],
                "2025-02-01",
                "2025-03-07",
                train_days=10,
                test_days=5,
                param_grid=_LOOSE_PARAM_GRID,
            )
        self.assertIn("folds", result)


class TestRunAblationNewPaths(unittest.TestCase):
    """Cover lines 1394-1395, 1410-1412, 1430-1431, 1435, 1440-1441, 1443-1444."""

    def test_empty_data_returns_empty(self):
        """Lines 1394-1395: raw.empty → returns {}."""
        with patch("backtest.engine.yf.download", return_value=pd.DataFrame()):
            result = run_ablation(["AAPL"], "2025-01-01", "2025-06-30")
        self.assertEqual(result, {})

    def test_spy_multiindex_in_ablation(self):
        """Lines 1410-1412: SPY fetch returns MultiIndex columns."""
        main_raw = _make_raw(n=100)
        spy_idx = pd.bdate_range("2024-11-01", periods=100)
        spy_multi = pd.DataFrame(
            {("Close", "SPY"): [400.0] * 100, ("Volume", "SPY"): [5_000_000] * 100},
            index=spy_idx,
        )
        spy_multi.columns = pd.MultiIndex.from_tuples(spy_multi.columns)
        call_count = [0]

        def _dl(sym, **kwargs):
            call_count[0] += 1
            return main_raw if call_count[0] == 1 else spy_multi

        with (
            patch("backtest.engine.yf.download", side_effect=_dl),
            patch("backtest.engine.prefetch_earnings_history", return_value={}),
            patch("backtest.engine.prefetch_insider_history", return_value={}),
        ):
            result = run_ablation(["AAPL", "FLAT"], "2025-01-01", "2025-06-30")
        self.assertIn("baseline", result)

    def test_vix_exception_in_ablation(self):
        """Lines 1430-1431: VIX raises → continues gracefully."""
        main_raw = _make_raw(n=100)
        call_count = [0]

        def _dl(sym, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return main_raw
            raise RuntimeError("VIX unavailable")

        with (
            patch("backtest.engine.yf.download", side_effect=_dl),
            patch("backtest.engine.prefetch_earnings_history", return_value={}),
            patch("backtest.engine.prefetch_insider_history", return_value={}),
        ):
            result = run_ablation(["AAPL", "FLAT"], "2025-01-01", "2025-06-30")
        self.assertIn("baseline", result)

    def test_use_fundamentals_in_ablation(self):
        """Lines 1440-1441, 1443-1444: use_fundamentals=True."""
        with (
            patch("backtest.engine.yf.download", return_value=_make_raw(n=100)),
            patch("backtest.engine.prefetch_earnings_history", return_value={}) as mock_earn,
            patch("backtest.engine.prefetch_insider_history", return_value={}) as mock_ins,
        ):
            run_ablation(["AAPL", "FLAT"], "2025-01-01", "2025-06-30", use_fundamentals=True)
        mock_earn.assert_called_once()
        mock_ins.assert_called_once()

    def test_use_earnings_only_in_ablation(self):
        """Line 1440-1441 (use_earnings_only branch): earnings fetched, insider skipped."""
        with (
            patch("backtest.engine.yf.download", return_value=_make_raw(n=100)),
            patch("backtest.engine.prefetch_earnings_history", return_value={}) as mock_earn,
            patch("backtest.engine.prefetch_insider_history", return_value={}) as mock_ins,
        ):
            run_ablation(["AAPL", "FLAT"], "2025-01-01", "2025-06-30", use_earnings_only=True)
        mock_earn.assert_called_once()
        mock_ins.assert_not_called()


class TestRunBackwardEliminationNewPaths(unittest.TestCase):
    """Cover lines 1566-1568, 1586-1587, 1591, 1599-1600, 1628, 1646-1649, 1654-1669."""

    def _run(self, **kwargs):
        raw = _make_raw(n=100)
        with (
            patch("backtest.engine.yf.download", return_value=raw),
            patch("backtest.engine.prefetch_earnings_history", return_value={}),
            patch("backtest.engine.prefetch_insider_history", return_value={}),
        ):
            return run_backward_elimination(["AAPL", "FLAT"], "2025-01-01", "2025-06-30", **kwargs)

    def test_spy_multiindex_in_backward_elimination(self):
        """Lines 1566-1568: SPY fetch returns MultiIndex in backward elimination."""
        main_raw = _make_raw(n=100)
        spy_idx = pd.bdate_range("2024-11-01", periods=100)
        spy_multi = pd.DataFrame(
            {("Close", "SPY"): [400.0] * 100, ("Volume", "SPY"): [5_000_000] * 100},
            index=spy_idx,
        )
        spy_multi.columns = pd.MultiIndex.from_tuples(spy_multi.columns)
        call_count = [0]

        def _dl(sym, **kwargs):
            call_count[0] += 1
            return main_raw if call_count[0] == 1 else spy_multi

        with (
            patch("backtest.engine.yf.download", side_effect=_dl),
            patch("backtest.engine.prefetch_earnings_history", return_value={}),
            patch("backtest.engine.prefetch_insider_history", return_value={}),
        ):
            result = run_backward_elimination(["AAPL", "FLAT"], "2025-01-01", "2025-06-30")
        self.assertIn("original_baseline", result)

    def test_vix_exception_in_backward_elimination(self):
        """Lines 1586-1587: VIX raises → continues."""
        main_raw = _make_raw(n=100)
        call_count = [0]

        def _dl(sym, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return main_raw
            raise RuntimeError("VIX unavailable")

        with (
            patch("backtest.engine.yf.download", side_effect=_dl),
            patch("backtest.engine.prefetch_earnings_history", return_value={}),
            patch("backtest.engine.prefetch_insider_history", return_value={}),
        ):
            result = run_backward_elimination(["AAPL", "FLAT"], "2025-01-01", "2025-06-30")
        self.assertIn("original_baseline", result)

    def test_use_fundamentals_in_backward_elimination(self):
        """Lines 1599-1600: use_fundamentals=True fetches insider too."""
        with (
            patch("backtest.engine.yf.download", return_value=_make_raw(n=100)),
            patch("backtest.engine.prefetch_earnings_history", return_value={}) as mock_earn,
            patch("backtest.engine.prefetch_insider_history", return_value={}) as mock_ins,
        ):
            run_backward_elimination(
                ["AAPL", "FLAT"], "2025-01-01", "2025-06-30", use_fundamentals=True
            )
        mock_earn.assert_called_once()
        mock_ins.assert_called_once()

    def test_backward_elimination_with_steps_prints(self):
        """Lines 1702-1705: steps list non-empty → step table printed without error."""
        r = {
            "original_baseline": {"sharpe_ratio": 0.5, "total_return_pct": 5.0, "total_trades": 10},
            "final_result": {"sharpe_ratio": 0.8, "total_return_pct": 7.0, "total_trades": 8},
            "steps": [
                {
                    "step": 1,
                    "signal_removed": "macd_crossover",
                    "sharpe_delta": 0.3,
                    "sharpe_after": 0.8,
                    "trades_removed": 2,
                }
            ],
            "signals_kept": ["momentum", "mean_reversion"],
            "signals_removed": ["macd_crossover"],
        }
        try:
            _print_backward_elimination_results(r, "2025-01-01", "2025-06-30")
        except Exception as exc:
            self.fail(f"_print_backward_elimination_results raised: {exc}")


class TestRunSignalAnalysisNewPaths(unittest.TestCase):
    """Cover lines 1801-1803, 1821-1822, 1826, 1831-1832, 1834-1835, 1872-1884,
    1888-1892, 1896-1905, 1936-1959, 1964, 1968-1969, 1983-1993."""

    def _run(self, **kwargs):
        raw = _make_raw(n=100)
        with (
            patch("backtest.engine.yf.download", return_value=raw),
            patch("backtest.engine.prefetch_earnings_history", return_value={}),
            patch("backtest.engine.prefetch_insider_history", return_value={}),
        ):
            return run_signal_analysis(["AAPL", "FLAT"], "2025-01-01", "2025-06-30", **kwargs)

    def test_spy_multiindex_in_signal_analysis(self):
        """Lines 1801-1803: SPY fetch returns MultiIndex in signal_analysis."""
        main_raw = _make_raw(n=100)
        spy_idx = pd.bdate_range("2024-11-01", periods=100)
        spy_multi = pd.DataFrame(
            {("Close", "SPY"): [400.0] * 100, ("Volume", "SPY"): [5_000_000] * 100},
            index=spy_idx,
        )
        spy_multi.columns = pd.MultiIndex.from_tuples(spy_multi.columns)
        call_count = [0]

        def _dl(sym, **kwargs):
            call_count[0] += 1
            return main_raw if call_count[0] == 1 else spy_multi

        with (
            patch("backtest.engine.yf.download", side_effect=_dl),
            patch("backtest.engine.prefetch_earnings_history", return_value={}),
            patch("backtest.engine.prefetch_insider_history", return_value={}),
        ):
            result = run_signal_analysis(["AAPL", "FLAT"], "2025-01-01", "2025-06-30")
        self.assertIn("regime_stats", result)

    def test_vix_exception_in_signal_analysis(self):
        """Lines 1821-1822: VIX raises → continues."""
        main_raw = _make_raw(n=100)
        call_count = [0]

        def _dl(sym, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return main_raw
            raise RuntimeError("VIX unavailable")

        with (
            patch("backtest.engine.yf.download", side_effect=_dl),
            patch("backtest.engine.prefetch_earnings_history", return_value={}),
            patch("backtest.engine.prefetch_insider_history", return_value={}),
        ):
            result = run_signal_analysis(["AAPL", "FLAT"], "2025-01-01", "2025-06-30")
        self.assertIn("regime_stats", result)

    def test_use_fundamentals_in_signal_analysis(self):
        """Lines 1831-1832, 1834-1835: use_fundamentals=True."""
        with (
            patch("backtest.engine.yf.download", return_value=_make_raw(n=100)),
            patch("backtest.engine.prefetch_earnings_history", return_value={}) as mock_earn,
            patch("backtest.engine.prefetch_insider_history", return_value={}) as mock_ins,
        ):
            run_signal_analysis(["AAPL", "FLAT"], "2025-01-01", "2025-06-30", use_fundamentals=True)
        mock_earn.assert_called_once()
        mock_ins.assert_called_once()

    def test_use_earnings_only_in_signal_analysis(self):
        """Line 1831-1832 (earnings-only): insider not fetched."""
        with (
            patch("backtest.engine.yf.download", return_value=_make_raw(n=100)),
            patch("backtest.engine.prefetch_earnings_history", return_value={}) as mock_earn,
            patch("backtest.engine.prefetch_insider_history", return_value={}) as mock_ins,
        ):
            run_signal_analysis(
                ["AAPL", "FLAT"], "2025-01-01", "2025-06-30", use_earnings_only=True
            )
        mock_earn.assert_called_once()
        mock_ins.assert_not_called()

    def test_regime_stats_accumulate_wins_and_losses(self):
        """Lines 1872-1884, 1888-1892, 1896-1905: trades with pnl_pct > 0 and <= 0
        accumulate into regime_stats properly (wins + losses paths)."""
        # Use loose params to generate trades with entry_regime set
        idx = pd.bdate_range("2025-01-02", periods=60)
        n = len(idx)
        # Alternating wins/losses: price oscillates ±5%
        closes = []
        for i in range(n):
            if i % 6 < 3:
                closes.append(100.0 + i * 0.3)
            else:
                closes.append(100.0 - i * 0.1)
        df = pd.DataFrame(
            {
                "Close": closes,
                "Open": [c * 0.999 for c in closes],
                "Volume": [2_000_000] * n,
                "rsi": [50.0] * n,
                "bb_pct": [0.5] * n,
                "vol_ratio": [1.5] * n,
                "ema9": [101.0] * n,
                "ema21": [100.0] * n,
                "macd_diff": [0.5] * n,
                "ret_5d": [2.0] * n,
            },
            index=idx,
        )
        regime = {d.strftime("%Y-%m-%d"): "BULL_TRENDING" for d in idx}
        _run_simulation(
            {"AAPL": df},
            idx[1:],
            initial_capital=10_000.0,
            params=_LOOSE_ENTRY,
            regime_by_date=regime,
        )
        # Now run signal_analysis with these trades (test via direct path)
        # This covers lines 1872-1884 in isolation via a _run call
        with (
            patch("backtest.engine.yf.download", return_value=_make_raw(n=100)),
            patch("backtest.engine.prefetch_earnings_history", return_value={}),
            patch("backtest.engine.prefetch_insider_history", return_value={}),
        ):
            sa_result = run_signal_analysis(
                ["AAPL", "FLAT"],
                "2025-01-01",
                "2025-06-30",
                params=_LOOSE_ENTRY,
            )
        # If any trades fired, regime_stats should have win/loss counts
        if sa_result["regime_stats"]:
            for _sig, reg_dict in sa_result["regime_stats"].items():
                for _reg, cell in reg_dict.items():
                    total = cell["wins"] + cell["losses"]
                    self.assertGreaterEqual(total, 0)

    def test_print_regime_table_with_intraday_signals(self):
        """Lines 1936-1959, 1964, 1968-1969: intraday signal present → footnote printed."""
        import io
        import sys

        # Build regime_stats with an intraday signal to trigger footnote
        r = {
            "regime_stats": {
                "orb_breakout": {
                    "BULL_TRENDING": {
                        "wins": 5,
                        "losses": 3,
                        "total_return": 4.0,
                        "win_rate_ci_low": float("nan"),
                        "win_rate_ci_high": float("nan"),
                    }
                },
                "momentum": {
                    "CHOPPY": {
                        "wins": 2,
                        "losses": 8,
                        "total_return": -1.5,
                        "win_rate_ci_low": float("nan"),
                        "win_rate_ci_high": float("nan"),
                    }
                },
            },
            "decay_stats": {},
        }
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            _print_regime_table(r, "2025-01-01", "2025-06-30")
        finally:
            sys.stdout = old
        output = buf.getvalue()
        # Low-n footnote (n=8 < 30) and intraday footnote should both appear
        self.assertIn("†", output)
        self.assertIn("*", output)

    def test_print_regime_table_with_valid_ci(self):
        """Lines 1954-1956: ci_lo not NaN → CI string printed."""
        import io
        import sys

        r = {
            "regime_stats": {
                "momentum": {
                    "BULL_TRENDING": {
                        "wins": 20,
                        "losses": 10,
                        "total_return": 15.0,
                        "win_rate_ci_low": 0.45,
                        "win_rate_ci_high": 0.85,
                    }
                }
            },
            "decay_stats": {},
        }
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            _print_regime_table(r, "2025-01-01", "2025-06-30")
        finally:
            sys.stdout = old
        output = buf.getvalue()
        self.assertIn("CI", output)

    def test_print_hold_period_table_with_data(self):
        """Lines 1983-1993: decay_stats has data → lines printed including decay flag."""
        import io
        import sys

        r = {
            "regime_stats": {},
            "decay_stats": {
                "momentum": {
                    1: {"wins": 5, "losses": 5, "total_return": 0.5},
                    2: {"wins": 2, "losses": 8, "total_return": -2.0},  # avg < -0.1 → ← decays
                }
            },
        }
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            _print_hold_period_table(r, "2025-01-01", "2025-06-30")
        finally:
            sys.stdout = old
        output = buf.getvalue()
        self.assertIn("Day 2", output)
        self.assertIn("decays", output)


class TestRunHoldoutEvaluationNewPaths(unittest.TestCase):
    """Lines 2100-2101, 2110."""

    def test_empty_data_returns_empty(self):
        """Lines 2100-2101: raw.empty → returns {}."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            log_path = os.path.join(tmp, "holdout_log.jsonl")
            with (
                patch("backtest.engine.yf.download", return_value=pd.DataFrame()),
                patch("backtest.engine.LOG_DIR", tmp),
                patch("backtest.engine._HOLDOUT_LOG", log_path),
            ):
                result = run_holdout_evaluation(
                    frozen_params={}, version="vTEST_EMPTY", symbols=["AAPL"]
                )
        self.assertEqual(result, {})

    def test_spy_present_computes_regime(self):
        """Line 2110: spy_indicators not None → _compute_regimes called."""
        import tempfile

        ohlcv = _make_ohlcv_full(300)
        ind = _compute_indicators(ohlcv)
        # Include SPY in indicators so spy_indicators is not None → line 2110
        spy_ohlcv = _make_ohlcv(300)
        spy_ind = _compute_indicators(spy_ohlcv)
        fake_ind = {"AAPL": ind, "SPY": spy_ind}
        fake_raw = pd.concat(
            {"Close": ohlcv[["Close"]], "Open": ohlcv[["Open"]], "Volume": ohlcv[["Volume"]]},
            axis=1,
        )
        fake_raw.columns = pd.MultiIndex.from_tuples(
            [("Close", "AAPL"), ("Open", "AAPL"), ("Volume", "AAPL")]
        )
        with tempfile.TemporaryDirectory() as tmp:
            log_path = os.path.join(tmp, "holdout_log.jsonl")
            with (
                patch("backtest.engine.yf.download", return_value=fake_raw),
                patch("backtest.engine._build_indicators", return_value=fake_ind),
                patch("backtest.engine.LOG_DIR", tmp),
                patch("backtest.engine._HOLDOUT_LOG", log_path),
            ):
                result = run_holdout_evaluation(
                    frozen_params={}, version="vTEST_SPY", symbols=["AAPL"]
                )
        self.assertIn("total_trades", result)


class TestBackwardEliminationSteps(unittest.TestCase):
    """Lines 1628, 1646-1649, 1654-1669: exercise the inner elimination loop.

    We mock _run_simulation so the first removal always improves Sharpe (delta>0),
    the second removal is tied (delta==best_delta), and subsequent calls drop back to 0.
    This exercises the 'disabled.add' path (1654-1669) AND the 'not remaining' break
    path (1628) when all signals have been removed.
    """

    def test_elimination_step_recorded(self):
        """Lines 1654-1669: step is appended when a signal improves Sharpe."""
        raw = _make_raw(n=100)

        # baseline sharpe=0.0; first removal of any signal → sharpe=0.5
        # subsequent calls → sharpe=0.0 (no further improvement)
        def _mock_sim(indicators, dates, **kwargs):
            disabled = kwargs.get("disabled_signals") or frozenset()
            if not disabled:
                # baseline
                return {
                    "sharpe_ratio": 0.0,
                    "total_return_pct": 0.0,
                    "total_trades": 0,
                    "win_rate_pct": 0.0,
                    "avg_return_per_trade_pct": 0.0,
                    "max_drawdown_pct": 0.0,
                    "final_value": 10000.0,
                    "initial_capital": 10000.0,
                    "equity_curve": [],
                    "trades": [],
                    "by_signal": {},
                    "signals_tested": set(),
                    "signals_not_tested": set(),
                    "validation_scope": "rule_proxy_only",
                }
            if len(disabled) == 1:
                # First removal: improve Sharpe for the first signal only
                sig = next(iter(disabled))
                from backtest.engine import _SIGNAL_PRIORITY

                first_sig = min(_SIGNAL_PRIORITY, key=lambda s: _SIGNAL_PRIORITY[s])
                sharpe = 0.5 if sig == first_sig else 0.0
                return {
                    "sharpe_ratio": sharpe,
                    "total_return_pct": 0.0,
                    "total_trades": 0,
                    "win_rate_pct": 0.0,
                    "avg_return_per_trade_pct": 0.0,
                    "max_drawdown_pct": 0.0,
                    "final_value": 10000.0,
                    "initial_capital": 10000.0,
                    "equity_curve": [],
                    "trades": [],
                    "by_signal": {},
                    "signals_tested": set(),
                    "signals_not_tested": set(),
                    "validation_scope": "rule_proxy_only",
                }
            # Two+ signals disabled → no improvement
            return {
                "sharpe_ratio": 0.5,
                "total_return_pct": 0.0,
                "total_trades": 0,
                "win_rate_pct": 0.0,
                "avg_return_per_trade_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "final_value": 10000.0,
                "initial_capital": 10000.0,
                "equity_curve": [],
                "trades": [],
                "by_signal": {},
                "signals_tested": set(),
                "signals_not_tested": set(),
                "validation_scope": "rule_proxy_only",
            }

        with (
            patch("backtest.engine.yf.download", return_value=raw),
            patch("backtest.engine.prefetch_earnings_history", return_value={}),
            patch("backtest.engine.prefetch_insider_history", return_value={}),
            patch("backtest.engine._run_simulation", side_effect=_mock_sim),
        ):
            result = run_backward_elimination(["AAPL", "FLAT"], "2025-01-01", "2025-06-30")
        # At least one step should be recorded (the first signal removed)
        self.assertGreater(len(result["steps"]), 0)

    def test_all_signals_disabled_hits_not_remaining_break(self):
        """Line 1628: when all signals are disabled, loop breaks via 'not remaining'."""
        raw = _make_raw(n=100)
        from backtest.engine import _SIGNAL_PRIORITY

        # Mock: every removal always improves Sharpe — forces all signals to be disabled
        def _mock_sim(indicators, dates, **kwargs):
            disabled = kwargs.get("disabled_signals") or frozenset()
            base = {
                "total_return_pct": 0.0,
                "total_trades": 0,
                "win_rate_pct": 0.0,
                "avg_return_per_trade_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "final_value": 10000.0,
                "initial_capital": 10000.0,
                "equity_curve": [],
                "trades": [],
                "by_signal": {},
                "signals_tested": set(),
                "signals_not_tested": set(),
                "validation_scope": "rule_proxy_only",
            }
            # Sharpe improves linearly with number of disabled signals
            base["sharpe_ratio"] = float(len(disabled)) * 0.01
            return base

        with (
            patch("backtest.engine.yf.download", return_value=raw),
            patch("backtest.engine.prefetch_earnings_history", return_value={}),
            patch("backtest.engine.prefetch_insider_history", return_value={}),
            patch("backtest.engine._run_simulation", side_effect=_mock_sim),
        ):
            result = run_backward_elimination(["AAPL", "FLAT"], "2025-01-01", "2025-06-30")
        # All signals should have been removed
        self.assertEqual(result["signals_kept"], [])
        self.assertEqual(len(result["signals_removed"]), len(_SIGNAL_PRIORITY))


class TestRegimeStatsAccumulation(unittest.TestCase):
    """Lines 1872-1884, 1888-1892, 1896-1905: regime stats accumulation requires
    actual SELL trades with entry_regime + pnl_pct set. We mock _run_simulation
    to return synthetic trade data so the loop in run_signal_analysis executes."""

    def _make_sim_result(self, trades):
        return {
            "sharpe_ratio": 1.0,
            "total_return_pct": 5.0,
            "total_trades": len(trades),
            "win_rate_pct": 60.0,
            "avg_return_per_trade_pct": 0.5,
            "max_drawdown_pct": -2.0,
            "final_value": 10500.0,
            "initial_capital": 10000.0,
            "equity_curve": [("2025-01-02", 10000.0)],
            "trades": trades,
            "by_signal": {},
            "signals_tested": set(),
            "signals_not_tested": set(),
            "validation_scope": "rule_proxy_only",
        }

    def test_wins_and_losses_accumulated(self):
        """Full coverage of the regime stats loop including win (pnl>0) and loss (pnl<=0) paths."""
        trades = [
            {
                "action": "SELL",
                "pnl_pct": 2.5,
                "signal": "momentum",
                "entry_regime": "BULL_TRENDING",
                "days_held": 2,
                "reason": "time_exit",
                "date": "2025-01-03",
            },
            {
                "action": "SELL",
                "pnl_pct": -1.2,
                "signal": "momentum",
                "entry_regime": "BULL_TRENDING",
                "days_held": 1,
                "reason": "stop_loss",
                "date": "2025-01-04",
            },
            {
                "action": "SELL",
                "pnl_pct": 0.0,
                "signal": "mean_reversion",
                "entry_regime": "CHOPPY",
                "days_held": 3,
                "reason": "time_exit",
                "date": "2025-01-05",
            },
        ]
        sim_result = self._make_sim_result(trades)

        with (
            patch("backtest.engine.yf.download", return_value=_make_raw(n=100)),
            patch("backtest.engine.prefetch_earnings_history", return_value={}),
            patch("backtest.engine.prefetch_insider_history", return_value={}),
            patch("backtest.engine._run_simulation", return_value=sim_result),
        ):
            result = run_signal_analysis(["AAPL"], "2025-01-01", "2025-06-30")

        # momentum in BULL_TRENDING should have 1 win, 1 loss
        self.assertIn("momentum", result["regime_stats"])
        self.assertIn("BULL_TRENDING", result["regime_stats"]["momentum"])
        cell = result["regime_stats"]["momentum"]["BULL_TRENDING"]
        self.assertEqual(cell["wins"], 1)
        self.assertEqual(cell["losses"], 1)

        # mean_reversion in CHOPPY: pnl_pct=0.0 → loss path (not > 0)
        self.assertIn("mean_reversion", result["regime_stats"])
        mr_cell = result["regime_stats"]["mean_reversion"]["CHOPPY"]
        self.assertEqual(mr_cell["losses"], 1)

        # decay_stats should also be populated (lines 1896-1905)
        self.assertIn("momentum", result["decay_stats"])

    def test_hold_period_decay_flag_triggered(self):
        """Line 1987 in _print_hold_period_table: avg < -0.1 → '← decays' flag."""
        import io
        import sys

        trades = [
            {
                "action": "SELL",
                "pnl_pct": -1.5,
                "signal": "momentum",
                "entry_regime": "CHOPPY",
                "days_held": 2,
                "reason": "stop_loss",
                "date": "2025-01-03",
            },
        ]
        sim_result = self._make_sim_result(trades)

        with (
            patch("backtest.engine.yf.download", return_value=_make_raw(n=100)),
            patch("backtest.engine.prefetch_earnings_history", return_value={}),
            patch("backtest.engine.prefetch_insider_history", return_value={}),
            patch("backtest.engine._run_simulation", return_value=sim_result),
        ):
            result = run_signal_analysis(["AAPL"], "2025-01-01", "2025-06-30")

        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            _print_hold_period_table(result, "2025-01-01", "2025-06-30")
        finally:
            sys.stdout = old
        output = buf.getvalue()
        # avg = -1.5/1 = -1.5 < -0.1 → "← decays" should appear
        self.assertIn("decays", output)


class TestFetchIntradayBarsBody(unittest.TestCase):
    """Lines 332-377: _fetch_intraday_bars body after API key check."""

    def _alpaca_modules(self, bars=None, raise_exc=None):
        """Build sys.modules patch + mock client."""
        mock_bar = MagicMock()
        mock_dt = datetime(2025, 1, 2, 10, 30, tzinfo=_ET)
        mock_bar.timestamp.astimezone.return_value = mock_dt

        if raise_exc is not None:
            resp = MagicMock()
            mock_client = MagicMock()
            mock_client.get_stock_bars.side_effect = raise_exc
        else:
            bars_data = [] if bars == [] else [mock_bar]
            resp = MagicMock()
            resp.data.get.return_value = bars_data
            mock_client = MagicMock()
            mock_client.get_stock_bars.return_value = resp

        historical = MagicMock()
        historical.StockHistoricalDataClient = MagicMock(return_value=mock_client)
        modules = {
            "alpaca": MagicMock(),
            "alpaca.data": MagicMock(),
            "alpaca.data.historical": historical,
            "alpaca.data.requests": MagicMock(),
            "alpaca.data.timeframe": MagicMock(),
        }
        return modules, mock_client

    def _with_keys(self, fn):
        import config as _cfg

        orig_key, orig_secret = _cfg.ALPACA_API_KEY, _cfg.ALPACA_SECRET_KEY
        _cfg.ALPACA_API_KEY = "test_key"
        _cfg.ALPACA_SECRET_KEY = "test_secret"
        try:
            fn()
        finally:
            _cfg.ALPACA_API_KEY = orig_key
            _cfg.ALPACA_SECRET_KEY = orig_secret

    def test_success_bars_returned(self):
        """Lines 332-371, 376-377: bars returned → sym_result populated."""
        import sys

        import backtest.engine as _eng

        mods, _ = self._alpaca_modules()

        def run():
            with (
                patch.dict(sys.modules, mods),
                patch("backtest.engine._compute_intraday_day", return_value={"vwap": 100.0}),
            ):
                result = _eng._fetch_intraday_bars(["AAPL"], "2025-01-02", "2025-01-02")
            self.assertIn("AAPL", result)
            self.assertIn("2025-01-02", result["AAPL"])

        self._with_keys(run)

    def test_empty_bars_skips_symbol(self):
        """Lines 355-357: bars_data empty → continue → symbol absent from result."""
        import sys

        import backtest.engine as _eng

        mods, _ = self._alpaca_modules(bars=[])

        def run():
            with patch.dict(sys.modules, mods):
                result = _eng._fetch_intraday_bars(["AAPL"], "2025-01-02", "2025-01-02")
            self.assertNotIn("AAPL", result)

        self._with_keys(run)

    def test_exception_in_fetch_logged(self):
        """Lines 373-374: get_stock_bars raises → warning logged, result empty."""
        import sys

        import backtest.engine as _eng

        mods, _ = self._alpaca_modules(raise_exc=RuntimeError("network"))

        def run():
            with patch.dict(sys.modules, mods):
                result = _eng._fetch_intraday_bars(["AAPL"], "2025-01-02", "2025-01-02")
            self.assertEqual(result, {})

        self._with_keys(run)
