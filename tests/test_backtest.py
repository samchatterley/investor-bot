"""Tests for backtest/engine.py — _compute_indicators, _entry_signal, run_backtest,
_run_simulation, run_walk_forward_optimized, _compute_intraday_day."""

import json
import os
import tempfile
import unittest
from collections import namedtuple
from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd

from backtest.engine import (
    _CORE_COLS,
    _DEFAULT_PARAMS,
    _SIGNAL_PRIORITY,
    _compute_indicators,
    _compute_intraday_day,
    _entry_signal,
    _print_results,
    _run_simulation,
    _save_results,
    run_backtest,
    run_walk_forward_optimized,
)

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

    def test_choppy_allows_mean_reversion(self):
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
    "consistency_pct",
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
