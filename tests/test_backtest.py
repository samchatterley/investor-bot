"""Tests for backtest/engine.py — _compute_indicators, _entry_signal, run_backtest,
_run_simulation, run_walk_forward_optimized."""
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from backtest.engine import (
    _DEFAULT_PARAMS,
    _MIN_TRAIN_TRADES,
    _compute_indicators,
    _entry_signal,
    _print_results,
    _run_simulation,
    _save_results,
    run_backtest,
    run_walk_forward_optimized,
)


def _make_ohlcv(n: int = 60) -> pd.DataFrame:
    idx = pd.bdate_range("2024-11-01", periods=n)
    prices = [100.0 + i * 0.5 for i in range(n)]
    return pd.DataFrame({"Close": prices, "Volume": [1_000_000] * n}, index=idx)


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
        "rsi": 50.0, "bb_pct": 0.5, "vol_ratio": 1.0,
        "ema9": 100.0, "ema21": 100.0, "macd_diff": 0.0, "ret_5d": 0.0,
    }
    defaults.update(kwargs)
    return pd.Series(defaults)


def _make_results_dict() -> dict:
    return {
        "start": "2025-01-01", "end": "2025-12-31",
        "initial_capital": 10_000, "final_value": 11_000,
        "total_return_pct": 10.0, "total_trades": 5,
        "win_rate_pct": 60.0, "avg_return_per_trade_pct": 1.5,
        "max_drawdown_pct": -3.0, "sharpe_ratio": 1.2,
        "by_signal": {"momentum": {"wins": 3, "losses": 2, "total_return": 7.5}},
        "trades": [],
        "equity_curve": [("2025-01-02", 10000.0), ("2025-01-03", 10100.0)],
    }


# ── _compute_indicators ───────────────────────────────────────────────────────

class TestComputeIndicators(unittest.TestCase):

    def test_returns_dataframe(self):
        self.assertIsInstance(_compute_indicators(_make_ohlcv(60)), pd.DataFrame)

    def test_has_expected_columns(self):
        result = _compute_indicators(_make_ohlcv(60))
        for col in ("rsi", "macd_diff", "ema9", "ema21", "bb_pct", "vol_ratio", "ret_5d"):
            self.assertIn(col, result.columns, f"Missing column: {col}")

    def test_result_has_no_nans(self):
        result = _compute_indicators(_make_ohlcv(60))
        self.assertFalse(result.isnull().any().any())

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


# ── _entry_signal ─────────────────────────────────────────────────────────────

class TestEntrySignal(unittest.TestCase):

    def test_mean_reversion_fires_when_all_conditions_met(self):
        self.assertEqual(_entry_signal(_make_row(rsi=30, bb_pct=0.20, vol_ratio=1.5)), "mean_reversion")

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
            rsi=30, bb_pct=0.20, vol_ratio=1.5,
            ema9=105, ema21=100, macd_diff=0.5, ret_5d=2.0,
        )
        self.assertEqual(_entry_signal(row), "mean_reversion")


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
        for key in ("initial_capital", "final_value", "total_return_pct", "total_trades",
                    "win_rate_pct", "sharpe_ratio", "by_signal", "equity_curve", "trades"):
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
        tight = {"rsi_threshold": 1, "bb_threshold": 0.01, "mr_vol_threshold": 999,
                 "mom_vol_threshold": 999, "mom_ret5d_threshold": 999}
        result = _run_simulation(indicators, dates, params=tight)
        self.assertEqual(result["total_trades"], 0)

    def test_loose_params_fire_momentum_signal(self):
        # Uptrend data: EMA9 > EMA21, macd_diff > 0; loosen vol and ret5d thresholds
        indicators = self._build_indicators()
        dates = pd.bdate_range("2025-02-03", "2025-02-28")
        loose = {"mom_vol_threshold": 0.9, "mom_ret5d_threshold": 0.3}
        result = _run_simulation(indicators, dates, initial_capital=10_000.0,
                                 max_hold_days=3, params=loose)
        self.assertGreater(result["total_trades"], 0)

    def test_custom_params_produce_different_trades_than_defaults(self):
        indicators = self._build_indicators()
        dates = pd.bdate_range("2025-02-03", "2025-02-28")
        default_result = _run_simulation(indicators, dates, initial_capital=10_000.0, max_hold_days=3)
        loose = {"mom_vol_threshold": 0.9, "mom_ret5d_threshold": 0.3}
        loose_result = _run_simulation(indicators, dates, initial_capital=10_000.0,
                                       max_hold_days=3, params=loose)
        # Loose params should produce at least as many trades as defaults
        self.assertGreaterEqual(loose_result["total_trades"], default_result["total_trades"])

    def test_slippage_reduces_final_value_when_trades_fire(self):
        indicators = self._build_indicators()
        dates = pd.bdate_range("2025-02-03", "2025-02-28")
        loose = {"mom_vol_threshold": 0.9, "mom_ret5d_threshold": 0.3}
        zero = _run_simulation(indicators, dates, initial_capital=10_000.0,
                               max_hold_days=3, params=loose, slippage_bps=0, spread_bps=0)
        costly = _run_simulation(indicators, dates, initial_capital=10_000.0,
                                 max_hold_days=3, params=loose, slippage_bps=20, spread_bps=10)
        if zero["total_trades"] > 0:
            self.assertGreaterEqual(zero["final_value"], costly["final_value"])


# ── run_backtest ──────────────────────────────────────────────────────────────

_EXPECTED_RESULT_KEYS = {
    "start", "end", "initial_capital", "final_value",
    "total_return_pct", "total_trades", "win_rate_pct",
    "avg_return_per_trade_pct", "max_drawdown_pct",
    "sharpe_ratio", "by_signal", "equity_curve", "trades",
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
        tight = {"rsi_threshold": 1, "bb_threshold": 0.01,
                 "mr_vol_threshold": 999, "mom_vol_threshold": 999,
                 "mom_ret5d_threshold": 999}
        result = self._run(params=tight)
        self.assertEqual(result["total_trades"], 0)


# ── run_walk_forward_optimized ────────────────────────────────────────────────

# Tiny param grid: 4 combos (fast grid search); vol/ret thresholds loose enough
# that momentum fires on the uptrend data used in _make_raw
_LOOSE_PARAM_GRID: dict = {
    "rsi_threshold": [35],
    "bb_threshold": [0.25],
    "mr_vol_threshold": [1.2],
    "mom_vol_threshold": [0.9],   # fires with vol_ratio=1.0
    "mom_ret5d_threshold": [0.3], # fires with ret_5d≈0.47% on 0.1/bar uptrend
}

_TIGHT_PARAM_GRID: dict = {
    "rsi_threshold": [35, 40],
    "bb_threshold": [0.25],
    "mr_vol_threshold": [1.2],
    "mom_vol_threshold": [999.0],  # nothing fires
    "mom_ret5d_threshold": [999.0],
}

_WF_FOLD_KEYS = {
    "train_start", "train_end", "test_start", "test_end",
    "best_params", "train_sharpe", "train_total_trades",
    "oos_total_return_pct", "oos_win_rate_pct", "oos_total_trades", "oos_sharpe",
    "oos_degradation", "random_baseline_return_pct",
}

_WF_SUMMARY_KEYS = {
    "n_folds", "mean_oos_return_pct", "mean_oos_win_rate_pct",
    "mean_oos_sharpe", "profitable_folds", "consistency_pct",
    "param_stability_pct", "mean_oos_degradation", "random_baseline_return_pct",
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
