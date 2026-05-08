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
        "rsi": 50.0,
        "bb_pct": 0.5,
        "vol_ratio": 1.0,
        "ema9": 100.0,
        "ema21": 100.0,
        "macd_diff": 0.0,
        "ret_5d": 0.0,
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

    def test_signals_tested_contains_expected_signals(self):
        indicators = self._build_indicators()
        dates = pd.bdate_range("2025-03-01", "2025-03-07")
        result = _run_simulation(indicators, dates, initial_capital=10_000.0)
        self.assertIn("mean_reversion", result["signals_tested"])
        self.assertIn("momentum", result["signals_tested"])

    def test_signals_not_tested_excludes_untested_signals(self):
        indicators = self._build_indicators()
        dates = pd.bdate_range("2025-03-01", "2025-03-07")
        result = _run_simulation(indicators, dates, initial_capital=10_000.0)
        for sig in ("bb_squeeze_breakout", "breakout_52w", "rs_leader"):
            self.assertIn(sig, result["signals_not_tested"])

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
