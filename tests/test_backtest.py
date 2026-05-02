"""Tests for backtest/engine.py — _compute_indicators, _entry_signal, run_backtest."""
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from backtest.engine import (
    _compute_indicators,
    _entry_signal,
    _print_results,
    _save_results,
    run_backtest,
)


def _make_ohlcv(n: int = 60) -> pd.DataFrame:
    idx = pd.bdate_range("2024-11-01", periods=n)
    prices = [100.0 + i * 0.5 for i in range(n)]
    return pd.DataFrame({"Close": prices, "Volume": [1_000_000] * n}, index=idx)


def _make_raw(n: int = 100, symbols: tuple = ("AAPL", "FLAT")) -> pd.DataFrame:
    """Build a realistic multi-symbol yfinance download mock (MultiIndex columns)."""
    idx = pd.bdate_range("2024-11-01", periods=n)
    data: dict = {}
    for i, sym in enumerate(symbols):
        data[("Close", sym)] = [100.0 + j * 0.1 + i * 50 for j in range(n)]
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
        # Uptrend gives RSI ≈ 100 (no mean_reversion) and constant volume (no momentum)
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
        # Flat equity → std = 0 → Sharpe = 0
        result = self._run()
        self.assertEqual(result["sharpe_ratio"], 0.0)

    def test_max_drawdown_is_zero_when_equity_flat(self):
        result = self._run()
        self.assertEqual(result["max_drawdown_pct"], 0.0)


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
