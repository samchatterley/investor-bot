"""Tests for data/macro_data.py — 100% coverage."""

from __future__ import annotations

import contextlib
import json
import os
from unittest import TestCase
from unittest.mock import patch

import pandas as pd

from data.macro_data import (
    MacroSnapshot,
    _compute_snapshot,
    _download_macro_prices,
    _load_cache,
    _period_roc,
    _ratio_roc,
    _save_cache,
    _zero_snapshot,
    get_copper_gold_positive,
    get_credit_stress,
    get_duration_flight,
    get_macro_snapshot,
    get_usd_strong,
)


def _make_series(values: list[float]) -> pd.Series:
    """Create a simple numeric Series with integer index."""
    return pd.Series(values, dtype=float)


def _all_prices(n: int = 35) -> dict[str, pd.Series]:
    """Return a complete prices dict with all tickers having n values."""
    # HYG flat at 100, LQD flat at 100 → no trend
    return {
        "HYG": _make_series([100.0] * n),
        "LQD": _make_series([100.0] * n),
        "IEF": _make_series([100.0] * n),
        "TLT": _make_series([100.0] * n),
        "CPER": _make_series([100.0] * n),
        "GLD": _make_series([100.0] * n),
        "UUP": _make_series([100.0] * n),
        "SPY": _make_series([100.0] * n),
    }


class TestPeriodRoc(TestCase):
    def test_positive_roc(self):
        s = _make_series([100.0] * 10 + [110.0])
        self.assertAlmostEqual(_period_roc(s, 10), 10.0)

    def test_negative_roc(self):
        s = _make_series([110.0] * 6 + [100.0])
        result = _period_roc(s, 6)
        self.assertIsNotNone(result)
        self.assertLess(result, 0)

    def test_insufficient_data(self):
        s = _make_series([100.0] * 5)
        self.assertIsNone(_period_roc(s, 5))  # len == n, need > n

    def test_zero_previous(self):
        s = _make_series([0.0] + [100.0] * 5)
        self.assertIsNone(_period_roc(s, 5))

    def test_none_series(self):
        self.assertIsNone(_period_roc(None, 5))  # type: ignore[arg-type]

    def test_empty_series(self):
        self.assertIsNone(_period_roc(pd.Series(dtype=float), 5))

    def test_type_error_returns_none(self):
        """Object-dtype series with dicts triggers TypeError in float() → return None."""
        s = pd.Series([{"v": i} for i in range(12)])  # float(dict) raises TypeError
        self.assertIsNone(_period_roc(s, 10))


class TestRatioRoc(TestCase):
    def test_basic_ratio_roc(self):
        # HYG up 10%, LQD flat → ratio up ~10%
        hyg = _make_series([100.0] * 10 + [110.0])
        lqd = _make_series([100.0] * 11)
        result = _ratio_roc(hyg, lqd, 10)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 10.0, places=1)

    def test_none_inputs(self):
        s = _make_series([100.0] * 15)
        self.assertIsNone(_ratio_roc(None, s, 10))
        self.assertIsNone(_ratio_roc(s, None, 10))
        self.assertIsNone(_ratio_roc(None, None, 10))

    def test_insufficient_data(self):
        a = _make_series([100.0] * 5)
        b = _make_series([100.0] * 5)
        self.assertIsNone(_ratio_roc(a, b, 5))

    def test_zero_denominator(self):
        # b is zero → ratio undefined
        a = _make_series([100.0] * 15)
        b = _make_series([0.0] * 15)
        self.assertIsNone(_ratio_roc(a, b, 10))

    def test_exception_in_compute_returns_none(self):
        """String series causes TypeError in division → except Exception → None."""
        a = pd.Series(["x"] * 15)
        b = pd.Series(["y"] * 15)
        self.assertIsNone(_ratio_roc(a, b, 10))

    def test_zero_ratio_prev_returns_none(self):
        """ratio at iloc[-(n+1)] == 0 → if prev == 0 → return None (line 179)."""
        # a is 0 at index 4 (= iloc[-11] for n=10, len=15) → ratio 0/100 = 0
        a_vals = [100.0] * 4 + [0.0] + [100.0] * 10
        a = _make_series(a_vals)
        b = _make_series([100.0] * 15)
        self.assertIsNone(_ratio_roc(a, b, 10))


class TestComputeSnapshot(TestCase):
    def test_flat_prices_no_stress(self):
        snap = _compute_snapshot(_all_prices())
        self.assertFalse(snap.credit_stress)
        self.assertFalse(snap.duration_flight)
        self.assertFalse(snap.copper_gold_positive)
        self.assertFalse(snap.usd_strong)
        # Flat prices → zero ROC → data_available = True (None for each numeric field)
        # Actually flat means 0% ROC which is not None
        self.assertTrue(snap.data_available)

    def test_credit_stress_triggered(self):
        # HYG falls 3%: iloc[-11]=100.0 (prev), iloc[-1]=97.0 (curr) → ROC=-3% → stress
        prices = _all_prices(30)
        hyg_vals = [100.0] * 20 + [97.0] * 10
        prices["HYG"] = _make_series(hyg_vals)
        snap = _compute_snapshot(prices)
        self.assertTrue(snap.credit_stress)

    def test_credit_stress_below_threshold(self):
        # HYG falls 1% only → ROC=-1% < threshold of -2% → not stress
        prices = _all_prices(30)
        hyg_vals = [100.0] * 20 + [99.0] * 10
        prices["HYG"] = _make_series(hyg_vals)
        snap = _compute_snapshot(prices)
        self.assertFalse(snap.credit_stress)

    def test_duration_flight_triggered(self):
        # TLT up 5%: iloc[-6]=100.0 (prev), iloc[-1]=105.0 (curr) → ROC=5%, SPY flat → spread>3%
        prices = _all_prices(30)
        tlt_vals = [100.0] * 25 + [105.0] * 5
        prices["TLT"] = _make_series(tlt_vals)
        snap = _compute_snapshot(prices)
        self.assertTrue(snap.duration_flight)
        self.assertIsNotNone(snap.tlt_spy_spread_5d)

    def test_duration_flight_not_triggered(self):
        # TLT up 1%, SPY up 1% → spread = 0% ≤ 3%
        prices = _all_prices(30)
        tlt_vals = [100.0] * 6 + [101.0] * (30 - 6)
        spy_vals = [100.0] * 6 + [101.0] * (30 - 6)
        prices["TLT"] = _make_series(tlt_vals)
        prices["SPY"] = _make_series(spy_vals)
        snap = _compute_snapshot(prices)
        self.assertFalse(snap.duration_flight)

    def test_copper_gold_positive(self):
        # CPER up 5%, GLD flat → copper/gold ratio up ~5% > 0
        prices = _all_prices(30)
        cper_vals = [100.0] * 21 + [105.0] * (30 - 21)
        prices["CPER"] = _make_series(cper_vals)
        snap = _compute_snapshot(prices)
        self.assertTrue(snap.copper_gold_positive)

    def test_copper_gold_negative(self):
        # CPER down 5% → ratio negative
        prices = _all_prices(30)
        cper_vals = [100.0] * 21 + [95.0] * (30 - 21)
        prices["CPER"] = _make_series(cper_vals)
        snap = _compute_snapshot(prices)
        self.assertFalse(snap.copper_gold_positive)

    def test_usd_strong(self):
        # UUP up 3% over 20 days
        prices = _all_prices(30)
        uup_vals = [100.0] * 21 + [103.0] * (30 - 21)
        prices["UUP"] = _make_series(uup_vals)
        snap = _compute_snapshot(prices)
        self.assertTrue(snap.usd_strong)

    def test_usd_weak(self):
        # UUP up 0.5% — below 1% threshold
        prices = _all_prices(30)
        uup_vals = [100.0] * 21 + [100.5] * (30 - 21)
        prices["UUP"] = _make_series(uup_vals)
        snap = _compute_snapshot(prices)
        self.assertFalse(snap.usd_strong)

    def test_missing_tickers_handled(self):
        # Partial prices — TLT missing → tlt_spy_spread_5d is None
        prices = _all_prices()
        del prices["TLT"]
        snap = _compute_snapshot(prices)
        self.assertIsNone(snap.tlt_spy_spread_5d)
        self.assertFalse(snap.duration_flight)

    def test_all_tickers_missing(self):
        snap = _compute_snapshot({})
        self.assertIsNone(snap.credit_spread_roc)
        self.assertIsNone(snap.usd_trend_20d)
        self.assertFalse(snap.data_available)

    def test_hyg_ief_roc_computed(self):
        prices = _all_prices(30)
        # HYG up 2%: iloc[-11]=100.0 (prev), iloc[-1]=102.0 (curr) → ROC=2% > 0
        hyg_vals = [100.0] * 20 + [102.0] * 10
        prices["HYG"] = _make_series(hyg_vals)
        snap = _compute_snapshot(prices)
        self.assertIsNotNone(snap.hyg_ief_roc_10d)
        self.assertGreater(snap.hyg_ief_roc_10d, 0)


class TestZeroSnapshot(TestCase):
    def test_zero_snapshot_fields(self):
        z = _zero_snapshot()
        self.assertFalse(z.credit_stress)
        self.assertFalse(z.duration_flight)
        self.assertFalse(z.data_available)
        self.assertIsNone(z.credit_spread_roc)


class TestCacheIO(TestCase):
    def test_save_and_load(self):
        snap = MacroSnapshot(
            credit_spread_roc=-1.5,
            credit_stress=False,
            tlt_spy_spread_5d=2.0,
            duration_flight=False,
            copper_gold_trend_20d=1.0,
            copper_gold_positive=True,
            usd_trend_20d=0.5,
            usd_strong=False,
            hyg_ief_roc_10d=-0.3,
            data_available=True,
        )
        with (
            patch("data.macro_data._CACHE_PATH", "/tmp/_test_macro_cache.json"),
            patch("data.macro_data.today_et") as mock_today,
        ):
            mock_today.return_value = __import__("datetime").date(2026, 6, 4)
            _save_cache(snap)
            loaded = _load_cache()
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.credit_spread_roc, -1.5)
        self.assertFalse(loaded.credit_stress)
        self.assertTrue(loaded.data_available)
        # Cleanup
        with contextlib.suppress(FileNotFoundError):
            os.remove("/tmp/_test_macro_cache.json")

    def test_load_stale_cache(self):
        with (
            patch("data.macro_data._CACHE_PATH", "/tmp/_test_macro_stale.json"),
            patch("data.macro_data.today_et") as mock_today,
        ):
            # Save with yesterday's date
            mock_today.return_value = __import__("datetime").date(2026, 6, 3)
            _save_cache(_zero_snapshot())
            # Load with today's date → stale
            mock_today.return_value = __import__("datetime").date(2026, 6, 4)
            loaded = _load_cache()
        self.assertIsNone(loaded)
        with contextlib.suppress(FileNotFoundError):
            os.remove("/tmp/_test_macro_stale.json")

    def test_load_missing_cache(self):
        with patch("data.macro_data._CACHE_PATH", "/tmp/_nonexistent_macro.json"):
            result = _load_cache()
        self.assertIsNone(result)

    def test_load_corrupt_cache(self):
        with patch("data.macro_data._CACHE_PATH", "/tmp/_corrupt_macro.json"):
            with open("/tmp/_corrupt_macro.json", "w") as f:
                f.write("{not valid json")
            result = _load_cache()
        self.assertIsNone(result)
        with contextlib.suppress(FileNotFoundError):
            os.remove("/tmp/_corrupt_macro.json")

    def test_load_cache_bad_structure(self):
        """load_cache returns None when JSON is not a dict."""
        with patch("data.macro_data._CACHE_PATH", "/tmp/_bad_struct_macro.json"):
            with open("/tmp/_bad_struct_macro.json", "w") as f:
                json.dump([1, 2, 3], f)
            result = _load_cache()
        self.assertIsNone(result)
        with contextlib.suppress(FileNotFoundError):
            os.remove("/tmp/_bad_struct_macro.json")

    def test_save_oserror(self):
        with (
            patch("data.macro_data._CACHE_PATH", "/no_such_dir/x.json"),
            patch("data.macro_data.os.makedirs", side_effect=OSError("fail")),
        ):
            # Should not raise
            _save_cache(_zero_snapshot())


class TestDownloadMacroPrices(TestCase):
    def _make_multiindex_df(self, tickers, n=30) -> pd.DataFrame:
        """Return a synthetic MultiIndex yfinance download result."""

        dates = pd.date_range("2026-01-01", periods=n, freq="B")
        arrays = [
            ["Close"] * len(tickers),
            tickers,
        ]
        pd.MultiIndex.from_arrays(arrays)
        data = pd.DataFrame(
            {(t, s): [100.0 + i * 0.1 for i in range(n)] for t in ["Close"] for s in tickers},
            index=dates,
        )
        data.columns = pd.MultiIndex.from_tuples([(t, s) for t in ["Close"] for s in tickers])
        return data

    def test_download_success(self):
        tickers = ["HYG", "LQD", "IEF", "TLT", "CPER", "GLD", "UUP", "SPY"]
        fake_df = self._make_multiindex_df(tickers)
        with patch("data.macro_data.yf.download", return_value=fake_df):
            result = _download_macro_prices()
        self.assertIn("HYG", result)
        self.assertIn("SPY", result)

    def test_download_empty(self):
        with patch("data.macro_data.yf.download", return_value=pd.DataFrame()):
            result = _download_macro_prices()
        self.assertEqual(result, {})

    def test_download_exception(self):
        with patch("data.macro_data.yf.download", side_effect=RuntimeError("network")):
            result = _download_macro_prices()
        self.assertEqual(result, {})

    def test_download_none(self):
        with patch("data.macro_data.yf.download", return_value=None):
            result = _download_macro_prices()
        self.assertEqual(result, {})

    def test_short_series_excluded(self):
        """Tickers with < 22 bars are excluded from result."""
        tickers = ["HYG", "SPY"]
        # Only 10 bars — too short
        dates = pd.date_range("2026-01-01", periods=10, freq="B")
        df = pd.DataFrame({("Close", t): [100.0] * 10 for t in tickers}, index=dates)
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        with patch("data.macro_data.yf.download", return_value=df):
            result = _download_macro_prices()
        self.assertEqual(result, {})

    def test_download_no_close_key(self):
        """MultiIndex df with no 'Close' column → except KeyError → return {}."""
        tickers = ["HYG", "LQD"]
        dates = pd.date_range("2026-01-01", periods=30, freq="B")
        df = pd.DataFrame({("Open", t): [100.0] * 30 for t in tickers}, index=dates)
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        with patch("data.macro_data.yf.download", return_value=df):
            result = _download_macro_prices()
        self.assertEqual(result, {})

    def test_download_single_ticker_fallback(self):
        """With a single-element _TICKERS, non-MultiIndex df uses the elif branch."""
        import data.macro_data as macro_mod

        dates = pd.date_range("2026-01-01", periods=30, freq="B")
        df = pd.DataFrame({"Close": [100.0 + i * 0.1 for i in range(30)]}, index=dates)
        with (
            patch.object(macro_mod, "_TICKERS", ["HYG"]),
            patch("data.macro_data.yf.download", return_value=df),
        ):
            result = _download_macro_prices()
        self.assertIn("HYG", result)

    def test_download_single_ticker_short_series(self):
        """Single-ticker mode with < 22 bars → series excluded → empty dict (143->146)."""
        import data.macro_data as macro_mod

        dates = pd.date_range("2026-01-01", periods=10, freq="B")
        df = pd.DataFrame({"Close": [100.0] * 10}, index=dates)
        with (
            patch.object(macro_mod, "_TICKERS", ["HYG"]),
            patch("data.macro_data.yf.download", return_value=df),
        ):
            result = _download_macro_prices()
        self.assertEqual(result, {})

    def test_download_non_multiindex_multi_tickers(self):
        """Non-MultiIndex df with multiple tickers → elif False → empty result (141->146)."""
        dates = pd.date_range("2026-01-01", periods=30, freq="B")
        flat_df = pd.DataFrame({"Close": [100.0] * 30}, index=dates)
        with patch("data.macro_data.yf.download", return_value=flat_df):
            result = _download_macro_prices()
        self.assertEqual(result, {})


class TestGetMacroSnapshot(TestCase):
    def _make_multiindex_df(self, n=35):
        tickers = ["HYG", "LQD", "IEF", "TLT", "CPER", "GLD", "UUP", "SPY"]
        dates = pd.date_range("2026-01-01", periods=n, freq="B")
        df = pd.DataFrame({("Close", t): [100.0] * n for t in tickers}, index=dates)
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        return df

    def test_cache_hit(self):
        cached = MacroSnapshot(
            credit_spread_roc=0.0,
            credit_stress=False,
            tlt_spy_spread_5d=0.0,
            duration_flight=False,
            copper_gold_trend_20d=0.0,
            copper_gold_positive=False,
            usd_trend_20d=0.0,
            usd_strong=False,
            hyg_ief_roc_10d=0.0,
            data_available=True,
        )
        with patch("data.macro_data._load_cache", return_value=cached):
            result = get_macro_snapshot()
        self.assertIs(result, cached)

    def test_cache_miss_downloads(self):
        fake_df = self._make_multiindex_df()
        with (
            patch("data.macro_data._load_cache", return_value=None),
            patch("data.macro_data.yf.download", return_value=fake_df),
            patch("data.macro_data._save_cache"),
        ):
            result = get_macro_snapshot()
        self.assertIsInstance(result, MacroSnapshot)

    def test_download_fail_returns_zero(self):
        with (
            patch("data.macro_data._load_cache", return_value=None),
            patch("data.macro_data._download_macro_prices", return_value={}),
        ):
            result = get_macro_snapshot()
        self.assertFalse(result.data_available)

    def test_force_refresh_bypasses_cache(self):
        cached_snap = MacroSnapshot(
            credit_spread_roc=99.0,
            credit_stress=True,
            tlt_spy_spread_5d=99.0,
            duration_flight=True,
            copper_gold_trend_20d=99.0,
            copper_gold_positive=True,
            usd_trend_20d=99.0,
            usd_strong=True,
            hyg_ief_roc_10d=99.0,
            data_available=True,
        )
        fake_df = self._make_multiindex_df()
        with (
            patch("data.macro_data._load_cache", return_value=cached_snap),
            patch("data.macro_data.yf.download", return_value=fake_df),
            patch("data.macro_data._save_cache"),
        ):
            result = get_macro_snapshot(force_refresh=True)
        # Should NOT be the cached snapshot (force_refresh=True)
        self.assertIsNot(result, cached_snap)


class TestConvenienceGetters(TestCase):
    def _patch_snapshot(self, **kwargs):
        defaults = {
            "credit_spread_roc": None,
            "credit_stress": False,
            "tlt_spy_spread_5d": None,
            "duration_flight": False,
            "copper_gold_trend_20d": None,
            "copper_gold_positive": False,
            "usd_trend_20d": None,
            "usd_strong": False,
            "hyg_ief_roc_10d": None,
            "data_available": False,
        }
        defaults.update(kwargs)
        return MacroSnapshot(**defaults)

    def test_get_credit_stress_true(self):
        snap = self._patch_snapshot(credit_stress=True)
        with patch("data.macro_data.get_macro_snapshot", return_value=snap):
            self.assertTrue(get_credit_stress())

    def test_get_credit_stress_false(self):
        snap = self._patch_snapshot(credit_stress=False)
        with patch("data.macro_data.get_macro_snapshot", return_value=snap):
            self.assertFalse(get_credit_stress())

    def test_get_duration_flight_true(self):
        snap = self._patch_snapshot(duration_flight=True)
        with patch("data.macro_data.get_macro_snapshot", return_value=snap):
            self.assertTrue(get_duration_flight())

    def test_get_duration_flight_false(self):
        snap = self._patch_snapshot(duration_flight=False)
        with patch("data.macro_data.get_macro_snapshot", return_value=snap):
            self.assertFalse(get_duration_flight())

    def test_get_copper_gold_positive(self):
        snap = self._patch_snapshot(copper_gold_positive=True)
        with patch("data.macro_data.get_macro_snapshot", return_value=snap):
            self.assertTrue(get_copper_gold_positive())

    def test_get_copper_gold_negative(self):
        snap = self._patch_snapshot(copper_gold_positive=False)
        with patch("data.macro_data.get_macro_snapshot", return_value=snap):
            self.assertFalse(get_copper_gold_positive())

    def test_get_usd_strong(self):
        snap = self._patch_snapshot(usd_strong=True)
        with patch("data.macro_data.get_macro_snapshot", return_value=snap):
            self.assertTrue(get_usd_strong())

    def test_get_usd_weak(self):
        snap = self._patch_snapshot(usd_strong=False)
        with patch("data.macro_data.get_macro_snapshot", return_value=snap):
            self.assertFalse(get_usd_strong())
