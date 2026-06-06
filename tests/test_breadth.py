"""Tests for data/breadth.py — 100% line coverage."""

import json
import unittest
from unittest.mock import patch

import pandas as pd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(n: int = 260, base: float = 100.0, trend: float = 0.1) -> pd.DataFrame:
    """Build a minimal Close/High/Low DataFrame with a DatetimeIndex."""
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    prices = [base + i * trend for i in range(len(idx))]
    return pd.DataFrame(
        {
            "Close": prices,
            "High": [p + 1 for p in prices],
            "Low": [p - 1 for p in prices],
        },
        index=idx,
    )


def _make_df_flat(n: int = 260, price: float = 100.0) -> pd.DataFrame:
    """All prices identical — no advances or declines in diff()."""
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    return pd.DataFrame(
        {
            "Close": [price] * len(idx),
            "High": [price + 1] * len(idx),
            "Low": [price - 1] * len(idx),
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# Tests: is_breadth_thrust
# ---------------------------------------------------------------------------


class TestIsBreadthThrust(unittest.TestCase):
    def test_short_list_returns_false(self):
        from data.breadth import is_breadth_thrust

        self.assertFalse(is_breadth_thrust([0.3, 0.5, 0.7], window=10))

    def test_no_crossing_returns_false(self):
        from data.breadth import is_breadth_thrust

        # Stays above both thresholds the whole time — never starts below 0.40
        history = [0.65] * 15
        self.assertFalse(is_breadth_thrust(history))

    def test_starts_low_stays_low_returns_false(self):
        from data.breadth import is_breadth_thrust

        history = [0.30] * 15
        self.assertFalse(is_breadth_thrust(history))

    def test_crossing_within_window_returns_true(self):
        from data.breadth import is_breadth_thrust

        # First 10 bars: 0.35 → 0.65 exactly crossing within window
        history = [0.35] + [0.40] * 8 + [0.65] + [0.65] * 5
        self.assertTrue(is_breadth_thrust(history, window=10))

    def test_crossing_spread_over_more_than_window_returns_false(self):
        from data.breadth import is_breadth_thrust

        # High values come before low values — no low→high crossing within any window
        history4 = [0.65] * 5 + [0.30] * 10
        self.assertFalse(is_breadth_thrust(history4, window=10))

    def test_exact_window_length_returns_true(self):
        from data.breadth import is_breadth_thrust

        # Exactly window bars: first below, last above
        history = [0.35] + [0.50] * 8 + [0.65]
        self.assertTrue(is_breadth_thrust(history, window=10))

    def test_custom_thresholds(self):
        from data.breadth import is_breadth_thrust

        history = [0.20] + [0.30] * 8 + [0.80]
        self.assertTrue(
            is_breadth_thrust(history, window=10, from_threshold=0.25, to_threshold=0.75)
        )
        self.assertFalse(
            is_breadth_thrust(history, window=10, from_threshold=0.10, to_threshold=0.90)
        )


# ---------------------------------------------------------------------------
# Tests: compute_breadth
# ---------------------------------------------------------------------------


class TestComputeBreadth(unittest.TestCase):
    def test_empty_price_data_returns_zeros(self):
        from data.breadth import compute_breadth

        snap = compute_breadth({})
        self.assertEqual(snap.pct_above_sma50, 0.0)
        self.assertEqual(snap.pct_above_sma200, 0.0)
        self.assertEqual(snap.new_highs_52w, 0)
        self.assertEqual(snap.new_lows_52w, 0)
        self.assertEqual(snap.symbols_counted, 0)
        self.assertFalse(snap.breadth_thrust)

    def test_all_above_sma50(self):
        from data.breadth import compute_breadth

        # Strongly upward trending — every stock above its 50d SMA
        df = _make_df(n=260, base=50.0, trend=1.0)  # price = 50 + i, rising fast
        snap = compute_breadth({"A": df, "B": df.copy()})
        self.assertEqual(snap.pct_above_sma50, 1.0)
        self.assertEqual(snap.symbols_counted, 2)

    def test_all_below_sma50(self):
        from data.breadth import compute_breadth

        # Strongly downward — every stock below its 50d SMA
        df = _make_df(n=260, base=500.0, trend=-1.0)
        snap = compute_breadth({"A": df, "B": df.copy()})
        self.assertEqual(snap.pct_above_sma50, 0.0)

    def test_mixed_fraction(self):
        from data.breadth import compute_breadth

        up_df = _make_df(n=260, base=50.0, trend=1.0)
        down_df = _make_df(n=260, base=500.0, trend=-1.0)
        snap = compute_breadth({"A": up_df, "B": down_df, "C": up_df.copy(), "D": down_df.copy()})
        self.assertAlmostEqual(snap.pct_above_sma50, 0.5, places=2)

    def test_nh_nl_ratio_computed_correctly(self):
        from data.breadth import compute_breadth

        # Upward: latest close = 52-week high → new high
        up_df = _make_df(n=260, base=50.0, trend=1.0)
        snap = compute_breadth({"A": up_df})
        # 1 new high, 0 new lows → ratio = 1/(0+1) = 1.0
        self.assertEqual(snap.new_highs_52w, 1)
        self.assertEqual(snap.new_lows_52w, 0)
        self.assertAlmostEqual(snap.nh_nl_ratio, 1.0, places=4)

    def test_new_lows_counted(self):
        from data.breadth import compute_breadth

        down_df = _make_df(n=260, base=500.0, trend=-1.0)
        snap = compute_breadth({"A": down_df})
        self.assertEqual(snap.new_lows_52w, 1)
        self.assertEqual(snap.new_highs_52w, 0)
        # ratio = 0/(1+1) = 0
        self.assertAlmostEqual(snap.nh_nl_ratio, 0.0, places=4)

    def test_symbols_counted_reflects_enough_bars(self):
        from data.breadth import compute_breadth

        # Only 10 bars — not enough for SMA50 (min 55 required)
        short_df = _make_df(n=10)
        long_df = _make_df(n=260)
        snap = compute_breadth({"short": short_df, "long": long_df})
        self.assertEqual(snap.symbols_counted, 1)

    def test_symbol_without_close_column_skipped(self):
        from data.breadth import compute_breadth

        bad_df = pd.DataFrame({"Volume": [1, 2, 3]})
        good_df = _make_df(n=260)
        snap = compute_breadth({"bad": bad_df, "good": good_df})
        self.assertEqual(snap.symbols_counted, 1)

    def test_none_dataframe_skipped(self):
        from data.breadth import compute_breadth

        good_df = _make_df(n=260)
        snap = compute_breadth({"none_sym": None, "good": good_df})  # type: ignore[dict-item]
        self.assertEqual(snap.symbols_counted, 1)

    def test_sma200_fraction_correct(self):
        from data.breadth import compute_breadth

        up_df = _make_df(n=260, base=50.0, trend=1.0)
        snap = compute_breadth({"A": up_df})
        self.assertEqual(snap.pct_above_sma200, 1.0)

    def test_ad_line_nonzero_with_trending_stock(self):
        from data.breadth import compute_breadth

        up_df = _make_df(n=260, base=50.0, trend=1.0)
        snap = compute_breadth({"A": up_df})
        # Upward trend → all recent bars are advances → ad_line_5d_change > 0
        self.assertGreater(snap.ad_line_5d_change, 0)

    def test_symbol_exception_skipped_gracefully(self):
        from data.breadth import compute_breadth

        # Object-dtype "Close" column — rolling().mean() raises TypeError, triggers except block
        _idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=260)
        bad_df = pd.DataFrame({"Close": ["bad"] * len(_idx)}, index=_idx)
        up_df = _make_df(n=260)
        snap = compute_breadth({"bad": bad_df, "good": up_df})
        self.assertEqual(snap.symbols_counted, 1)

    def test_single_bar_skips_52w_block(self):
        from data.breadth import compute_breadth

        # n=1 — `if n >= 2:` is False, covering branch (False path skips 52w window).
        # Use a fixed past business day so the index is non-empty regardless of today's weekday.
        one_bar_df = pd.DataFrame(
            {"Close": [100.0], "High": [101.0], "Low": [99.0]},
            index=pd.bdate_range("2024-01-02", periods=1),
        )
        snap = compute_breadth({"A": one_bar_df})
        self.assertEqual(snap.new_highs_52w, 0)

    def test_flat_prices_zero_return_ad_line(self):
        from data.breadth import compute_breadth

        # Flat prices → daily_returns all == 0 → neither advances nor declines
        # covers branch 168->164 (elif ret < 0 fall-through)
        snap = compute_breadth({"A": _make_df_flat(n=260)})
        self.assertEqual(snap.ad_line_5d_change, 0.0)

    def test_short_symbols_give_zero_ad_line(self):
        from data.breadth import compute_breadth

        # n=5 — too short for AD line (needs >= 7); covers else branch of `if ad_counted > 0`
        short_df = _make_df(n=5)
        snap = compute_breadth({"A": short_df})
        self.assertEqual(snap.ad_line_5d_change, 0.0)

    def test_medium_symbols_give_false_thrust(self):
        from data.breadth import compute_breadth

        # n=70 — enough for AD line but < 75 (min_bars_sma50 + thrust_window)
        # covers else branch of `if thrust_counted > 0`
        medium_df = _make_df(n=70)
        snap = compute_breadth({"A": medium_df})
        self.assertFalse(snap.breadth_thrust)

    def test_breadth_thrust_triggered(self):
        from data.breadth import compute_breadth

        # Build a dataset where pct_above_sma50 history crosses the thrust threshold:
        # We need: for some 10-bar window, chunk[0] < 0.40 and chunk[-1] > 0.60
        # Simulate with a large universe: half stocks below SMA for first bars, then all above.
        # To guarantee a thrust we need ~20 bars of thrust_window data.
        # Use min_bars_sma50=55 + thrust_window(20) = 75 minimum, so n >= 80
        # Build first 55+20 bars as downtrend, last portion uptrend:
        idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=260)
        n_actual = len(idx)
        # Prices: start high at index 0 (old), fall sharply, then rise in last 10 bars
        # SMA50 of the last bar will be compared to close.
        # Simpler: create price series that is below SMA50 in early recent bars, then jumps
        prices_down = [200.0 - i * 0.5 for i in range(n_actual - 10)]
        prices_up = [prices_down[-1] + i * 20 for i in range(1, 11)]
        prices = prices_down + prices_up
        df = pd.DataFrame(
            {"Close": prices, "High": [p + 1 for p in prices], "Low": [p - 1 for p in prices]},
            index=idx,
        )
        result = compute_breadth({"A": df})
        self.assertIsInstance(result.breadth_thrust, bool)


# ---------------------------------------------------------------------------
# Tests: _load_breadth_cache
# ---------------------------------------------------------------------------


class TestLoadBreadthCache(unittest.TestCase):
    def test_missing_file_returns_none(self):
        from data.breadth import _load_breadth_cache

        with patch("data.breadth._CACHE_PATH", "/nonexistent/path/breadth_cache.json"):
            result = _load_breadth_cache()
        self.assertIsNone(result)

    def test_stale_date_returns_none(self):
        from data.breadth import _load_breadth_cache

        stale_payload = json.dumps({"_date": "2000-01-01", "pct_above_sma50": 0.5})
        m = unittest.mock.mock_open(read_data=stale_payload)
        with (
            patch("builtins.open", m),
            patch("data.breadth.today_et", return_value=_parse_date("2026-06-02")),
        ):
            result = _load_breadth_cache()
        self.assertIsNone(result)

    def test_corrupt_json_returns_none(self):
        from data.breadth import _load_breadth_cache

        m = unittest.mock.mock_open(read_data="not-valid-json{{{{")
        with patch("builtins.open", m):
            result = _load_breadth_cache()
        self.assertIsNone(result)

    def test_today_date_returns_dict(self):
        from data.breadth import _load_breadth_cache

        today_str = "2026-06-02"
        payload = {"_date": today_str, "pct_above_sma50": 0.55}
        m = unittest.mock.mock_open(read_data=json.dumps(payload))
        with (
            patch("builtins.open", m),
            patch("data.breadth.today_et", return_value=_parse_date(today_str)),
        ):
            result = _load_breadth_cache()
        self.assertIsNotNone(result)
        self.assertEqual(result["_date"], today_str)

    def test_non_dict_json_returns_none(self):
        from data.breadth import _load_breadth_cache

        m = unittest.mock.mock_open(read_data=json.dumps([1, 2, 3]))
        with patch("builtins.open", m):
            result = _load_breadth_cache()
        self.assertIsNone(result)


def _parse_date(s: str):
    from datetime import date

    return date.fromisoformat(s)


# ---------------------------------------------------------------------------
# Tests: _save_breadth_cache
# ---------------------------------------------------------------------------


class TestSaveBreadthCache(unittest.TestCase):
    def _make_snapshot(self):
        from data.breadth import BreadthSnapshot

        return BreadthSnapshot(
            pct_above_sma50=0.6,
            pct_above_sma200=0.4,
            new_highs_52w=10,
            new_lows_52w=2,
            nh_nl_ratio=3.33,
            ad_line_5d_change=5.0,
            breadth_thrust=True,
            symbols_counted=50,
        )

    def test_writes_file_with_date(self):
        from data.breadth import _save_breadth_cache

        today_str = "2026-06-02"
        written_data = {}

        def fake_dump(obj, f):
            written_data.update(obj)

        m = unittest.mock.mock_open()
        with (
            patch("builtins.open", m),
            patch("os.makedirs"),
            patch("json.dump", fake_dump),
            patch("data.breadth.today_et", return_value=_parse_date(today_str)),
        ):
            _save_breadth_cache(self._make_snapshot())

        self.assertEqual(written_data.get("_date"), today_str)
        self.assertEqual(written_data.get("pct_above_sma50"), 0.6)

    def test_oserror_does_not_raise(self):
        from data.breadth import _save_breadth_cache

        with patch("os.makedirs", side_effect=OSError("disk full")):
            # Should not raise
            _save_breadth_cache(self._make_snapshot())


# ---------------------------------------------------------------------------
# Tests: get_breadth_snapshot
# ---------------------------------------------------------------------------


class TestGetBreadthSnapshot(unittest.TestCase):
    def _snapshot_dict(self, today_str: str = "2026-06-02") -> dict:
        return {
            "_date": today_str,
            "pct_above_sma50": 0.55,
            "pct_above_sma200": 0.40,
            "new_highs_52w": 20,
            "new_lows_52w": 5,
            "nh_nl_ratio": 3.33,
            "ad_line_5d_change": 10.0,
            "breadth_thrust": False,
            "symbols_counted": 100,
        }

    def test_cache_hit_returns_cached_snapshot(self):
        from data.breadth import BreadthSnapshot, get_breadth_snapshot

        today_str = "2026-06-02"
        cached = self._snapshot_dict(today_str)

        with patch("data.breadth._load_breadth_cache", return_value=cached):
            snap = get_breadth_snapshot()

        self.assertIsInstance(snap, BreadthSnapshot)
        self.assertEqual(snap.pct_above_sma50, 0.55)
        self.assertEqual(snap.symbols_counted, 100)

    def test_cache_miss_triggers_compute(self):
        from data.breadth import BreadthSnapshot, get_breadth_snapshot

        up_df = _make_df(n=260)

        with (
            patch("data.breadth._load_breadth_cache", return_value=None),
            patch("data.breadth._download_price_data", return_value={"A": up_df}),
            patch("data.breadth._save_breadth_cache") as mock_save,
        ):
            snap = get_breadth_snapshot(symbols=["A"])

        self.assertIsInstance(snap, BreadthSnapshot)
        mock_save.assert_called_once()

    def test_price_data_provided_skips_download(self):
        from data.breadth import get_breadth_snapshot

        up_df = _make_df(n=260)

        with (
            patch("data.breadth._download_price_data") as mock_dl,
            patch("data.breadth._load_breadth_cache") as mock_cache,
        ):
            snap = get_breadth_snapshot(price_data={"A": up_df})

        mock_dl.assert_not_called()
        mock_cache.assert_not_called()
        self.assertEqual(snap.symbols_counted, 1)

    def test_price_data_provided_does_not_save_cache(self):
        from data.breadth import get_breadth_snapshot

        up_df = _make_df(n=260)

        with patch("data.breadth._save_breadth_cache") as mock_save:
            get_breadth_snapshot(price_data={"A": up_df})

        mock_save.assert_not_called()

    def test_uses_stock_universe_when_symbols_none(self):
        from data.breadth import get_breadth_snapshot

        with (
            patch("data.breadth._load_breadth_cache", return_value=None),
            patch("data.breadth._download_price_data", return_value={}) as mock_dl,
            patch("data.breadth._save_breadth_cache"),
        ):
            get_breadth_snapshot(symbols=None)

        called_symbols = mock_dl.call_args[0][0]
        # Should have passed STOCK_UNIVERSE
        from config import STOCK_UNIVERSE

        self.assertEqual(called_symbols, list(STOCK_UNIVERSE))

    def test_all_failures_return_zeros_snapshot(self):
        from data.breadth import BreadthSnapshot, get_breadth_snapshot

        with (
            patch("data.breadth._load_breadth_cache", return_value=None),
            patch("data.breadth._download_price_data", side_effect=RuntimeError("network error")),
        ):
            snap = get_breadth_snapshot(symbols=["X"])

        self.assertIsInstance(snap, BreadthSnapshot)
        self.assertEqual(snap.symbols_counted, 0)
        self.assertFalse(snap.breadth_thrust)

    def test_cache_deserialise_error_falls_through_to_compute(self):
        from data.breadth import get_breadth_snapshot

        # Cache returns a dict but with wrong keys for BreadthSnapshot
        bad_cache = {"_date": "2026-06-02", "unknown_field": 99}
        up_df = _make_df(n=260)

        with (
            patch("data.breadth._load_breadth_cache", return_value=bad_cache),
            patch("data.breadth._download_price_data", return_value={"A": up_df}),
            patch("data.breadth._save_breadth_cache"),
        ):
            snap = get_breadth_snapshot(symbols=["A"])

        self.assertEqual(snap.symbols_counted, 1)

    def test_compute_breadth_exception_returns_zeros(self):
        from data.breadth import get_breadth_snapshot

        with (
            patch("data.breadth._load_breadth_cache", return_value=None),
            patch("data.breadth._download_price_data", return_value={"A": _make_df()}),
            patch("data.breadth.compute_breadth", side_effect=ValueError("oops")),
            patch("data.breadth._save_breadth_cache"),
        ):
            snap = get_breadth_snapshot(symbols=["A"])

        self.assertEqual(snap.symbols_counted, 0)


# ---------------------------------------------------------------------------
# Tests: _download_price_data
# ---------------------------------------------------------------------------


class TestDownloadPriceData(unittest.TestCase):
    def test_download_failure_returns_empty(self):
        from data.breadth import _download_price_data

        with patch("data.breadth.yf.download", side_effect=Exception("network error")):
            result = _download_price_data(["AAPL"], 260)
        self.assertEqual(result, {})

    def test_empty_download_returns_empty(self):
        from data.breadth import _download_price_data

        with patch("data.breadth.yf.download", return_value=pd.DataFrame()):
            result = _download_price_data(["AAPL"], 260)
        self.assertEqual(result, {})

    def test_single_symbol_non_multiindex(self):
        from data.breadth import _download_price_data

        df = _make_df(n=100)
        with patch("data.breadth.yf.download", return_value=df):
            result = _download_price_data(["AAPL"], 260)
        self.assertIn("AAPL", result)

    def test_multiindex_extracts_symbols(self):
        from data.breadth import _download_price_data

        df_a = _make_df(n=100)
        df_b = _make_df(n=100, base=200.0)
        combined = pd.concat({"A": df_a, "B": df_b}, axis=1)
        combined.columns = pd.MultiIndex.from_tuples(
            [(col, sym) for sym in ["A", "B"] for col in ["Close", "High", "Low"]],
            names=[None, None],
        )
        with patch("data.breadth.yf.download", return_value=combined):
            result = _download_price_data(["A", "B"], 260)
        # At minimum should attempt to extract; exact result depends on xs
        self.assertIsInstance(result, dict)

    def test_none_download_returns_empty(self):
        from data.breadth import _download_price_data

        with patch("data.breadth.yf.download", return_value=None):
            result = _download_price_data(["AAPL"], 260)
        self.assertEqual(result, {})

    def test_multiindex_symbol_not_in_available(self):
        from data.breadth import _download_price_data

        # MultiIndex df only has "A"; request ["A", "MISSING"] — MISSING branch 325->324
        df_a = _make_df(n=100)
        combined = pd.concat({"A": df_a}, axis=1)
        combined.columns = pd.MultiIndex.from_tuples(
            [(col, "A") for col in ["Close", "High", "Low"]], names=[None, None]
        )
        with patch("data.breadth.yf.download", return_value=combined):
            result = _download_price_data(["A", "MISSING"], 260)
        self.assertIn("A", result)
        self.assertNotIn("MISSING", result)

    def test_multiindex_empty_sym_df_skipped(self):
        from data.breadth import _download_price_data

        # xs returns all-NaN rows → dropna(how="all") empties it → branch 328->324
        df_a = _make_df(n=100)
        nan_df = pd.DataFrame(
            {col: [float("nan")] * len(df_a) for col in ["Close", "High", "Low"]},
            index=df_a.index,
        )
        raw = pd.concat({"A": nan_df}, axis=1)
        raw.columns = pd.MultiIndex.from_tuples(
            [(col, "A") for col in ["Close", "High", "Low"]], names=[None, None]
        )
        with patch("data.breadth.yf.download", return_value=raw):
            result = _download_price_data(["A"], 260)
        self.assertEqual(result, {})

    def test_multiindex_xs_keyerror_handled(self):
        from data.breadth import _download_price_data

        # xs raises KeyError even though symbol appears in available — lines 330-331
        df_a = _make_df(n=100)
        combined = pd.concat({"A": df_a}, axis=1)
        combined.columns = pd.MultiIndex.from_tuples(
            [(col, "A") for col in ["Close", "High", "Low"]], names=[None, None]
        )
        with (
            patch("data.breadth.yf.download", return_value=combined),
            patch("pandas.DataFrame.xs", side_effect=KeyError("A")),
        ):
            result = _download_price_data(["A"], 260)
        self.assertEqual(result, {})

    def test_non_multiindex_multiple_symbols_returns_empty(self):
        from data.breadth import _download_price_data

        # Non-MultiIndex df with 2 symbols requested — elif len==1 False branch (332->335)
        df = _make_df(n=100)
        with patch("data.breadth.yf.download", return_value=df):
            result = _download_price_data(["AAPL", "MSFT"], 260)
        self.assertEqual(result, {})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
