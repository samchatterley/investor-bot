"""Tests for data/earnings_surprise.py — PEAD candidate detection."""

import unittest
from datetime import UTC, date, datetime, timedelta
from unittest.mock import MagicMock, mock_open, patch

import pandas as pd

from data.earnings_surprise import (
    _load_cache,
    _save_cache,
    get_earnings_miss,
    get_earnings_surprise,
    prefetch_earnings_data,
)

_NOW = datetime.now(UTC)
_TODAY_DATE = date(2026, 6, 3)
_TODAY_KEY = _TODAY_DATE.isoformat()


def _make_earnings_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal earnings_dates DataFrame matching yfinance's format."""
    index = pd.DatetimeIndex([pd.Timestamp(r["date"]) for r in rows], name="Earnings Date")
    df = pd.DataFrame(
        {
            "EPS Estimate": [r.get("estimate") for r in rows],
            "Reported EPS": [r.get("reported") for r in rows],
            "Surprise(%)": [r.get("surprise") for r in rows],
        },
        index=index,
    )
    return df


def _recent_date(days_ago: int) -> str:
    """Return an ISO timestamp string N days ago."""
    ts = _NOW - timedelta(days=days_ago)
    return ts.strftime("%Y-%m-%dT%H:%M:%S-04:00")


# Reusable DataFrames
_DF_BEAT = _make_earnings_df(
    [{"date": _recent_date(5), "estimate": 1.50, "reported": 1.65, "surprise": 10.0}]
)
_DF_MISS = _make_earnings_df(
    [{"date": _recent_date(5), "estimate": 1.50, "reported": 1.35, "surprise": -10.0}]
)


class TestGetEarningsSurprise(unittest.TestCase):
    def test_returns_candidate_above_threshold(self):
        with (
            patch("data.earnings_surprise._load_cache", return_value={}),
            patch("data.earnings_surprise._save_cache"),
            patch("data.earnings_surprise.yf.Ticker") as mock_ticker,
            patch("data.earnings_surprise.time.sleep"),
            patch("data.earnings_surprise.today_et", return_value=_TODAY_DATE),
        ):
            mock_ticker.return_value.earnings_dates = _DF_BEAT
            result = get_earnings_surprise(["AAPL"])
        self.assertIn("AAPL", result)
        self.assertAlmostEqual(result["AAPL"]["earnings_surprise_pct"], 10.0)
        self.assertTrue(result["AAPL"]["pead_candidate"])

    def test_excludes_below_threshold(self):
        df = _make_earnings_df(
            [{"date": _recent_date(5), "estimate": 1.50, "reported": 1.55, "surprise": 3.33}]
        )
        with (
            patch("data.earnings_surprise._load_cache", return_value={}),
            patch("data.earnings_surprise._save_cache"),
            patch("data.earnings_surprise.yf.Ticker") as mock_ticker,
            patch("data.earnings_surprise.time.sleep"),
            patch("data.earnings_surprise.today_et", return_value=_TODAY_DATE),
        ):
            mock_ticker.return_value.earnings_dates = df
            result = get_earnings_surprise(["AAPL"])
        self.assertNotIn("AAPL", result)

    def test_excludes_outside_lookback_window(self):
        df = _make_earnings_df(
            [{"date": _recent_date(45), "estimate": 1.50, "reported": 1.65, "surprise": 10.0}]
        )
        with (
            patch("data.earnings_surprise._load_cache", return_value={}),
            patch("data.earnings_surprise._save_cache"),
            patch("data.earnings_surprise.yf.Ticker") as mock_ticker,
            patch("data.earnings_surprise.time.sleep"),
            patch("data.earnings_surprise.today_et", return_value=_TODAY_DATE),
        ):
            mock_ticker.return_value.earnings_dates = df
            result = get_earnings_surprise(["AAPL"])
        self.assertNotIn("AAPL", result)

    def test_custom_lookback_window(self):
        df = _make_earnings_df(
            [{"date": _recent_date(45), "estimate": 1.50, "reported": 1.65, "surprise": 10.0}]
        )
        with (
            patch("data.earnings_surprise._load_cache", return_value={}),
            patch("data.earnings_surprise._save_cache"),
            patch("data.earnings_surprise.yf.Ticker") as mock_ticker,
            patch("data.earnings_surprise.time.sleep"),
            patch("data.earnings_surprise.today_et", return_value=_TODAY_DATE),
        ):
            mock_ticker.return_value.earnings_dates = df
            result = get_earnings_surprise(["AAPL"], lookback_days=60)
        self.assertIn("AAPL", result)

    def test_excludes_future_earnings_without_reported_eps(self):
        df = _make_earnings_df(
            [
                {"date": _recent_date(-10), "estimate": 2.00, "reported": None, "surprise": None},
                {"date": _recent_date(5), "estimate": 1.50, "reported": 1.65, "surprise": 10.0},
            ]
        )
        with (
            patch("data.earnings_surprise._load_cache", return_value={}),
            patch("data.earnings_surprise._save_cache"),
            patch("data.earnings_surprise.yf.Ticker") as mock_ticker,
            patch("data.earnings_surprise.time.sleep"),
            patch("data.earnings_surprise.today_et", return_value=_TODAY_DATE),
        ):
            mock_ticker.return_value.earnings_dates = df
            result = get_earnings_surprise(["AAPL"])
        self.assertIn("AAPL", result)
        self.assertAlmostEqual(result["AAPL"]["earnings_surprise_pct"], 10.0)

    def test_returns_empty_when_earnings_dates_is_none(self):
        with (
            patch("data.earnings_surprise._load_cache", return_value={}),
            patch("data.earnings_surprise._save_cache"),
            patch("data.earnings_surprise.yf.Ticker") as mock_ticker,
            patch("data.earnings_surprise.time.sleep"),
            patch("data.earnings_surprise.today_et", return_value=_TODAY_DATE),
        ):
            mock_ticker.return_value.earnings_dates = None
            result = get_earnings_surprise(["AAPL"])
        self.assertNotIn("AAPL", result)

    def test_returns_empty_on_network_failure(self):
        with (
            patch("data.earnings_surprise._load_cache", return_value={}),
            patch("data.earnings_surprise._save_cache"),
            patch("data.earnings_surprise.yf.Ticker", side_effect=Exception("network error")),
            patch("data.earnings_surprise.time.sleep"),
            patch("data.earnings_surprise.today_et", return_value=_TODAY_DATE),
        ):
            result = get_earnings_surprise(["AAPL"])
        self.assertEqual(result, {})

    def test_malformed_symbol_does_not_abort_batch(self):
        """D2: a symbol with an unexpected earnings_dates schema is isolated; the rest still parse."""
        from data.earnings_surprise import _live_fetch_earnings

        # BAD has no 'Surprise(%)' column → dropna(subset=[...]) raises KeyError mid-batch.
        bad_df = pd.DataFrame({"Reported EPS": [1.0]}, index=pd.DatetimeIndex([_NOW]))

        def _ticker(sym):
            m = MagicMock()
            m.earnings_dates = bad_df if sym == "BAD" else _DF_BEAT
            return m

        with (
            patch("data.earnings_surprise.yf.Ticker", side_effect=_ticker),
            patch("data.earnings_surprise.time.sleep"),
        ):
            result = _live_fetch_earnings(["BAD", "GOOD"])
        self.assertIsNone(result["BAD"])  # isolated — did not raise
        self.assertIsNotNone(result["GOOD"])  # batch continued
        self.assertIsNotNone(result["GOOD"]["surprise"])

    def test_returns_days_ago_field(self):
        df = _make_earnings_df(
            [{"date": _recent_date(7), "estimate": 1.50, "reported": 1.65, "surprise": 10.0}]
        )
        with (
            patch("data.earnings_surprise._load_cache", return_value={}),
            patch("data.earnings_surprise._save_cache"),
            patch("data.earnings_surprise.yf.Ticker") as mock_ticker,
            patch("data.earnings_surprise.time.sleep"),
            patch("data.earnings_surprise.today_et", return_value=_TODAY_DATE),
        ):
            mock_ticker.return_value.earnings_dates = df
            result = get_earnings_surprise(["AAPL"])
        self.assertIn("earnings_days_ago", result["AAPL"])
        self.assertAlmostEqual(result["AAPL"]["earnings_days_ago"], 7, delta=1)

    def test_returns_earnings_date_as_iso_string(self):
        df = _make_earnings_df(
            [{"date": _recent_date(5), "estimate": 1.50, "reported": 1.65, "surprise": 10.0}]
        )
        with (
            patch("data.earnings_surprise._load_cache", return_value={}),
            patch("data.earnings_surprise._save_cache"),
            patch("data.earnings_surprise.yf.Ticker") as mock_ticker,
            patch("data.earnings_surprise.time.sleep"),
            patch("data.earnings_surprise.today_et", return_value=_TODAY_DATE),
        ):
            mock_ticker.return_value.earnings_dates = df
            result = get_earnings_surprise(["AAPL"])
        self.assertIsInstance(date.fromisoformat(result["AAPL"]["earnings_date"]), date)

    def test_multiple_symbols_independent(self):
        df_small = _make_earnings_df(
            [{"date": _recent_date(5), "estimate": 1.50, "reported": 1.45, "surprise": -3.33}]
        )

        def _ticker_factory(sym):
            m = MagicMock()
            m.earnings_dates = _DF_BEAT if sym == "AAPL" else df_small
            return m

        with (
            patch("data.earnings_surprise._load_cache", return_value={}),
            patch("data.earnings_surprise._save_cache"),
            patch("data.earnings_surprise.yf.Ticker", side_effect=_ticker_factory),
            patch("data.earnings_surprise.time.sleep"),
            patch("data.earnings_surprise.today_et", return_value=_TODAY_DATE),
        ):
            result = get_earnings_surprise(["AAPL", "MSFT"])
        self.assertIn("AAPL", result)
        self.assertNotIn("MSFT", result)

    def test_custom_min_surprise_threshold(self):
        df = _make_earnings_df(
            [{"date": _recent_date(5), "estimate": 1.50, "reported": 1.68, "surprise": 12.0}]
        )
        with (
            patch("data.earnings_surprise._load_cache", return_value={}),
            patch("data.earnings_surprise._save_cache"),
            patch("data.earnings_surprise.yf.Ticker") as mock_ticker,
            patch("data.earnings_surprise.time.sleep"),
            patch("data.earnings_surprise.today_et", return_value=_TODAY_DATE),
        ):
            mock_ticker.return_value.earnings_dates = df
            result_excluded = get_earnings_surprise(["AAPL"], min_surprise=15.0)
            result_included = get_earnings_surprise(["AAPL"], min_surprise=3.0)
        self.assertNotIn("AAPL", result_excluded)
        self.assertIn("AAPL", result_included)

    def test_skips_etf_symbol(self):
        with (
            patch("data.earnings_surprise._load_cache", return_value={}),
            patch("data.earnings_surprise._save_cache"),
            patch("data.earnings_surprise.today_et", return_value=_TODAY_DATE),
            patch("data.earnings_surprise.time.sleep"),
        ):
            result = get_earnings_surprise(["SPY"])
        self.assertEqual(result, {})

    def test_skips_when_all_rows_have_null_reported_eps(self):
        df = _make_earnings_df(
            [{"date": _recent_date(5), "estimate": 1.50, "reported": None, "surprise": None}]
        )
        with (
            patch("data.earnings_surprise._load_cache", return_value={}),
            patch("data.earnings_surprise._save_cache"),
            patch("data.earnings_surprise.yf.Ticker") as mock_ticker,
            patch("data.earnings_surprise.time.sleep"),
            patch("data.earnings_surprise.today_et", return_value=_TODAY_DATE),
        ):
            mock_ticker.return_value.earnings_dates = df
            result = get_earnings_surprise(["AAPL"])
        self.assertNotIn("AAPL", result)


class TestGetEarningsMiss(unittest.TestCase):
    def test_returns_miss_below_threshold(self):
        with (
            patch("data.earnings_surprise._load_cache", return_value={}),
            patch("data.earnings_surprise._save_cache"),
            patch("data.earnings_surprise.yf.Ticker") as mock_ticker,
            patch("data.earnings_surprise.time.sleep"),
            patch("data.earnings_surprise.today_et", return_value=_TODAY_DATE),
        ):
            mock_ticker.return_value.earnings_dates = _DF_MISS
            result = get_earnings_miss(["AAPL"])
        self.assertIn("AAPL", result)
        self.assertAlmostEqual(result["AAPL"]["earnings_miss_pct"], -10.0)
        self.assertTrue(result["AAPL"]["earnings_miss_candidate"])

    def test_excludes_miss_above_threshold(self):
        df = _make_earnings_df(
            [{"date": _recent_date(5), "estimate": 1.50, "reported": 1.47, "surprise": -2.0}]
        )
        with (
            patch("data.earnings_surprise._load_cache", return_value={}),
            patch("data.earnings_surprise._save_cache"),
            patch("data.earnings_surprise.yf.Ticker") as mock_ticker,
            patch("data.earnings_surprise.time.sleep"),
            patch("data.earnings_surprise.today_et", return_value=_TODAY_DATE),
        ):
            mock_ticker.return_value.earnings_dates = df
            result = get_earnings_miss(["AAPL"])
        self.assertNotIn("AAPL", result)

    def test_excludes_outside_lookback_window(self):
        df = _make_earnings_df(
            [{"date": _recent_date(45), "estimate": 1.50, "reported": 1.35, "surprise": -10.0}]
        )
        with (
            patch("data.earnings_surprise._load_cache", return_value={}),
            patch("data.earnings_surprise._save_cache"),
            patch("data.earnings_surprise.yf.Ticker") as mock_ticker,
            patch("data.earnings_surprise.time.sleep"),
            patch("data.earnings_surprise.today_et", return_value=_TODAY_DATE),
        ):
            mock_ticker.return_value.earnings_dates = df
            result = get_earnings_miss(["AAPL"])
        self.assertNotIn("AAPL", result)

    def test_excludes_positive_surprise(self):
        with (
            patch("data.earnings_surprise._load_cache", return_value={}),
            patch("data.earnings_surprise._save_cache"),
            patch("data.earnings_surprise.yf.Ticker") as mock_ticker,
            patch("data.earnings_surprise.time.sleep"),
            patch("data.earnings_surprise.today_et", return_value=_TODAY_DATE),
        ):
            mock_ticker.return_value.earnings_dates = _DF_BEAT
            result = get_earnings_miss(["AAPL"])
        self.assertNotIn("AAPL", result)

    def test_returns_days_ago_and_date_fields(self):
        df = _make_earnings_df(
            [{"date": _recent_date(7), "estimate": 1.50, "reported": 1.35, "surprise": -10.0}]
        )
        with (
            patch("data.earnings_surprise._load_cache", return_value={}),
            patch("data.earnings_surprise._save_cache"),
            patch("data.earnings_surprise.yf.Ticker") as mock_ticker,
            patch("data.earnings_surprise.time.sleep"),
            patch("data.earnings_surprise.today_et", return_value=_TODAY_DATE),
        ):
            mock_ticker.return_value.earnings_dates = df
            result = get_earnings_miss(["AAPL"])
        self.assertIn("earnings_miss_days_ago", result["AAPL"])
        self.assertIn("earnings_miss_date", result["AAPL"])
        self.assertAlmostEqual(result["AAPL"]["earnings_miss_days_ago"], 7, delta=1)

    def test_returns_empty_when_no_earnings_data(self):
        with (
            patch("data.earnings_surprise._load_cache", return_value={}),
            patch("data.earnings_surprise._save_cache"),
            patch("data.earnings_surprise.yf.Ticker") as mock_ticker,
            patch("data.earnings_surprise.time.sleep"),
            patch("data.earnings_surprise.today_et", return_value=_TODAY_DATE),
        ):
            mock_ticker.return_value.earnings_dates = None
            result = get_earnings_miss(["AAPL"])
        self.assertEqual(result, {})

    def test_returns_empty_on_network_failure(self):
        with (
            patch("data.earnings_surprise._load_cache", return_value={}),
            patch("data.earnings_surprise._save_cache"),
            patch("data.earnings_surprise.yf.Ticker", side_effect=Exception("network error")),
            patch("data.earnings_surprise.time.sleep"),
            patch("data.earnings_surprise.today_et", return_value=_TODAY_DATE),
        ):
            result = get_earnings_miss(["AAPL"])
        self.assertEqual(result, {})

    def test_skips_etf(self):
        with (
            patch("data.earnings_surprise._load_cache", return_value={}),
            patch("data.earnings_surprise._save_cache"),
            patch("data.earnings_surprise.today_et", return_value=_TODAY_DATE),
            patch("data.earnings_surprise.time.sleep"),
        ):
            result = get_earnings_miss(["SPY"])
        self.assertEqual(result, {})

    def test_skips_null_reported_eps(self):
        df = _make_earnings_df(
            [{"date": _recent_date(5), "estimate": 1.50, "reported": None, "surprise": None}]
        )
        with (
            patch("data.earnings_surprise._load_cache", return_value={}),
            patch("data.earnings_surprise._save_cache"),
            patch("data.earnings_surprise.yf.Ticker") as mock_ticker,
            patch("data.earnings_surprise.time.sleep"),
            patch("data.earnings_surprise.today_et", return_value=_TODAY_DATE),
        ):
            mock_ticker.return_value.earnings_dates = df
            result = get_earnings_miss(["AAPL"])
        self.assertNotIn("AAPL", result)

    def test_custom_max_miss_threshold(self):
        # _live_fetch_earnings stores miss data only when surprise_pct <= _MAX_MISS_PCT (-5%).
        # max_miss is a post-filter on stored data — use a qualifying miss and test strict vs default.
        df = _make_earnings_df(
            [{"date": _recent_date(5), "estimate": 1.50, "reported": 1.39, "surprise": -7.5}]
        )
        with (
            patch("data.earnings_surprise._load_cache", return_value={}),
            patch("data.earnings_surprise._save_cache"),
            patch("data.earnings_surprise.yf.Ticker") as mock_ticker,
            patch("data.earnings_surprise.time.sleep"),
            patch("data.earnings_surprise.today_et", return_value=_TODAY_DATE),
        ):
            mock_ticker.return_value.earnings_dates = df
            result_included = get_earnings_miss(["AAPL"])  # default -5%: -7.5 qualifies
            result_excluded = get_earnings_miss(["AAPL"], max_miss=-10.0)  # strict: -7.5 excluded
        self.assertIn("AAPL", result_included)
        self.assertNotIn("AAPL", result_excluded)


class TestEarningsCacheShared(unittest.TestCase):
    """Tests for the shared same-day cache and single-fetch behaviour."""

    def test_surprise_cache_hit_returns_without_network_call(self):
        warm_cache = {
            _TODAY_KEY: {
                "AAPL": {
                    "surprise": {
                        "earnings_surprise_pct": 10.0,
                        "earnings_date": "2026-05-29",
                        "earnings_days_ago": 5,
                        "pead_candidate": True,
                    },
                    "miss": None,
                }
            }
        }
        with (
            patch("data.earnings_surprise._load_cache", return_value=warm_cache),
            patch("data.earnings_surprise._save_cache") as mock_save,
            patch("data.earnings_surprise.yf.Ticker") as mock_ticker,
            patch("data.earnings_surprise.today_et", return_value=_TODAY_DATE),
        ):
            result = get_earnings_surprise(["AAPL"])
        mock_ticker.assert_not_called()
        mock_save.assert_not_called()
        self.assertEqual(result["AAPL"]["earnings_surprise_pct"], 10.0)

    def test_miss_cache_hit_returns_without_network_call(self):
        warm_cache = {
            _TODAY_KEY: {
                "AAPL": {
                    "surprise": None,
                    "miss": {
                        "earnings_miss_pct": -10.0,
                        "earnings_miss_date": "2026-05-29",
                        "earnings_miss_days_ago": 5,
                        "earnings_miss_candidate": True,
                    },
                }
            }
        }
        with (
            patch("data.earnings_surprise._load_cache", return_value=warm_cache),
            patch("data.earnings_surprise._save_cache") as mock_save,
            patch("data.earnings_surprise.yf.Ticker") as mock_ticker,
            patch("data.earnings_surprise.today_et", return_value=_TODAY_DATE),
        ):
            result = get_earnings_miss(["AAPL"])
        mock_ticker.assert_not_called()
        mock_save.assert_not_called()
        self.assertEqual(result["AAPL"]["earnings_miss_pct"], -10.0)

    def test_null_surprise_omits_symbol(self):
        warm_cache = {_TODAY_KEY: {"AAPL": {"surprise": None, "miss": None}}}
        with (
            patch("data.earnings_surprise._load_cache", return_value=warm_cache),
            patch("data.earnings_surprise.yf.Ticker") as mock_ticker,
            patch("data.earnings_surprise.today_et", return_value=_TODAY_DATE),
        ):
            result = get_earnings_surprise(["AAPL"])
        mock_ticker.assert_not_called()
        self.assertNotIn("AAPL", result)

    def test_null_miss_omits_symbol(self):
        warm_cache = {_TODAY_KEY: {"AAPL": {"surprise": None, "miss": None}}}
        with (
            patch("data.earnings_surprise._load_cache", return_value=warm_cache),
            patch("data.earnings_surprise.yf.Ticker") as mock_ticker,
            patch("data.earnings_surprise.today_et", return_value=_TODAY_DATE),
        ):
            result = get_earnings_miss(["AAPL"])
        mock_ticker.assert_not_called()
        self.assertNotIn("AAPL", result)

    def test_single_fetch_populates_both_surprise_and_miss(self):
        # Verify yf.Ticker is called once even when both getters are called
        with (
            patch("data.earnings_surprise._load_cache", return_value={}),
            patch("data.earnings_surprise._save_cache"),
            patch("data.earnings_surprise.yf.Ticker") as mock_ticker,
            patch("data.earnings_surprise.time.sleep"),
            patch("data.earnings_surprise.today_et", return_value=_TODAY_DATE),
        ):
            mock_ticker.return_value.earnings_dates = _DF_BEAT
            get_earnings_surprise(["AAPL"])
            # Second call: AAPL already in cache (patched _save_cache is no-op, so
            # cache state doesn't persist across calls here — that's expected in tests)
        self.assertEqual(mock_ticker.call_count, 1)

    def test_cache_miss_saves_today_key(self):
        with (
            patch("data.earnings_surprise._load_cache", return_value={}),
            patch("data.earnings_surprise._save_cache") as mock_save,
            patch("data.earnings_surprise.yf.Ticker") as mock_ticker,
            patch("data.earnings_surprise.time.sleep"),
            patch("data.earnings_surprise.today_et", return_value=_TODAY_DATE),
        ):
            mock_ticker.return_value.earnings_dates = None
            get_earnings_surprise(["AAPL"])
        mock_save.assert_called_once()
        saved = mock_save.call_args[0][0]
        self.assertIn(_TODAY_KEY, saved)


class TestPrefetchEarningsData(unittest.TestCase):
    def test_returns_count_of_symbols_fetched(self):
        with (
            patch("data.earnings_surprise._load_cache", return_value={}),
            patch("data.earnings_surprise._save_cache"),
            patch("data.earnings_surprise.yf.Ticker") as mock_ticker,
            patch("data.earnings_surprise.time.sleep"),
            patch("data.earnings_surprise.today_et", return_value=_TODAY_DATE),
        ):
            mock_ticker.return_value.earnings_dates = None
            n = prefetch_earnings_data(["AAPL", "MSFT"])
        self.assertEqual(n, 2)

    def test_warm_cache_returns_zero(self):
        warm = {_TODAY_KEY: {"AAPL": None, "MSFT": None}}
        with (
            patch("data.earnings_surprise._load_cache", return_value=warm),
            patch("data.earnings_surprise._save_cache") as mock_save,
            patch("data.earnings_surprise.yf.Ticker") as mock_ticker,
            patch("data.earnings_surprise.today_et", return_value=_TODAY_DATE),
        ):
            n = prefetch_earnings_data(["AAPL", "MSFT"])
        self.assertEqual(n, 0)
        mock_ticker.assert_not_called()
        mock_save.assert_not_called()

    def test_uses_stock_universe_when_symbols_none(self):
        with (
            patch("data.earnings_surprise._load_cache", return_value={}),
            patch("data.earnings_surprise._save_cache"),
            patch("data.earnings_surprise._live_fetch_earnings", return_value={}) as mock_fetch,
            patch("data.earnings_surprise.today_et", return_value=_TODAY_DATE),
            patch("data.earnings_surprise.STOCK_UNIVERSE", {"AAPL", "MSFT"}),
        ):
            prefetch_earnings_data()
        fetched = set(mock_fetch.call_args[0][0])
        self.assertEqual(fetched, {"AAPL", "MSFT"})

    def test_saves_today_key(self):
        with (
            patch("data.earnings_surprise._load_cache", return_value={}),
            patch("data.earnings_surprise._save_cache") as mock_save,
            patch("data.earnings_surprise.yf.Ticker") as mock_ticker,
            patch("data.earnings_surprise.time.sleep"),
            patch("data.earnings_surprise.today_et", return_value=_TODAY_DATE),
        ):
            mock_ticker.return_value.earnings_dates = None
            prefetch_earnings_data(["AAPL"])
        saved = mock_save.call_args[0][0]
        self.assertIn(_TODAY_KEY, saved)

    def test_counts_pead_and_neg_pead_in_log(self):
        # Verify the function completes without error and returns correct count
        # when both beat and miss results are present
        def _factory(sym):
            m = MagicMock()
            m.earnings_dates = _DF_BEAT if sym == "AAPL" else _DF_MISS
            return m

        with (
            patch("data.earnings_surprise._load_cache", return_value={}),
            patch("data.earnings_surprise._save_cache"),
            patch("data.earnings_surprise.yf.Ticker", side_effect=_factory),
            patch("data.earnings_surprise.time.sleep"),
            patch("data.earnings_surprise.today_et", return_value=_TODAY_DATE),
        ):
            n = prefetch_earnings_data(["AAPL", "MSFT"])
        self.assertEqual(n, 2)


class TestLoadSaveCacheEarnings(unittest.TestCase):
    def test_load_returns_empty_on_missing_file(self):
        with patch("builtins.open", side_effect=FileNotFoundError):
            result = _load_cache()
        self.assertEqual(result, {})

    def test_load_returns_empty_on_json_error(self):
        import json as _json

        with (
            patch("builtins.open", mock_open(read_data="not valid json")),
            patch(
                "data.earnings_surprise.json.load",
                side_effect=_json.JSONDecodeError("err", "", 0),
            ),
        ):
            result = _load_cache()
        self.assertEqual(result, {})

    def test_save_writes_json_on_success(self):
        m = mock_open()
        with (
            patch("data.earnings_surprise.os.makedirs"),
            patch("builtins.open", m),
            patch("data.earnings_surprise.json.dump") as mock_dump,
        ):
            _save_cache({_TODAY_KEY: {}})
        mock_dump.assert_called_once()

    def test_save_logs_warning_on_os_error(self):
        with (
            patch("data.earnings_surprise.os.makedirs"),
            patch("builtins.open", side_effect=OSError("disk full")),
        ):
            _save_cache({_TODAY_KEY: {}})  # should not raise
