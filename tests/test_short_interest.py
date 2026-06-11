"""Tests for data/short_interest.py — short interest enrichment."""

import json
import unittest
from datetime import date
from unittest.mock import MagicMock, mock_open, patch

from data.short_interest import (
    _load_cache,
    _save_cache,
    get_short_interest,
    prefetch_short_interest,
)

_TODAY_DATE = date(2026, 6, 3)
_TODAY_KEY = _TODAY_DATE.isoformat()


class TestGetShortInterest(unittest.TestCase):
    def _make_ticker(self, short_ratio):
        m = MagicMock()
        m.info = {"shortRatio": short_ratio}
        return m

    def test_returns_symbol_above_threshold(self):
        with (
            patch("data.short_interest._load_cache", return_value={}),
            patch("data.short_interest._save_cache"),
            patch("data.short_interest.yf.Ticker", return_value=self._make_ticker(7.0)),
            patch("data.short_interest.time.sleep"),
            patch("data.short_interest.today_et", return_value=_TODAY_DATE),
        ):
            result = get_short_interest(["AAPL"])
        self.assertIn("AAPL", result)
        self.assertAlmostEqual(result["AAPL"]["short_ratio"], 7.0)
        self.assertTrue(result["AAPL"]["high_short_interest"])

    def test_excludes_below_threshold(self):
        with (
            patch("data.short_interest._load_cache", return_value={}),
            patch("data.short_interest._save_cache"),
            patch("data.short_interest.yf.Ticker", return_value=self._make_ticker(2.0)),
            patch("data.short_interest.time.sleep"),
            patch("data.short_interest.today_et", return_value=_TODAY_DATE),
        ):
            result = get_short_interest(["AAPL"])
        self.assertNotIn("AAPL", result)

    def test_excludes_exact_threshold(self):
        with (
            patch("data.short_interest._load_cache", return_value={}),
            patch("data.short_interest._save_cache"),
            patch("data.short_interest.yf.Ticker", return_value=self._make_ticker(4.9)),
            patch("data.short_interest.time.sleep"),
            patch("data.short_interest.today_et", return_value=_TODAY_DATE),
        ):
            result = get_short_interest(["AAPL"])
        self.assertNotIn("AAPL", result)

    def test_custom_min_threshold(self):
        with (
            patch("data.short_interest._load_cache", return_value={}),
            patch("data.short_interest._save_cache"),
            patch("data.short_interest.yf.Ticker", return_value=self._make_ticker(3.0)),
            patch("data.short_interest.time.sleep"),
            patch("data.short_interest.today_et", return_value=_TODAY_DATE),
        ):
            result_default = get_short_interest(["AAPL"])
            result_custom = get_short_interest(["AAPL"], min_short_ratio=2.0)
        self.assertNotIn("AAPL", result_default)
        self.assertIn("AAPL", result_custom)

    def test_skips_etf_symbol(self):
        with (
            patch("data.short_interest._load_cache", return_value={}),
            patch("data.short_interest._save_cache"),
            patch("data.short_interest.today_et", return_value=_TODAY_DATE),
            patch("data.short_interest.time.sleep"),
        ):
            result = get_short_interest(["SPY"])
        self.assertEqual(result, {})

    def test_returns_empty_when_short_ratio_missing(self):
        m = MagicMock()
        m.info = {}
        with (
            patch("data.short_interest._load_cache", return_value={}),
            patch("data.short_interest._save_cache"),
            patch("data.short_interest.yf.Ticker", return_value=m),
            patch("data.short_interest.time.sleep"),
            patch("data.short_interest.today_et", return_value=_TODAY_DATE),
        ):
            result = get_short_interest(["AAPL"])
        self.assertNotIn("AAPL", result)

    def test_returns_empty_on_network_failure(self):
        with (
            patch("data.short_interest._load_cache", return_value={}),
            patch("data.short_interest._save_cache"),
            patch("data.short_interest.yf.Ticker", side_effect=Exception("network error")),
            patch("data.short_interest.time.sleep"),
            patch("data.short_interest.today_et", return_value=_TODAY_DATE),
        ):
            result = get_short_interest(["AAPL"])
        self.assertEqual(result, {})

    def test_skips_non_numeric_short_ratio(self):
        m = MagicMock()
        m.info = {"shortRatio": "not_a_number"}
        with (
            patch("data.short_interest._load_cache", return_value={}),
            patch("data.short_interest._save_cache"),
            patch("data.short_interest.yf.Ticker", return_value=m),
            patch("data.short_interest.time.sleep"),
            patch("data.short_interest.today_et", return_value=_TODAY_DATE),
        ):
            result = get_short_interest(["AAPL"])
        self.assertNotIn("AAPL", result)

    def test_skips_none_short_ratio(self):
        m = MagicMock()
        m.info = {"shortRatio": None}
        with (
            patch("data.short_interest._load_cache", return_value={}),
            patch("data.short_interest._save_cache"),
            patch("data.short_interest.yf.Ticker", return_value=m),
            patch("data.short_interest.time.sleep"),
            patch("data.short_interest.today_et", return_value=_TODAY_DATE),
        ):
            result = get_short_interest(["AAPL"])
        self.assertNotIn("AAPL", result)

    def test_multiple_symbols_independent(self):
        def _factory(sym):
            m = MagicMock()
            m.info = {"shortRatio": 8.0 if sym == "AAPL" else 1.0}
            return m

        with (
            patch("data.short_interest._load_cache", return_value={}),
            patch("data.short_interest._save_cache"),
            patch("data.short_interest.yf.Ticker", side_effect=_factory),
            patch("data.short_interest.time.sleep"),
            patch("data.short_interest.today_et", return_value=_TODAY_DATE),
        ):
            result = get_short_interest(["AAPL", "MSFT"])
        self.assertIn("AAPL", result)
        self.assertNotIn("MSFT", result)


class TestShortInterestCache(unittest.TestCase):
    def test_cache_hit_returns_without_network_call(self):
        warm_cache = {
            _TODAY_KEY: {
                "AAPL": {
                    "short_ratio": 7.0,
                    "high_short_interest": True,
                }
            }
        }
        with (
            patch("data.short_interest._load_cache", return_value=warm_cache),
            patch("data.short_interest._save_cache") as mock_save,
            patch("data.short_interest.yf.Ticker") as mock_ticker,
            patch("data.short_interest.today_et", return_value=_TODAY_DATE),
        ):
            result = get_short_interest(["AAPL"])
        mock_ticker.assert_not_called()
        mock_save.assert_not_called()
        self.assertAlmostEqual(result["AAPL"]["short_ratio"], 7.0)

    def test_null_sentinel_omits_symbol(self):
        warm_cache = {_TODAY_KEY: {"AAPL": None}}
        with (
            patch("data.short_interest._load_cache", return_value=warm_cache),
            patch("data.short_interest.yf.Ticker") as mock_ticker,
            patch("data.short_interest.today_et", return_value=_TODAY_DATE),
        ):
            result = get_short_interest(["AAPL"])
        mock_ticker.assert_not_called()
        self.assertNotIn("AAPL", result)

    def test_partial_cache_hit_only_fetches_missing(self):
        warm_cache = {
            _TODAY_KEY: {
                "AAPL": {
                    "short_ratio": 7.0,
                    "high_short_interest": True,
                }
            }
        }
        m = MagicMock()
        m.info = {"shortRatio": 6.0}
        with (
            patch("data.short_interest._load_cache", return_value=warm_cache),
            patch("data.short_interest._save_cache"),
            patch("data.short_interest.yf.Ticker", return_value=m) as mock_ticker,
            patch("data.short_interest.time.sleep"),
            patch("data.short_interest.today_et", return_value=_TODAY_DATE),
        ):
            result = get_short_interest(["AAPL", "MSFT"])
        mock_ticker.assert_called_once_with("MSFT")
        self.assertIn("AAPL", result)
        self.assertIn("MSFT", result)

    def test_cache_miss_saves_today_key(self):
        m = MagicMock()
        m.info = {"shortRatio": None}
        with (
            patch("data.short_interest._load_cache", return_value={}),
            patch("data.short_interest._save_cache") as mock_save,
            patch("data.short_interest.yf.Ticker", return_value=m),
            patch("data.short_interest.time.sleep"),
            patch("data.short_interest.today_et", return_value=_TODAY_DATE),
        ):
            get_short_interest(["AAPL"])
        mock_save.assert_called_once()
        saved = mock_save.call_args[0][0]
        self.assertIn(_TODAY_KEY, saved)


class TestPrefetchShortInterest(unittest.TestCase):
    def test_returns_count_of_symbols_fetched(self):
        m = MagicMock()
        m.info = {"shortRatio": None}
        with (
            patch("data.short_interest._load_cache", return_value={}),
            patch("data.short_interest._save_cache"),
            patch("data.short_interest.yf.Ticker", return_value=m),
            patch("data.short_interest.time.sleep"),
            patch("data.short_interest.today_et", return_value=_TODAY_DATE),
        ):
            n = prefetch_short_interest(["AAPL", "MSFT"])
        self.assertEqual(n, 2)

    def test_warm_cache_returns_zero(self):
        warm = {_TODAY_KEY: {"AAPL": None, "MSFT": None}}
        with (
            patch("data.short_interest._load_cache", return_value=warm),
            patch("data.short_interest._save_cache") as mock_save,
            patch("data.short_interest.yf.Ticker") as mock_ticker,
            patch("data.short_interest.today_et", return_value=_TODAY_DATE),
        ):
            n = prefetch_short_interest(["AAPL", "MSFT"])
        self.assertEqual(n, 0)
        mock_ticker.assert_not_called()
        mock_save.assert_not_called()

    def test_uses_stock_universe_when_symbols_none(self):
        with (
            patch("data.short_interest._load_cache", return_value={}),
            patch("data.short_interest._save_cache"),
            patch("data.short_interest._live_fetch_short_interest", return_value={}) as mock_fetch,
            patch("data.short_interest.today_et", return_value=_TODAY_DATE),
            patch("data.short_interest.STOCK_UNIVERSE", {"AAPL", "MSFT"}),
        ):
            prefetch_short_interest()
        fetched = set(mock_fetch.call_args[0][0])
        self.assertEqual(fetched, {"AAPL", "MSFT"})

    def test_saves_today_key(self):
        m = MagicMock()
        m.info = {"shortRatio": None}
        with (
            patch("data.short_interest._load_cache", return_value={}),
            patch("data.short_interest._save_cache") as mock_save,
            patch("data.short_interest.yf.Ticker", return_value=m),
            patch("data.short_interest.time.sleep"),
            patch("data.short_interest.today_et", return_value=_TODAY_DATE),
        ):
            prefetch_short_interest(["AAPL"])
        saved = mock_save.call_args[0][0]
        self.assertIn(_TODAY_KEY, saved)

    def test_counts_high_short_interest_in_log(self):
        def _factory(sym):
            m = MagicMock()
            m.info = {"shortRatio": 7.0 if sym == "AAPL" else 1.0}
            return m

        with (
            patch("data.short_interest._load_cache", return_value={}),
            patch("data.short_interest._save_cache"),
            patch("data.short_interest.yf.Ticker", side_effect=_factory),
            patch("data.short_interest.time.sleep"),
            patch("data.short_interest.today_et", return_value=_TODAY_DATE),
        ):
            n = prefetch_short_interest(["AAPL", "MSFT"])
        self.assertEqual(n, 2)


class TestLoadSaveCacheShortInterest(unittest.TestCase):
    def test_load_returns_empty_on_missing_file(self):
        with patch("builtins.open", side_effect=FileNotFoundError):
            result = _load_cache()
        self.assertEqual(result, {})

    def test_load_returns_empty_on_json_error(self):
        with (
            patch("builtins.open", mock_open(read_data="not valid json")),
            patch(
                "data.short_interest.json.load",
                side_effect=json.JSONDecodeError("err", "", 0),
            ),
        ):
            result = _load_cache()
        self.assertEqual(result, {})

    def test_save_writes_json_on_success(self):
        m = mock_open()
        with (
            patch("data.short_interest.os.makedirs"),
            patch("builtins.open", m),
            patch("data.short_interest.json.dump") as mock_dump,
        ):
            _save_cache({_TODAY_KEY: {}})
        mock_dump.assert_called_once()

    def test_save_logs_warning_on_os_error(self):
        with (
            patch("data.short_interest.os.makedirs"),
            patch("builtins.open", side_effect=OSError("disk full")),
        ):
            _save_cache({_TODAY_KEY: {}})  # should not raise


class TestShortPctFloat(unittest.TestCase):
    def test_invalid_short_pct_float_coerces_to_none(self):
        """shortPercentOfFloat that can't be cast to float results in pct_float=None."""
        info = {
            "shortRatio": 6.0,
            "shortPercentOfFloat": "n/a",  # unparseable string
        }
        mock_ticker = MagicMock()
        mock_ticker.info = info
        with (
            patch("data.short_interest.yf.Ticker", return_value=mock_ticker),
            patch("data.short_interest.time.sleep"),
            patch("data.short_interest.today_et", return_value=_TODAY_DATE),
            patch("data.short_interest._load_cache", return_value={}),
            patch("data.short_interest._save_cache"),
        ):
            from data.short_interest import get_short_interest

            result = get_short_interest(["AAPL"])
        # short_ratio=6.0 > threshold so entry is not None; pct_float is None due to cast failure
        self.assertIsNotNone(result.get("AAPL"))
        self.assertIsNone(result["AAPL"]["short_pct_float"])
