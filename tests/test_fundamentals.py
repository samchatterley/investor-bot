"""Tests for data/fundamentals.py — Finnhub fundamentals and analyst consensus."""

import json
import os
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _stale_iso() -> str:
    return (datetime.now(UTC) - timedelta(hours=25)).isoformat()


# ---------------------------------------------------------------------------
# _load_cache
# ---------------------------------------------------------------------------


class TestLoadCache(unittest.TestCase):
    def test_returns_empty_dict_on_corrupt_json(self):
        """json.JSONDecodeError (e.g. truncated file) → returns {}."""
        from data.fundamentals import _load_cache

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{ corrupt json !!!")
            path = f.name
        try:
            result = _load_cache(path)
        finally:
            os.unlink(path)
        self.assertEqual(result, {})

    def test_returns_empty_dict_on_missing_file(self):
        from data.fundamentals import _load_cache

        result = _load_cache("/tmp/does_not_exist_fund_test.json")
        self.assertEqual(result, {})


# ---------------------------------------------------------------------------
# _save_cache
# ---------------------------------------------------------------------------


class TestSaveCache(unittest.TestCase):
    def test_writes_json_to_disk(self):
        from data.fundamentals import _save_cache

        data = {"AAPL": {"fetched_at": _now_iso(), "data": {"roe": 0.5}}}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cache.json")
            with patch("data.fundamentals.LOG_DIR", tmpdir):
                _save_cache(path, data)
            with open(path) as f:
                loaded = json.load(f)
        self.assertEqual(loaded["AAPL"]["data"]["roe"], 0.5)


# ---------------------------------------------------------------------------
# _is_stale
# ---------------------------------------------------------------------------


class TestIsStale(unittest.TestCase):
    def test_returns_true_when_fetched_at_missing(self):
        from data.fundamentals import _is_stale

        self.assertTrue(_is_stale({}))

    def test_returns_true_when_fetched_at_invalid(self):
        from data.fundamentals import _is_stale

        self.assertTrue(_is_stale({"fetched_at": "not-a-date"}))


# ---------------------------------------------------------------------------
# _get
# ---------------------------------------------------------------------------


class TestGet(unittest.TestCase):
    def test_returns_none_when_no_api_key(self):
        from data.fundamentals import _get

        with patch("data.fundamentals.FINNHUB_API_KEY", ""):
            result = _get("/stock/metric", {"symbol": "AAPL"})
        self.assertIsNone(result)

    def test_returns_none_on_request_exception(self):
        from data.fundamentals import _get

        with (
            patch("data.fundamentals.FINNHUB_API_KEY", "testkey"),
            patch("data.fundamentals.requests.get", side_effect=Exception("timeout")),
        ):
            result = _get("/stock/metric", {"symbol": "AAPL"})
        self.assertIsNone(result)

    def test_returns_none_on_http_error(self):
        from data.fundamentals import _get

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("403 Forbidden")
        with (
            patch("data.fundamentals.FINNHUB_API_KEY", "testkey"),
            patch("data.fundamentals.requests.get", return_value=mock_resp),
        ):
            result = _get("/stock/metric", {"symbol": "AAPL"})
        self.assertIsNone(result)

    def test_returns_json_on_success(self):
        from data.fundamentals import _get

        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"metric": {"roeTTM": 50.0}}
        with (
            patch("data.fundamentals.FINNHUB_API_KEY", "testkey"),
            patch("data.fundamentals.requests.get", return_value=mock_resp),
        ):
            result = _get("/stock/metric", {"symbol": "AAPL"})
        self.assertEqual(result, {"metric": {"roeTTM": 50.0}})


# ---------------------------------------------------------------------------
# _fetch_ratios
# ---------------------------------------------------------------------------


class TestFetchRatios(unittest.TestCase):
    def _call(self, raw):
        from data.fundamentals import _fetch_ratios

        with patch("data.fundamentals._get", return_value=raw):
            return _fetch_ratios("AAPL")

    def test_extracts_and_converts_fields(self):
        raw = {
            "metric": {
                "roeTTM": 146.69,
                "netProfitMarginTTM": 27.15,
                "totalDebt/totalEquityAnnual": 1.35,
                "currentRatioAnnual": 0.89,
            }
        }
        sym, data = self._call(raw)
        self.assertEqual(sym, "AAPL")
        # ROE and margin are divided by 100
        self.assertAlmostEqual(data["roe"], 1.4669, places=4)
        self.assertAlmostEqual(data["profit_margin"], 0.2715, places=4)
        # D/E and current ratio pass through unchanged
        self.assertAlmostEqual(data["debt_to_equity"], 1.35)
        self.assertAlmostEqual(data["current_ratio"], 0.89)

    def test_returns_none_for_missing_fields(self):
        raw = {"metric": {"roeTTM": 50.0}}
        _, data = self._call(raw)
        self.assertAlmostEqual(data["roe"], 0.5)
        self.assertIsNone(data["profit_margin"])
        self.assertIsNone(data["debt_to_equity"])
        self.assertIsNone(data["current_ratio"])

    def test_returns_empty_dict_when_all_fields_none(self):
        raw = {"metric": {}}
        sym, data = self._call(raw)
        self.assertEqual(sym, "AAPL")
        self.assertEqual(data, {})

    def test_returns_empty_dict_on_api_failure(self):
        sym, data = self._call(None)
        self.assertEqual(sym, "AAPL")
        self.assertEqual(data, {})

    def test_returns_empty_dict_on_missing_metric_key(self):
        _, data = self._call({})
        self.assertEqual(data, {})


# ---------------------------------------------------------------------------
# _fetch_analyst
# ---------------------------------------------------------------------------


class TestFetchAnalyst(unittest.TestCase):
    def _rec(self, strong_buy=10, buy=5, hold=8, sell=2, strong_sell=1):
        return [
            {
                "strongBuy": strong_buy,
                "buy": buy,
                "hold": hold,
                "sell": sell,
                "strongSell": strong_sell,
                "period": "2026-05-01",
                "symbol": "AAPL",
            }
        ]

    def _call(self, raw):
        from data.fundamentals import _fetch_analyst

        with patch("data.fundamentals._get", return_value=raw):
            return _fetch_analyst("AAPL")

    def test_computes_exact_bullish_pct(self):
        # strongBuy=10, buy=5 → bullish=15; sell=2, strongSell=1 → bearish=3; total=26
        _, data = self._call(self._rec())
        self.assertEqual(data["bullish_pct"], round(15 / 26 * 100))
        self.assertEqual(data["bearish_pct"], round(3 / 26 * 100))

    def test_bullish_and_bearish_within_100(self):
        _, data = self._call(self._rec())
        self.assertLessEqual(data["bullish_pct"] + data["bearish_pct"], 100)

    def test_includes_analyst_count(self):
        _, data = self._call(self._rec(strong_buy=10, buy=5, hold=8, sell=2, strong_sell=1))
        self.assertEqual(data["analyst_count"], 26)

    def test_returns_empty_when_total_is_zero(self):
        _, data = self._call(self._rec(0, 0, 0, 0, 0))
        self.assertEqual(data, {})

    def test_returns_empty_when_api_returns_none(self):
        _, data = self._call(None)
        self.assertEqual(data, {})

    def test_returns_empty_when_api_returns_empty_list(self):
        _, data = self._call([])
        self.assertEqual(data, {})

    def test_uses_most_recent_period(self):
        raw = [
            {
                "strongBuy": 20,
                "buy": 10,
                "hold": 5,
                "sell": 1,
                "strongSell": 0,
                "period": "2026-05-01",
            },
            {
                "strongBuy": 5,
                "buy": 2,
                "hold": 10,
                "sell": 5,
                "strongSell": 3,
                "period": "2026-04-01",
            },
        ]
        _, data = self._call(raw)
        # Should use first element (most recent)
        self.assertEqual(data["analyst_count"], 36)


# ---------------------------------------------------------------------------
# get_fundamentals
# ---------------------------------------------------------------------------


class TestGetFundamentals(unittest.TestCase):
    def test_returns_empty_when_no_api_key(self):
        from data.fundamentals import get_fundamentals

        with patch("data.fundamentals.FINNHUB_API_KEY", ""):
            result = get_fundamentals(["AAPL"])
        self.assertEqual(result, {})

    def test_returns_empty_when_no_symbols(self):
        from data.fundamentals import get_fundamentals

        with patch("data.fundamentals.FINNHUB_API_KEY", "testkey"):
            result = get_fundamentals([])
        self.assertEqual(result, {})

    def test_cache_hit_skips_fetch(self):
        from data.fundamentals import get_fundamentals

        cached_data = {
            "roe": 0.4,
            "profit_margin": 0.2,
            "debt_to_equity": 1.0,
            "current_ratio": 1.5,
        }
        cache = {"AAPL": {"fetched_at": _now_iso(), "data": cached_data}}
        with (
            patch("data.fundamentals.FINNHUB_API_KEY", "testkey"),
            patch("data.fundamentals._load_cache", return_value=cache),
            patch("data.fundamentals._fetch_ratios") as mock_fetch,
        ):
            result = get_fundamentals(["AAPL"])
        mock_fetch.assert_not_called()
        self.assertEqual(result["AAPL"]["roe"], 0.4)

    def test_stale_cache_triggers_fetch(self):
        from data.fundamentals import get_fundamentals

        stale_cache = {"AAPL": {"fetched_at": _stale_iso(), "data": {"roe": 0.1}}}
        fresh_data = {"roe": 0.5, "profit_margin": 0.2, "debt_to_equity": 0.8, "current_ratio": 2.0}
        with (
            patch("data.fundamentals.FINNHUB_API_KEY", "testkey"),
            patch("data.fundamentals._load_cache", return_value=stale_cache),
            patch("data.fundamentals._save_cache"),
            patch("data.fundamentals._fetch_ratios", return_value=("AAPL", fresh_data)),
        ):
            result = get_fundamentals(["AAPL"])
        self.assertAlmostEqual(result["AAPL"]["roe"], 0.5)

    def test_saves_cache_after_fetch(self):
        from data.fundamentals import get_fundamentals

        fresh_data = {
            "roe": 0.3,
            "profit_margin": 0.15,
            "debt_to_equity": 1.1,
            "current_ratio": 1.9,
        }
        with (
            patch("data.fundamentals.FINNHUB_API_KEY", "testkey"),
            patch("data.fundamentals._load_cache", return_value={}),
            patch("data.fundamentals._save_cache") as mock_save,
            patch("data.fundamentals._fetch_ratios", return_value=("AAPL", fresh_data)),
        ):
            get_fundamentals(["AAPL"])
        mock_save.assert_called_once()

    def test_empty_fetch_excluded_from_result(self):
        from data.fundamentals import get_fundamentals

        with (
            patch("data.fundamentals.FINNHUB_API_KEY", "testkey"),
            patch("data.fundamentals._load_cache", return_value={}),
            patch("data.fundamentals._save_cache"),
            patch("data.fundamentals._fetch_ratios", return_value=("AAPL", {})),
        ):
            result = get_fundamentals(["AAPL"])
        self.assertNotIn("AAPL", result)

    def test_partial_results_when_some_symbols_fail(self):
        from data.fundamentals import get_fundamentals

        def fake_fetch(sym):
            if sym == "AAPL":
                return (
                    "AAPL",
                    {"roe": 0.4, "profit_margin": 0.2, "debt_to_equity": 1.0, "current_ratio": 1.5},
                )
            return (sym, {})

        with (
            patch("data.fundamentals.FINNHUB_API_KEY", "testkey"),
            patch("data.fundamentals._load_cache", return_value={}),
            patch("data.fundamentals._save_cache"),
            patch("data.fundamentals._fetch_ratios", side_effect=fake_fetch),
        ):
            result = get_fundamentals(["AAPL", "FAIL"])
        self.assertIn("AAPL", result)
        self.assertNotIn("FAIL", result)

    def test_fetch_exception_is_caught_and_does_not_propagate(self):
        """Exception raised by _fetch_ratios is caught (lines 127-128) → symbol excluded."""
        from data.fundamentals import get_fundamentals

        with (
            patch("data.fundamentals.FINNHUB_API_KEY", "testkey"),
            patch("data.fundamentals._load_cache", return_value={}),
            patch("data.fundamentals._save_cache"),
            patch("data.fundamentals._fetch_ratios", side_effect=RuntimeError("network error")),
        ):
            result = get_fundamentals(["AAPL"])
        self.assertNotIn("AAPL", result)


# ---------------------------------------------------------------------------
# get_analyst_consensus
# ---------------------------------------------------------------------------


class TestGetAnalystConsensus(unittest.TestCase):
    def test_returns_empty_when_no_api_key(self):
        from data.fundamentals import get_analyst_consensus

        with patch("data.fundamentals.FINNHUB_API_KEY", ""):
            result = get_analyst_consensus(["AAPL"])
        self.assertEqual(result, {})

    def test_returns_empty_when_no_symbols(self):
        from data.fundamentals import get_analyst_consensus

        with patch("data.fundamentals.FINNHUB_API_KEY", "testkey"):
            result = get_analyst_consensus([])
        self.assertEqual(result, {})

    def test_cache_hit_returns_cached_data(self):
        from data.fundamentals import get_analyst_consensus

        cached = {"bullish_pct": 70, "bearish_pct": 10, "analyst_count": 20}
        cache = {"AAPL": {"fetched_at": _now_iso(), "data": cached}}
        with (
            patch("data.fundamentals.FINNHUB_API_KEY", "testkey"),
            patch("data.fundamentals._load_cache", return_value=cache),
            patch("data.fundamentals._fetch_analyst") as mock_fetch,
        ):
            result = get_analyst_consensus(["AAPL"])
        mock_fetch.assert_not_called()
        self.assertEqual(result["AAPL"]["bullish_pct"], 70)

    def test_api_data_included_in_result(self):
        from data.fundamentals import get_analyst_consensus

        analyst_data = {"bullish_pct": 65, "bearish_pct": 15, "analyst_count": 25}
        with (
            patch("data.fundamentals.FINNHUB_API_KEY", "testkey"),
            patch("data.fundamentals._load_cache", return_value={}),
            patch("data.fundamentals._save_cache"),
            patch("data.fundamentals._fetch_analyst", return_value=("AAPL", analyst_data)),
        ):
            result = get_analyst_consensus(["AAPL"])
        self.assertEqual(result["AAPL"]["analyst_count"], 25)

    def test_empty_fetch_excluded_from_result(self):
        from data.fundamentals import get_analyst_consensus

        with (
            patch("data.fundamentals.FINNHUB_API_KEY", "testkey"),
            patch("data.fundamentals._load_cache", return_value={}),
            patch("data.fundamentals._save_cache"),
            patch("data.fundamentals._fetch_analyst", return_value=("AAPL", {})),
        ):
            result = get_analyst_consensus(["AAPL"])
        self.assertNotIn("AAPL", result)

    def test_saves_cache_after_fetch(self):
        from data.fundamentals import get_analyst_consensus

        analyst_data = {"bullish_pct": 60, "bearish_pct": 20, "analyst_count": 15}
        with (
            patch("data.fundamentals.FINNHUB_API_KEY", "testkey"),
            patch("data.fundamentals._load_cache", return_value={}),
            patch("data.fundamentals._save_cache") as mock_save,
            patch("data.fundamentals._fetch_analyst", return_value=("AAPL", analyst_data)),
        ):
            get_analyst_consensus(["AAPL"])
        mock_save.assert_called_once()

    def test_fetch_exception_is_caught_and_does_not_propagate(self):
        """Exception raised by _fetch_analyst is caught (lines 165-166) → symbol excluded."""
        from data.fundamentals import get_analyst_consensus

        with (
            patch("data.fundamentals.FINNHUB_API_KEY", "testkey"),
            patch("data.fundamentals._load_cache", return_value={}),
            patch("data.fundamentals._save_cache"),
            patch("data.fundamentals._fetch_analyst", side_effect=RuntimeError("api error")),
        ):
            result = get_analyst_consensus(["AAPL"])
        self.assertNotIn("AAPL", result)


# ---------------------------------------------------------------------------
# get_sentiment delegates to get_analyst_consensus
# ---------------------------------------------------------------------------


class TestGetSentimentDelegates(unittest.TestCase):
    def test_delegates_to_get_analyst_consensus(self):
        from data.sentiment import get_sentiment

        expected = {"AAPL": {"bullish_pct": 70, "bearish_pct": 10, "analyst_count": 20}}
        with patch("data.sentiment.get_analyst_consensus", return_value=expected) as mock:
            result = get_sentiment(["AAPL"])
        mock.assert_called_once_with(["AAPL"])
        self.assertEqual(result, expected)

    def test_returns_empty_when_consensus_empty(self):
        from data.sentiment import get_sentiment

        with patch("data.sentiment.get_analyst_consensus", return_value={}):
            result = get_sentiment(["AAPL"])
        self.assertEqual(result, {})
