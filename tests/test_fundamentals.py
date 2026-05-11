"""Tests for data/fundamentals.py — FMP fundamentals and analyst consensus."""

import unittest
from datetime import UTC, datetime, timedelta
from unittest.mock import patch


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _stale_iso() -> str:
    return (datetime.now(UTC) - timedelta(hours=25)).isoformat()


# ---------------------------------------------------------------------------
# _fetch_ratios
# ---------------------------------------------------------------------------


class TestFetchRatios(unittest.TestCase):
    def _call(self, raw):
        from data.fundamentals import _fetch_ratios

        with patch("data.fundamentals._get", return_value=raw):
            return _fetch_ratios("AAPL")

    def test_extracts_fields_from_list_response(self):
        raw = [
            {
                "returnOnEquityTTM": 0.5,
                "netProfitMarginTTM": 0.25,
                "debtEquityRatioTTM": 1.2,
                "currentRatioTTM": 1.8,
            }
        ]
        sym, data = self._call(raw)
        self.assertEqual(sym, "AAPL")
        self.assertAlmostEqual(data["roe"], 0.5)
        self.assertAlmostEqual(data["profit_margin"], 0.25)
        self.assertAlmostEqual(data["debt_to_equity"], 1.2)
        self.assertAlmostEqual(data["current_ratio"], 1.8)

    def test_extracts_fields_from_dict_response(self):
        raw = {
            "returnOnEquityTTM": 0.3,
            "netProfitMarginTTM": 0.1,
            "debtEquityRatioTTM": 0.8,
            "currentRatioTTM": 2.0,
        }
        sym, data = self._call(raw)
        self.assertAlmostEqual(data["roe"], 0.3)

    def test_returns_none_for_missing_fields(self):
        raw = [{}]
        _, data = self._call(raw)
        self.assertIsNone(data["roe"])
        self.assertIsNone(data["profit_margin"])

    def test_returns_empty_dict_on_api_failure(self):
        sym, data = self._call(None)
        self.assertEqual(sym, "AAPL")
        self.assertEqual(data, {})

    def test_returns_empty_dict_on_empty_list(self):
        sym, data = self._call([])
        self.assertEqual(data, {})


# ---------------------------------------------------------------------------
# _fetch_analyst
# ---------------------------------------------------------------------------


class TestFetchAnalyst(unittest.TestCase):
    def _recs(self, strong_buy=10, buy=5, hold=8, sell=2, strong_sell=1):
        return [
            {
                "analystRatingsStrongBuy": strong_buy,
                "analystRatingsbuy": buy,
                "analystRatingsHold": hold,
                "analystRatingsSell": sell,
                "analystRatingsStrongSell": strong_sell,
            }
        ]

    def _call(self, recs=None, targets=None):
        from data.fundamentals import _fetch_analyst

        def fake_get(path, params=None):
            if "analyst-stock-recommendations" in path:
                return recs
            if "price-target-consensus" in path:
                return targets
            return None

        with patch("data.fundamentals._get", side_effect=fake_get):
            return _fetch_analyst("AAPL")

    def test_computes_bullish_pct(self):
        # strong_buy=10, buy=5 → bullish=15; sell=2, strong_sell=1 → bearish=3; total=26
        _, data = self._call(recs=self._recs())
        self.assertIn("bullish_pct", data)
        self.assertIn("bearish_pct", data)
        expected_bullish = round(15 / 26 * 100)
        self.assertEqual(data["bullish_pct"], expected_bullish)

    def test_bullish_and_bearish_within_100(self):
        _, data = self._call(recs=self._recs())
        self.assertLessEqual(data["bullish_pct"] + data["bearish_pct"], 100)

    def test_includes_analyst_count(self):
        _, data = self._call(recs=self._recs(strong_buy=10, buy=5, hold=8, sell=2, strong_sell=1))
        self.assertEqual(data["analyst_count"], 26)

    def test_returns_empty_when_total_is_zero(self):
        _, data = self._call(recs=self._recs(0, 0, 0, 0, 0))
        self.assertNotIn("bullish_pct", data)

    def test_returns_empty_when_no_recs(self):
        _, data = self._call(recs=None)
        self.assertEqual(data, {})

    def test_includes_target_price_from_dict(self):
        targets = {"targetConsensus": 200.0, "targetMedian": 190.0}
        _, data = self._call(recs=self._recs(), targets=targets)
        self.assertEqual(data["target_price"], 200.0)

    def test_includes_target_price_from_list(self):
        targets = [{"targetConsensus": 195.0}]
        _, data = self._call(recs=self._recs(), targets=targets)
        self.assertEqual(data["target_price"], 195.0)

    def test_falls_back_to_median_when_no_consensus(self):
        targets = {"targetMedian": 185.0}
        _, data = self._call(recs=self._recs(), targets=targets)
        self.assertEqual(data["target_price"], 185.0)

    def test_no_target_price_when_targets_none(self):
        _, data = self._call(recs=self._recs(), targets=None)
        self.assertNotIn("target_price", data)


# ---------------------------------------------------------------------------
# get_fundamentals
# ---------------------------------------------------------------------------


class TestGetFundamentals(unittest.TestCase):
    def test_returns_empty_when_no_api_key(self):
        from data.fundamentals import get_fundamentals

        with patch("data.fundamentals.FMP_API_KEY", ""):
            result = get_fundamentals(["AAPL"])
        self.assertEqual(result, {})

    def test_returns_empty_when_no_symbols(self):
        from data.fundamentals import get_fundamentals

        with patch("data.fundamentals.FMP_API_KEY", "testkey"):
            result = get_fundamentals([])
        self.assertEqual(result, {})

    def test_cache_hit_skips_api(self):
        from data.fundamentals import get_fundamentals

        cached_data = {
            "roe": 0.4,
            "profit_margin": 0.2,
            "debt_to_equity": 1.0,
            "current_ratio": 1.5,
        }
        cache = {"AAPL": {"fetched_at": _now_iso(), "data": cached_data}}
        with (
            patch("data.fundamentals.FMP_API_KEY", "testkey"),
            patch("data.fundamentals._load_cache", return_value=cache),
            patch("data.fundamentals._fetch_ratios") as mock_fetch,
        ):
            result = get_fundamentals(["AAPL"])
        mock_fetch.assert_not_called()
        self.assertEqual(result["AAPL"]["roe"], 0.4)

    def test_stale_cache_triggers_api_call(self):
        from data.fundamentals import get_fundamentals

        stale_cache = {"AAPL": {"fetched_at": _stale_iso(), "data": {"roe": 0.1}}}
        fresh_data = {"roe": 0.5, "profit_margin": 0.2, "debt_to_equity": 0.8, "current_ratio": 2.0}
        with (
            patch("data.fundamentals.FMP_API_KEY", "testkey"),
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
            patch("data.fundamentals.FMP_API_KEY", "testkey"),
            patch("data.fundamentals._load_cache", return_value={}),
            patch("data.fundamentals._save_cache") as mock_save,
            patch("data.fundamentals._fetch_ratios", return_value=("AAPL", fresh_data)),
        ):
            get_fundamentals(["AAPL"])
        mock_save.assert_called_once()

    def test_api_failure_excluded_from_result(self):
        from data.fundamentals import get_fundamentals

        with (
            patch("data.fundamentals.FMP_API_KEY", "testkey"),
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
            patch("data.fundamentals.FMP_API_KEY", "testkey"),
            patch("data.fundamentals._load_cache", return_value={}),
            patch("data.fundamentals._save_cache"),
            patch("data.fundamentals._fetch_ratios", side_effect=fake_fetch),
        ):
            result = get_fundamentals(["AAPL", "FAIL"])
        self.assertIn("AAPL", result)
        self.assertNotIn("FAIL", result)


# ---------------------------------------------------------------------------
# get_analyst_consensus
# ---------------------------------------------------------------------------


class TestGetAnalystConsensus(unittest.TestCase):
    def test_returns_empty_when_no_api_key(self):
        from data.fundamentals import get_analyst_consensus

        with patch("data.fundamentals.FMP_API_KEY", ""):
            result = get_analyst_consensus(["AAPL"])
        self.assertEqual(result, {})

    def test_returns_empty_when_no_symbols(self):
        from data.fundamentals import get_analyst_consensus

        with patch("data.fundamentals.FMP_API_KEY", "testkey"):
            result = get_analyst_consensus([])
        self.assertEqual(result, {})

    def test_cache_hit_returns_cached_data(self):
        from data.fundamentals import get_analyst_consensus

        cached = {"bullish_pct": 70, "bearish_pct": 10, "analyst_count": 20}
        cache = {"AAPL": {"fetched_at": _now_iso(), "data": cached}}
        with (
            patch("data.fundamentals.FMP_API_KEY", "testkey"),
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
            patch("data.fundamentals.FMP_API_KEY", "testkey"),
            patch("data.fundamentals._load_cache", return_value={}),
            patch("data.fundamentals._save_cache"),
            patch("data.fundamentals._fetch_analyst", return_value=("AAPL", analyst_data)),
        ):
            result = get_analyst_consensus(["AAPL"])
        self.assertEqual(result["AAPL"]["analyst_count"], 25)

    def test_empty_fetch_excluded_from_result(self):
        from data.fundamentals import get_analyst_consensus

        with (
            patch("data.fundamentals.FMP_API_KEY", "testkey"),
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
            patch("data.fundamentals.FMP_API_KEY", "testkey"),
            patch("data.fundamentals._load_cache", return_value={}),
            patch("data.fundamentals._save_cache") as mock_save,
            patch("data.fundamentals._fetch_analyst", return_value=("AAPL", analyst_data)),
        ):
            get_analyst_consensus(["AAPL"])
        mock_save.assert_called_once()


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
