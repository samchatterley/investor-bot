"""Tests for data/fundamentals.py — yfinance fundamentals and analyst consensus."""

import unittest
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _stale_iso() -> str:
    return (datetime.now(UTC) - timedelta(hours=25)).isoformat()


def _ticker_mock(info: dict) -> MagicMock:
    m = MagicMock()
    m.info = info
    return m


# ---------------------------------------------------------------------------
# _fetch_ratios
# ---------------------------------------------------------------------------


class TestFetchRatios(unittest.TestCase):
    def _call(self, info: dict):
        from data.fundamentals import _fetch_ratios

        with patch("data.fundamentals.yf.Ticker", return_value=_ticker_mock(info)):
            return _fetch_ratios("AAPL")

    def test_extracts_all_fields(self):
        sym, data = self._call(
            {
                "returnOnEquity": 0.5,
                "profitMargins": 0.25,
                "debtToEquity": 79.5,
                "currentRatio": 1.8,
            }
        )
        self.assertEqual(sym, "AAPL")
        self.assertAlmostEqual(data["roe"], 0.5)
        self.assertAlmostEqual(data["profit_margin"], 0.25)
        self.assertAlmostEqual(data["debt_to_equity"], 79.5)
        self.assertAlmostEqual(data["current_ratio"], 1.8)

    def test_partial_info_returns_none_for_missing_fields(self):
        _, data = self._call({"returnOnEquity": 0.3})
        self.assertAlmostEqual(data["roe"], 0.3)
        self.assertIsNone(data["profit_margin"])
        self.assertIsNone(data["debt_to_equity"])

    def test_returns_empty_dict_when_all_fields_none(self):
        sym, data = self._call({})
        self.assertEqual(sym, "AAPL")
        self.assertEqual(data, {})

    def test_returns_empty_dict_when_info_is_empty_dict(self):
        _, data = self._call({})
        self.assertEqual(data, {})

    def test_exception_returns_empty_dict(self):
        from data.fundamentals import _fetch_ratios

        with patch("data.fundamentals.yf.Ticker", side_effect=Exception("network error")):
            sym, data = _fetch_ratios("AAPL")
        self.assertEqual(sym, "AAPL")
        self.assertEqual(data, {})


# ---------------------------------------------------------------------------
# _fetch_analyst
# ---------------------------------------------------------------------------


class TestFetchAnalyst(unittest.TestCase):
    def _call(self, info: dict):
        from data.fundamentals import _fetch_analyst

        with patch("data.fundamentals.yf.Ticker", return_value=_ticker_mock(info)):
            return _fetch_analyst("AAPL")

    def test_buy_recommendation_gives_high_bullish_pct(self):
        _, data = self._call(
            {"recommendationKey": "buy", "numberOfAnalystOpinions": 30, "targetMeanPrice": 200.0}
        )
        self.assertIn("bullish_pct", data)
        self.assertIn("bearish_pct", data)
        self.assertGreater(data["bullish_pct"], 50)

    def test_sell_recommendation_gives_high_bearish_pct(self):
        _, data = self._call({"recommendationKey": "sell", "numberOfAnalystOpinions": 20})
        self.assertGreater(data["bearish_pct"], 50)

    def test_strong_buy_gives_highest_bullish_pct(self):
        _, buy = self._call({"recommendationKey": "buy", "numberOfAnalystOpinions": 10})
        _, strong_buy = self._call(
            {"recommendationKey": "strong_buy", "numberOfAnalystOpinions": 10}
        )
        self.assertGreater(strong_buy["bullish_pct"], buy["bullish_pct"])

    def test_bullish_and_bearish_within_100(self):
        for key in ("strong_buy", "buy", "hold", "underperform", "sell", "strong_sell"):
            _, data = self._call({"recommendationKey": key, "numberOfAnalystOpinions": 25})
            self.assertLessEqual(
                data["bullish_pct"] + data["bearish_pct"],
                100,
                f"bullish+bearish > 100 for {key}",
            )

    def test_includes_analyst_count(self):
        _, data = self._call({"recommendationKey": "buy", "numberOfAnalystOpinions": 42})
        self.assertEqual(data["analyst_count"], 42)

    def test_unknown_recommendation_key_omits_pct(self):
        _, data = self._call({"recommendationKey": "neutral", "numberOfAnalystOpinions": 10})
        self.assertNotIn("bullish_pct", data)

    def test_no_recommendation_key_returns_empty(self):
        _, data = self._call({})
        self.assertEqual(data, {})

    def test_includes_target_price(self):
        _, data = self._call({"recommendationKey": "buy", "targetMeanPrice": 305.5})
        self.assertEqual(data["target_price"], 305.5)

    def test_no_target_price_when_absent(self):
        _, data = self._call({"recommendationKey": "buy", "numberOfAnalystOpinions": 10})
        self.assertNotIn("target_price", data)

    def test_exception_returns_empty_dict(self):
        from data.fundamentals import _fetch_analyst

        with patch("data.fundamentals.yf.Ticker", side_effect=Exception("timeout")):
            sym, data = _fetch_analyst("AAPL")
        self.assertEqual(sym, "AAPL")
        self.assertEqual(data, {})


# ---------------------------------------------------------------------------
# get_fundamentals
# ---------------------------------------------------------------------------


class TestGetFundamentals(unittest.TestCase):
    def test_returns_empty_when_no_symbols(self):
        from data.fundamentals import get_fundamentals

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
            patch("data.fundamentals._load_cache", return_value={}),
            patch("data.fundamentals._save_cache") as mock_save,
            patch("data.fundamentals._fetch_ratios", return_value=("AAPL", fresh_data)),
        ):
            get_fundamentals(["AAPL"])
        mock_save.assert_called_once()

    def test_empty_fetch_excluded_from_result(self):
        from data.fundamentals import get_fundamentals

        with (
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
    def test_returns_empty_when_no_symbols(self):
        from data.fundamentals import get_analyst_consensus

        result = get_analyst_consensus([])
        self.assertEqual(result, {})

    def test_cache_hit_returns_cached_data(self):
        from data.fundamentals import get_analyst_consensus

        cached = {"bullish_pct": 70, "bearish_pct": 10, "analyst_count": 20}
        cache = {"AAPL": {"fetched_at": _now_iso(), "data": cached}}
        with (
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
            patch("data.fundamentals._load_cache", return_value={}),
            patch("data.fundamentals._save_cache"),
            patch("data.fundamentals._fetch_analyst", return_value=("AAPL", analyst_data)),
        ):
            result = get_analyst_consensus(["AAPL"])
        self.assertEqual(result["AAPL"]["analyst_count"], 25)

    def test_empty_fetch_excluded_from_result(self):
        from data.fundamentals import get_analyst_consensus

        with (
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
