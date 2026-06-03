"""Tests for data/av_sentiment.py — Alpha Vantage news sentiment enrichment."""

import json
import unittest
from datetime import UTC, date, datetime, timedelta
from unittest.mock import MagicMock, mock_open, patch

from data.av_sentiment import (
    _live_fetch_av_sentiment,
    _load_cache,
    _save_cache,
    get_av_sentiment,
    prefetch_av_sentiment,
)

_TODAY_DATE = date(2026, 6, 3)
_TODAY_KEY = _TODAY_DATE.isoformat()


def _ts(hours_ago: float = 1.0) -> str:
    """AV-format timestamp N hours in the past (always within the 24h lookback window)."""
    return (datetime.now(UTC) - timedelta(hours=hours_ago)).strftime("%Y%m%dT%H%M%S")


_AV_RESPONSE = {
    "feed": [
        {
            "title": "Apple hits record high on AI announcement",
            "time_published": _ts(1),
            "ticker_sentiment": [
                {"ticker": "AAPL", "ticker_sentiment_score": "0.45", "relevance_score": "0.85"},
                {"ticker": "MSFT", "ticker_sentiment_score": "0.10", "relevance_score": "0.20"},
            ],
        },
        {
            "title": "Apple supply chain concerns ease",
            "time_published": _ts(2),
            "ticker_sentiment": [
                {"ticker": "AAPL", "ticker_sentiment_score": "0.25", "relevance_score": "0.70"},
            ],
        },
    ]
}

_AV_RESPONSE_BEARISH = {
    "feed": [
        {
            "title": "NVDA misses estimates",
            "time_published": _ts(1),
            "ticker_sentiment": [
                {"ticker": "NVDA", "ticker_sentiment_score": "-0.50", "relevance_score": "0.90"},
            ],
        }
    ]
}


def _make_response(json_data):
    m = MagicMock()
    m.raise_for_status = MagicMock()
    m.json.return_value = json_data
    return m


class TestLiveFetchAvSentiment(unittest.TestCase):
    def test_returns_empty_when_no_api_key(self):
        with patch("data.av_sentiment.ALPHA_VANTAGE_API_KEY", None):
            result = _live_fetch_av_sentiment(["AAPL"])
        self.assertEqual(result, {})

    def test_aggregates_scores_for_symbol(self):
        with (
            patch("data.av_sentiment.ALPHA_VANTAGE_API_KEY", "fake_key"),
            patch("data.av_sentiment.requests.get") as mock_get,
            patch("data.av_sentiment.time.sleep"),
        ):
            mock_get.return_value = _make_response(_AV_RESPONSE)
            result = _live_fetch_av_sentiment(["AAPL", "MSFT"])
        self.assertIn("AAPL", result)
        self.assertIsNotNone(result["AAPL"])
        self.assertAlmostEqual(result["AAPL"]["av_sentiment_score"], 0.35, places=2)
        self.assertEqual(result["AAPL"]["av_article_count"], 2)
        self.assertEqual(result["AAPL"]["av_sentiment_label"], "Bullish")

    def test_low_relevance_article_produces_none_sentinel(self):
        with (
            patch("data.av_sentiment.ALPHA_VANTAGE_API_KEY", "fake_key"),
            patch("data.av_sentiment.requests.get") as mock_get,
            patch("data.av_sentiment.time.sleep"),
        ):
            mock_get.return_value = _make_response(_AV_RESPONSE)
            result = _live_fetch_av_sentiment(["AAPL", "MSFT"])
        # MSFT relevance 0.20 < _MIN_RELEVANCE 0.30 → None sentinel
        self.assertIn("MSFT", result)
        self.assertIsNone(result["MSFT"])

    def test_bearish_label_for_negative_score(self):
        with (
            patch("data.av_sentiment.ALPHA_VANTAGE_API_KEY", "fake_key"),
            patch("data.av_sentiment.requests.get") as mock_get,
            patch("data.av_sentiment.time.sleep"),
        ):
            mock_get.return_value = _make_response(_AV_RESPONSE_BEARISH)
            result = _live_fetch_av_sentiment(["NVDA"])
        self.assertIsNotNone(result["NVDA"])
        self.assertEqual(result["NVDA"]["av_sentiment_label"], "Bearish")

    def test_none_sentinel_when_no_articles(self):
        with (
            patch("data.av_sentiment.ALPHA_VANTAGE_API_KEY", "fake_key"),
            patch("data.av_sentiment.requests.get") as mock_get,
            patch("data.av_sentiment.time.sleep"),
        ):
            mock_get.return_value = _make_response({"feed": []})
            result = _live_fetch_av_sentiment(["AAPL"])
        self.assertIn("AAPL", result)
        self.assertIsNone(result["AAPL"])

    def test_none_sentinel_on_network_failure(self):
        with (
            patch("data.av_sentiment.ALPHA_VANTAGE_API_KEY", "fake_key"),
            patch("data.av_sentiment.requests.get", side_effect=Exception("timeout")),
            patch("data.av_sentiment.time.sleep"),
        ):
            result = _live_fetch_av_sentiment(["AAPL"])
        self.assertEqual(result, {"AAPL": None})

    def test_none_sentinel_on_rate_limit_response(self):
        rate_limit = {"Information": "Thank you for using Alpha Vantage! rate limit reached"}
        with (
            patch("data.av_sentiment.ALPHA_VANTAGE_API_KEY", "fake_key"),
            patch("data.av_sentiment.requests.get") as mock_get,
            patch("data.av_sentiment.time.sleep"),
        ):
            mock_get.return_value = _make_response(rate_limit)
            result = _live_fetch_av_sentiment(["AAPL"])
        self.assertEqual(result, {"AAPL": None})

    def test_top_headline_is_first_article_for_symbol(self):
        with (
            patch("data.av_sentiment.ALPHA_VANTAGE_API_KEY", "fake_key"),
            patch("data.av_sentiment.requests.get") as mock_get,
            patch("data.av_sentiment.time.sleep"),
        ):
            mock_get.return_value = _make_response(_AV_RESPONSE)
            result = _live_fetch_av_sentiment(["AAPL"])
        self.assertEqual(
            result["AAPL"]["av_top_headline"], "Apple hits record high on AI announcement"
        )

    def test_batches_symbols_correctly(self):
        symbols = [f"SYM{i}" for i in range(12)]
        with (
            patch("data.av_sentiment.ALPHA_VANTAGE_API_KEY", "fake_key"),
            patch("data.av_sentiment.requests.get") as mock_get,
            patch("data.av_sentiment.time.sleep"),
        ):
            mock_get.return_value = _make_response({"feed": []})
            _live_fetch_av_sentiment(symbols)
        self.assertEqual(mock_get.call_count, 2)

    def test_neutral_label_for_near_zero_score(self):
        neutral_feed = {
            "feed": [
                {
                    "title": "Stock update",
                    "time_published": _ts(1),
                    "ticker_sentiment": [
                        {
                            "ticker": "AAPL",
                            "ticker_sentiment_score": "0.05",
                            "relevance_score": "0.60",
                        },
                    ],
                }
            ]
        }
        with (
            patch("data.av_sentiment.ALPHA_VANTAGE_API_KEY", "fake_key"),
            patch("data.av_sentiment.requests.get") as mock_get,
            patch("data.av_sentiment.time.sleep"),
        ):
            mock_get.return_value = _make_response(neutral_feed)
            result = _live_fetch_av_sentiment(["AAPL"])
        self.assertEqual(result["AAPL"]["av_sentiment_label"], "Neutral")

    def test_sleep_before_continue_on_exception_with_more_batches(self):
        symbols = [f"SYM{i}" for i in range(11)]
        sleep_mock = MagicMock()
        with (
            patch("data.av_sentiment.ALPHA_VANTAGE_API_KEY", "fake_key"),
            patch(
                "data.av_sentiment.requests.get",
                side_effect=[Exception("timeout"), _make_response({"feed": []})],
            ),
            patch("data.av_sentiment.time.sleep", sleep_mock),
        ):
            _live_fetch_av_sentiment(symbols)
        sleep_mock.assert_called()

    def test_sleep_before_continue_on_no_feed_with_more_batches(self):
        symbols = [f"SYM{i}" for i in range(11)]
        sleep_mock = MagicMock()
        rate_limit = {"Information": "rate limited"}
        with (
            patch("data.av_sentiment.ALPHA_VANTAGE_API_KEY", "fake_key"),
            patch(
                "data.av_sentiment.requests.get",
                side_effect=[_make_response(rate_limit), _make_response({"feed": []})],
            ),
            patch("data.av_sentiment.time.sleep", sleep_mock),
        ):
            _live_fetch_av_sentiment(symbols)
        sleep_mock.assert_called()

    def test_skips_article_with_invalid_time_published(self):
        bad_time_feed = {
            "feed": [
                {
                    "title": "Bad timestamp article",
                    "time_published": "not-a-timestamp",
                    "ticker_sentiment": [
                        {
                            "ticker": "AAPL",
                            "ticker_sentiment_score": "0.5",
                            "relevance_score": "0.8",
                        },
                    ],
                }
            ]
        }
        with (
            patch("data.av_sentiment.ALPHA_VANTAGE_API_KEY", "fake_key"),
            patch("data.av_sentiment.requests.get") as mock_get,
            patch("data.av_sentiment.time.sleep"),
        ):
            mock_get.return_value = _make_response(bad_time_feed)
            result = _live_fetch_av_sentiment(["AAPL"])
        self.assertIsNone(result["AAPL"])

    def test_skips_article_published_before_cutoff(self):
        old_time = (datetime.now(UTC) - timedelta(hours=48)).strftime("%Y%m%dT%H%M%S")
        old_feed = {
            "feed": [
                {
                    "title": "Old article",
                    "time_published": old_time,
                    "ticker_sentiment": [
                        {
                            "ticker": "AAPL",
                            "ticker_sentiment_score": "0.5",
                            "relevance_score": "0.8",
                        },
                    ],
                }
            ]
        }
        with (
            patch("data.av_sentiment.ALPHA_VANTAGE_API_KEY", "fake_key"),
            patch("data.av_sentiment.requests.get") as mock_get,
            patch("data.av_sentiment.time.sleep"),
        ):
            mock_get.return_value = _make_response(old_feed)
            result = _live_fetch_av_sentiment(["AAPL"])
        self.assertIsNone(result["AAPL"])

    def test_skips_ticker_sentiment_with_invalid_score(self):
        bad_score_feed = {
            "feed": [
                {
                    "title": "Article with bad score",
                    "time_published": _ts(1),
                    "ticker_sentiment": [
                        {
                            "ticker": "AAPL",
                            "ticker_sentiment_score": "not-a-float",
                            "relevance_score": "0.8",
                        },
                    ],
                }
            ]
        }
        with (
            patch("data.av_sentiment.ALPHA_VANTAGE_API_KEY", "fake_key"),
            patch("data.av_sentiment.requests.get") as mock_get,
            patch("data.av_sentiment.time.sleep"),
        ):
            mock_get.return_value = _make_response(bad_score_feed)
            result = _live_fetch_av_sentiment(["AAPL"])
        self.assertIsNone(result["AAPL"])

    def test_no_feed_and_no_info_key_skips_silently(self):
        no_feed_no_info = {"status": "OK"}
        with (
            patch("data.av_sentiment.ALPHA_VANTAGE_API_KEY", "fake_key"),
            patch("data.av_sentiment.requests.get") as mock_get,
            patch("data.av_sentiment.time.sleep"),
        ):
            mock_get.return_value = _make_response(no_feed_no_info)
            result = _live_fetch_av_sentiment(["AAPL"])
        self.assertEqual(result, {"AAPL": None})


class TestGetAvSentiment(unittest.TestCase):
    def test_returns_empty_when_no_api_key(self):
        with patch("data.av_sentiment.ALPHA_VANTAGE_API_KEY", None):
            result = get_av_sentiment(["AAPL"])
        self.assertEqual(result, {})

    def test_cache_miss_fetches_and_returns_result(self):
        with (
            patch("data.av_sentiment._load_cache", return_value={}),
            patch("data.av_sentiment._save_cache"),
            patch("data.av_sentiment.ALPHA_VANTAGE_API_KEY", "fake_key"),
            patch("data.av_sentiment.requests.get") as mock_get,
            patch("data.av_sentiment.time.sleep"),
            patch("data.av_sentiment.today_et", return_value=_TODAY_DATE),
        ):
            mock_get.return_value = _make_response(_AV_RESPONSE)
            result = get_av_sentiment(["AAPL", "MSFT"])
        self.assertIn("AAPL", result)
        # MSFT below relevance threshold → None sentinel → omitted from result
        self.assertNotIn("MSFT", result)

    def test_cache_hit_skips_network(self):
        warm_cache = {
            _TODAY_KEY: {
                "AAPL": {
                    "av_sentiment_score": 0.35,
                    "av_article_count": 2,
                    "av_sentiment_label": "Bullish",
                    "av_top_headline": "cached headline",
                }
            }
        }
        with (
            patch("data.av_sentiment._load_cache", return_value=warm_cache),
            patch("data.av_sentiment._save_cache") as mock_save,
            patch("data.av_sentiment.ALPHA_VANTAGE_API_KEY", "fake_key"),
            patch("data.av_sentiment.requests.get") as mock_get,
            patch("data.av_sentiment.today_et", return_value=_TODAY_DATE),
        ):
            result = get_av_sentiment(["AAPL"])
        mock_get.assert_not_called()
        mock_save.assert_not_called()
        self.assertAlmostEqual(result["AAPL"]["av_sentiment_score"], 0.35)

    def test_null_sentinel_omits_symbol(self):
        warm_cache = {_TODAY_KEY: {"AAPL": None}}
        with (
            patch("data.av_sentiment._load_cache", return_value=warm_cache),
            patch("data.av_sentiment.ALPHA_VANTAGE_API_KEY", "fake_key"),
            patch("data.av_sentiment.requests.get") as mock_get,
            patch("data.av_sentiment.today_et", return_value=_TODAY_DATE),
        ):
            result = get_av_sentiment(["AAPL"])
        mock_get.assert_not_called()
        self.assertNotIn("AAPL", result)

    def test_partial_cache_hit_only_fetches_missing(self):
        warm_cache = {
            _TODAY_KEY: {
                "AAPL": {
                    "av_sentiment_score": 0.35,
                    "av_article_count": 2,
                    "av_sentiment_label": "Bullish",
                    "av_top_headline": "cached",
                }
            }
        }
        with (
            patch("data.av_sentiment._load_cache", return_value=warm_cache),
            patch("data.av_sentiment._save_cache"),
            patch("data.av_sentiment.ALPHA_VANTAGE_API_KEY", "fake_key"),
            patch("data.av_sentiment.requests.get") as mock_get,
            patch("data.av_sentiment.time.sleep"),
            patch("data.av_sentiment.today_et", return_value=_TODAY_DATE),
        ):
            mock_get.return_value = _make_response(_AV_RESPONSE_BEARISH)
            result = get_av_sentiment(["AAPL", "NVDA"])
        mock_get.assert_called_once()
        self.assertIn("AAPL", result)
        self.assertIn("NVDA", result)

    def test_cache_miss_saves_today_key(self):
        with (
            patch("data.av_sentiment._load_cache", return_value={}),
            patch("data.av_sentiment._save_cache") as mock_save,
            patch("data.av_sentiment.ALPHA_VANTAGE_API_KEY", "fake_key"),
            patch("data.av_sentiment.requests.get") as mock_get,
            patch("data.av_sentiment.time.sleep"),
            patch("data.av_sentiment.today_et", return_value=_TODAY_DATE),
        ):
            mock_get.return_value = _make_response({"feed": []})
            get_av_sentiment(["AAPL"])
        mock_save.assert_called_once()
        saved = mock_save.call_args[0][0]
        self.assertIn(_TODAY_KEY, saved)


class TestPrefetchAvSentiment(unittest.TestCase):
    def test_returns_zero_when_no_api_key(self):
        with patch("data.av_sentiment.ALPHA_VANTAGE_API_KEY", None):
            n = prefetch_av_sentiment(["AAPL"])
        self.assertEqual(n, 0)

    def test_returns_count_of_symbols_fetched(self):
        with (
            patch("data.av_sentiment._load_cache", return_value={}),
            patch("data.av_sentiment._save_cache"),
            patch("data.av_sentiment.ALPHA_VANTAGE_API_KEY", "fake_key"),
            patch("data.av_sentiment.requests.get") as mock_get,
            patch("data.av_sentiment.time.sleep"),
            patch("data.av_sentiment.today_et", return_value=_TODAY_DATE),
        ):
            mock_get.return_value = _make_response({"feed": []})
            n = prefetch_av_sentiment(["AAPL", "MSFT"])
        self.assertEqual(n, 2)

    def test_warm_cache_returns_zero(self):
        warm = {_TODAY_KEY: {"AAPL": None, "MSFT": None}}
        with (
            patch("data.av_sentiment._load_cache", return_value=warm),
            patch("data.av_sentiment._save_cache") as mock_save,
            patch("data.av_sentiment.ALPHA_VANTAGE_API_KEY", "fake_key"),
            patch("data.av_sentiment.requests.get") as mock_get,
            patch("data.av_sentiment.today_et", return_value=_TODAY_DATE),
        ):
            n = prefetch_av_sentiment(["AAPL", "MSFT"])
        self.assertEqual(n, 0)
        mock_get.assert_not_called()
        mock_save.assert_not_called()

    def test_uses_stock_universe_when_symbols_none(self):
        with (
            patch("data.av_sentiment._load_cache", return_value={}),
            patch("data.av_sentiment._save_cache"),
            patch("data.av_sentiment.ALPHA_VANTAGE_API_KEY", "fake_key"),
            patch("data.av_sentiment._live_fetch_av_sentiment", return_value={}) as mock_fetch,
            patch("data.av_sentiment.today_et", return_value=_TODAY_DATE),
            patch("data.av_sentiment.STOCK_UNIVERSE", {"AAPL", "MSFT"}),
        ):
            prefetch_av_sentiment()
        fetched = set(mock_fetch.call_args[0][0])
        self.assertEqual(fetched, {"AAPL", "MSFT"})

    def test_saves_today_key(self):
        with (
            patch("data.av_sentiment._load_cache", return_value={}),
            patch("data.av_sentiment._save_cache") as mock_save,
            patch("data.av_sentiment.ALPHA_VANTAGE_API_KEY", "fake_key"),
            patch("data.av_sentiment.requests.get") as mock_get,
            patch("data.av_sentiment.time.sleep"),
            patch("data.av_sentiment.today_et", return_value=_TODAY_DATE),
        ):
            mock_get.return_value = _make_response({"feed": []})
            prefetch_av_sentiment(["AAPL"])
        saved = mock_save.call_args[0][0]
        self.assertIn(_TODAY_KEY, saved)


class TestLoadSaveCacheAvSentiment(unittest.TestCase):
    def test_load_returns_empty_on_missing_file(self):
        with patch("builtins.open", side_effect=FileNotFoundError):
            result = _load_cache()
        self.assertEqual(result, {})

    def test_load_returns_empty_on_json_error(self):
        with (
            patch("builtins.open", mock_open(read_data="not valid json")),
            patch(
                "data.av_sentiment.json.load",
                side_effect=json.JSONDecodeError("err", "", 0),
            ),
        ):
            result = _load_cache()
        self.assertEqual(result, {})

    def test_save_writes_json_on_success(self):
        m = mock_open()
        with (
            patch("data.av_sentiment.os.makedirs"),
            patch("builtins.open", m),
            patch("data.av_sentiment.json.dump") as mock_dump,
        ):
            _save_cache({_TODAY_KEY: {}})
        mock_dump.assert_called_once()

    def test_save_logs_warning_on_os_error(self):
        with (
            patch("data.av_sentiment.os.makedirs"),
            patch("builtins.open", side_effect=OSError("disk full")),
        ):
            _save_cache({_TODAY_KEY: {}})  # should not raise
