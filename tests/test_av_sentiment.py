"""Tests for data/av_sentiment.py — Alpha Vantage news sentiment enrichment."""

import unittest
from unittest.mock import MagicMock, patch

from data.av_sentiment import get_av_sentiment

_AV_RESPONSE = {
    "feed": [
        {
            "title": "Apple hits record high on AI announcement",
            "time_published": "20260509T100000",
            "ticker_sentiment": [
                {"ticker": "AAPL", "ticker_sentiment_score": "0.45", "relevance_score": "0.85"},
                {"ticker": "MSFT", "ticker_sentiment_score": "0.10", "relevance_score": "0.20"},
            ],
        },
        {
            "title": "Apple supply chain concerns ease",
            "time_published": "20260509T080000",
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
            "time_published": "20260509T120000",
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


class TestGetAvSentiment(unittest.TestCase):
    def test_returns_empty_when_no_api_key(self):
        with patch("data.av_sentiment.ALPHA_VANTAGE_API_KEY", None):
            result = get_av_sentiment(["AAPL"])
        self.assertEqual(result, {})

    def test_aggregates_scores_for_symbol(self):
        with (
            patch("data.av_sentiment.ALPHA_VANTAGE_API_KEY", "fake_key"),
            patch("data.av_sentiment.requests.get") as mock_get,
            patch("data.av_sentiment.time.sleep"),
        ):
            mock_get.return_value = _make_response(_AV_RESPONSE)
            result = get_av_sentiment(["AAPL", "MSFT"])
        self.assertIn("AAPL", result)
        # avg of 0.45 and 0.25 = 0.35
        self.assertAlmostEqual(result["AAPL"]["av_sentiment_score"], 0.35, places=2)
        self.assertEqual(result["AAPL"]["av_article_count"], 2)
        self.assertEqual(result["AAPL"]["av_sentiment_label"], "Bullish")

    def test_low_relevance_article_excluded(self):
        # MSFT relevance=0.20 < _MIN_RELEVANCE=0.30 → excluded
        with (
            patch("data.av_sentiment.ALPHA_VANTAGE_API_KEY", "fake_key"),
            patch("data.av_sentiment.requests.get") as mock_get,
            patch("data.av_sentiment.time.sleep"),
        ):
            mock_get.return_value = _make_response(_AV_RESPONSE)
            result = get_av_sentiment(["AAPL", "MSFT"])
        self.assertNotIn("MSFT", result)

    def test_bearish_label_for_negative_score(self):
        with (
            patch("data.av_sentiment.ALPHA_VANTAGE_API_KEY", "fake_key"),
            patch("data.av_sentiment.requests.get") as mock_get,
            patch("data.av_sentiment.time.sleep"),
        ):
            mock_get.return_value = _make_response(_AV_RESPONSE_BEARISH)
            result = get_av_sentiment(["NVDA"])
        self.assertIn("NVDA", result)
        self.assertEqual(result["NVDA"]["av_sentiment_label"], "Bearish")

    def test_symbol_absent_when_no_articles(self):
        with (
            patch("data.av_sentiment.ALPHA_VANTAGE_API_KEY", "fake_key"),
            patch("data.av_sentiment.requests.get") as mock_get,
            patch("data.av_sentiment.time.sleep"),
        ):
            mock_get.return_value = _make_response({"feed": []})
            result = get_av_sentiment(["AAPL"])
        self.assertNotIn("AAPL", result)

    def test_returns_empty_on_network_failure(self):
        with (
            patch("data.av_sentiment.ALPHA_VANTAGE_API_KEY", "fake_key"),
            patch("data.av_sentiment.requests.get", side_effect=Exception("timeout")),
            patch("data.av_sentiment.time.sleep"),
        ):
            result = get_av_sentiment(["AAPL"])
        self.assertEqual(result, {})

    def test_returns_empty_on_rate_limit_response(self):
        rate_limit = {"Information": "Thank you for using Alpha Vantage! rate limit reached"}
        with (
            patch("data.av_sentiment.ALPHA_VANTAGE_API_KEY", "fake_key"),
            patch("data.av_sentiment.requests.get") as mock_get,
            patch("data.av_sentiment.time.sleep"),
        ):
            mock_get.return_value = _make_response(rate_limit)
            result = get_av_sentiment(["AAPL"])
        self.assertEqual(result, {})

    def test_top_headline_is_first_article_for_symbol(self):
        with (
            patch("data.av_sentiment.ALPHA_VANTAGE_API_KEY", "fake_key"),
            patch("data.av_sentiment.requests.get") as mock_get,
            patch("data.av_sentiment.time.sleep"),
        ):
            mock_get.return_value = _make_response(_AV_RESPONSE)
            result = get_av_sentiment(["AAPL"])
        self.assertEqual(
            result["AAPL"]["av_top_headline"], "Apple hits record high on AI announcement"
        )

    def test_batches_symbols_correctly(self):
        # 12 symbols → 2 batches of 10 and 2
        symbols = [f"SYM{i}" for i in range(12)]
        with (
            patch("data.av_sentiment.ALPHA_VANTAGE_API_KEY", "fake_key"),
            patch("data.av_sentiment.requests.get") as mock_get,
            patch("data.av_sentiment.time.sleep"),
        ):
            mock_get.return_value = _make_response({"feed": []})
            get_av_sentiment(symbols)
        self.assertEqual(mock_get.call_count, 2)

    def test_neutral_label_for_near_zero_score(self):
        neutral_feed = {
            "feed": [
                {
                    "title": "Stock update",
                    "time_published": "20260509T100000",
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
            result = get_av_sentiment(["AAPL"])
        self.assertEqual(result["AAPL"]["av_sentiment_label"], "Neutral")
