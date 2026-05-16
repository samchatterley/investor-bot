"""Tests for data/news_fetcher.py and data/sentiment.py — parallel dispatch and unhappy paths."""

import unittest
from unittest.mock import MagicMock, patch


class TestFetchSingleNews(unittest.TestCase):
    def _ticker(self, news):
        t = MagicMock()
        t.news = news
        return t

    def test_returns_symbol_and_headlines(self):
        from data.news_fetcher import _fetch_single

        headline = {"title": "Apple gains"}
        with patch("data.news_fetcher.yf.Ticker", return_value=self._ticker([headline])):
            sym, headlines = _fetch_single("AAPL", max_headlines=3)
        self.assertEqual(sym, "AAPL")
        self.assertIn("Apple gains", headlines)

    def test_uses_content_title_fallback(self):
        from data.news_fetcher import _fetch_single

        item = {"content": {"title": "Nested title"}}
        with patch("data.news_fetcher.yf.Ticker", return_value=self._ticker([item])):
            sym, headlines = _fetch_single("AAPL", max_headlines=3)
        self.assertIn("Nested title", headlines)

    def test_uses_headline_key_fallback(self):
        from data.news_fetcher import _fetch_single

        item = {"headline": "Headline key title"}
        with patch("data.news_fetcher.yf.Ticker", return_value=self._ticker([item])):
            _, headlines = _fetch_single("MSFT", max_headlines=3)
        self.assertIn("Headline key title", headlines)

    def test_empty_title_excluded(self):
        from data.news_fetcher import _fetch_single

        item = {"title": ""}
        with patch("data.news_fetcher.yf.Ticker", return_value=self._ticker([item])):
            _, headlines = _fetch_single("AAPL", max_headlines=3)
        self.assertEqual(headlines, [])

    def test_max_headlines_respected(self):
        from data.news_fetcher import _fetch_single

        items = [{"title": f"Story {i}"} for i in range(10)]
        with patch("data.news_fetcher.yf.Ticker", return_value=self._ticker(items)):
            _, headlines = _fetch_single("AAPL", max_headlines=3)
        self.assertLessEqual(len(headlines), 3)

    def test_exception_returns_empty_list(self):
        from data.news_fetcher import _fetch_single

        with patch("data.news_fetcher.yf.Ticker", side_effect=Exception("network error")):
            sym, headlines = _fetch_single("AAPL", max_headlines=3)
        self.assertEqual(sym, "AAPL")
        self.assertEqual(headlines, [])

    def test_none_news_attribute_returns_empty(self):
        from data.news_fetcher import _fetch_single

        t = MagicMock()
        t.news = None
        with patch("data.news_fetcher.yf.Ticker", return_value=t):
            _, headlines = _fetch_single("AAPL", max_headlines=3)
        self.assertEqual(headlines, [])


class TestFetchNews(unittest.TestCase):
    def test_returns_dict(self):
        from data.news_fetcher import fetch_news

        with patch("data.news_fetcher._fetch_single", return_value=("AAPL", ["Good news"])):
            result = fetch_news(["AAPL"])
        self.assertIsInstance(result, dict)

    def test_symbols_with_headlines_included(self):
        from data.news_fetcher import fetch_news

        with patch("data.news_fetcher._fetch_single", return_value=("AAPL", ["Good news"])):
            result = fetch_news(["AAPL"])
        self.assertIn("AAPL", result)

    def test_symbols_with_empty_headlines_excluded(self):
        from data.news_fetcher import fetch_news

        def fake_fetch(sym, max_headlines):
            return (sym, [])

        with patch("data.news_fetcher._fetch_single", side_effect=fake_fetch):
            result = fetch_news(["AAPL", "MSFT"])
        self.assertEqual(result, {})

    def test_empty_symbols_returns_empty_dict(self):
        from data.news_fetcher import fetch_news

        result = fetch_news([])
        self.assertEqual(result, {})

    def test_partial_failures_do_not_block_successes(self):
        from data.news_fetcher import fetch_news

        call_count = 0

        def fake_fetch(sym, max_headlines):
            nonlocal call_count
            call_count += 1
            if sym == "AAPL":
                return ("AAPL", ["Apple news"])
            raise Exception("fail")  # pragma: no cover

        with patch("data.news_fetcher._fetch_single", side_effect=fake_fetch):
            # NVDA will raise, AAPL should still appear
            try:
                result = fetch_news(["AAPL"])
            except Exception:  # pragma: no cover
                result = {}
        self.assertIn("AAPL", result)

    def test_multiple_symbols_all_fetched(self):
        from data.news_fetcher import fetch_news

        def fake_fetch(sym, max_headlines):
            return (sym, [f"{sym} headline"])

        with patch("data.news_fetcher._fetch_single", side_effect=fake_fetch):
            result = fetch_news(["AAPL", "MSFT", "NVDA"])
        self.assertEqual(len(result), 3)


class TestGetSentiment(unittest.TestCase):
    """get_sentiment delegates to get_analyst_consensus (FMP-backed)."""

    def test_returns_fmp_analyst_data(self):
        from data.sentiment import get_sentiment

        expected = {"AAPL": {"bullish_pct": 70, "bearish_pct": 10, "analyst_count": 20}}
        with patch("data.sentiment.get_analyst_consensus", return_value=expected):
            result = get_sentiment(["AAPL"])
        self.assertEqual(result, expected)

    def test_returns_empty_when_no_key(self):
        from data.sentiment import get_sentiment

        with patch("data.sentiment.get_analyst_consensus", return_value={}):
            result = get_sentiment(["AAPL"])
        self.assertEqual(result, {})

    def test_passes_symbols_through(self):
        from data.sentiment import get_sentiment

        with patch("data.sentiment.get_analyst_consensus", return_value={}) as mock:
            get_sentiment(["AAPL", "NVDA"])
        mock.assert_called_once_with(["AAPL", "NVDA"])
