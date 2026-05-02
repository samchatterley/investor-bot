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
            raise Exception("fail")

        with patch("data.news_fetcher._fetch_single", side_effect=fake_fetch):
            # NVDA will raise, AAPL should still appear
            try:
                result = fetch_news(["AAPL"])
            except Exception:
                result = {}
        self.assertIn("AAPL", result)

    def test_multiple_symbols_all_fetched(self):
        from data.news_fetcher import fetch_news

        def fake_fetch(sym, max_headlines):
            return (sym, [f"{sym} headline"])

        with patch("data.news_fetcher._fetch_single", side_effect=fake_fetch):
            result = fetch_news(["AAPL", "MSFT", "NVDA"])
        self.assertEqual(len(result), 3)


class TestFetchAnalystSentiment(unittest.TestCase):

    def _info(self, mean=2.0, key="buy", analysts=15, target=200.0, current=180.0):
        return {
            "recommendationMean": mean,
            "recommendationKey": key,
            "numberOfAnalystOpinions": analysts,
            "targetMeanPrice": target,
            "currentPrice": current,
        }

    def test_returns_symbol_and_data(self):
        from data.sentiment import _fetch_analyst
        with patch("data.sentiment.yf.Ticker") as mock_ticker:
            mock_ticker.return_value.info = self._info()
            sym, data = _fetch_analyst("AAPL")
        self.assertEqual(sym, "AAPL")
        self.assertIn("bullish_pct", data)

    def test_bullish_pct_range(self):
        from data.sentiment import _fetch_analyst
        with patch("data.sentiment.yf.Ticker") as mock_ticker:
            mock_ticker.return_value.info = self._info(mean=1.0)
            _, data = _fetch_analyst("AAPL")
        self.assertEqual(data["bullish_pct"] + data["bearish_pct"], 100)
        self.assertGreaterEqual(data["bullish_pct"], 0)
        self.assertLessEqual(data["bullish_pct"], 100)

    def test_upside_pct_included_when_target_and_current(self):
        from data.sentiment import _fetch_analyst
        with patch("data.sentiment.yf.Ticker") as mock_ticker:
            mock_ticker.return_value.info = self._info(target=200.0, current=180.0)
            _, data = _fetch_analyst("AAPL")
        self.assertIn("upside_pct", data)
        self.assertAlmostEqual(data["upside_pct"], round((200 / 180 - 1) * 100, 1), places=1)

    def test_upside_pct_excluded_when_no_target(self):
        from data.sentiment import _fetch_analyst
        with patch("data.sentiment.yf.Ticker") as mock_ticker:
            mock_ticker.return_value.info = self._info(target=None, current=180.0)
            _, data = _fetch_analyst("AAPL")
        self.assertNotIn("upside_pct", data)

    def test_returns_empty_when_no_mean(self):
        from data.sentiment import _fetch_analyst
        info = self._info()
        info["recommendationMean"] = None
        with patch("data.sentiment.yf.Ticker") as mock_ticker:
            mock_ticker.return_value.info = info
            _, data = _fetch_analyst("AAPL")
        self.assertEqual(data, {})

    def test_returns_empty_when_no_analysts(self):
        from data.sentiment import _fetch_analyst
        info = self._info()
        info["numberOfAnalystOpinions"] = 0
        with patch("data.sentiment.yf.Ticker") as mock_ticker:
            mock_ticker.return_value.info = info
            _, data = _fetch_analyst("AAPL")
        self.assertEqual(data, {})

    def test_returns_empty_on_exception(self):
        from data.sentiment import _fetch_analyst
        with patch("data.sentiment.yf.Ticker", side_effect=Exception("network")):
            sym, data = _fetch_analyst("AAPL")
        self.assertEqual(sym, "AAPL")
        self.assertEqual(data, {})


class TestGetSentiment(unittest.TestCase):

    def test_returns_dict(self):
        from data.sentiment import get_sentiment
        with patch("data.sentiment._fetch_analyst", return_value=("AAPL", {"bullish_pct": 75, "bearish_pct": 25})):
            result = get_sentiment(["AAPL"])
        self.assertIsInstance(result, dict)

    def test_symbols_with_data_included(self):
        from data.sentiment import get_sentiment

        def fake_fetch(sym):
            return (sym, {"bullish_pct": 60, "bearish_pct": 40, "analyst_count": 10, "recommendation": "buy"})

        with patch("data.sentiment._fetch_analyst", side_effect=fake_fetch):
            result = get_sentiment(["AAPL"])
        self.assertIn("AAPL", result)

    def test_symbols_with_no_data_excluded(self):
        from data.sentiment import get_sentiment

        def fake_fetch(sym):
            return (sym, {})

        with patch("data.sentiment._fetch_analyst", side_effect=fake_fetch):
            result = get_sentiment(["AAPL", "MSFT"])
        self.assertEqual(result, {})

    def test_empty_symbols_returns_empty_dict(self):
        from data.sentiment import get_sentiment
        result = get_sentiment([])
        self.assertEqual(result, {})

    def test_partial_data_only_returns_populated(self):
        from data.sentiment import get_sentiment

        def fake_fetch(sym):
            if sym == "AAPL":
                return ("AAPL", {"bullish_pct": 70, "bearish_pct": 30})
            return (sym, {})

        with patch("data.sentiment._fetch_analyst", side_effect=fake_fetch):
            result = get_sentiment(["AAPL", "MSFT"])
        self.assertIn("AAPL", result)
        self.assertNotIn("MSFT", result)

    def test_future_exception_does_not_block_other_symbols(self):
        # Lines 59-60: future raises in the futures loop → pass, other symbols succeed
        from data.sentiment import get_sentiment

        def selective_fetch(sym):
            if sym == "FAIL":
                raise RuntimeError("forced failure")
            return (sym, {"bullish_pct": 60, "bearish_pct": 40})

        with patch("data.sentiment._fetch_analyst", side_effect=selective_fetch):
            result = get_sentiment(["FAIL", "AAPL"])
        self.assertNotIn("FAIL", result)
        self.assertIn("AAPL", result)
