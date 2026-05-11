"""
Regression tests for the six incidents found on the first live paper-trading run
(2026-04-27). Each class maps to one incident. These tests exist to prevent
silent regressions if the affected code is refactored.
"""

import math
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Incident 1 — Python 3.9 compatibility (dict | None annotations)
# ---------------------------------------------------------------------------


class TestPython39Compat(unittest.TestCase):
    """
    The '|' union syntax for type hints requires Python 3.10+.
    'from __future__ import annotations' must be present in every module
    that uses it so annotations are treated as strings, not evaluated.

    If any of these imports raise TypeError, the fix has been removed.
    """

    def test_emailer_imports_without_error(self):
        try:
            import notifications.emailer  # noqa: F401
        except TypeError as e:
            self.fail(f"emailer.py raised TypeError on import (Python 3.9 compat broken): {e}")

    def test_dashboard_imports_without_error(self):
        try:
            import dashboard  # noqa: F401
        except TypeError as e:
            self.fail(f"dashboard.py raised TypeError on import (Python 3.9 compat broken): {e}")
        except ImportError:
            self.skipTest("Dashboard optional dependencies not installed in this environment")

    def test_sentiment_imports_without_error(self):
        try:
            import data.sentiment  # noqa: F401
        except TypeError as e:
            self.fail(f"data/sentiment.py raised TypeError on import: {e}")

    def test_trader_imports_without_error(self):
        try:
            import execution.trader  # noqa: F401
        except TypeError as e:
            self.fail(f"execution/trader.py raised TypeError on import: {e}")


# ---------------------------------------------------------------------------
# Incident 2 — News fetcher: yfinance API format change
# ---------------------------------------------------------------------------


class TestNewsFetcherTitleFallback(unittest.TestCase):
    """
    yfinance moved news titles from item['title'] to item['content']['title'].
    The fetcher must handle both formats and not crash on either being absent.
    """

    def _call(self, items, max_headlines=5):
        from data.news_fetcher import _fetch_single

        ticker_mock = MagicMock()
        ticker_mock.news = items
        with patch("data.news_fetcher.yf.Ticker", return_value=ticker_mock):
            _, headlines = _fetch_single("AAPL", max_headlines)
        return headlines

    def test_old_format_title_key(self):
        headlines = self._call([{"title": "Stock rises on earnings"}])
        self.assertEqual(headlines, ["Stock rises on earnings"])

    def test_new_format_content_title_key(self):
        headlines = self._call([{"content": {"title": "Stock rises on earnings"}}])
        self.assertEqual(headlines, ["Stock rises on earnings"])

    def test_headline_key_also_accepted(self):
        headlines = self._call([{"headline": "Stock rises on earnings"}])
        self.assertEqual(headlines, ["Stock rises on earnings"])

    def test_old_format_takes_priority_over_new(self):
        # If both present, top-level title wins (it's more likely to be correct)
        headlines = self._call([{"title": "Top level", "content": {"title": "Nested"}}])
        self.assertEqual(headlines, ["Top level"])

    def test_item_with_no_title_is_skipped(self):
        headlines = self._call([{"link": "https://example.com"}])
        self.assertEqual(headlines, [])

    def test_item_with_none_content_does_not_crash(self):
        headlines = self._call([{"content": None}])
        self.assertEqual(headlines, [])

    def test_mixed_items_only_valid_titles_returned(self):
        items = [
            {"title": "Good headline"},
            {"link": "no title"},
            {"content": {"title": "Also good"}},
        ]
        headlines = self._call(items)
        self.assertEqual(headlines, ["Good headline", "Also good"])

    def test_max_headlines_respected(self):
        items = [{"title": f"Headline {i}"} for i in range(10)]
        headlines = self._call(items, max_headlines=3)
        self.assertEqual(len(headlines), 3)


# ---------------------------------------------------------------------------
# Incident 3 — Sentiment fetcher: Stocktwits replaced with yfinance analyst data
# ---------------------------------------------------------------------------


class TestSentimentAnalystConversion(unittest.TestCase):
    """get_sentiment delegates to FMP analyst consensus."""

    def test_get_sentiment_delegates_to_fmp(self):
        from data.sentiment import get_sentiment

        expected = {"AAPL": {"bullish_pct": 75, "bearish_pct": 5, "analyst_count": 30}}
        with patch("data.sentiment.get_analyst_consensus", return_value=expected):
            result = get_sentiment(["AAPL"])
        self.assertEqual(result, expected)

    def test_get_sentiment_empty_symbols(self):
        from data.sentiment import get_sentiment

        with patch("data.sentiment.get_analyst_consensus", return_value={}):
            result = get_sentiment([])
        self.assertEqual(result, {})


# ---------------------------------------------------------------------------
# Incident 4 — Trailing stops rejected for fractional positions
# Incident 5 — Stop qty rounding above available quantity
# ---------------------------------------------------------------------------


class TestPlaceTrailingStopFractionalRouting(unittest.TestCase):
    """
    Alpaca rejects TrailingStopOrderRequest for fractional share quantities.
    place_trailing_stop must route to StopOrderRequest for fractional qty
    and TrailingStopOrderRequest for whole-share qty.
    """

    def _make_client(self):
        client = MagicMock()
        order = MagicMock()
        order.id = "test-order-id"
        client.submit_order.return_value = order
        return client

    def test_fractional_qty_uses_stop_order(self):
        from alpaca.trading.requests import StopOrderRequest

        from execution.trader import place_trailing_stop

        client = self._make_client()
        place_trailing_stop(client, "NVDA", qty=132.652248, current_price=210.0)
        # First call is the stop; second liquidates the fractional remainder
        stop_req = client.submit_order.call_args_list[0][0][0]
        self.assertIsInstance(stop_req, StopOrderRequest)

    def test_whole_qty_uses_trailing_stop_order(self):
        from alpaca.trading.requests import TrailingStopOrderRequest

        from execution.trader import place_trailing_stop

        client = self._make_client()
        place_trailing_stop(client, "AAPL", qty=10.0, current_price=180.0)
        submitted = client.submit_order.call_args[0][0]
        self.assertIsInstance(submitted, TrailingStopOrderRequest)

    def test_fractional_without_current_price_returns_stop_failed(self):
        from execution.trader import place_trailing_stop
        from models import OrderStatus

        client = self._make_client()
        result = place_trailing_stop(client, "NVDA", qty=132.652248, current_price=None)
        self.assertIsNotNone(result)
        self.assertEqual(result.status, OrderStatus.STOP_FAILED)
        client.submit_order.assert_not_called()

    def test_fractional_stop_price_calculated_from_current_price(self):
        from config import TRAILING_STOP_PCT
        from execution.trader import place_trailing_stop

        client = self._make_client()
        current_price = 200.0
        place_trailing_stop(client, "NVDA", qty=5.5, current_price=current_price)
        # First call is the stop order
        stop_req = client.submit_order.call_args_list[0][0][0]
        expected_stop = round(current_price * (1 - TRAILING_STOP_PCT / 100), 2)
        self.assertAlmostEqual(float(stop_req.stop_price), expected_stop, places=2)

    def test_zero_qty_returns_none(self):
        from execution.trader import place_trailing_stop

        client = self._make_client()
        result = place_trailing_stop(client, "AAPL", qty=0.0)
        self.assertIsNone(result)
        client.submit_order.assert_not_called()

    def test_negative_qty_returns_none(self):
        from execution.trader import place_trailing_stop

        client = self._make_client()
        result = place_trailing_stop(client, "AAPL", qty=-1.0)
        self.assertIsNone(result)
        client.submit_order.assert_not_called()


class TestStopQtyTruncation(unittest.TestCase):
    """
    Incident 5: round(64.075231525, 6) = 64.075232 which exceeds the available
    quantity Alpaca reports. Submitted qty must be floored, never rounded up.
    """

    def _safe_qty(self, qty):
        return math.floor(qty * 1_000_000) / 1_000_000

    def test_truncation_never_exceeds_input(self):
        for qty in [64.075231525, 132.652248, 10.999999999, 1.0000005]:
            safe = self._safe_qty(qty)
            self.assertLessEqual(safe, qty, f"safe_qty({qty}) = {safe} exceeds input")

    def test_known_lmt_case(self):
        # The exact value that caused the Alpaca rejection on 2026-04-27
        qty = 64.075231525
        safe = self._safe_qty(qty)
        self.assertEqual(safe, 64.075231)
        self.assertLess(safe, qty)

    def test_whole_number_unchanged(self):
        self.assertEqual(self._safe_qty(10.0), 10.0)

    def test_already_6_decimals_unchanged(self):
        self.assertEqual(self._safe_qty(5.123456), 5.123456)

    def test_submitted_qty_does_not_exceed_position_qty(self):

        from execution.trader import place_trailing_stop

        client = MagicMock()
        order = MagicMock()
        order.id = "x"
        client.submit_order.return_value = order

        raw_qty = 64.075231525
        place_trailing_stop(client, "LMT", qty=raw_qty, current_price=500.0)
        submitted = client.submit_order.call_args[0][0]
        self.assertLessEqual(float(submitted.qty), raw_qty)


if __name__ == "__main__":
    unittest.main()
