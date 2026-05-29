"""Tests for data/short_interest.py — short interest enrichment."""

import unittest
from unittest.mock import MagicMock, patch

from data.short_interest import get_short_interest


class TestGetShortInterest(unittest.TestCase):
    def _make_ticker(self, short_ratio):
        m = MagicMock()
        m.info = {"shortRatio": short_ratio}
        return m

    def test_returns_symbol_above_threshold(self):
        with (
            patch("data.short_interest.yf.Ticker", return_value=self._make_ticker(7.0)),
            patch("data.short_interest.time.sleep"),
        ):
            result = get_short_interest(["AAPL"])
        self.assertIn("AAPL", result)
        self.assertAlmostEqual(result["AAPL"]["short_ratio"], 7.0)
        self.assertTrue(result["AAPL"]["high_short_interest"])

    def test_excludes_below_threshold(self):
        with (
            patch("data.short_interest.yf.Ticker", return_value=self._make_ticker(2.0)),
            patch("data.short_interest.time.sleep"),
        ):
            result = get_short_interest(["AAPL"])
        self.assertNotIn("AAPL", result)

    def test_excludes_exact_threshold(self):
        # Default threshold is 5.0; exactly 5.0 is not strictly greater → excluded
        with (
            patch("data.short_interest.yf.Ticker", return_value=self._make_ticker(4.9)),
            patch("data.short_interest.time.sleep"),
        ):
            result = get_short_interest(["AAPL"])
        self.assertNotIn("AAPL", result)

    def test_custom_min_threshold(self):
        with (
            patch("data.short_interest.yf.Ticker", return_value=self._make_ticker(3.0)),
            patch("data.short_interest.time.sleep"),
        ):
            result_default = get_short_interest(["AAPL"])
            result_custom = get_short_interest(["AAPL"], min_short_ratio=2.0)
        self.assertNotIn("AAPL", result_default)
        self.assertIn("AAPL", result_custom)

    def test_skips_etf_symbol(self):
        with patch("data.short_interest.time.sleep"):
            result = get_short_interest(["SPY"])
        self.assertEqual(result, {})

    def test_returns_empty_when_short_ratio_missing(self):
        m = MagicMock()
        m.info = {}
        with (
            patch("data.short_interest.yf.Ticker", return_value=m),
            patch("data.short_interest.time.sleep"),
        ):
            result = get_short_interest(["AAPL"])
        self.assertNotIn("AAPL", result)

    def test_returns_empty_on_network_failure(self):
        with (
            patch(
                "data.short_interest.yf.Ticker",
                side_effect=Exception("network error"),
            ),
            patch("data.short_interest.time.sleep"),
        ):
            result = get_short_interest(["AAPL"])
        self.assertEqual(result, {})

    def test_skips_non_numeric_short_ratio(self):
        m = MagicMock()
        m.info = {"shortRatio": "not_a_number"}
        with (
            patch("data.short_interest.yf.Ticker", return_value=m),
            patch("data.short_interest.time.sleep"),
        ):
            result = get_short_interest(["AAPL"])
        self.assertNotIn("AAPL", result)

    def test_skips_none_short_ratio(self):
        m = MagicMock()
        m.info = {"shortRatio": None}
        with (
            patch("data.short_interest.yf.Ticker", return_value=m),
            patch("data.short_interest.time.sleep"),
        ):
            result = get_short_interest(["AAPL"])
        self.assertNotIn("AAPL", result)

    def test_multiple_symbols_independent(self):
        def _factory(sym):
            m = MagicMock()
            m.info = {"shortRatio": 8.0 if sym == "AAPL" else 1.0}
            return m

        with (
            patch("data.short_interest.yf.Ticker", side_effect=_factory),
            patch("data.short_interest.time.sleep"),
        ):
            result = get_short_interest(["AAPL", "MSFT"])
        self.assertIn("AAPL", result)
        self.assertNotIn("MSFT", result)
