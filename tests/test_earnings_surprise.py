"""Tests for data/earnings_surprise.py — PEAD candidate detection."""

import unittest
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd

from data.earnings_surprise import get_earnings_surprise

_TODAY = datetime.now(UTC)


def _make_earnings_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal earnings_dates DataFrame matching yfinance's format."""
    index = pd.DatetimeIndex([pd.Timestamp(r["date"]) for r in rows], name="Earnings Date")
    df = pd.DataFrame(
        {
            "EPS Estimate": [r.get("estimate") for r in rows],
            "Reported EPS": [r.get("reported") for r in rows],
            "Surprise(%)": [r.get("surprise") for r in rows],
        },
        index=index,
    )
    return df


def _recent_date(days_ago: int) -> str:
    """Return an ISO timestamp string N days ago."""
    ts = _TODAY - timedelta(days=days_ago)
    return ts.strftime("%Y-%m-%dT%H:%M:%S-04:00")


class TestGetEarningsSurprise(unittest.TestCase):
    def test_returns_candidate_above_threshold(self):
        df = _make_earnings_df(
            [{"date": _recent_date(5), "estimate": 1.50, "reported": 1.65, "surprise": 10.0}]
        )
        with (
            patch("data.earnings_surprise.yf.Ticker") as mock_ticker,
            patch("data.earnings_surprise.time.sleep"),
        ):
            mock_ticker.return_value.earnings_dates = df
            result = get_earnings_surprise(["AAPL"])
        self.assertIn("AAPL", result)
        self.assertAlmostEqual(result["AAPL"]["earnings_surprise_pct"], 10.0)
        self.assertTrue(result["AAPL"]["pead_candidate"])

    def test_excludes_below_threshold(self):
        df = _make_earnings_df(
            [{"date": _recent_date(5), "estimate": 1.50, "reported": 1.55, "surprise": 3.33}]
        )
        with (
            patch("data.earnings_surprise.yf.Ticker") as mock_ticker,
            patch("data.earnings_surprise.time.sleep"),
        ):
            mock_ticker.return_value.earnings_dates = df
            result = get_earnings_surprise(["AAPL"])
        self.assertNotIn("AAPL", result)

    def test_excludes_outside_lookback_window(self):
        # Surprise is big but happened 45 days ago — outside default 30-day window.
        df = _make_earnings_df(
            [{"date": _recent_date(45), "estimate": 1.50, "reported": 1.65, "surprise": 10.0}]
        )
        with (
            patch("data.earnings_surprise.yf.Ticker") as mock_ticker,
            patch("data.earnings_surprise.time.sleep"),
        ):
            mock_ticker.return_value.earnings_dates = df
            result = get_earnings_surprise(["AAPL"])
        self.assertNotIn("AAPL", result)

    def test_custom_lookback_window(self):
        # With lookback_days=60, 45-day-old surprise should be included.
        df = _make_earnings_df(
            [{"date": _recent_date(45), "estimate": 1.50, "reported": 1.65, "surprise": 10.0}]
        )
        with (
            patch("data.earnings_surprise.yf.Ticker") as mock_ticker,
            patch("data.earnings_surprise.time.sleep"),
        ):
            mock_ticker.return_value.earnings_dates = df
            result = get_earnings_surprise(["AAPL"], lookback_days=60)
        self.assertIn("AAPL", result)

    def test_excludes_future_earnings_without_reported_eps(self):
        # Future earnings have NaN for Reported EPS — should be ignored.
        df = _make_earnings_df(
            [
                {
                    "date": _recent_date(-10),  # future
                    "estimate": 2.00,
                    "reported": None,
                    "surprise": None,
                },
                {
                    "date": _recent_date(5),
                    "estimate": 1.50,
                    "reported": 1.65,
                    "surprise": 10.0,
                },
            ]
        )
        with (
            patch("data.earnings_surprise.yf.Ticker") as mock_ticker,
            patch("data.earnings_surprise.time.sleep"),
        ):
            mock_ticker.return_value.earnings_dates = df
            result = get_earnings_surprise(["AAPL"])
        self.assertIn("AAPL", result)
        self.assertAlmostEqual(result["AAPL"]["earnings_surprise_pct"], 10.0)

    def test_returns_empty_when_earnings_dates_is_none(self):
        with (
            patch("data.earnings_surprise.yf.Ticker") as mock_ticker,
            patch("data.earnings_surprise.time.sleep"),
        ):
            mock_ticker.return_value.earnings_dates = None
            result = get_earnings_surprise(["AAPL"])
        self.assertNotIn("AAPL", result)

    def test_returns_empty_on_network_failure(self):
        with (
            patch(
                "data.earnings_surprise.yf.Ticker",
                side_effect=Exception("network error"),
            ),
            patch("data.earnings_surprise.time.sleep"),
        ):
            result = get_earnings_surprise(["AAPL"])
        self.assertEqual(result, {})

    def test_returns_days_ago_field(self):
        df = _make_earnings_df(
            [{"date": _recent_date(7), "estimate": 1.50, "reported": 1.65, "surprise": 10.0}]
        )
        with (
            patch("data.earnings_surprise.yf.Ticker") as mock_ticker,
            patch("data.earnings_surprise.time.sleep"),
        ):
            mock_ticker.return_value.earnings_dates = df
            result = get_earnings_surprise(["AAPL"])
        self.assertIn("earnings_days_ago", result["AAPL"])
        # Within 1 day of 7 (timezone rounding).
        self.assertAlmostEqual(result["AAPL"]["earnings_days_ago"], 7, delta=1)

    def test_returns_earnings_date_as_iso_string(self):
        df = _make_earnings_df(
            [{"date": _recent_date(5), "estimate": 1.50, "reported": 1.65, "surprise": 10.0}]
        )
        with (
            patch("data.earnings_surprise.yf.Ticker") as mock_ticker,
            patch("data.earnings_surprise.time.sleep"),
        ):
            mock_ticker.return_value.earnings_dates = df
            result = get_earnings_surprise(["AAPL"])
        date_str = result["AAPL"]["earnings_date"]
        # Should be parseable as ISO date.
        from datetime import date

        self.assertIsInstance(date.fromisoformat(date_str), date)

    def test_multiple_symbols_independent(self):
        df_beat = _make_earnings_df(
            [{"date": _recent_date(5), "estimate": 1.50, "reported": 1.65, "surprise": 10.0}]
        )
        df_miss = _make_earnings_df(
            [{"date": _recent_date(5), "estimate": 1.50, "reported": 1.45, "surprise": -3.33}]
        )

        def _ticker_factory(sym):
            m = MagicMock()
            m.earnings_dates = df_beat if sym == "AAPL" else df_miss
            return m

        with (
            patch("data.earnings_surprise.yf.Ticker", side_effect=_ticker_factory),
            patch("data.earnings_surprise.time.sleep"),
        ):
            result = get_earnings_surprise(["AAPL", "MSFT"])
        self.assertIn("AAPL", result)
        self.assertNotIn("MSFT", result)

    def test_custom_min_surprise_threshold(self):
        df = _make_earnings_df(
            [{"date": _recent_date(5), "estimate": 1.50, "reported": 1.58, "surprise": 5.33}]
        )
        with (
            patch("data.earnings_surprise.yf.Ticker") as mock_ticker,
            patch("data.earnings_surprise.time.sleep"),
        ):
            mock_ticker.return_value.earnings_dates = df
            # With min_surprise=10 this should be excluded.
            result_excluded = get_earnings_surprise(["AAPL"], min_surprise=10.0)
            # With min_surprise=3 it should be included.
            result_included = get_earnings_surprise(["AAPL"], min_surprise=3.0)
        self.assertNotIn("AAPL", result_excluded)
        self.assertIn("AAPL", result_included)

    def test_skips_etf_symbol(self):
        # Line 58: `continue` when sym in ETF_SYMBOLS — SPY is an ETF, skipped immediately
        with patch("data.earnings_surprise.time.sleep"):
            result = get_earnings_surprise(["SPY"])
        self.assertEqual(result, {})

    def test_skips_when_all_rows_have_null_reported_eps(self):
        # Line 73: `continue` when df.empty after dropna on Reported EPS / Surprise(%)
        df = _make_earnings_df(
            [
                {
                    "date": _recent_date(5),
                    "estimate": 1.50,
                    "reported": None,
                    "surprise": None,
                }
            ]
        )
        with (
            patch("data.earnings_surprise.yf.Ticker") as mock_ticker,
            patch("data.earnings_surprise.time.sleep"),
        ):
            mock_ticker.return_value.earnings_dates = df
            result = get_earnings_surprise(["AAPL"])
        self.assertNotIn("AAPL", result)
