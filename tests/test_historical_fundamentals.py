"""Tests for backtest/historical_fundamentals.py — point-in-time fundamental pre-fetchers."""

import unittest
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd

from backtest.historical_fundamentals import (
    insider_state_on_date,
    pead_active_on_date,
    prefetch_earnings_history,
    prefetch_insider_history,
)


def _make_earnings_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal earnings_dates DataFrame matching yfinance's format."""
    index = pd.DatetimeIndex([pd.Timestamp(r["date"]) for r in rows], name="Earnings Date")
    return pd.DataFrame(
        {
            "EPS Estimate": [r.get("estimate") for r in rows],
            "Reported EPS": [r.get("reported") for r in rows],
            "Surprise(%)": [r.get("surprise") for r in rows],
        },
        index=index,
    )


# ── prefetch_earnings_history ─────────────────────────────────────────────────


class TestPrefetchEarningsHistory(unittest.TestCase):
    def _mock_ticker(self, df):
        m = MagicMock()
        m.earnings_dates = df
        return m

    def test_returns_dict_of_lists(self):
        df = _make_earnings_df(
            [{"date": "2025-01-15", "estimate": 1.0, "reported": 1.1, "surprise": 10.0}]
        )
        with patch(
            "backtest.historical_fundamentals.yf.Ticker", return_value=self._mock_ticker(df)
        ):
            result = prefetch_earnings_history(["AAPL"])
        self.assertIsInstance(result, dict)
        self.assertIn("AAPL", result)
        self.assertIsInstance(result["AAPL"], list)

    def test_event_has_date_and_surprise_pct(self):
        df = _make_earnings_df(
            [{"date": "2025-01-15", "estimate": 1.0, "reported": 1.1, "surprise": 10.0}]
        )
        with patch(
            "backtest.historical_fundamentals.yf.Ticker", return_value=self._mock_ticker(df)
        ):
            result = prefetch_earnings_history(["AAPL"])
        evt = result["AAPL"][0]
        self.assertIn("date", evt)
        self.assertIn("surprise_pct", evt)
        self.assertAlmostEqual(evt["surprise_pct"], 10.0)

    def test_sorted_oldest_first(self):
        df = _make_earnings_df(
            [
                {"date": "2025-06-01", "estimate": 1.0, "reported": 1.1, "surprise": 8.0},
                {"date": "2025-01-15", "estimate": 1.0, "reported": 1.1, "surprise": 6.0},
            ]
        )
        with patch(
            "backtest.historical_fundamentals.yf.Ticker", return_value=self._mock_ticker(df)
        ):
            result = prefetch_earnings_history(["AAPL"])
        dates = [e["date"] for e in result["AAPL"]]
        self.assertEqual(dates, sorted(dates))

    def test_drops_rows_with_nan_surprise(self):
        df = _make_earnings_df(
            [
                {"date": "2025-01-15", "estimate": 1.0, "reported": None, "surprise": None},
                {"date": "2025-04-15", "estimate": 1.0, "reported": 1.1, "surprise": 7.0},
            ]
        )
        with patch(
            "backtest.historical_fundamentals.yf.Ticker", return_value=self._mock_ticker(df)
        ):
            result = prefetch_earnings_history(["AAPL"])
        self.assertEqual(len(result["AAPL"]), 1)
        self.assertAlmostEqual(result["AAPL"][0]["surprise_pct"], 7.0)

    def test_symbol_absent_when_fetch_raises(self):
        with patch(
            "backtest.historical_fundamentals.yf.Ticker",
            side_effect=Exception("network error"),
        ):
            result = prefetch_earnings_history(["AAPL"])
        self.assertNotIn("AAPL", result)

    def test_symbol_absent_when_no_data(self):
        m = MagicMock()
        m.earnings_dates = None
        with patch("backtest.historical_fundamentals.yf.Ticker", return_value=m):
            result = prefetch_earnings_history(["AAPL"])
        self.assertNotIn("AAPL", result)

    def test_multiple_symbols_fetched(self):
        df = _make_earnings_df(
            [{"date": "2025-01-15", "estimate": 1.0, "reported": 1.1, "surprise": 6.0}]
        )
        with patch(
            "backtest.historical_fundamentals.yf.Ticker", return_value=self._mock_ticker(df)
        ):
            result = prefetch_earnings_history(["AAPL", "MSFT"])
        self.assertIn("AAPL", result)
        self.assertIn("MSFT", result)


# ── pead_active_on_date ───────────────────────────────────────────────────────


class TestPeadActiveOnDate(unittest.TestCase):
    _BASE = date(2025, 6, 1)

    def _hist(self, **kwargs):
        days_ago = kwargs.get("days_ago", 10)
        surprise = kwargs.get("surprise", 8.0)
        return {"AAPL": [{"date": self._BASE - timedelta(days=days_ago), "surprise_pct": surprise}]}

    def test_returns_true_within_window(self):
        self.assertTrue(pead_active_on_date("AAPL", self._BASE, self._hist(days_ago=10)))

    def test_returns_false_outside_window(self):
        self.assertFalse(pead_active_on_date("AAPL", self._BASE, self._hist(days_ago=35)))

    def test_returns_false_below_min_surprise(self):
        self.assertFalse(
            pead_active_on_date("AAPL", self._BASE, self._hist(days_ago=5, surprise=3.0))
        )

    def test_returns_false_on_exact_sim_date(self):
        hist = {"AAPL": [{"date": self._BASE, "surprise_pct": 10.0}]}
        self.assertFalse(pead_active_on_date("AAPL", self._BASE, hist))

    def test_returns_false_when_symbol_missing(self):
        self.assertFalse(pead_active_on_date("MSFT", self._BASE, self._hist()))

    def test_returns_false_when_history_empty(self):
        self.assertFalse(pead_active_on_date("AAPL", self._BASE, {}))

    def test_custom_lookback(self):
        hist = {"AAPL": [{"date": self._BASE - timedelta(days=7), "surprise_pct": 8.0}]}
        self.assertFalse(pead_active_on_date("AAPL", self._BASE, hist, lookback_days=5))
        self.assertTrue(pead_active_on_date("AAPL", self._BASE, hist, lookback_days=10))

    def test_custom_min_surprise(self):
        hist = {"AAPL": [{"date": self._BASE - timedelta(days=5), "surprise_pct": 4.0}]}
        self.assertFalse(pead_active_on_date("AAPL", self._BASE, hist, min_surprise=5.0))
        self.assertTrue(pead_active_on_date("AAPL", self._BASE, hist, min_surprise=3.0))

    def test_picks_up_any_qualifying_event_in_window(self):
        hist = {
            "AAPL": [
                {"date": self._BASE - timedelta(days=25), "surprise_pct": 2.0},
                {"date": self._BASE - timedelta(days=15), "surprise_pct": 9.0},
            ]
        }
        self.assertTrue(pead_active_on_date("AAPL", self._BASE, hist))


# ── prefetch_insider_history ──────────────────────────────────────────────────


class TestPrefetchInsiderHistory(unittest.TestCase):
    def test_returns_dict_of_lists(self):
        with (
            patch(
                "backtest.historical_fundamentals._get_cik_map",
                return_value={"AAPL": "0000320193"},
            ),
            patch(
                "backtest.historical_fundamentals._recent_form4_filings",
                return_value=[
                    {
                        "filing_date": "2025-02-10",
                        "accession": "0001234567-25-001",
                        "doc": "form4.xml",
                    }
                ],
            ),
            patch(
                "backtest.historical_fundamentals._parse_form4",
                return_value=[{"reporter": "CEO John", "shares": 1000.0, "price": 150.0}],
            ),
        ):
            result = prefetch_insider_history(["AAPL"])
        self.assertIn("AAPL", result)
        self.assertIsInstance(result["AAPL"], list)

    def test_event_has_expected_fields(self):
        with (
            patch(
                "backtest.historical_fundamentals._get_cik_map",
                return_value={"AAPL": "0000320193"},
            ),
            patch(
                "backtest.historical_fundamentals._recent_form4_filings",
                return_value=[
                    {
                        "filing_date": "2025-02-10",
                        "accession": "0001234567-25-001",
                        "doc": "form4.xml",
                    }
                ],
            ),
            patch(
                "backtest.historical_fundamentals._parse_form4",
                return_value=[{"reporter": "CEO John", "shares": 500.0, "price": 200.0}],
            ),
        ):
            result = prefetch_insider_history(["AAPL"])
        evt = result["AAPL"][0]
        self.assertIn("filing_date", evt)
        self.assertIn("reporter", evt)
        self.assertIn("shares", evt)
        self.assertIn("price", evt)

    def test_symbol_absent_when_no_cik(self):
        with patch("backtest.historical_fundamentals._get_cik_map", return_value={}):
            result = prefetch_insider_history(["UNKNOWN"])
        self.assertNotIn("UNKNOWN", result)

    def test_symbol_absent_when_no_filings(self):
        with (
            patch(
                "backtest.historical_fundamentals._get_cik_map",
                return_value={"AAPL": "0000320193"},
            ),
            patch(
                "backtest.historical_fundamentals._recent_form4_filings",
                return_value=[],
            ),
        ):
            result = prefetch_insider_history(["AAPL"])
        self.assertNotIn("AAPL", result)

    def test_sorted_oldest_first(self):
        with (
            patch(
                "backtest.historical_fundamentals._get_cik_map",
                return_value={"AAPL": "0000320193"},
            ),
            patch(
                "backtest.historical_fundamentals._recent_form4_filings",
                return_value=[
                    {
                        "filing_date": "2025-06-01",
                        "accession": "acc-1",
                        "doc": "f4.xml",
                    },
                    {
                        "filing_date": "2025-01-15",
                        "accession": "acc-2",
                        "doc": "f4.xml",
                    },
                ],
            ),
            patch(
                "backtest.historical_fundamentals._parse_form4",
                return_value=[{"reporter": "CFO", "shares": 100.0, "price": 150.0}],
            ),
        ):
            result = prefetch_insider_history(["AAPL"])
        dates = [e["filing_date"] for e in result["AAPL"]]
        self.assertEqual(dates, sorted(dates))

    def test_invalid_filing_date_skipped(self):
        with (
            patch(
                "backtest.historical_fundamentals._get_cik_map",
                return_value={"AAPL": "0000320193"},
            ),
            patch(
                "backtest.historical_fundamentals._recent_form4_filings",
                return_value=[{"filing_date": "not-a-date", "accession": "acc-1", "doc": "f4.xml"}],
            ),
            patch(
                "backtest.historical_fundamentals._parse_form4",
                return_value=[{"reporter": "CEO", "shares": 100.0, "price": 50.0}],
            ),
        ):
            result = prefetch_insider_history(["AAPL"])
        self.assertNotIn("AAPL", result)


# ── insider_state_on_date ─────────────────────────────────────────────────────


class TestInsiderStateOnDate(unittest.TestCase):
    _BASE = date(2025, 6, 1)

    def _hist(self, txns):
        return {"AAPL": txns}

    def test_cluster_detected_with_two_insiders(self):
        hist = self._hist(
            [
                {
                    "filing_date": self._BASE - timedelta(days=5),
                    "reporter": "CEO A",
                    "shares": 100.0,
                    "price": 100.0,
                },
                {
                    "filing_date": self._BASE - timedelta(days=3),
                    "reporter": "CFO B",
                    "shares": 200.0,
                    "price": 100.0,
                },
            ]
        )
        state = insider_state_on_date("AAPL", self._BASE, hist)
        self.assertTrue(state["insider_cluster"])
        self.assertEqual(state["insider_unique_insiders"], 2)

    def test_no_cluster_with_one_insider(self):
        hist = self._hist(
            [
                {
                    "filing_date": self._BASE - timedelta(days=5),
                    "reporter": "CEO A",
                    "shares": 100.0,
                    "price": 100.0,
                }
            ]
        )
        state = insider_state_on_date("AAPL", self._BASE, hist)
        self.assertFalse(state["insider_cluster"])
        self.assertEqual(state["insider_unique_insiders"], 1)

    def test_large_buy_detected(self):
        hist = self._hist(
            [
                {
                    "filing_date": self._BASE - timedelta(days=5),
                    "reporter": "CEO A",
                    "shares": 1000.0,
                    "price": 200.0,
                }
            ]
        )
        state = insider_state_on_date("AAPL", self._BASE, hist)
        self.assertTrue(state["insider_large_buy"])

    def test_no_large_buy_below_threshold(self):
        hist = self._hist(
            [
                {
                    "filing_date": self._BASE - timedelta(days=5),
                    "reporter": "CEO A",
                    "shares": 10.0,
                    "price": 50.0,
                }
            ]
        )
        state = insider_state_on_date("AAPL", self._BASE, hist)
        self.assertFalse(state["insider_large_buy"])

    def test_txn_on_sim_date_excluded(self):
        hist = self._hist(
            [
                {
                    "filing_date": self._BASE,
                    "reporter": "CEO A",
                    "shares": 1000.0,
                    "price": 200.0,
                },
                {
                    "filing_date": self._BASE,
                    "reporter": "CFO B",
                    "shares": 1000.0,
                    "price": 200.0,
                },
            ]
        )
        state = insider_state_on_date("AAPL", self._BASE, hist)
        self.assertFalse(state["insider_cluster"])

    def test_txn_outside_lookback_excluded(self):
        hist = self._hist(
            [
                {
                    "filing_date": self._BASE - timedelta(days=20),
                    "reporter": "CEO A",
                    "shares": 1000.0,
                    "price": 200.0,
                },
                {
                    "filing_date": self._BASE - timedelta(days=20),
                    "reporter": "CFO B",
                    "shares": 1000.0,
                    "price": 200.0,
                },
            ]
        )
        state = insider_state_on_date("AAPL", self._BASE, hist)
        self.assertFalse(state["insider_cluster"])

    def test_returns_empty_state_when_symbol_missing(self):
        state = insider_state_on_date("MSFT", self._BASE, {})
        self.assertFalse(state["insider_cluster"])
        self.assertFalse(state["insider_large_buy"])
        self.assertEqual(state["insider_unique_insiders"], 0)

    def test_returns_empty_state_when_no_txns_in_window(self):
        state = insider_state_on_date("AAPL", self._BASE, {"AAPL": []})
        self.assertFalse(state["insider_cluster"])

    def test_same_reporter_twice_counts_as_one(self):
        hist = self._hist(
            [
                {
                    "filing_date": self._BASE - timedelta(days=5),
                    "reporter": "CEO A",
                    "shares": 100.0,
                    "price": 100.0,
                },
                {
                    "filing_date": self._BASE - timedelta(days=3),
                    "reporter": "CEO A",
                    "shares": 200.0,
                    "price": 100.0,
                },
            ]
        )
        state = insider_state_on_date("AAPL", self._BASE, hist)
        self.assertFalse(state["insider_cluster"])
        self.assertEqual(state["insider_unique_insiders"], 1)
