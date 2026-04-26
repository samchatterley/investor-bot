import unittest
from datetime import date
from unittest.mock import patch, MagicMock

import pandas as pd

from risk.earnings_calendar import (
    get_next_earnings_date,
    days_until_earnings,
    get_earnings_risk_positions,
)

_TODAY = date(2026, 4, 26)


def _mock_ticker(earnings_date):
    """Build a mock yf.Ticker whose .calendar returns the given date."""
    mock = MagicMock()
    if earnings_date is None:
        mock.calendar = None
    else:
        mock.calendar = {"Earnings Date": [pd.Timestamp(earnings_date.isoformat())]}
    return mock


class TestGetNextEarningsDate(unittest.TestCase):

    def test_returns_future_date(self):
        future = date(2026, 5, 15)
        with patch("risk.earnings_calendar.yf.Ticker", return_value=_mock_ticker(future)):
            result = get_next_earnings_date("AAPL")
        self.assertEqual(result, future)

    def test_returns_none_when_calendar_is_none(self):
        with patch("risk.earnings_calendar.yf.Ticker", return_value=_mock_ticker(None)):
            result = get_next_earnings_date("AAPL")
        self.assertIsNone(result)

    def test_returns_none_on_yfinance_exception(self):
        with patch("risk.earnings_calendar.yf.Ticker", side_effect=Exception("API down")):
            result = get_next_earnings_date("AAPL")
        self.assertIsNone(result)


class TestDaysUntilEarnings(unittest.TestCase):

    def test_future_earnings_returns_positive(self):
        future = date(2026, 4, 28)
        with patch("risk.earnings_calendar.yf.Ticker", return_value=_mock_ticker(future)):
            with patch("risk.earnings_calendar.today_et", return_value=_TODAY):
                result = days_until_earnings("AAPL")
        self.assertEqual(result, 2)

    def test_past_earnings_returns_none(self):
        past = date(2026, 4, 20)
        with patch("risk.earnings_calendar.yf.Ticker", return_value=_mock_ticker(past)):
            with patch("risk.earnings_calendar.today_et", return_value=_TODAY):
                result = days_until_earnings("AAPL")
        self.assertIsNone(result)

    def test_today_earnings_returns_zero(self):
        with patch("risk.earnings_calendar.yf.Ticker", return_value=_mock_ticker(_TODAY)):
            with patch("risk.earnings_calendar.today_et", return_value=_TODAY):
                result = days_until_earnings("AAPL")
        self.assertEqual(result, 0)

    def test_no_data_returns_none(self):
        with patch("risk.earnings_calendar.yf.Ticker", return_value=_mock_ticker(None)):
            result = days_until_earnings("AAPL")
        self.assertIsNone(result)


class TestGetEarningsRiskPositions(unittest.TestCase):

    def test_position_within_warning_days_flagged(self):
        tomorrow = date(2026, 4, 27)
        with patch("risk.earnings_calendar.yf.Ticker", return_value=_mock_ticker(tomorrow)):
            with patch("risk.earnings_calendar.today_et", return_value=_TODAY):
                at_risk = get_earnings_risk_positions(["AAPL"], warning_days=2)
        self.assertIn("AAPL", at_risk)

    def test_position_outside_warning_days_not_flagged(self):
        far_future = date(2026, 5, 30)
        with patch("risk.earnings_calendar.yf.Ticker", return_value=_mock_ticker(far_future)):
            with patch("risk.earnings_calendar.today_et", return_value=_TODAY):
                at_risk = get_earnings_risk_positions(["AAPL"], warning_days=2)
        self.assertNotIn("AAPL", at_risk)

    def test_no_earnings_data_not_flagged(self):
        with patch("risk.earnings_calendar.yf.Ticker", return_value=_mock_ticker(None)):
            with patch("risk.earnings_calendar.today_et", return_value=_TODAY):
                at_risk = get_earnings_risk_positions(["AAPL"], warning_days=2)
        self.assertNotIn("AAPL", at_risk)

    def test_empty_symbols_returns_empty(self):
        at_risk = get_earnings_risk_positions([], warning_days=2)
        self.assertEqual(at_risk, {})
