import unittest
from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd

from risk.earnings_calendar import (
    days_until_earnings,
    get_earnings_risk_positions,
    get_next_earnings_date,
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

    def test_returns_none_on_unexpected_exception(self):
        # except Exception as e — covers unexpected errors beyond API failures
        with patch("risk.earnings_calendar.yf.Ticker", side_effect=AttributeError("unexpected attr")):
            result = get_next_earnings_date("AAPL")
        self.assertIsNone(result)

    def test_returns_none_when_raw_earnings_date_is_none(self):
        # Line 35: raw is None inside the try block → return None
        mock = MagicMock()
        # Dict calendar with no Earnings Date key → raw = None
        mock.calendar = {}
        with patch("risk.earnings_calendar.yf.Ticker", return_value=mock):
            result = get_next_earnings_date("AAPL")
        self.assertIsNone(result)


class TestDaysUntilEarnings(unittest.TestCase):

    def test_future_earnings_returns_positive(self):
        future = date(2026, 4, 28)
        with (
            patch("risk.earnings_calendar.yf.Ticker", return_value=_mock_ticker(future)),
            patch("risk.earnings_calendar.today_et", return_value=_TODAY),
        ):
            result = days_until_earnings("AAPL")
        self.assertEqual(result, 2)

    def test_past_earnings_returns_none(self):
        past = date(2026, 4, 20)
        with (
            patch("risk.earnings_calendar.yf.Ticker", return_value=_mock_ticker(past)),
            patch("risk.earnings_calendar.today_et", return_value=_TODAY),
        ):
            result = days_until_earnings("AAPL")
        self.assertIsNone(result)

    def test_today_earnings_returns_zero(self):
        with (
            patch("risk.earnings_calendar.yf.Ticker", return_value=_mock_ticker(_TODAY)),
            patch("risk.earnings_calendar.today_et", return_value=_TODAY),
        ):
            result = days_until_earnings("AAPL")
        self.assertEqual(result, 0)

    def test_no_data_returns_none(self):
        with patch("risk.earnings_calendar.yf.Ticker", return_value=_mock_ticker(None)):
            result = days_until_earnings("AAPL")
        self.assertIsNone(result)


class TestGetEarningsRiskPositions(unittest.TestCase):

    def test_position_within_warning_days_flagged(self):
        tomorrow = date(2026, 4, 27)
        with (
            patch("risk.earnings_calendar.yf.Ticker", return_value=_mock_ticker(tomorrow)),
            patch("risk.earnings_calendar.today_et", return_value=_TODAY),
        ):
            at_risk = get_earnings_risk_positions(["AAPL"], warning_days=2)
        self.assertIn("AAPL", at_risk)

    def test_position_outside_warning_days_not_flagged(self):
        far_future = date(2026, 5, 30)
        with (
            patch("risk.earnings_calendar.yf.Ticker", return_value=_mock_ticker(far_future)),
            patch("risk.earnings_calendar.today_et", return_value=_TODAY),
        ):
            at_risk = get_earnings_risk_positions(["AAPL"], warning_days=2)
        self.assertNotIn("AAPL", at_risk)

    def test_no_earnings_data_not_flagged(self):
        with (
            patch("risk.earnings_calendar.yf.Ticker", return_value=_mock_ticker(None)),
            patch("risk.earnings_calendar.today_et", return_value=_TODAY),
        ):
            at_risk = get_earnings_risk_positions(["AAPL"], warning_days=2)
        self.assertNotIn("AAPL", at_risk)

    def test_empty_symbols_returns_empty(self):
        at_risk = get_earnings_risk_positions([], warning_days=2)
        self.assertEqual(at_risk, {})


class TestGetNextEarningsDateFormats(unittest.TestCase):
    """Covers alternative yfinance calendar return formats."""

    def _ticker_with_calendar(self, cal):
        mock = MagicMock()
        mock.calendar = cal
        return mock

    def test_dataframe_with_earnings_date_column(self):
        future = date(2026, 5, 20)
        df = pd.DataFrame({"Earnings Date": [pd.Timestamp(future.isoformat())]})
        with patch("risk.earnings_calendar.yf.Ticker",
                   return_value=self._ticker_with_calendar(df)):
            result = get_next_earnings_date("AAPL")
        self.assertEqual(result, future)

    def test_dataframe_without_earnings_date_column_uses_first_cell(self):
        future = date(2026, 5, 20)
        df = pd.DataFrame([[pd.Timestamp(future.isoformat())]], columns=["Date"])
        with patch("risk.earnings_calendar.yf.Ticker",
                   return_value=self._ticker_with_calendar(df)):
            result = get_next_earnings_date("AAPL")
        self.assertEqual(result, future)

    def test_empty_dataframe_returns_none(self):
        df = pd.DataFrame()
        with patch("risk.earnings_calendar.yf.Ticker",
                   return_value=self._ticker_with_calendar(df)):
            result = get_next_earnings_date("AAPL")
        self.assertIsNone(result)

    def test_non_dict_non_dataframe_returns_none(self):
        with patch("risk.earnings_calendar.yf.Ticker",
                   return_value=self._ticker_with_calendar("unexpected string")):
            result = get_next_earnings_date("AAPL")
        self.assertIsNone(result)
