"""Tests for execution/quote_gate.py — live quote validation."""

import unittest
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch


def _mock_quote(bid=149.9, ask=150.1, timestamp=None, tzinfo_none=False):
    """Build a mock quote object."""
    q = MagicMock()
    q.bid_price = bid
    q.ask_price = ask
    if timestamp is None:
        ts = datetime.now(UTC) - timedelta(seconds=1)
    else:
        ts = timestamp
    if tzinfo_none:
        ts = ts.replace(tzinfo=None)
    q.timestamp = ts
    return q


def _mock_trade(seconds_old=1):
    t = MagicMock()
    t.timestamp = datetime.now(UTC) - timedelta(seconds=seconds_old)
    return t


def _make_client(quote=None, trade=None, quote_raises=None, trade_raises=None):
    """Build a mock data client."""
    client = MagicMock()
    if quote_raises:
        client.get_stock_latest_quote.side_effect = quote_raises
    else:
        client.get_stock_latest_quote.return_value = {"AAPL": quote}
    if trade_raises:
        client.get_stock_latest_trade.side_effect = trade_raises
    else:
        client.get_stock_latest_trade.return_value = {"AAPL": trade}
    return client


class TestGetDataClient(unittest.TestCase):
    """Line 52: _get_data_client() constructs StockHistoricalDataClient."""

    def test_returns_stock_historical_data_client(self):
        from execution.quote_gate import _get_data_client

        with patch("execution.quote_gate.StockHistoricalDataClient") as mock_cls:
            mock_cls.return_value = MagicMock()
            result = _get_data_client()
        mock_cls.assert_called_once()
        self.assertIsNotNone(result)


class TestCheckQuoteGateBrokerError(unittest.TestCase):
    """Line 77: network error → BrokerStateUnavailable raised."""

    def test_quote_fetch_error_raises_broker_unavailable(self):
        from execution.quote_gate import check_quote_gate
        from models import BrokerStateUnavailable

        client = _make_client(quote_raises=ConnectionError("timeout"))
        with self.assertRaises(BrokerStateUnavailable):
            check_quote_gate("AAPL", 500.0, data_client=client)


class TestCheckQuoteGateNoQuote(unittest.TestCase):
    """Line 81: quotes.get(symbol) returns None → rejected."""

    def test_no_quote_returned_is_rejected(self):
        from execution.quote_gate import check_quote_gate

        client = MagicMock()
        client.get_stock_latest_quote.return_value = {}  # No AAPL key
        result = check_quote_gate("AAPL", 500.0, data_client=client)
        self.assertFalse(result.approved)
        self.assertIn("no quote returned", result.reject_reason)


class TestCheckQuoteGateMissingBidAsk(unittest.TestCase):
    """Lines 87-93: bid or ask is zero → rejected."""

    def test_zero_bid_is_rejected(self):
        from execution.quote_gate import check_quote_gate

        q = _mock_quote(bid=0.0, ask=150.1)
        trade = _mock_trade(seconds_old=1)
        client = _make_client(quote=q, trade=trade)
        result = check_quote_gate("AAPL", 500.0, data_client=client)
        self.assertFalse(result.approved)
        self.assertIn("bid/ask", result.reject_reason)

    def test_zero_ask_is_rejected(self):
        from execution.quote_gate import check_quote_gate

        q = _mock_quote(bid=149.9, ask=0.0)
        trade = _mock_trade(seconds_old=1)
        client = _make_client(quote=q, trade=trade)
        result = check_quote_gate("AAPL", 500.0, data_client=client)
        self.assertFalse(result.approved)


class TestCheckQuoteGateTimezone(unittest.TestCase):
    """Line 99: quote_ts with no tzinfo → replace with UTC."""

    def test_naive_timestamp_treated_as_utc(self):
        from execution.quote_gate import check_quote_gate

        q = _mock_quote(tzinfo_none=True)
        trade = _mock_trade(seconds_old=1)
        client = _make_client(quote=q, trade=trade)
        result = check_quote_gate("AAPL", 500.0, data_client=client)
        # Should not raise — the naive timestamp gets UTC attached
        self.assertIsNotNone(result)


class TestCheckQuoteGateNoTimestamp(unittest.TestCase):
    """Line 102: quote_ts is None → quote_age = 0.0 (assume fresh)."""

    def test_none_timestamp_assumed_fresh(self):
        from execution.quote_gate import check_quote_gate

        q = _mock_quote()
        q.timestamp = None  # Override to None
        trade = _mock_trade(seconds_old=1)
        client = _make_client(quote=q, trade=trade)
        result = check_quote_gate("AAPL", 500.0, data_client=client)
        self.assertEqual(result.quote_age_seconds, 0.0)


class TestCheckQuoteGateStaleQuote(unittest.TestCase):
    """Lines 104-112: quote_age > max_quote_age → rejected."""

    def test_stale_quote_rejected(self):
        from execution.quote_gate import check_quote_gate

        stale_ts = datetime.now(UTC) - timedelta(seconds=60)
        q = _mock_quote(timestamp=stale_ts)
        trade = _mock_trade(seconds_old=1)
        client = _make_client(quote=q, trade=trade)
        result = check_quote_gate("AAPL", 500.0, data_client=client, max_quote_age=10)
        self.assertFalse(result.approved)
        self.assertIn("stale", result.reject_reason)


class TestCheckQuoteGateWideSpread(unittest.TestCase):
    """Lines 117-126: spread_bps > max_spread_bps → rejected."""

    def test_wide_spread_rejected(self):
        from execution.quote_gate import check_quote_gate

        # bid=100, ask=101 → spread = 1/100.5 * 10000 ≈ 99.5 bps > 30
        q = _mock_quote(bid=100.0, ask=101.0)
        trade = _mock_trade(seconds_old=1)
        client = _make_client(quote=q, trade=trade)
        result = check_quote_gate("AAPL", 500.0, data_client=client)
        self.assertFalse(result.approved)
        self.assertIn("spread", result.reject_reason)


class TestCheckQuoteGateStaleLastTrade(unittest.TestCase):
    """Lines 138-148: last trade stale → rejected."""

    def test_stale_last_trade_rejected(self):
        from execution.quote_gate import check_quote_gate

        q = _mock_quote()
        stale_trade = _mock_trade(seconds_old=120)
        client = _make_client(quote=q, trade=stale_trade)
        result = check_quote_gate("AAPL", 500.0, data_client=client, max_trade_age=30)
        self.assertFalse(result.approved)
        self.assertIn("last trade stale", result.reject_reason)

    def test_trade_fetch_error_raises_broker_unavailable(self):
        """Lines 149-152: trade fetch fails → BrokerStateUnavailable."""
        from execution.quote_gate import check_quote_gate
        from models import BrokerStateUnavailable

        q = _mock_quote()
        client = _make_client(quote=q, trade_raises=ConnectionError("timeout"))
        with self.assertRaises(BrokerStateUnavailable):
            check_quote_gate("AAPL", 500.0, data_client=client)

    def test_naive_trade_timestamp_treated_as_utc(self):
        """Line 135-136: trade timestamp without tzinfo gets UTC attached."""
        from execution.quote_gate import check_quote_gate

        q = _mock_quote()
        trade = _mock_trade(seconds_old=1)
        trade.timestamp = trade.timestamp.replace(tzinfo=None)
        client = _make_client(quote=q, trade=trade)
        result = check_quote_gate("AAPL", 500.0, data_client=client)
        self.assertIsNotNone(result)


class TestCheckQuoteGateAffordability(unittest.TestCase):
    """Lines 155-168: notional too small for even 1 share → rejected."""

    def test_sub_share_notional_rejected(self):
        from execution.quote_gate import check_quote_gate

        # ask=200, notional=50 → 50/200 = 0.25 < 1 share
        q = _mock_quote(bid=199.9, ask=200.0)
        trade = _mock_trade(seconds_old=1)
        client = _make_client(quote=q, trade=trade)
        result = check_quote_gate("AAPL", 50.0, data_client=client)
        self.assertFalse(result.approved)
        self.assertIn("cannot buy 1 whole share", result.reject_reason)


class TestCheckQuoteGateApproved(unittest.TestCase):
    """Lines 170-178: all checks pass → approved=True."""

    def test_valid_quote_approved(self):
        from execution.quote_gate import check_quote_gate

        q = _mock_quote(bid=149.9, ask=150.1)
        trade = _mock_trade(seconds_old=1)
        client = _make_client(quote=q, trade=trade)
        result = check_quote_gate("AAPL", 500.0, data_client=client)
        self.assertTrue(result.approved)
        self.assertEqual(result.symbol, "AAPL")
        self.assertAlmostEqual(result.bid, 149.9)
        self.assertAlmostEqual(result.ask, 150.1)


if __name__ == "__main__":
    unittest.main()
