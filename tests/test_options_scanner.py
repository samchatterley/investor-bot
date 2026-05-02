"""Tests for data/options_scanner.py — put/call ratio and unusual call detection."""
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd


def _make_chain(call_vol, put_vol, call_oi=1000):
    calls = pd.DataFrame({"volume": [call_vol], "openInterest": [call_oi]})
    puts = pd.DataFrame({"volume": [put_vol], "openInterest": [500]})
    chain = MagicMock()
    chain.calls = calls
    chain.puts = puts
    return chain


def _mock_ticker(chain=None, expirations=("2026-05-01",)):
    ticker = MagicMock()
    ticker.options = expirations
    ticker.option_chain.return_value = chain
    return ticker


class TestGetSignal(unittest.TestCase):

    def test_bullish_ratio_below_0_7(self):
        from data.options_scanner import _get_signal
        chain = _make_chain(call_vol=1000, put_vol=500)
        with patch("data.options_scanner.yf.Ticker", return_value=_mock_ticker(chain)):
            result = _get_signal("AAPL")
        self.assertIsNotNone(result)
        # put_vol / (call_vol + 1) ≈ 0.50
        self.assertLess(result["put_call_ratio"], 0.7)

    def test_bearish_ratio_above_1_3(self):
        from data.options_scanner import _get_signal
        chain = _make_chain(call_vol=300, put_vol=1000)
        with patch("data.options_scanner.yf.Ticker", return_value=_mock_ticker(chain)):
            result = _get_signal("AAPL")
        self.assertIsNotNone(result)
        self.assertGreater(result["put_call_ratio"], 1.3)

    def test_unusual_calls_detected_when_volume_exceeds_oi(self):
        from data.options_scanner import _get_signal
        # call_vol > call_oi * 1.5 → unusual
        chain = _make_chain(call_vol=2000, put_vol=500, call_oi=1000)
        with patch("data.options_scanner.yf.Ticker", return_value=_mock_ticker(chain)):
            result = _get_signal("AAPL")
        self.assertTrue(result["unusual_calls"])

    def test_no_unusual_calls_when_volume_normal(self):
        from data.options_scanner import _get_signal
        chain = _make_chain(call_vol=800, put_vol=500, call_oi=1000)
        with patch("data.options_scanner.yf.Ticker", return_value=_mock_ticker(chain)):
            result = _get_signal("AAPL")
        self.assertFalse(result["unusual_calls"])

    def test_returns_none_when_no_expirations(self):
        from data.options_scanner import _get_signal
        ticker = _mock_ticker(expirations=())
        with patch("data.options_scanner.yf.Ticker", return_value=ticker):
            result = _get_signal("AAPL")
        self.assertIsNone(result)

    def test_returns_none_when_total_volume_below_minimum(self):
        from data.options_scanner import _get_signal
        chain = _make_chain(call_vol=10, put_vol=5)   # total 15 < _MIN_VOLUME (50)
        with patch("data.options_scanner.yf.Ticker", return_value=_mock_ticker(chain)):
            result = _get_signal("AAPL")
        self.assertIsNone(result)

    def test_returns_none_on_empty_chain(self):
        from data.options_scanner import _get_signal
        chain = MagicMock()
        chain.calls = pd.DataFrame()
        chain.puts = pd.DataFrame()
        with patch("data.options_scanner.yf.Ticker", return_value=_mock_ticker(chain)):
            result = _get_signal("AAPL")
        self.assertIsNone(result)

    def test_returns_none_on_exception(self):
        from data.options_scanner import _get_signal
        with patch("data.options_scanner.yf.Ticker", side_effect=Exception("network")):
            result = _get_signal("AAPL")
        self.assertIsNone(result)

    def test_result_has_required_keys(self):
        from data.options_scanner import _get_signal
        chain = _make_chain(call_vol=500, put_vol=300)
        with patch("data.options_scanner.yf.Ticker", return_value=_mock_ticker(chain)):
            result = _get_signal("AAPL")
        self.assertIn("put_call_ratio", result)
        self.assertIn("unusual_calls", result)

    def test_put_call_ratio_rounded_to_2dp(self):
        from data.options_scanner import _get_signal
        chain = _make_chain(call_vol=333, put_vol=200)
        with patch("data.options_scanner.yf.Ticker", return_value=_mock_ticker(chain)):
            result = _get_signal("AAPL")
        # Should be rounded to 2 decimal places
        self.assertEqual(result["put_call_ratio"], round(result["put_call_ratio"], 2))


class TestGetOptionsSignals(unittest.TestCase):

    def test_returns_dict_of_results(self):
        from data.options_scanner import get_options_signals
        with patch("data.options_scanner._get_signal", return_value={"put_call_ratio": 0.5, "unusual_calls": False}):
            result = get_options_signals(["AAPL", "NVDA"])
        self.assertIsInstance(result, dict)

    def test_none_results_excluded(self):
        from data.options_scanner import get_options_signals
        with patch("data.options_scanner._get_signal", return_value=None):
            result = get_options_signals(["AAPL", "NVDA"])
        self.assertEqual(result, {})

    def test_empty_symbols_returns_empty(self):
        from data.options_scanner import get_options_signals
        result = get_options_signals([])
        self.assertEqual(result, {})

    def test_future_exception_does_not_block_other_symbols(self):
        # Lines 62-63: future.result() raises → logs debug, continues
        from data.options_scanner import get_options_signals

        def selective_signal(sym):
            if sym == "FAIL":
                raise RuntimeError("forced failure")
            return {"put_call_ratio": 0.5, "unusual_calls": False}

        with patch("data.options_scanner._get_signal", side_effect=selective_signal):
            result = get_options_signals(["FAIL", "AAPL"])
        self.assertNotIn("FAIL", result)
        self.assertIn("AAPL", result)
