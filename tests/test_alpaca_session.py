"""Tests for utils/alpaca_session.py — bounded request timeout injection for alpaca-py clients."""

import unittest

from utils.alpaca_session import _DEFAULT_TIMEOUT, with_request_timeout


class _FakeSession:
    def __init__(self):
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return "resp"


class _FakeClient:
    def __init__(self, session):
        self._session = session


class TestWithRequestTimeout(unittest.TestCase):
    def test_injects_default_timeout(self):
        sess = _FakeSession()
        client = with_request_timeout(_FakeClient(sess))
        result = client._session.request("GET", "https://example.com")
        self.assertEqual(result, "resp")
        self.assertEqual(sess.calls[0][2]["timeout"], _DEFAULT_TIMEOUT)

    def test_custom_timeout_is_used(self):
        sess = _FakeSession()
        with_request_timeout(_FakeClient(sess), timeout=(1.0, 2.0))
        sess.request("POST", "https://example.com")
        self.assertEqual(sess.calls[0][2]["timeout"], (1.0, 2.0))

    def test_does_not_override_explicit_per_call_timeout(self):
        sess = _FakeSession()
        with_request_timeout(_FakeClient(sess))
        sess.request("GET", "https://example.com", timeout=5)
        self.assertEqual(sess.calls[0][2]["timeout"], 5)

    def test_noop_when_client_has_no_session_attr(self):
        class _NoSession:
            pass

        c = _NoSession()
        self.assertIs(with_request_timeout(c), c)

    def test_noop_when_session_is_none(self):
        c = _FakeClient(None)
        self.assertIs(with_request_timeout(c), c)

    def test_noop_when_session_has_no_request(self):
        class _SessionNoRequest:
            pass

        c = _FakeClient(_SessionNoRequest())
        self.assertIs(with_request_timeout(c), c)

    def test_real_alpaca_client_session_gets_patched(self):
        # Guard: the patch must actually take effect on the pinned alpaca-py — if a future version
        # drops the private `_session`, with_request_timeout silently no-ops, and this fails loudly.
        from alpaca.data.historical import StockHistoricalDataClient

        client = with_request_timeout(StockHistoricalDataClient(api_key="test", secret_key="test"))
        self.assertEqual(client._session.request.__name__, "_request_with_timeout")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
