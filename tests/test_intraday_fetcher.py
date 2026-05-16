"""Tests for data/intraday_fetcher.py — lines 53-133."""

import builtins
import os
import pickle
import sys
import tempfile
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

from data.intraday_fetcher import _cache_path, fetch_intraday_bars

_ET = ZoneInfo("America/New_York")
_START = "2025-01-02"
_END = "2025-01-02"
_MOCK_DATE = "2025-01-02"


def _make_mock_bar():
    bar = MagicMock()
    mock_dt = datetime(2025, 1, 2, 10, 30, tzinfo=_ET)
    bar.timestamp.astimezone.return_value = mock_dt
    return bar


def _alpaca_modules(bars=None, raise_exc=None):
    """Return (sys.modules patch dict, mock_client)."""
    client = MagicMock()
    if raise_exc is not None:
        client.get_stock_bars.side_effect = raise_exc
    else:
        bars_data = [] if bars == [] else (bars if bars is not None else [_make_mock_bar()])
        resp = MagicMock()
        resp.data.get.return_value = bars_data
        client.get_stock_bars.return_value = resp

    historical = MagicMock()
    historical.StockHistoricalDataClient = MagicMock(return_value=client)
    modules = {
        "alpaca": MagicMock(),
        "alpaca.data": MagicMock(),
        "alpaca.data.historical": historical,
        "alpaca.data.requests": MagicMock(),
        "alpaca.data.timeframe": MagicMock(),
    }
    return modules, client


class _KeysMixin:
    """Set/restore Alpaca API keys around a test callable."""

    def _with_keys(self, fn):
        import config as _cfg

        orig_key, orig_secret = _cfg.ALPACA_API_KEY, _cfg.ALPACA_SECRET_KEY
        _cfg.ALPACA_API_KEY = "test_key"
        _cfg.ALPACA_SECRET_KEY = "test_secret"
        try:
            fn()
        finally:
            _cfg.ALPACA_API_KEY = orig_key
            _cfg.ALPACA_SECRET_KEY = orig_secret


class TestFetchIntradayBarsImportError(unittest.TestCase):
    """Lines 60-68: alpaca import fails → empty dict."""

    def test_import_error_returns_empty(self):
        real_import = builtins.__import__

        def blocking(name, *a, **kw):
            if "alpaca" in name:
                raise ImportError("not installed")
            return real_import(name, *a, **kw)  # pragma: no cover

        blocked = {k: sys.modules.pop(k) for k in list(sys.modules) if "alpaca" in k}
        try:
            with patch("builtins.__import__", side_effect=blocking):
                result = fetch_intraday_bars(["AAPL"], _START, _END, cache_dir="")
            self.assertEqual(result, {})
        finally:
            sys.modules.update(blocked)


class TestFetchIntradayBarsNoApiKeys(unittest.TestCase):
    """Lines 70-72: blank/None API keys → empty dict."""

    def test_no_keys_returns_empty(self):
        import config as _cfg

        orig_key, orig_secret = _cfg.ALPACA_API_KEY, _cfg.ALPACA_SECRET_KEY
        _cfg.ALPACA_API_KEY = ""
        _cfg.ALPACA_SECRET_KEY = ""
        mods, _ = _alpaca_modules()
        try:
            with patch.dict(sys.modules, mods):
                result = fetch_intraday_bars(["AAPL"], _START, _END, cache_dir="")
            self.assertEqual(result, {})
        finally:
            _cfg.ALPACA_API_KEY = orig_key
            _cfg.ALPACA_SECRET_KEY = orig_secret


class TestFetchIntradayBarsNoCaching(_KeysMixin, unittest.TestCase):
    """Lines 74-133: fetch with cache_dir='' (no caching)."""

    def test_success_returns_bars_by_date(self):
        """Lines 74-122, 132-133: bars returned → result keyed by symbol and date."""
        mods, _ = _alpaca_modules()

        def run():
            with patch.dict(sys.modules, mods):
                result = fetch_intraday_bars(["AAPL"], _START, _END, cache_dir="")
            self.assertIn("AAPL", result)
            self.assertIn(_MOCK_DATE, result["AAPL"])

        self._with_keys(run)

    def test_empty_bars_skips_symbol(self):
        """Lines 110-111: empty bars_data → continue → symbol absent."""
        mods, _ = _alpaca_modules(bars=[])

        def run():
            with patch.dict(sys.modules, mods):
                result = fetch_intraday_bars(["AAPL"], _START, _END, cache_dir="")
            self.assertNotIn("AAPL", result)

        self._with_keys(run)

    def test_exception_in_fetch_skips_symbol(self):
        """Lines 129-130: get_stock_bars raises → warning logged, symbol absent."""
        mods, _ = _alpaca_modules(raise_exc=RuntimeError("network"))

        def run():
            with patch.dict(sys.modules, mods):
                result = fetch_intraday_bars(["AAPL"], _START, _END, cache_dir="")
            self.assertNotIn("AAPL", result)

        self._with_keys(run)


class TestFetchIntradayBarsCaching(_KeysMixin, unittest.TestCase):
    """Lines 53-58, 84-94, 124-127: caching paths."""

    def test_default_cache_dir_used_when_none(self):
        """Lines 54-55: cache_dir=None → _DEFAULT_CACHE_DIR used, makedirs called."""
        mods, _ = _alpaca_modules(bars=[])

        with tempfile.TemporaryDirectory() as tmpdir:

            def run():
                with (
                    patch.dict(sys.modules, mods),
                    patch("data.intraday_fetcher._DEFAULT_CACHE_DIR", tmpdir),
                ):
                    result = fetch_intraday_bars(["AAPL"], _START, _END, cache_dir=None)
                self.assertEqual(result, {})
                self.assertTrue(os.path.isdir(tmpdir))

            self._with_keys(run)

    def test_cache_hit_returns_pickled_data(self):
        """Lines 84-92: pre-existing cache file → loaded, fetch not called."""
        mods, mock_client = _alpaca_modules()

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = _cache_path(tmpdir, "AAPL", _START, _END)
            cached = {_MOCK_DATE: [("dt", "fake_bar")]}
            with open(cache_file, "wb") as f:
                pickle.dump(cached, f)

            def run():
                with patch.dict(sys.modules, mods):
                    result = fetch_intraday_bars(["AAPL"], _START, _END, cache_dir=tmpdir)
                self.assertEqual(result["AAPL"], cached)
                mock_client.get_stock_bars.assert_not_called()

            self._with_keys(run)

    def test_broken_cache_falls_through_to_fetch(self):
        """Lines 93-94: corrupt pickle → exception caught, falls through to real fetch."""
        mods, mock_client = _alpaca_modules()

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = _cache_path(tmpdir, "AAPL", _START, _END)
            with open(cache_file, "wb") as f:
                f.write(b"not_valid_pickle")

            def run():
                with patch.dict(sys.modules, mods):
                    result = fetch_intraday_bars(["AAPL"], _START, _END, cache_dir=tmpdir)
                mock_client.get_stock_bars.assert_called_once()
                self.assertIn("AAPL", result)

            self._with_keys(run)

    def test_result_written_to_cache(self):
        """Lines 124-127: after successful fetch, cache file is written."""
        mods, _ = _alpaca_modules()

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = _cache_path(tmpdir, "AAPL", _START, _END)
            self.assertFalse(os.path.exists(cache_file))

            def run():
                with patch.dict(sys.modules, mods):
                    result = fetch_intraday_bars(["AAPL"], _START, _END, cache_dir=tmpdir)
                self.assertTrue(os.path.exists(cache_file))
                self.assertIn("AAPL", result)

            self._with_keys(run)
