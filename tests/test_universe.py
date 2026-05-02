"""Tests for execution/universe.py — build_scan_universe and helpers."""
import contextlib
import json
import os
import sys
import unittest
from datetime import datetime, timedelta  # noqa: F401 (timedelta used in tests)
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Stub out heavy / credential-requiring imports before the module is loaded
# ---------------------------------------------------------------------------

def _make_stubs():
    config_stub = MagicMock()
    config_stub.ALPACA_API_KEY = "test-key"
    config_stub.ALPACA_SECRET_KEY = "test-secret"
    config_stub.LOG_DIR = "/tmp/test_universe_logs"
    config_stub.MIN_VOLUME = 500_000
    config_stub.STOCK_UNIVERSE = ["AAPL", "MSFT", "GOOGL"]
    return {
        "config": config_stub,
        "alpaca.data.historical": MagicMock(),
        "alpaca.data.requests": MagicMock(),
        "alpaca.trading.client": MagicMock(),
        "alpaca.trading.enums": MagicMock(),
        "alpaca.trading.requests": MagicMock(),
    }


def _load_universe_module():
    import importlib.util
    import types

    stubs = _make_stubs()
    with patch.dict(sys.modules, stubs):
        sys.modules.pop("execution.universe", None)

        spec = importlib.util.spec_from_file_location(
            "execution.universe",
            os.path.join(os.path.dirname(__file__), "..", "execution", "universe.py"),
        )
        mod = types.ModuleType(spec.name)
        mod.__spec__ = spec
        mod.__file__ = spec.origin
        mod.__package__ = "execution"
        spec.loader.exec_module(mod)
        # Patch module-level constants to use test values
        mod._CACHE_PATH = "/tmp/test_universe_cache.json"
        mod.LOG_DIR = "/tmp/test_universe_logs"
        mod.MIN_VOLUME = 500_000
        mod.STOCK_UNIVERSE = ["AAPL", "MSFT", "GOOGL"]
        mod.ALPACA_API_KEY = "test-key"
        mod.ALPACA_SECRET_KEY = "test-secret"
    return mod


def _make_asset(symbol: str, tradable=True, fractionable=True, exchange="NYSE"):
    a = MagicMock()
    a.symbol = symbol
    a.tradable = tradable
    a.fractionable = fractionable
    a.exchange = exchange
    return a


def _make_snap(close: float, volume: float):
    bar = MagicMock()
    bar.close = close
    bar.volume = volume
    snap = MagicMock()
    snap.daily_bar = bar
    return snap


class TestLoadCache(unittest.TestCase):

    def setUp(self):
        self.mod = _load_universe_module()
        with contextlib.suppress(FileNotFoundError):
            os.remove(self.mod._CACHE_PATH)

    def tearDown(self):
        with contextlib.suppress(FileNotFoundError):
            os.remove(self.mod._CACHE_PATH)

    def test_missing_cache_returns_none(self):
        self.assertIsNone(self.mod._load_cache())

    def test_fresh_cache_returns_symbols(self):
        symbols = ["AAPL", "MSFT"]
        data = {"saved_at": datetime.now().isoformat(), "symbols": symbols}
        with open(self.mod._CACHE_PATH, "w") as f:
            json.dump(data, f)
        result = self.mod._load_cache()
        self.assertEqual(result, symbols)

    def test_stale_cache_returns_none(self):
        stale_time = datetime.now() - timedelta(hours=25)
        data = {"saved_at": stale_time.isoformat(), "symbols": ["AAPL"]}
        with open(self.mod._CACHE_PATH, "w") as f:
            json.dump(data, f)
        self.assertIsNone(self.mod._load_cache())

    def test_corrupt_cache_returns_none(self):
        with open(self.mod._CACHE_PATH, "w") as f:
            f.write("not valid json{{{")
        self.assertIsNone(self.mod._load_cache())


class TestSaveCache(unittest.TestCase):

    def setUp(self):
        self.mod = _load_universe_module()
        with contextlib.suppress(FileNotFoundError):
            os.remove(self.mod._CACHE_PATH)

    def tearDown(self):
        with contextlib.suppress(FileNotFoundError):
            os.remove(self.mod._CACHE_PATH)

    def test_saves_symbols_to_json(self):
        symbols = ["AAPL", "TSLA", "NVDA"]
        with patch("os.makedirs"):
            self.mod._save_cache(symbols)
        with open(self.mod._CACHE_PATH) as f:
            data = json.load(f)
        self.assertEqual(data["symbols"], symbols)
        self.assertIn("saved_at", data)

    def test_handles_io_error_gracefully(self):
        with (
            patch("os.makedirs"),
            patch("builtins.open", side_effect=OSError("disk full")),
        ):
            try:
                self.mod._save_cache(["AAPL"])
            except Exception as exc:
                self.fail(f"_save_cache raised unexpectedly: {exc}")


class TestGetEligibleSymbols(unittest.TestCase):

    def setUp(self):
        self.mod = _load_universe_module()

    def test_filters_non_tradable(self):
        assets = [
            _make_asset("AAPL", tradable=True, fractionable=True, exchange="NYSE"),
            _make_asset("SHADY", tradable=False, fractionable=True, exchange="NYSE"),
        ]
        client = MagicMock()
        client.get_all_assets.return_value = assets
        # Patch the enum set so exchange string comparison works
        self.mod._MAJOR_EXCHANGES = {"NYSE"}
        result = self.mod._get_eligible_symbols(client)
        self.assertIn("AAPL", result)
        self.assertNotIn("SHADY", result)

    def test_filters_non_fractionable(self):
        assets = [
            _make_asset("AAPL", tradable=True, fractionable=True, exchange="NYSE"),
            _make_asset("PRICEY", tradable=True, fractionable=False, exchange="NYSE"),
        ]
        client = MagicMock()
        client.get_all_assets.return_value = assets
        self.mod._MAJOR_EXCHANGES = {"NYSE"}
        result = self.mod._get_eligible_symbols(client)
        self.assertIn("AAPL", result)
        self.assertNotIn("PRICEY", result)

    def test_filters_otc_exchange(self):
        assets = [
            _make_asset("AAPL", tradable=True, fractionable=True, exchange="NASDAQ"),
            _make_asset("OTCCO", tradable=True, fractionable=True, exchange="OTC"),
        ]
        client = MagicMock()
        client.get_all_assets.return_value = assets
        self.mod._MAJOR_EXCHANGES = {"NYSE", "NASDAQ"}
        result = self.mod._get_eligible_symbols(client)
        self.assertIn("AAPL", result)
        self.assertNotIn("OTCCO", result)

    def test_returns_all_eligible(self):
        assets = [
            _make_asset("AAPL", tradable=True, fractionable=True, exchange="NASDAQ"),
            _make_asset("JPM", tradable=True, fractionable=True, exchange="NYSE"),
        ]
        client = MagicMock()
        client.get_all_assets.return_value = assets
        self.mod._MAJOR_EXCHANGES = {"NYSE", "NASDAQ"}
        result = self.mod._get_eligible_symbols(client)
        self.assertEqual(sorted(result), ["AAPL", "JPM"])


class TestApplySnapshotFilter(unittest.TestCase):

    def setUp(self):
        self.mod = _load_universe_module()
        self.mod._MIN_PRICE = 5.0
        self.mod.MIN_VOLUME = 500_000
        self.mod._SNAPSHOT_CHUNK_SIZE = 3

    def _mock_data_client(self, snap_map: dict):
        mock_dc = MagicMock()
        mock_dc.get_stock_snapshots.return_value = snap_map
        return mock_dc

    def test_passes_price_and_volume_threshold(self):
        snaps = {
            "AAPL": _make_snap(close=150.0, volume=1_000_000),
        }
        mock_cls = MagicMock(return_value=self._mock_data_client(snaps))
        with patch.object(self.mod, "StockHistoricalDataClient", mock_cls):
            result = self.mod._apply_snapshot_filter(["AAPL"])
        self.assertIn("AAPL", result)

    def test_filters_below_min_price(self):
        snaps = {
            "PENNY": _make_snap(close=1.0, volume=5_000_000),
        }
        mock_cls = MagicMock(return_value=self._mock_data_client(snaps))
        with patch.object(self.mod, "StockHistoricalDataClient", mock_cls):
            result = self.mod._apply_snapshot_filter(["PENNY"])
        self.assertNotIn("PENNY", result)

    def test_filters_below_min_volume(self):
        snaps = {
            "ILLIQ": _make_snap(close=50.0, volume=100),
        }
        mock_cls = MagicMock(return_value=self._mock_data_client(snaps))
        with patch.object(self.mod, "StockHistoricalDataClient", mock_cls):
            result = self.mod._apply_snapshot_filter(["ILLIQ"])
        self.assertNotIn("ILLIQ", result)

    def test_skips_symbol_with_none_daily_bar(self):
        snap = MagicMock()
        snap.daily_bar = None
        snaps = {"NOBAR": snap}
        mock_cls = MagicMock(return_value=self._mock_data_client(snaps))
        with patch.object(self.mod, "StockHistoricalDataClient", mock_cls):
            result = self.mod._apply_snapshot_filter(["NOBAR"])
        self.assertNotIn("NOBAR", result)

    def test_chunk_error_excludes_chunk_symbols(self):
        mock_dc = MagicMock()
        mock_dc.get_stock_snapshots.side_effect = RuntimeError("API error")
        mock_cls = MagicMock(return_value=mock_dc)
        with patch.object(self.mod, "StockHistoricalDataClient", mock_cls):
            result = self.mod._apply_snapshot_filter(["AAPL", "MSFT"])
        # Fail-closed: on error the chunk is dropped, not passed through
        self.assertNotIn("AAPL", result)
        self.assertNotIn("MSFT", result)
        self.assertEqual(result, [])

    def test_multiple_chunks_processed(self):
        """With chunk size 3 and 5 symbols, 2 API calls are made."""
        snaps = {
            "A": _make_snap(10.0, 1_000_000),
            "B": _make_snap(10.0, 1_000_000),
            "C": _make_snap(10.0, 1_000_000),
        }
        snaps2 = {
            "D": _make_snap(10.0, 1_000_000),
            "E": _make_snap(10.0, 1_000_000),
        }
        mock_dc = MagicMock()
        mock_dc.get_stock_snapshots.side_effect = [snaps, snaps2]
        mock_cls = MagicMock(return_value=mock_dc)
        with patch.object(self.mod, "StockHistoricalDataClient", mock_cls):
            result = self.mod._apply_snapshot_filter(["A", "B", "C", "D", "E"])
        self.assertEqual(mock_dc.get_stock_snapshots.call_count, 2)
        self.assertEqual(sorted(result), ["A", "B", "C", "D", "E"])


class TestBuildScanUniverse(unittest.TestCase):

    def setUp(self):
        self.mod = _load_universe_module()
        self.mod._MAX_UNIVERSE_SIZE = 10
        self.mod.STOCK_UNIVERSE = ["AAPL", "MSFT", "GOOGL"]
        with contextlib.suppress(FileNotFoundError):
            os.remove(self.mod._CACHE_PATH)

    def tearDown(self):
        with contextlib.suppress(FileNotFoundError):
            os.remove(self.mod._CACHE_PATH)

    def test_cache_hit_skips_api_calls(self):
        symbols = ["AAPL", "MSFT", "AMZN"]
        data = {"saved_at": datetime.now().isoformat(), "symbols": symbols}
        with open(self.mod._CACHE_PATH, "w") as f:
            json.dump(data, f)
        client = MagicMock()
        result = self.mod.build_scan_universe(client)
        client.get_all_assets.assert_not_called()
        self.assertEqual(result, symbols)

    def test_builds_universe_on_cache_miss(self):
        assets = [
            _make_asset("AMZN", tradable=True, fractionable=True, exchange="NASDAQ"),
            _make_asset("NVDA", tradable=True, fractionable=True, exchange="NASDAQ"),
        ]
        client = MagicMock()
        client.get_all_assets.return_value = assets
        self.mod._MAJOR_EXCHANGES = {"NASDAQ"}

        snaps = {
            "AMZN": _make_snap(100.0, 2_000_000),
            "NVDA": _make_snap(200.0, 3_000_000),
        }
        mock_dc = MagicMock()
        mock_dc.get_stock_snapshots.return_value = snaps
        mock_cls = MagicMock(return_value=mock_dc)

        with (
            patch.object(self.mod, "StockHistoricalDataClient", mock_cls),
            patch.object(self.mod, "_save_cache"),
        ):
            result = self.mod.build_scan_universe(client)

        # Core symbols always present
        for sym in self.mod.STOCK_UNIVERSE:
            self.assertIn(sym, result)
        # Dynamic additions present
        self.assertIn("AMZN", result)
        self.assertIn("NVDA", result)

    def test_core_symbols_always_included(self):
        """Core STOCK_UNIVERSE symbols are present even when Alpaca returns nothing."""
        client = MagicMock()
        client.get_all_assets.return_value = []
        self.mod._MAJOR_EXCHANGES = {"NYSE"}

        mock_dc = MagicMock()
        mock_dc.get_stock_snapshots.return_value = {}
        mock_cls = MagicMock(return_value=mock_dc)

        with (
            patch.object(self.mod, "StockHistoricalDataClient", mock_cls),
            patch.object(self.mod, "_save_cache"),
        ):
            result = self.mod.build_scan_universe(client)

        for sym in self.mod.STOCK_UNIVERSE:
            self.assertIn(sym, result)

    def test_max_universe_size_respected(self):
        """Total symbols must not exceed _MAX_UNIVERSE_SIZE."""
        self.mod._MAX_UNIVERSE_SIZE = 5
        # Return 10 dynamic symbols
        assets = [
            _make_asset(f"SYM{i}", tradable=True, fractionable=True, exchange="NYSE")
            for i in range(10)
        ]
        client = MagicMock()
        client.get_all_assets.return_value = assets
        self.mod._MAJOR_EXCHANGES = {"NYSE"}

        snaps = {f"SYM{i}": _make_snap(50.0, 1_000_000) for i in range(10)}
        mock_dc = MagicMock()
        mock_dc.get_stock_snapshots.return_value = snaps
        mock_cls = MagicMock(return_value=mock_dc)

        with (
            patch.object(self.mod, "StockHistoricalDataClient", mock_cls),
            patch.object(self.mod, "_save_cache"),
        ):
            result = self.mod.build_scan_universe(client)

        self.assertLessEqual(len(result), self.mod._MAX_UNIVERSE_SIZE)

    def test_fallback_on_alpaca_error(self):
        """If get_all_assets raises, fall back to config.STOCK_UNIVERSE."""
        client = MagicMock()
        client.get_all_assets.side_effect = RuntimeError("Alpaca down")

        result = self.mod.build_scan_universe(client)
        self.assertEqual(result, list(self.mod.STOCK_UNIVERSE))

    def test_fallback_on_snapshot_error(self):
        """If the data client constructor raises, fall back to config.STOCK_UNIVERSE."""
        assets = [_make_asset("AMZN", tradable=True, fractionable=True, exchange="NYSE")]
        client = MagicMock()
        client.get_all_assets.return_value = assets
        self.mod._MAJOR_EXCHANGES = {"NYSE"}

        with patch.object(
            self.mod, "StockHistoricalDataClient", side_effect=RuntimeError("auth failed")
        ):
            result = self.mod.build_scan_universe(client)

        self.assertEqual(result, list(self.mod.STOCK_UNIVERSE))

    def test_no_duplicate_symbols(self):
        """No symbol should appear more than once in the result."""
        assets = [
            _make_asset("AAPL", tradable=True, fractionable=True, exchange="NASDAQ"),
            _make_asset("AMZN", tradable=True, fractionable=True, exchange="NASDAQ"),
        ]
        client = MagicMock()
        client.get_all_assets.return_value = assets
        self.mod._MAJOR_EXCHANGES = {"NASDAQ"}

        snaps = {
            "AAPL": _make_snap(150.0, 2_000_000),
            "AMZN": _make_snap(100.0, 2_000_000),
        }
        mock_dc = MagicMock()
        mock_dc.get_stock_snapshots.return_value = snaps
        mock_cls = MagicMock(return_value=mock_dc)

        with (
            patch.object(self.mod, "StockHistoricalDataClient", mock_cls),
            patch.object(self.mod, "_save_cache"),
        ):
            result = self.mod.build_scan_universe(client)

        self.assertEqual(len(result), len(set(result)), "Duplicate symbols in universe")

    def test_cache_is_written_on_success(self):
        assets = [_make_asset("AMZN", tradable=True, fractionable=True, exchange="NYSE")]
        client = MagicMock()
        client.get_all_assets.return_value = assets
        self.mod._MAJOR_EXCHANGES = {"NYSE"}

        snaps = {"AMZN": _make_snap(100.0, 1_000_000)}
        mock_dc = MagicMock()
        mock_dc.get_stock_snapshots.return_value = snaps
        mock_cls = MagicMock(return_value=mock_dc)
        mock_save = MagicMock()

        with (
            patch.object(self.mod, "StockHistoricalDataClient", mock_cls),
            patch.object(self.mod, "_save_cache", mock_save),
        ):
            self.mod.build_scan_universe(client)

        mock_save.assert_called_once()

    def test_cache_not_written_on_fallback(self):
        """Cache must NOT be written when we fall back due to error."""
        client = MagicMock()
        client.get_all_assets.side_effect = RuntimeError("oops")
        mock_save = MagicMock()

        with patch.object(self.mod, "_save_cache", mock_save):
            self.mod.build_scan_universe(client)

        mock_save.assert_not_called()


if __name__ == "__main__":
    unittest.main()
