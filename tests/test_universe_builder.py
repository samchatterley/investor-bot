"""Tests for data/universe_builder.py — the dynamic rules-based tradeable universe."""

import os
import tempfile
import types
import unittest
from datetime import date
from unittest.mock import patch

import data.universe_builder as ub


def _asset(symbol, name="Acme Corp", tradable=True, fractionable=True, exchange="NASDAQ"):
    return types.SimpleNamespace(
        symbol=symbol,
        name=name,
        tradable=tradable,
        fractionable=fractionable,
        exchange=types.SimpleNamespace(value=exchange),
    )


class TestIsOperatingStock(unittest.TestCase):
    def test_plain_ticker_ok(self):
        self.assertTrue(ub._is_operating_stock("AAPL", "Apple Inc"))

    def test_five_letter_ok_when_not_wur(self):
        self.assertTrue(ub._is_operating_stock("GOOGL", "Alphabet"))

    def test_dot_symbol_rejected(self):
        self.assertFalse(ub._is_operating_stock("BRK.B", "Berkshire"))

    def test_too_long_rejected(self):
        self.assertFalse(ub._is_operating_stock("TOOLONG", "Whatever"))

    def test_digit_symbol_rejected(self):
        self.assertFalse(ub._is_operating_stock("ABC1", "Whatever"))

    def test_warrant_unit_right_suffix_rejected(self):
        self.assertFalse(ub._is_operating_stock("ABCDW", "Acme Warrant"))
        self.assertFalse(ub._is_operating_stock("ABCDU", "Acme Unit"))
        self.assertFalse(ub._is_operating_stock("ABCDR", "Acme Right"))

    def test_fund_name_rejected(self):
        self.assertFalse(ub._is_operating_stock("SPY", "SPDR S&P 500 ETF Trust"))

    def test_empty_symbol_rejected(self):
        self.assertFalse(ub._is_operating_stock("", "x"))


class TestScreenAssets(unittest.TestCase):
    def test_applies_all_filters_and_sorts(self):
        assets = [
            _asset("ZZZZ"),  # ok (sorts last)
            _asset("AAPL"),  # ok
            _asset("NOTR", tradable=False),  # not tradable
            _asset("NOFR", fractionable=False),  # not fractionable
            _asset("ARCA", exchange="ARCA"),  # wrong exchange
            _asset("SPY", name="SPDR ETF Trust"),  # fund name
            _asset("ABCDW"),  # warrant
        ]
        out = ub._screen_assets(assets, ub._DEFAULT_EXCHANGES, require_fractionable=True)
        self.assertEqual(out, ["AAPL", "ZZZZ"])

    def test_require_fractionable_false_keeps_nonfractionable(self):
        assets = [_asset("AAPL", fractionable=False)]
        self.assertEqual(
            ub._screen_assets(assets, ub._DEFAULT_EXCHANGES, require_fractionable=False), ["AAPL"]
        )


class TestCacheAndBuild(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._p = patch.object(ub, "_CACHE_PATH", os.path.join(self._tmp, "u.json"))
        self._p.start()

    def tearDown(self):
        self._p.stop()

    def test_load_missing_returns_empty(self):
        self.assertEqual(ub._load_cache(), {})

    def test_load_corrupt_returns_empty(self):
        with open(ub._CACHE_PATH, "w") as f:
            f.write("{bad")
        self.assertEqual(ub._load_cache(), {})

    def test_save_then_load(self):
        ub._save_cache("2024-01-01", ["AAA"])
        self.assertEqual(ub._load_cache()["symbols"], ["AAA"])

    def test_cache_hit_skips_fetch(self):
        ub._save_cache(date.today().isoformat(), ["AAA", "BBB"])
        with patch.object(ub, "_fetch_assets") as mf:
            out = ub.build_universe()
        mf.assert_not_called()
        self.assertEqual(out, ["AAA", "BBB"])

    def test_cache_miss_fetches_screens_saves(self):
        with patch.object(ub, "_fetch_assets", return_value=[_asset("AAPL"), _asset("MSFT")]):
            out = ub.build_universe()
        self.assertEqual(out, ["AAPL", "MSFT"])
        self.assertEqual(ub._load_cache()["symbols"], ["AAPL", "MSFT"])

    def test_fetch_failure_returns_empty(self):
        with patch.object(ub, "_fetch_assets", side_effect=RuntimeError("boom")):
            self.assertEqual(ub.build_universe(), [])

    def test_no_cache_does_not_write(self):
        with patch.object(ub, "_fetch_assets", return_value=[_asset("AAPL")]):
            out = ub.build_universe(use_cache=False)
        self.assertEqual(out, ["AAPL"])
        self.assertEqual(ub._load_cache(), {})


class TestFetchAssets(unittest.TestCase):
    def test_fetch_assets_calls_alpaca(self):
        fake_client = types.SimpleNamespace(get_all_assets=lambda req: [_asset("AAPL")])
        with (
            patch("alpaca.trading.client.TradingClient", return_value=fake_client),
            patch("alpaca.trading.requests.GetAssetsRequest"),
            patch("alpaca.trading.enums.AssetClass"),
            patch("alpaca.trading.enums.AssetStatus"),
        ):
            out = ub._fetch_assets()
        self.assertEqual(out[0].symbol, "AAPL")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
