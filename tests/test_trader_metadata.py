"""
Tests for the pure file-I/O position metadata functions in execution/trader.py.
Alpaca API calls (place_buy_order, get_account_info, etc.) are excluded — those
require a live/mock broker connection and belong in integration tests.
"""
import os
import shutil
import tempfile
import unittest
from datetime import date
from unittest.mock import patch


def _meta_patcher(tmpdir):
    meta_path = os.path.join(tmpdir, "positions_meta.json")
    return patch("execution.trader._META_PATH", meta_path), patch("execution.trader.LOG_DIR", tmpdir)


class TestRecordBuyAndSell(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        p1, p2 = _meta_patcher(self.tmpdir)
        self.p1 = p1.start()
        self.p2 = p2.start()
        self.addCleanup(p1.stop)
        self.addCleanup(p2.stop)
        self.addCleanup(shutil.rmtree, self.tmpdir)

    def test_record_buy_creates_entry(self):
        from execution.trader import record_buy, get_position_meta
        record_buy("AAPL", 175.50, signal="momentum", regime="BULL_TRENDING", confidence=8)
        meta = get_position_meta("AAPL")
        self.assertEqual(meta["signal"], "momentum")
        self.assertEqual(meta["regime"], "BULL_TRENDING")
        self.assertEqual(meta["confidence"], 8)
        self.assertAlmostEqual(meta["entry_price"], 175.50, places=2)

    def test_record_buy_stores_entry_date(self):
        from execution.trader import record_buy, get_position_meta
        record_buy("MSFT", 400.0)
        meta = get_position_meta("MSFT")
        self.assertIn("entry_date", meta)
        date.fromisoformat(meta["entry_date"])  # should not raise

    def test_record_sell_removes_entry(self):
        from execution.trader import record_buy, record_sell, get_position_meta
        record_buy("NVDA", 800.0, signal="momentum")
        record_sell("NVDA")
        meta = get_position_meta("NVDA")
        self.assertEqual(meta["signal"], "unknown")  # default — entry gone

    def test_record_sell_missing_symbol_is_safe(self):
        from execution.trader import record_sell
        record_sell("GHOST")  # should not raise

    def test_multiple_positions_stored_independently(self):
        from execution.trader import record_buy, get_position_meta
        record_buy("AAPL", 175.0, signal="momentum", confidence=8)
        record_buy("MSFT", 400.0, signal="mean_reversion", confidence=7)
        self.assertEqual(get_position_meta("AAPL")["signal"], "momentum")
        self.assertEqual(get_position_meta("MSFT")["signal"], "mean_reversion")


class TestGetPositionMeta(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        p1, p2 = _meta_patcher(self.tmpdir)
        self.p1 = p1.start()
        self.p2 = p2.start()
        self.addCleanup(p1.stop)
        self.addCleanup(p2.stop)
        self.addCleanup(shutil.rmtree, self.tmpdir)

    def test_unknown_symbol_returns_defaults(self):
        from execution.trader import get_position_meta
        meta = get_position_meta("UNKNOWN")
        self.assertEqual(meta["signal"], "unknown")
        self.assertEqual(meta["regime"], "UNKNOWN")
        self.assertEqual(meta["confidence"], 0)
        self.assertEqual(meta["entry_price"], 0.0)

    def test_defaults_not_overwritten_by_partial_metadata(self):
        from execution.trader import record_buy, get_position_meta
        record_buy("AAPL", 180.0)  # no regime or confidence
        meta = get_position_meta("AAPL")
        self.assertEqual(meta["regime"], "UNKNOWN")
        self.assertEqual(meta["confidence"], 0)


class TestPositionAges(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        p1, p2 = _meta_patcher(self.tmpdir)
        self.p1 = p1.start()
        self.p2 = p2.start()
        self.addCleanup(p1.stop)
        self.addCleanup(p2.stop)
        self.addCleanup(shutil.rmtree, self.tmpdir)

    def test_position_entered_today_has_age_one(self):
        from execution.trader import record_buy, get_position_ages
        record_buy("AAPL", 175.0)
        ages = get_position_ages()
        self.assertGreaterEqual(ages["AAPL"], 1)

    def test_get_stale_positions_below_threshold(self):
        from execution.trader import record_buy, get_stale_positions
        record_buy("AAPL", 175.0)
        # A position entered today won't be stale at max_days=3
        stale = get_stale_positions(max_days=30)
        self.assertNotIn("AAPL", stale)

    def test_get_stale_positions_with_low_threshold(self):
        from execution.trader import record_buy, get_stale_positions
        record_buy("AAPL", 175.0)
        # With max_days=1, a 1-day position is stale
        stale = get_stale_positions(max_days=1)
        self.assertIn("AAPL", stale)

    def test_no_positions_returns_empty(self):
        from execution.trader import get_position_ages, get_stale_positions
        self.assertEqual(get_position_ages(), {})
        self.assertEqual(get_stale_positions(), [])


class TestLoadMetaCorruption(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        p1, p2 = _meta_patcher(self.tmpdir)
        self.p1 = p1.start()
        self.p2 = p2.start()
        self.addCleanup(p1.stop)
        self.addCleanup(p2.stop)
        self.addCleanup(shutil.rmtree, self.tmpdir)

    def test_corrupted_json_returns_empty_dict_not_exception(self):
        import json
        from execution.trader import _META_PATH, get_position_meta
        with open(_META_PATH, "w") as f:
            f.write("not valid json {{{")
        meta = get_position_meta("AAPL")
        self.assertEqual(meta["signal"], "unknown")

    def test_empty_file_returns_defaults(self):
        from execution.trader import _META_PATH, get_position_meta
        with open(_META_PATH, "w") as f:
            f.write("")
        meta = get_position_meta("AAPL")
        self.assertEqual(meta["signal"], "unknown")
        self.assertEqual(meta["confidence"], 0)
