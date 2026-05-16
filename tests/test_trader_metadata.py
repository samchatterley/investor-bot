"""
Tests for position metadata functions in execution/trader.py (SQLite backend).
Alpaca API calls are excluded — those require a live/mock broker connection.
"""

import os
import shutil
import tempfile
import unittest
from datetime import date
from unittest.mock import patch


class _MetaTestBase(unittest.TestCase):
    def setUp(self):
        import utils.db as db_module

        self.tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(self.tmpdir, "test.db")

        # Patch DB path, reset init flag, and suppress JSON migration so the
        # temp DB starts empty rather than inheriting live positions_meta.json data.
        self.p_path = patch.object(db_module, "_DB_PATH", db_path)
        self.p_init = patch.object(db_module, "_initialized", False)
        self.p_migrate = patch.object(db_module, "_migrate_json_state", lambda: None)
        self.p_path.start()
        self.p_init.start()
        self.p_migrate.start()
        self.addCleanup(self.p_path.stop)
        self.addCleanup(self.p_init.stop)
        self.addCleanup(self.p_migrate.stop)
        self.addCleanup(shutil.rmtree, self.tmpdir)

        # Initialise schema in the isolated temp DB
        db_module._initialized = False
        from utils.db import init_db

        init_db()


class TestRecordBuyAndSell(_MetaTestBase):
    def test_record_buy_creates_entry(self):
        from execution.trader import get_position_meta, record_buy

        record_buy("AAPL", 175.50, signal="momentum", regime="BULL_TRENDING", confidence=8)
        meta = get_position_meta("AAPL")
        self.assertEqual(meta["signal"], "momentum")
        self.assertEqual(meta["regime"], "BULL_TRENDING")
        self.assertEqual(meta["confidence"], 8)
        self.assertAlmostEqual(meta["entry_price"], 175.50, places=2)

    def test_record_buy_stores_entry_date(self):
        from execution.trader import get_position_meta, record_buy

        record_buy("MSFT", 400.0)
        meta = get_position_meta("MSFT")
        self.assertIn("entry_date", meta)
        date.fromisoformat(meta["entry_date"])  # should not raise

    def test_record_sell_removes_entry(self):
        from execution.trader import get_position_meta, record_buy, record_sell

        record_buy("NVDA", 800.0, signal="momentum")
        record_sell("NVDA")
        meta = get_position_meta("NVDA")
        self.assertEqual(meta["signal"], "unknown")  # default — entry gone

    def test_record_sell_missing_symbol_is_safe(self):
        from execution.trader import record_sell

        record_sell("GHOST")  # should not raise

    def test_multiple_positions_stored_independently(self):
        from execution.trader import get_position_meta, record_buy

        record_buy("AAPL", 175.0, signal="momentum", confidence=8)
        record_buy("MSFT", 400.0, signal="mean_reversion", confidence=7)
        self.assertEqual(get_position_meta("AAPL")["signal"], "momentum")
        self.assertEqual(get_position_meta("MSFT")["signal"], "mean_reversion")


class TestGetPositionMeta(_MetaTestBase):
    def test_unknown_symbol_returns_defaults(self):
        from execution.trader import get_position_meta

        meta = get_position_meta("UNKNOWN")
        self.assertEqual(meta["signal"], "unknown")
        self.assertEqual(meta["regime"], "UNKNOWN")
        self.assertEqual(meta["confidence"], 0)
        self.assertEqual(meta["entry_price"], 0.0)

    def test_defaults_not_overwritten_by_partial_metadata(self):
        from execution.trader import get_position_meta, record_buy

        record_buy("AAPL", 180.0)  # no regime or confidence supplied → defaults
        meta = get_position_meta("AAPL")
        self.assertEqual(meta["regime"], "UNKNOWN")
        self.assertEqual(meta["confidence"], 0)


class TestPositionAges(_MetaTestBase):
    def test_position_entered_today_has_age_one(self):
        from execution.trader import get_position_ages, record_buy

        record_buy("AAPL", 175.0)
        ages = get_position_ages()
        self.assertGreaterEqual(ages["AAPL"], 1)

    def test_get_stale_positions_below_threshold(self):
        from execution.trader import get_stale_positions, record_buy

        record_buy("AAPL", 175.0)
        stale = get_stale_positions(max_days=30)
        self.assertNotIn("AAPL", stale)

    def test_get_stale_positions_with_low_threshold(self):
        from execution.trader import get_stale_positions, record_buy

        record_buy("AAPL", 175.0)
        stale = get_stale_positions(max_days=1)
        self.assertIn("AAPL", stale)

    def test_no_positions_returns_empty(self):
        from execution.trader import get_position_ages, get_stale_positions

        self.assertEqual(get_position_ages(), {})
        self.assertEqual(get_stale_positions(), [])


class TestRecordPartialExit(_MetaTestBase):
    """Line 281: record_partial_exit happy path sets partial_exit_taken_at."""

    def test_record_partial_exit_sets_timestamp(self):
        from execution.trader import get_position_meta, record_buy, record_partial_exit

        record_buy("AAPL", 180.0)
        record_partial_exit("AAPL")
        meta = get_position_meta("AAPL")
        self.assertIsNotNone(meta.get("partial_exit_taken_at"))

    def test_record_partial_exit_unknown_symbol_does_not_raise(self):
        from execution.trader import record_partial_exit

        # Symbol not in DB — UPDATE touches no rows but should not raise
        try:
            record_partial_exit("GHOST")
        except Exception:  # pragma: no cover
            self.fail("record_partial_exit raised on unknown symbol")
