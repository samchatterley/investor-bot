"""Tests for utils/order_ledger.py — exception handlers and branch coverage."""

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import utils.db as db_module


class OrderLedgerBase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self._db_patchers = [
            patch.object(db_module, "_DB_PATH", self.db_path),
            patch.object(db_module, "_initialized", False),
            patch.object(db_module, "_migrate_json_state", lambda: None),
        ]
        for p in self._db_patchers:
            p.start()
        from utils.db import init_db

        init_db()

    def tearDown(self):
        for p in self._db_patchers:
            p.stop()
        shutil.rmtree(self.tmpdir)


class TestUpdateIntentException(OrderLedgerBase):
    """Lines 84-85: update_intent swallows DB exceptions."""

    def test_db_error_does_not_raise(self):
        from utils.order_ledger import update_intent

        with patch("utils.order_ledger.get_db", side_effect=RuntimeError("db locked")):
            update_intent("missing-id", "submitted")

    def test_db_error_with_broker_id_does_not_raise(self):
        from utils.order_ledger import update_intent

        with patch("utils.order_ledger.get_db", side_effect=RuntimeError("db locked")):
            update_intent("missing-id", "filled", broker_order_id="broker-123")


class TestLogOrderEventException(OrderLedgerBase):
    """Lines 104-105: log_order_event swallows DB exceptions."""

    def test_db_error_does_not_raise(self):
        from utils.order_ledger import log_order_event

        with patch("utils.order_ledger.get_db", side_effect=RuntimeError("table missing")):
            log_order_event("test-id", "FILL", {"qty": 10})


class TestGetUnresolvedIntents(OrderLedgerBase):
    """Lines 204, 209-211: get_unresolved_intents else branch and exception path."""

    def _insert_intent(self, symbol, status, trade_date="2026-01-15"):
        from utils.db import get_db

        with get_db() as conn:
            conn.execute(
                "INSERT INTO order_intents "
                "(symbol, side, trade_date, intended_notional, client_order_id, "
                "status, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
                (
                    symbol,
                    "BUY",
                    trade_date,
                    500.0,
                    f"ib-{symbol}-BUY-{trade_date}",
                    status,
                    "2026-01-15T09:00:00+00:00",
                    "2026-01-15T09:00:00+00:00",
                ),
            )

    def test_no_trade_date_returns_all_unresolved(self):
        """Line 204: else branch — no trade_date filter queries all unresolved statuses."""
        from utils.order_ledger import get_unresolved_intents

        self._insert_intent("AAPL", "timeout", "2026-01-14")
        self._insert_intent("MSFT", "timeout", "2026-01-15")

        result = get_unresolved_intents()
        symbols = {r["symbol"] for r in result}
        self.assertIn("AAPL", symbols)
        self.assertIn("MSFT", symbols)

    def test_trade_date_filters_to_that_day(self):
        """When trade_date is provided, only that day's intents are returned."""
        from utils.order_ledger import get_unresolved_intents

        self._insert_intent("AAPL", "timeout", "2026-01-14")
        self._insert_intent("MSFT", "timeout", "2026-01-15")

        result = get_unresolved_intents(trade_date="2026-01-15")
        symbols = {r["symbol"] for r in result}
        self.assertIn("MSFT", symbols)
        self.assertNotIn("AAPL", symbols)

    def test_exception_returns_empty_list(self):
        """Lines 209-211: exception is swallowed, empty list returned."""
        from utils.order_ledger import get_unresolved_intents

        with patch("utils.order_ledger.get_db", side_effect=RuntimeError("db gone")):
            result = get_unresolved_intents()
        self.assertEqual(result, [])

    def test_exception_with_trade_date_returns_empty_list(self):
        from utils.order_ledger import get_unresolved_intents

        with patch("utils.order_ledger.get_db", side_effect=RuntimeError("db gone")):
            result = get_unresolved_intents(trade_date="2026-01-15")
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
