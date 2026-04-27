import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch


class AuditLogBase(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.audit_path = os.path.join(self.tmpdir, "audit.jsonl")
        self.patcher = patch("utils.audit_log._AUDIT_PATH", self.audit_path)
        self.log_dir_patcher = patch("utils.audit_log.LOG_DIR", self.tmpdir)
        self.patcher.start()
        self.log_dir_patcher.start()
        self.addCleanup(self.patcher.stop)
        self.addCleanup(self.log_dir_patcher.stop)
        self.addCleanup(shutil.rmtree, self.tmpdir)

    def _read_events(self):
        if not os.path.exists(self.audit_path):
            return []
        with open(self.audit_path) as f:
            return [json.loads(line) for line in f if line.strip()]


class TestAuditLogWrites(AuditLogBase):

    def test_log_run_start_writes_event(self):
        from utils.audit_log import log_run_start
        log_run_start("open", 100_000, 10_000, True)
        events = self._read_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event"], "RUN_START")
        self.assertEqual(events[0]["mode"], "open")

    def test_log_run_end_writes_event(self):
        from utils.audit_log import log_run_end
        log_run_end("close", 1234.56, 3, 101_234.56)
        events = self._read_events()
        self.assertEqual(events[0]["event"], "RUN_END")
        self.assertEqual(events[0]["trades_executed"], 3)

    def test_log_order_placed(self):
        from utils.audit_log import log_order_placed
        log_order_placed("AAPL", "BUY", 5000.0, "order-123")
        events = self._read_events()
        self.assertEqual(events[0]["event"], "ORDER_PLACED")
        self.assertEqual(events[0]["symbol"], "AAPL")
        self.assertEqual(events[0]["order_id"], "order-123")

    def test_log_order_filled(self):
        from utils.audit_log import log_order_filled
        log_order_filled("AAPL", "order-123", 28.571)
        events = self._read_events()
        self.assertEqual(events[0]["event"], "ORDER_FILLED")

    def test_log_position_closed(self):
        from utils.audit_log import log_position_closed
        log_position_closed("MSFT", "earnings_exit", -1.23)
        events = self._read_events()
        self.assertEqual(events[0]["event"], "POSITION_CLOSED")
        self.assertEqual(events[0]["reason"], "earnings_exit")

    def test_log_ai_decision(self):
        from utils.audit_log import log_ai_decision
        log_ai_decision("Bullish day", 2, 1)
        events = self._read_events()
        self.assertEqual(events[0]["event"], "AI_DECISION")
        self.assertEqual(events[0]["buy_count"], 2)

    def test_log_validation_failure(self):
        from utils.audit_log import log_validation_failure
        log_validation_failure(["bad symbol GHOST"])
        events = self._read_events()
        self.assertEqual(events[0]["event"], "VALIDATION_FAILURE")
        self.assertIn("GHOST", events[0]["errors"][0])

    def test_log_circuit_breaker(self):
        from utils.audit_log import log_circuit_breaker
        log_circuit_breaker(-13.5)
        events = self._read_events()
        self.assertEqual(events[0]["event"], "CIRCUIT_BREAKER")
        self.assertAlmostEqual(events[0]["drawdown_pct"], -13.5)

    def test_log_kill_switch(self):
        from utils.audit_log import log_kill_switch
        log_kill_switch(4)
        events = self._read_events()
        self.assertEqual(events[0]["event"], "KILL_SWITCH")
        self.assertEqual(events[0]["positions_closed"], 4)

    def test_log_halt_cleared(self):
        from utils.audit_log import log_halt_cleared
        log_halt_cleared()
        events = self._read_events()
        self.assertEqual(events[0]["event"], "HALT_CLEARED")

    def test_events_are_appended_not_overwritten(self):
        from utils.audit_log import log_run_start, log_run_end
        log_run_start("open", 100_000, 10_000, True)
        log_run_end("open", 500.0, 2, 100_500)
        events = self._read_events()
        self.assertEqual(len(events), 2)

    def test_each_event_has_timestamp(self):
        from utils.audit_log import log_run_start
        log_run_start("open", 100_000, 10_000, False)
        events = self._read_events()
        self.assertIn("ts", events[0])
        self.assertIn("T", events[0]["ts"])  # ISO format check

    def test_log_daily_loss_limit(self):
        from utils.audit_log import log_daily_loss_limit
        log_daily_loss_limit(-4.2)
        events = self._read_events()
        self.assertEqual(events[0]["event"], "DAILY_LOSS_LIMIT")
        self.assertAlmostEqual(events[0]["loss_pct"], -4.2)

    def test_log_earnings_exit(self):
        from utils.audit_log import log_earnings_exit
        log_earnings_exit("AAPL", "2026-05-01")
        events = self._read_events()
        self.assertEqual(events[0]["event"], "EARNINGS_EXIT")
        self.assertEqual(events[0]["symbol"], "AAPL")
        self.assertEqual(events[0]["earnings_date"], "2026-05-01")

    def test_log_macro_skip(self):
        from utils.audit_log import log_macro_skip
        log_macro_skip("FOMC Rate Decision")
        events = self._read_events()
        self.assertEqual(events[0]["event"], "MACRO_SKIP")
        self.assertIn("FOMC", events[0]["macro_event"])
