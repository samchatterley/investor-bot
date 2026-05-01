"""Tests for utils/decision_log.py — AI decision audit trail."""
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch


def _decisions(buy_symbols=None, sell_symbols=None, market="Quiet day"):
    buys = [
        {"symbol": sym, "confidence": 8, "reasoning": "Good setup", "key_signal": "momentum"}
        for sym in (buy_symbols or [])
    ]
    sells = [
        {"symbol": sym, "action": "SELL", "confidence": 7, "reasoning": "Weak momentum"}
        for sym in (sell_symbols or [])
    ]
    return {
        "date": "2026-01-15",
        "market_summary": market,
        "buy_candidates": buys,
        "position_decisions": sells,
    }


class DecisionLogBase(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.decisions_path = os.path.join(self.tmpdir, "decisions.jsonl")
        self.patcher = patch("utils.decision_log._DECISIONS_PATH", self.decisions_path)
        self.log_dir_patcher = patch("utils.decision_log.LOG_DIR", self.tmpdir)
        self.patcher.start()
        self.log_dir_patcher.start()
        self.addCleanup(self.patcher.stop)
        self.addCleanup(self.log_dir_patcher.stop)
        self.addCleanup(shutil.rmtree, self.tmpdir)

    def _read_entries(self):
        if not os.path.exists(self.decisions_path):
            return []
        with open(self.decisions_path) as f:
            return [json.loads(line) for line in f if line.strip()]


class TestLogDecisions(DecisionLogBase):

    def test_buy_candidate_written(self):
        from utils.decision_log import log_decisions
        log_decisions(_decisions(buy_symbols=["AAPL"]), "open", set())
        entries = self._read_entries()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["symbol"], "AAPL")
        self.assertEqual(entries[0]["action"], "BUY")

    def test_position_decision_written(self):
        from utils.decision_log import log_decisions
        log_decisions(_decisions(sell_symbols=["MSFT"]), "midday", set())
        entries = self._read_entries()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["symbol"], "MSFT")
        self.assertEqual(entries[0]["action"], "SELL")

    def test_multiple_candidates_all_written(self):
        from utils.decision_log import log_decisions
        log_decisions(_decisions(buy_symbols=["AAPL", "NVDA"]), "open", set())
        entries = self._read_entries()
        self.assertEqual(len(entries), 2)

    def test_executed_flag_true_when_in_executed_set(self):
        from utils.decision_log import log_decisions
        log_decisions(_decisions(buy_symbols=["AAPL"]), "open", {"AAPL"})
        entries = self._read_entries()
        self.assertTrue(entries[0]["executed"])

    def test_executed_flag_false_when_not_in_executed_set(self):
        from utils.decision_log import log_decisions
        log_decisions(_decisions(buy_symbols=["AAPL"]), "open", set())
        entries = self._read_entries()
        self.assertFalse(entries[0]["executed"])

    def test_market_summary_included(self):
        from utils.decision_log import log_decisions
        log_decisions(_decisions(buy_symbols=["AAPL"], market="Strong bull day"), "open", set())
        entries = self._read_entries()
        self.assertEqual(entries[0]["market_summary"], "Strong bull day")

    def test_mode_included(self):
        from utils.decision_log import log_decisions
        log_decisions(_decisions(buy_symbols=["AAPL"]), "midday", set())
        entries = self._read_entries()
        self.assertEqual(entries[0]["mode"], "midday")

    def test_confidence_and_reasoning_included(self):
        from utils.decision_log import log_decisions
        log_decisions(_decisions(buy_symbols=["AAPL"]), "open", set())
        entries = self._read_entries()
        self.assertEqual(entries[0]["confidence"], 8)
        self.assertEqual(entries[0]["reasoning"], "Good setup")

    def test_key_signal_none_for_position_decisions(self):
        from utils.decision_log import log_decisions
        log_decisions(_decisions(sell_symbols=["MSFT"]), "open", set())
        entries = self._read_entries()
        self.assertIsNone(entries[0]["key_signal"])

    def test_empty_decisions_writes_nothing(self):
        from utils.decision_log import log_decisions
        log_decisions(_decisions(), "open", set())
        self.assertEqual(self._read_entries(), [])

    def test_entries_appended_across_calls(self):
        from utils.decision_log import log_decisions
        log_decisions(_decisions(buy_symbols=["AAPL"]), "open", set())
        log_decisions(_decisions(buy_symbols=["NVDA"]), "midday", set())
        self.assertEqual(len(self._read_entries()), 2)

    def test_each_entry_has_timestamp(self):
        from utils.decision_log import log_decisions
        log_decisions(_decisions(buy_symbols=["AAPL"]), "open", set())
        entries = self._read_entries()
        self.assertIn("ts", entries[0])


class TestLoadDecisions(DecisionLogBase):

    def test_returns_empty_when_file_missing(self):
        from utils.decision_log import load_decisions
        result = load_decisions()
        self.assertEqual(result, [])

    def test_returns_written_entries(self):
        from utils.decision_log import load_decisions, log_decisions
        log_decisions(_decisions(buy_symbols=["AAPL", "NVDA"]), "open", {"AAPL"})
        result = load_decisions()
        self.assertEqual(len(result), 2)

    def test_respects_n_limit(self):
        from utils.decision_log import load_decisions, log_decisions
        for i in range(10):
            log_decisions(_decisions(buy_symbols=[f"SYM{i}"]), "open", set())
        result = load_decisions(n=5)
        self.assertEqual(len(result), 5)

    def test_returns_last_n_not_first_n(self):
        from utils.decision_log import load_decisions, log_decisions
        for i in range(5):
            log_decisions(_decisions(buy_symbols=[f"SYM{i}"]), "open", set())
        result = load_decisions(n=2)
        symbols = [e["symbol"] for e in result]
        self.assertIn("SYM4", symbols)
        self.assertNotIn("SYM0", symbols)

    def test_skips_malformed_lines(self):
        from utils.decision_log import load_decisions
        with open(self.decisions_path, "w") as f:
            f.write('{"valid": true}\n')
            f.write('not json at all\n')
            f.write('{"also_valid": true}\n')
        result = load_decisions()
        self.assertEqual(len(result), 2)
