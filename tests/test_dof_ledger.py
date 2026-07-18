"""Tests for experiment/dof_ledger.py — the global degrees-of-freedom ledger (online FDR)."""

import json
import os
import shutil
import tempfile
import unittest

from experiment.dof_ledger import (
    LedgerState,
    build_ledger_lines,
    invest_level,
    load_ledger,
    new_state,
    record_batch,
    record_test,
    save_ledger,
)


class TestState(unittest.TestCase):
    def test_new_state_defaults(self):
        s = new_state()
        self.assertEqual((s.alpha, s.gamma, s.n_tests, s.n_rejections), (0.10, 0.5, 0, 0))
        self.assertAlmostEqual(s.wealth, 0.05)  # alpha / 2
        self.assertEqual(s.looks, [])

    def test_new_state_custom(self):
        s = new_state(alpha=0.05, gamma=0.25)
        self.assertEqual((s.alpha, s.gamma), (0.05, 0.25))
        self.assertAlmostEqual(s.wealth, 0.025)

    def test_invest_level_uncapped(self):
        self.assertAlmostEqual(invest_level(new_state()), 0.025)  # 0.5 * 0.05 < alpha

    def test_invest_level_capped_at_alpha(self):
        s = LedgerState(alpha=0.10, gamma=0.5, wealth=0.8, n_tests=0, n_rejections=0)
        self.assertAlmostEqual(invest_level(s), 0.10)  # min(0.4, 0.10)


class TestRecordTest(unittest.TestCase):
    def test_reject_refunds_wealth(self):
        new, look = record_test(new_state(), "h1", "miner", "d", 0.001, now="t0")
        self.assertTrue(look.rejected)
        self.assertAlmostEqual(new.wealth, 0.10)  # 0.05 - 0 + payout 0.05
        self.assertEqual((new.n_tests, new.n_rejections), (1, 1))
        self.assertEqual((look.created, look.alpha_level), ("t0", 0.025))

    def test_accept_pays_penalty(self):
        new, look = record_test(new_state(), "h1", "miner", "d", 0.9, now="t0")
        self.assertFalse(look.rejected)
        self.assertAlmostEqual(new.wealth, 0.05 - 0.025 / 0.975)
        self.assertEqual((new.n_tests, new.n_rejections), (1, 0))

    def test_default_timestamp_is_populated(self):
        _, look = record_test(new_state(), "h1", "miner", "d", 0.9)
        self.assertTrue(look.created)  # _now_iso() produced an ISO string

    def test_wealth_stays_positive_under_many_failures(self):
        s = new_state()
        for i in range(300):
            s, _ = record_test(s, f"h{i}", "miner", "d", 0.99, now="t")
        self.assertGreater(s.wealth, 0.0)  # the bar can never dead-end at zero

    def test_bar_tightens_after_a_failed_look(self):
        s = new_state()
        before = invest_level(s)
        s, _ = record_test(s, "h", "miner", "d", 0.99, now="t")
        self.assertLess(invest_level(s), before)  # spent-down -> stricter next bar


class TestRecordBatch(unittest.TestCase):
    def test_processes_ascending_by_pvalue(self):
        tests = [("weak", "miner", "d", 0.9), ("strong", "miner", "d", 0.001)]
        new, looks = record_batch(new_state(), tests, now="t")
        self.assertEqual(looks[0].id, "strong")  # smallest p tested first, on freshest budget
        self.assertEqual(new.n_tests, 2)


class TestBuildLedgerLines(unittest.TestCase):
    def test_empty_ledger(self):
        lines = build_ledger_lines(new_state())
        self.assertEqual(len(lines), 1)
        self.assertIn("no formal tests recorded yet", lines[0])

    def test_populated_ledger_reports_families_and_discoveries(self):
        s = new_state()
        s, _ = record_test(s, "h1", "miner", "d", 0.001, now="t")
        s, _ = record_test(s, "h2", "specialization", "d", 0.9, now="t")
        joined = "\n".join(build_ledger_lines(s))
        self.assertIn("2 formal test(s)", joined)
        self.assertIn("miner", joined)
        self.assertIn("specialization", joined)
        self.assertIn("1 discovery", joined)


class TestPersistence(unittest.TestCase):
    def _p(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d)
        return os.path.join(d, "sub", "dof_ledger.json")

    def test_missing_file_seeds_and_persists(self):
        p = self._p()
        loaded = load_ledger(p)
        self.assertTrue(os.path.exists(p))  # seeded to disk
        self.assertEqual(loaded.n_tests, 0)
        self.assertAlmostEqual(loaded.wealth, 0.05)

    def test_round_trip(self):
        p = self._p()
        s, _ = record_test(new_state(), "h1", "miner", "desc", 0.001, now="t0")
        save_ledger(s, p)
        loaded = load_ledger(p)
        self.assertEqual((loaded.n_tests, loaded.n_rejections), (1, 1))
        self.assertAlmostEqual(loaded.wealth, 0.10)
        self.assertEqual(loaded.looks[0].id, "h1")
        self.assertTrue(loaded.looks[0].rejected)

    def test_malformed_returns_fresh(self):
        p = self._p()
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write("{nope")
        self.assertEqual(load_ledger(p).n_tests, 0)

    def test_missing_key_returns_fresh(self):
        p = self._p()
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            json.dump({"alpha": 0.10}, f)  # missing gamma/wealth/... -> KeyError
        self.assertEqual(load_ledger(p).n_tests, 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
