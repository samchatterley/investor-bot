"""Tests for utils/feed_status.py — the live feed-status recorder."""

import unittest

from utils import feed_status


class TestFeedStatus(unittest.TestCase):
    def setUp(self):
        feed_status.reset()

    def tearDown(self):
        feed_status.reset()

    def test_record_ok_not_in_degraded(self):
        feed_status.record("spy_5d", True)
        self.assertEqual(feed_status.degraded(), [])

    def test_record_failure_with_detail_appears_and_logs(self):
        with self.assertLogs("utils.feed_status", level="WARNING") as cm:
            feed_status.record("vix", False, "empty history")
        self.assertIn("vix", feed_status.degraded())
        self.assertTrue(any("feed degraded: vix" in m and "empty history" in m for m in cm.output))

    def test_record_failure_without_detail_still_logs(self):
        with self.assertLogs("utils.feed_status", level="WARNING") as cm:
            feed_status.record("x", False)
        self.assertIn("x", feed_status.degraded())
        self.assertTrue(any("feed degraded: x" in m for m in cm.output))

    def test_latest_outcome_wins(self):
        feed_status.record("spy_5d", False, "down")
        feed_status.record("spy_5d", True)  # recovered
        self.assertNotIn("spy_5d", feed_status.degraded())

    def test_degraded_is_sorted(self):
        feed_status.record("zeta", False)
        feed_status.record("alpha", False)
        self.assertEqual(feed_status.degraded(), ["alpha", "zeta"])

    def test_snapshot_returns_defensive_copy(self):
        feed_status.record("vix", True)
        snap = feed_status.snapshot()
        self.assertIn("vix", snap)
        snap["vix"]["ok"] = False  # mutate the copy
        self.assertTrue(feed_status.snapshot()["vix"]["ok"])  # original unaffected

    def test_reset_clears_state(self):
        feed_status.record("vix", False)
        feed_status.reset()
        self.assertEqual(feed_status.degraded(), [])
        self.assertEqual(feed_status.snapshot(), {})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
