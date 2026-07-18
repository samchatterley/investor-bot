"""Tests for data/as_of.py — the canonical point-in-time selector + tripwire."""

import unittest
from datetime import date

from data.as_of import LookaheadError, assert_no_future, latest_as_of, visible_as_of

_EVENTS = [
    {"d": "2026-01-01", "v": "a"},
    {"d": "2026-01-05", "v": "b"},
    {"d": "2026-01-10", "v": "c"},
]


def _d(r):
    return r["d"]


class TestVisibleAsOf(unittest.TestCase):
    def test_excludes_the_future_and_preserves_order(self):
        got = visible_as_of(_EVENTS, "2026-01-05", date_of=_d)
        self.assertEqual([r["v"] for r in got], ["a", "b"])  # c (01-10) excluded

    def test_within_days_lower_bound(self):
        # on 01-10 with a 3-day window: only 01-10 (01-05 is 5 days back, excluded)
        got = visible_as_of(_EVENTS, "2026-01-10", date_of=_d, within_days=3)
        self.assertEqual([r["v"] for r in got], ["c"])

    def test_accepts_date_objects(self):
        got = visible_as_of(_EVENTS, date(2026, 1, 1), date_of=lambda r: date.fromisoformat(r["d"]))
        self.assertEqual([r["v"] for r in got], ["a"])


class TestLatestAsOf(unittest.TestCase):
    def test_returns_most_recent_visible(self):
        self.assertEqual(latest_as_of(_EVENTS, "2026-01-07", date_of=_d)["v"], "b")

    def test_none_when_nothing_visible(self):
        self.assertIsNone(latest_as_of(_EVENTS, "2025-12-31", date_of=_d))


class TestAssertNoFuture(unittest.TestCase):
    def test_passes_when_clean(self):
        assert_no_future(_EVENTS[:2], "2026-01-05", date_of=_d)  # no raise

    def test_raises_on_a_future_record(self):
        with self.assertRaises(LookaheadError):
            assert_no_future(_EVENTS, "2026-01-05", date_of=_d)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
