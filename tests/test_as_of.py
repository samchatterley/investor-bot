"""Tests for data/as_of.py — the canonical point-in-time selector + tripwire."""

import unittest
from datetime import date

import pandas as pd

from data.as_of import (
    LookaheadError,
    assert_no_future,
    latest_as_of,
    split_adjust_as_of,
    visible_as_of,
)

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


class TestSplitAdjustAsOf(unittest.TestCase):
    def _df(self):
        idx = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"])
        # raw prices: $100 pre-split, 2:1 split on 01-03 -> $50 post-split
        return pd.DataFrame(
            {
                "Open": [100.0, 100.0, 50.0],
                "High": [100.0, 100.0, 50.0],
                "Low": [100.0, 100.0, 50.0],
                "Close": [100.0, 100.0, 50.0],
                "Volume": [10.0, 10.0, 20.0],
                "Stock Splits": [0.0, 0.0, 2.0],
            },
            index=idx,
        )

    def test_adjusts_pre_split_prices_as_of_split_date(self):
        adj = split_adjust_as_of(self._df(), "2026-01-03")
        self.assertEqual(
            list(adj["Close"]), [50.0, 50.0, 50.0]
        )  # pre-split halved to post-split scale
        self.assertEqual(list(adj["Volume"]), [20.0, 20.0, 20.0])  # pre-split volume doubled

    def test_no_adjustment_before_the_split(self):
        adj = split_adjust_as_of(self._df(), "2026-01-02")  # split not yet known
        self.assertEqual(list(adj["Close"]), [100.0, 100.0])  # raw, point-in-time correct

    def test_empty_when_as_of_precedes_data(self):
        self.assertTrue(split_adjust_as_of(self._df(), "2025-01-01").empty)

    def test_missing_splits_column_returns_sliced_unadjusted(self):
        out = split_adjust_as_of(self._df().drop(columns=["Stock Splits"]), "2026-01-03")
        self.assertEqual(list(out["Close"]), [100.0, 100.0, 50.0])  # untouched


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
