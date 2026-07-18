"""Lookahead audit of the existing point-in-time providers.

Substrate brick 2 (Phase B): the fundamentals PIT helpers only *assert* "point-in-time safe" in a
docstring. Here we drive the lookahead guard against them on synthetic event data so the claim is
enforced -- masking every event after the as-of date must not change the answer. Also exercises
data.as_of.assert_no_future as the tripwire on the same provider data shape.
"""

import unittest
from datetime import date

from backtest.historical_fundamentals import earnings_miss_active_on_date, pead_active_on_date
from data.as_of import assert_no_future
from experiment.lookahead_guard import audit_no_lookahead

_HIST = {
    "SYM": [
        {"date": date(2026, 1, 5), "surprise_pct": 12.0},  # beat
        {"date": date(2026, 1, 12), "surprise_pct": -15.0},  # miss
        {"date": date(2026, 1, 20), "surprise_pct": 12.0},  # beat
    ]
}
_DATES = ["2026-01-06", "2026-01-13", "2026-01-15", "2026-01-21", "2026-01-25"]


def _poison(hist, as_of):
    """Drop every earnings event strictly after as_of (the future did not exist)."""
    cutoff = date.fromisoformat(as_of)
    return {sym: [e for e in evs if e["date"] <= cutoff] for sym, evs in hist.items()}


def _pead_on(hist, as_of):
    return pead_active_on_date("SYM", date.fromisoformat(as_of), hist)


def _miss_on(hist, as_of):
    return earnings_miss_active_on_date("SYM", date.fromisoformat(as_of), hist)


class TestFundamentalsPITAudit(unittest.TestCase):
    def test_pead_is_lookahead_clean(self):
        self.assertEqual(audit_no_lookahead(_pead_on, _HIST, _DATES, poison=_poison), [])

    def test_earnings_miss_is_lookahead_clean(self):
        self.assertEqual(audit_no_lookahead(_miss_on, _HIST, _DATES, poison=_poison), [])

    def test_assert_no_future_tripwire_on_visible_events(self):
        # the visible slice a point-in-time provider works from must never carry a future event
        for d in _DATES:
            assert_no_future(_poison(_HIST, d)["SYM"], d, date_of=lambda e: e["date"])  # no raise


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
