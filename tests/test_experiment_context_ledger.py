import unittest
from datetime import datetime, timedelta

from experiment.context_ledger import (
    TIMESTAMP_CLEAN,
    TIMESTAMP_REJECTED,
    TIMESTAMP_UNCERTAIN,
    ContextItem,
    assess_admissibility,
    filter_admissible,
)

_DECISION = datetime(2026, 6, 14, 15, 0, 0)


def _item(**kwargs) -> ContextItem:
    base = {
        "symbol": "AAA",
        "decision_id": "d1",
        "decision_time": _DECISION,
        "source": "edgar",
        "source_id": "0001",
        "text": "8-K filed",
        "provider_published_at": _DECISION - timedelta(hours=2),
        "provider_seen_at": _DECISION - timedelta(hours=1),
        "retrieved_at": _DECISION - timedelta(minutes=50),
    }
    base.update(kwargs)
    return ContextItem(**base)


class TestAssessAdmissibility(unittest.TestCase):
    def test_clean_item_admissible(self):
        a = assess_admissibility(_item())
        self.assertTrue(a.admissible)
        self.assertEqual(a.timestamp_confidence, TIMESTAMP_CLEAN)

    def test_missing_seen_at_rejected(self):
        a = assess_admissibility(_item(provider_seen_at=None))
        self.assertFalse(a.admissible)
        self.assertEqual(a.timestamp_confidence, TIMESTAMP_REJECTED)

    def test_missing_retrieved_at_rejected(self):
        a = assess_admissibility(_item(retrieved_at=None))
        self.assertEqual(a.timestamp_confidence, TIMESTAMP_REJECTED)

    def test_retrieved_after_decision_rejected(self):
        a = assess_admissibility(_item(retrieved_at=_DECISION + timedelta(minutes=1)))
        self.assertFalse(a.admissible)
        self.assertEqual(a.timestamp_confidence, TIMESTAMP_REJECTED)
        self.assertIn("retrieved", a.reason)

    def test_seen_after_decision_rejected(self):
        a = assess_admissibility(
            _item(
                provider_seen_at=_DECISION + timedelta(minutes=1),
                retrieved_at=_DECISION,  # retrieved exactly at decision (not after)
            )
        )
        self.assertFalse(a.admissible)
        self.assertEqual(a.timestamp_confidence, TIMESTAMP_REJECTED)
        self.assertIn("seen after", a.reason)

    def test_within_safety_buffer_uncertain(self):
        # seen 2 minutes before decision, buffer 5 minutes -> within buffer
        a = assess_admissibility(
            _item(
                provider_seen_at=_DECISION - timedelta(minutes=2),
                retrieved_at=_DECISION - timedelta(minutes=1),
                provider_published_at=_DECISION - timedelta(minutes=10),
            )
        )
        self.assertFalse(a.admissible)
        self.assertEqual(a.timestamp_confidence, TIMESTAMP_UNCERTAIN)
        self.assertIn("buffer", a.reason)

    def test_published_after_seen_uncertain_backfill(self):
        a = assess_admissibility(
            _item(provider_published_at=_DECISION - timedelta(minutes=30))  # after seen (-1h)
        )
        self.assertFalse(a.admissible)
        self.assertEqual(a.timestamp_confidence, TIMESTAMP_UNCERTAIN)
        self.assertIn("backfill", a.reason)

    def test_no_published_at_is_fine(self):
        a = assess_admissibility(_item(provider_published_at=None))
        self.assertTrue(a.admissible)

    def test_custom_buffer_can_admit(self):
        # seen 2 min before; with a 60s buffer it is now outside the buffer -> admissible
        a = assess_admissibility(
            _item(
                provider_seen_at=_DECISION - timedelta(minutes=2),
                retrieved_at=_DECISION - timedelta(minutes=1),
                provider_published_at=_DECISION - timedelta(minutes=10),
            ),
            safety_buffer_seconds=60,
        )
        self.assertTrue(a.admissible)


class TestFilterAdmissible(unittest.TestCase):
    def test_filters_out_inadmissible(self):
        good = _item(symbol="GOOD")
        bad = _item(symbol="BAD", retrieved_at=None)
        result = filter_admissible([good, bad])
        self.assertEqual([it.symbol for it in result], ["GOOD"])

    def test_empty_list(self):
        self.assertEqual(filter_admissible([]), [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
