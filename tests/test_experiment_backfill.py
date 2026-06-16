"""Tests for experiment/backfill.py — forward-outcome backfill (point-in-time)."""

import unittest

from experiment.backfill import (
    _entry_index,
    backfill,
    cost_r_estimate,
    score_observation,
)


class TestEntryIndex(unittest.TestCase):
    def setUp(self):
        self.dates = ["2024-01-02", "2024-01-03", "2024-01-05"]

    def test_exact_match(self):
        self.assertEqual(_entry_index(self.dates, "2024-01-03"), 1)

    def test_first_bar_on_or_after(self):
        # decision on a non-trading day falls to the next available bar
        self.assertEqual(_entry_index(self.dates, "2024-01-04"), 2)

    def test_past_end_returns_none(self):
        self.assertIsNone(_entry_index(self.dates, "2024-01-06"))


class TestCostREstimate(unittest.TestCase):
    def test_valid(self):
        # (100 * 10/10000) / 2.0 = 0.05
        self.assertAlmostEqual(cost_r_estimate(100.0, 2.0, 10.0), 0.05)

    def test_none_when_no_price(self):
        self.assertIsNone(cost_r_estimate(None, 2.0))

    def test_none_when_no_atr(self):
        self.assertIsNone(cost_r_estimate(100.0, None))

    def test_none_when_atr_nonpositive(self):
        self.assertIsNone(cost_r_estimate(100.0, 0.0))


class TestScoreObservation(unittest.TestCase):
    def _series(self):
        dates = [f"2024-01-{d:02d}" for d in range(1, 9)]  # 8 bars
        closes = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0]
        return dates, closes

    def _obs(self, date="2024-01-01", atr=2.0, price=100.0):
        feats = {"current_price": price}
        if atr is not None:
            feats["atr"] = atr
        return {"symbol": "AAPL", "date": date, "features": feats}

    def test_closed_and_open_horizons(self):
        dates, closes = self._series()
        out = score_observation(self._obs(), dates, closes)
        o = out["outcomes"]
        self.assertAlmostEqual(o["forward_r_1d"], 0.5)  # (101-100)/2
        self.assertAlmostEqual(o["forward_r_3d"], 1.5)  # (103-100)/2
        self.assertAlmostEqual(o["forward_r_5d"], 2.5)  # (105-100)/2
        self.assertIsNone(o["forward_r_10d"])  # exit bar beyond the 8-bar series → not closed
        self.assertEqual(o["scored_horizons"], 3)
        self.assertAlmostEqual(o["cost_r_estimate"], 0.05)

    def test_missing_atr_yields_no_outcomes(self):
        dates, closes = self._series()
        out = score_observation(self._obs(atr=None), dates, closes)
        self.assertTrue(all(out["outcomes"][f"forward_r_{h}d"] is None for h in (1, 3, 5, 10)))
        self.assertEqual(out["outcomes"]["scored_horizons"], 0)

    def test_decision_date_past_series_yields_no_outcomes(self):
        dates, closes = self._series()
        out = score_observation(self._obs(date="2099-01-01"), dates, closes)
        self.assertEqual(out["outcomes"]["scored_horizons"], 0)


class TestBackfill(unittest.TestCase):
    def test_scores_with_series_skips_without(self):
        dates = [f"2024-01-{d:02d}" for d in range(1, 9)]
        closes = [100.0 + i for i in range(8)]
        obs = [
            {
                "symbol": "AAPL",
                "date": "2024-01-01",
                "features": {"atr": 2.0, "current_price": 100.0},
            },
            {"symbol": "NOPRICE", "date": "2024-01-01", "features": {"atr": 1.0}},
        ]
        out = backfill(obs, {"AAPL": (dates, closes)})
        self.assertEqual(len(out), 1)  # NOPRICE skipped
        self.assertEqual(out[0]["symbol"], "AAPL")
        self.assertAlmostEqual(out[0]["outcomes"]["forward_r_5d"], 2.5)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
