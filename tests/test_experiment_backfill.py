"""Tests for experiment/backfill.py — forward-outcome backfill (point-in-time, ATR from history)."""

import unittest

from experiment.backfill import (
    _atr_at,
    _entry_index,
    backfill,
    cost_r_estimate,
    score_observation,
)


def _ohlc(n: int = 20):
    """n bars with a constant true range of 2.0 → ATR(14) == 2.0, closes rising by 1/bar.

    closes[i]=100+i, highs=close+1, lows=close-1 → TR = max(2, 2, 0) = 2 every bar.
    """
    dates = [f"2024-01-{i + 1:02d}" for i in range(n)]
    closes = [100.0 + i for i in range(n)]
    highs = [c + 1.0 for c in closes]
    lows = [c - 1.0 for c in closes]
    return dates, highs, lows, closes


class TestEntryIndex(unittest.TestCase):
    def setUp(self):
        self.dates = ["2024-01-02", "2024-01-03", "2024-01-05"]

    def test_exact_match(self):
        self.assertEqual(_entry_index(self.dates, "2024-01-03"), 1)

    def test_first_bar_on_or_after(self):
        self.assertEqual(_entry_index(self.dates, "2024-01-04"), 2)

    def test_past_end_returns_none(self):
        self.assertIsNone(_entry_index(self.dates, "2024-01-06"))


class TestAtrAt(unittest.TestCase):
    def test_constant_true_range(self):
        _d, highs, lows, closes = _ohlc()
        self.assertAlmostEqual(_atr_at(highs, lows, closes, 14), 2.0)

    def test_none_when_insufficient_history(self):
        _d, highs, lows, closes = _ohlc()
        self.assertIsNone(_atr_at(highs, lows, closes, 5))  # idx < period(14)

    def test_none_when_idx_none(self):
        _d, highs, lows, closes = _ohlc()
        self.assertIsNone(_atr_at(highs, lows, closes, None))

    def test_none_when_zero_range(self):
        flat = [100.0] * 20
        self.assertIsNone(_atr_at(flat, flat, flat, 14))  # TR=0 → atr 0 → None


class TestCostREstimate(unittest.TestCase):
    def test_valid(self):
        self.assertAlmostEqual(cost_r_estimate(100.0, 2.0, 10.0), 0.05)  # (100*10/10000)/2

    def test_none_when_no_price(self):
        self.assertIsNone(cost_r_estimate(None, 2.0))

    def test_none_when_no_atr(self):
        self.assertIsNone(cost_r_estimate(100.0, None))

    def test_none_when_atr_nonpositive(self):
        self.assertIsNone(cost_r_estimate(100.0, 0.0))


class TestScoreObservation(unittest.TestCase):
    def _obs(self, date="2024-01-15", price=100.0):
        return {"symbol": "AAPL", "date": date, "features": {"current_price": price}}

    def test_closed_and_open_horizons(self):
        dates, highs, lows, closes = _ohlc()  # ATR=2.0; decision at idx 14 (close 114)
        out = score_observation(self._obs(), dates, highs, lows, closes)
        o = out["outcomes"]
        self.assertAlmostEqual(o["forward_r_1d"], 0.5)  # (115-114)/2
        self.assertAlmostEqual(o["forward_r_3d"], 1.5)  # (117-114)/2
        self.assertAlmostEqual(o["forward_r_5d"], 2.5)  # (119-114)/2
        self.assertIsNone(o["forward_r_10d"])  # exit bar beyond the 20-bar series
        self.assertEqual(o["scored_horizons"], 3)
        self.assertAlmostEqual(o["cost_r_estimate"], 0.05)

    def test_insufficient_history_yields_no_outcomes(self):
        dates, highs, lows, closes = _ohlc()
        out = score_observation(self._obs(date="2024-01-04"), dates, highs, lows, closes)  # idx 3
        self.assertTrue(all(out["outcomes"][f"forward_r_{h}d"] is None for h in (1, 3, 5, 10)))
        self.assertEqual(out["outcomes"]["scored_horizons"], 0)

    def test_decision_date_past_series_yields_no_outcomes(self):
        dates, highs, lows, closes = _ohlc()
        out = score_observation(self._obs(date="2099-01-01"), dates, highs, lows, closes)
        self.assertEqual(out["outcomes"]["scored_horizons"], 0)


class TestBackfill(unittest.TestCase):
    def test_scores_with_series_skips_without(self):
        dates, highs, lows, closes = _ohlc()
        obs = [
            {"symbol": "AAPL", "date": "2024-01-15", "features": {"current_price": 100.0}},
            {"symbol": "NOPRICE", "date": "2024-01-15", "features": {}},
        ]
        out = backfill(obs, {"AAPL": (dates, highs, lows, closes)})
        self.assertEqual(len(out), 1)  # NOPRICE skipped
        self.assertEqual(out[0]["symbol"], "AAPL")
        self.assertAlmostEqual(out[0]["outcomes"]["forward_r_5d"], 2.5)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
