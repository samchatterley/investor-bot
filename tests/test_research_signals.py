"""Tests for experiment/research_signals.py — the shadow-only research-candidate signal tier."""

import json
import os
import tempfile
import unittest

from experiment.research_signals import (
    ResearchSignal,
    fires,
    load_research_signals,
    save_research_signals,
    score_research_signal,
)


def _sig(feature="rsi_14", op=">=", threshold=60.0, direction="long", created="2026-07-01"):
    return ResearchSignal(
        id="s1", feature=feature, op=op, threshold=threshold, direction=direction, created=created
    )


def _obs(rsi, fr5, date="2026-07-10"):
    return {"date": date, "features": {"rsi_14": rsi}, "outcomes": {"forward_r_5d": fr5}}


class TestFires(unittest.TestCase):
    def test_ge_and_le(self):
        self.assertTrue(fires(_sig(op=">=", threshold=60), _obs(65, 0)))
        self.assertFalse(fires(_sig(op=">=", threshold=60), _obs(55, 0)))
        self.assertTrue(fires(_sig(op="<=", threshold=40), _obs(30, 0)))
        self.assertFalse(fires(_sig(op="<=", threshold=40), _obs(50, 0)))

    def test_missing_or_nonnumeric_feature_does_not_fire(self):
        self.assertFalse(fires(_sig(feature="absent"), _obs(65, 0)))
        self.assertFalse(fires(_sig(), {"features": {"rsi_14": "high"}}))
        self.assertFalse(fires(_sig(), {"features": {"rsi_14": True}}))  # bool excluded

    def test_flat_snapshot_without_features_dict(self):
        self.assertTrue(fires(_sig(op=">=", threshold=60), {"rsi_14": 70}))


class TestScoreResearchSignal(unittest.TestCase):
    def test_long_excess_vs_field(self):
        obs = [_obs(70, 2.0), _obs(80, 4.0), _obs(30, 0.0), _obs(20, 0.0)]  # field mean 1.5
        n, effect = score_research_signal(_sig(op=">=", threshold=60), obs)
        self.assertEqual(n, 2)  # fired: rsi 70,80
        self.assertAlmostEqual(effect, 3.0 - 1.5)  # fired mean 3.0 - field 1.5

    def test_short_direction_negates(self):
        obs = [_obs(70, -2.0), _obs(30, 2.0)]  # field mean 0; fired(rsi>=60) mean -2
        n, effect = score_research_signal(_sig(op=">=", threshold=60, direction="short"), obs)
        self.assertAlmostEqual(effect, 2.0)  # -(−2 − 0) = +2 (short profits when name falls)

    def test_only_scores_on_or_after_created(self):
        obs = [
            _obs(70, 5.0, date="2026-06-01"),
            _obs(70, 1.0, date="2026-07-10"),
        ]  # created 2026-07-01
        n, _ = score_research_signal(_sig(created="2026-07-01"), obs)
        self.assertEqual(n, 1)  # the 06-01 obs is pre-registration, excluded

    def test_no_field_returns_none(self):
        self.assertEqual(score_research_signal(_sig(), [_obs(70, None)]), (0, None))

    def test_nothing_fires_returns_none(self):
        self.assertEqual(score_research_signal(_sig(threshold=99), [_obs(10, 1.0)]), (0, None))


class TestPersistence(unittest.TestCase):
    def _p(self):
        d = tempfile.mkdtemp()
        self.addCleanup(__import__("shutil").rmtree, d)
        return os.path.join(d, "sub", "research_signals.json")

    def test_round_trip(self):
        p = self._p()
        save_research_signals([_sig(), ResearchSignal("s2", "bb_pct", "<=", 0.2, "long")], p)
        loaded = load_research_signals(p)
        self.assertEqual([s.id for s in loaded], ["s1", "s2"])

    def test_missing_file_returns_empty(self):
        self.assertEqual(load_research_signals("/no/such/rs.json"), [])

    def test_malformed_returns_empty(self):
        p = self._p()
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write("{nope")
        self.assertEqual(load_research_signals(p), [])

    def test_missing_signals_key_returns_empty(self):
        p = self._p()
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            json.dump({"updated": "x"}, f)
        self.assertEqual(load_research_signals(p), [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
