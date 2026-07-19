"""Tests for experiment/specialization.py — AI-vs-deterministic ΔR by regime (self-specialization)."""

import unittest

from experiment.specialization import (
    ai_edge_by_slice,
    build_specialization_lines,
    to_candidate,
)


def _cand(*, ai, rank, r5, regime="NEUTRAL_CHOP", cost=0.0, dtype="buy_candidate"):
    return {
        "extra": {
            "decision_type": dtype,
            "arm3_ai_selected": ai,
            "arm1_deterministic_rank": rank,
            "market_context": {"regime": regime},
        },
        "outcomes": {"forward_r_5d": r5, "cost_r_estimate": cost},
    }


class TestAiEdgeBySlice(unittest.TestCase):
    def test_delta_r_ai_minus_det(self):
        obs = [
            _cand(ai=True, rank=50, r5=2.0),  # AI-picked (not in det top-5)
            _cand(ai=False, rank=1, r5=0.5),  # det top-5, AI passed
            _cand(ai=False, rank=2, r5=0.5),  # det top-5
        ]
        edge = ai_edge_by_slice(obs, top_k=5)
        e = edge["NEUTRAL_CHOP"]
        self.assertEqual((e["n_ai"], e["n_det"]), (1, 2))
        self.assertAlmostEqual(e["ai_mean"], 2.0)
        self.assertAlmostEqual(e["det_mean"], 0.5)
        self.assertAlmostEqual(e["delta_r"], 1.5)

    def test_slice_needs_both_arms(self):
        obs = [_cand(ai=True, rank=50, r5=1.0)]  # AI arm only, no det top-K in this slice
        self.assertEqual(ai_edge_by_slice(obs, top_k=5), {})

    def test_skips_unclosed_and_untagged(self):
        obs = [
            _cand(ai=True, rank=1, r5=None),  # unclosed horizon
            {
                "extra": {"decision_type": "buy_candidate"},
                "outcomes": {"forward_r_5d": 1.0},
            },  # no arm fields
            {
                "extra": {
                    "decision_type": "held_position",
                    "arm3_ai_selected": True,
                    "arm1_deterministic_rank": 1,
                },
                "outcomes": {"forward_r_5d": 1.0},
            },  # not a buy_candidate
        ]
        self.assertEqual(ai_edge_by_slice(obs), {})

    def test_regime_missing_falls_back_to_unknown(self):
        obs = [
            {
                "extra": {
                    "decision_type": "buy_candidate",
                    "arm3_ai_selected": True,
                    "arm1_deterministic_rank": 1,
                },
                "outcomes": {"forward_r_5d": 1.0},
            },
            {
                "extra": {
                    "decision_type": "buy_candidate",
                    "arm3_ai_selected": False,
                    "arm1_deterministic_rank": 1,
                },
                "outcomes": {"forward_r_5d": 0.0},
            },
        ]
        self.assertIn("UNKNOWN", ai_edge_by_slice(obs))


class TestBuildLines(unittest.TestCase):
    def test_empty(self):
        self.assertIn("no matured", build_specialization_lines({})[0])

    def test_edge_and_noise_verdicts(self):
        edge = {
            "BULL_TREND": {
                "n_ai": 40,
                "n_det": 40,
                "ai_mean": 0.6,
                "det_mean": 0.2,
                "delta_r": 0.4,
            },
            "NEUTRAL_CHOP": {
                "n_ai": 40,
                "n_det": 40,
                "ai_mean": 0.1,
                "det_mean": 0.3,
                "delta_r": -0.2,
            },
        }
        joined = "\n".join(build_specialization_lines(edge))
        self.assertIn("BULL_TREND", joined)
        self.assertIn("EDGE: candidate to concentrate", joined)
        self.assertIn("NOISE: candidate to defer", joined)

    def test_no_verdict_below_min_n(self):
        edge = {"X": {"n_ai": 5, "n_det": 5, "ai_mean": 0.6, "det_mean": 0.1, "delta_r": 0.5}}
        joined = "\n".join(build_specialization_lines(edge))
        self.assertNotIn("EDGE", joined)  # too few samples to claim edge

    def test_no_verdict_for_small_positive_edge(self):
        # enough sample but delta between 0 and min_edge -> neither EDGE nor NOISE
        edge = {"X": {"n_ai": 40, "n_det": 40, "ai_mean": 0.35, "det_mean": 0.3, "delta_r": 0.05}}
        joined = "\n".join(build_specialization_lines(edge))
        self.assertNotIn("EDGE", joined)
        self.assertNotIn("NOISE", joined)


class TestToCandidate(unittest.TestCase):
    def test_authors_on_clearing_bar(self):
        c = to_candidate("BULL_TREND", {"n_ai": 80, "n_det": 80, "delta_r": 0.3}, "2026-07-19")
        self.assertEqual((c.id, c.source), ("specialize_ai_BULL_TREND", "specialization"))
        self.assertIn("reversible", c.action)

    def test_none_below_sample_floor(self):
        self.assertIsNone(
            to_candidate("X", {"n_ai": 10, "n_det": 80, "delta_r": 0.3}, "2026-07-19")
        )

    def test_none_below_effect_floor(self):
        self.assertIsNone(
            to_candidate(
                "X", {"n_ai": 80, "n_det": 80, "delta_r": 0.05}, "2026-07-19", min_effect=0.1
            )
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
