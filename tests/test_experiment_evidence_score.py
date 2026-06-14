import unittest

from experiment.evidence_score import EVIDENCE_SCORE_VERSION, evidence_score_v1

_FULL = {
    "signal_edge_score": 2.0,
    "signal_regime_score": 1.0,
    "confluence_score": 0.5,
    "liquidity_score": 0.3,
    "trend_quality_score": 0.4,
    "volatility_penalty": 0.2,
    "spread_penalty": 0.1,
    "decay_penalty": 0.05,
    "event_risk_penalty": 0.3,
}


class TestEvidenceScoreV1(unittest.TestCase):
    def test_version_tag(self):
        self.assertEqual(evidence_score_v1(_FULL).version, EVIDENCE_SCORE_VERSION)

    def test_full_score_sums_positives_minus_penalties(self):
        # (2 + 1 + 0.5 + 0.3 + 0.4) - (0.2 + 0.1 + 0.05 + 0.3) = 4.2 - 0.65 = 3.55
        r = evidence_score_v1(_FULL)
        self.assertAlmostEqual(r.score, 3.55)
        self.assertTrue(r.expectancy_present)
        self.assertEqual(r.missing, [])

    def test_penalties_reduce_score(self):
        no_pen = dict(_FULL)
        for p in ("volatility_penalty", "spread_penalty", "decay_penalty", "event_risk_penalty"):
            no_pen[p] = 0.0
        self.assertGreater(evidence_score_v1(no_pen).score, evidence_score_v1(_FULL).score)

    def test_missing_expectancy_flagged_not_scoreable(self):
        c = dict(_FULL)
        del c["signal_edge_score"]
        r = evidence_score_v1(c)
        self.assertFalse(r.expectancy_present)
        self.assertIn("signal_edge_score", r.missing)

    def test_none_treated_as_missing(self):
        c = dict(_FULL)
        c["signal_regime_score"] = None
        r = evidence_score_v1(c)
        self.assertFalse(r.expectancy_present)
        self.assertIn("signal_regime_score", r.missing)

    def test_optional_component_missing_still_scoreable(self):
        c = dict(_FULL)
        del c["confluence_score"]
        r = evidence_score_v1(c)
        self.assertTrue(r.expectancy_present)  # expectancy backbone still present
        self.assertIn("confluence_score", r.missing)
        # confluence (0.5) no longer contributes
        self.assertAlmostEqual(r.score, 3.55 - 0.5)

    def test_empty_components(self):
        r = evidence_score_v1({})
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.expectancy_present)
        self.assertEqual(len(r.missing), 9)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
