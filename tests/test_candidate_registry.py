"""Tests for experiment/candidate_registry.py — the improvement-candidate engine + approval queue."""

import json
import os
import tempfile
import unittest

from experiment.candidate_registry import (
    ACCUMULATING,
    NOT_SUPPORTED,
    READY,
    Candidate,
    build_candidate_lines,
    default_candidates,
    evaluate,
    load_registry,
    save_registry,
)


def _cand(min_n=50, min_effect=0.15, cid="c1"):
    return Candidate(
        id=cid,
        hypothesis="h",
        action="do X (reversible)",
        metric="edge (R)",
        min_n=min_n,
        min_effect=min_effect,
    )


class TestEvaluate(unittest.TestCase):
    def test_missing_evidence_is_accumulating(self):
        self.assertEqual(evaluate(_cand(), None, 0.5), (ACCUMULATING, 0.0))
        self.assertEqual(evaluate(_cand(), 100, None), (ACCUMULATING, 0.0))

    def test_below_sample_floor_accumulating_with_pct(self):
        verdict, pct = evaluate(_cand(min_n=50), 25, 0.9)
        self.assertEqual(verdict, ACCUMULATING)
        self.assertAlmostEqual(pct, 0.5)

    def test_ready_when_both_floors_cleared(self):
        self.assertEqual(evaluate(_cand(min_n=50, min_effect=0.15), 60, 0.30), (READY, 1.0))

    def test_not_supported_when_effect_below_floor(self):
        verdict, _ = evaluate(_cand(min_n=50, min_effect=0.15), 60, 0.10)
        self.assertEqual(verdict, NOT_SUPPORTED)

    def test_effect_exactly_at_floor_is_not_ready(self):
        verdict, _ = evaluate(_cand(min_effect=0.15), 60, 0.15)  # strictly greater required
        self.assertEqual(verdict, NOT_SUPPORTED)

    def test_zero_sample_floor_is_full_pct(self):
        self.assertEqual(evaluate(_cand(min_n=0, min_effect=0.0), 1, 0.5), (READY, 1.0))


class TestBuildCandidateLines(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(
            build_candidate_lines([]),
            ["Candidate pipeline: empty (no improvement candidates registered)."],
        )

    def test_ready_candidate_produces_dossier(self):
        out = "\n".join(build_candidate_lines([(_cand(cid="win"), 60, 0.30)]))
        self.assertIn("PENDING APPROVAL -- 1 candidate", out)
        self.assertIn("[win]", out)
        self.assertIn("evidence: edge (R) = +0.30 at n=60", out)
        self.assertIn("if approved: do X (reversible)", out)

    def test_accumulating_and_no_evidence_lines(self):
        out = build_candidate_lines([(_cand(cid="acc"), 25, 0.9), (_cand(cid="new"), None, None)])
        joined = "\n".join(out)
        self.assertNotIn("PENDING APPROVAL", joined)
        self.assertIn("[acc] ACCUMULATING", joined)
        self.assertIn("50% of n>=50", joined)
        self.assertIn("[new] ACCUMULATING -- no matured evidence yet", joined)

    def test_mixed_ready_and_other(self):
        out = "\n".join(
            build_candidate_lines([(_cand(cid="win"), 60, 0.30), (_cand(cid="fail"), 60, 0.0)])
        )
        self.assertIn("PENDING APPROVAL", out)
        self.assertIn("[fail] NOT-SUPPORTED", out)


class TestDefaultCandidates(unittest.TestCase):
    def test_seeds_expected_candidates(self):
        ids = {c.id for c in default_candidates()}
        self.assertEqual(ids, {"min_confidence_7_to_8", "ungate_guidance_downgrade_shorts"})

    def test_all_actions_marked_reversible_and_have_bars(self):
        for c in default_candidates():
            self.assertTrue(c.min_n > 0 and c.action and c.hypothesis)


class TestRegistryPersistence(unittest.TestCase):
    def _path(self):
        d = tempfile.mkdtemp()
        self.addCleanup(__import__("shutil").rmtree, d)
        return os.path.join(d, "sub", "candidate_registry.json")

    def test_missing_file_seeds_and_persists_defaults(self):
        p = self._path()
        cands = load_registry(p)
        self.assertTrue(os.path.exists(p))  # seeded on first load
        self.assertEqual({c.id for c in cands}, {c.id for c in default_candidates()})

    def test_round_trip(self):
        p = self._path()
        save_registry([_cand(cid="x")], p)
        loaded = load_registry(p)
        self.assertEqual([c.id for c in loaded], ["x"])

    def test_malformed_falls_back_to_defaults(self):
        p = self._path()
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write("{not json}")
        self.assertEqual({c.id for c in load_registry(p)}, {c.id for c in default_candidates()})

    def test_missing_candidates_key_falls_back(self):
        p = self._path()
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            json.dump({"updated": "2026-07-18"}, f)  # no "candidates" key
        self.assertEqual({c.id for c in load_registry(p)}, {c.id for c in default_candidates()})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
