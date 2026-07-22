"""Tests for experiment/counterfactual.py — paired hold-horizon counterfactual replay."""

import unittest

from experiment.counterfactual import (
    _net_r,
    _paired_diffs,
    _ttest_1samp,
    author_online,
    build_counterfactual_lines,
    horizon_counterfactuals,
    to_candidate,
)
from experiment.dof_ledger import new_state


def _obs(r3=None, r5=None, cost=0.0):
    return {"outcomes": {"forward_r_3d": r3, "forward_r_5d": r5, "cost_r_estimate": cost}}


class TestNetR(unittest.TestCase):
    def test_gross_minus_cost(self):
        self.assertAlmostEqual(
            _net_r({"outcomes": {"forward_r_5d": 1.0, "cost_r_estimate": 0.1}}, 5), 0.9
        )

    def test_unclosed_horizon_is_none(self):
        self.assertIsNone(_net_r({"outcomes": {"forward_r_5d": None}}, 5))

    def test_missing_cost_defaults_zero(self):
        self.assertAlmostEqual(_net_r({"outcomes": {"forward_r_5d": 1.0}}, 5), 1.0)

    def test_none_cost_defaults_zero(self):
        self.assertAlmostEqual(
            _net_r({"outcomes": {"forward_r_5d": 1.0, "cost_r_estimate": None}}, 5), 1.0
        )


class TestPairedDiffs(unittest.TestCase):
    def test_pairs_only_when_both_closed(self):
        obs = [_obs(2.0, 1.0), _obs(3.0, None), _obs(None, 1.0)]  # only the first has both
        self.assertEqual(_paired_diffs(obs, 3, 5), [1.0])  # 2.0 - 1.0


class TestTTest(unittest.TestCase):
    def test_significant(self):
        m, p = _ttest_1samp([2.0, 4.0])
        self.assertAlmostEqual(m, 3.0)
        self.assertLess(p, 0.05)

    def test_zero_se_nonzero_mean_is_significant(self):
        self.assertEqual(_ttest_1samp([0.05, 0.05]), (0.05, 0.0))

    def test_zero_se_zero_mean_is_null(self):
        self.assertEqual(_ttest_1samp([0.0, 0.0]), (0.0, 1.0))


class TestHorizonCounterfactuals(unittest.TestCase):
    def test_paired_uplift_and_best(self):
        obs = [_obs(2.0, 1.0) for _ in range(10)]  # 3d beats 5d by +1.0 on the same decisions
        r = horizon_counterfactuals(obs)
        self.assertEqual(r["per_horizon"][3], {"n": 10, "uplift": 1.0})
        self.assertEqual(r["best"], 3)
        self.assertAlmostEqual(r["uplift"], 1.0)

    def test_empty_when_baseline_unmatured(self):
        obs = [_obs(2.0, None) for _ in range(10)]  # no 5d -> nothing pairs
        r = horizon_counterfactuals(obs)
        self.assertEqual((r["per_horizon"], r["best"], r["uplift"]), ({}, None, None))


class TestBuildLines(unittest.TestCase):
    def test_empty(self):
        lines = build_counterfactual_lines(
            {"per_horizon": {}, "baseline": 5, "best": None, "uplift": None}
        )
        self.assertIn("no decisions with both a matured baseline", lines[0])

    def test_flags_a_winning_horizon(self):
        r = {"per_horizon": {3: {"n": 80, "uplift": 0.4}}, "baseline": 5, "best": 3, "uplift": 0.4}
        joined = "\n".join(build_counterfactual_lines(r))
        self.assertIn("3d +0.40R", joined)
        self.assertIn("candidate: a 3d hold beats the 5d baseline", joined)

    def test_no_candidate_line_below_min_uplift(self):
        r = {
            "per_horizon": {3: {"n": 80, "uplift": 0.05}},
            "baseline": 5,
            "best": 3,
            "uplift": 0.05,
        }
        self.assertEqual(len(build_counterfactual_lines(r)), 1)


class TestToCandidate(unittest.TestCase):
    def test_authors_on_win(self):
        r = {"per_horizon": {3: {"n": 80, "uplift": 0.4}}, "baseline": 5, "best": 3, "uplift": 0.4}
        c = to_candidate(r, "2026-07-22")
        self.assertEqual((c.id, c.source), ("hold_horizon_5_to_3", "counterfactual"))
        self.assertIn("paired", c.hypothesis)

    def test_none_when_no_best(self):
        self.assertIsNone(
            to_candidate(
                {"per_horizon": {}, "baseline": 5, "best": None, "uplift": None}, "2026-07-22"
            )
        )

    def test_none_below_floor(self):
        r = {
            "per_horizon": {3: {"n": 80, "uplift": 0.05}},
            "baseline": 5,
            "best": 3,
            "uplift": 0.05,
        }
        self.assertIsNone(to_candidate(r, "2026-07-22", min_effect=0.1))


class TestAuthorOnline(unittest.TestCase):
    def test_authors_when_rejected_and_uplift_clears(self):
        obs = [_obs(1.0, 0.0) for _ in range(50)]  # 3d beats 5d by +1.0, paired, zero variance
        ledger, cands = author_online(obs, new_state(), "2026-07-22")
        self.assertEqual([c.id for c in cands], ["hold_horizon_5_to_3"])
        self.assertEqual(ledger.n_tests, 1)

    def test_empty_when_nothing_pairs(self):
        obs = [_obs(None, 1.0) for _ in range(10)]  # 5d only, no alternative horizon closed
        ledger, cands = author_online(obs, new_state(), "2026-07-22")
        self.assertEqual(cands, [])
        self.assertEqual(ledger.n_tests, 0)

    def test_charged_but_not_rejected(self):
        obs = [_obs(v, 0.0) for v in [10, -9] * 10]  # high-variance diffs -> weak paired signal
        ledger, cands = author_online(obs, new_state(), "2026-07-22")
        self.assertEqual(cands, [])
        self.assertEqual((ledger.n_tests, ledger.n_rejections), (1, 0))

    def test_rejected_but_uplift_below_floor(self):
        obs = [
            _obs(0.05, 0.0) for _ in range(50)
        ]  # rejected (perfect separation) but +0.05R < floor
        ledger, cands = author_online(obs, new_state(), "2026-07-22")
        self.assertEqual(cands, [])
        self.assertEqual(ledger.n_rejections, 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
