"""Tests for experiment/counterfactual.py — hold-horizon counterfactual replay."""

import unittest

from experiment.counterfactual import (
    _net_r,
    author_online,
    build_counterfactual_lines,
    horizon_counterfactuals,
    to_candidate,
)
from experiment.dof_ledger import new_state


def _obs(**outcomes):
    return {"outcomes": outcomes}


class TestNetR(unittest.TestCase):
    def test_gross_minus_cost(self):
        self.assertAlmostEqual(_net_r(_obs(forward_r_5d=1.0, cost_r_estimate=0.1), 5), 0.9)

    def test_unclosed_horizon_is_none(self):
        self.assertIsNone(_net_r(_obs(forward_r_5d=None, cost_r_estimate=0.1), 5))

    def test_missing_cost_defaults_zero(self):
        self.assertAlmostEqual(_net_r(_obs(forward_r_5d=1.0), 5), 1.0)

    def test_none_cost_defaults_zero(self):
        self.assertAlmostEqual(_net_r(_obs(forward_r_5d=1.0, cost_r_estimate=None), 5), 1.0)


class TestHorizonCounterfactuals(unittest.TestCase):
    def test_best_and_uplift(self):
        obs = [
            _obs(forward_r_3d=2.0, forward_r_5d=1.0, cost_r_estimate=0.0),
            _obs(forward_r_3d=2.0, forward_r_5d=1.0, cost_r_estimate=0.0),
        ]
        r = horizon_counterfactuals(obs, horizons=(3, 5), baseline=5)
        self.assertEqual(r["per_horizon"][3], {"n": 2, "mean_net_r": 2.0})
        self.assertEqual(r["best"], 3)
        self.assertAlmostEqual(r["uplift"], 1.0)  # 2.0 (3d) - 1.0 (5d)

    def test_unclosed_horizon_excluded(self):
        obs = [_obs(forward_r_5d=1.0), _obs(forward_r_5d=None)]  # second not closed
        r = horizon_counterfactuals(obs, horizons=(5,), baseline=5)
        self.assertEqual(r["per_horizon"][5]["n"], 1)

    def test_uplift_none_when_baseline_absent(self):
        obs = [_obs(forward_r_3d=2.0)]  # no 5d sample
        r = horizon_counterfactuals(obs, horizons=(3, 5), baseline=5)
        self.assertEqual(r["best"], 3)
        self.assertIsNone(r["uplift"])

    def test_empty(self):
        r = horizon_counterfactuals([_obs(forward_r_5d=None)], horizons=(5,))
        self.assertEqual((r["per_horizon"], r["best"], r["uplift"]), ({}, None, None))


class TestBuildLines(unittest.TestCase):
    def test_empty(self):
        lines = build_counterfactual_lines(
            {"per_horizon": {}, "baseline": 5, "best": None, "uplift": None}
        )
        self.assertIn("no closed-horizon sample yet", lines[0])

    def test_flags_a_winning_horizon(self):
        r = {
            "per_horizon": {3: {"n": 80, "mean_net_r": 0.6}, 5: {"n": 80, "mean_net_r": 0.2}},
            "baseline": 5,
            "best": 3,
            "uplift": 0.4,
        }
        joined = "\n".join(build_counterfactual_lines(r))
        self.assertIn("3d=+0.60R", joined)
        self.assertIn("candidate: 3d hold beats the 5d baseline", joined)

    def test_no_candidate_line_below_min_uplift(self):
        r = {
            "per_horizon": {3: {"n": 80, "mean_net_r": 0.25}, 5: {"n": 80, "mean_net_r": 0.2}},
            "baseline": 5,
            "best": 3,
            "uplift": 0.05,  # below default min_uplift 0.1
        }
        self.assertEqual(len(build_counterfactual_lines(r)), 1)  # table only, no candidate line


class TestToCandidate(unittest.TestCase):
    def _win(self):
        return {
            "per_horizon": {3: {"n": 80, "mean_net_r": 0.6}, 5: {"n": 80, "mean_net_r": 0.2}},
            "baseline": 5,
            "best": 3,
            "uplift": 0.4,
        }

    def test_authors_candidate_on_win(self):
        c = to_candidate(self._win(), "2026-07-19")
        self.assertEqual((c.id, c.source), ("hold_horizon_5_to_3", "counterfactual"))
        self.assertIn("reversible", c.action)

    def test_none_when_best_is_baseline(self):
        r = {**self._win(), "best": 5, "uplift": 0.0}
        self.assertIsNone(to_candidate(r, "2026-07-19"))

    def test_none_when_uplift_below_floor(self):
        r = {**self._win(), "uplift": 0.05}
        self.assertIsNone(to_candidate(r, "2026-07-19", min_effect=0.1))

    def test_none_when_no_sample(self):
        r = {"per_horizon": {}, "baseline": 5, "best": None, "uplift": None}
        self.assertIsNone(to_candidate(r, "2026-07-19"))


class TestAuthorOnline(unittest.TestCase):
    def test_authors_when_ledger_rejects_and_uplift_clears(self):
        obs = [_obs(forward_r_3d=1.0, forward_r_5d=0.0, cost_r_estimate=0.0) for _ in range(50)]
        ledger, cands = author_online(obs, new_state(), "2026-07-19", horizons=(3, 5), baseline=5)
        self.assertEqual([c.id for c in cands], ["hold_horizon_5_to_3"])
        self.assertEqual(ledger.n_tests, 1)

    def test_empty_when_baseline_absent(self):
        obs = [_obs(forward_r_3d=1.0) for _ in range(10)]  # no 5d sample
        ledger, cands = author_online(obs, new_state(), "2026-07-19", horizons=(3, 5), baseline=5)
        self.assertEqual(cands, [])
        self.assertEqual(ledger.n_tests, 0)  # nothing charged

    def test_empty_when_best_is_baseline(self):
        obs = [_obs(forward_r_3d=0.0, forward_r_5d=1.0, cost_r_estimate=0.0) for _ in range(50)]
        _ledger, cands = author_online(obs, new_state(), "2026-07-19", horizons=(3, 5), baseline=5)
        self.assertEqual(cands, [])

    def test_charged_but_not_rejected(self):
        obs = [_obs(forward_r_3d=v, forward_r_5d=0.0, cost_r_estimate=0.0) for v in [10, -9] * 10]
        ledger, cands = author_online(obs, new_state(), "2026-07-19", horizons=(3, 5), baseline=5)
        self.assertEqual(cands, [])
        self.assertEqual((ledger.n_tests, ledger.n_rejections), (1, 0))  # a look, no discovery

    def test_rejected_but_uplift_below_floor(self):
        obs = [_obs(forward_r_3d=0.05, forward_r_5d=0.0, cost_r_estimate=0.0) for _ in range(50)]
        ledger, cands = author_online(obs, new_state(), "2026-07-19", horizons=(3, 5), baseline=5)
        self.assertEqual(cands, [])  # rejected (perfect separation) but +0.05R < 0.1 floor
        self.assertEqual(ledger.n_rejections, 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
