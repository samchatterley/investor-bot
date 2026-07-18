"""Tests for experiment/candidate_miner.py — autonomous, multiple-testing-corrected edge mining."""

import unittest

from experiment.candidate_miner import (
    _holm,
    _mean_var,
    _pct,
    _welch_p,
    mine_edges_online,
    mine_feature_edges,
    to_candidate,
    to_research_signal,
)
from experiment.dof_ledger import new_state


def _obs(rsi, fr5):
    return {"features": {"rsi_14": rsi}, "outcomes": {"forward_r_5d": fr5}}


class TestHelpers(unittest.TestCase):
    def test_pct_single_value(self):
        self.assertEqual(_pct([5.0], 0.8), 5.0)

    def test_pct_interpolates(self):
        self.assertAlmostEqual(_pct([0.0, 10.0], 0.5), 5.0)

    def test_mean_var_single(self):
        self.assertEqual(_mean_var([3.0]), (3.0, 0.0))

    def test_mean_var_multiple(self):
        m, v = _mean_var([1.0, 3.0])
        self.assertEqual((m, v), (2.0, 2.0))  # sample variance

    def test_welch_p_zero_se_is_one(self):
        diff, p = _welch_p([1.0, 1.0], [1.0, 1.0])
        self.assertEqual((diff, p), (0.0, 1.0))

    def test_welch_p_significant_difference(self):
        diff, p = _welch_p([5.0] * 20, [0.0] * 20)
        self.assertEqual(diff, 5.0)
        self.assertLess(p, 0.01)

    def test_holm_monotone_and_scaled(self):
        adj = _holm([0.01, 0.04])
        self.assertAlmostEqual(adj[0], 0.02)  # smallest * m(2)
        self.assertGreaterEqual(adj[1], adj[0])  # non-decreasing


class TestMineFeatureEdges(unittest.TestCase):
    def test_surfaces_a_real_top_quintile_edge(self):
        # rsi 0..49; top-quintile (rsi>=~39) forward_r +3, rest 0 -> a strong long edge
        obs = [_obs(i, 3.0 if i >= 40 else 0.0) for i in range(50)]
        edges = mine_feature_edges(
            obs, features=("rsi_14",), min_n=5, min_abs_excess=0.5, alpha=0.05
        )
        self.assertTrue(
            any(e.feature == "rsi_14" and e.op == ">=" and e.direction == "long" for e in edges)
        )

    def test_no_edge_when_no_relationship(self):
        obs = [_obs(i, 0.0) for i in range(50)]  # forward_r flat -> zero excess
        self.assertEqual(mine_feature_edges(obs, features=("rsi_14",), min_n=5), [])

    def test_insufficient_data_skipped(self):
        obs = [_obs(i, 1.0) for i in range(8)]  # < 2*min_n
        self.assertEqual(mine_feature_edges(obs, features=("rsi_14",), min_n=5), [])

    def test_quantile_subset_below_min_n_skipped(self):
        # 12 pairs clears 2*min_n(10), but the top/bottom quintile subsets (~2) are < min_n(5)
        obs = [_obs(i, 3.0 if i >= 10 else 0.0) for i in range(12)]
        self.assertEqual(mine_feature_edges(obs, features=("rsi_14",), min_n=5), [])

    def test_welch_p_perfect_separation_is_significant(self):
        diff, p = _welch_p([5.0, 5.0], [0.0, 0.0])  # zero variance, different means
        self.assertEqual((diff, p), (5.0, 0.0))

    def test_ignores_unclosed_horizon(self):
        obs = [_obs(i, None) for i in range(50)]  # no closed forward_r
        self.assertEqual(mine_feature_edges(obs, features=("rsi_14",), min_n=5), [])

    def test_below_effect_floor_filtered(self):
        # a tiny but "significant" difference is dropped by the effect floor
        obs = [_obs(i, 0.10 if i >= 40 else 0.0) for i in range(50)]
        self.assertEqual(
            mine_feature_edges(obs, features=("rsi_14",), min_n=5, min_abs_excess=0.5), []
        )


class TestAuthoring(unittest.TestCase):
    def test_to_research_signal(self):
        obs = [_obs(i, 3.0 if i >= 40 else 0.0) for i in range(50)]
        edge = mine_feature_edges(obs, features=("rsi_14",), min_n=5, min_abs_excess=0.5)[0]
        sig = to_research_signal(edge, created="2026-07-18")
        self.assertEqual((sig.feature, sig.source, sig.created), ("rsi_14", "mined", "2026-07-18"))
        self.assertTrue(sig.id.startswith("mined_rsi_14"))

    def test_to_candidate_has_bar_and_reversible_action(self):
        obs = [_obs(i, 3.0 if i >= 40 else 0.0) for i in range(50)]
        edge = mine_feature_edges(obs, features=("rsi_14",), min_n=5, min_abs_excess=0.5)[0]
        cand = to_candidate(edge, created="2026-07-18", min_n=60, min_effect=0.15)
        self.assertEqual(cand.source, "mined")
        self.assertEqual((cand.min_n, cand.min_effect), (60, 0.15))
        self.assertIn("Promote research signal", cand.action)


class TestMineEdgesOnline(unittest.TestCase):
    def test_surfaces_edge_and_charges_every_look(self):
        # strong top-quintile long edge; both quantile splits are charged against the ledger
        obs = [_obs(i, 3.0 if i >= 40 else 0.0) for i in range(50)]
        ledger, edges = mine_edges_online(
            obs, new_state(), features=("rsi_14",), min_n=5, min_abs_excess=0.5
        )
        self.assertEqual(ledger.n_tests, 2)  # >= and <= splits both counted as looks
        self.assertGreater(ledger.wealth, 0.05)  # a discovery refunded budget
        self.assertTrue(any(e.op == ">=" and e.direction == "long" for e in edges))

    def test_empty_tests_returns_ledger_unchanged(self):
        led = new_state()
        obs = [_obs(i, 1.0) for i in range(8)]  # < 2*min_n -> no looks collected
        out_ledger, edges = mine_edges_online(obs, led, features=("rsi_14",), min_n=5)
        self.assertEqual(edges, [])
        self.assertIs(out_ledger, led)  # short-circuit: budget untouched

    def test_effect_floor_filters_significant_but_tiny(self):
        # a highly-significant but tiny-excess split: the ledger rejects it, the floor drops it
        obs = [_obs(i, 0.10 if i >= 40 else 0.0) for i in range(50)]
        ledger, edges = mine_edges_online(
            obs, new_state(), features=("rsi_14",), min_n=5, min_abs_excess=0.5
        )
        self.assertEqual(edges, [])
        self.assertEqual(ledger.n_tests, 2)  # still charged as looks

    def test_null_data_rejects_nothing(self):
        obs = [_obs(i, 0.0) for i in range(50)]  # flat forward_r -> p=1, no rejections
        ledger, edges = mine_edges_online(obs, new_state(), features=("rsi_14",), min_n=5)
        self.assertEqual(edges, [])
        self.assertEqual((ledger.n_tests, ledger.n_rejections), (2, 0))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
