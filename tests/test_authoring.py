"""Tests for experiment/authoring.py — the single weekly candidate-authoring orchestration."""

import unittest
from unittest.mock import patch

from experiment.authoring import run_authoring
from experiment.candidate_miner import MinedEdge
from experiment.candidate_registry import load_registry
from experiment.dof_ledger import new_state
from experiment.research_signals import load_research_signals


def _edge():
    return MinedEdge(
        feature="rsi_14",
        op=">=",
        threshold=70.0,
        direction="long",
        n=50,
        excess=0.5,
        p_value=0.001,
        p_corrected=0.01,
    )


class TestRunAuthoring(unittest.TestCase):
    def test_registers_mined_candidate_and_research_signal(self):
        # mock the miner to yield one edge; capabilities run for real on empty obs (author nothing)
        with patch("experiment.authoring.mine_edges_online", return_value=(new_state(), [_edge()])):
            result = run_authoring([], created="2026-07-22")
        self.assertEqual(result["registered"], 1)
        self.assertIn("mined_rsi_14_ge", [c.id for c in load_registry()])
        self.assertIn("mined_rsi_14_ge", [s.id for s in load_research_signals()])

    def test_idempotent_second_run_registers_nothing(self):
        with patch("experiment.authoring.mine_edges_online", return_value=(new_state(), [_edge()])):
            run_authoring([], created="2026-07-22")
            again = run_authoring([], created="2026-07-22")  # same edge, already registered
        self.assertEqual(again["registered"], 0)

    def test_defaults_load_observations_and_date(self):
        # observations=None -> loads (mocked empty); created=None -> today; nothing to author
        with patch("experiment.authoring.load_scored_observations", return_value=[]):
            result = run_authoring()
        self.assertEqual(result["registered"], 0)
        self.assertEqual(result["research_signals"], 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
