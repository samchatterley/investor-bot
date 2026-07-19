"""Tests for experiment/case_memory.py — instance-based case-memory + held-out falsification."""

import unittest

from experiment.case_memory import (
    Case,
    _feature_scales,
    _situation,
    build_case_memory_lines,
    evaluate_case_memory,
    neighbor_outcome,
    retrieve_neighbors,
    to_candidate,
)

_F = ("rsi_14",)
_SCALES = {"rsi_14": 1.0}


def _o(rsi, r5, date, cost=0.0):
    return {
        "features": {"rsi_14": rsi},
        "outcomes": {"forward_r_5d": r5, "cost_r_estimate": cost},
        "date": date,
    }


def _case(rsi, outcome, date="2026-01-01"):
    return Case({"rsi_14": rsi}, outcome, date)


class TestSituation(unittest.TestCase):
    def test_full_vector(self):
        self.assertEqual(_situation({"features": {"rsi_14": 30}}, _F), {"rsi_14": 30.0})

    def test_missing_feature_is_none(self):
        self.assertIsNone(_situation({"features": {}}, _F))

    def test_non_numeric_and_bool_are_none(self):
        self.assertIsNone(_situation({"features": {"rsi_14": "x"}}, _F))
        self.assertIsNone(_situation({"features": {"rsi_14": True}}, _F))


class TestScales(unittest.TestCase):
    def test_std_over_cases(self):
        scales = _feature_scales([_case(0.0, 0), _case(2.0, 0)], _F)
        self.assertAlmostEqual(scales["rsi_14"], 1.0)  # population std of {0,2} = 1

    def test_zero_variance_is_floored(self):
        scales = _feature_scales([_case(5.0, 0), _case(5.0, 0)], _F)
        self.assertEqual(scales["rsi_14"], 1e-9)


class TestRetrieval(unittest.TestCase):
    def test_k_nearest(self):
        cases = [_case(10, 0), _case(50, 0), _case(90, 0)]
        nb = retrieve_neighbors({"rsi_14": 45}, cases, k=2, scales=_SCALES, features=_F)
        self.assertEqual(sorted(c.situation["rsi_14"] for c in nb), [10.0, 50.0])

    def test_neighbor_outcome_is_mean(self):
        cases = [_case(48, 1.0), _case(52, 3.0), _case(90, -5.0)]
        out = neighbor_outcome({"rsi_14": 50}, cases, k=2, scales=_SCALES, features=_F)
        self.assertAlmostEqual(out, 2.0)  # mean of the two nearest (1.0, 3.0)

    def test_neighbor_outcome_none_when_no_cases(self):
        self.assertIsNone(neighbor_outcome({"rsi_14": 50}, [], k=2, scales=_SCALES, features=_F))


class TestEvaluate(unittest.TestCase):
    def _split_data(self):
        # 4 cases (earlier): high rsi -> +1, low rsi -> -1; 2 held-out (later)
        return [
            _o(80, 1.0, "2026-01-01"),
            _o(85, 1.0, "2026-01-02"),
            _o(10, -1.0, "2026-01-03"),
            _o(15, -1.0, "2026-01-04"),
            _o(82, 2.0, "2026-01-05"),  # high -> positive neighbours -> actual +2
            _o(12, -2.0, "2026-01-06"),  # low  -> negative neighbours -> actual -2
        ]

    def test_positive_out_of_sample_edge(self):
        r = evaluate_case_memory(self._split_data(), k=1, split_frac=0.7, features=_F)
        self.assertEqual((r["n_cases"], r["n_test"], r["n_pos"], r["n_neg"]), (4, 2, 1, 1))
        self.assertAlmostEqual(r["edge"], 4.0)  # +2 (pos-neighbour) - (-2) (neg-neighbour)
        self.assertTrue(r["helps"])

    def test_helps_false_below_min_edge(self):
        r = evaluate_case_memory(
            self._split_data(), k=1, split_frac=0.7, features=_F, min_edge=10.0
        )
        self.assertIsNotNone(r["edge"])
        self.assertFalse(r["helps"])

    def test_edge_none_when_one_group_empty(self):
        # all held-out rows are high-rsi -> only positive-neighbour group -> no contrast
        obs = [_o(80, 1.0, "2026-01-01"), _o(85, 1.0, "2026-01-02"), _o(82, 2.0, "2026-01-05")]
        r = evaluate_case_memory(obs, k=1, split_frac=0.66, features=_F)
        self.assertIsNone(r["edge"])
        self.assertFalse(r["helps"])

    def test_guard_returns_empty_on_thin_data(self):
        r = evaluate_case_memory([_o(50, 1.0, "2026-01-01")], features=_F)  # split=0
        self.assertEqual((r["n_cases"], r["n_test"], r["edge"]), (0, 0, None))

    def test_max_test_caps_held_out(self):
        obs = [_o(80, 1.0, f"2026-01-0{i}") for i in range(1, 5)] + [
            _o(82, 2.0, "2026-01-05"),
            _o(83, 2.0, "2026-01-06"),
            _o(84, 2.0, "2026-01-07"),
        ]
        r = evaluate_case_memory(obs, k=1, split_frac=0.5, features=_F, max_test=1)
        self.assertEqual(r["n_test"], 1)  # only the most-recent held-out row scored


class TestBuildLines(unittest.TestCase):
    def test_no_sample(self):
        self.assertIn("not enough matured sample", build_case_memory_lines({"edge": None})[0])

    def test_helps_verdict(self):
        line = build_case_memory_lines(
            {"edge": 0.4, "helps": True, "k": 10, "n_test": 200, "n_pos": 120, "n_neg": 80}
        )[0]
        self.assertIn("HELPS: candidate to consult", line)

    def test_no_edge_verdict(self):
        line = build_case_memory_lines(
            {"edge": 0.02, "helps": False, "k": 10, "n_test": 200, "n_pos": 120, "n_neg": 80}
        )[0]
        self.assertIn("do not consult", line)


class TestToCandidate(unittest.TestCase):
    def test_authors_when_edge_clears_bar(self):
        c = to_candidate({"edge": 0.3, "n_test": 200, "k": 10}, "2026-07-19")
        self.assertEqual((c.id, c.source), ("consult_case_memory", "case_memory"))

    def test_none_when_no_edge(self):
        self.assertIsNone(to_candidate({"edge": None, "n_test": 200, "k": 10}, "2026-07-19"))

    def test_none_below_sample_floor(self):
        self.assertIsNone(to_candidate({"edge": 0.3, "n_test": 10, "k": 10}, "2026-07-19"))

    def test_none_below_effect_floor(self):
        self.assertIsNone(
            to_candidate({"edge": 0.05, "n_test": 200, "k": 10}, "2026-07-19", min_effect=0.1)
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
