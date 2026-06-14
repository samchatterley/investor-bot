import unittest

from experiment.noise_audit import (
    format_report,
    run_noise_audit,
    summarise_stability,
)


class TestSummariseStability(unittest.TestCase):
    def test_all_stable_passes(self):
        r = summarise_stability({"A": ["neutral"] * 5, "B": ["positive"] * 5})
        self.assertEqual(r.mean_flip_rate, 0.0)
        self.assertEqual(r.frac_fully_stable, 1.0)
        self.assertTrue(r.passed)
        self.assertEqual(r.n_runs, 5)

    def test_flip_rate_computed(self):
        # 4 neutral, 1 positive -> mode neutral, flip rate 0.2
        r = summarise_stability({"A": ["neutral", "neutral", "neutral", "neutral", "positive"]})
        self.assertAlmostEqual(r.per_candidate[0].flip_rate, 0.2)
        self.assertEqual(r.per_candidate[0].mode, "neutral")
        self.assertFalse(r.per_candidate[0].fully_stable)

    def test_high_flip_fails(self):
        # alternating -> flip rate ~0.4-0.5, above default 0.2 threshold
        r = summarise_stability({"A": ["neutral", "positive", "negative", "positive", "neutral"]})
        self.assertFalse(r.passed)

    def test_threshold_is_respected(self):
        runs = {"A": ["neutral", "neutral", "neutral", "neutral", "positive"]}  # flip 0.2
        self.assertTrue(summarise_stability(runs, pass_flip_threshold=0.2).passed)
        self.assertFalse(summarise_stability(runs, pass_flip_threshold=0.1).passed)

    def test_empty_dict_raises(self):
        with self.assertRaises(ValueError):
            summarise_stability({})

    def test_all_empty_runs_raises(self):
        with self.assertRaises(ValueError):
            summarise_stability({"A": [], "B": []})

    def test_skips_empty_run_candidates(self):
        r = summarise_stability({"A": ["neutral"] * 3, "B": []})
        self.assertEqual(r.n_candidates, 1)


class TestRunNoiseAudit(unittest.TestCase):
    def test_deterministic_caller_is_stable(self):
        r = run_noise_audit(lambda c: "neutral", [{"symbol": "A"}, {"symbol": "B"}], runs=5)
        self.assertTrue(r.passed)
        self.assertEqual(r.frac_fully_stable, 1.0)

    def test_flipping_caller_is_unstable(self):
        state = {"n": 0}

        def flip(_c):
            state["n"] += 1
            return "positive" if state["n"] % 2 else "negative"

        r = run_noise_audit(flip, [{"symbol": "A"}], runs=6)
        self.assertFalse(r.passed)

    def test_runs_below_two_raises(self):
        with self.assertRaises(ValueError):
            run_noise_audit(lambda c: "neutral", [{"symbol": "A"}], runs=1)

    def test_empty_candidates_raises(self):
        with self.assertRaises(ValueError):
            run_noise_audit(lambda c: "neutral", [], runs=5)

    def test_candidate_without_symbol_gets_placeholder(self):
        r = run_noise_audit(lambda c: "neutral", [{}], runs=3)
        self.assertEqual(r.per_candidate[0].symbol, "candidate_0")


class TestFormatReport(unittest.TestCase):
    def test_pass_report(self):
        r = summarise_stability({"A": ["neutral"] * 5})
        text = format_report(r)
        self.assertIn("PASS", text)
        self.assertIn("noise audit", text)

    def test_fail_report(self):
        r = summarise_stability({"A": ["neutral", "positive", "negative", "positive", "neutral"]})
        self.assertIn("FAIL", format_report(r))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
