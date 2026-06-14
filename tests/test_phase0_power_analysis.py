import math
import unittest

from scripts.phase0_power_analysis import (
    Assumptions,
    PowerResult,
    design_effect,
    effective_n,
    main,
    min_detectable_ic,
    project,
    required_n_for_ic,
)


class TestMinDetectableIC(unittest.TestCase):
    def test_known_value(self):
        # 1.96 / sqrt(384) ≈ 0.1000
        self.assertAlmostEqual(min_detectable_ic(384, z=1.96), 0.10002, places=4)

    def test_larger_n_gives_smaller_mdi(self):
        self.assertLess(min_detectable_ic(1000), min_detectable_ic(100))

    def test_nonpositive_n_raises(self):
        with self.assertRaises(ValueError):
            min_detectable_ic(0)
        with self.assertRaises(ValueError):
            min_detectable_ic(-5)


class TestRequiredNForIC(unittest.TestCase):
    def test_known_value(self):
        # (1.96 / 0.05)^2 = 1536.64
        self.assertAlmostEqual(required_n_for_ic(0.05, z=1.96), 1536.64, places=2)

    def test_smaller_ic_needs_more_n(self):
        self.assertGreater(required_n_for_ic(0.02), required_n_for_ic(0.05))

    def test_nonpositive_ic_raises(self):
        with self.assertRaises(ValueError):
            required_n_for_ic(0)
        with self.assertRaises(ValueError):
            required_n_for_ic(-0.1)


class TestDesignEffect(unittest.TestCase):
    def test_known_value(self):
        # 1 + (4 - 1) * 0.1 = 1.3
        self.assertAlmostEqual(design_effect(4, 0.1), 1.3)

    def test_single_member_cluster_is_one(self):
        self.assertEqual(design_effect(1, 0.5), 1.0)

    def test_zero_icc_is_one(self):
        self.assertEqual(design_effect(10, 0.0), 1.0)

    def test_invalid_cluster_size_raises(self):
        with self.assertRaises(ValueError):
            design_effect(0.5, 0.1)

    def test_invalid_icc_raises(self):
        with self.assertRaises(ValueError):
            design_effect(4, 1.5)
        with self.assertRaises(ValueError):
            design_effect(4, -0.1)


class TestEffectiveN(unittest.TestCase):
    def test_known_value(self):
        # 130 / 1.3 = 100
        self.assertAlmostEqual(effective_n(130, 4, 0.1), 100.0)

    def test_no_clustering_returns_raw(self):
        self.assertEqual(effective_n(130, 1, 0.1), 130.0)

    def test_negative_raw_raises(self):
        with self.assertRaises(ValueError):
            effective_n(-1, 4, 0.1)


class TestProject(unittest.TestCase):
    def test_defaults_return_powerresult(self):
        r = project(Assumptions())
        self.assertIsInstance(r, PowerResult)
        self.assertGreater(r.eligible_per_day, 0)
        self.assertGreater(r.mdi_material, 0)

    def test_material_n_not_greater_than_context_present_n(self):
        r = project(Assumptions())
        self.assertLessEqual(r.raw_material_n, r.raw_context_present_n)

    def test_material_mdi_not_better_than_context_present(self):
        # Fewer effective observations → larger (worse) minimum detectable IC.
        r = project(Assumptions())
        self.assertGreaterEqual(r.mdi_material, r.mdi_context_present)

    def test_daily_cap_binds(self):
        # A tiny cap forces context_present_per_day down to the cap.
        a = Assumptions(daily_arm3_cap=2)
        r = project(a)
        self.assertLessEqual(r.context_present_per_day, 2)

    def test_small_account_setup_is_underpowered(self):
        # The realistic small-universe/short-window case cannot detect a modest IC.
        a = Assumptions(eligible_rate=0.04, context_present_rate=0.1, trading_days=126)
        r = project(a)
        self.assertFalse(r.powered)

    def test_generous_flow_can_be_powered(self):
        # Enough flow over a long window to detect a 0.05 IC.
        a = Assumptions(
            eligible_rate=0.3,
            context_present_rate=0.5,
            material_rate=0.8,
            trading_days=750,
            daily_arm3_cap=200,
            icc=0.02,
        )
        r = project(a)
        self.assertTrue(r.powered)

    def test_powered_matches_threshold(self):
        r = project(Assumptions())
        self.assertEqual(r.powered, r.mdi_material <= Assumptions().plausible_ic)

    def test_mdi_consistent_with_n_eff(self):
        r = project(Assumptions())
        self.assertAlmostEqual(r.mdi_material, 1.96 / math.sqrt(r.n_eff_material), places=6)


class TestMain(unittest.TestCase):
    def test_main_runs_and_prints(self):
        # Smoke test: default args, no exception, prints a verdict.
        main([])

    def test_main_accepts_overrides(self):
        main(["--universe", "100", "--trading-days", "60", "--plausible-ic", "0.1"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
