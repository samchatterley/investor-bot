import unittest

from experiment.material_context import (
    MATERIAL_CONTEXT_CATEGORIES,
    detect_material_context,
    is_material_context,
)


class TestDetectMaterialContext(unittest.TestCase):
    def test_empty_candidate_has_no_categories(self):
        self.assertEqual(detect_material_context({}), [])
        self.assertFalse(is_material_context({}))

    def test_output_is_subset_in_registry_order(self):
        cand = {
            "matched_signals": ["pead", "insider_buying", "squeeze_setup_long"],
            "secondary_offering": True,
        }
        result = detect_material_context(cand)
        self.assertTrue(set(result) <= set(MATERIAL_CONTEXT_CATEGORIES))
        # registry order preserved
        self.assertEqual(result, [c for c in MATERIAL_CONTEXT_CATEGORIES if c in result])
        self.assertTrue(is_material_context(cand))

    # earnings: three independent triggers
    def test_earnings_via_pead_signal(self):
        self.assertIn(
            "earnings_surprise_or_drift", detect_material_context({"matched_signals": ["pead"]})
        )

    def test_earnings_via_gap(self):
        self.assertIn(
            "earnings_surprise_or_drift", detect_material_context({"earnings_gap_pct": -8.0})
        )

    def test_small_gap_does_not_trigger_earnings(self):
        self.assertNotIn(
            "earnings_surprise_or_drift", detect_material_context({"earnings_gap_pct": -2.0})
        )

    def test_earnings_via_flag(self):
        self.assertIn(
            "earnings_surprise_or_drift", detect_material_context({"earnings_event": True})
        )

    def test_guidance_via_signal(self):
        self.assertIn(
            "guidance_change", detect_material_context({"signals": ["guidance_raise_signal"]})
        )

    def test_insider_via_signal(self):
        self.assertIn(
            "insider_cluster_buying",
            detect_material_context({"matched_signals": ["insider_buying"]}),
        )

    def test_analyst_via_signal(self):
        self.assertIn(
            "analyst_action",
            detect_material_context({"matched_signals": ["analyst_upgrade_signal"]}),
        )

    def test_squeeze_via_signal(self):
        for sig in ("squeeze_setup_long", "squeeze_momentum_long", "short_interest_trend_long"):
            self.assertIn(
                "short_squeeze_setup", detect_material_context({"matched_signals": [sig]}), sig
            )

    def test_flag_only_categories(self):
        cases = {
            "secondary_offering_dilution": "secondary_offering",
            "regulatory_event": "regulatory_event",
            "index_change": "index_change",
            "ma_event": "ma_event",
            "accounting_concern": "accounting_concern",
        }
        for category, flag in cases.items():
            self.assertIn(category, detect_material_context({flag: True}), category)
            self.assertNotIn(category, detect_material_context({flag: False}), category)

    def test_signals_fallback_key(self):
        # uses "signals" when "matched_signals" absent
        self.assertTrue(is_material_context({"signals": ["pead"]}))

    def test_multiple_categories(self):
        cand = {"matched_signals": ["pead", "guidance_raise_signal"], "ma_event": True}
        result = detect_material_context(cand)
        self.assertEqual(result, ["earnings_surprise_or_drift", "guidance_change", "ma_event"])

    def test_non_material_signal_only(self):
        # a signal not in any material category does not qualify the candidate
        self.assertEqual(detect_material_context({"matched_signals": ["momentum"]}), [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
