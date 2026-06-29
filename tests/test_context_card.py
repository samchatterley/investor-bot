"""Tests for analysis/context_card.py — the prompt-affordable distillation layer."""

import unittest

import analysis.context_card as cc


class TestCatalystLine(unittest.TestCase):
    def test_bearish_only(self):
        line = cc._catalyst_line("AAA", {"eps_estimate_cut": True, "guidance_negative": True}, None)
        self.assertIn("bearish: ", line)
        self.assertIn("EPS estimate cut", line)
        self.assertNotIn("bullish", line)

    def test_bullish_only(self):
        line = cc._catalyst_line("AAA", {"insider_cluster": True}, None)
        self.assertEqual(line, "bullish: insider buying")

    def test_both_sides(self):
        line = cc._catalyst_line("AAA", {"eps_estimate_cut": True, "analyst_upgrade": True}, None)
        self.assertIn("bearish:", line)
        self.assertIn("bullish:", line)

    def test_none_when_no_catalysts(self):
        self.assertIsNone(cc._catalyst_line("AAA", {"rsi_14": 50}, None))


class TestShortInterestLine(unittest.TestCase):
    def test_with_float_and_days(self):
        line = cc._short_interest_line("AAA", {"short_pct_float": 0.18, "days_to_cover": 4.2}, None)
        self.assertEqual(line, "short interest 18% of float, 4.2d to cover")

    def test_with_float_only(self):
        line = cc._short_interest_line("AAA", {"short_pct_float": 0.09}, None)
        self.assertEqual(line, "short interest 9% of float")

    def test_none_without_float(self):
        self.assertIsNone(cc._short_interest_line("AAA", {}, None))


class TestFilingChangeLine(unittest.TestCase):
    def test_line_when_above_threshold(self):
        ctx = {"filing_change": {"change_score": 0.15}}
        line = cc._filing_change_line("AAA", {}, ctx)
        self.assertIn("10-K language changed 15% YoY", line)

    def test_none_below_threshold(self):
        ctx = {"filing_change": {"change_score": 0.02}}
        self.assertIsNone(cc._filing_change_line("AAA", {}, ctx))

    def test_none_without_record(self):
        self.assertIsNone(cc._filing_change_line("AAA", {}, {}))
        self.assertIsNone(cc._filing_change_line("AAA", {}, None))


class TestBuildContextCard(unittest.TestCase):
    def test_assembles_non_empty_lines(self):
        snap = {"eps_estimate_cut": True, "short_pct_float": 0.2, "days_to_cover": 5.0}
        ctx = {"filing_change": {"change_score": 0.2}}
        card = cc.build_context_card("NVDA", snap, context=ctx)
        self.assertTrue(card.startswith("NVDA context:"))
        self.assertIn("EPS estimate cut", card)
        self.assertIn("short interest 20% of float", card)
        self.assertIn("Lazy Prices", card)
        self.assertEqual(card.count("•"), 3)

    def test_empty_when_nothing_material(self):
        self.assertEqual(cc.build_context_card("AAA", {"rsi_14": 50}), "")

    def test_contributor_exception_is_fail_safe(self):
        def _boom(symbol, snapshot, context):
            raise ValueError("contributor blew up")

        snap = {"insider_cluster": True}
        card = cc.build_context_card("AAA", snap, contributors=(_boom, cc._catalyst_line))
        self.assertIn("bullish: insider buying", card)  # the good contributor still ran

    def test_respects_char_budget(self):
        snap = dict.fromkeys(cc._BEARISH_CATALYSTS, True) | dict.fromkeys(
            cc._BULLISH_CATALYSTS, True
        )
        card = cc.build_context_card("AAA", snap, max_chars=40)
        self.assertLessEqual(len(card), 40)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
