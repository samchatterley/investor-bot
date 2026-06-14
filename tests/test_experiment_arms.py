import unittest

from experiment.arms import (
    ADJUSTMENT_VALUE,
    ARM2_TOOL,
    ARM3_TOOL,
    COARSE_ADJUSTMENTS,
    REASON_CODES,
    adjustment_value,
    build_arm2_prompt,
    build_arm3_prompt,
    build_structured_prose,
    final_conviction,
)


class TestConstants(unittest.TestCase):
    def test_adjustment_value_keys_match_scale(self):
        self.assertEqual(set(ADJUSTMENT_VALUE), set(COARSE_ADJUSTMENTS))
        self.assertEqual(ADJUSTMENT_VALUE["negative"], -1)
        self.assertEqual(ADJUSTMENT_VALUE["neutral"], 0)
        self.assertEqual(ADJUSTMENT_VALUE["positive"], 1)

    def test_reason_codes_include_other(self):
        self.assertIn("other", REASON_CODES)


class TestStructuredProse(unittest.TestCase):
    def test_includes_present_fields_in_order_skips_none(self):
        prose = build_structured_prose(
            {"symbol": "AAA", "evidence_score": 1.2, "rsi_14": None, "vol_ratio": 1.5}
        )
        self.assertIn("Symbol: AAA", prose)
        self.assertIn("Evidence score: 1.2", prose)
        self.assertIn("Volume ratio: 1.5", prose)
        self.assertNotIn("RSI(14)", prose)  # None skipped
        # Symbol appears before Evidence score (fixed order)
        self.assertLess(prose.index("Symbol"), prose.index("Evidence score"))

    def test_matched_signals_appended(self):
        prose = build_structured_prose({"symbol": "B", "matched_signals": ["pead", "momentum"]})
        self.assertIn("Fired signals: pead, momentum", prose)

    def test_signals_fallback_when_no_matched(self):
        prose = build_structured_prose({"symbol": "C", "signals": ["macd_crossover"]})
        self.assertIn("Fired signals: macd_crossover", prose)

    def test_no_signals_line_when_absent(self):
        prose = build_structured_prose({"symbol": "D"})
        self.assertNotIn("Fired signals", prose)


class TestPrompts(unittest.TestCase):
    def test_arm2_prompt_contains_prose_no_context(self):
        prose = build_structured_prose({"symbol": "AAA", "evidence_score": 1.0})
        p = build_arm2_prompt(prose)
        self.assertIn("Symbol: AAA", p)
        self.assertIn("structured_conviction", p)
        self.assertNotIn("CONTEXT", p)

    def test_arm3_prompt_is_arm2_plus_context_block(self):
        prose = build_structured_prose({"symbol": "AAA"})
        arm2 = {"structured_conviction": 7, "structured_reasoning": "clean trend"}
        p = build_arm3_prompt(prose, arm2, "Earnings beat reported pre-decision.")
        self.assertTrue(p.startswith(build_arm2_prompt(prose)))  # exact base prompt preserved
        self.assertIn("structured_conviction=7", p)
        self.assertIn("clean trend", p)
        self.assertIn("Earnings beat reported pre-decision.", p)
        self.assertIn("context_adjustment", p)

    def test_arm3_prompt_empty_context_fallback(self):
        p = build_arm3_prompt("x", {"structured_conviction": 5, "structured_reasoning": "r"}, "")
        self.assertIn("(no admissible context)", p)


class TestConvictionMath(unittest.TestCase):
    def test_adjustment_value_valid(self):
        self.assertEqual(adjustment_value("positive"), 1)

    def test_adjustment_value_invalid_raises(self):
        with self.assertRaises(ValueError):
            adjustment_value("strong_buy")

    def test_final_conviction_sums(self):
        self.assertEqual(final_conviction(5, "positive"), 6)
        self.assertEqual(final_conviction(5, "neutral"), 5)
        self.assertEqual(final_conviction(5, "negative"), 4)

    def test_final_conviction_clamps_high(self):
        self.assertEqual(final_conviction(10, "positive"), 10)

    def test_final_conviction_clamps_low(self):
        self.assertEqual(final_conviction(1, "negative"), 1)


class TestToolSchemas(unittest.TestCase):
    def test_arm2_tool_bounds_and_required(self):
        props = ARM2_TOOL["input_schema"]["properties"]
        self.assertEqual(props["structured_conviction"]["maximum"], 10)
        self.assertIn("structured_conviction", ARM2_TOOL["input_schema"]["required"])

    def test_arm3_tool_enums_match_constants(self):
        props = ARM3_TOOL["input_schema"]["properties"]
        self.assertEqual(props["context_adjustment"]["enum"], list(COARSE_ADJUSTMENTS))
        self.assertEqual(props["reason_code"]["enum"], list(REASON_CODES))
        self.assertIn("veto", ARM3_TOOL["input_schema"]["required"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
