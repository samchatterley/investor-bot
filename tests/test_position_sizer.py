import unittest
from unittest.mock import patch

import config
from risk.position_sizer import get_max_positions, kelly_fraction, risk_budget_size


class TestKellyFraction(unittest.TestCase):

    def test_zero_confidence_returns_zero(self):
        self.assertEqual(kelly_fraction(0), 0.0)

    def test_confidence_ten_capped_at_max_position_pct(self):
        result = kelly_fraction(10)
        self.assertEqual(result, config.MAX_POSITION_PCT)

    def test_confidence_seven_is_positive(self):
        result = kelly_fraction(7)
        self.assertGreater(result, 0.0)

    def test_confidence_seven_below_cap(self):
        result = kelly_fraction(7)
        self.assertLessEqual(result, config.MAX_POSITION_PCT)

    def test_higher_confidence_gives_larger_fraction(self):
        self.assertGreater(kelly_fraction(9), kelly_fraction(7))

    def test_result_never_negative(self):
        for conf in range(0, 11):
            self.assertGreaterEqual(kelly_fraction(conf), 0.0)

    def test_result_never_exceeds_max_position_pct(self):
        for conf in range(0, 11):
            self.assertLessEqual(kelly_fraction(conf), config.MAX_POSITION_PCT)


class TestRiskBudgetSize(unittest.TestCase):
    """
    risk_budget_size: notional = (equity * RISK_PER_TRADE_PCT) / (TRAILING_STOP_PCT / 100)
    capped at equity * MAX_POSITION_WEIGHT.
    """

    def test_returns_positive_for_normal_inputs(self):
        result = risk_budget_size(10_000, confidence=8)
        self.assertGreater(result, 0.0)

    def test_zero_equity_returns_zero(self):
        self.assertEqual(risk_budget_size(0, confidence=8), 0.0)

    def test_negative_equity_returns_zero(self):
        self.assertEqual(risk_budget_size(-5_000, confidence=8), 0.0)

    def test_scales_with_equity(self):
        small = risk_budget_size(10_000, confidence=8)
        large = risk_budget_size(100_000, confidence=8)
        self.assertGreater(large, small)

    def test_capped_at_max_position_weight(self):
        equity = 10_000
        result = risk_budget_size(equity, confidence=10)
        self.assertLessEqual(result, equity * config.MAX_POSITION_WEIGHT)

    def test_never_exceeds_cap_regardless_of_confidence(self):
        equity = 50_000
        for conf in range(0, 11):
            result = risk_budget_size(equity, confidence=conf)
            self.assertLessEqual(result, equity * config.MAX_POSITION_WEIGHT)

    def test_never_negative_regardless_of_confidence(self):
        for conf in range(0, 11):
            self.assertGreaterEqual(risk_budget_size(10_000, confidence=conf), 0.0)

    def test_formula_matches_expected_value(self):
        # notional = equity * RISK_PER_TRADE_PCT / (TRAILING_STOP_PCT / 100)
        # then capped at equity * MAX_POSITION_WEIGHT
        equity = 10_000
        risk_usd = equity * config.RISK_PER_TRADE_PCT
        stop_pct = config.TRAILING_STOP_PCT / 100.0
        expected = min(risk_usd / stop_pct, equity * config.MAX_POSITION_WEIGHT)
        self.assertAlmostEqual(risk_budget_size(equity, confidence=8), expected, places=4)

    def test_notional_unaffected_when_kelly_unavailable(self):
        """Kelly is telemetry only — risk_budget_size returns valid notional even if kelly_fraction fails."""
        def forbidden(*args, **kwargs):
            raise AssertionError("kelly_fraction must not determine live notional")
        with patch("risk.position_sizer.kelly_fraction", side_effect=forbidden):
            result = risk_budget_size(10_000, confidence=8)
        self.assertGreater(result, 0.0)

    def test_confidence_does_not_change_notional(self):
        # Confidence affects Kelly telemetry only — risk-budget notional is identical
        # for confidence=7 and confidence=9 (both above MIN_CONFIDENCE gate)
        equity = 20_000
        self.assertEqual(
            risk_budget_size(equity, confidence=7),
            risk_budget_size(equity, confidence=9),
        )


class TestGetMaxPositions(unittest.TestCase):

    def test_large_account_returns_five(self):
        self.assertEqual(get_max_positions(100_000), 5)

    def test_exactly_fifty_thousand_returns_five(self):
        self.assertEqual(get_max_positions(50_000), 5)

    def test_medium_account_returns_four(self):
        self.assertEqual(get_max_positions(30_000), 4)

    def test_exactly_twenty_thousand_returns_four(self):
        self.assertEqual(get_max_positions(20_000), 4)

    def test_small_account_returns_three(self):
        self.assertEqual(get_max_positions(5_000), 3)
