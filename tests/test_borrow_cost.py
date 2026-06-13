"""Tests for data.borrow_cost — borrow-rate estimation and dollar-cost conversion."""

import unittest

from data.borrow_cost import (
    GC_RATE,
    HTB_RATE_THRESHOLD,
    borrow_cost_usd,
    borrow_rate_for_symbol,
    estimate_borrow_rate,
    is_hard_to_borrow,
)


class TestEstimateBorrowRateByPctFloat(unittest.TestCase):
    def test_low_si_returns_gc_rate(self):
        self.assertEqual(estimate_borrow_rate(0.02), GC_RATE)

    def test_boundary_5pct_is_gc(self):
        self.assertEqual(estimate_borrow_rate(0.05), GC_RATE)

    def test_moderate_si_tier(self):
        self.assertEqual(estimate_borrow_rate(0.10), 0.03)

    def test_elevated_si_tier(self):
        self.assertEqual(estimate_borrow_rate(0.25), 0.10)

    def test_hard_tier(self):
        self.assertEqual(estimate_borrow_rate(0.40), 0.30)

    def test_special_tier_above_50pct(self):
        self.assertEqual(estimate_borrow_rate(0.65), 0.80)

    def test_percentage_form_normalised(self):
        """A value > 1.5 is treated as a percentage (18.0 → 0.18 → 10% tier)."""
        self.assertEqual(estimate_borrow_rate(18.0), 0.10)


class TestEstimateBorrowRateByShortRatio(unittest.TestCase):
    def test_low_days_to_cover_is_gc(self):
        self.assertEqual(estimate_borrow_rate(None, short_ratio=2.0), GC_RATE)

    def test_mid_days_to_cover(self):
        self.assertEqual(estimate_borrow_rate(None, short_ratio=5.0), 0.03)

    def test_elevated_days_to_cover(self):
        self.assertEqual(estimate_borrow_rate(None, short_ratio=9.0), 0.10)

    def test_high_days_to_cover_special(self):
        self.assertEqual(estimate_borrow_rate(None, short_ratio=15.0), 0.30)

    def test_no_data_defaults_to_gc(self):
        self.assertEqual(estimate_borrow_rate(None, None), GC_RATE)

    def test_pct_float_takes_precedence_over_ratio(self):
        # short_pct_float present → ratio ignored
        self.assertEqual(estimate_borrow_rate(0.02, short_ratio=20.0), GC_RATE)


class TestIsHardToBorrow(unittest.TestCase):
    def test_low_si_not_htb(self):
        self.assertFalse(is_hard_to_borrow(0.10))

    def test_high_si_is_htb(self):
        self.assertTrue(is_hard_to_borrow(0.40))

    def test_threshold_exactly_at_boundary_is_htb(self):
        # 30–50% tier returns 0.30, equal to HTB_RATE_THRESHOLD → HTB
        self.assertEqual(estimate_borrow_rate(0.40), HTB_RATE_THRESHOLD)
        self.assertTrue(is_hard_to_borrow(0.40))

    def test_ratio_fallback_htb(self):
        self.assertTrue(is_hard_to_borrow(None, short_ratio=15.0))


class TestBorrowCostUsd(unittest.TestCase):
    def test_basic_cost(self):
        # $10,000 notional at 10%/yr for 252 days = full year = $1,000
        self.assertAlmostEqual(borrow_cost_usd(0.10, 10_000.0, 252), 1_000.0)

    def test_pro_rated_for_partial_year(self):
        # 5 trading days at 10%/yr on $10,000 = 10000 * 0.10 * 5/252
        self.assertAlmostEqual(borrow_cost_usd(0.10, 10_000.0, 5), 10_000.0 * 0.10 * 5 / 252)

    def test_zero_rate_is_free(self):
        self.assertEqual(borrow_cost_usd(0.0, 10_000.0, 5), 0.0)

    def test_zero_notional(self):
        self.assertEqual(borrow_cost_usd(0.10, 0.0, 5), 0.0)

    def test_zero_days(self):
        self.assertEqual(borrow_cost_usd(0.10, 10_000.0, 0), 0.0)

    def test_negative_rate_clamped(self):
        self.assertEqual(borrow_cost_usd(-0.10, 10_000.0, 5), 0.0)


class TestBorrowRateForSymbol(unittest.TestCase):
    def test_known_symbol_uses_record(self):
        si = {"GME": {"short_pct_float": 0.40, "short_ratio": 5.0}}
        self.assertEqual(borrow_rate_for_symbol("GME", si), 0.30)

    def test_missing_symbol_defaults_gc(self):
        self.assertEqual(borrow_rate_for_symbol("AAPL", {}), GC_RATE)

    def test_none_record_defaults_gc(self):
        self.assertEqual(borrow_rate_for_symbol("AAPL", {"AAPL": None}), GC_RATE)

    def test_record_without_pct_float_uses_ratio(self):
        si = {"XYZ": {"short_pct_float": None, "short_ratio": 9.0}}
        self.assertEqual(borrow_rate_for_symbol("XYZ", si), 0.10)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
