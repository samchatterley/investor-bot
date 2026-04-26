import unittest
import config
from risk.position_sizer import kelly_fraction, get_max_positions


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
