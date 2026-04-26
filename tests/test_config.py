import unittest
from datetime import date
import config


class TestValidate(unittest.TestCase):

    def _patch(self, attr, value):
        original = getattr(config, attr)
        setattr(config, attr, value)
        self.addCleanup(setattr, config, attr, original)

    def test_valid_config_passes(self):
        config.validate()

    def test_max_position_pct_above_one_fails(self):
        self._patch("MAX_POSITION_PCT", 1.5)
        with self.assertRaises(ValueError):
            config.validate()

    def test_max_position_pct_zero_fails(self):
        self._patch("MAX_POSITION_PCT", 0.0)
        with self.assertRaises(ValueError):
            config.validate()

    def test_cash_reserve_zero_fails(self):
        self._patch("CASH_RESERVE_PCT", 0.0)
        with self.assertRaises(ValueError):
            config.validate()

    def test_cash_reserve_above_one_fails(self):
        self._patch("CASH_RESERVE_PCT", 1.1)
        with self.assertRaises(ValueError):
            config.validate()

    def test_min_confidence_above_ten_fails(self):
        self._patch("MIN_CONFIDENCE", 11)
        with self.assertRaises(ValueError):
            config.validate()

    def test_min_confidence_zero_fails(self):
        self._patch("MIN_CONFIDENCE", 0)
        with self.assertRaises(ValueError):
            config.validate()

    def test_trailing_stop_zero_fails(self):
        self._patch("TRAILING_STOP_PCT", 0.0)
        with self.assertRaises(ValueError):
            config.validate()

    def test_max_hold_days_zero_fails(self):
        self._patch("MAX_HOLD_DAYS", 0)
        with self.assertRaises(ValueError):
            config.validate()

    def test_max_positions_zero_fails(self):
        self._patch("MAX_POSITIONS", 0)
        with self.assertRaises(ValueError):
            config.validate()

    def test_max_single_order_zero_fails(self):
        self._patch("MAX_SINGLE_ORDER_USD", 0.0)
        with self.assertRaises(ValueError):
            config.validate()

    def test_error_message_lists_all_failures(self):
        self._patch("MAX_POSITION_PCT", 2.0)
        self._patch("MIN_CONFIDENCE", 0)
        with self.assertRaises(ValueError) as ctx:
            config.validate()
        msg = str(ctx.exception)
        self.assertIn("MAX_POSITION_PCT", msg)
        self.assertIn("MIN_CONFIDENCE", msg)


class TestTodayEt(unittest.TestCase):

    def test_returns_date_object(self):
        self.assertIsInstance(config.today_et(), date)

    def test_date_year_is_plausible(self):
        self.assertGreater(config.today_et().year, 2020)
