import unittest
from risk.risk_manager import (
    check_circuit_breaker,
    check_daily_loss,
    check_vix_stop_adjustment,
    validate_buy_candidates,
)


def _record(value):
    return {"account_after": {"portfolio_value": value}}


class TestCircuitBreaker(unittest.TestCase):

    def test_not_triggered_on_flat_portfolio(self):
        history = [_record(100_000)] * 5
        triggered, drawdown = check_circuit_breaker(history)
        self.assertFalse(triggered)
        self.assertEqual(drawdown, 0.0)

    def test_triggered_on_large_drawdown(self):
        history = [_record(100_000), _record(100_000), _record(100_000),
                   _record(100_000), _record(85_000)]
        triggered, drawdown = check_circuit_breaker(history)
        self.assertTrue(triggered)
        self.assertLess(drawdown, -12.0)

    def test_not_triggered_on_small_dip(self):
        history = [_record(100_000)] * 4 + [_record(95_000)]
        triggered, _ = check_circuit_breaker(history)
        self.assertFalse(triggered)

    def test_insufficient_history_returns_false(self):
        triggered, drawdown = check_circuit_breaker([_record(100_000)])
        self.assertFalse(triggered)
        self.assertEqual(drawdown, 0.0)

    def test_empty_history_returns_false(self):
        triggered, drawdown = check_circuit_breaker([])
        self.assertFalse(triggered)

    def test_drawdown_value_is_correct(self):
        history = [_record(100_000), _record(100_000), _record(100_000),
                   _record(100_000), _record(80_000)]
        _, drawdown = check_circuit_breaker(history)
        self.assertAlmostEqual(drawdown, -20.0, places=1)


class TestDailyLoss(unittest.TestCase):

    def test_not_triggered_on_gain(self):
        triggered, pct = check_daily_loss(100_000, 102_000)
        self.assertFalse(triggered)
        self.assertGreater(pct, 0)

    def test_triggered_on_large_loss(self):
        triggered, pct = check_daily_loss(100_000, 94_000)
        self.assertTrue(triggered)
        self.assertLess(pct, -5.0)

    def test_not_triggered_on_small_loss(self):
        triggered, _ = check_daily_loss(100_000, 98_000)
        self.assertFalse(triggered)

    def test_zero_open_value_returns_false(self):
        triggered, pct = check_daily_loss(0, 100_000)
        self.assertFalse(triggered)
        self.assertEqual(pct, 0.0)

    def test_loss_percentage_is_correct(self):
        _, pct = check_daily_loss(100_000, 95_000)
        self.assertAlmostEqual(pct, -5.0, places=1)


class TestVixStopAdjustment(unittest.TestCase):

    def test_none_returns_default(self):
        self.assertEqual(check_vix_stop_adjustment(None), 4.0)

    def test_low_vix_tight_stop(self):
        self.assertEqual(check_vix_stop_adjustment(15.0), 3.0)

    def test_normal_vix(self):
        self.assertEqual(check_vix_stop_adjustment(20.0), 4.0)

    def test_elevated_vix(self):
        self.assertEqual(check_vix_stop_adjustment(30.0), 5.5)

    def test_very_high_vix(self):
        self.assertEqual(check_vix_stop_adjustment(40.0), 7.0)


class TestValidateBuyCandidates(unittest.TestCase):

    def _sector_map(self, sym):
        return {"AAPL": "Technology", "MSFT": "Technology",
                "JPM": "Financials", "SPY": "ETF"}.get(sym, "Unknown")

    def test_filters_already_held_symbols(self):
        candidates = [{"symbol": "AAPL", "confidence": 8}]
        result = validate_buy_candidates(candidates, {"AAPL"}, self._sector_map)
        self.assertEqual(result, [])

    def test_passes_unrelated_symbols(self):
        candidates = [{"symbol": "JPM", "confidence": 8}]
        result = validate_buy_candidates(candidates, {"AAPL"}, self._sector_map)
        self.assertEqual(len(result), 1)

    def test_sector_cap_blocks_third_in_same_sector(self):
        candidates = [{"symbol": "MSFT", "confidence": 8}]
        held = {"AAPL"}
        # Artificially fill Technology to the cap (2)
        result = validate_buy_candidates(
            candidates, held, self._sector_map, max_per_sector=1
        )
        self.assertEqual(result, [])

    def test_etf_bypasses_sector_cap(self):
        candidates = [{"symbol": "SPY", "confidence": 8}]
        result = validate_buy_candidates(
            candidates, {"AAPL", "MSFT"}, self._sector_map, max_per_sector=1
        )
        self.assertEqual(len(result), 1)

    def test_empty_candidates_returns_empty(self):
        result = validate_buy_candidates([], {"AAPL"}, self._sector_map)
        self.assertEqual(result, [])
