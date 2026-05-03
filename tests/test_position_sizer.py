import json
import os
import tempfile
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
        # Without empirical data, conviction_scale is flat 0.75
        equity = 10_000
        risk_usd = equity * config.RISK_PER_TRADE_PCT
        stop_pct = config.TRAILING_STOP_PCT / 100.0
        base = min(risk_usd / stop_pct, equity * config.MAX_POSITION_WEIGHT)
        self.assertAlmostEqual(risk_budget_size(equity, confidence=8), base * 0.75, places=4)

    def test_notional_unaffected_when_kelly_unavailable(self):
        """Kelly is telemetry only — risk_budget_size returns valid notional even if kelly_fraction fails."""

        def forbidden(*args, **kwargs):
            raise AssertionError("kelly_fraction must not determine live notional")

        with patch("risk.position_sizer.kelly_fraction", side_effect=forbidden):
            result = risk_budget_size(10_000, confidence=8)
        self.assertGreater(result, 0.0)

    def test_confidence_does_not_affect_size_without_empirical_data(self):
        equity = 20_000
        self.assertAlmostEqual(
            risk_budget_size(equity, confidence=9),
            risk_budget_size(equity, confidence=7),
            places=4,
        )

    def test_no_empirical_data_gives_neutral_conviction(self):
        equity = 10_000
        risk_usd = equity * config.RISK_PER_TRADE_PCT
        stop_pct = config.TRAILING_STOP_PCT / 100.0
        base = min(risk_usd / stop_pct, equity * config.MAX_POSITION_WEIGHT)
        result = risk_budget_size(equity, confidence=config.MIN_CONFIDENCE)
        self.assertAlmostEqual(result, base * 0.75, places=4)

    def test_empirical_win_rate_overrides_neutral_scale(self):
        equity = 10_000
        risk_usd = equity * config.RISK_PER_TRADE_PCT
        stop_pct = config.TRAILING_STOP_PCT / 100.0
        base = min(risk_usd / stop_pct, equity * config.MAX_POSITION_WEIGHT)
        with patch("risk.position_sizer._empirical_win_rate", return_value=0.60):
            result = risk_budget_size(equity, confidence=8)
        self.assertAlmostEqual(result, base * 1.0, places=4)

    def test_higher_empirical_win_rate_gives_larger_notional(self):
        with patch("risk.position_sizer._empirical_win_rate", return_value=0.65):
            full = risk_budget_size(10_000, confidence=8)
        with patch("risk.position_sizer._empirical_win_rate", return_value=0.40):
            smaller = risk_budget_size(10_000, confidence=8)
        self.assertGreater(full, smaller)

    def test_empirical_win_rate_scales_notional(self):
        with patch("risk.position_sizer._empirical_win_rate", return_value=0.65):
            full = risk_budget_size(10_000, confidence=8)
        with patch("risk.position_sizer._empirical_win_rate", return_value=0.40):
            smaller = risk_budget_size(10_000, confidence=8)
        self.assertGreater(full, smaller)

    def test_poor_empirical_win_rate_is_floored(self):
        with patch("risk.position_sizer._empirical_win_rate", return_value=0.10):
            result = risk_budget_size(10_000, confidence=8)
        self.assertGreater(result, 0.0)  # floor at 25% of base, never zero


class TestEmpiricalWinRate(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.stats_path = os.path.join(self.tmpdir, "signal_stats.json")
        self.patcher = patch("risk.position_sizer._SIGNAL_STATS_PATH", self.stats_path)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        import shutil

        self.addCleanup(shutil.rmtree, self.tmpdir)

    def _write_stats(self, data):
        with open(self.stats_path, "w") as f:
            json.dump(data, f)

    def test_load_signal_stats_missing_file_returns_empty(self):
        # Lines 27-28: FileNotFoundError → return {}
        from risk.position_sizer import _load_signal_stats

        # stats_path does not exist (setUp doesn't create it)
        result = _load_signal_stats()
        self.assertEqual(result, {})

    def test_load_signal_stats_corrupt_json_returns_empty(self):
        # Lines 27-28: json.JSONDecodeError → return {}
        from risk.position_sizer import _load_signal_stats

        with open(self.stats_path, "w") as f:
            f.write("{ not valid json {{")
        result = _load_signal_stats()
        self.assertEqual(result, {})

    def test_regime_bucket_with_enough_samples_used(self):
        # Line 44: regime bucket has >= _MIN_SAMPLE_SIZE (5) trades → return regime win rate
        from risk.position_sizer import _empirical_win_rate

        stats = {
            "momentum": {
                "trades": 10,
                "wins": 6,
                "losses": 4,
                "total_return_pct": 20.0,
                "by_regime": {
                    "BULL_TRENDING": {"trades": 5, "wins": 4, "losses": 1, "total_return_pct": 10.0}
                },
            }
        }
        self._write_stats(stats)
        result = _empirical_win_rate("momentum", "BULL_TRENDING")
        self.assertAlmostEqual(result, 4 / 5)

    def test_signal_level_fallback_when_regime_insufficient(self):
        # Line 47: regime bucket < 5 samples, signal-level has >= 5 → uses signal-level
        from risk.position_sizer import _empirical_win_rate

        stats = {
            "momentum": {
                "trades": 8,
                "wins": 5,
                "losses": 3,
                "total_return_pct": 15.0,
                "by_regime": {
                    "BULL_TRENDING": {"trades": 2, "wins": 1, "losses": 1, "total_return_pct": 2.0}
                },
            }
        }
        self._write_stats(stats)
        result = _empirical_win_rate("momentum", "BULL_TRENDING")
        self.assertAlmostEqual(result, 5 / 8)

    def test_returns_none_when_insufficient_samples(self):
        # Line 49: both regime and signal-level have < 5 samples → return None
        from risk.position_sizer import _empirical_win_rate

        stats = {
            "momentum": {
                "trades": 2,
                "wins": 1,
                "losses": 1,
                "total_return_pct": 3.0,
                "by_regime": {
                    "BULL_TRENDING": {"trades": 1, "wins": 1, "losses": 0, "total_return_pct": 5.0}
                },
            }
        }
        self._write_stats(stats)
        result = _empirical_win_rate("momentum", "BULL_TRENDING")
        self.assertIsNone(result)


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
