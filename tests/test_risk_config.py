"""Tests for RiskConfig dataclass and its integration with position sizer."""

import unittest
from unittest.mock import patch

from risk.position_sizer import kelly_fraction, risk_budget_size
from risk.risk_config import RiskConfig


class TestRiskConfigConstruction(unittest.TestCase):
    def test_direct_construction(self):
        rc = RiskConfig(
            stop_loss_pct=0.05,
            take_profit_pct=0.15,
            trailing_stop_pct=4.0,
            partial_profit_pct=8.0,
            max_hold_days=3,
        )
        self.assertEqual(rc.stop_loss_pct, 0.05)
        self.assertEqual(rc.take_profit_pct, 0.15)
        self.assertEqual(rc.trailing_stop_pct, 4.0)
        self.assertEqual(rc.partial_profit_pct, 8.0)
        self.assertEqual(rc.max_hold_days, 3)

    def test_frozen_immutable(self):
        rc = RiskConfig(0.05, 0.15, 4.0, 8.0, 3)
        with self.assertRaises(AttributeError):
            rc.stop_loss_pct = 0.10  # type: ignore[misc]

    def test_from_config_reads_live_values(self):
        with (
            patch("config.STOP_LOSS_PCT", 0.08),
            patch("config.TAKE_PROFIT_PCT", 0.20),
            patch("config.TRAILING_STOP_PCT", 5.0),
            patch("config.PARTIAL_PROFIT_PCT", 10.0),
            patch("config.MAX_HOLD_DAYS", 4),
        ):
            rc = RiskConfig.from_config()
        self.assertEqual(rc.stop_loss_pct, 0.08)
        self.assertEqual(rc.take_profit_pct, 0.20)
        self.assertEqual(rc.trailing_stop_pct, 5.0)
        self.assertEqual(rc.partial_profit_pct, 10.0)
        self.assertEqual(rc.max_hold_days, 4)


class TestKellyFractionWithRiskConfig(unittest.TestCase):
    def test_custom_risk_config_used(self):
        """Tighter stop / larger target should raise Kelly vs default."""
        rc_wide = RiskConfig(0.05, 0.20, 5.0, 10.0, 3)
        rc_tight = RiskConfig(0.15, 0.10, 5.0, 10.0, 3)
        with patch("risk.position_sizer._load_signal_stats", return_value={}):
            f_wide = kelly_fraction(7, risk_config=rc_wide)
            f_tight = kelly_fraction(7, risk_config=rc_tight)
        # Wide target/narrow stop → better b ratio → higher Kelly
        self.assertGreater(f_wide, f_tight)

    def test_none_risk_config_uses_config_defaults(self):
        with patch("risk.position_sizer._load_signal_stats", return_value={}):
            result_default = kelly_fraction(7)
            result_none = kelly_fraction(7, risk_config=None)
        self.assertAlmostEqual(result_default, result_none)


class TestRiskBudgetSizeWithRiskConfig(unittest.TestCase):
    def test_tighter_trailing_stop_increases_size(self):
        """Smaller trailing stop % → lower stop_pct denominator → larger base notional."""
        rc_tight = RiskConfig(0.08, 0.20, 2.0, 8.0, 4)
        rc_wide = RiskConfig(0.08, 0.20, 8.0, 8.0, 4)
        with patch("risk.position_sizer._load_signal_stats", return_value={}):
            size_tight = risk_budget_size(10_000, 7, risk_config=rc_tight)
            size_wide = risk_budget_size(10_000, 7, risk_config=rc_wide)
        self.assertGreater(size_tight, size_wide)

    def test_none_risk_config_uses_config_defaults(self):
        with patch("risk.position_sizer._load_signal_stats", return_value={}):
            result_default = risk_budget_size(10_000, 7)
            result_none = risk_budget_size(10_000, 7, risk_config=None)
        self.assertAlmostEqual(result_default, result_none)


class TestRegimeBlockedCanonical(unittest.TestCase):
    """Verify REGIME_BLOCKED in evaluator is the single source of truth."""

    def test_bear_day_blocks_momentum(self):
        from signals.evaluator import REGIME_BLOCKED

        self.assertIn("momentum", REGIME_BLOCKED["BEAR_DAY"])

    def test_bear_day_blocks_iv_compression(self):
        from signals.evaluator import REGIME_BLOCKED

        self.assertIn("iv_compression", REGIME_BLOCKED["BEAR_DAY"])

    def test_choppy_blocks_mean_reversion(self):
        from signals.evaluator import REGIME_BLOCKED

        self.assertIn("mean_reversion", REGIME_BLOCKED["CHOPPY"])

    def test_choppy_blocks_macd_crossover(self):
        from signals.evaluator import REGIME_BLOCKED

        self.assertIn("macd_crossover", REGIME_BLOCKED["CHOPPY"])

    def test_high_vol_blocks_breakout_52w(self):
        from signals.evaluator import REGIME_BLOCKED

        self.assertIn("breakout_52w", REGIME_BLOCKED["HIGH_VOL"])

    def test_scanner_uses_canonical_not_local(self):
        """stock_scanner.py must not define its own _LIVE_REGIME_BLOCKED."""
        import ast
        import pathlib

        src = pathlib.Path(
            "/Users/samchatterley/Development/Claude Test/InvestorBotHard"
            "/execution/stock_scanner.py"
        ).read_text()
        tree = ast.parse(src)
        names = {
            node.targets[0].id
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
        }
        self.assertNotIn("_LIVE_REGIME_BLOCKED", names)

    def test_engine_does_not_define_local_regime_blocked(self):
        """backtest/engine.py must not define its own _REGIME_BLOCKED."""
        import ast
        import pathlib

        src = pathlib.Path(
            "/Users/samchatterley/Development/Claude Test/InvestorBotHard/backtest/engine.py"
        ).read_text()
        tree = ast.parse(src)
        names = {
            node.targets[0].id
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
        }
        self.assertNotIn("_REGIME_BLOCKED", names)

    def test_range_reversion_blocked_in_choppy(self):
        # Blocked: WR 46%, p>0.05, n=52 in NEUTRAL_CHOP
        from signals.evaluator import REGIME_BLOCKED

        self.assertIn("range_reversion", REGIME_BLOCKED["CHOPPY"])

    def test_range_reversion_blocked_in_defensive_downtrend(self):
        # Blocked: WR 30%, avg -2.1% in DEFENSIVE_DOWNTREND (n=10)
        from signals.evaluator import REGIME_BLOCKED

        self.assertIn("range_reversion", REGIME_BLOCKED["DEFENSIVE_DOWNTREND"])

    def test_momentum_12_1_globally_disabled(self):
        # Globally disabled: WR 48%, avg -0.2%, n=97 in BULL_TREND; no edge in any regime
        from signals.evaluator import GLOBALLY_DISABLED

        self.assertIn("momentum_12_1", GLOBALLY_DISABLED)

    def test_iv_compression_blocked_in_neutral_chop(self):
        # Blocked: WR 51%, avg +0.0%, n=506 — doesn't clear 0.32% round-trip cost threshold
        from signals.evaluator import REGIME_BLOCKED

        self.assertIn("iv_compression", REGIME_BLOCKED["NEUTRAL_CHOP"])

    def test_rsi_divergence_not_blocked_in_neutral_chop(self):
        from signals.evaluator import REGIME_BLOCKED

        self.assertNotIn("rsi_divergence", REGIME_BLOCKED["NEUTRAL_CHOP"])

    def test_rsi_divergence_blocked_in_bull_trend(self):
        from signals.evaluator import REGIME_BLOCKED

        self.assertIn("rsi_divergence", REGIME_BLOCKED["BULL_TREND"])

    def test_rsi_divergence_blocked_in_stress_risk_off(self):
        from signals.evaluator import REGIME_BLOCKED

        self.assertIn("rsi_divergence", REGIME_BLOCKED["STRESS_RISK_OFF"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
