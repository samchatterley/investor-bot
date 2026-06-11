import json
import os
import tempfile
import unittest
from unittest.mock import patch

import config
from risk.position_sizer import (
    SIGNAL_SHARPE_MULTIPLIER,
    amihud_size_scalar,
    atr_position_size,
    cofiring_boost,
    correlation_scalar,
    drawdown_scalar,
    get_max_positions,
    get_signal_size_multiplier,
    kelly_fraction,
    macro_scalar,
    momentum_quality_score,
    mqr_size_multiplier,
    nhl_scalar,
    risk_budget_size,
    seasonal_scalar,
    small_account_size,
    vol_of_vol_scalar,
)


class TestKellyFraction(unittest.TestCase):
    def setUp(self):
        # Isolate from live logs/signal_stats.json — these tests exercise
        # the LLM-confidence fallback path (no empirical data).
        self._patcher = patch("risk.position_sizer._SIGNAL_STATS_PATH", "/nonexistent/path")
        self._patcher.start()
        self.addCleanup(self._patcher.stop)

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
        # Without empirical data, conviction_scale is flat 0.75.
        # Patch signal stats so live logs/signal_stats.json doesn't interfere.
        equity = 10_000
        risk_usd = equity * config.RISK_PER_TRADE_PCT
        stop_pct = config.TRAILING_STOP_PCT / 100.0
        base = min(risk_usd / stop_pct, equity * config.MAX_POSITION_WEIGHT)
        with patch("risk.position_sizer._SIGNAL_STATS_PATH", "/nonexistent/path"):
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
        with patch("risk.position_sizer._SIGNAL_STATS_PATH", "/nonexistent/path"):
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

    def test_returns_none_when_signal_absent_from_stats(self):
        # Line 41: signal not in stats dict at all → return None immediately
        from risk.position_sizer import _empirical_win_rate

        self._write_stats({"other_signal": {"trades": 10, "wins": 6}})
        result = _empirical_win_rate("unknown_signal", "BULL_TREND")
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


class TestDrawdownScalar(unittest.TestCase):
    def _history(self, values: list[float]) -> list[dict]:
        return [{"account_after": {"portfolio_value": v}} for v in values]

    def test_empty_history_returns_one(self):
        self.assertEqual(drawdown_scalar([]), 1.0)

    def test_single_record_returns_one(self):
        self.assertEqual(drawdown_scalar(self._history([100_000])), 1.0)

    def test_no_drawdown_returns_one(self):
        history = self._history([100_000, 102_000, 105_000])
        self.assertEqual(drawdown_scalar(history), 1.0)

    def test_small_drawdown_below_threshold_returns_one(self):
        # 3% drawdown — below the 5% threshold
        history = self._history([100_000, 105_000, 101_850])
        self.assertEqual(drawdown_scalar(history), 1.0)

    def test_exactly_at_threshold_returns_half(self):
        # exactly -5% from peak
        history = self._history([100_000, 100_000, 95_000])
        self.assertEqual(drawdown_scalar(history), 0.5)

    def test_beyond_threshold_returns_half(self):
        # -10% drawdown
        history = self._history([100_000, 110_000, 99_000])
        self.assertEqual(drawdown_scalar(history), 0.5)

    def test_recovery_above_threshold_returns_one(self):
        # Was down, now recovered to within 5%
        history = self._history([100_000, 110_000, 90_000, 105_000])
        self.assertEqual(drawdown_scalar(history), 1.0)

    def test_implausible_values_filtered_out(self):
        # Records below $1000 are ignored; only the plausible ones count
        history = self._history([100_000, 100_000, 500])
        # Only two plausible values: 100_000 and 100_000 — no drawdown → 1.0
        self.assertEqual(drawdown_scalar(history), 1.0)

    def test_only_one_plausible_value_returns_one(self):
        history = self._history([500, 100_000])
        # Only one plausible value → can't compute drawdown
        self.assertEqual(drawdown_scalar(history), 1.0)

    def test_all_implausible_returns_one(self):
        history = self._history([100, 200, 300])
        self.assertEqual(drawdown_scalar(history), 1.0)

    def test_missing_key_returns_one(self):
        # Malformed records with 2 entries pass the length check but KeyError in values extraction
        bad_history = [{"wrong_key": {}}, {"wrong_key": {}}]
        self.assertEqual(drawdown_scalar(bad_history), 1.0)


class TestSignalSharpeMultiplier(unittest.TestCase):
    def test_iv_compression_is_highest(self):
        self.assertEqual(SIGNAL_SHARPE_MULTIPLIER["iv_compression"], 1.5)

    def test_breakout_52w_is_zero(self):
        self.assertEqual(SIGNAL_SHARPE_MULTIPLIER["breakout_52w"], 0.0)

    def test_rsi_divergence_is_zero(self):
        self.assertEqual(SIGNAL_SHARPE_MULTIPLIER["rsi_divergence"], 0.0)

    def test_vix_fear_reversion_is_zero(self):
        # vix_fear_reversion is globally disabled; zeroed to prevent AI from inflating size
        self.assertEqual(SIGNAL_SHARPE_MULTIPLIER["vix_fear_reversion"], 0.0)

    def test_range_reversion_is_baseline(self):
        self.assertEqual(SIGNAL_SHARPE_MULTIPLIER["range_reversion"], 1.0)


class TestGetSignalSizeMultiplier(unittest.TestCase):
    def test_known_signal_returns_correct_multiplier(self):
        self.assertEqual(get_signal_size_multiplier("iv_compression"), 1.5)

    def test_pead_returns_correct_multiplier(self):
        self.assertEqual(get_signal_size_multiplier("pead"), 1.3)

    def test_unknown_signal_returns_one(self):
        self.assertEqual(get_signal_size_multiplier("brand_new_signal"), 1.0)

    def test_disabled_signal_breakout_52w_returns_zero(self):
        self.assertEqual(get_signal_size_multiplier("breakout_52w"), 0.0)

    def test_disabled_signal_rsi_divergence_returns_zero(self):
        self.assertEqual(get_signal_size_multiplier("rsi_divergence"), 0.0)


class TestAtrPositionSize(unittest.TestCase):
    def test_normal_case_computes_correctly(self):
        # equity=10000, risk_pct=0.01, atr_pct=25.0
        # size = (10000 * 0.01) / (25.0 / 100) = 100 / 0.25 = 400 — below cap
        result = atr_position_size(10_000, 25.0, risk_pct=0.01)
        self.assertAlmostEqual(result, 400.0)

    def test_zero_atr_pct_returns_zero(self):
        self.assertEqual(atr_position_size(10_000, 0.0), 0.0)

    def test_negative_atr_pct_returns_zero(self):
        self.assertEqual(atr_position_size(10_000, -1.0), 0.0)

    def test_zero_equity_returns_zero(self):
        self.assertEqual(atr_position_size(0.0, 3.5), 0.0)

    def test_negative_equity_returns_zero(self):
        self.assertEqual(atr_position_size(-5_000, 3.5), 0.0)

    def test_caps_at_max_position_weight(self):
        # Very small atr_pct → huge uncapped size; should be capped at equity * MAX_POSITION_WEIGHT
        equity = 10_000
        result = atr_position_size(equity, 0.001, risk_pct=0.10)
        self.assertAlmostEqual(result, equity * config.MAX_POSITION_WEIGHT)

    def test_custom_risk_pct_overrides_default(self):
        equity = 10_000
        atr_pct = 2.0
        custom_risk = 0.005
        expected = min(
            (equity * custom_risk) / (atr_pct / 100),
            equity * config.MAX_POSITION_WEIGHT,
        )
        result = atr_position_size(equity, atr_pct, risk_pct=custom_risk)
        self.assertAlmostEqual(result, expected)

    def test_default_risk_pct_uses_risk_per_trade_pct(self):
        equity = 10_000
        atr_pct = 2.0
        expected = min(
            (equity * config.RISK_PER_TRADE_PCT) / (atr_pct / 100),
            equity * config.MAX_POSITION_WEIGHT,
        )
        result = atr_position_size(equity, atr_pct)
        self.assertAlmostEqual(result, expected)


class TestCofiringBoost(unittest.TestCase):
    def test_zero_signals_returns_one(self):
        self.assertEqual(cofiring_boost(0), 1.0)

    def test_one_signal_returns_one(self):
        self.assertEqual(cofiring_boost(1), 1.0)

    def test_two_signals_returns_one_point_five(self):
        self.assertEqual(cofiring_boost(2), 1.5)

    def test_five_signals_still_capped_at_one_point_five(self):
        self.assertEqual(cofiring_boost(5), 1.5)


class TestAmihudSizeScalar(unittest.TestCase):
    def test_liquid_symbol_returns_one(self):
        self.assertEqual(amihud_size_scalar(False), 1.0)

    def test_illiquid_symbol_returns_half(self):
        self.assertEqual(amihud_size_scalar(True), 0.5)

    def test_return_type_is_float(self):
        self.assertIsInstance(amihud_size_scalar(True), float)
        self.assertIsInstance(amihud_size_scalar(False), float)


class TestMomentumQualityScore(unittest.TestCase):
    def test_empty_snapshot_scores_zero(self):
        self.assertEqual(momentum_quality_score({}), 0)

    def test_high_rs_rank_scores_one(self):
        self.assertEqual(momentum_quality_score({"rs_rank_pct": 60.0}), 1)

    def test_rs_rank_below_threshold_scores_zero(self):
        self.assertEqual(momentum_quality_score({"rs_rank_pct": 59.9}), 0)

    def test_pead_candidate_scores_one(self):
        self.assertEqual(momentum_quality_score({"pead_candidate": True}), 1)

    def test_pead_false_scores_zero(self):
        self.assertEqual(momentum_quality_score({"pead_candidate": False}), 0)

    def test_quality_proxy_positive_roe_and_margin_scores_one(self):
        self.assertEqual(momentum_quality_score({"roe": 0.15, "profit_margin": 0.10}), 1)

    def test_negative_roe_quality_zero(self):
        self.assertEqual(momentum_quality_score({"roe": -0.05, "profit_margin": 0.10}), 0)

    def test_negative_margin_quality_zero(self):
        self.assertEqual(momentum_quality_score({"roe": 0.10, "profit_margin": -0.02}), 0)

    def test_all_three_components_scores_three(self):
        snap = {"rs_rank_pct": 75.0, "pead_candidate": True, "roe": 0.20, "profit_margin": 0.15}
        self.assertEqual(momentum_quality_score(snap), 3)

    def test_two_components_scores_two(self):
        snap = {"rs_rank_pct": 80.0, "pead_candidate": True}
        self.assertEqual(momentum_quality_score(snap), 2)

    def test_mqr_multiplier_one_for_score_below_three(self):
        self.assertEqual(mqr_size_multiplier(0), 1.0)
        self.assertEqual(mqr_size_multiplier(1), 1.0)
        self.assertEqual(mqr_size_multiplier(2), 1.0)

    def test_mqr_multiplier_one_point_five_for_score_three(self):
        self.assertEqual(mqr_size_multiplier(3), 1.5)


# ── Batch 2: vol_of_vol_scalar ────────────────────────────────────────────────


class TestVolOfVolScalar(unittest.TestCase):
    """vol_of_vol_scalar: position-size multiplier based on VIX volatility-of-volatility."""

    def test_high_vov_reduces_size(self):
        self.assertAlmostEqual(vol_of_vol_scalar(5.0), 0.7)

    def test_low_vov_boosts_size(self):
        self.assertAlmostEqual(vol_of_vol_scalar(0.5), 1.2)

    def test_mid_vov_returns_one(self):
        self.assertAlmostEqual(vol_of_vol_scalar(2.0), 1.0)

    def test_none_returns_one(self):
        self.assertAlmostEqual(vol_of_vol_scalar(None), 1.0)

    def test_exact_reduce_threshold_returns_one(self):
        self.assertAlmostEqual(vol_of_vol_scalar(3.5), 1.0)

    def test_just_above_reduce_threshold_reduces(self):
        self.assertAlmostEqual(vol_of_vol_scalar(3.51), 0.7)

    def test_exact_boost_threshold_returns_one(self):
        self.assertAlmostEqual(vol_of_vol_scalar(1.0), 1.0)

    def test_just_below_boost_threshold_boosts(self):
        self.assertAlmostEqual(vol_of_vol_scalar(0.99), 1.2)

    def test_return_type_is_float(self):
        self.assertIsInstance(vol_of_vol_scalar(2.0), float)


def _neutral_ctx(**overrides) -> dict:
    """Return a fully-neutral seasonal context with optional field overrides."""
    base = {
        "turn_of_month": False,
        "opex_week": False,
        "post_opex": False,
        "halloween_bullish": False,
        "quarter_end_dressing": False,
        "pre_holiday": False,
    }
    base.update(overrides)
    return base


class TestSeasonalScalar(unittest.TestCase):
    def _patch(self, ctx: dict):
        return patch("risk.macro_calendar.get_seasonal_context", return_value=ctx)

    def test_all_neutral_bearish_season_gives_0_90(self):
        ctx = _neutral_ctx(halloween_bullish=False)
        with self._patch(ctx):
            self.assertAlmostEqual(seasonal_scalar("momentum"), 0.90)

    def test_halloween_bullish_gives_1_10(self):
        ctx = _neutral_ctx(halloween_bullish=True)
        with self._patch(ctx):
            self.assertAlmostEqual(seasonal_scalar("momentum"), 1.10)

    def test_opex_week_dampens_gap_and_go(self):
        ctx = _neutral_ctx(halloween_bullish=True, opex_week=True)
        with self._patch(ctx):
            result = seasonal_scalar("gap_and_go")
        self.assertAlmostEqual(result, 1.10 * 0.70)

    def test_opex_week_does_not_dampen_pead(self):
        ctx = _neutral_ctx(halloween_bullish=True, opex_week=True)
        with self._patch(ctx):
            result = seasonal_scalar("pead")
        self.assertAlmostEqual(result, 1.10)

    def test_post_opex_boosts_all_signals(self):
        ctx = _neutral_ctx(halloween_bullish=True, post_opex=True)
        with self._patch(ctx):
            result = seasonal_scalar("pead")
        self.assertAlmostEqual(result, 1.10 * 1.10)

    def test_turn_of_month_boosts(self):
        ctx = _neutral_ctx(halloween_bullish=True, turn_of_month=True)
        with self._patch(ctx):
            result = seasonal_scalar("mean_reversion")
        self.assertAlmostEqual(result, 1.10 * 1.05)

    def test_quarter_end_boosts_momentum(self):
        ctx = _neutral_ctx(halloween_bullish=True, quarter_end_dressing=True)
        with self._patch(ctx):
            result = seasonal_scalar("momentum")
        self.assertAlmostEqual(result, 1.10 * 1.10)

    def test_quarter_end_does_not_boost_gap_and_go(self):
        ctx = _neutral_ctx(halloween_bullish=True, quarter_end_dressing=True)
        with self._patch(ctx):
            result = seasonal_scalar("gap_and_go")
        self.assertAlmostEqual(result, 1.10)

    def test_pre_holiday_boosts_all_signals(self):
        ctx = _neutral_ctx(halloween_bullish=True, pre_holiday=True)
        with self._patch(ctx):
            result = seasonal_scalar("mean_reversion")
        self.assertAlmostEqual(result, 1.10 * 1.05)

    def test_result_clamped_at_1_25(self):
        ctx = _neutral_ctx(
            halloween_bullish=True,
            post_opex=True,
            turn_of_month=True,
            quarter_end_dressing=True,
            pre_holiday=True,
        )
        with self._patch(ctx):
            result = seasonal_scalar("momentum")
        self.assertLessEqual(result, 1.25)

    def test_result_clamped_at_0_70(self):
        ctx = _neutral_ctx(halloween_bullish=False, opex_week=True)
        with self._patch(ctx):
            result = seasonal_scalar("gap_and_go")
        self.assertGreaterEqual(result, 0.70)

    def test_result_is_float(self):
        ctx = _neutral_ctx(halloween_bullish=True)
        with self._patch(ctx):
            self.assertIsInstance(seasonal_scalar("momentum"), float)


class TestMacroScalar(unittest.TestCase):
    def _neutral(self) -> dict:
        return {
            "macro_yield_curve": 0.5,
            "macro_yield_curve_inverted_days": 0,
            "macro_copper_gold_positive": False,
            "macro_usd_strong": False,
            "macro_pmi_expanding": False,
        }

    def test_neutral_flags_return_one(self):
        self.assertEqual(macro_scalar(self._neutral(), "momentum"), 1.0)

    def test_recession_scalar_when_deep_inversion(self):
        snap = {**self._neutral(), "macro_yield_curve": -0.5, "macro_yield_curve_inverted_days": 60}
        result = macro_scalar(snap, "mean_reversion")
        self.assertLess(result, 1.0)

    def test_expansion_boost_for_cyclical_with_steep_curve(self):
        snap = {**self._neutral(), "macro_yield_curve": 1.8}
        result = macro_scalar(snap, "momentum")
        self.assertGreater(result, 1.0)

    def test_expansion_boost_not_applied_to_non_cyclical(self):
        snap = {**self._neutral(), "macro_yield_curve": 1.8}
        result = macro_scalar(snap, "mean_reversion")
        self.assertEqual(result, 1.0)

    def test_copper_gold_positive_boosts_cyclicals(self):
        snap = {**self._neutral(), "macro_copper_gold_positive": True}
        result = macro_scalar(snap, "momentum")
        self.assertGreater(result, 1.0)

    def test_copper_gold_positive_no_boost_for_non_cyclical(self):
        snap = {**self._neutral(), "macro_copper_gold_positive": True}
        result = macro_scalar(snap, "mean_reversion")
        self.assertEqual(result, 1.0)

    def test_usd_strong_reduces_all(self):
        snap = {**self._neutral(), "macro_usd_strong": True}
        result = macro_scalar(snap, "momentum")
        self.assertLess(result, 1.0)

    def test_pmi_expanding_boosts_cyclicals(self):
        snap = {**self._neutral(), "macro_pmi_expanding": True}
        result = macro_scalar(snap, "momentum")
        self.assertGreater(result, 1.0)

    def test_pmi_expanding_no_boost_for_non_cyclical(self):
        snap = {**self._neutral(), "macro_pmi_expanding": True}
        result = macro_scalar(snap, "mean_reversion")
        self.assertEqual(result, 1.0)

    def test_clamp_at_max_125(self):
        snap = {
            "macro_yield_curve": 2.0,
            "macro_yield_curve_inverted_days": 0,
            "macro_copper_gold_positive": True,
            "macro_usd_strong": False,
            "macro_pmi_expanding": True,
        }
        result = macro_scalar(snap, "momentum")
        self.assertLessEqual(result, 1.25)

    def test_clamp_at_min_070(self):
        snap = {
            "macro_yield_curve": -0.5,
            "macro_yield_curve_inverted_days": 65,
            "macro_copper_gold_positive": False,
            "macro_usd_strong": True,
            "macro_pmi_expanding": False,
        }
        result = macro_scalar(snap, "momentum")
        self.assertGreaterEqual(result, 0.70)

    def test_missing_keys_default_to_neutral(self):
        result = macro_scalar({}, "momentum")
        self.assertEqual(result, 1.0)


class TestSmallAccountSize(unittest.TestCase):
    def test_zero_portfolio_returns_zero(self):
        self.assertEqual(small_account_size(0), 0.0)

    def test_negative_portfolio_returns_zero(self):
        self.assertEqual(small_account_size(-50), 0.0)

    def test_small_portfolio_floored_at_40(self):
        # 50 * 0.8 / 2 = 20 < 40 → floor
        self.assertEqual(small_account_size(50), 40.0)

    def test_large_portfolio_capped_at_max_single_order(self):
        # 500 * 0.8 / 2 = 200 > 55 → cap
        self.assertEqual(small_account_size(500), 55.0)

    def test_medium_portfolio_returns_computed_value(self):
        # 120 * 0.8 / 2 = 48; 40 <= 48 <= 55 → 48
        self.assertAlmostEqual(small_account_size(120), 48.0)

    def test_custom_max_single_order(self):
        # 500 * 0.8 / 2 = 200 > 100 → cap at custom max
        self.assertEqual(small_account_size(500, max_single_order=100.0), 100.0)


class TestCorrelationScalar(unittest.TestCase):
    def test_high_corr_dampens_size(self):
        self.assertAlmostEqual(correlation_scalar(0.80), 0.85)

    def test_low_corr_boosts_size(self):
        self.assertAlmostEqual(correlation_scalar(0.20), 1.10)

    def test_neutral_corr_returns_one(self):
        self.assertAlmostEqual(correlation_scalar(0.55), 1.0)

    def test_none_returns_one(self):
        self.assertAlmostEqual(correlation_scalar(None), 1.0)

    def test_exactly_at_high_threshold_returns_one(self):
        # 0.75 is NOT > 0.75, so no dampening
        self.assertAlmostEqual(correlation_scalar(0.75), 1.0)

    def test_exactly_at_low_threshold_returns_one(self):
        # 0.35 is NOT < 0.35, so no boost
        self.assertAlmostEqual(correlation_scalar(0.35), 1.0)

    def test_just_above_high_threshold_dampens(self):
        self.assertAlmostEqual(correlation_scalar(0.76), 0.85)

    def test_just_below_low_threshold_boosts(self):
        self.assertAlmostEqual(correlation_scalar(0.34), 1.10)


class TestNHLScalar(unittest.TestCase):
    def test_high_nhl_boosts_size(self):
        self.assertAlmostEqual(nhl_scalar(3.0), 1.10)

    def test_low_nhl_dampens_size(self):
        self.assertAlmostEqual(nhl_scalar(0.3), 0.80)

    def test_neutral_nhl_returns_one(self):
        self.assertAlmostEqual(nhl_scalar(1.0), 1.0)

    def test_none_returns_one(self):
        self.assertAlmostEqual(nhl_scalar(None), 1.0)

    def test_exactly_at_expansion_threshold_returns_one(self):
        # 2.0 is NOT > 2.0, so no boost
        self.assertAlmostEqual(nhl_scalar(2.0), 1.0)

    def test_exactly_at_contraction_threshold_returns_one(self):
        # 0.5 is NOT < 0.5, so no dampening
        self.assertAlmostEqual(nhl_scalar(0.5), 1.0)

    def test_just_above_expansion_threshold_boosts(self):
        self.assertAlmostEqual(nhl_scalar(2.1), 1.10)

    def test_just_below_contraction_threshold_dampens(self):
        self.assertAlmostEqual(nhl_scalar(0.4), 0.80)
