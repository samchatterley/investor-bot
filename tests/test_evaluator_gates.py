"""Tests for dynamic gates in signals/evaluator.py — premarket_gap_quality."""

import unittest


def _gap_and_go_snapshot(premarket_gap_retrace: bool = False) -> dict:
    """Return a snapshot that should fire gap_and_go when premarket_gap_retrace is False."""
    return {
        "gap_pct": 3.5,
        "close_above_open": True,
        "vol_ratio": 2.5,
        "adx": 30.0,
        "rsi_14": 55.0,
        "bb_pct": 0.5,
        "macd_diff": 0.1,
        "ema9_above_ema21": True,
        "ret_5d_pct": 2.0,
        "ret_10d_pct": 4.0,
        "price_vs_ema21_pct": 1.0,
        "price_vs_52w_high_pct": -5.0,
        "hv_rank": 0.5,
        "bb_squeeze": False,
        "bb_squeeze_days": 0,
        "is_inside_day": False,
        "spread_proxy_20d": 0.001,
        "weekly_trend_up": True,
        "weekly_rsi": 55.0,
        "calendar_month": 6,
        "premarket_gap_retrace": premarket_gap_retrace,
    }


class TestPremarketGapQualityGate(unittest.TestCase):
    def test_premarket_gap_retrace_suppresses_gap_and_go(self):
        """When premarket_gap_retrace=True, gap_and_go must not appear in signals."""
        from signals.evaluator import evaluate_signals

        snap = _gap_and_go_snapshot(premarket_gap_retrace=True)
        signals = evaluate_signals(snap)
        self.assertNotIn("gap_and_go", signals)

    def test_no_retrace_allows_gap_and_go(self):
        """When premarket_gap_retrace=False, gap_and_go fires normally."""
        from signals.evaluator import evaluate_signals

        snap = _gap_and_go_snapshot(premarket_gap_retrace=False)
        signals = evaluate_signals(snap)
        self.assertIn("gap_and_go", signals)

    def test_missing_field_defaults_to_no_suppression(self):
        """When premarket_gap_retrace is absent from snapshot, gap_and_go fires normally."""
        from signals.evaluator import evaluate_signals

        snap = _gap_and_go_snapshot()
        del snap["premarket_gap_retrace"]
        signals = evaluate_signals(snap)
        self.assertIn("gap_and_go", signals)

    def test_premarket_retrace_does_not_suppress_other_signals(self):
        """premarket_gap_retrace only gates gap_and_go, not momentum or other signals."""
        from signals.evaluator import evaluate_signals

        snap = _gap_and_go_snapshot(premarket_gap_retrace=True)
        # The snapshot also carries momentum characteristics (ema_up, macd_diff>0, ret_5d>0, adx>=20).
        signals = evaluate_signals(snap)
        # gap_and_go suppressed
        self.assertNotIn("gap_and_go", signals)
        # momentum should still fire (macd_crossover was retired)
        self.assertIn("momentum", signals)

    def test_lottery_pop_blocks_momentum_family(self):
        """A recent >=+10% single-day pop gates momentum + gap_and_go (MAX effect)."""
        from signals.evaluator import evaluate_signals

        snap = _gap_and_go_snapshot()
        snap["recent_lottery_pop"] = True
        signals = evaluate_signals(snap)
        self.assertNotIn("momentum", signals)
        self.assertNotIn("gap_and_go", signals)

    def test_no_lottery_pop_allows_momentum(self):
        """Without a recent pop the momentum family fires normally (control)."""
        from signals.evaluator import evaluate_signals

        snap = _gap_and_go_snapshot()
        snap["recent_lottery_pop"] = False
        signals = evaluate_signals(snap)
        self.assertIn("momentum", signals)


class TestSpreadProxyGateDirection(unittest.TestCase):
    """Finding 11: a missing spread_proxy_20d must read as WIDE (illiquid) at both evaluator sites —
    previously the execution gate defaulted 0.0 (fail-open) while the capitulation guardrail defaulted
    1.0 (fail-closed), so an absent field meant 'liquid' in one place and 'illiquid' in the other.
    gap_and_go is in _SPREAD_PROXY_GATED, so it's the probe for the execution-cost gate."""

    def test_liquid_spread_allows_gated_signal(self):
        from signals.evaluator import evaluate_signals

        snap = _gap_and_go_snapshot()
        snap["spread_proxy_20d"] = 0.0  # tight spread → liquid → not gated (control)
        self.assertIn("gap_and_go", evaluate_signals(snap))

    def test_wide_spread_gates_signal(self):
        from signals.evaluator import evaluate_signals

        snap = _gap_and_go_snapshot()
        snap["spread_proxy_20d"] = 0.05  # 5% spread ≫ spread_proxy_max → gated
        self.assertNotIn("gap_and_go", evaluate_signals(snap))

    def test_absent_spread_fails_closed_and_gates_signal(self):
        """The reconciliation: an absent field now behaves like a wide spread (fail-closed)."""
        from signals.evaluator import evaluate_signals

        snap = _gap_and_go_snapshot()
        del snap["spread_proxy_20d"]
        self.assertNotIn("gap_and_go", evaluate_signals(snap))
