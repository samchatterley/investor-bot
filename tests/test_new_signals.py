"""Tests for all new signals and gates added in v1.97.

Covers:
- Fundamental quality gates (altman_z, piotroski, forward_pe, gross_margin, accruals)
- Market microstructure gates (nhl_ratio, sector_correlation, earnings_yield_vs_bonds, aaii)
- New long signals (activist_13d, guidance_raise, iv_vs_rv_spread, fcf_yield, options_skew,
  unusual_options, put_call_contrarian, squeeze_setup, squeeze_momentum, short_interest_trend,
  analyst_upgrade, aaii_extreme_fear, fear_greed_extreme_fear, sector_pair_mr, google_trends)
- New short signals (altman_distress, piotroski_distress, gross_margin_deterioration,
  accruals_quality, lockup_expiry, analyst_downgrade)
"""


# ── shared snapshot helpers ────────────────────────────────────────────────────


def _base_snap(**kwargs) -> dict:
    """Return a minimal snapshot that doesn't fire any signal by default."""
    snap = {
        "rsi_14": 55.0,
        "bb_pct": 0.5,
        "vol_ratio": 1.0,
        "macd_diff": 0.0,
        "macd_crossed_up": False,
        "ema9_above_ema21": True,
        "adx": 25.0,
        "ret_5d_pct": 0.5,
        "ret_10d_pct": 1.0,
        "price_vs_ema21_pct": 0.0,
        "price_vs_52w_high_pct": -10.0,
        "hv_rank": 0.5,
        "bb_squeeze": False,
        "bb_squeeze_days": 0,
        "is_inside_day": False,
        "gap_pct": 0.0,
        "close_above_open": False,
        "spread_proxy_20d": 0.001,
        "calendar_month": 6,
    }
    snap.update(kwargs)
    return snap


def _momentum_snap(**kwargs) -> dict:
    """Snapshot that fires momentum when gates allow it."""
    snap = _base_snap(
        ema9_above_ema21=True,
        macd_diff=0.5,
        ret_5d_pct=2.0,
        vol_ratio=1.5,
        adx=25.0,
    )
    snap.update(kwargs)
    return snap


# ── Fundamental quality gates ─────────────────────────────────────────────────


class TestAltmanZGate:
    def test_distress_blocks_momentum(self):
        """altman_z < 1.1 blocks momentum/breakout trend signals."""
        from signals.evaluator import evaluate_signals

        snap = _momentum_snap(altman_z=0.8)
        signals = evaluate_signals(snap)
        assert "momentum" not in signals

    def test_distress_blocks_gap_and_go(self):
        """altman_z < 1.1 blocks gap_and_go."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(
            altman_z=0.5,
            gap_pct=4.0,
            close_above_open=True,
            vol_ratio=2.5,
            adx=25.0,
            ema9_above_ema21=True,
        )
        signals = evaluate_signals(snap)
        assert "gap_and_go" not in signals

    def test_distress_allows_mean_reversion(self):
        """altman_z < 1.1 does NOT block mean_reversion (counter-cyclical)."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(altman_z=0.5, rsi_14=28.0, bb_pct=0.05, vol_ratio=1.5)
        signals = evaluate_signals(snap)
        assert "mean_reversion" in signals

    def test_safe_zone_allows_all_signals(self):
        """altman_z > 2.6 doesn't block any signals."""
        from signals.evaluator import evaluate_signals

        snap = _momentum_snap(altman_z=3.5)
        signals = evaluate_signals(snap)
        assert "momentum" in signals

    def test_none_altman_z_is_neutral(self):
        """Missing altman_z field doesn't trigger any gate."""
        from signals.evaluator import evaluate_signals

        snap = _momentum_snap()  # no altman_z key
        signals = evaluate_signals(snap)
        assert "momentum" in signals

    def test_grey_zone_allows_momentum(self):
        """altman_z between 1.1 and 2.6 (grey zone) doesn't trigger distress gate."""
        from signals.evaluator import evaluate_signals

        snap = _momentum_snap(altman_z=1.8)
        signals = evaluate_signals(snap)
        assert "momentum" in signals


class TestPiotroskiGate:
    def test_low_piotroski_blocks_pead(self):
        """piotroski_f < 3 blocks pead."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(
            piotroski_f=2,
            pead_candidate=True,
            ret_5d_pct=2.0,
        )
        signals = evaluate_signals(snap)
        assert "pead" not in signals

    def test_adequate_piotroski_allows_pead(self):
        """piotroski_f >= 3 allows pead."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(
            piotroski_f=5,
            pead_candidate=True,
            ret_5d_pct=2.0,
        )
        signals = evaluate_signals(snap)
        assert "pead" in signals

    def test_none_piotroski_is_neutral(self):
        """Missing piotroski_f doesn't block pead."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(pead_candidate=True, ret_5d_pct=2.0)
        signals = evaluate_signals(snap)
        assert "pead" in signals


class TestForwardPeGate:
    def test_high_pe_blocks_momentum(self):
        """forward_pe > 60 blocks momentum signals."""
        from signals.evaluator import evaluate_signals

        snap = _momentum_snap(forward_pe=75.0)
        signals = evaluate_signals(snap)
        assert "momentum" not in signals

    def test_reasonable_pe_allows_momentum(self):
        """forward_pe <= 60 doesn't block momentum."""
        from signals.evaluator import evaluate_signals

        snap = _momentum_snap(forward_pe=20.0)
        signals = evaluate_signals(snap)
        assert "momentum" in signals

    def test_none_pe_is_neutral(self):
        """Missing forward_pe doesn't block momentum."""
        from signals.evaluator import evaluate_signals

        snap = _momentum_snap()
        signals = evaluate_signals(snap)
        assert "momentum" in signals


class TestGrossMarginGate:
    def test_gm_deterioration_blocks_pead(self):
        """gross_margin_trend < -0.03 blocks pead."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(
            gross_margin_trend=-0.05,
            pead_candidate=True,
            ret_5d_pct=2.0,
        )
        signals = evaluate_signals(snap)
        assert "pead" not in signals

    def test_gm_stable_allows_pead(self):
        """gross_margin_trend >= -0.03 allows pead."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(
            gross_margin_trend=0.01,
            pead_candidate=True,
            ret_5d_pct=2.0,
        )
        signals = evaluate_signals(snap)
        assert "pead" in signals

    def test_none_gm_trend_is_neutral(self):
        """Missing gross_margin_trend doesn't block pead."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(pead_candidate=True, ret_5d_pct=2.0)
        signals = evaluate_signals(snap)
        assert "pead" in signals


class TestAccrualsGate:
    def test_high_accruals_blocks_momentum(self):
        """accruals_ratio > 0.10 blocks momentum/breakout signals."""
        from signals.evaluator import evaluate_signals

        snap = _momentum_snap(accruals_ratio=0.15)
        signals = evaluate_signals(snap)
        assert "momentum" not in signals

    def test_low_accruals_allows_momentum(self):
        """accruals_ratio <= 0.10 doesn't block momentum."""
        from signals.evaluator import evaluate_signals

        snap = _momentum_snap(accruals_ratio=0.05)
        signals = evaluate_signals(snap)
        assert "momentum" in signals

    def test_none_accruals_is_neutral(self):
        """Missing accruals_ratio doesn't block."""
        from signals.evaluator import evaluate_signals

        snap = _momentum_snap()
        signals = evaluate_signals(snap)
        assert "momentum" in signals


# ── Market microstructure gates ───────────────────────────────────────────────


class TestNHLRatioGate:
    def test_weak_breadth_blocks_momentum(self):
        """nhl_ratio < 0.5 blocks momentum (weak breadth — no rising tide)."""
        from signals.evaluator import evaluate_signals

        snap = _momentum_snap(nhl_ratio=0.3)
        signals = evaluate_signals(snap)
        assert "momentum" not in signals

    def test_healthy_breadth_allows_momentum(self):
        """nhl_ratio >= 0.5 allows momentum."""
        from signals.evaluator import evaluate_signals

        snap = _momentum_snap(nhl_ratio=1.5)
        signals = evaluate_signals(snap)
        assert "momentum" in signals

    def test_none_nhl_ratio_is_neutral(self):
        """Missing nhl_ratio doesn't block momentum."""
        from signals.evaluator import evaluate_signals

        snap = _momentum_snap()
        signals = evaluate_signals(snap)
        assert "momentum" in signals


class TestSectorCorrelationGate:
    def test_high_correlation_blocks_momentum(self):
        """sector_correlation_20d > 0.75 blocks momentum, breakout_52w, bb_squeeze."""
        from signals.evaluator import evaluate_signals

        snap = _momentum_snap(sector_correlation_20d=0.85)
        signals = evaluate_signals(snap)
        assert "momentum" not in signals

    def test_low_correlation_allows_momentum(self):
        """sector_correlation_20d <= 0.75 doesn't block momentum."""
        from signals.evaluator import evaluate_signals

        snap = _momentum_snap(sector_correlation_20d=0.40)
        signals = evaluate_signals(snap)
        assert "momentum" in signals

    def test_none_correlation_is_neutral(self):
        """Missing sector_correlation_20d doesn't block."""
        from signals.evaluator import evaluate_signals

        snap = _momentum_snap()
        signals = evaluate_signals(snap)
        assert "momentum" in signals


class TestEarningsYieldGate:
    def test_low_erp_blocks_momentum(self):
        """ERP < 1% (low forward earnings yield vs bonds) blocks momentum."""
        from signals.evaluator import evaluate_signals

        # forward_pe=25 → earnings yield = 4%; 10y_yield=4.5% → ERP = -0.5% < 1%
        snap = _momentum_snap(forward_pe=25.0, macro_10y_yield=4.5)
        signals = evaluate_signals(snap)
        assert "momentum" not in signals

    def test_high_erp_allows_momentum(self):
        """ERP > 1% allows momentum."""
        from signals.evaluator import evaluate_signals

        # forward_pe=15 → earnings yield = 6.67%; 10y_yield=3.0% → ERP = 3.67%
        snap = _momentum_snap(forward_pe=15.0, macro_10y_yield=3.0)
        signals = evaluate_signals(snap)
        assert "momentum" in signals

    def test_missing_either_field_skips_gate(self):
        """Missing forward_pe or macro_10y_yield skips ERP gate."""
        from signals.evaluator import evaluate_signals

        snap = _momentum_snap(macro_10y_yield=5.0)  # no forward_pe
        signals = evaluate_signals(snap)
        assert "momentum" in signals

        snap2 = _momentum_snap(forward_pe=15.0)  # no 10y_yield
        signals2 = evaluate_signals(snap2)
        assert "momentum" in signals2

    def test_zero_forward_pe_skips_gate(self):
        """forward_pe=0 skips ERP gate (division by zero guard)."""
        from signals.evaluator import evaluate_signals

        snap = _momentum_snap(forward_pe=0.0, macro_10y_yield=4.0)
        signals = evaluate_signals(snap)
        assert "momentum" in signals


class TestAaiiBullsGate:
    def test_excessive_bulls_blocks_gap_and_go(self):
        """aaii_excessive_bulls=True blocks gap_and_go and momentum."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(
            aaii_excessive_bulls=True,
            gap_pct=4.0,
            close_above_open=True,
            vol_ratio=2.5,
            adx=25.0,
            ema9_above_ema21=True,
        )
        signals = evaluate_signals(snap)
        assert "gap_and_go" not in signals

    def test_no_excessive_bulls_allows_gap_and_go(self):
        """aaii_excessive_bulls=False doesn't block gap_and_go."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(
            aaii_excessive_bulls=False,
            gap_pct=4.0,
            close_above_open=True,
            vol_ratio=2.5,
            adx=25.0,
            ema9_above_ema21=True,
        )
        signals = evaluate_signals(snap)
        assert "gap_and_go" in signals

    def test_missing_aaii_field_is_neutral(self):
        """Missing aaii_excessive_bulls field is treated as False."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(
            gap_pct=4.0,
            close_above_open=True,
            vol_ratio=2.5,
            adx=25.0,
            ema9_above_ema21=True,
        )
        signals = evaluate_signals(snap)
        assert "gap_and_go" in signals


# ── New long signals ──────────────────────────────────────────────────────────


class TestActivist13dSignal:
    def test_activist_filing_fires_activist_13d(self):
        """activist_filing=True fires activist_13d_signal."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(activist_filing=True)
        signals = evaluate_signals(snap)
        assert "activist_13d_signal" in signals

    def test_no_activist_filing_no_signal(self):
        """activist_filing=False does not fire activist_13d_signal."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(activist_filing=False)
        signals = evaluate_signals(snap)
        assert "activist_13d_signal" not in signals

    def test_activist_13d_blocked_suppresses(self):
        """Explicitly blocking activist_13d_signal suppresses it."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(activist_filing=True)
        signals = evaluate_signals(snap, blocked=frozenset({"activist_13d_signal"}))
        assert "activist_13d_signal" not in signals


class TestGuidanceRaiseSignal:
    def test_guidance_positive_fires_guidance_raise(self):
        """guidance_positive=True fires guidance_raise_signal (even without ret_5d > 0)."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(guidance_positive=True, ret_5d_pct=-0.5)
        signals = evaluate_signals(snap)
        assert "guidance_raise_signal" in signals

    def test_no_guidance_no_signal(self):
        """guidance_positive=False does not fire guidance_raise_signal."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(guidance_positive=False)
        signals = evaluate_signals(snap)
        assert "guidance_raise_signal" not in signals

    def test_guidance_raise_blocked_suppresses(self):
        """Blocking guidance_raise_signal suppresses it."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(guidance_positive=True)
        signals = evaluate_signals(snap, blocked=frozenset({"guidance_raise_signal"}))
        assert "guidance_raise_signal" not in signals

    def test_guidance_positive_also_fires_pead_when_price_confirms(self):
        """When guidance_positive=True AND ret_5d>0, both pead and guidance_raise_signal fire."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(guidance_positive=True, ret_5d_pct=2.0)
        signals = evaluate_signals(snap)
        assert "guidance_raise_signal" in signals
        assert "pead" in signals


class TestIvVsRvSpread:
    def test_cheap_iv_rv_fires_signal(self):
        """iv_rv_spread < 0.70 with ema_up fires iv_vs_rv_spread."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(
            iv_rv_spread=0.55,
            ema9_above_ema21=True,
            vol_ratio=1.5,
        )
        signals = evaluate_signals(snap)
        assert "iv_vs_rv_spread" in signals

    def test_expensive_iv_rv_no_signal(self):
        """iv_rv_spread >= 0.70 does not fire iv_vs_rv_spread."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(
            iv_rv_spread=0.90,
            ema9_above_ema21=True,
            vol_ratio=1.5,
        )
        signals = evaluate_signals(snap)
        assert "iv_vs_rv_spread" not in signals

    def test_none_iv_rv_spread_skips(self):
        """Missing iv_rv_spread field doesn't fire signal."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(ema9_above_ema21=True, vol_ratio=1.5)
        signals = evaluate_signals(snap)
        assert "iv_vs_rv_spread" not in signals

    def test_iv_vs_rv_blocked_suppresses(self):
        """Blocking iv_vs_rv_spread suppresses it."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(iv_rv_spread=0.50, ema9_above_ema21=True, vol_ratio=1.5)
        signals = evaluate_signals(snap, blocked=frozenset({"iv_vs_rv_spread"}))
        assert "iv_vs_rv_spread" not in signals


class TestFcfYieldSignal:
    def test_high_fcf_with_quality_fires(self):
        """fcf_yield > 5% with piotroski_f >= 5 fires fcf_yield_signal."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(fcf_yield=0.07, piotroski_f=6)
        signals = evaluate_signals(snap)
        assert "fcf_yield_signal" in signals

    def test_high_fcf_no_piotroski_fires(self):
        """fcf_yield > 5% with no piotroski_f (None) also fires."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(fcf_yield=0.08)
        signals = evaluate_signals(snap)
        assert "fcf_yield_signal" in signals

    def test_low_piotroski_with_high_fcf_blocked(self):
        """fcf_yield > 5% but piotroski_f < 5 does NOT fire."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(fcf_yield=0.07, piotroski_f=3)
        signals = evaluate_signals(snap)
        assert "fcf_yield_signal" not in signals

    def test_low_fcf_yield_no_signal(self):
        """fcf_yield < 5% doesn't fire."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(fcf_yield=0.03, piotroski_f=7)
        signals = evaluate_signals(snap)
        assert "fcf_yield_signal" not in signals

    def test_none_fcf_yield_no_signal(self):
        """Missing fcf_yield doesn't fire."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(piotroski_f=7)
        signals = evaluate_signals(snap)
        assert "fcf_yield_signal" not in signals

    def test_fcf_yield_blocked_suppresses(self):
        """Blocking fcf_yield_signal suppresses it."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(fcf_yield=0.07, piotroski_f=6)
        signals = evaluate_signals(snap, blocked=frozenset({"fcf_yield_signal"}))
        assert "fcf_yield_signal" not in signals


class TestOptionsSkewSignal:
    def test_panic_put_skew_fires(self):
        """panic_put_skew=True fires options_skew_signal."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(panic_put_skew=True)
        signals = evaluate_signals(snap)
        assert "options_skew_signal" in signals

    def test_call_skew_spike_fires(self):
        """call_skew_spike=True fires options_skew_signal."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(call_skew_spike=True)
        signals = evaluate_signals(snap)
        assert "options_skew_signal" in signals

    def test_neither_skew_no_signal(self):
        """No skew flags → no signal."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(panic_put_skew=False, call_skew_spike=False)
        signals = evaluate_signals(snap)
        assert "options_skew_signal" not in signals

    def test_missing_skew_fields_no_signal(self):
        """Absent skew fields → no signal (both default to False)."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap()
        signals = evaluate_signals(snap)
        assert "options_skew_signal" not in signals

    def test_options_skew_blocked_suppresses(self):
        """Blocking options_skew_signal suppresses it."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(panic_put_skew=True)
        signals = evaluate_signals(snap, blocked=frozenset({"options_skew_signal"}))
        assert "options_skew_signal" not in signals


class TestUnusualOptionsActivity:
    def test_unusual_call_oi_disabled_does_not_fire(self):
        """Retired 2026-07 (options kill/keep: premise inverted, -0.178%/3d t=-2.4, 0/3 yrs)."""
        from signals.evaluator import GLOBALLY_DISABLED, evaluate_signals

        assert "unusual_options_activity" in GLOBALLY_DISABLED
        snap = _base_snap(unusual_call_oi=True)
        signals = evaluate_signals(snap)
        assert "unusual_options_activity" not in signals

    def test_no_unusual_call_oi_no_signal(self):
        """unusual_call_oi=False doesn't fire."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(unusual_call_oi=False)
        signals = evaluate_signals(snap)
        assert "unusual_options_activity" not in signals

    def test_unusual_options_blocked_suppresses(self):
        """Blocking unusual_options_activity suppresses it."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(unusual_call_oi=True)
        signals = evaluate_signals(snap, blocked=frozenset({"unusual_options_activity"}))
        assert "unusual_options_activity" not in signals


class TestPutCallContrarian:
    def test_extreme_put_call_disabled_does_not_fire(self):
        """Retired 2026-07 (options kill/keep: flat on volume proxy, no supporting evidence)."""
        from signals.evaluator import GLOBALLY_DISABLED, evaluate_signals

        assert "put_call_contrarian" in GLOBALLY_DISABLED
        snap = _base_snap(put_call_oi_ratio=3.0, ema9_above_ema21=True)
        signals = evaluate_signals(snap)
        assert "put_call_contrarian" not in signals

    def test_extreme_put_call_with_macd_disabled_does_not_fire(self):
        """Same via the macd_diff path — the disable covers both trend confirmations."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(
            put_call_oi_ratio=3.0,
            ema9_above_ema21=False,
            macd_diff=0.5,
        )
        signals = evaluate_signals(snap)
        assert "put_call_contrarian" not in signals

    def test_low_put_call_no_signal(self):
        """put_call_oi_ratio < 2.5 doesn't fire."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(put_call_oi_ratio=1.5, ema9_above_ema21=True)
        signals = evaluate_signals(snap)
        assert "put_call_contrarian" not in signals

    def test_none_put_call_ratio_no_signal(self):
        """Missing put_call_oi_ratio doesn't fire."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(ema9_above_ema21=True)
        signals = evaluate_signals(snap)
        assert "put_call_contrarian" not in signals

    def test_extreme_put_call_no_trend_no_signal(self):
        """put_call_oi_ratio > 2.5 but no ema_up or macd doesn't fire."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(
            put_call_oi_ratio=3.5,
            ema9_above_ema21=False,
            macd_diff=-0.1,
        )
        signals = evaluate_signals(snap)
        assert "put_call_contrarian" not in signals

    def test_put_call_contrarian_blocked_suppresses(self):
        """Blocking put_call_contrarian suppresses it."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(put_call_oi_ratio=3.0, ema9_above_ema21=True)
        signals = evaluate_signals(snap, blocked=frozenset({"put_call_contrarian"}))
        assert "put_call_contrarian" not in signals


class TestSqueezeSetupLong:
    def _squeeze_snap(self, **kwargs) -> dict:
        snap = _base_snap(
            short_pct_float=0.20,
            short_ratio=6.0,
            ret_5d_pct=1.0,
            near_20d_low=True,
        )
        snap.update(kwargs)
        return snap

    def test_full_setup_fires(self):
        """All conditions met fires squeeze_setup_long."""
        from signals.evaluator import evaluate_signals

        snap = self._squeeze_snap()
        signals = evaluate_signals(snap)
        assert "squeeze_setup_long" in signals

    def test_low_si_pct_no_signal(self):
        """short_pct_float < 0.15 doesn't fire."""
        from signals.evaluator import evaluate_signals

        snap = self._squeeze_snap(short_pct_float=0.10)
        signals = evaluate_signals(snap)
        assert "squeeze_setup_long" not in signals

    def test_low_dtc_no_signal(self):
        """short_ratio < 5 doesn't fire."""
        from signals.evaluator import evaluate_signals

        snap = self._squeeze_snap(short_ratio=3.0)
        signals = evaluate_signals(snap)
        assert "squeeze_setup_long" not in signals

    def test_not_near_20d_low_no_signal(self):
        """near_20d_low=False doesn't fire squeeze_setup."""
        from signals.evaluator import evaluate_signals

        snap = self._squeeze_snap(near_20d_low=False)
        signals = evaluate_signals(snap)
        assert "squeeze_setup_long" not in signals

    def test_none_si_pct_no_signal(self):
        """Missing short_pct_float doesn't fire."""
        from signals.evaluator import evaluate_signals

        snap = self._squeeze_snap()
        del snap["short_pct_float"]
        signals = evaluate_signals(snap)
        assert "squeeze_setup_long" not in signals

    def test_none_si_dtc_no_signal(self):
        """Missing short_ratio doesn't fire."""
        from signals.evaluator import evaluate_signals

        snap = self._squeeze_snap()
        del snap["short_ratio"]
        signals = evaluate_signals(snap)
        assert "squeeze_setup_long" not in signals

    def test_squeeze_setup_blocked_suppresses(self):
        """Blocking squeeze_setup_long suppresses it."""
        from signals.evaluator import evaluate_signals

        snap = self._squeeze_snap()
        signals = evaluate_signals(snap, blocked=frozenset({"squeeze_setup_long"}))
        assert "squeeze_setup_long" not in signals


class TestSqueezeMomentumLong:
    def _squeeze_mom_snap(self, **kwargs) -> dict:
        snap = _base_snap(
            short_pct_float=0.20,
            short_ratio=6.0,
            ret_5d_pct=12.0,
            near_20d_high=True,
        )
        snap.update(kwargs)
        return snap

    def test_full_momentum_fires(self):
        """All conditions met fires squeeze_momentum_long."""
        from signals.evaluator import evaluate_signals

        snap = self._squeeze_mom_snap()
        signals = evaluate_signals(snap)
        assert "squeeze_momentum_long" in signals

    def test_low_ret5d_no_signal(self):
        """ret_5d_pct < 10% doesn't fire squeeze_momentum."""
        from signals.evaluator import evaluate_signals

        snap = self._squeeze_mom_snap(ret_5d_pct=5.0)
        signals = evaluate_signals(snap)
        assert "squeeze_momentum_long" not in signals

    def test_not_near_20d_high_no_signal(self):
        """near_20d_high=False doesn't fire."""
        from signals.evaluator import evaluate_signals

        snap = self._squeeze_mom_snap(near_20d_high=False)
        signals = evaluate_signals(snap)
        assert "squeeze_momentum_long" not in signals

    def test_squeeze_momentum_blocked_suppresses(self):
        """Blocking squeeze_momentum_long suppresses it."""
        from signals.evaluator import evaluate_signals

        snap = self._squeeze_mom_snap()
        signals = evaluate_signals(snap, blocked=frozenset({"squeeze_momentum_long"}))
        assert "squeeze_momentum_long" not in signals


class TestShortInterestTrendLong:
    def test_si_falling_from_peak_fires(self):
        """SI% fallen >30% from peak + price rising fires short_interest_trend_long."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(short_pct_float=0.10, si_peak=0.20, ret_5d_pct=1.5)
        signals = evaluate_signals(snap)
        assert "short_interest_trend_long" in signals

    def test_si_not_fallen_enough_no_signal(self):
        """SI% fallen < 30% from peak doesn't fire."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(short_pct_float=0.18, si_peak=0.20, ret_5d_pct=1.5)
        signals = evaluate_signals(snap)
        assert "short_interest_trend_long" not in signals

    def test_price_not_rising_no_signal(self):
        """ret_5d <= 0 doesn't fire."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(short_pct_float=0.10, si_peak=0.20, ret_5d_pct=-1.0)
        signals = evaluate_signals(snap)
        assert "short_interest_trend_long" not in signals

    def test_none_si_pct_no_signal(self):
        """Missing short_pct_float doesn't fire."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(si_peak=0.20, ret_5d_pct=1.5)
        signals = evaluate_signals(snap)
        assert "short_interest_trend_long" not in signals

    def test_none_si_peak_no_signal(self):
        """Missing si_peak doesn't fire."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(short_pct_float=0.10, ret_5d_pct=1.5)
        signals = evaluate_signals(snap)
        assert "short_interest_trend_long" not in signals

    def test_zero_si_peak_no_signal(self):
        """si_peak=0 doesn't fire (division by zero guard)."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(short_pct_float=0.10, si_peak=0.0, ret_5d_pct=1.5)
        signals = evaluate_signals(snap)
        assert "short_interest_trend_long" not in signals

    def test_si_trend_blocked_suppresses(self):
        """Blocking short_interest_trend_long suppresses it."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(short_pct_float=0.10, si_peak=0.20, ret_5d_pct=1.5)
        signals = evaluate_signals(snap, blocked=frozenset({"short_interest_trend_long"}))
        assert "short_interest_trend_long" not in signals


class TestAnalystUpgradeSignal:
    def test_analyst_upgrade_fires(self):
        """analyst_upgrade=True fires analyst_upgrade_signal."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(analyst_upgrade=True)
        signals = evaluate_signals(snap)
        assert "analyst_upgrade_signal" in signals

    def test_no_upgrade_no_signal(self):
        """analyst_upgrade=False doesn't fire."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(analyst_upgrade=False)
        signals = evaluate_signals(snap)
        assert "analyst_upgrade_signal" not in signals

    def test_analyst_upgrade_blocked_suppresses(self):
        """Blocking analyst_upgrade_signal suppresses it."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(analyst_upgrade=True)
        signals = evaluate_signals(snap, blocked=frozenset({"analyst_upgrade_signal"}))
        assert "analyst_upgrade_signal" not in signals


class TestAaiiExtremeFear:
    def test_extreme_fear_fires(self):
        """aaii_extreme_fear=True fires aaii_extreme_fear_long."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(aaii_extreme_fear=True)
        signals = evaluate_signals(snap)
        assert "aaii_extreme_fear_long" in signals

    def test_no_extreme_fear_no_signal(self):
        """aaii_extreme_fear=False doesn't fire."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(aaii_extreme_fear=False)
        signals = evaluate_signals(snap)
        assert "aaii_extreme_fear_long" not in signals

    def test_missing_field_no_signal(self):
        """Missing aaii_extreme_fear defaults to False."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap()
        signals = evaluate_signals(snap)
        assert "aaii_extreme_fear_long" not in signals

    def test_aaii_extreme_fear_blocked_suppresses(self):
        """Blocking aaii_extreme_fear_long suppresses it."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(aaii_extreme_fear=True)
        signals = evaluate_signals(snap, blocked=frozenset({"aaii_extreme_fear_long"}))
        assert "aaii_extreme_fear_long" not in signals


class TestFearGreedExtremeFear:
    def test_low_score_fires(self):
        """fear_greed_score < 20 fires fear_greed_extreme_fear."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(fear_greed_score=15.0)
        signals = evaluate_signals(snap)
        assert "fear_greed_extreme_fear" in signals

    def test_high_score_no_signal(self):
        """fear_greed_score >= 20 doesn't fire."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(fear_greed_score=50.0)
        signals = evaluate_signals(snap)
        assert "fear_greed_extreme_fear" not in signals

    def test_none_score_no_signal(self):
        """Missing fear_greed_score doesn't fire."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap()
        signals = evaluate_signals(snap)
        assert "fear_greed_extreme_fear" not in signals

    def test_fear_greed_blocked_suppresses(self):
        """Blocking fear_greed_extreme_fear suppresses it."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(fear_greed_score=5.0)
        signals = evaluate_signals(snap, blocked=frozenset({"fear_greed_extreme_fear"}))
        assert "fear_greed_extreme_fear" not in signals


class TestSectorPairMeanReversion:
    def test_high_z_score_ema_up_fires(self):
        """pairs_spread_z > 1.5 with ema_up fires sector_pair_mean_reversion."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(pairs_spread_z=2.0, ema9_above_ema21=True)
        signals = evaluate_signals(snap)
        assert "sector_pair_mean_reversion" in signals

    def test_negative_z_score_no_signal(self):
        """pairs_spread_z < 0 (expensive leg) doesn't fire."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(pairs_spread_z=-2.0, ema9_above_ema21=True)
        signals = evaluate_signals(snap)
        assert "sector_pair_mean_reversion" not in signals

    def test_low_z_score_no_signal(self):
        """pairs_spread_z < 1.5 doesn't fire."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(pairs_spread_z=1.0, ema9_above_ema21=True)
        signals = evaluate_signals(snap)
        assert "sector_pair_mean_reversion" not in signals

    def test_none_pairs_z_no_signal(self):
        """Missing pairs_spread_z doesn't fire."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(ema9_above_ema21=True)
        signals = evaluate_signals(snap)
        assert "sector_pair_mean_reversion" not in signals

    def test_no_ema_up_no_signal(self):
        """pairs_spread_z > 1.5 but ema9_above_ema21=False doesn't fire."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(pairs_spread_z=2.0, ema9_above_ema21=False)
        signals = evaluate_signals(snap)
        assert "sector_pair_mean_reversion" not in signals

    def test_pairs_blocked_suppresses(self):
        """Blocking sector_pair_mean_reversion suppresses it."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(pairs_spread_z=2.0, ema9_above_ema21=True)
        signals = evaluate_signals(snap, blocked=frozenset({"sector_pair_mean_reversion"}))
        assert "sector_pair_mean_reversion" not in signals


class TestGoogleTrendsBullish:
    def test_trends_spike_fires(self):
        """google_trends_spike=True fires google_trends_bullish."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(google_trends_spike=True)
        signals = evaluate_signals(snap)
        assert "google_trends_bullish" in signals

    def test_no_trends_spike_no_signal(self):
        """google_trends_spike=False doesn't fire."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(google_trends_spike=False)
        signals = evaluate_signals(snap)
        assert "google_trends_bullish" not in signals

    def test_missing_field_no_signal(self):
        """Missing google_trends_spike defaults to False."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap()
        signals = evaluate_signals(snap)
        assert "google_trends_bullish" not in signals

    def test_google_trends_blocked_suppresses(self):
        """Blocking google_trends_bullish suppresses it."""
        from signals.evaluator import evaluate_signals

        snap = _base_snap(google_trends_spike=True)
        signals = evaluate_signals(snap, blocked=frozenset({"google_trends_bullish"}))
        assert "google_trends_bullish" not in signals


# ── New short signals ─────────────────────────────────────────────────────────


class TestAltmanDistressShort:
    def test_distress_globally_disabled(self):
        """v1.99: altman_distress_short is in SHORT_GLOBALLY_DISABLED — never fires even when
        altman_z < 1.1. Disabled as a lagging, multi-month thesis wrong for a 1-5d hold."""
        from signals.evaluator import SHORT_GLOBALLY_DISABLED, evaluate_short_signals

        assert "altman_distress_short" in SHORT_GLOBALLY_DISABLED
        snap = {"altman_z": 0.8}
        signals = evaluate_short_signals(snap)
        assert "altman_distress_short" not in signals

    def test_safe_zone_no_short(self):
        """altman_z >= 1.1 doesn't fire."""
        from signals.evaluator import evaluate_short_signals

        snap = {"altman_z": 2.0}
        signals = evaluate_short_signals(snap)
        assert "altman_distress_short" not in signals

    def test_none_altman_no_short(self):
        """Missing altman_z doesn't fire."""
        from signals.evaluator import evaluate_short_signals

        snap = {}
        signals = evaluate_short_signals(snap)
        assert "altman_distress_short" not in signals

    def test_altman_distress_blocked_suppresses(self):
        """Blocking altman_distress_short suppresses it."""
        from signals.evaluator import evaluate_short_signals

        snap = {"altman_z": 0.5}
        signals = evaluate_short_signals(snap, blocked=frozenset({"altman_distress_short"}))
        assert "altman_distress_short" not in signals


class TestPostEarningsGapdownFailedBounce:
    def test_fires_on_gap_plus_failed_bounce(self):
        """Recent ≥7% earnings gap-down + failed bounce + volume → signal fires."""
        from signals.evaluator import evaluate_short_signals

        snap = {
            "earnings_gap_pct": -9.0,
            "gap_failed_bounce": True,
            "vol_ratio": 1.8,
        }
        assert "post_earnings_gapdown_failed_bounce" in evaluate_short_signals(snap)

    def test_no_fire_without_failed_bounce(self):
        """Gap-down present but the bounce has not failed → no entry (avoids dead-cat bounce)."""
        from signals.evaluator import evaluate_short_signals

        snap = {"earnings_gap_pct": -9.0, "gap_failed_bounce": False, "vol_ratio": 1.8}
        assert "post_earnings_gapdown_failed_bounce" not in evaluate_short_signals(snap)

    def test_no_fire_when_gap_too_small(self):
        from signals.evaluator import evaluate_short_signals

        snap = {"earnings_gap_pct": -3.0, "gap_failed_bounce": True, "vol_ratio": 1.8}
        assert "post_earnings_gapdown_failed_bounce" not in evaluate_short_signals(snap)

    def test_no_fire_on_low_volume(self):
        from signals.evaluator import evaluate_short_signals

        snap = {"earnings_gap_pct": -9.0, "gap_failed_bounce": True, "vol_ratio": 0.8}
        assert "post_earnings_gapdown_failed_bounce" not in evaluate_short_signals(snap)

    def test_no_fire_when_no_gap_data(self):
        from signals.evaluator import evaluate_short_signals

        snap = {"gap_failed_bounce": True, "vol_ratio": 1.8}
        assert "post_earnings_gapdown_failed_bounce" not in evaluate_short_signals(snap)

    def test_blocked_when_explicitly_blocked(self):
        from signals.evaluator import evaluate_short_signals

        snap = {"earnings_gap_pct": -9.0, "gap_failed_bounce": True, "vol_ratio": 1.8}
        result = evaluate_short_signals(
            snap, blocked=frozenset({"post_earnings_gapdown_failed_bounce"})
        )
        assert "post_earnings_gapdown_failed_bounce" not in result

    def test_in_short_priority(self):
        from signals.evaluator import SHORT_SIGNAL_PRIORITY

        assert "post_earnings_gapdown_failed_bounce" in SHORT_SIGNAL_PRIORITY


class TestPiotroskiDistressShort:
    def test_distress_f_score_fires(self):
        """piotroski_f <= 2 with price_below_sma200=True fires piotroski_distress_short."""
        from signals.evaluator import evaluate_short_signals

        snap = {"piotroski_f": 2, "price_below_sma200": True}
        signals = evaluate_short_signals(snap)
        assert "piotroski_distress_short" in signals

    def test_adequate_f_score_no_short(self):
        """piotroski_f > 2 doesn't fire."""
        from signals.evaluator import evaluate_short_signals

        snap = {"piotroski_f": 4, "price_below_sma200": True}
        signals = evaluate_short_signals(snap)
        assert "piotroski_distress_short" not in signals

    def test_not_below_sma200_no_short(self):
        """price_below_sma200=False doesn't fire even with low F-score."""
        from signals.evaluator import evaluate_short_signals

        snap = {"piotroski_f": 1, "price_below_sma200": False}
        signals = evaluate_short_signals(snap)
        assert "piotroski_distress_short" not in signals

    def test_none_piotroski_no_short(self):
        """Missing piotroski_f doesn't fire."""
        from signals.evaluator import evaluate_short_signals

        snap = {"price_below_sma200": True}
        signals = evaluate_short_signals(snap)
        assert "piotroski_distress_short" not in signals

    def test_piotroski_distress_blocked_suppresses(self):
        """Blocking piotroski_distress_short suppresses it."""
        from signals.evaluator import evaluate_short_signals

        snap = {"piotroski_f": 1, "price_below_sma200": True}
        signals = evaluate_short_signals(snap, blocked=frozenset({"piotroski_distress_short"}))
        assert "piotroski_distress_short" not in signals


class TestGrossMarginDeteriorationShort:
    def test_gm_deterioration_globally_disabled(self):
        """v1.99: gross_margin_deterioration_short is in SHORT_GLOBALLY_DISABLED — never fires.
        Disabled as a slow fundamental signal (n=5, worst avg of the lagging short trio)."""
        from signals.evaluator import SHORT_GLOBALLY_DISABLED, evaluate_short_signals

        assert "gross_margin_deterioration_short" in SHORT_GLOBALLY_DISABLED
        snap = {"gross_margin_trend": -0.05, "price_below_sma200": True}
        signals = evaluate_short_signals(snap)
        assert "gross_margin_deterioration_short" not in signals

    def test_mild_gm_deterioration_no_short(self):
        """gross_margin_trend > -0.03 doesn't fire."""
        from signals.evaluator import evaluate_short_signals

        snap = {"gross_margin_trend": -0.01, "price_below_sma200": True}
        signals = evaluate_short_signals(snap)
        assert "gross_margin_deterioration_short" not in signals

    def test_not_below_sma200_no_short(self):
        """price_below_sma200=False doesn't fire."""
        from signals.evaluator import evaluate_short_signals

        snap = {"gross_margin_trend": -0.05, "price_below_sma200": False}
        signals = evaluate_short_signals(snap)
        assert "gross_margin_deterioration_short" not in signals

    def test_none_gm_trend_no_short(self):
        """Missing gross_margin_trend doesn't fire."""
        from signals.evaluator import evaluate_short_signals

        snap = {"price_below_sma200": True}
        signals = evaluate_short_signals(snap)
        assert "gross_margin_deterioration_short" not in signals

    def test_gm_deterioration_blocked_suppresses(self):
        """Blocking gross_margin_deterioration_short suppresses it."""
        from signals.evaluator import evaluate_short_signals

        snap = {"gross_margin_trend": -0.05, "price_below_sma200": True}
        signals = evaluate_short_signals(
            snap, blocked=frozenset({"gross_margin_deterioration_short"})
        )
        assert "gross_margin_deterioration_short" not in signals


class TestAccrualsQualityShort:
    def test_high_accruals_extended_price_fires(self):
        """accruals_ratio > 0.15 + ret_5d > 5% fires accruals_quality_short."""
        from signals.evaluator import evaluate_short_signals

        snap = {"accruals_ratio": 0.20, "ret_5d_pct": 7.0}
        signals = evaluate_short_signals(snap)
        assert "accruals_quality_short" in signals

    def test_low_accruals_no_short(self):
        """accruals_ratio < 0.15 doesn't fire."""
        from signals.evaluator import evaluate_short_signals

        snap = {"accruals_ratio": 0.10, "ret_5d_pct": 7.0}
        signals = evaluate_short_signals(snap)
        assert "accruals_quality_short" not in signals

    def test_high_accruals_flat_price_no_short(self):
        """High accruals but ret_5d < 5% doesn't fire (not extended)."""
        from signals.evaluator import evaluate_short_signals

        snap = {"accruals_ratio": 0.20, "ret_5d_pct": 3.0}
        signals = evaluate_short_signals(snap)
        assert "accruals_quality_short" not in signals

    def test_none_accruals_no_short(self):
        """Missing accruals_ratio doesn't fire."""
        from signals.evaluator import evaluate_short_signals

        snap = {"ret_5d_pct": 8.0}
        signals = evaluate_short_signals(snap)
        assert "accruals_quality_short" not in signals

    def test_accruals_short_blocked_suppresses(self):
        """Blocking accruals_quality_short suppresses it."""
        from signals.evaluator import evaluate_short_signals

        snap = {"accruals_ratio": 0.20, "ret_5d_pct": 7.0}
        signals = evaluate_short_signals(snap, blocked=frozenset({"accruals_quality_short"}))
        assert "accruals_quality_short" not in signals


class TestLockupExpiryShort:
    def test_lockup_expiry_soon_fires(self):
        """lockup_expiry_soon=True fires lockup_expiry_short."""
        from signals.evaluator import evaluate_short_signals

        snap = {"lockup_expiry_soon": True}
        signals = evaluate_short_signals(snap)
        assert "lockup_expiry_short" in signals

    def test_no_lockup_no_short(self):
        """lockup_expiry_soon=False doesn't fire."""
        from signals.evaluator import evaluate_short_signals

        snap = {"lockup_expiry_soon": False}
        signals = evaluate_short_signals(snap)
        assert "lockup_expiry_short" not in signals

    def test_missing_lockup_field_no_short(self):
        """Missing lockup_expiry_soon doesn't fire."""
        from signals.evaluator import evaluate_short_signals

        snap = {}
        signals = evaluate_short_signals(snap)
        assert "lockup_expiry_short" not in signals

    def test_lockup_blocked_suppresses(self):
        """Blocking lockup_expiry_short suppresses it."""
        from signals.evaluator import evaluate_short_signals

        snap = {"lockup_expiry_soon": True}
        signals = evaluate_short_signals(snap, blocked=frozenset({"lockup_expiry_short"}))
        assert "lockup_expiry_short" not in signals


class TestAnalystDowngradeSignal:
    def test_analyst_downgrade_fires(self):
        """analyst_downgrade=True fires analyst_downgrade_signal."""
        from signals.evaluator import evaluate_short_signals

        snap = {"analyst_downgrade": True}
        signals = evaluate_short_signals(snap)
        assert "analyst_downgrade_signal" in signals

    def test_no_downgrade_no_short(self):
        """analyst_downgrade=False doesn't fire."""
        from signals.evaluator import evaluate_short_signals

        snap = {"analyst_downgrade": False}
        signals = evaluate_short_signals(snap)
        assert "analyst_downgrade_signal" not in signals

    def test_missing_field_no_short(self):
        """Missing analyst_downgrade doesn't fire."""
        from signals.evaluator import evaluate_short_signals

        snap = {}
        signals = evaluate_short_signals(snap)
        assert "analyst_downgrade_signal" not in signals

    def test_analyst_downgrade_blocked_suppresses(self):
        """Blocking analyst_downgrade_signal suppresses it."""
        from signals.evaluator import evaluate_short_signals

        snap = {"analyst_downgrade": True}
        signals = evaluate_short_signals(snap, blocked=frozenset({"analyst_downgrade_signal"}))
        assert "analyst_downgrade_signal" not in signals


# ── Signal priority ordering ───────────────────────────────────────────────────


class TestSignalPriorityOrdering:
    def test_activist_13d_before_pead(self):
        """activist_13d_signal has lower priority number than pead (fires first)."""
        from signals.evaluator import SIGNAL_PRIORITY

        assert SIGNAL_PRIORITY["activist_13d_signal"] < SIGNAL_PRIORITY["pead"]

    def test_guidance_raise_before_gap_and_go(self):
        """guidance_raise_signal has lower priority number than gap_and_go."""
        from signals.evaluator import SIGNAL_PRIORITY

        assert SIGNAL_PRIORITY["guidance_raise_signal"] < SIGNAL_PRIORITY["gap_and_go"]

    def test_new_signals_in_signal_priority(self):
        """All new long signals are in SIGNAL_PRIORITY."""
        from signals.evaluator import SIGNAL_PRIORITY

        new_signals = [
            "activist_13d_signal",
            "guidance_raise_signal",
            "iv_vs_rv_spread",
            "fcf_yield_signal",
            "options_skew_signal",
            "unusual_options_activity",
            "put_call_contrarian",
            "squeeze_setup_long",
            "squeeze_momentum_long",
            "short_interest_trend_long",
            "analyst_upgrade_signal",
            "aaii_extreme_fear_long",
            "fear_greed_extreme_fear",
            "sector_pair_mean_reversion",
            "google_trends_bullish",
        ]
        for sig in new_signals:
            assert sig in SIGNAL_PRIORITY, f"{sig} missing from SIGNAL_PRIORITY"

    def test_new_short_signals_in_short_priority(self):
        """All new short signals are in SHORT_SIGNAL_PRIORITY."""
        from signals.evaluator import SHORT_SIGNAL_PRIORITY

        new_shorts = [
            "altman_distress_short",
            "piotroski_distress_short",
            "gross_margin_deterioration_short",
            "accruals_quality_short",
            "lockup_expiry_short",
            "analyst_downgrade_signal",
        ]
        for sig in new_shorts:
            assert sig in SHORT_SIGNAL_PRIORITY, f"{sig} missing from SHORT_SIGNAL_PRIORITY"
