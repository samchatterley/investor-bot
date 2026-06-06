"""Tests for execution/stock_scanner.py — get_market_regime, prefilter_candidates, get_top_movers."""

import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from execution.stock_scanner import (
    _passes_quality_screen,
    get_market_regime,
    prefilter_candidates,
    score_candidate,
)


def _snap(**kwargs):
    defaults = {
        "symbol": "TEST",
        "rsi_14": 50,
        "bb_pct": 0.5,
        "vol_ratio": 1.0,
        "ema9_above_ema21": False,
        "macd_diff": 0,
        "macd_crossed_up": False,
        "weekly_trend_up": True,
        "ret_5d_pct": 0,
        "avg_volume": 1_000_000,  # above MIN_VOLUME = 500_000
        # New strategy fields
        "bb_squeeze": False,
        "price_vs_52w_high_pct": -10.0,  # 10% below high (default — not near breakout)
        "rel_strength_5d": 0.0,
        "rel_strength_10d": 0.0,
        "is_inside_day": False,
        "price_vs_ema21_pct": 0.0,
    }
    defaults.update(kwargs)
    return defaults


class TestGetMarketRegime(unittest.TestCase):
    """Tests for the thin wrapper in execution/stock_scanner.py.

    The wrapper calls fetch_spy_vix_history, load_regime_state, the shared
    classify function, and save_regime_state.  We mock those four seams so
    tests don't need network access or a live yfinance install.
    """

    def _mock_shared(self, regime_name: str, is_bearish: bool = False):
        """Return a context manager that injects a pre-built snapshot."""
        from data.market_regime import (
            MarketRegime,
            MarketRegimeSnapshot,
            RegimeFeatures,
        )

        idx = pd.bdate_range("2024-01-01", periods=10)
        spy_df = pd.DataFrame({"Close": [400.0] * 10}, index=idx)
        vix_df = pd.DataFrame({"Close": [18.0] * 10}, index=idx)
        features = RegimeFeatures(
            spy_ret_1d=1.0,
            spy_ret_5d=3.0,
            spy_ret_20d=5.0,
            spy_above_ma200=True,
            spy_drawdown_pct=-1.0,
            vix=18.0,
            vix_ma20=17.0,
            vix_vs_ma=1.06,
            vix_5d_change=-2.0,
            vix9d=None,
            data_quality="full",
        )
        snapshot = MarketRegimeSnapshot(
            regime=MarketRegime(regime_name),
            reasons=(f"test {regime_name}",),
            features=features,
        )
        import contextlib

        return contextlib.ExitStack(), [
            patch("execution.stock_scanner.fetch_spy_vix_history", return_value=(spy_df, vix_df)),
            patch("execution.stock_scanner.fetch_vix9d_history", return_value=None),
            patch("execution.stock_scanner.load_regime_state", return_value=None),
            patch("execution.stock_scanner._compute_regime", return_value=snapshot),
            patch("execution.stock_scanner.save_regime_state"),
        ]

    def _with_regime(self, regime_name: str):
        _, patches = self._mock_shared(regime_name)
        return patches

    def test_bull_trend_regime(self):
        patchers = self._with_regime("BULL_TREND")
        with patchers[0], patchers[1], patchers[2], patchers[3], patchers[4]:
            result = get_market_regime(threshold_pct=-1.5)
        self.assertEqual(result["regime"], "BULL_TREND")
        self.assertFalse(result["is_bearish"])

    def test_stress_regime_sets_is_bearish(self):
        patchers = self._with_regime("STRESS_RISK_OFF")
        with patchers[0], patchers[1], patchers[2], patchers[3], patchers[4]:
            result = get_market_regime(threshold_pct=-1.5)
        self.assertEqual(result["regime"], "STRESS_RISK_OFF")
        self.assertTrue(result["is_bearish"])

    def test_high_vol_downtrend_regime(self):
        patchers = self._with_regime("HIGH_VOL_DOWNTREND")
        with patchers[0], patchers[1], patchers[2], patchers[3], patchers[4]:
            result = get_market_regime(threshold_pct=-1.5, vix=30.0)
        self.assertEqual(result["regime"], "HIGH_VOL_DOWNTREND")

    def test_neutral_chop_regime(self):
        patchers = self._with_regime("NEUTRAL_CHOP")
        with patchers[0], patchers[1], patchers[2], patchers[3], patchers[4]:
            result = get_market_regime(threshold_pct=-1.5)
        self.assertEqual(result["regime"], "NEUTRAL_CHOP")

    def test_exception_returns_unknown(self):
        with patch(
            "execution.stock_scanner.fetch_spy_vix_history",
            side_effect=Exception("network"),
        ):
            result = get_market_regime()
        self.assertEqual(result["regime"], "UNKNOWN")
        self.assertFalse(result["is_bearish"])

    def test_result_has_required_keys(self):
        patchers = self._with_regime("NEUTRAL_CHOP")
        with patchers[0], patchers[1], patchers[2], patchers[3], patchers[4]:
            result = get_market_regime()
        for key in ("is_bearish", "spy_change_pct", "spy_5d_pct", "regime"):
            self.assertIn(key, result)


class TestPrefilterCandidates(unittest.TestCase):
    def test_momentum_signal_passes(self):
        snap = _snap(ema9_above_ema21=True, macd_diff=0.5, ret_5d_pct=2.0, vol_ratio=1.5)
        result = prefilter_candidates([snap])
        self.assertEqual(len(result), 1)

    def test_mean_reversion_signal_passes(self):
        # rsi<35, bb<0.15, vol>1.2 (canonical thresholds from signals/evaluator.py)
        snap = _snap(rsi_14=30, bb_pct=0.10, vol_ratio=1.3)
        result = prefilter_candidates([snap])
        self.assertEqual(len(result), 1)

    def test_macd_crossover_passes(self):
        snap = _snap(macd_crossed_up=True, vol_ratio=1.5)
        result = prefilter_candidates([snap])
        self.assertEqual(len(result), 1)

    def test_no_signal_filtered_out(self):
        snap = _snap()  # all defaults — no signal
        result = prefilter_candidates([snap])
        self.assertEqual(len(result), 0)

    def test_against_weekly_trend_filtered(self):
        # Momentum setup but weekly trend is down
        snap = _snap(
            ema9_above_ema21=True,
            macd_diff=0.5,
            ret_5d_pct=2.0,
            vol_ratio=1.5,
            weekly_trend_up=False,
        )
        result = prefilter_candidates([snap])
        self.assertEqual(len(result), 0)

    def test_deeply_oversold_bypasses_weekly_trend(self):
        # RSI < 30, BB < 0.15 → allowed even against weekly trend
        snap = _snap(rsi_14=25, bb_pct=0.10, vol_ratio=1.5, weekly_trend_up=False)
        result = prefilter_candidates([snap])
        self.assertEqual(len(result), 1)

    def test_illiquid_stock_filtered_by_min_volume(self):
        snap = _snap(rsi_14=30, bb_pct=0.20, vol_ratio=1.2, avg_volume=100_000)
        result = prefilter_candidates([snap])
        self.assertEqual(len(result), 0)

    def test_empty_input_returns_empty(self):
        self.assertEqual(prefilter_candidates([]), [])

    def test_mixed_batch_filters_correctly(self):
        good = _snap(symbol="GOOD", rsi_14=30, bb_pct=0.10, vol_ratio=1.3)
        bad = _snap(symbol="BAD")
        result = prefilter_candidates([good, bad])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["symbol"], "GOOD")

    # ── bb_squeeze_breakout ──────────────────────────────────────────────────

    def _bbs_snap(self, **overrides):
        """Minimal valid bb_squeeze snapshot — all new gates satisfied by default."""
        base = {
            "bb_squeeze": True,
            "bb_squeeze_days": 5,
            "ema9_above_ema21": True,
            "vol_ratio": 1.5,
            "adx": 27,
            "rs_rank_pct": 70.0,
            "current_price": 20.0,
        }
        base.update(overrides)
        return _snap(**base)

    def test_bb_squeeze_with_ema_up_and_volume_passes(self):
        snap = self._bbs_snap()
        self.assertEqual(len(prefilter_candidates([snap])), 1)

    def test_bb_squeeze_with_positive_macd_passes(self):
        snap = self._bbs_snap(ema9_above_ema21=False, macd_diff=0.3, vol_ratio=1.3)
        self.assertEqual(len(prefilter_candidates([snap])), 1)

    def test_bb_squeeze_without_directional_confirmation_fails(self):
        snap = self._bbs_snap(ema9_above_ema21=False, macd_diff=-0.1)
        self.assertEqual(len(prefilter_candidates([snap])), 0)

    def test_bb_squeeze_without_volume_fails(self):
        snap = self._bbs_snap(vol_ratio=0.9)
        self.assertEqual(len(prefilter_candidates([snap])), 0)

    def test_bb_squeeze_insufficient_days_fails(self):
        snap = self._bbs_snap(bb_squeeze_days=4)
        self.assertEqual(len(prefilter_candidates([snap])), 0)

    def test_bb_squeeze_low_rs_rank_fails(self):
        snap = self._bbs_snap(rs_rank_pct=59.9)
        self.assertEqual(len(prefilter_candidates([snap])), 0)

    def test_bb_squeeze_penny_stock_fails(self):
        snap = self._bbs_snap(current_price=9.99)
        self.assertEqual(len(prefilter_candidates([snap])), 0)

    def test_bb_squeeze_low_adx_fails(self):
        snap = self._bbs_snap(adx=24)
        self.assertEqual(len(prefilter_candidates([snap])), 0)

    # ── breakout_52w ─────────────────────────────────────────────────────────

    def test_breakout_52w_globally_disabled(self):
        from signals.evaluator import GLOBALLY_DISABLED

        self.assertIn("breakout_52w", GLOBALLY_DISABLED)
        snap = _snap(price_vs_52w_high_pct=-1.5, vol_ratio=1.5, ema9_above_ema21=True)
        self.assertEqual(len(prefilter_candidates([snap])), 0)

    def test_breakout_52w_too_far_from_high_fails(self):
        snap = _snap(price_vs_52w_high_pct=-5.0, vol_ratio=1.5, weekly_trend_up=True)
        self.assertEqual(len(prefilter_candidates([snap])), 0)

    def test_breakout_52w_against_weekly_trend_fails(self):
        snap = _snap(price_vs_52w_high_pct=-1.0, vol_ratio=1.5, weekly_trend_up=False)
        self.assertEqual(len(prefilter_candidates([snap])), 0)

    def test_breakout_52w_low_volume_fails(self):
        snap = _snap(price_vs_52w_high_pct=-1.0, vol_ratio=0.9, weekly_trend_up=True)
        self.assertEqual(len(prefilter_candidates([snap])), 0)

    # ── inside_day_breakout ──────────────────────────────────────────────────

    def test_inside_day_ema_up_and_volume_passes(self):
        snap = _snap(is_inside_day=True, ema9_above_ema21=True, vol_ratio=1.2)
        self.assertEqual(len(prefilter_candidates([snap])), 1)

    def test_inside_day_positive_macd_passes(self):
        snap = _snap(is_inside_day=True, macd_diff=0.5, vol_ratio=1.2)
        self.assertEqual(len(prefilter_candidates([snap])), 1)

    def test_inside_day_no_directional_confirmation_fails(self):
        snap = _snap(is_inside_day=True, ema9_above_ema21=False, macd_diff=-0.1, vol_ratio=1.2)
        self.assertEqual(len(prefilter_candidates([snap])), 0)

    def test_inside_day_insufficient_volume_fails(self):
        snap = _snap(is_inside_day=True, ema9_above_ema21=True, vol_ratio=0.9)
        self.assertEqual(len(prefilter_candidates([snap])), 0)

    # ── trend_pullback ───────────────────────────────────────────────────────

    def test_trend_pullback_in_buy_zone_passes(self):
        # EMA up, price 2% below EMA21, RSI 50 (mid-range), normal volume
        snap = _snap(ema9_above_ema21=True, price_vs_ema21_pct=-2.0, rsi_14=50, vol_ratio=1.1)
        self.assertEqual(len(prefilter_candidates([snap])), 1)

    def test_trend_pullback_just_below_ema21_passes(self):
        snap = _snap(ema9_above_ema21=True, price_vs_ema21_pct=-0.6, rsi_14=53, vol_ratio=1.1)
        self.assertEqual(len(prefilter_candidates([snap])), 1)

    def test_trend_pullback_ema_not_up_fails(self):
        snap = _snap(ema9_above_ema21=False, price_vs_ema21_pct=-2.0, rsi_14=50, vol_ratio=1.1)
        self.assertEqual(len(prefilter_candidates([snap])), 0)

    def test_trend_pullback_too_deep_fails(self):
        # price_vs_ema21_pct < -3.0 — too extended downward
        snap = _snap(ema9_above_ema21=True, price_vs_ema21_pct=-4.0, rsi_14=50, vol_ratio=1.1)
        self.assertEqual(len(prefilter_candidates([snap])), 0)

    def test_trend_pullback_too_close_fails(self):
        # price_vs_ema21_pct > -0.5 — not pulled back enough
        snap = _snap(ema9_above_ema21=True, price_vs_ema21_pct=-0.3, rsi_14=50, vol_ratio=1.1)
        self.assertEqual(len(prefilter_candidates([snap])), 0)

    def test_trend_pullback_overbought_rsi_fails(self):
        snap = _snap(ema9_above_ema21=True, price_vs_ema21_pct=-2.0, rsi_14=65, vol_ratio=1.1)
        self.assertEqual(len(prefilter_candidates([snap])), 0)

    def test_trend_pullback_oversold_rsi_fails(self):
        # RSI < 40 → likely mean_reversion territory, not a pullback
        snap = _snap(ema9_above_ema21=True, price_vs_ema21_pct=-2.0, rsi_14=35, vol_ratio=1.1)
        # This may pass mean_reversion if bb_pct is also low, but with defaults it fails both
        # (default bb_pct=0.5, so mean_reversion doesn't fire; trend_pullback rsi guard blocks it)
        self.assertEqual(len(prefilter_candidates([snap])), 0)


class TestInsiderBuyingSignal(unittest.TestCase):
    """insider_buying signal in prefilter_candidates."""

    def test_passes_and_signals_when_cluster_true(self):
        snap = _snap(insider_cluster=True)
        result = prefilter_candidates([snap])
        self.assertEqual(len(result), 1)
        self.assertIn("insider_buying", result[0]["matched_signals"])

    def test_no_signal_when_cluster_false(self):
        snap = _snap(insider_cluster=False)
        result = prefilter_candidates([snap])
        self.assertEqual(len(result), 0)

    def test_no_signal_when_cluster_field_absent(self):
        snap = _snap()  # no insider_cluster key
        result = prefilter_candidates([snap])
        self.assertEqual(len(result), 0)

    def test_insider_buying_combined_with_technical_signal(self):
        # insider_cluster AND mean_reversion both fire → both appear in matched_signals
        snap = _snap(insider_cluster=True, rsi_14=30, bb_pct=0.10, vol_ratio=1.3)
        result = prefilter_candidates([snap])
        self.assertEqual(len(result), 1)
        signals = result[0]["matched_signals"]
        self.assertIn("insider_buying", signals)
        self.assertIn("mean_reversion", signals)

    def test_weekly_trend_filter_does_not_block_insider_buying(self):
        # insider_buying is standalone — weekly trend guard applies only when
        # no matched signals pass, but insider_buying passes so the stock is kept.
        snap = _snap(insider_cluster=True, weekly_trend_up=False, rsi_14=50)
        result = prefilter_candidates([snap])
        self.assertEqual(len(result), 1)


class TestMatchedSignals(unittest.TestCase):
    """prefilter_candidates annotates each result with matched_signals."""

    def test_matched_signals_present_on_all_results(self):
        snap = _snap(rsi_14=30, bb_pct=0.10, vol_ratio=1.3)
        result = prefilter_candidates([snap])
        self.assertIn("matched_signals", result[0])

    def test_mean_reversion_signal_annotated(self):
        snap = _snap(rsi_14=30, bb_pct=0.10, vol_ratio=1.3)
        result = prefilter_candidates([snap])
        self.assertIn("mean_reversion", result[0]["matched_signals"])

    def test_momentum_signal_annotated(self):
        snap = _snap(ema9_above_ema21=True, macd_diff=0.5, ret_5d_pct=2.0, vol_ratio=1.5)
        result = prefilter_candidates([snap])
        self.assertIn("momentum", result[0]["matched_signals"])

    def test_multiple_signals_all_annotated(self):
        snap = _snap(
            rsi_14=30,
            bb_pct=0.10,
            vol_ratio=1.5,
            ema9_above_ema21=True,
            macd_diff=0.5,
            ret_5d_pct=2.0,
        )
        result = prefilter_candidates([snap])
        signals = result[0]["matched_signals"]
        self.assertIn("mean_reversion", signals)
        self.assertIn("momentum", signals)

    def test_original_snapshot_not_mutated(self):
        snap = _snap(rsi_14=30, bb_pct=0.20, vol_ratio=1.2)
        prefilter_candidates([snap])
        self.assertNotIn("matched_signals", snap)


class TestScoreCandidate(unittest.TestCase):
    def test_returns_float(self):
        snap = _snap(rsi_14=30, bb_pct=0.20, vol_ratio=1.5, matched_signals=["mean_reversion"])
        self.assertIsInstance(score_candidate(snap), float)

    def test_score_between_zero_and_one(self):
        for snap in [
            _snap(
                rsi_14=30,
                bb_pct=0.10,
                vol_ratio=2.0,
                rel_strength_5d=5.0,
                matched_signals=["mean_reversion"] * 8,
            ),
            _snap(),
        ]:
            result = score_candidate(snap)
            self.assertGreaterEqual(result, 0.0)
            self.assertLessEqual(result, 1.0)

    def test_higher_vol_ratio_scores_higher(self):
        low = score_candidate(_snap(vol_ratio=1.1, matched_signals=["momentum"]))
        high = score_candidate(_snap(vol_ratio=2.0, matched_signals=["momentum"]))
        self.assertGreater(high, low)

    def test_more_signals_scores_higher(self):
        one = score_candidate(_snap(matched_signals=["momentum"]))
        three = score_candidate(_snap(matched_signals=["momentum", "mean_reversion", "rs_leader"]))
        self.assertGreater(three, one)

    def test_missing_matched_signals_key_handled(self):
        snap = _snap()
        snap.pop("matched_signals", None)
        result = score_candidate(snap)
        self.assertIsInstance(result, float)

    def test_deterministic_same_input(self):
        snap = _snap(rsi_14=35, bb_pct=0.3, vol_ratio=1.4, matched_signals=["momentum"])
        self.assertEqual(score_candidate(snap), score_candidate(snap))


class TestGetTopMovers(unittest.TestCase):
    def _make_data(self, symbols, n_rows=5):
        idx = pd.date_range("2026-01-01", periods=n_rows, freq="B")
        closes = pd.DataFrame({sym: [100 + i for i in range(n_rows)] for sym in symbols}, index=idx)
        volumes = pd.DataFrame({sym: [1_000_000] * n_rows for sym in symbols}, index=idx)
        mock = MagicMock()
        mock.empty = False
        mock.__len__ = MagicMock(return_value=n_rows)
        mock.__getitem__ = lambda self, key: closes if key == "Close" else volumes
        return mock

    def test_returns_list_on_success(self):
        from execution.stock_scanner import get_top_movers

        syms = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN"]
        with patch("execution.stock_scanner.yf.download", return_value=self._make_data(syms)):
            result = get_top_movers(n=3)
        self.assertIsInstance(result, list)

    def test_returns_empty_on_exception(self):
        from execution.stock_scanner import get_top_movers

        with patch("execution.stock_scanner.yf.download", side_effect=Exception("network error")):
            result = get_top_movers()
        self.assertEqual(result, [])

    def test_returns_empty_on_empty_data(self):
        from execution.stock_scanner import get_top_movers

        mock = MagicMock()
        mock.empty = True
        with patch("execution.stock_scanner.yf.download", return_value=mock):
            result = get_top_movers()
        self.assertEqual(result, [])

    def test_returns_empty_when_insufficient_rows(self):
        from execution.stock_scanner import get_top_movers

        mock = MagicMock()
        mock.empty = False
        mock.__len__ = MagicMock(return_value=1)
        with patch("execution.stock_scanner.yf.download", return_value=mock):
            result = get_top_movers()
        self.assertEqual(result, [])


class TestPassesQualityScreen(unittest.TestCase):
    """_passes_quality_screen: fundamental quality gate."""

    def test_passes_when_all_fields_absent(self):
        self.assertTrue(_passes_quality_screen({}))

    def test_passes_when_only_debt_equity_present(self):
        # roe and profit_margin both absent → permissive pass regardless of d/e
        self.assertTrue(_passes_quality_screen({"debt_to_equity": 50.0}))

    def test_passes_positive_roe(self):
        self.assertTrue(_passes_quality_screen({"roe": 0.15, "profit_margin": 0.10}))

    def test_fails_negative_roe(self):
        self.assertFalse(_passes_quality_screen({"roe": -0.05, "profit_margin": 0.10}))

    def test_fails_negative_profit_margin(self):
        self.assertFalse(_passes_quality_screen({"roe": 0.10, "profit_margin": -0.02}))

    def test_fails_excessive_debt_to_equity(self):
        self.assertFalse(
            _passes_quality_screen({"roe": 0.10, "profit_margin": 0.05, "debt_to_equity": 350.0})
        )

    def test_passes_high_but_acceptable_debt_to_equity(self):
        # Financials/REITs can have D/E up to 300
        self.assertTrue(
            _passes_quality_screen({"roe": 0.10, "profit_margin": 0.05, "debt_to_equity": 280.0})
        )

    def test_passes_when_profit_margin_absent_but_roe_positive(self):
        self.assertTrue(_passes_quality_screen({"roe": 0.12}))

    def test_fails_when_profit_margin_absent_but_roe_negative(self):
        self.assertFalse(_passes_quality_screen({"roe": -0.01}))

    def test_prefilter_rejects_stock_with_negative_roe(self):
        snap = _snap(rsi_14=30, bb_pct=0.20, vol_ratio=1.2, roe=-0.05, profit_margin=0.05)
        self.assertEqual(prefilter_candidates([snap]), [])

    def test_prefilter_passes_stock_without_quality_fields(self):
        snap = _snap(rsi_14=30, bb_pct=0.10, vol_ratio=1.3)
        result = prefilter_candidates([snap])
        self.assertEqual(len(result), 1)


class TestPeadSignal(unittest.TestCase):
    """prefilter_candidates: pead signal — post-earnings announcement drift."""

    def test_pead_fires_with_beat_and_positive_return(self):
        snap = _snap(pead_candidate=True, ret_5d_pct=2.5)
        result = prefilter_candidates([snap])
        self.assertEqual(len(result), 1)
        self.assertIn("pead", result[0]["matched_signals"])

    def test_pead_absent_when_no_beat(self):
        # pead_candidate not set → signal absent; no other signal → rejected
        snap = _snap(ret_5d_pct=3.0)
        result = prefilter_candidates([snap])
        self.assertEqual(result, [])

    def test_pead_absent_when_negative_return(self):
        # Beat happened but price is now drifting down → no signal
        snap = _snap(pead_candidate=True, ret_5d_pct=-1.0)
        result = prefilter_candidates([snap])
        self.assertEqual(result, [])

    def test_pead_fires_against_weekly_downtrend(self):
        # PEAD bypasses the weekly trend filter (fundamental conviction)
        snap = _snap(pead_candidate=True, ret_5d_pct=2.5, weekly_trend_up=False)
        result = prefilter_candidates([snap])
        self.assertEqual(len(result), 1)
        self.assertIn("pead", result[0]["matched_signals"])

    def test_pead_absent_when_pead_candidate_is_false(self):
        snap = _snap(pead_candidate=False, ret_5d_pct=3.0)
        result = prefilter_candidates([snap])
        self.assertEqual(result, [])


class TestIvCompressionSignal(unittest.TestCase):
    """prefilter_candidates: iv_compression — historical volatility percentile squeeze."""

    def test_iv_compression_fires_with_ema_confirmation(self):
        snap = _snap(hv_rank=0.05, ema9_above_ema21=True, vol_ratio=1.3)
        result = prefilter_candidates([snap])
        self.assertEqual(len(result), 1)
        self.assertIn("iv_compression", result[0]["matched_signals"])

    def test_iv_compression_fires_with_macd_confirmation(self):
        snap = _snap(hv_rank=0.05, ema9_above_ema21=False, macd_diff=0.05, vol_ratio=1.3)
        result = prefilter_candidates([snap])
        self.assertEqual(len(result), 1)
        self.assertIn("iv_compression", result[0]["matched_signals"])

    def test_iv_compression_absent_when_hv_rank_above_threshold(self):
        # hv_rank=0.50 → mid-range vol; no other signals → rejected
        snap = _snap(hv_rank=0.50, ema9_above_ema21=True, vol_ratio=1.5)
        result = prefilter_candidates([snap])
        self.assertEqual(result, [])

    def test_iv_compression_absent_without_directional_confirmation(self):
        # hv_rank is low but neither EMA nor MACD confirms direction → no signal
        snap = _snap(hv_rank=0.10, ema9_above_ema21=False, macd_diff=-0.1, vol_ratio=1.2)
        result = prefilter_candidates([snap])
        self.assertEqual(result, [])

    def test_iv_compression_absent_without_volume(self):
        # hv_rank is low, EMA confirms, but volume is below threshold
        snap = _snap(hv_rank=0.10, ema9_above_ema21=True, vol_ratio=1.0)
        result = prefilter_candidates([snap])
        self.assertEqual(result, [])


class TestRegimeBlocking(unittest.TestCase):
    """prefilter_candidates: regime-conditional signal suppression."""

    # ── CHOPPY blocks ────────────────────────────────────────────────────────

    def test_choppy_filters_insufficient_signals(self):
        # Boundary snap (vol below threshold, rsi at boundary) — no signal fires
        snap = _snap(rsi_14=35, bb_pct=0.25, vol_ratio=1.1)
        result = prefilter_candidates([snap], regime="CHOPPY")
        self.assertEqual(result, [])

    def test_choppy_blocks_macd_crossover(self):
        snap = _snap(macd_crossed_up=True, vol_ratio=1.3)
        result = prefilter_candidates([snap], regime="CHOPPY")
        self.assertEqual(result, [])

    def test_choppy_blocks_inside_day_breakout(self):
        snap = _snap(is_inside_day=True, ema9_above_ema21=True, vol_ratio=1.2)
        result = prefilter_candidates([snap], regime="CHOPPY")
        self.assertEqual(result, [])

    def test_choppy_blocks_momentum(self):
        snap = _snap(ema9_above_ema21=True, macd_diff=0.5, ret_5d_pct=1.5, vol_ratio=1.4)
        result = prefilter_candidates([snap], regime="CHOPPY")
        self.assertEqual(result, [])

    def test_choppy_blocks_mean_reversion(self):
        # mean_reversion blocked in CHOPPY/NEUTRAL_CHOP: WR 49%, avg -0.1%, n=687 (p>0.05)
        snap = _snap(rsi_14=30, bb_pct=0.20, vol_ratio=1.3)
        result = prefilter_candidates([snap], regime="CHOPPY")
        self.assertEqual(result, [])

    # ── BEAR_DAY blocks ──────────────────────────────────────────────────────

    def test_bear_day_blocks_iv_compression(self):
        snap = _snap(hv_rank=0.10, ema9_above_ema21=True, vol_ratio=1.2)
        result = prefilter_candidates([snap], regime="BEAR_DAY")
        self.assertEqual(result, [])

    def test_bear_day_blocks_mean_reversion(self):
        # mean_reversion blocked in BEAR_DAY/STRESS_RISK_OFF: WR 47%, p>0.05 (n=129)
        snap = _snap(rsi_14=32, bb_pct=0.22, vol_ratio=1.3)
        result = prefilter_candidates([snap], regime="BEAR_DAY")
        self.assertEqual(result, [])

    # ── No blocking when regime is None or unrecognised ──────────────────────

    def test_no_regime_does_not_block_mean_reversion(self):
        snap = _snap(rsi_14=32, bb_pct=0.10, vol_ratio=1.3)
        result = prefilter_candidates([snap])
        self.assertEqual(len(result), 1)
        self.assertIn("mean_reversion", result[0]["matched_signals"])

    def test_bull_trending_blocks_rs_leader(self):
        # rs_leader blocked in BULL_TRENDING/BULL_TREND: WR 51%, avg -0.13%, n=246 (no edge)
        snap = _snap(rsi_14=32, bb_pct=0.10, vol_ratio=1.3)
        result = prefilter_candidates([snap], regime="BULL_TRENDING")
        self.assertEqual(len(result), 1)
        self.assertIn("mean_reversion", result[0]["matched_signals"])

    def test_range_reversion_blocked_in_neutral_chop(self):
        # range_reversion blocked in CHOPPY/NEUTRAL_CHOP: WR 46%, p>0.05, n=52
        snap = _snap(rsi_14=25, bb_pct=0.05, adx=15)
        result = prefilter_candidates([snap], regime="CHOPPY")
        self.assertEqual(result, [])

    def test_iv_compression_blocked_in_neutral_chop(self):
        # iv_compression blocked in NEUTRAL_CHOP: WR 51%, avg +0.0%, n=506
        snap = _snap(hv_rank=0.10, ema9_above_ema21=True, vol_ratio=1.2)
        result = prefilter_candidates([snap], regime="NEUTRAL_CHOP")
        signals = result[0]["matched_signals"] if result else []
        self.assertNotIn("iv_compression", signals)

    def test_rsi_divergence_globally_disabled(self):
        from signals.evaluator import GLOBALLY_DISABLED

        self.assertIn("rsi_divergence", GLOBALLY_DISABLED)
        snap = _snap(rsi_divergence=True, adx=20, rsi_14=38, vol_ratio=1.2, bb_pct=0.20)
        result = prefilter_candidates([snap], regime="NEUTRAL_CHOP")
        signals = result[0]["matched_signals"] if result else []
        self.assertNotIn("rsi_divergence", signals)

    def test_rsi_divergence_blocked_in_bull_trend(self):
        snap = _snap(rsi_divergence=True, adx=20, rsi_14=38, vol_ratio=1.2)
        result = prefilter_candidates([snap], regime="BULL_TRENDING")
        signals = result[0]["matched_signals"] if result else []
        self.assertNotIn("rsi_divergence", signals)


class TestEvaluateSignalsNoneGuard(unittest.TestCase):
    """evaluate_signals must not crash when snapshot fields are None (market_data gaps)."""

    def test_none_gap_pct_does_not_crash(self):
        from signals.evaluator import evaluate_signals

        snap = _snap(gap_pct=None)
        result = evaluate_signals(snap)
        self.assertIsInstance(result, list)

    def test_none_ret_5d_does_not_crash(self):
        from signals.evaluator import evaluate_signals

        snap = _snap(ret_5d_pct=None)
        result = evaluate_signals(snap)
        self.assertIsInstance(result, list)

    def test_none_rsi_does_not_crash(self):
        from signals.evaluator import evaluate_signals

        snap = _snap(rsi_14=None)
        result = evaluate_signals(snap)
        self.assertIsInstance(result, list)

    def test_none_vol_ratio_does_not_crash(self):
        from signals.evaluator import evaluate_signals

        snap = _snap(vol_ratio=None)
        result = evaluate_signals(snap)
        self.assertIsInstance(result, list)

    def test_all_numeric_fields_none_does_not_crash(self):
        from signals.evaluator import evaluate_signals

        snap = _snap(
            rsi_14=None,
            bb_pct=None,
            vol_ratio=None,
            macd_diff=None,
            gap_pct=None,
            ret_5d_pct=None,
            ret_10d_pct=None,
            price_vs_ema21_pct=None,
            price_vs_52w_high_pct=None,
            hv_rank=None,
            pct_vs_vwap=None,
        )
        result = evaluate_signals(snap)
        self.assertIsInstance(result, list)


class TestEvaluateSignalsSignalPaths(unittest.TestCase):
    """Coverage for evaluate_signals signal paths not hit by existing tests."""

    def test_breakout_52w_in_globally_disabled(self):
        from signals.evaluator import GLOBALLY_DISABLED

        self.assertIn("breakout_52w", GLOBALLY_DISABLED)

    def test_rsi_divergence_in_globally_disabled(self):
        from signals.evaluator import GLOBALLY_DISABLED

        self.assertIn("rsi_divergence", GLOBALLY_DISABLED)


# ── Extended signal wiring (v1.77) ───────────────────────────────────────────


class TestActivistFilingSignal(unittest.TestCase):
    """activist_filing=True fires insider_buying even when insider_cluster=False."""

    def test_activist_fires_insider_buying(self):
        snap = _snap(activist_filing=True, insider_cluster=False)
        result = prefilter_candidates([snap])
        self.assertEqual(len(result), 1)
        self.assertIn("insider_buying", result[0]["matched_signals"])

    def test_activist_absent_no_signal(self):
        snap = _snap(activist_filing=False, insider_cluster=False)
        result = prefilter_candidates([snap])
        self.assertEqual(len(result), 0)

    def test_activist_and_cluster_both_true_fires_once(self):
        snap = _snap(activist_filing=True, insider_cluster=True)
        result = prefilter_candidates([snap])
        signals = result[0]["matched_signals"]
        self.assertEqual(signals.count("insider_buying"), 1)


class TestGuidancePositiveSignal(unittest.TestCase):
    """guidance_positive=True fires pead when ret_5d > 0."""

    def test_guidance_positive_fires_pead(self):
        snap = _snap(guidance_positive=True, pead_candidate=False, ret_5d_pct=1.5)
        result = prefilter_candidates([snap])
        self.assertEqual(len(result), 1)
        self.assertIn("pead", result[0]["matched_signals"])

    def test_guidance_positive_with_negative_return_no_signal(self):
        snap = _snap(guidance_positive=True, pead_candidate=False, ret_5d_pct=-1.0)
        result = prefilter_candidates([snap])
        self.assertEqual(len(result), 0)

    def test_guidance_positive_and_pead_candidate_fires_once(self):
        snap = _snap(guidance_positive=True, pead_candidate=True, ret_5d_pct=2.0)
        result = prefilter_candidates([snap])
        signals = result[0]["matched_signals"]
        self.assertEqual(signals.count("pead"), 1)


class TestIvCheapSignal(unittest.TestCase):
    """iv_cheap=True fires iv_compression even when hv_rank is above threshold."""

    def test_iv_cheap_fires_iv_compression(self):
        # hv_rank above threshold but iv_cheap=True — should still fire
        snap = _snap(hv_rank=0.50, iv_cheap=True, ema9_above_ema21=True, vol_ratio=1.3)
        result = prefilter_candidates([snap])
        self.assertEqual(len(result), 1)
        self.assertIn("iv_compression", result[0]["matched_signals"])

    def test_iv_cheap_false_hv_rank_high_no_signal(self):
        snap = _snap(hv_rank=0.50, iv_cheap=False, ema9_above_ema21=True, vol_ratio=1.3)
        result = prefilter_candidates([snap])
        self.assertEqual(len(result), 0)

    def test_iv_cheap_without_directional_confirmation_no_signal(self):
        snap = _snap(
            hv_rank=0.50, iv_cheap=True, ema9_above_ema21=False, macd_diff=-0.1, vol_ratio=1.3
        )
        result = prefilter_candidates([snap])
        self.assertEqual(len(result), 0)


class TestGuidanceDowngradeShortSignal(unittest.TestCase):
    """guidance_downgrade short signal fires on guidance_negative=True."""

    def test_guidance_downgrade_fires(self):
        from signals.evaluator import evaluate_short_signals

        snap = {"guidance_negative": True}
        signals = evaluate_short_signals(snap)
        self.assertIn("guidance_downgrade", signals)

    def test_guidance_downgrade_absent_when_false(self):
        from signals.evaluator import evaluate_short_signals

        snap = {"guidance_negative": False}
        signals = evaluate_short_signals(snap)
        self.assertNotIn("guidance_downgrade", signals)

    def test_guidance_downgrade_blocked_when_in_blocked_set(self):
        from signals.evaluator import evaluate_short_signals

        snap = {"guidance_negative": True}
        signals = evaluate_short_signals(snap, blocked=frozenset({"guidance_downgrade"}))
        self.assertNotIn("guidance_downgrade", signals)

    def test_guidance_downgrade_in_short_signal_priority(self):
        from signals.evaluator import SHORT_SIGNAL_PRIORITY

        self.assertIn("guidance_downgrade", SHORT_SIGNAL_PRIORITY)


class TestSecondaryOfferingShortSignal(unittest.TestCase):
    """secondary_offering_short fires on secondary_offering=True."""

    def test_secondary_offering_short_fires(self):
        from signals.evaluator import evaluate_short_signals

        snap = {"secondary_offering": True}
        signals = evaluate_short_signals(snap)
        self.assertIn("secondary_offering_short", signals)

    def test_secondary_offering_short_absent_when_false(self):
        from signals.evaluator import evaluate_short_signals

        snap = {"secondary_offering": False}
        signals = evaluate_short_signals(snap)
        self.assertNotIn("secondary_offering_short", signals)

    def test_secondary_offering_short_blocked_when_in_blocked_set(self):
        from signals.evaluator import evaluate_short_signals

        snap = {"secondary_offering": True}
        signals = evaluate_short_signals(snap, blocked=frozenset({"secondary_offering_short"}))
        self.assertNotIn("secondary_offering_short", signals)

    def test_secondary_offering_short_in_priority(self):
        from signals.evaluator import SHORT_SIGNAL_PRIORITY

        self.assertIn("secondary_offering_short", SHORT_SIGNAL_PRIORITY)


class TestGloballyDisabledSignals(unittest.TestCase):
    """Signals in GLOBALLY_DISABLED / SHORT_GLOBALLY_DISABLED never fire."""

    def test_faded_earnings_gap_up_in_short_globally_disabled(self):
        from signals.evaluator import SHORT_GLOBALLY_DISABLED

        self.assertIn("faded_earnings_gap_up", SHORT_GLOBALLY_DISABLED)

    def test_faded_earnings_gap_up_never_fires(self):
        from signals.evaluator import evaluate_short_signals

        snap = {"faded_earnings_gap_up_pct": 6.0, "close_pct_of_range": 0.20, "vol_ratio": 2.0}
        self.assertNotIn("faded_earnings_gap_up", evaluate_short_signals(snap))

    def test_vix_fear_reversion_in_globally_disabled(self):
        from signals.evaluator import GLOBALLY_DISABLED

        self.assertIn("vix_fear_reversion", GLOBALLY_DISABLED)

    def test_vix_fear_reversion_never_fires(self):
        from signals.evaluator import evaluate_signals

        snap = _snap(vol_ratio=2.0)
        self.assertNotIn("vix_fear_reversion", evaluate_signals(snap, vix_spike=True))
