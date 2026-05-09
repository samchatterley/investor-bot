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


def _spy_history(prices: list[float]) -> pd.DataFrame:
    """Build a minimal history DataFrame from a list of close prices."""
    return pd.DataFrame({"Close": prices, "Volume": [1_000_000] * len(prices)})


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
    def _mock_spy(self, prices):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = _spy_history(prices)
        return patch("execution.stock_scanner.yf.Ticker", return_value=mock_ticker)

    def test_bull_trending_regime(self):
        # 5-day gain > 2%, today positive
        prices = [100, 101, 102, 103, 104, 106]
        with self._mock_spy(prices):
            result = get_market_regime(threshold_pct=-1.5)
        self.assertEqual(result["regime"], "BULL_TRENDING")
        self.assertFalse(result["is_bearish"])

    def test_bear_day_regime(self):
        # Today drops > 1.5%
        prices = [100, 100, 100, 100, 100, 98]
        with self._mock_spy(prices):
            result = get_market_regime(threshold_pct=-1.5)
        self.assertEqual(result["regime"], "BEAR_DAY")
        self.assertTrue(result["is_bearish"])

    def test_high_vol_regime(self):
        # VIX > 25, 5-day return < -3%
        prices = [100, 100, 100, 100, 100, 96]
        # spy_1d = -4% → triggers BEAR_DAY first; we need 1d flat but 5d down
        prices = [100, 96, 96, 96, 96, 96]
        with self._mock_spy(prices):
            result = get_market_regime(threshold_pct=-1.5, vix=30.0)
        # 1d = 0%, 5d ≈ -4%, VIX 30 → HIGH_VOL
        self.assertEqual(result["regime"], "HIGH_VOL")

    def test_choppy_regime(self):
        prices = [100, 100, 100, 100, 100, 100]
        with self._mock_spy(prices):
            result = get_market_regime(threshold_pct=-1.5)
        self.assertEqual(result["regime"], "CHOPPY")

    def test_insufficient_history_returns_unknown(self):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = _spy_history([100, 101])
        with patch("execution.stock_scanner.yf.Ticker", return_value=mock_ticker):
            result = get_market_regime()
        self.assertEqual(result["regime"], "UNKNOWN")

    def test_exception_returns_unknown(self):
        with patch("execution.stock_scanner.yf.Ticker", side_effect=Exception("network")):
            result = get_market_regime()
        self.assertEqual(result["regime"], "UNKNOWN")
        self.assertFalse(result["is_bearish"])

    def test_result_has_required_keys(self):
        prices = [100] * 6
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = _spy_history(prices)
        with patch("execution.stock_scanner.yf.Ticker", return_value=mock_ticker):
            result = get_market_regime()
        for key in ("is_bearish", "spy_change_pct", "spy_5d_pct", "regime"):
            self.assertIn(key, result)


class TestPrefilterCandidates(unittest.TestCase):
    def test_momentum_signal_passes(self):
        snap = _snap(ema9_above_ema21=True, macd_diff=0.5, ret_5d_pct=2.0, vol_ratio=1.5)
        result = prefilter_candidates([snap])
        self.assertEqual(len(result), 1)

    def test_mean_reversion_signal_passes(self):
        snap = _snap(rsi_14=30, bb_pct=0.20, vol_ratio=1.2)
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
        good = _snap(symbol="GOOD", rsi_14=30, bb_pct=0.20, vol_ratio=1.2)
        bad = _snap(symbol="BAD")
        result = prefilter_candidates([good, bad])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["symbol"], "GOOD")

    # ── bb_squeeze_breakout ──────────────────────────────────────────────────

    def test_bb_squeeze_with_ema_up_and_volume_passes(self):
        snap = _snap(bb_squeeze=True, ema9_above_ema21=True, vol_ratio=1.5)
        self.assertEqual(len(prefilter_candidates([snap])), 1)

    def test_bb_squeeze_with_positive_macd_passes(self):
        snap = _snap(bb_squeeze=True, macd_diff=0.3, vol_ratio=1.3)
        self.assertEqual(len(prefilter_candidates([snap])), 1)

    def test_bb_squeeze_without_directional_confirmation_fails(self):
        # bb_squeeze True but ema not up and macd_diff <= 0
        snap = _snap(bb_squeeze=True, ema9_above_ema21=False, macd_diff=-0.1, vol_ratio=1.5)
        self.assertEqual(len(prefilter_candidates([snap])), 0)

    def test_bb_squeeze_without_volume_fails(self):
        snap = _snap(bb_squeeze=True, ema9_above_ema21=True, vol_ratio=0.9)
        self.assertEqual(len(prefilter_candidates([snap])), 0)

    # ── breakout_52w ─────────────────────────────────────────────────────────

    def test_breakout_52w_near_high_with_volume_passes(self):
        snap = _snap(price_vs_52w_high_pct=-1.5, vol_ratio=1.5, weekly_trend_up=True)
        self.assertEqual(len(prefilter_candidates([snap])), 1)

    def test_breakout_52w_at_high_passes(self):
        snap = _snap(price_vs_52w_high_pct=0.0, vol_ratio=1.3, weekly_trend_up=True)
        self.assertEqual(len(prefilter_candidates([snap])), 1)

    def test_breakout_52w_too_far_from_high_fails(self):
        snap = _snap(price_vs_52w_high_pct=-5.0, vol_ratio=1.5, weekly_trend_up=True)
        self.assertEqual(len(prefilter_candidates([snap])), 0)

    def test_breakout_52w_against_weekly_trend_fails(self):
        snap = _snap(price_vs_52w_high_pct=-1.0, vol_ratio=1.5, weekly_trend_up=False)
        self.assertEqual(len(prefilter_candidates([snap])), 0)

    def test_breakout_52w_low_volume_fails(self):
        snap = _snap(price_vs_52w_high_pct=-1.0, vol_ratio=0.9, weekly_trend_up=True)
        self.assertEqual(len(prefilter_candidates([snap])), 0)

    # ── rs_leader ────────────────────────────────────────────────────────────

    def test_rs_leader_strong_outperformance_ema_up_passes(self):
        snap = _snap(rel_strength_5d=3.0, rel_strength_10d=4.0, ema9_above_ema21=True)
        self.assertEqual(len(prefilter_candidates([snap])), 1)

    def test_rs_leader_weak_5d_fails(self):
        snap = _snap(rel_strength_5d=1.0, rel_strength_10d=4.0, ema9_above_ema21=True)
        self.assertEqual(len(prefilter_candidates([snap])), 0)

    def test_rs_leader_weak_10d_fails(self):
        snap = _snap(rel_strength_5d=3.0, rel_strength_10d=2.0, ema9_above_ema21=True)
        self.assertEqual(len(prefilter_candidates([snap])), 0)

    def test_rs_leader_ema_not_up_fails(self):
        snap = _snap(rel_strength_5d=3.0, rel_strength_10d=4.0, ema9_above_ema21=False)
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
        snap = _snap(ema9_above_ema21=True, price_vs_ema21_pct=-0.6, rsi_14=45, vol_ratio=1.1)
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
        snap = _snap(insider_cluster=True, rsi_14=30, bb_pct=0.20, vol_ratio=1.2)
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
        snap = _snap(rsi_14=30, bb_pct=0.20, vol_ratio=1.2)
        result = prefilter_candidates([snap])
        self.assertIn("matched_signals", result[0])

    def test_mean_reversion_signal_annotated(self):
        snap = _snap(rsi_14=30, bb_pct=0.20, vol_ratio=1.2)
        result = prefilter_candidates([snap])
        self.assertIn("mean_reversion", result[0]["matched_signals"])

    def test_momentum_signal_annotated(self):
        snap = _snap(ema9_above_ema21=True, macd_diff=0.5, ret_5d_pct=2.0, vol_ratio=1.5)
        result = prefilter_candidates([snap])
        self.assertIn("momentum", result[0]["matched_signals"])

    def test_multiple_signals_all_annotated(self):
        snap = _snap(
            rsi_14=30,
            bb_pct=0.20,
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
        snap = _snap(rsi_14=30, bb_pct=0.20, vol_ratio=1.2)
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
        snap = _snap(hv_rank=0.10, ema9_above_ema21=True, vol_ratio=1.2)
        result = prefilter_candidates([snap])
        self.assertEqual(len(result), 1)
        self.assertIn("iv_compression", result[0]["matched_signals"])

    def test_iv_compression_fires_with_macd_confirmation(self):
        snap = _snap(hv_rank=0.15, ema9_above_ema21=False, macd_diff=0.05, vol_ratio=1.15)
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
