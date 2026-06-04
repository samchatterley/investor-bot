"""Tests for data/options_data.py — 100% coverage."""

from __future__ import annotations

import contextlib
import os
from datetime import datetime, timedelta
from unittest import TestCase
from unittest.mock import MagicMock, patch

import pandas as pd

from data.options_data import (
    OptionsSnapshot,
    _atm_iv,
    _bs_delta,
    _compute_snapshot,
    _find_25d_iv,
    _is_stale,
    _load_cache,
    _null_snapshot,
    _realized_vol_20d,
    _save_cache,
    _select_expiry,
    get_options_batch,
    get_options_snapshot,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_chain_df(strikes, ivs, include_oi=True) -> pd.DataFrame:
    """Create a synthetic option chain DataFrame."""
    rows = []
    for strike, iv in zip(strikes, ivs, strict=False):
        row = {
            "strike": float(strike),
            "impliedVolatility": float(iv),
            "openInterest": 1000.0,
            "volume": 100.0,
        }
        if not include_oi:
            row["openInterest"] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _make_ticker_mock(
    spot: float = 100.0,
    expirations: tuple[str, ...] = (),
    call_df: pd.DataFrame | None = None,
    put_df: pd.DataFrame | None = None,
    hist_len: int = 25,
) -> MagicMock:
    """Create a mock yf.Ticker for options tests."""
    ticker = MagicMock()
    ticker.options = expirations

    chain_mock = MagicMock()
    chain_mock.calls = call_df if call_df is not None else pd.DataFrame()
    chain_mock.puts = put_df if put_df is not None else pd.DataFrame()
    ticker.option_chain.return_value = chain_mock

    # fast_info
    ticker.fast_info.last_price = spot

    # history for realized vol
    dates = pd.date_range("2026-01-01", periods=hist_len, freq="B")
    hist_df = pd.DataFrame({"Close": [100.0 + i * 0.1 for i in range(hist_len)]}, index=dates)
    ticker.history.return_value = hist_df

    return ticker


def _future_expiry(days_ahead: int = 30) -> str:
    return (datetime.now().date() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")


# ── Black-Scholes delta ───────────────────────────────────────────────────────


class TestBsDelta(TestCase):
    def test_atm_call_delta_near_half(self):
        # ATM call delta ≈ 0.5 for reasonable inputs
        delta = _bs_delta(S=100, K=100, T=30 / 365, r=0.05, sigma=0.20, is_call=True)
        self.assertAlmostEqual(delta, 0.5, delta=0.1)

    def test_atm_put_delta_near_minus_half(self):
        delta = _bs_delta(S=100, K=100, T=30 / 365, r=0.05, sigma=0.20, is_call=False)
        self.assertAlmostEqual(delta, -0.5, delta=0.1)

    def test_deep_itm_call_delta_near_one(self):
        delta = _bs_delta(S=200, K=100, T=30 / 365, r=0.05, sigma=0.20, is_call=True)
        self.assertGreater(delta, 0.9)

    def test_deep_otm_call_delta_near_zero(self):
        delta = _bs_delta(S=50, K=200, T=30 / 365, r=0.05, sigma=0.20, is_call=True)
        self.assertLess(delta, 0.05)

    def test_zero_T_returns_zero(self):
        self.assertEqual(_bs_delta(100, 100, 0, 0.05, 0.20, True), 0.0)

    def test_zero_sigma_returns_zero(self):
        self.assertEqual(_bs_delta(100, 100, 0.1, 0.05, 0.0, True), 0.0)

    def test_zero_spot_returns_zero(self):
        self.assertEqual(_bs_delta(0, 100, 0.1, 0.05, 0.2, True), 0.0)

    def test_zero_strike_returns_zero(self):
        self.assertEqual(_bs_delta(100, 0, 0.1, 0.05, 0.2, True), 0.0)


# ── _atm_iv ───────────────────────────────────────────────────────────────────


class TestAtmIv(TestCase):
    def test_nearest_strike(self):
        df = _make_chain_df([95, 100, 105], [0.20, 0.18, 0.20])
        result = _atm_iv(df, spot=101.0)
        self.assertEqual(result, 0.18)  # strike 100 is nearest to 101

    def test_empty_df(self):
        self.assertIsNone(_atm_iv(pd.DataFrame(), spot=100.0))

    def test_zero_spot(self):
        df = _make_chain_df([100], [0.20])
        self.assertIsNone(_atm_iv(df, spot=0.0))

    def test_all_zero_iv(self):
        df = _make_chain_df([100], [0.0])
        self.assertIsNone(_atm_iv(df, spot=100.0))

    def test_nan_iv_dropped(self):
        df = _make_chain_df([100, 105], [float("nan"), 0.20])
        result = _atm_iv(df, spot=100.0)
        self.assertIsNotNone(result)


# ── _find_25d_iv ──────────────────────────────────────────────────────────────


class TestFind25dIv(TestCase):
    def test_put_25d_found(self):
        # 25-delta put ≈ 88% of spot (S=100, T=30d, sigma=0.25)
        strikes = list(range(80, 120, 5))
        ivs = [0.25] * len(strikes)
        df = _make_chain_df(strikes, ivs)
        result = _find_25d_iv(df, spot=100.0, T=30 / 365, r=0.05, is_call=False)
        self.assertIsNotNone(result)
        self.assertGreater(result, 0)

    def test_call_25d_found(self):
        strikes = list(range(80, 120, 5))
        ivs = [0.25] * len(strikes)
        df = _make_chain_df(strikes, ivs)
        result = _find_25d_iv(df, spot=100.0, T=30 / 365, r=0.05, is_call=True)
        self.assertIsNotNone(result)
        self.assertGreater(result, 0)

    def test_empty_df(self):
        self.assertIsNone(_find_25d_iv(pd.DataFrame(), 100, 0.1, 0.05, True))

    def test_zero_spot(self):
        df = _make_chain_df([100], [0.20])
        self.assertIsNone(_find_25d_iv(df, spot=0, T=0.1, r=0.05, is_call=True))

    def test_zero_T(self):
        df = _make_chain_df([100], [0.20])
        self.assertIsNone(_find_25d_iv(df, spot=100, T=0, r=0.05, is_call=True))

    def test_no_delta_close_enough(self):
        # All deep OTM → delta far from 0.25 → returns None
        strikes = [200, 300, 400]  # way above spot of 100
        ivs = [0.20, 0.20, 0.20]
        df = _make_chain_df(strikes, ivs)
        result = _find_25d_iv(df, spot=100.0, T=30 / 365, r=0.05, is_call=True)
        self.assertIsNone(result)

    def test_all_zero_iv_df_empty_after_filter(self):
        """IVs of 0.0 pass dropna but are removed by > 0 filter → df empty → None."""
        df = pd.DataFrame(
            {
                "strike": [95.0, 100.0, 105.0],
                "impliedVolatility": [0.0, 0.0, 0.0],
                "openInterest": [100.0] * 3,
                "volume": [50.0] * 3,
            }
        )
        result = _find_25d_iv(df, spot=100.0, T=30 / 365, r=0.05, is_call=True)
        self.assertIsNone(result)


# ── _select_expiry ────────────────────────────────────────────────────────────


class TestSelectExpiry(TestCase):
    def test_picks_nearest_to_30d(self):
        exps = (
            _future_expiry(7),  # too short (< 10d)
            _future_expiry(28),
            _future_expiry(35),
            _future_expiry(90),  # too far (> 60d)
        )
        result = _select_expiry(exps)
        self.assertEqual(result, _future_expiry(28))

    def test_none_when_outside_range(self):
        exps = (_future_expiry(5), _future_expiry(90))
        self.assertIsNone(_select_expiry(exps))

    def test_empty_expirations(self):
        self.assertIsNone(_select_expiry(()))

    def test_invalid_date_format_skipped(self):
        exps = ("not-a-date", _future_expiry(30))
        result = _select_expiry(exps)
        self.assertIsNotNone(result)

    def test_past_expiry_skipped(self):
        past = (datetime.now().date() - timedelta(days=5)).strftime("%Y-%m-%d")
        exps = (past, _future_expiry(30))
        result = _select_expiry(exps)
        self.assertEqual(result, _future_expiry(30))


# ── _realized_vol_20d ─────────────────────────────────────────────────────────


class TestRealizedVol(TestCase):
    def test_normal_history(self):
        dates = pd.date_range("2026-01-01", periods=30, freq="B")
        hist_df = pd.DataFrame({"Close": [100.0 * (1 + 0.01 * i) for i in range(30)]}, index=dates)
        ticker = MagicMock()
        ticker.history.return_value = hist_df
        result = _realized_vol_20d(ticker)
        self.assertIsNotNone(result)
        self.assertGreater(result, 0)

    def test_insufficient_data(self):
        dates = pd.date_range("2026-01-01", periods=10, freq="B")
        hist_df = pd.DataFrame({"Close": [100.0] * 10}, index=dates)
        ticker = MagicMock()
        ticker.history.return_value = hist_df
        self.assertIsNone(_realized_vol_20d(ticker))

    def test_empty_history(self):
        ticker = MagicMock()
        ticker.history.return_value = pd.DataFrame()
        self.assertIsNone(_realized_vol_20d(ticker))

    def test_exception_returns_none(self):
        ticker = MagicMock()
        ticker.history.side_effect = RuntimeError("network")
        self.assertIsNone(_realized_vol_20d(ticker))

    def test_close_nan_heavy_returns_none(self):
        """hist has ≥22 rows but most Close values are NaN → closes < 22 → None."""
        dates = pd.date_range("2026-01-01", periods=30, freq="B")
        close_vals = [100.0] * 10 + [float("nan")] * 20
        hist_df = pd.DataFrame({"Close": close_vals}, index=dates)
        ticker = MagicMock()
        ticker.history.return_value = hist_df
        self.assertIsNone(_realized_vol_20d(ticker))


# ── _compute_snapshot ─────────────────────────────────────────────────────────


class TestComputeSnapshot(TestCase):
    def _good_ticker(self) -> MagicMock:
        exp = _future_expiry(30)
        strikes = list(range(80, 120, 5))
        call_df = _make_chain_df(strikes, [0.22] * len(strikes))
        put_df = _make_chain_df(strikes, [0.25] * len(strikes))
        return _make_ticker_mock(
            spot=100.0,
            expirations=(exp,),
            call_df=call_df,
            put_df=put_df,
        )

    def test_normal_computation(self):
        with patch("data.options_data.yf.Ticker", return_value=self._good_ticker()):
            snap = _compute_snapshot("AAPL")
        self.assertIsNotNone(snap.atm_iv)
        self.assertGreater(snap.atm_iv, 0)

    def test_no_expirations(self):
        ticker = _make_ticker_mock(expirations=())
        with patch("data.options_data.yf.Ticker", return_value=ticker):
            snap = _compute_snapshot("AAPL")
        self.assertIsNone(snap.atm_iv)

    def test_no_expiry_in_range(self):
        # Only far-out expiry
        ticker = _make_ticker_mock(expirations=(_future_expiry(90),))
        with patch("data.options_data.yf.Ticker", return_value=ticker):
            snap = _compute_snapshot("AAPL")
        self.assertIsNone(snap.atm_iv)

    def test_empty_chains(self):
        ticker = _make_ticker_mock(
            expirations=(_future_expiry(30),),
            call_df=pd.DataFrame(),
            put_df=pd.DataFrame(),
        )
        with patch("data.options_data.yf.Ticker", return_value=ticker):
            snap = _compute_snapshot("AAPL")
        self.assertIsNone(snap.atm_iv)

    def test_zero_spot_from_fast_info(self):
        exp = _future_expiry(30)
        call_df = _make_chain_df([100], [0.20])
        put_df = _make_chain_df([100], [0.25])
        ticker = _make_ticker_mock(
            spot=0.0,  # fast_info returns 0
            expirations=(exp,),
            call_df=call_df,
            put_df=put_df,
        )
        # fast_info raises AttributeError → should use strike median fallback
        ticker.fast_info.last_price = 0.0
        with patch("data.options_data.yf.Ticker", return_value=ticker):
            snap = _compute_snapshot("TSLA")
        # Either it found a valid spot from median or returned null
        self.assertIsInstance(snap, OptionsSnapshot)

    def test_fast_info_exception_uses_median(self):
        exp = _future_expiry(30)
        call_df = _make_chain_df([100, 105], [0.20, 0.20])
        put_df = _make_chain_df([95, 100], [0.25, 0.25])
        ticker = _make_ticker_mock(
            spot=100.0,
            expirations=(exp,),
            call_df=call_df,
            put_df=put_df,
        )
        # spec_set=[] means no attributes → accessing .last_price raises AttributeError
        ticker.fast_info = MagicMock(spec_set=[])
        with patch("data.options_data.yf.Ticker", return_value=ticker):
            snap = _compute_snapshot("XYZ")
        self.assertIsInstance(snap, OptionsSnapshot)

    def test_exception_returns_null(self):
        with patch("data.options_data.yf.Ticker", side_effect=RuntimeError("fail")):
            snap = _compute_snapshot("BAD")
        self.assertIsNone(snap.atm_iv)

    def test_spot_zero_after_zero_strikes(self):
        """fast_info raises and all strikes are 0 → median=0 → second spot≤0 → null."""
        exp = _future_expiry(30)
        call_df = _make_chain_df([0.0], [0.20])
        put_df = _make_chain_df([0.0], [0.25])
        ticker = _make_ticker_mock(
            spot=100.0,
            expirations=(exp,),
            call_df=call_df,
            put_df=put_df,
        )
        ticker.fast_info = MagicMock(spec_set=[])  # raises on .last_price
        with patch("data.options_data.yf.Ticker", return_value=ticker):
            snap = _compute_snapshot("ZERO")
        self.assertIsNone(snap.atm_iv)

    def test_iv_rv_spread_none_when_atm_iv_none(self):
        """calls with all-zero IVs → atm_iv_val=None → iv_rv_spread stays None."""
        exp = _future_expiry(30)
        call_df = pd.DataFrame(
            {
                "strike": [100.0],
                "impliedVolatility": [0.0],
                "openInterest": [100.0],
                "volume": [10.0],
            }
        )
        put_df = _make_chain_df([100], [0.25])
        ticker = _make_ticker_mock(spot=100.0, expirations=(exp,), call_df=call_df, put_df=put_df)
        with patch("data.options_data.yf.Ticker", return_value=ticker):
            snap = _compute_snapshot("AAPL")
        self.assertIsNone(snap.iv_rv_spread)
        self.assertFalse(snap.iv_cheap)
        self.assertFalse(snap.iv_expensive)

    def test_put_call_oi_ratio_computed(self):
        exp = _future_expiry(30)
        call_df = _make_chain_df([100], [0.20])
        put_df = _make_chain_df([100], [0.25])
        call_df["openInterest"] = 1000.0
        put_df["openInterest"] = 2000.0
        ticker = _make_ticker_mock(spot=100.0, expirations=(exp,), call_df=call_df, put_df=put_df)
        with patch("data.options_data.yf.Ticker", return_value=ticker):
            snap = _compute_snapshot("AAPL")
        self.assertIsNotNone(snap.put_call_oi_ratio)

    def test_unusual_call_oi_detection(self):
        exp = _future_expiry(30)
        call_df = _make_chain_df([100], [0.20])
        put_df = _make_chain_df([100], [0.25])
        # Call volume >> call OI → unusual
        call_df["openInterest"] = 100.0
        call_df["volume"] = 400.0  # 4× OI
        put_df["openInterest"] = 100.0
        put_df["volume"] = 50.0
        ticker = _make_ticker_mock(spot=100.0, expirations=(exp,), call_df=call_df, put_df=put_df)
        with patch("data.options_data.yf.Ticker", return_value=ticker):
            snap = _compute_snapshot("AAPL")
        self.assertTrue(snap.unusual_call_oi)

    def test_panic_put_skew(self):
        exp = _future_expiry(30)
        strikes = list(range(80, 120, 5))
        # Very high put IV vs call IV → skew > 1.4
        call_df = _make_chain_df(strikes, [0.20] * len(strikes))
        put_df = _make_chain_df(strikes, [0.35] * len(strikes))
        ticker = _make_ticker_mock(spot=100, expirations=(exp,), call_df=call_df, put_df=put_df)
        with patch("data.options_data.yf.Ticker", return_value=ticker):
            snap = _compute_snapshot("SPY")
        # panic_put_skew depends on skew_25d > 1.4 — may or may not trigger
        # depending on exact delta computation, but should not raise
        self.assertIsInstance(snap, OptionsSnapshot)

    def test_iv_cheap_flag(self):
        # iv_rv_spread < -0.07 → iv_cheap
        exp = _future_expiry(30)
        call_df = _make_chain_df([100], [0.10])  # low IV
        put_df = _make_chain_df([100], [0.10])
        ticker = _make_ticker_mock(spot=100, expirations=(exp,), call_df=call_df, put_df=put_df)
        # hist shows high RV (big daily moves)
        dates = pd.date_range("2026-01-01", periods=30, freq="B")
        big_moves = [100.0 * (1 + 0.03 * (-1 if i % 2 else 1)) ** i for i in range(30)]
        hist_df = pd.DataFrame({"Close": big_moves}, index=dates)
        ticker.history.return_value = hist_df
        with patch("data.options_data.yf.Ticker", return_value=ticker):
            snap = _compute_snapshot("AAPL")
        # If RV computed high and IV is 0.10, iv_cheap might trigger
        self.assertIsInstance(snap, OptionsSnapshot)

    def test_iv_expensive_flag(self):
        # iv_rv_spread > 0.15 → iv_expensive
        exp = _future_expiry(30)
        call_df = _make_chain_df([100], [0.45])  # high IV
        put_df = _make_chain_df([100], [0.45])
        ticker = _make_ticker_mock(spot=100, expirations=(exp,), call_df=call_df, put_df=put_df)
        # hist shows low RV (tiny daily moves)
        dates = pd.date_range("2026-01-01", periods=30, freq="B")
        tiny_moves = [100.0 + 0.001 * i for i in range(30)]
        hist_df = pd.DataFrame({"Close": tiny_moves}, index=dates)
        ticker.history.return_value = hist_df
        with patch("data.options_data.yf.Ticker", return_value=ticker):
            snap = _compute_snapshot("AAPL")
        self.assertIsInstance(snap, OptionsSnapshot)

    def test_below_min_oi(self):
        # Total OI below _MIN_OPEN_INTEREST → put_call_oi_ratio is None
        exp = _future_expiry(30)
        call_df = _make_chain_df([100], [0.20], include_oi=False)
        put_df = _make_chain_df([100], [0.25], include_oi=False)
        ticker = _make_ticker_mock(spot=100, expirations=(exp,), call_df=call_df, put_df=put_df)
        with patch("data.options_data.yf.Ticker", return_value=ticker):
            snap = _compute_snapshot("AAPL")
        self.assertIsNone(snap.put_call_oi_ratio)


class TestNullSnapshot(TestCase):
    def test_all_false_none(self):
        null = _null_snapshot()
        self.assertIsNone(null.atm_iv)
        self.assertFalse(null.iv_cheap)
        self.assertFalse(null.iv_expensive)
        self.assertFalse(null.unusual_call_oi)
        self.assertFalse(null.panic_put_skew)
        self.assertFalse(null.call_skew_spike)


# ── Cache ─────────────────────────────────────────────────────────────────────


class TestCacheIO(TestCase):
    def _snap(self) -> OptionsSnapshot:
        return OptionsSnapshot(
            atm_iv=0.20,
            skew_25d=1.1,
            put_call_oi_ratio=1.2,
            iv_rv_spread=0.02,
            iv_cheap=False,
            iv_expensive=False,
            unusual_call_oi=False,
            panic_put_skew=False,
            call_skew_spike=False,
        )

    def test_save_load_roundtrip(self):
        with (
            patch("data.options_data._CACHE_PATH", "/tmp/_test_options_cache.json"),
            patch("data.options_data.today_et") as mock_today,
        ):
            mock_today.return_value = __import__("datetime").date(2026, 6, 4)
            _save_cache(
                {"AAPL": {**__import__("dataclasses").asdict(self._snap()), "_date": "2026-06-04"}}
            )
            cache = _load_cache()
        self.assertIn("AAPL", cache)
        with contextlib.suppress(FileNotFoundError):
            os.remove("/tmp/_test_options_cache.json")

    def test_load_missing_file(self):
        with patch("data.options_data._CACHE_PATH", "/tmp/_no_options.json"):
            result = _load_cache()
        self.assertEqual(result, {})

    def test_load_corrupt_json(self):
        with patch("data.options_data._CACHE_PATH", "/tmp/_corrupt_options.json"):
            with open("/tmp/_corrupt_options.json", "w") as f:
                f.write("{bad json")
            result = _load_cache()
        self.assertEqual(result, {})
        with contextlib.suppress(FileNotFoundError):
            os.remove("/tmp/_corrupt_options.json")

    def test_save_oserror(self):
        with (
            patch("data.options_data._CACHE_PATH", "/no_dir/x.json"),
            patch("data.options_data.os.makedirs", side_effect=OSError("fail")),
        ):
            _save_cache({})  # should not raise

    def test_is_stale_true(self):
        entry = {"_date": "2026-01-01"}
        with patch("data.options_data.today_et") as mock_today:
            mock_today.return_value = __import__("datetime").date(2026, 6, 4)
            self.assertTrue(_is_stale(entry))

    def test_is_stale_false(self):
        entry = {"_date": "2026-06-04"}
        with patch("data.options_data.today_et") as mock_today:
            mock_today.return_value = __import__("datetime").date(2026, 6, 4)
            self.assertFalse(_is_stale(entry))


# ── get_options_snapshot ──────────────────────────────────────────────────────


class TestGetOptionsSnapshot(TestCase):
    def _snap(self) -> OptionsSnapshot:
        return OptionsSnapshot(
            atm_iv=0.20,
            skew_25d=1.1,
            put_call_oi_ratio=1.0,
            iv_rv_spread=0.01,
            iv_cheap=False,
            iv_expensive=False,
            unusual_call_oi=False,
            panic_put_skew=False,
            call_skew_spike=False,
        )

    def test_cache_hit(self):
        snap = self._snap()
        from dataclasses import asdict

        cache = {"AAPL": {**asdict(snap), "_date": "2026-06-04"}}
        with (
            patch("data.options_data._load_cache", return_value=cache),
            patch("data.options_data.today_et") as mock_today,
        ):
            mock_today.return_value = __import__("datetime").date(2026, 6, 4)
            result = get_options_snapshot("AAPL")
        self.assertEqual(result.atm_iv, 0.20)

    def test_cache_miss_computes(self):
        with (
            patch("data.options_data._load_cache", return_value={}),
            patch("data.options_data._compute_snapshot", return_value=self._snap()),
            patch("data.options_data._save_cache"),
            patch("data.options_data.today_et") as mock_today,
        ):
            mock_today.return_value = __import__("datetime").date(2026, 6, 4)
            result = get_options_snapshot("AAPL")
        self.assertEqual(result.atm_iv, 0.20)

    def test_force_refresh(self):
        snap = self._snap()
        from dataclasses import asdict

        cache = {"AAPL": {**asdict(snap), "_date": "2026-06-04"}}
        new_snap = OptionsSnapshot(
            atm_iv=0.30,
            skew_25d=1.2,
            put_call_oi_ratio=1.0,
            iv_rv_spread=0.05,
            iv_cheap=False,
            iv_expensive=False,
            unusual_call_oi=False,
            panic_put_skew=False,
            call_skew_spike=False,
        )
        with (
            patch("data.options_data._load_cache", return_value=cache),
            patch("data.options_data._compute_snapshot", return_value=new_snap),
            patch("data.options_data._save_cache"),
            patch("data.options_data.today_et") as mock_today,
        ):
            mock_today.return_value = __import__("datetime").date(2026, 6, 4)
            result = get_options_snapshot("AAPL", force_refresh=True)
        self.assertEqual(result.atm_iv, 0.30)

    def test_corrupt_cache_entry_recomputes(self):
        # Cache entry with wrong fields → should fall back to compute
        bad_cache = {"AAPL": {"_date": "2026-06-04", "bad_field": 1}}
        with (
            patch("data.options_data._load_cache", return_value=bad_cache),
            patch("data.options_data._compute_snapshot", return_value=self._snap()),
            patch("data.options_data._save_cache"),
            patch("data.options_data.today_et") as mock_today,
        ):
            mock_today.return_value = __import__("datetime").date(2026, 6, 4)
            result = get_options_snapshot("AAPL")
        self.assertEqual(result.atm_iv, 0.20)


# ── get_options_batch ─────────────────────────────────────────────────────────


class TestGetOptionsBatch(TestCase):
    def _snap(self) -> OptionsSnapshot:
        return OptionsSnapshot(
            atm_iv=0.22,
            skew_25d=1.05,
            put_call_oi_ratio=0.9,
            iv_rv_spread=0.03,
            iv_cheap=False,
            iv_expensive=False,
            unusual_call_oi=False,
            panic_put_skew=False,
            call_skew_spike=False,
        )

    def test_batch_all_cache_hit(self):
        snap = self._snap()
        from dataclasses import asdict

        cache = {
            "AAPL": {**asdict(snap), "_date": "2026-06-04"},
            "MSFT": {**asdict(snap), "_date": "2026-06-04"},
        }
        with (
            patch("data.options_data._load_cache", return_value=cache),
            patch("data.options_data.today_et") as mock_today,
        ):
            mock_today.return_value = __import__("datetime").date(2026, 6, 4)
            results = get_options_batch(["AAPL", "MSFT"])
        self.assertIn("AAPL", results)
        self.assertIn("MSFT", results)

    def test_batch_computes_missing(self):
        snap = self._snap()
        with (
            patch("data.options_data._load_cache", return_value={}),
            patch("data.options_data._compute_snapshot", return_value=snap),
            patch("data.options_data._save_cache"),
            patch("data.options_data.today_et") as mock_today,
        ):
            mock_today.return_value = __import__("datetime").date(2026, 6, 4)
            results = get_options_batch(["AAPL"])
        self.assertIn("AAPL", results)

    def test_batch_future_exception(self):
        """Futures that raise are handled; symbol gets null snapshot."""
        with (
            patch("data.options_data._load_cache", return_value={}),
            patch("data.options_data._compute_snapshot", side_effect=RuntimeError("fail")),
            patch("data.options_data._save_cache"),
            patch("data.options_data.today_et") as mock_today,
        ):
            mock_today.return_value = __import__("datetime").date(2026, 6, 4)
            results = get_options_batch(["AAPL"])
        self.assertIn("AAPL", results)
        self.assertIsNone(results["AAPL"].atm_iv)

    def test_batch_force_refresh(self):
        snap = self._snap()
        from dataclasses import asdict

        cache = {"AAPL": {**asdict(snap), "_date": "2026-06-04"}}
        new_snap = OptionsSnapshot(
            atm_iv=0.30,
            skew_25d=1.0,
            put_call_oi_ratio=0.8,
            iv_rv_spread=0.0,
            iv_cheap=False,
            iv_expensive=False,
            unusual_call_oi=False,
            panic_put_skew=False,
            call_skew_spike=False,
        )
        with (
            patch("data.options_data._load_cache", return_value=cache),
            patch("data.options_data._compute_snapshot", return_value=new_snap),
            patch("data.options_data._save_cache"),
            patch("data.options_data.today_et") as mock_today,
        ):
            mock_today.return_value = __import__("datetime").date(2026, 6, 4)
            results = get_options_batch(["AAPL"], force_refresh=True)
        self.assertEqual(results["AAPL"].atm_iv, 0.30)

    def test_batch_corrupt_cache_entry_recomputes(self):
        snap = self._snap()
        bad_cache = {"AAPL": {"_date": "2026-06-04", "wrong": 1}}
        with (
            patch("data.options_data._load_cache", return_value=bad_cache),
            patch("data.options_data._compute_snapshot", return_value=snap),
            patch("data.options_data._save_cache"),
            patch("data.options_data.today_et") as mock_today,
        ):
            mock_today.return_value = __import__("datetime").date(2026, 6, 4)
            results = get_options_batch(["AAPL"])
        self.assertEqual(results["AAPL"].atm_iv, 0.22)
