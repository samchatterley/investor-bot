"""Tests for data/sentiment_client.py — 100% coverage."""

from __future__ import annotations

import contextlib
import os
from unittest import TestCase
from unittest.mock import MagicMock, patch

import pandas as pd

from data.sentiment_client import (
    AAIISentiment,
    FearGreedSnapshot,
    _build_aaii_snapshot,
    _download_fg_prices,
    _fg_hyg_ief_trend,
    _fg_spy_momentum,
    _fg_tlt_spy_spread,
    _fg_vix_vs_ma,
    _is_stale,
    _load_cache,
    _parse_aaii_df,
    _save_cache,
    compute_fear_greed,
    get_aaii_sentiment,
    get_fear_greed_composite,
    get_google_trends,
    get_sentiment_snapshot,
)


def _make_series(values: list[float]) -> pd.Series:
    return pd.Series(values, dtype=float)


# ── Cache helpers ─────────────────────────────────────────────────────────────


class TestCacheIO(TestCase):
    def test_save_load_roundtrip(self):
        data = {"aaii": {"_fetched_date": "2026-06-04", "records": []}}
        with patch("data.sentiment_client._CACHE_PATH", "/tmp/_sent_cache.json"):
            _save_cache(data)
            loaded = _load_cache()
        self.assertEqual(loaded["aaii"]["_fetched_date"], "2026-06-04")
        with contextlib.suppress(FileNotFoundError):
            os.remove("/tmp/_sent_cache.json")

    def test_load_missing(self):
        with patch("data.sentiment_client._CACHE_PATH", "/tmp/_no_sent.json"):
            result = _load_cache()
        self.assertEqual(result, {})

    def test_load_corrupt(self):
        with patch("data.sentiment_client._CACHE_PATH", "/tmp/_corrupt_sent.json"):
            with open("/tmp/_corrupt_sent.json", "w") as f:
                f.write("{bad")
            result = _load_cache()
        self.assertEqual(result, {})
        with contextlib.suppress(FileNotFoundError):
            os.remove("/tmp/_corrupt_sent.json")

    def test_save_oserror(self):
        with (
            patch("data.sentiment_client._CACHE_PATH", "/no_dir/x.json"),
            patch("data.sentiment_client.os.makedirs", side_effect=OSError),
        ):
            _save_cache({})  # must not raise

    def test_is_stale_true(self):
        entry = {"_fetched_date": "2026-01-01"}
        with patch("data.sentiment_client.today_et") as m:
            m.return_value = __import__("datetime").date(2026, 6, 4)
            self.assertTrue(_is_stale(entry, ttl_days=7))

    def test_is_stale_false_within_ttl(self):
        entry = {"_fetched_date": "2026-06-01"}
        with patch("data.sentiment_client.today_et") as m:
            m.return_value = __import__("datetime").date(2026, 6, 4)
            self.assertFalse(_is_stale(entry, ttl_days=7))

    def test_is_stale_missing_key(self):
        self.assertTrue(_is_stale({}, ttl_days=7))

    def test_is_stale_bad_format(self):
        self.assertTrue(_is_stale({"_fetched_date": "not-a-date"}, ttl_days=7))


# ── AAII parsing ──────────────────────────────────────────────────────────────


class TestParseAaiiDf(TestCase):
    def _make_df(self, rows: list[list]) -> pd.DataFrame:
        """Create a DataFrame that looks like the AAII XLS."""
        header = ["Date", "Bullish", "Neutral", "Bearish", "Total"]
        full = [header] + rows
        return pd.DataFrame(full)

    def test_parses_decimal_fractions(self):
        rows = [
            ["2026-05-01", 0.35, 0.30, 0.35, 1.0],
            ["2026-05-08", 0.40, 0.28, 0.32, 1.0],
        ]
        df = self._make_df(rows)
        result = _parse_aaii_df(df)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[-1]["bullish"], 0.40, places=3)

    def test_parses_percentage_values(self):
        # AV sometimes gives 35.0 instead of 0.35 — needs ≥2 rows to return non-None
        rows = [
            ["2026-04-24", 35.0, 30.0, 35.0, 100.0],
            ["2026-05-01", 40.0, 28.0, 32.0, 100.0],
        ]
        df = self._make_df(rows)
        result = _parse_aaii_df(df)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result[-1]["bullish"], 0.40, places=3)

    def test_returns_none_when_no_header(self):
        df = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
        result = _parse_aaii_df(df)
        self.assertIsNone(result)

    def test_returns_none_insufficient_rows(self):
        # Only 1 data row → still valid if sum matches
        rows = [["2026-05-01", 0.35, 0.30, 0.35, 1.0]]
        df = self._make_df(rows)
        # _parse_aaii_df returns records[-8:] if len >= 2 else None
        result = _parse_aaii_df(df)
        self.assertIsNone(result)

    def test_skips_bad_sum_rows(self):
        # Row where bull+neutral+bear != 1.0 within tolerance
        rows = [
            ["2026-05-01", 0.35, 0.30, 0.35, 1.0],
            ["2026-05-08", 0.99, 0.99, 0.99, 1.0],  # bad sum
            ["2026-05-15", 0.40, 0.28, 0.32, 1.0],
        ]
        df = self._make_df(rows)
        result = _parse_aaii_df(df)
        self.assertIsNotNone(result)
        # Only 2 valid rows
        self.assertEqual(len(result), 2)

    def test_exception_returns_none(self):
        # Pass something totally unexpected
        result = _parse_aaii_df(pd.DataFrame([[None, None, None]]))
        self.assertIsNone(result)

    def test_inner_except_on_invalid_value(self):
        """Row with non-numeric 'Bullish' triggers ValueError → inner except → skip."""
        rows = [
            ["2026-05-01", 0.35, 0.30, 0.35, 1.0],
            ["2026-05-08", "INVALID", 0.30, 0.35, 1.0],
            ["2026-05-15", 0.40, 0.28, 0.32, 1.0],
        ]
        df = self._make_df(rows)
        result = _parse_aaii_df(df)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)  # middle row skipped

    def test_outer_except_on_non_dataframe(self):
        """Passing None triggers AttributeError in outer try → outer except → None."""
        result = _parse_aaii_df(None)  # type: ignore[arg-type]
        self.assertIsNone(result)


class TestBuildAaiiSnapshot(TestCase):
    def _records(self, n: int, bull: float = 0.40, bear: float = 0.35) -> list[dict]:
        return [
            {
                "bullish": bull,
                "bearish": bear,
                "neutral": round(1 - bull - bear, 4),
                "date": f"2026-0{i + 1}-01",
            }
            for i in range(n)
        ]

    def test_basic_build(self):
        records = self._records(3)
        snap = _build_aaii_snapshot(records)
        self.assertAlmostEqual(snap.bullish_pct, 0.40, places=3)
        self.assertAlmostEqual(snap.bearish_pct, 0.35, places=3)
        self.assertAlmostEqual(snap.bull_bear_spread, 0.05, places=3)

    def test_extreme_bearish_two_weeks(self):
        records = self._records(4, bull=0.20, bear=0.52)
        snap = _build_aaii_snapshot(records)
        self.assertTrue(snap.extreme_bearish)

    def test_not_extreme_bearish_only_one_week(self):
        records = self._records(1, bull=0.20, bear=0.52)
        snap = _build_aaii_snapshot(records)
        self.assertFalse(snap.extreme_bearish)

    def test_extreme_bullish_three_weeks(self):
        records = self._records(4, bull=0.62, bear=0.15)
        snap = _build_aaii_snapshot(records)
        self.assertTrue(snap.extreme_bullish)

    def test_not_extreme_bullish_only_two_weeks(self):
        records = self._records(2, bull=0.62, bear=0.15)
        snap = _build_aaii_snapshot(records)
        self.assertFalse(snap.extreme_bullish)


class TestGetAaiiSentiment(TestCase):
    def _records(self) -> list[dict]:
        return [
            {"bullish": 0.40, "bearish": 0.35, "neutral": 0.25, "date": f"2026-0{i + 1}-01"}
            for i in range(3)
        ]

    def test_cache_hit(self):
        cache = {
            "aaii": {
                "_fetched_date": "2026-06-04",
                "records": self._records(),
            }
        }
        with (
            patch("data.sentiment_client._load_cache", return_value=cache),
            patch("data.sentiment_client.today_et") as m,
        ):
            m.return_value = __import__("datetime").date(2026, 6, 4)
            result = get_aaii_sentiment()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, AAIISentiment)

    def test_cache_miss_fetches(self):
        import pandas as pd

        rows = [
            [f"2026-0{i + 1}-01", 0.35 + i * 0.01, 0.30, 0.35 - i * 0.01, 1.0] for i in range(3)
        ]
        header = ["Date", "Bullish", "Neutral", "Bearish", "Total"]
        fake_df = pd.DataFrame([header] + rows)

        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.content = b"fake xls bytes"

        with (
            patch("data.sentiment_client._load_cache", return_value={}),
            patch("data.sentiment_client._save_cache"),
            patch("data.sentiment_client.requests.get", return_value=mock_resp),
            patch("data.sentiment_client.pd.read_excel", return_value=fake_df),
            patch("data.sentiment_client.today_et") as m,
        ):
            m.return_value = __import__("datetime").date(2026, 6, 4)
            result = get_aaii_sentiment()
        self.assertIsNotNone(result)

    def test_fetch_failure_returns_none(self):
        with (
            patch("data.sentiment_client._load_cache", return_value={}),
            patch("data.sentiment_client.requests.get", side_effect=RuntimeError("network")),
        ):
            result = get_aaii_sentiment()
        self.assertIsNone(result)

    def test_parse_failure_returns_none(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.content = b"xls"
        # pd.read_excel returns empty DataFrame → _parse_aaii_df returns None
        with (
            patch("data.sentiment_client._load_cache", return_value={}),
            patch("data.sentiment_client.requests.get", return_value=mock_resp),
            patch("data.sentiment_client.pd.read_excel", return_value=pd.DataFrame()),
        ):
            result = get_aaii_sentiment()
        self.assertIsNone(result)

    def test_corrupt_cache_entry_refetches(self):
        cache = {"aaii": {"_fetched_date": "2026-06-04", "records": "broken"}}
        with (
            patch("data.sentiment_client._load_cache", return_value=cache),
            patch("data.sentiment_client.today_et") as m,
            patch("data.sentiment_client.requests.get", side_effect=RuntimeError),
        ):
            m.return_value = __import__("datetime").date(2026, 6, 4)
            result = get_aaii_sentiment()
        self.assertIsNone(result)

    def test_force_refresh(self):
        cache = {"aaii": {"_fetched_date": "2026-06-04", "records": self._records()}}
        rows = [[f"2026-0{i + 1}-01", 0.35, 0.30, 0.35, 1.0] for i in range(3)]
        header = ["Date", "Bullish", "Neutral", "Bearish", "Total"]
        fake_df = pd.DataFrame([header] + rows)
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.content = b"xls"
        with (
            patch("data.sentiment_client._load_cache", return_value=cache),
            patch("data.sentiment_client._save_cache"),
            patch("data.sentiment_client.requests.get", return_value=mock_resp),
            patch("data.sentiment_client.pd.read_excel", return_value=fake_df),
            patch("data.sentiment_client.today_et") as m,
        ):
            m.return_value = __import__("datetime").date(2026, 6, 4)
            result = get_aaii_sentiment(force_refresh=True)
        # force_refresh bypasses cache even if not stale
        self.assertIsNotNone(result)


# ── Fear & Greed components ───────────────────────────────────────────────────


class TestFgComponents(TestCase):
    def test_spy_momentum_above_sma(self):
        # SPY above 125-day SMA → greed component > 50
        vals = [100.0] * 130
        vals[-1] = 115.0  # current price well above SMA (~100)
        s = _make_series(vals)
        result = _fg_spy_momentum(s)
        self.assertIsNotNone(result)
        self.assertGreater(result, 50.0)

    def test_spy_momentum_below_sma(self):
        vals = [100.0] * 130
        vals[-1] = 85.0  # below SMA
        s = _make_series(vals)
        result = _fg_spy_momentum(s)
        self.assertIsNotNone(result)
        self.assertLess(result, 50.0)

    def test_spy_momentum_insufficient_data(self):
        self.assertIsNone(_fg_spy_momentum(_make_series([100.0] * 100)))

    def test_spy_momentum_zero_sma(self):
        self.assertIsNone(_fg_spy_momentum(_make_series([0.0] * 130)))

    def test_vix_vs_ma_calm(self):
        # VIX below 50-day MA → greed > 50
        vals = [20.0] * 55
        vals[-1] = 15.0  # below MA
        s = _make_series(vals)
        result = _fg_vix_vs_ma(s)
        self.assertGreater(result, 50.0)

    def test_vix_vs_ma_elevated(self):
        vals = [15.0] * 55
        vals[-1] = 30.0  # elevated VIX
        s = _make_series(vals)
        result = _fg_vix_vs_ma(s)
        self.assertLess(result, 50.0)

    def test_vix_vs_ma_insufficient(self):
        self.assertIsNone(_fg_vix_vs_ma(_make_series([20.0] * 40)))

    def test_vix_vs_ma_zero_ma(self):
        self.assertIsNone(_fg_vix_vs_ma(_make_series([0.0] * 55)))

    def test_tlt_spy_spread_safety_flight(self):
        # TLT up a lot, SPY flat → score < 50 (fear)
        tlt = _make_series([100.0] * 5 + [120.0] * 6)
        spy = _make_series([100.0] * 11)
        result = _fg_tlt_spy_spread(tlt, spy)
        self.assertLess(result, 50.0)

    def test_tlt_spy_spread_risk_on(self):
        # SPY up a lot, TLT flat → score > 50 (greed)
        spy = _make_series([100.0] * 5 + [120.0] * 6)
        tlt = _make_series([100.0] * 11)
        result = _fg_tlt_spy_spread(tlt, spy)
        self.assertGreater(result, 50.0)

    def test_tlt_spy_spread_insufficient(self):
        self.assertIsNone(_fg_tlt_spy_spread(_make_series([100.0] * 5), _make_series([100.0] * 11)))

    def test_hyg_ief_rising(self):
        # HYG/IEF rising → risk appetite → greed > 50
        hyg = _make_series([100.0] * 5 + [110.0] * 6)
        ief = _make_series([100.0] * 11)
        result = _fg_hyg_ief_trend(hyg, ief)
        self.assertGreater(result, 50.0)

    def test_hyg_ief_falling(self):
        hyg = _make_series([110.0] * 5 + [100.0] * 6)
        ief = _make_series([100.0] * 11)
        result = _fg_hyg_ief_trend(hyg, ief)
        self.assertLess(result, 50.0)

    def test_hyg_ief_insufficient(self):
        self.assertIsNone(_fg_hyg_ief_trend(_make_series([100.0] * 5), _make_series([100.0] * 11)))

    def test_hyg_ief_exception(self):
        # Pass non-Series to trigger exception path
        result = _fg_hyg_ief_trend(None, _make_series([100.0] * 15))  # type: ignore[arg-type]
        self.assertIsNone(result)

    def test_hyg_ief_nonoverlapping_combined_short(self):
        """After concat+dropna, combined has < 11 rows → return None."""
        hyg = pd.Series([100.0] * 15, index=range(0, 15))
        ief = pd.Series([100.0] * 15, index=range(10, 25))
        # Overlapping indices: 10–14 only → 5 rows after dropna, < 11
        result = _fg_hyg_ief_trend(hyg, ief)
        self.assertIsNone(result)

    def test_hyg_ief_string_series_exception(self):
        """String-valued series raises TypeError inside try → except → None."""
        hyg = pd.Series(["a"] * 15)
        ief = pd.Series(["b"] * 15)
        result = _fg_hyg_ief_trend(hyg, ief)
        self.assertIsNone(result)


class TestComputeFearGreed(TestCase):
    def _good_prices(self) -> dict:
        spy = _make_series([100.0] * 130)
        vix = _make_series([15.0] * 55)
        tlt = _make_series([100.0] * 15)
        hyg = _make_series([100.0] * 15)
        ief = _make_series([100.0] * 15)
        return {"SPY": spy, "^VIX": vix, "TLT": tlt, "HYG": hyg, "IEF": ief}

    def test_all_components_computed(self):
        snap = compute_fear_greed(self._good_prices(), nh_nl_ratio=2.0, pct_above_sma50=0.60)
        self.assertIsNotNone(snap)
        self.assertIn("spy_momentum", snap.components)
        self.assertIn("breadth", snap.components)
        self.assertIn("nh_nl", snap.components)
        self.assertIn("vix_vs_ma", snap.components)

    def test_empty_prices_returns_neutral(self):
        snap = compute_fear_greed({})
        self.assertEqual(snap.label, "Neutral")
        self.assertEqual(snap.score, 50.0)

    def test_missing_optional_breadth_inputs(self):
        snap = compute_fear_greed(self._good_prices())
        self.assertIsNone(snap.components.get("breadth"))
        self.assertIsNone(snap.components.get("nh_nl"))

    def test_extreme_fear_label(self):
        # All components should give 0 → extreme fear
        spy = _make_series([100.0] * 130 + [50.0])  # huge crash
        vix = _make_series([15.0] * 50 + [50.0] * 5)  # VIX spiked
        tlt = _make_series([100.0] * 5 + [130.0] * 6)  # TLT outperforming → fear
        hyg = _make_series([100.0] * 5 + [80.0] * 6)  # HYG falling
        ief = _make_series([100.0] * 11)
        prices = {"SPY": spy, "^VIX": vix, "TLT": tlt, "HYG": hyg, "IEF": ief}
        snap = compute_fear_greed(prices, nh_nl_ratio=0.1, pct_above_sma50=0.05)
        # Expect fear or extreme fear (score < 50)
        self.assertLessEqual(snap.score, 50.0)

    def test_extreme_greed_flag(self):
        snap = FearGreedSnapshot(
            score=85.0, label="Extreme Greed", extreme_fear=False, extreme_greed=True, components={}
        )
        self.assertTrue(snap.extreme_greed)
        self.assertFalse(snap.extreme_fear)

    def test_clamps_at_100(self):
        spy = _make_series([100.0] * 130 + [300.0])  # insane rally
        snap = compute_fear_greed({"SPY": spy})
        self.assertLessEqual(snap.score, 100.0)

    def test_clamps_at_zero(self):
        spy = _make_series([300.0] * 130 + [10.0])  # crash
        snap = compute_fear_greed({"SPY": spy})
        self.assertGreaterEqual(snap.score, 0.0)

    def test_fear_label(self):
        """Score ~33 (SPY mildly below SMA) → 'Fear' label."""
        spy = _make_series([100.0] * 129 + [95.0])
        snap = compute_fear_greed({"SPY": spy})
        self.assertEqual(snap.label, "Fear")
        self.assertFalse(snap.extreme_fear)

    def test_greed_label(self):
        """Score ~70 (SPY mildly above SMA) → 'Greed' label."""
        spy = _make_series([100.0] * 129 + [105.0])
        snap = compute_fear_greed({"SPY": spy})
        self.assertEqual(snap.label, "Greed")
        self.assertFalse(snap.extreme_greed)


class TestGetFearGreedComposite(TestCase):
    def _fake_prices(self) -> dict:
        return {"SPY": _make_series([100.0] * 130)}

    def test_cache_hit(self):
        snap_data = {
            "score": 55.0,
            "label": "Neutral",
            "extreme_fear": False,
            "extreme_greed": False,
            "components": {},
            "_fetched_date": "2026-06-04",
        }
        cache = {"fear_greed": snap_data}
        with (
            patch("data.sentiment_client._load_cache", return_value=cache),
            patch("data.sentiment_client.today_et") as m,
        ):
            m.return_value = __import__("datetime").date(2026, 6, 4)
            result = get_fear_greed_composite()
        self.assertIsNotNone(result)
        self.assertEqual(result.score, 55.0)

    def test_cache_miss_downloads(self):
        with (
            patch("data.sentiment_client._load_cache", return_value={}),
            patch("data.sentiment_client._download_fg_prices", return_value=self._fake_prices()),
            patch("data.sentiment_client._save_cache"),
            patch("data.sentiment_client.today_et") as m,
        ):
            m.return_value = __import__("datetime").date(2026, 6, 4)
            result = get_fear_greed_composite()
        self.assertIsNotNone(result)

    def test_force_refresh(self):
        snap_data = {
            "score": 99.0,
            "label": "Extreme Greed",
            "extreme_fear": False,
            "extreme_greed": True,
            "components": {},
            "_fetched_date": "2026-06-04",
        }
        cache = {"fear_greed": snap_data}
        with (
            patch("data.sentiment_client._load_cache", return_value=cache),
            patch("data.sentiment_client._download_fg_prices", return_value=self._fake_prices()),
            patch("data.sentiment_client._save_cache"),
            patch("data.sentiment_client.today_et") as m,
        ):
            m.return_value = __import__("datetime").date(2026, 6, 4)
            result = get_fear_greed_composite(force_refresh=True)
        # Should recompute — score should no longer be 99.0
        self.assertIsNotNone(result)
        self.assertNotEqual(result.score, 99.0)

    def test_corrupt_cache_entry_refetches(self):
        cache = {"fear_greed": {"_fetched_date": "2026-06-04", "score": "bad"}}
        with (
            patch("data.sentiment_client._load_cache", return_value=cache),
            patch("data.sentiment_client._download_fg_prices", return_value=self._fake_prices()),
            patch("data.sentiment_client._save_cache"),
            patch("data.sentiment_client.today_et") as m,
        ):
            m.return_value = __import__("datetime").date(2026, 6, 4)
            result = get_fear_greed_composite()
        self.assertIsNotNone(result)

    def test_returns_none_when_no_prices(self):
        with (
            patch("data.sentiment_client._load_cache", return_value={}),
            patch("data.sentiment_client._download_fg_prices", return_value={}),
            patch("data.sentiment_client._save_cache"),
            patch("data.sentiment_client.today_et") as m,
        ):
            m.return_value = __import__("datetime").date(2026, 6, 4)
            result = get_fear_greed_composite()
        # compute_fear_greed with {} prices → returns neutral, not None
        self.assertIsNotNone(result)
        self.assertEqual(result.label, "Neutral")


class TestDownloadFgPrices(TestCase):
    def _make_multiindex_df(self, n=50):
        tickers = ["SPY", "^VIX", "TLT", "HYG", "IEF"]
        dates = pd.date_range("2026-01-01", periods=n, freq="B")
        df = pd.DataFrame({("Close", t): [100.0] * n for t in tickers}, index=dates)
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        return df

    def test_download_success(self):
        fake_df = self._make_multiindex_df()
        with patch("data.sentiment_client.yf.download", return_value=fake_df):
            result = _download_fg_prices()
        self.assertIn("SPY", result)
        self.assertIn("^VIX", result)

    def test_download_empty(self):
        with patch("data.sentiment_client.yf.download", return_value=pd.DataFrame()):
            result = _download_fg_prices()
        self.assertEqual(result, {})

    def test_download_exception(self):
        with patch("data.sentiment_client.yf.download", side_effect=RuntimeError):
            result = _download_fg_prices()
        self.assertEqual(result, {})

    def test_download_none(self):
        with patch("data.sentiment_client.yf.download", return_value=None):
            result = _download_fg_prices()
        self.assertEqual(result, {})

    def test_download_non_multiindex_returns_empty(self):
        """When download returns a flat (non-MultiIndex) df, result is empty."""
        dates = pd.date_range("2026-01-01", periods=50, freq="B")
        flat_df = pd.DataFrame({"Close": [100.0] * 50}, index=dates)
        with patch("data.sentiment_client.yf.download", return_value=flat_df):
            result = _download_fg_prices()
        self.assertEqual(result, {})

    def test_download_missing_ticker_skipped(self):
        """Tickers absent from close.columns are silently skipped."""
        tickers_present = ["SPY", "^VIX", "TLT"]
        dates = pd.date_range("2026-01-01", periods=50, freq="B")
        df = pd.DataFrame({("Close", t): [100.0] * 50 for t in tickers_present}, index=dates)
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        with patch("data.sentiment_client.yf.download", return_value=df):
            result = _download_fg_prices()
        self.assertIn("SPY", result)
        self.assertNotIn("HYG", result)

    def test_download_short_series_excluded(self):
        """Tickers with < 11 bars are excluded (len(s) >= 11 branch not taken)."""
        all_tickers = ["SPY", "^VIX", "TLT", "HYG", "IEF"]
        dates = pd.date_range("2026-01-01", periods=8, freq="B")
        df = pd.DataFrame({("Close", t): [100.0] * 8 for t in all_tickers}, index=dates)
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        with patch("data.sentiment_client.yf.download", return_value=df):
            result = _download_fg_prices()
        self.assertEqual(result, {})


# ── Google Trends ─────────────────────────────────────────────────────────────


class TestGetGoogleTrends(TestCase):
    def _make_pytrends_mock(self, current: int = 80, history: list[int] | None = None) -> MagicMock:
        if history is None:
            history = [30] * 12 + [current]
        mock = MagicMock()
        dates = pd.date_range("2026-01-01", periods=len(history), freq="W")
        df = pd.DataFrame({"AAPL": history, "isPartial": [False] * len(history)}, index=dates)
        mock.interest_over_time.return_value = df
        return mock

    def test_cache_hit(self):
        cache = {
            "trends_AAPL": {
                "_fetched_date": "2026-06-04",
                "current_interest": 70,
                "avg_interest_12w": 30.0,
                "spike": True,
                "declining": False,
            }
        }
        with (
            patch("data.sentiment_client._load_cache", return_value=cache),
            patch("data.sentiment_client.today_et") as m,
        ):
            m.return_value = __import__("datetime").date(2026, 6, 4)
            result = get_google_trends("AAPL")
        self.assertIsNotNone(result)
        self.assertTrue(result["spike"])

    def test_spike_detection(self):
        fake_pt = self._make_pytrends_mock(current=100, history=[20] * 12 + [100])
        with (
            patch("data.sentiment_client._load_cache", return_value={}),
            patch("data.sentiment_client._save_cache"),
            patch("data.sentiment_client.today_et") as m,
            patch("data.sentiment_client._TrendReq", MagicMock(return_value=fake_pt)),
        ):
            m.return_value = __import__("datetime").date(2026, 6, 4)
            result = get_google_trends("AAPL")
        self.assertIsNotNone(result)
        self.assertTrue(result["spike"])

    def test_declining_detection(self):
        fake_pt = self._make_pytrends_mock(current=5, history=[50] * 12 + [5])
        with (
            patch("data.sentiment_client._load_cache", return_value={}),
            patch("data.sentiment_client._save_cache"),
            patch("data.sentiment_client.today_et") as m,
            patch("data.sentiment_client._TrendReq", MagicMock(return_value=fake_pt)),
        ):
            m.return_value = __import__("datetime").date(2026, 6, 4)
            result = get_google_trends("AAPL")
        self.assertIsNotNone(result)
        self.assertTrue(result["declining"])

    def test_pytrends_not_available(self):
        with (
            patch("data.sentiment_client._load_cache", return_value={}),
            patch("data.sentiment_client._TrendReq", None),
        ):
            result = get_google_trends("AAPL")
        self.assertIsNone(result)

    def test_pytrends_exception(self):
        with (
            patch("data.sentiment_client._load_cache", return_value={}),
            patch(
                "data.sentiment_client._TrendReq",
                MagicMock(side_effect=RuntimeError("rate limited")),
            ),
        ):
            result = get_google_trends("AAPL")
        self.assertIsNone(result)

    def test_empty_df_returns_none(self):
        mock_pt = MagicMock()
        mock_pt.interest_over_time.return_value = pd.DataFrame()
        with (
            patch("data.sentiment_client._load_cache", return_value={}),
            patch("data.sentiment_client._TrendReq", MagicMock(return_value=mock_pt)),
        ):
            result = get_google_trends("AAPL")
        self.assertIsNone(result)

    def test_symbol_not_in_df_returns_none(self):
        mock_pt = MagicMock()
        dates = pd.date_range("2026-01-01", periods=5, freq="W")
        df = pd.DataFrame({"OTHER": [10, 20, 30, 40, 50]}, index=dates)
        mock_pt.interest_over_time.return_value = df
        with (
            patch("data.sentiment_client._load_cache", return_value={}),
            patch("data.sentiment_client._TrendReq", MagicMock(return_value=mock_pt)),
        ):
            result = get_google_trends("AAPL")
        self.assertIsNone(result)

    def test_insufficient_series_returns_none(self):
        mock_pt = MagicMock()
        dates = pd.date_range("2026-01-01", periods=1, freq="W")
        df = pd.DataFrame({"AAPL": [50]}, index=dates)
        mock_pt.interest_over_time.return_value = df
        with (
            patch("data.sentiment_client._load_cache", return_value={}),
            patch("data.sentiment_client._TrendReq", MagicMock(return_value=mock_pt)),
        ):
            result = get_google_trends("AAPL")
        self.assertIsNone(result)

    def test_force_refresh(self):
        cache = {
            "trends_AAPL": {
                "_fetched_date": "2026-06-04",
                "current_interest": 10,
                "avg_interest_12w": 10.0,
                "spike": False,
                "declining": False,
            }
        }
        fake_pt = self._make_pytrends_mock(current=80, history=[20] * 12 + [80])
        with (
            patch("data.sentiment_client._load_cache", return_value=cache),
            patch("data.sentiment_client._save_cache"),
            patch("data.sentiment_client.today_et") as m,
            patch("data.sentiment_client._TrendReq", MagicMock(return_value=fake_pt)),
        ):
            m.return_value = __import__("datetime").date(2026, 6, 4)
            result = get_google_trends("AAPL", force_refresh=True)
        self.assertIsNotNone(result)
        self.assertEqual(result["current_interest"], 80)

    def test_cache_hit_exception_falls_through(self):
        """Cache entry whose .items() raises → except pass → fetches live (falls through)."""

        class BadEntry(dict):
            def items(self):
                raise RuntimeError("corrupt items")

        bad_entry = BadEntry({"_fetched_date": "2026-06-04"})
        cache = {"trends_AAPL": bad_entry}
        with (
            patch("data.sentiment_client._load_cache", return_value=cache),
            patch("data.sentiment_client._is_stale", return_value=False),
            patch("data.sentiment_client._TrendReq", side_effect=RuntimeError("rate limited")),
        ):
            result = get_google_trends("AAPL")
        self.assertIsNone(result)


# ── SentimentSnapshot ─────────────────────────────────────────────────────────


class TestGetSentimentSnapshot(TestCase):
    def test_contrarian_long_from_extreme_fear(self):
        fg = FearGreedSnapshot(
            score=15.0, label="Extreme Fear", extreme_fear=True, extreme_greed=False, components={}
        )
        with (
            patch("data.sentiment_client.get_aaii_sentiment", return_value=None),
            patch("data.sentiment_client.get_fear_greed_composite", return_value=fg),
        ):
            snap = get_sentiment_snapshot()
        self.assertTrue(snap.contrarian_long_signal)
        self.assertFalse(snap.contrarian_short_signal)

    def test_contrarian_long_from_aaii_bearish(self):
        aaii = AAIISentiment(
            bullish_pct=0.20,
            bearish_pct=0.55,
            neutral_pct=0.25,
            bull_bear_spread=-0.35,
            extreme_bearish=True,
            extreme_bullish=False,
        )
        fg = FearGreedSnapshot(
            score=50.0, label="Neutral", extreme_fear=False, extreme_greed=False, components={}
        )
        with (
            patch("data.sentiment_client.get_aaii_sentiment", return_value=aaii),
            patch("data.sentiment_client.get_fear_greed_composite", return_value=fg),
        ):
            snap = get_sentiment_snapshot()
        self.assertTrue(snap.contrarian_long_signal)

    def test_contrarian_short_from_extreme_greed(self):
        fg = FearGreedSnapshot(
            score=85.0, label="Extreme Greed", extreme_fear=False, extreme_greed=True, components={}
        )
        with (
            patch("data.sentiment_client.get_aaii_sentiment", return_value=None),
            patch("data.sentiment_client.get_fear_greed_composite", return_value=fg),
        ):
            snap = get_sentiment_snapshot()
        self.assertTrue(snap.contrarian_short_signal)

    def test_no_signals_when_neutral(self):
        fg = FearGreedSnapshot(
            score=50.0, label="Neutral", extreme_fear=False, extreme_greed=False, components={}
        )
        with (
            patch("data.sentiment_client.get_aaii_sentiment", return_value=None),
            patch("data.sentiment_client.get_fear_greed_composite", return_value=fg),
        ):
            snap = get_sentiment_snapshot()
        self.assertFalse(snap.contrarian_long_signal)
        self.assertFalse(snap.contrarian_short_signal)

    def test_both_none(self):
        with (
            patch("data.sentiment_client.get_aaii_sentiment", return_value=None),
            patch("data.sentiment_client.get_fear_greed_composite", return_value=None),
        ):
            snap = get_sentiment_snapshot()
        self.assertFalse(snap.contrarian_long_signal)
        self.assertFalse(snap.contrarian_short_signal)
        self.assertIsNone(snap.aaii)
        self.assertIsNone(snap.fear_greed)


# ── Module-level ImportError fallback ─────────────────────────────────────────


class TestTrendReqImportFallback(TestCase):
    def test_import_error_sets_trend_req_none(self):
        """Covers except ImportError: _TrendReq = None at module level (lines 44-45)."""
        import sys

        import data.sentiment_client as original_sc

        orig_pytrends_request = sys.modules.get("pytrends.request")

        try:
            sys.modules["pytrends.request"] = None  # causes ImportError on next import
            sys.modules.pop("data.sentiment_client", None)
            import data.sentiment_client as fresh_sc  # noqa: F401

            self.assertIsNone(fresh_sc._TrendReq)
        finally:
            sys.modules.pop("data.sentiment_client", None)
            sys.modules["data.sentiment_client"] = original_sc
            if orig_pytrends_request is not None:
                sys.modules["pytrends.request"] = orig_pytrends_request
            else:
                sys.modules.pop("pytrends.request", None)
