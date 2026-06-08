"""Tests for data/pairs.py.

All yfinance network calls and sector lookups are mocked.
100% line coverage required.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_close_df(symbols: list[str], n: int = 60) -> pd.DataFrame:
    """Return a fake multi-column Close DataFrame for the given symbols."""
    rng = np.random.default_rng(42)
    data = {sym: 100.0 + rng.standard_normal(n).cumsum() for sym in symbols}
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    df = pd.DataFrame(data, index=idx)
    return df


def _make_yf_return(symbols: list[str], n: int = 60) -> MagicMock:
    """Return a MagicMock that mimics the structure yf.download returns."""
    close_df = _make_close_df(symbols, n)
    mock_data = MagicMock()
    mock_data.empty = False
    mock_data.__len__ = lambda self: n  # noqa: ARG005
    mock_data.__getitem__ = lambda self, key: close_df if key == "Close" else MagicMock()
    return mock_data


def _make_yf_empty() -> MagicMock:
    mock_data = MagicMock()
    mock_data.empty = True
    return mock_data


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGetCointegratedPairsCacheHit(unittest.TestCase):
    """Cache is fresh — should return cached pairs without calling refresh_pairs."""

    def test_returns_cached_pairs_without_refresh(self) -> None:
        fresh_ts = datetime.now(UTC).isoformat()
        cache_content = {
            "generated_at": fresh_ts,
            "pairs": [
                {
                    "sym_a": "AAPL",
                    "sym_b": "MSFT",
                    "sector": "Technology",
                    "pvalue": 0.01,
                    "hedge_ratio": 1.2,
                    "spread_mean": 0.0,
                    "spread_std": 1.0,
                }
            ],
        }
        with (
            patch("data.pairs._load_cache", return_value=cache_content),
            patch("data.pairs.refresh_pairs") as mock_refresh,
        ):
            from data.pairs import get_cointegrated_pairs

            result = get_cointegrated_pairs(["AAPL", "MSFT"], max_age_days=7)

        mock_refresh.assert_not_called()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["sym_a"], "AAPL")


class TestGetCointegratedPairsCacheMiss(unittest.TestCase):
    """Cache is stale / missing — should call refresh_pairs."""

    def test_stale_cache_triggers_refresh(self) -> None:
        old_ts = (datetime.now(UTC) - timedelta(days=10)).isoformat()
        stale_cache = {"generated_at": old_ts, "pairs": []}
        fake_pairs = [
            {
                "sym_a": "JPM",
                "sym_b": "BAC",
                "sector": "Financials",
                "pvalue": 0.03,
                "hedge_ratio": 0.8,
                "spread_mean": 0.0,
                "spread_std": 2.0,
            }
        ]
        with (
            patch("data.pairs._load_cache", return_value=stale_cache),
            patch("data.pairs.refresh_pairs", return_value=fake_pairs) as mock_refresh,
        ):
            from data.pairs import get_cointegrated_pairs

            result = get_cointegrated_pairs(["JPM", "BAC"], max_age_days=7)

        mock_refresh.assert_called_once_with(["JPM", "BAC"])
        self.assertEqual(result, fake_pairs)

    def test_missing_cache_triggers_refresh(self) -> None:
        with (
            patch("data.pairs._load_cache", return_value={}),
            patch("data.pairs.refresh_pairs", return_value=[]) as mock_refresh,
        ):
            from data.pairs import get_cointegrated_pairs

            result = get_cointegrated_pairs(["AAPL"])

        mock_refresh.assert_called_once()
        self.assertEqual(result, [])


class TestGetCointegratedPairsGracefulDegradation(unittest.TestCase):
    def test_exception_returns_empty_list(self) -> None:
        with patch("data.pairs._load_cache", side_effect=RuntimeError("boom")):
            from data.pairs import get_cointegrated_pairs

            result = get_cointegrated_pairs(["AAPL"])

        self.assertEqual(result, [])


class TestRefreshPairsExcludesHighPvalue(unittest.TestCase):
    """Pairs with pvalue >= 0.05 must be excluded."""

    def test_high_pvalue_pair_excluded(self) -> None:
        symbols = ["AAPL", "MSFT"]
        mock_yf_data = _make_yf_return(symbols, n=60)

        with (
            patch("data.sector_data.get_sector", side_effect=lambda s: "Technology"),
            patch("yfinance.download", return_value=mock_yf_data),
            patch(
                "statsmodels.tsa.stattools.coint",
                return_value=(0.0, 0.99, None),
            ),
            patch("data.pairs._save_cache"),
        ):
            from data.pairs import refresh_pairs

            result = refresh_pairs(symbols, lookback_days=60)

        self.assertEqual(result, [])


class TestRefreshPairsIncludesLowPvalue(unittest.TestCase):
    """Pairs with pvalue < 0.05 must be included with correct keys."""

    def test_low_pvalue_pair_included(self) -> None:
        symbols = ["AAPL", "MSFT"]
        n = 60
        close_df = _make_close_df(symbols, n)

        mock_yf_data = MagicMock()
        mock_yf_data.empty = False
        mock_yf_data.__getitem__ = lambda self, key: close_df

        with (
            patch("data.sector_data.get_sector", side_effect=lambda s: "Technology"),
            patch("yfinance.download", return_value=mock_yf_data),
            patch(
                "statsmodels.tsa.stattools.coint",
                return_value=(0.0, 0.01, None),
            ),
            patch("data.pairs._save_cache"),
        ):
            from data.pairs import refresh_pairs

            result = refresh_pairs(symbols, lookback_days=60)

        self.assertEqual(len(result), 1)
        pair = result[0]
        for key in (
            "sym_a",
            "sym_b",
            "sector",
            "pvalue",
            "hedge_ratio",
            "spread_mean",
            "spread_std",
        ):
            self.assertIn(key, pair)
        self.assertEqual(pair["sector"], "Technology")
        self.assertLess(pair["pvalue"], 0.05)


class TestRefreshPairsYfinanceFailure(unittest.TestCase):
    """yfinance failure → graceful degradation → return []."""

    def test_yfinance_exception_returns_empty(self) -> None:
        with (
            patch("data.sector_data.get_sector", side_effect=lambda s: "Technology"),
            patch("yfinance.download", side_effect=RuntimeError("network error")),
        ):
            from data.pairs import refresh_pairs

            result = refresh_pairs(["AAPL", "MSFT"], lookback_days=60)

        self.assertEqual(result, [])

    def test_yfinance_empty_data_returns_empty(self) -> None:
        with (
            patch("data.sector_data.get_sector", side_effect=lambda s: "Technology"),
            patch("yfinance.download", return_value=_make_yf_empty()),
        ):
            from data.pairs import refresh_pairs

            result = refresh_pairs(["AAPL", "MSFT"], lookback_days=60)

        self.assertEqual(result, [])


class TestRefreshPairsUnknownSector(unittest.TestCase):
    """Symbols with unknown sector (not in get_sector or SECTOR_MAP) are skipped."""

    def test_unknown_sector_symbols_skipped(self) -> None:
        # get_sector returns Unknown and SECTOR_MAP has no entry
        with (
            patch("data.sector_data.get_sector", return_value="Unknown"),
            patch("data.pairs.SECTOR_MAP", {}),
        ):
            from data.pairs import refresh_pairs

            result = refresh_pairs(["FAKESYM1", "FAKESYM2"], lookback_days=60)

        self.assertEqual(result, [])

    def test_sector_map_fallback_used(self) -> None:
        """get_sector returns Unknown but SECTOR_MAP has a match — symbol is kept."""
        symbols = ["AAPL", "MSFT"]
        n = 60
        close_df = _make_close_df(symbols, n)

        mock_yf_data = MagicMock()
        mock_yf_data.empty = False
        mock_yf_data.__getitem__ = lambda self, key: close_df

        with (
            patch("data.sector_data.get_sector", return_value="Unknown"),
            patch("data.pairs.SECTOR_MAP", {"AAPL": "Technology", "MSFT": "Technology"}),
            patch("yfinance.download", return_value=mock_yf_data),
            patch("statsmodels.tsa.stattools.coint", return_value=(0.0, 0.01, None)),
            patch("data.pairs._save_cache"),
        ):
            from data.pairs import refresh_pairs

            result = refresh_pairs(symbols, lookback_days=60)

        self.assertGreaterEqual(len(result), 1)


class TestRefreshPairsSingleSymbolPerSector(unittest.TestCase):
    """Single symbol per sector → no pairs generated."""

    def test_single_symbol_per_sector_no_pairs(self) -> None:
        symbols = ["AAPL", "JPM"]
        n = 60
        close_df = _make_close_df(symbols, n)

        mock_yf_data = MagicMock()
        mock_yf_data.empty = False
        mock_yf_data.__getitem__ = lambda self, key: close_df

        sector_mapping = {"AAPL": "Technology", "JPM": "Financials"}

        with (
            patch(
                "data.sector_data.get_sector",
                side_effect=lambda s: sector_mapping.get(s, "Unknown"),
            ),
            patch("yfinance.download", return_value=mock_yf_data),
            patch("data.pairs._save_cache"),
        ):
            from data.pairs import refresh_pairs

            result = refresh_pairs(symbols, lookback_days=60)

        self.assertEqual(result, [])


class TestRefreshPairsSymbolMissingFromClose(unittest.TestCase):
    """Symbol in sector_map but absent from yfinance close columns is skipped."""

    def test_symbol_absent_from_close_produces_no_pairs(self) -> None:
        # sector_map has AAPL+MSFT but close only has AAPL — MSFT missing from response
        close_df = _make_close_df(["AAPL"], n=60)

        mock_yf_data = MagicMock()
        mock_yf_data.empty = False
        mock_yf_data.__getitem__ = lambda self, key: close_df

        with (
            patch("data.sector_data.get_sector", return_value="Technology"),
            patch("yfinance.download", return_value=mock_yf_data),
            patch("data.pairs._save_cache"),
        ):
            from data.pairs import refresh_pairs

            result = refresh_pairs(["AAPL", "MSFT"], lookback_days=60)

        self.assertEqual(result, [])


class TestComputeZscoreSuccess(unittest.TestCase):
    """compute_zscore returns correct dict on happy path."""

    def test_returns_zscore_dict(self) -> None:
        symbols = ["AAPL", "MSFT"]
        n = 80
        close_df = _make_close_df(symbols, n)

        mock_yf_data = MagicMock()
        mock_yf_data.empty = False
        mock_yf_data.__getitem__ = lambda self, key: close_df

        with patch("yfinance.download", return_value=mock_yf_data):
            from data.pairs import compute_zscore

            result = compute_zscore("AAPL", "MSFT", hedge_ratio=1.0, lookback_days=60)

        self.assertIsNotNone(result)
        self.assertIn("zscore", result)
        self.assertIn("spread", result)
        self.assertIn("hedge_ratio", result)
        self.assertIsInstance(result["zscore"], float)
        self.assertIsInstance(result["spread"], float)
        self.assertEqual(result["hedge_ratio"], 1.0)


class TestComputeZscoreDataFetchFailure(unittest.TestCase):
    """compute_zscore returns None when yfinance raises or returns empty."""

    def test_yfinance_exception_returns_none(self) -> None:
        with patch("yfinance.download", side_effect=RuntimeError("network error")):
            from data.pairs import compute_zscore

            result = compute_zscore("AAPL", "MSFT", hedge_ratio=1.0)

        self.assertIsNone(result)

    def test_empty_data_returns_none(self) -> None:
        with patch("yfinance.download", return_value=_make_yf_empty()):
            from data.pairs import compute_zscore

            result = compute_zscore("AAPL", "MSFT", hedge_ratio=1.0)

        self.assertIsNone(result)


class TestComputeZscoreMissingSymbols(unittest.TestCase):
    """compute_zscore returns None when requested symbols are absent from response."""

    def test_missing_symbol_in_response_returns_none(self) -> None:
        # Only AAPL in close, MSFT missing
        close_df = _make_close_df(["AAPL"], n=60)

        mock_yf_data = MagicMock()
        mock_yf_data.empty = False
        mock_yf_data.__getitem__ = lambda self, key: close_df

        with patch("yfinance.download", return_value=mock_yf_data):
            from data.pairs import compute_zscore

            result = compute_zscore("AAPL", "MSFT", hedge_ratio=1.0)

        self.assertIsNone(result)


class TestComputeZscoreTooFewRows(unittest.TestCase):
    """compute_zscore returns None when aligned data has fewer than 2 rows."""

    def test_too_few_rows_returns_none(self) -> None:
        # Only 1 row in close
        symbols = ["AAPL", "MSFT"]
        n = 1
        close_df = _make_close_df(symbols, n)

        mock_yf_data = MagicMock()
        mock_yf_data.empty = False
        mock_yf_data.__getitem__ = lambda self, key: close_df

        with patch("yfinance.download", return_value=mock_yf_data):
            from data.pairs import compute_zscore

            result = compute_zscore("AAPL", "MSFT", hedge_ratio=1.0)

        self.assertIsNone(result)


class TestComputeZscoreZeroStd(unittest.TestCase):
    """compute_zscore returns None when spread std is zero (flat spread)."""

    def test_zero_std_returns_none(self) -> None:
        n = 60
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        # Identical prices → spread = 0 constantly → std = 0
        close_df = pd.DataFrame({"AAPL": [100.0] * n, "MSFT": [100.0] * n}, index=idx)

        mock_yf_data = MagicMock()
        mock_yf_data.empty = False
        mock_yf_data.__getitem__ = lambda self, key: close_df

        with patch("yfinance.download", return_value=mock_yf_data):
            from data.pairs import compute_zscore

            result = compute_zscore("AAPL", "MSFT", hedge_ratio=1.0)

        self.assertIsNone(result)


class TestCacheHelpers(unittest.TestCase):
    """Unit tests for _load_cache, _save_cache, _cache_age_days."""

    def test_load_cache_missing_file(self) -> None:
        with patch("builtins.open", side_effect=FileNotFoundError):
            from data.pairs import _load_cache

            result = _load_cache()
        self.assertEqual(result, {})

    def test_load_cache_corrupt_json(self) -> None:
        import io

        with patch("builtins.open", return_value=io.StringIO("not-json{{")):
            from data.pairs import _load_cache

            result = _load_cache()
        self.assertEqual(result, {})

    def test_save_cache_oserror_swallowed(self) -> None:
        with (
            patch("os.makedirs"),
            patch("builtins.open", side_effect=OSError("disk full")),
        ):
            from data.pairs import _save_cache

            # Should not raise
            _save_cache({"generated_at": "2024-01-01", "pairs": []})

    def test_cache_age_days_no_key(self) -> None:
        from data.pairs import _cache_age_days

        age = _cache_age_days({})
        self.assertEqual(age, float("inf"))

    def test_cache_age_days_corrupt_value(self) -> None:
        from data.pairs import _cache_age_days

        age = _cache_age_days({"generated_at": "not-a-date"})
        self.assertEqual(age, float("inf"))

    def test_cache_age_days_naive_datetime(self) -> None:
        """Naive ISO string (no timezone) is treated as corrupt — returns inf."""
        from data.pairs import _cache_age_days

        naive_ts = datetime.now().isoformat()  # no tz info
        age = _cache_age_days({"generated_at": naive_ts})
        self.assertEqual(age, float("inf"))

    def test_save_and_load_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "pairs_cache.json")
            payload = {
                "generated_at": datetime.now(UTC).isoformat(),
                "pairs": [{"sym_a": "X", "sym_b": "Y"}],
            }
            with patch("data.pairs._CACHE_PATH", cache_path):
                from data.pairs import _load_cache, _save_cache

                _save_cache(payload)
                result = _load_cache()

        self.assertEqual(result["pairs"], payload["pairs"])


class TestRefreshPairsPairLevelException(unittest.TestCase):
    """An exception during coint for a specific pair is skipped; others continue."""

    def test_pair_exception_is_skipped(self) -> None:
        symbols = ["AAPL", "MSFT", "GOOGL"]
        n = 60
        close_df = _make_close_df(symbols, n)

        mock_yf_data = MagicMock()
        mock_yf_data.empty = False
        mock_yf_data.__getitem__ = lambda self, key: close_df

        call_count = {"n": 0}

        def coint_side_effect(a, b):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("coint boom")
            return (0.0, 0.01, None)

        with (
            patch("data.sector_data.get_sector", return_value="Technology"),
            patch("yfinance.download", return_value=mock_yf_data),
            patch("statsmodels.tsa.stattools.coint", side_effect=coint_side_effect),
            patch("data.pairs._save_cache"),
        ):
            from data.pairs import refresh_pairs

            result = refresh_pairs(symbols, lookback_days=60)

        # At least one pair should succeed (coint called 3 times for 3 pairs)
        self.assertGreaterEqual(len(result), 1)


class TestRefreshPairsTooFewRows(unittest.TestCase):
    """Pairs where aligned series has < 30 rows are skipped."""

    def test_too_few_rows_pair_skipped(self) -> None:
        symbols = ["AAPL", "MSFT"]
        n = 10  # fewer than 30
        close_df = _make_close_df(symbols, n)

        mock_yf_data = MagicMock()
        mock_yf_data.empty = False
        mock_yf_data.__getitem__ = lambda self, key: close_df

        with (
            patch("data.sector_data.get_sector", return_value="Technology"),
            patch("yfinance.download", return_value=mock_yf_data),
            patch("data.pairs._save_cache"),
        ):
            from data.pairs import refresh_pairs

            result = refresh_pairs(symbols, lookback_days=10)

        self.assertEqual(result, [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
