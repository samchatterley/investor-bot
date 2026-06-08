"""Tests for data/sector_momentum.py — sector ETF ranking and long/short gates."""

import json
import os
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from data.sector_momentum import (
    _SPDR_MAP,
    _load_cache,
    _save_cache,
    get_sector_momentum_ranks,
    sector_allowed_long,
    sector_allowed_short,
)


class TestSectorAllowedLong(unittest.TestCase):
    def test_empty_ranks_allows_long(self):
        self.assertTrue(sector_allowed_long("Technology", {}))

    def test_top_four_rank_allows_long(self):
        ranks = {"Technology": 1, "Healthcare": 2, "Financials": 3, "Energy": 4}
        self.assertTrue(sector_allowed_long("Technology", ranks))
        self.assertTrue(sector_allowed_long("Energy", ranks))

    def test_rank_five_blocks_long(self):
        ranks = {s: i + 1 for i, s in enumerate(_SPDR_MAP.values())}
        # Assign rank 5 to Industrials explicitly
        ranks["Industrials"] = 5
        self.assertFalse(sector_allowed_long("Industrials", ranks))

    def test_rank_eleven_blocks_long(self):
        ranks = {"Technology": 11}
        self.assertFalse(sector_allowed_long("Technology", ranks))

    def test_unknown_sector_allows_long(self):
        ranks = {"Technology": 1}
        self.assertTrue(sector_allowed_long("ETF", ranks))
        self.assertTrue(sector_allowed_long("Unknown", ranks))


class TestSectorAllowedShort(unittest.TestCase):
    def test_empty_ranks_allows_short(self):
        self.assertTrue(sector_allowed_short("Utilities", {}))

    def test_bottom_three_ranks_allow_short(self):
        ranks = {"Utilities": 9, "Real Estate": 10, "Materials": 11}
        self.assertTrue(sector_allowed_short("Utilities", ranks))
        self.assertTrue(sector_allowed_short("Materials", ranks))

    def test_rank_eight_blocks_short(self):
        ranks = {"Utilities": 8}
        self.assertFalse(sector_allowed_short("Utilities", ranks))

    def test_rank_one_blocks_short(self):
        ranks = {"Technology": 1}
        self.assertFalse(sector_allowed_short("Technology", ranks))

    def test_unknown_sector_allows_short(self):
        ranks = {"Technology": 1}
        self.assertTrue(sector_allowed_short("ETF", ranks))


class TestGetSectorMomentumRanks(unittest.TestCase):
    def _make_price_df(self, tickers: list[str], n: int = 25):
        """Build a fake multi-column close DataFrame for yf.download."""
        idx = pd.bdate_range("2024-01-01", periods=n)
        data = {t: [100.0 + i * (0.5 if t == "XLK" else -0.2) for i in range(n)] for t in tickers}
        closes = pd.DataFrame(data, index=idx)
        mock_df = MagicMock()
        mock_df.empty = False
        mock_df.__getitem__ = lambda self, key: closes
        mock_df.columns = pd.MultiIndex.from_tuples([("Close", t) for t in tickers])
        return mock_df, closes

    def test_returns_empty_dict_on_download_failure(self):
        with (
            patch("yfinance.download", side_effect=Exception("network")),
            patch("data.sector_momentum._load_cache", return_value=None),
        ):
            result = get_sector_momentum_ranks()
        self.assertEqual(result, {})

    def test_returns_empty_dict_on_empty_dataframe(self):
        mock_df = MagicMock()
        mock_df.empty = True
        with (
            patch("yfinance.download", return_value=mock_df),
            patch("data.sector_momentum._load_cache", return_value=None),
        ):
            result = get_sector_momentum_ranks()
        self.assertEqual(result, {})

    def test_returns_dict_with_all_sectors_ranked(self):
        tickers = list(_SPDR_MAP.keys())
        n = 25
        idx = pd.bdate_range("2024-01-01", periods=n)
        closes_data = {
            t: [100.0 + i * (0.5 if t == "XLK" else 0.1) for i in range(n)] for t in tickers
        }
        closes = pd.DataFrame(closes_data, index=idx)
        mock_df = MagicMock()
        mock_df.empty = False
        mock_df.columns = pd.MultiIndex.from_tuples([("Close", t) for t in tickers])
        mock_df.__getitem__ = lambda self, key: closes

        with (
            patch("yfinance.download", return_value=mock_df),
            patch("data.sector_momentum._load_cache", return_value=None),
            patch("data.sector_momentum._save_cache"),
        ):
            result = get_sector_momentum_ranks()

        self.assertGreater(len(result), 0)
        for sector in result:
            self.assertIn(result[sector], range(1, 12))

    def test_cache_returned_when_fresh(self):
        cached_ranks = {"Technology": 1, "Healthcare": 2}
        with patch("data.sector_momentum._load_cache", return_value=cached_ranks):
            result = get_sector_momentum_ranks()
        self.assertEqual(result, cached_ranks)

    def test_spdr_map_has_eleven_entries(self):
        self.assertEqual(len(_SPDR_MAP), 11)

    def test_all_spdr_sectors_in_map(self):
        sectors = set(_SPDR_MAP.values())
        expected = {
            "Technology",
            "Financials",
            "Healthcare",
            "Energy",
            "Industrials",
            "Communication Services",
            "Consumer Discretionary",
            "Consumer Staples",
            "Real Estate",
            "Materials",
            "Utilities",
        }
        self.assertEqual(sectors, expected)

    def test_non_multiindex_closes_handled(self):
        """Plain (non-MultiIndex) yfinance response uses df directly (line 87)."""
        tickers = list(_SPDR_MAP.keys())
        n = 25
        idx = pd.bdate_range("2024-01-01", periods=n)
        df = pd.DataFrame({t: [100.0 + i * 0.1 for i in range(n)] for t in tickers}, index=idx)
        with (
            patch("yfinance.download", return_value=df),
            patch("data.sector_momentum._load_cache", return_value=None),
            patch("data.sector_momentum._save_cache"),
        ):
            result = get_sector_momentum_ranks()
        self.assertGreater(len(result), 0)

    def test_ticker_missing_from_closes_is_skipped(self):
        """Missing tickers are skipped gracefully (line 92)."""
        tickers = list(_SPDR_MAP.keys())
        present = tickers[:6]
        n = 25
        idx = pd.bdate_range("2024-01-01", periods=n)
        closes = pd.DataFrame({t: [100.0 + i * 0.1 for i in range(n)] for t in present}, index=idx)
        mock_df = MagicMock()
        mock_df.empty = False
        mock_df.columns = pd.MultiIndex.from_tuples([("Close", t) for t in tickers])
        mock_df.__getitem__ = lambda self, key: closes
        with (
            patch("yfinance.download", return_value=mock_df),
            patch("data.sector_momentum._load_cache", return_value=None),
            patch("data.sector_momentum._save_cache"),
        ):
            result = get_sector_momentum_ranks()
        self.assertLessEqual(len(result), len(present))
        self.assertGreater(len(result), 0)

    def test_ticker_with_too_few_rows_is_skipped(self):
        """Ticker columns with < 5 non-NaN values are skipped (line 95)."""
        tickers = list(_SPDR_MAP.keys())
        n = 25
        idx = pd.bdate_range("2024-01-01", periods=n)
        closes_data = {t: [100.0 + i * 0.1 for i in range(n)] for t in tickers}
        closes_data["XLK"] = [float("nan")] * 22 + [100.0, 101.0, 102.0]
        closes = pd.DataFrame(closes_data, index=idx)
        mock_df = MagicMock()
        mock_df.empty = False
        mock_df.columns = pd.MultiIndex.from_tuples([("Close", t) for t in tickers])
        mock_df.__getitem__ = lambda self, key: closes
        with (
            patch("yfinance.download", return_value=mock_df),
            patch("data.sector_momentum._load_cache", return_value=None),
            patch("data.sector_momentum._save_cache"),
        ):
            result = get_sector_momentum_ranks()
        self.assertNotIn("Technology", result)

    def test_returns_empty_dict_when_all_tickers_have_too_few_rows(self):
        """Returns {} when all tickers have < 5 non-NaN rows (line 100)."""
        tickers = list(_SPDR_MAP.keys())
        n = 25
        idx = pd.bdate_range("2024-01-01", periods=n)
        closes_data = {t: [float("nan")] * 22 + [100.0, 101.0, 102.0] for t in tickers}
        closes = pd.DataFrame(closes_data, index=idx)
        mock_df = MagicMock()
        mock_df.empty = False
        mock_df.columns = pd.MultiIndex.from_tuples([("Close", t) for t in tickers])
        mock_df.__getitem__ = lambda self, key: closes
        with (
            patch("yfinance.download", return_value=mock_df),
            patch("data.sector_momentum._load_cache", return_value=None),
        ):
            result = get_sector_momentum_ranks()
        self.assertEqual(result, {})


class TestLoadSaveCache(unittest.TestCase):
    def test_stale_cache_returns_none(self):
        """_load_cache returns None when cached data is older than TTL (line 51->55)."""
        from data.sector_momentum import _CACHE_TTL_SECONDS

        stale = {"ts": time.time() - _CACHE_TTL_SECONDS - 1, "ranks": {"Technology": 1}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(stale, f)
            cache_path = f.name
        try:
            with patch("data.sector_momentum._CACHE_PATH", cache_path):
                result = _load_cache()
            self.assertIsNone(result)
        finally:
            os.unlink(cache_path)

    def test_fresh_cache_returns_ranks(self):
        """_load_cache returns ranks when cache is within TTL (line 52)."""
        fresh = {"ts": time.time(), "ranks": {"Technology": 1, "Healthcare": 2}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(fresh, f)
            cache_path = f.name
        try:
            with patch("data.sector_momentum._CACHE_PATH", cache_path):
                result = _load_cache()
            self.assertEqual(result, {"Technology": 1, "Healthcare": 2})
        finally:
            os.unlink(cache_path)

    def test_corrupt_cache_returns_none(self):
        """_load_cache returns None on corrupt JSON (lines 53-54)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not-valid-json{{{")
            cache_path = f.name
        try:
            with patch("data.sector_momentum._CACHE_PATH", cache_path):
                result = _load_cache()
            self.assertIsNone(result)
        finally:
            os.unlink(cache_path)

    def test_save_cache_writes_file(self):
        """_save_cache writes JSON successfully (line 61)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            cache_path = f.name
        try:
            with patch("data.sector_momentum._CACHE_PATH", cache_path):
                _save_cache({"Technology": 1})
            with open(cache_path) as f:
                data = json.load(f)
            self.assertEqual(data["ranks"], {"Technology": 1})
        finally:
            os.unlink(cache_path)

    def test_save_cache_silences_write_exception(self):
        """_save_cache logs a warning but does not raise on OSError (lines 62-63)."""
        with patch("builtins.open", side_effect=OSError("read-only fs")):
            _save_cache({"Technology": 1})
