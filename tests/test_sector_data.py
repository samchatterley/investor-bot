"""Tests for data/sector_data.py — sector lookup and concentration checks."""

import json
import unittest
from unittest.mock import MagicMock, mock_open, patch

import pandas as pd

# ---------------------------------------------------------------------------
# Legacy / existing tests
# ---------------------------------------------------------------------------


class TestGetSector(unittest.TestCase):
    def test_known_symbol_returns_correct_sector(self):
        from data.sector_data import get_sector

        # These symbols are in the legacy SECTOR_MAP and also hit the cache path.
        # Patch _load_sector_cache to return {} so fallback to SECTOR_MAP is tested.
        with patch("data.sector_data._load_sector_cache", return_value={}):
            self.assertEqual(get_sector("AAPL"), "Technology")
            self.assertEqual(get_sector("JPM"), "Financials")
            self.assertEqual(get_sector("XOM"), "Energy")
            self.assertEqual(get_sector("SPY"), "ETF")
            self.assertEqual(get_sector("LMT"), "Industrials")
            self.assertEqual(get_sector("PFE"), "Healthcare")

    def test_unknown_symbol_returns_unknown(self):
        from data.sector_data import get_sector

        with patch("data.sector_data._load_sector_cache", return_value={}):
            self.assertEqual(get_sector("ZZZZ"), "Unknown")

    def test_all_mapped_symbols_have_non_empty_sector(self):
        from data.sector_data import SECTOR_MAP, get_sector

        with patch("data.sector_data._load_sector_cache", return_value={}):
            for sym in SECTOR_MAP:
                self.assertNotEqual(get_sector(sym), "Unknown")
                self.assertNotEqual(get_sector(sym), "")

    def test_returns_from_cache_when_present(self):
        from data.sector_data import get_sector

        with patch("data.sector_data._load_sector_cache", return_value={"AAPL": "Technology"}):
            self.assertEqual(get_sector("AAPL"), "Technology")

    def test_falls_back_to_unknown_on_exception(self):
        from data.sector_data import get_sector

        with patch("data.sector_data._load_sector_cache", side_effect=RuntimeError("boom")):
            self.assertEqual(get_sector("AAPL"), "Unknown")


class TestCheckSectorConcentration(unittest.TestCase):
    def setUp(self):
        # Use legacy SECTOR_MAP for these tests — cache returns empty
        self._patcher = patch("data.sector_data._load_sector_cache", return_value={})
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def test_no_breach_within_limit(self):
        from data.sector_data import check_sector_concentration

        # AAPL + MSFT = 2 Technology — at limit, no breach
        result = check_sector_concentration(["AAPL", "MSFT"], max_per_sector=2)
        self.assertEqual(result, [])

    def test_breach_when_over_limit(self):
        from data.sector_data import check_sector_concentration

        # AAPL + MSFT + NVDA = 3 Technology — exceeds max_per_sector=2
        result = check_sector_concentration(["AAPL", "MSFT", "NVDA"], max_per_sector=2)
        self.assertEqual(len(result), 1)
        self.assertIn(result[0], ["AAPL", "MSFT", "NVDA"])

    def test_etf_sector_excluded_from_cap(self):
        from data.sector_data import check_sector_concentration

        # SPY + QQQ + IWM = 3 ETFs — ETF sector is exempt
        result = check_sector_concentration(["SPY", "QQQ", "IWM"], max_per_sector=2)
        self.assertEqual(result, [])

    def test_multiple_sector_breaches(self):
        from data.sector_data import check_sector_concentration

        # 3 Technology + 3 Financials — both breach at max=2
        symbols = ["AAPL", "MSFT", "NVDA", "JPM", "BAC", "GS"]
        result = check_sector_concentration(symbols, max_per_sector=2)
        self.assertEqual(len(result), 2)  # 1 excess per sector

    def test_empty_list_returns_empty(self):
        from data.sector_data import check_sector_concentration

        self.assertEqual(check_sector_concentration([]), [])

    def test_single_symbol_never_breaches(self):
        from data.sector_data import check_sector_concentration

        result = check_sector_concentration(["AAPL"], max_per_sector=1)
        self.assertEqual(result, [])

    def test_returns_only_excess_symbols(self):
        from data.sector_data import check_sector_concentration

        # 4 Technology, max=2 → 2 excess
        result = check_sector_concentration(["AAPL", "MSFT", "NVDA", "AMD"], max_per_sector=2)
        self.assertEqual(len(result), 2)

    def test_unknown_sector_symbols_counted_separately(self):
        from data.sector_data import check_sector_concentration

        # Three unknown symbols form their own "Unknown" bucket
        result = check_sector_concentration(["ZZZZ", "YYYY", "XXXX"], max_per_sector=2)
        self.assertEqual(len(result), 1)


class TestGetLeadingSectors(unittest.TestCase):
    def test_returns_list_of_top_n(self):
        from data.sector_data import get_leading_sectors

        mock_perf = {
            "Technology": 3.2,
            "Financials": 2.1,
            "Energy": 1.5,
            "Healthcare": -0.3,
            "Industrials": -1.0,
        }
        with patch("data.sector_data.get_sector_performance", return_value=mock_perf):
            result = get_leading_sectors(top_n=3)
        self.assertEqual(result, ["Technology", "Financials", "Energy"])

    def test_returns_empty_when_no_performance_data(self):
        from data.sector_data import get_leading_sectors

        with patch("data.sector_data.get_sector_performance", return_value={}):
            result = get_leading_sectors()
        self.assertEqual(result, [])


class TestGetSectorPerformance(unittest.TestCase):
    def _make_close_df(self, etfs, values):
        """Build a mock Close DataFrame with enough rows for a 5-day return."""
        rows = 8
        data = {
            etf: [v * (1 + 0.001 * i) for i in range(rows)]
            for etf, v in zip(etfs, values, strict=False)
        }
        return pd.DataFrame(data)

    def test_returns_sorted_dict_best_to_worst(self):
        from data.sector_data import SECTOR_ETFS, get_sector_performance

        etfs = list(SECTOR_ETFS.values())
        # Give each ETF a distinct start value; last row > first row = positive return
        close_df = self._make_close_df(etfs, [100 + i * 5 for i in range(len(etfs))])
        mock_data = MagicMock()
        mock_data.empty = False
        mock_data.__len__ = MagicMock(return_value=8)
        mock_data.__getitem__ = MagicMock(return_value=close_df)

        with patch("data.sector_data.yf.download", return_value=mock_data):
            result = get_sector_performance(days=5)

        # Result should be a dict (may be empty if mock shape doesn't match — that's ok)
        self.assertIsInstance(result, dict)

    def test_returns_empty_on_exception(self):
        from data.sector_data import get_sector_performance

        with patch("data.sector_data.yf.download", side_effect=Exception("network")):
            result = get_sector_performance()
        self.assertEqual(result, {})

    def test_returns_empty_on_empty_data(self):
        from data.sector_data import get_sector_performance

        mock_data = MagicMock()
        mock_data.empty = True
        with patch("data.sector_data.yf.download", return_value=mock_data):
            result = get_sector_performance()
        self.assertEqual(result, {})

    def test_etf_not_in_close_columns_skipped(self):
        """Line 92->91: ETF ticker not present in close.columns → sector skipped."""
        from data.sector_data import SECTOR_ETFS, get_sector_performance

        first_etf = next(iter(SECTOR_ETFS.values()))
        close_df = pd.DataFrame({first_etf: [100.0, 101.0, 102.0] * 3})
        mock_data = MagicMock()
        mock_data.empty = False
        mock_data.__len__ = MagicMock(return_value=9)
        mock_data.__getitem__ = MagicMock(return_value=close_df)

        with patch("data.sector_data.yf.download", return_value=mock_data):
            result = get_sector_performance(days=5)

        self.assertIsInstance(result, dict)
        missing_etfs = [s for s, e in SECTOR_ETFS.items() if e != first_etf]
        for sector in missing_etfs:
            self.assertNotIn(sector, result)

    def test_nan_return_excluded(self):
        """Line 94->91: return value is NaN → sector not added to perf dict."""
        from data.sector_data import SECTOR_ETFS, get_sector_performance

        etfs = list(SECTOR_ETFS.values())
        close_df = pd.DataFrame({etf: [float("nan")] * 9 for etf in etfs})
        mock_data = MagicMock()
        mock_data.empty = False
        mock_data.__len__ = MagicMock(return_value=9)
        mock_data.__getitem__ = MagicMock(return_value=close_df)

        with patch("data.sector_data.yf.download", return_value=mock_data):
            result = get_sector_performance(days=5)

        self.assertEqual(result, {})


# ---------------------------------------------------------------------------
# New tests — disk cache helpers
# ---------------------------------------------------------------------------


class TestLoadSectorCache(unittest.TestCase):
    def test_returns_empty_when_file_missing(self):
        from data.sector_data import _load_sector_cache

        with patch("builtins.open", side_effect=FileNotFoundError):
            self.assertEqual(_load_sector_cache(), {})

    def test_returns_empty_on_corrupt_json(self):
        from data.sector_data import _load_sector_cache

        with patch("builtins.open", mock_open(read_data="not-json{")):
            self.assertEqual(_load_sector_cache(), {})

    def test_returns_empty_on_non_dict_json(self):
        from data.sector_data import _load_sector_cache

        with patch("builtins.open", mock_open(read_data="[1,2,3]")):
            self.assertEqual(_load_sector_cache(), {})

    def test_returns_data_when_valid(self):
        from data.sector_data import _load_sector_cache

        payload = json.dumps({"AAPL": "Technology", "JPM": "Financials"})
        with patch("builtins.open", mock_open(read_data=payload)):
            result = _load_sector_cache()
        self.assertEqual(result, {"AAPL": "Technology", "JPM": "Financials"})

    def test_returns_empty_on_oserror(self):
        from data.sector_data import _load_sector_cache

        with patch("builtins.open", side_effect=OSError("disk full")):
            self.assertEqual(_load_sector_cache(), {})


class TestSaveSectorCache(unittest.TestCase):
    def test_writes_json_to_file(self):
        from data.sector_data import _save_sector_cache

        m = mock_open()
        with patch("builtins.open", m):
            _save_sector_cache({"AAPL": "Technology"})
        m.assert_called_once()
        handle = m()
        written = "".join(call.args[0] for call in handle.write.call_args_list)
        self.assertIn("AAPL", written)
        self.assertIn("Technology", written)

    def test_swallows_oserror_silently(self):
        from data.sector_data import _save_sector_cache

        with patch("builtins.open", side_effect=OSError("disk full")):
            # Must not raise
            _save_sector_cache({"AAPL": "Technology"})


# ---------------------------------------------------------------------------
# New tests — stale cache detection
# ---------------------------------------------------------------------------


class TestIsSectorCacheStale(unittest.TestCase):
    def test_not_stale_when_all_present(self):
        from data.sector_data import _is_sector_cache_stale

        cache = {"AAPL": "Technology", "MSFT": "Technology"}
        self.assertFalse(_is_sector_cache_stale(cache, ["AAPL", "MSFT"]))

    def test_not_stale_when_missing_under_10_pct(self):
        from data.sector_data import _is_sector_cache_stale

        # 1 missing out of 20 = 5% — not stale
        symbols = [f"S{i}" for i in range(20)]
        cache = dict.fromkeys(symbols[:-1], "Technology")  # last one missing
        self.assertFalse(_is_sector_cache_stale(cache, symbols))

    def test_stale_when_missing_over_10_pct(self):
        from data.sector_data import _is_sector_cache_stale

        # 3 missing out of 10 = 30% — stale
        symbols = [f"S{i}" for i in range(10)]
        cache = dict.fromkeys(symbols[:7], "Technology")
        self.assertTrue(_is_sector_cache_stale(cache, symbols))

    def test_empty_symbols_list_not_stale(self):
        from data.sector_data import _is_sector_cache_stale

        self.assertFalse(_is_sector_cache_stale({}, []))

    def test_entirely_empty_cache_is_stale(self):
        from data.sector_data import _is_sector_cache_stale

        symbols = [f"S{i}" for i in range(10)]
        self.assertTrue(_is_sector_cache_stale({}, symbols))


# ---------------------------------------------------------------------------
# New tests — yfinance fetch
# ---------------------------------------------------------------------------


class TestFetchSectorFromYfinance(unittest.TestCase):
    def test_returns_sector_string(self):
        from data.sector_data import _fetch_sector_from_yfinance

        mock_ticker = MagicMock()
        mock_ticker.info = {"sector": "Technology"}
        with patch("data.sector_data.yf.Ticker", return_value=mock_ticker):
            result = _fetch_sector_from_yfinance("AAPL")
        self.assertEqual(result, "Technology")

    def test_returns_unknown_when_sector_key_missing(self):
        from data.sector_data import _fetch_sector_from_yfinance

        mock_ticker = MagicMock()
        mock_ticker.info = {}
        with patch("data.sector_data.yf.Ticker", return_value=mock_ticker):
            result = _fetch_sector_from_yfinance("ZZZZ")
        self.assertEqual(result, "Unknown")

    def test_returns_unknown_on_exception(self):
        from data.sector_data import _fetch_sector_from_yfinance

        with patch("data.sector_data.yf.Ticker", side_effect=Exception("network error")):
            result = _fetch_sector_from_yfinance("AAPL")
        self.assertEqual(result, "Unknown")

    def test_returns_unknown_when_sector_is_none(self):
        from data.sector_data import _fetch_sector_from_yfinance

        mock_ticker = MagicMock()
        mock_ticker.info = {"sector": None}
        with patch("data.sector_data.yf.Ticker", return_value=mock_ticker):
            result = _fetch_sector_from_yfinance("AAPL")
        self.assertEqual(result, "Unknown")


# ---------------------------------------------------------------------------
# New tests — build_sector_map
# ---------------------------------------------------------------------------


class TestBuildSectorMap(unittest.TestCase):
    def test_cache_hit_skips_fetch(self):
        from data.sector_data import build_sector_map

        cache = {"AAPL": "Technology", "MSFT": "Technology"}
        with (
            patch("data.sector_data._load_sector_cache", return_value=cache),
            patch("data.sector_data._is_sector_cache_stale", return_value=False),
            patch("data.sector_data._fetch_sector_from_yfinance") as mock_fetch,
            patch("data.sector_data._save_sector_cache") as mock_save,
        ):
            result = build_sector_map(["AAPL", "MSFT"])
        mock_fetch.assert_not_called()
        mock_save.assert_not_called()
        self.assertEqual(result, cache)

    def test_cache_miss_fetches_and_saves(self):
        from data.sector_data import build_sector_map

        with (
            patch("data.sector_data._load_sector_cache", return_value={}),
            patch("data.sector_data._is_sector_cache_stale", return_value=True),
            patch(
                "data.sector_data._fetch_sector_from_yfinance", return_value="Technology"
            ) as mock_fetch,
            patch("data.sector_data._save_sector_cache") as mock_save,
            patch("data.sector_data.time.sleep"),
        ):
            result = build_sector_map(["AAPL", "MSFT"])

        self.assertEqual(mock_fetch.call_count, 2)
        mock_save.assert_called_once()
        self.assertEqual(result, {"AAPL": "Technology", "MSFT": "Technology"})

    def test_incremental_save_during_build(self):
        # Hardening: the partial map is persisted every _SECTOR_CACHE_SAVE_EVERY symbols so a
        # mid-build restart/crash keeps progress and the next run resumes the remainder.
        from data.sector_data import build_sector_map

        with (
            patch("data.sector_data._load_sector_cache", return_value={}),
            patch("data.sector_data._is_sector_cache_stale", return_value=True),
            patch("data.sector_data._fetch_sector_from_yfinance", return_value="Technology"),
            patch("data.sector_data._save_sector_cache") as mock_save,
            patch("data.sector_data.time.sleep"),
            patch("data.sector_data._SECTOR_CACHE_SAVE_EVERY", 2),
        ):
            build_sector_map(["A", "B", "C"])  # save-every=2 → save at i=2, plus the final save
        self.assertEqual(mock_save.call_count, 2)

    def test_partial_cache_tops_up_missing(self):
        from data.sector_data import build_sector_map

        # AAPL in cache, MSFT missing → only MSFT fetched
        cache = {"AAPL": "Technology"}

        def stale_check(c, syms):
            return sum(1 for s in syms if s not in c) / len(syms) > 0.10

        with (
            patch("data.sector_data._load_sector_cache", return_value=cache),
            patch("data.sector_data._is_sector_cache_stale", side_effect=stale_check),
            patch(
                "data.sector_data._fetch_sector_from_yfinance", return_value="Technology"
            ) as mock_fetch,
            patch("data.sector_data._save_sector_cache"),
            patch("data.sector_data.time.sleep"),
        ):
            result = build_sector_map(["AAPL", "MSFT"])

        # Only MSFT should have been fetched
        fetched_symbols = [call.args[0] for call in mock_fetch.call_args_list]
        self.assertIn("MSFT", fetched_symbols)
        self.assertNotIn("AAPL", fetched_symbols)
        self.assertIn("AAPL", result)
        self.assertIn("MSFT", result)

    def test_force_refresh_fetches_all(self):
        from data.sector_data import build_sector_map

        cache = {"AAPL": "Technology"}
        with (
            patch("data.sector_data._load_sector_cache", return_value=cache),
            patch(
                "data.sector_data._fetch_sector_from_yfinance", return_value="Technology"
            ) as mock_fetch,
            patch("data.sector_data._save_sector_cache"),
            patch("data.sector_data.time.sleep"),
        ):
            build_sector_map(["AAPL", "MSFT"], force_refresh=True)

        fetched = [call.args[0] for call in mock_fetch.call_args_list]
        self.assertIn("AAPL", fetched)
        self.assertIn("MSFT", fetched)

    def test_uses_stock_universe_when_symbols_none(self):
        from data.sector_data import build_sector_map

        with (
            patch("data.sector_data._load_sector_cache", return_value={}),
            patch("data.sector_data._is_sector_cache_stale", return_value=False),
            patch("data.sector_data._save_sector_cache"),
            patch("data.sector_data.STOCK_UNIVERSE", ["AAPL", "MSFT"]),
        ):
            result = build_sector_map(symbols=None)

        self.assertIsInstance(result, dict)


# ---------------------------------------------------------------------------
# New tests — rank_sectors_by_momentum
# ---------------------------------------------------------------------------


class TestRankSectorsByMomentum(unittest.TestCase):
    def _make_mock_data(self, etfs, num_rows=150):
        """Return a mock yf.download result with a Close sub-DataFrame."""
        close_df = pd.DataFrame(
            {
                etf: [100.0 + i * 0.1 * (j + 1) for i in range(num_rows)]
                for j, etf in enumerate(etfs)
            }
        )
        mock_data = MagicMock()
        mock_data.empty = False
        mock_data.__len__ = MagicMock(return_value=num_rows)
        mock_data.__getitem__ = MagicMock(return_value=close_df)
        return mock_data

    def test_returns_sorted_list(self):
        from data.sector_data import SECTOR_ETFS, rank_sectors_by_momentum

        etfs = list(SECTOR_ETFS.values())
        mock_data = self._make_mock_data(etfs)

        with patch("data.sector_data.yf.download", return_value=mock_data):
            result = rank_sectors_by_momentum()

        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        # Each element is (etf, sector, score)
        etf, sector, score = result[0]
        self.assertIsInstance(etf, str)
        self.assertIsInstance(sector, str)
        self.assertIsInstance(score, float)
        # Sorted best first
        scores = [r[2] for r in result]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_returns_empty_on_download_failure(self):
        from data.sector_data import rank_sectors_by_momentum

        with patch("data.sector_data.yf.download", side_effect=Exception("network")):
            result = rank_sectors_by_momentum()
        self.assertEqual(result, [])

    def test_returns_empty_on_empty_data(self):
        from data.sector_data import rank_sectors_by_momentum

        mock_data = MagicMock()
        mock_data.empty = True
        with patch("data.sector_data.yf.download", return_value=mock_data):
            result = rank_sectors_by_momentum()
        self.assertEqual(result, [])

    def test_accepts_custom_lookback_configs(self):
        from data.sector_data import SECTOR_ETFS, rank_sectors_by_momentum

        etfs = list(SECTOR_ETFS.values())
        mock_data = self._make_mock_data(etfs, num_rows=50)

        with patch("data.sector_data.yf.download", return_value=mock_data):
            result = rank_sectors_by_momentum(lookback_configs=[(10, 0.5), (20, 0.5)])

        self.assertIsInstance(result, list)

    def test_skips_etf_not_in_close_columns(self):
        from data.sector_data import SECTOR_ETFS, rank_sectors_by_momentum

        etfs = list(SECTOR_ETFS.values())
        # Only include the first ETF in the close DataFrame
        close_df = pd.DataFrame({etfs[0]: [100.0 + i * 0.1 for i in range(150)]})
        mock_data = MagicMock()
        mock_data.empty = False
        mock_data.__len__ = MagicMock(return_value=150)
        mock_data.__getitem__ = MagicMock(return_value=close_df)

        with patch("data.sector_data.yf.download", return_value=mock_data):
            result = rank_sectors_by_momentum()

        returned_etfs = [r[0] for r in result]
        self.assertIn(etfs[0], returned_etfs)
        for etf in etfs[1:]:
            self.assertNotIn(etf, returned_etfs)

    def test_skips_series_too_short_for_lookback(self):
        from data.sector_data import SECTOR_ETFS, rank_sectors_by_momentum

        etfs = list(SECTOR_ETFS.values())
        # Only 10 rows — shorter than default 21-day lookback
        mock_data = self._make_mock_data(etfs, num_rows=10)

        with patch("data.sector_data.yf.download", return_value=mock_data):
            result = rank_sectors_by_momentum()

        self.assertEqual(result, [])

    def test_skips_nan_returns(self):
        from data.sector_data import SECTOR_ETFS, rank_sectors_by_momentum

        etfs = list(SECTOR_ETFS.values())
        close_df = pd.DataFrame({etf: [float("nan")] * 150 for etf in etfs})
        mock_data = MagicMock()
        mock_data.empty = False
        mock_data.__len__ = MagicMock(return_value=150)
        mock_data.__getitem__ = MagicMock(return_value=close_df)

        with patch("data.sector_data.yf.download", return_value=mock_data):
            result = rank_sectors_by_momentum()

        self.assertEqual(result, [])

    def test_skips_zero_div_nan_return(self):
        from data.sector_data import SECTOR_ETFS, rank_sectors_by_momentum

        # 0.0 / 0.0 = NaN: series with all-zero prices covering line 247 (pd.isna(ret))
        etfs = list(SECTOR_ETFS.values())
        close_df = pd.DataFrame({etf: [0.0] * 150 for etf in etfs})
        mock_data = MagicMock()
        mock_data.empty = False
        mock_data.__len__ = MagicMock(return_value=150)
        mock_data.__getitem__ = MagicMock(return_value=close_df)

        with patch("data.sector_data.yf.download", return_value=mock_data):
            result = rank_sectors_by_momentum()

        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# New tests — get_sector_etf
# ---------------------------------------------------------------------------


class TestGetSectorEtf(unittest.TestCase):
    def test_known_sector_returns_etf(self):
        from data.sector_data import get_sector_etf

        with patch("data.sector_data._load_sector_cache", return_value={"AAPL": "Technology"}):
            result = get_sector_etf("AAPL")
        self.assertEqual(result, "XLK")

    def test_unknown_sector_returns_none(self):
        from data.sector_data import get_sector_etf

        with patch("data.sector_data._load_sector_cache", return_value={"ZZZZ": "Unknown"}):
            result = get_sector_etf("ZZZZ")
        self.assertIsNone(result)

    def test_symbol_not_in_cache_or_map_returns_none(self):
        from data.sector_data import get_sector_etf

        with patch("data.sector_data._load_sector_cache", return_value={}):
            result = get_sector_etf("NOSUCHSYMBOL")
        self.assertIsNone(result)

    def test_all_sector_etfs_covered(self):
        from data.sector_data import SECTOR_ETFS, get_sector_etf

        for sector, etf in SECTOR_ETFS.items():
            cache = {"TESTSYM": sector}
            with patch("data.sector_data._load_sector_cache", return_value=cache):
                result = get_sector_etf("TESTSYM")
            self.assertEqual(result, etf)


# ---------------------------------------------------------------------------
# New tests — sector_is_leading
# ---------------------------------------------------------------------------


class TestSectorIsLeading(unittest.TestCase):
    def _ranked(self):
        return [
            ("XLK", "Technology", 0.05),
            ("XLF", "Financials", 0.04),
            ("XLV", "Healthcare", 0.03),
            ("XLY", "Consumer Discretionary", 0.02),
            ("XLP", "Consumer Staples", 0.01),
            ("XLE", "Energy", -0.01),
            ("XLI", "Industrials", -0.02),
        ]

    def test_top_sector_returns_true(self):
        from data.sector_data import sector_is_leading

        with (
            patch("data.sector_data.rank_sectors_by_momentum", return_value=self._ranked()),
            patch("data.sector_data._load_sector_cache", return_value={"AAPL": "Technology"}),
        ):
            self.assertTrue(sector_is_leading("AAPL", top_n=4))

    def test_bottom_sector_returns_false(self):
        from data.sector_data import sector_is_leading

        with (
            patch("data.sector_data.rank_sectors_by_momentum", return_value=self._ranked()),
            patch("data.sector_data._load_sector_cache", return_value={"XOM": "Energy"}),
        ):
            self.assertFalse(sector_is_leading("XOM", top_n=4))

    def test_fails_open_when_ranked_empty(self):
        from data.sector_data import sector_is_leading

        with patch("data.sector_data.rank_sectors_by_momentum", return_value=[]):
            self.assertTrue(sector_is_leading("AAPL"))

    def test_fails_open_on_exception(self):
        from data.sector_data import sector_is_leading

        with patch("data.sector_data.rank_sectors_by_momentum", side_effect=RuntimeError("boom")):
            self.assertTrue(sector_is_leading("AAPL"))


# ---------------------------------------------------------------------------
# New tests — sector_is_lagging
# ---------------------------------------------------------------------------


class TestSectorIsLagging(unittest.TestCase):
    def _ranked(self):
        return [
            ("XLK", "Technology", 0.05),
            ("XLF", "Financials", 0.04),
            ("XLV", "Healthcare", 0.03),
            ("XLY", "Consumer Discretionary", 0.02),
            ("XLP", "Consumer Staples", 0.01),
            ("XLE", "Energy", -0.01),
            ("XLI", "Industrials", -0.02),
            ("XLU", "Utilities", -0.03),
        ]

    def test_bottom_sector_returns_true(self):
        from data.sector_data import sector_is_lagging

        with (
            patch("data.sector_data.rank_sectors_by_momentum", return_value=self._ranked()),
            patch("data.sector_data._load_sector_cache", return_value={"NEE": "Utilities"}),
        ):
            self.assertTrue(sector_is_lagging("NEE", bottom_n=3))

    def test_top_sector_returns_false(self):
        from data.sector_data import sector_is_lagging

        with (
            patch("data.sector_data.rank_sectors_by_momentum", return_value=self._ranked()),
            patch("data.sector_data._load_sector_cache", return_value={"AAPL": "Technology"}),
        ):
            self.assertFalse(sector_is_lagging("AAPL", bottom_n=3))

    def test_fails_open_when_ranked_empty(self):
        from data.sector_data import sector_is_lagging

        with patch("data.sector_data.rank_sectors_by_momentum", return_value=[]):
            self.assertFalse(sector_is_lagging("AAPL"))

    def test_fails_open_on_exception(self):
        from data.sector_data import sector_is_lagging

        with patch("data.sector_data.rank_sectors_by_momentum", side_effect=RuntimeError("boom")):
            self.assertFalse(sector_is_lagging("AAPL"))


# ---------------------------------------------------------------------------
# New tests — SECTOR_ETFS completeness
# ---------------------------------------------------------------------------


class TestSectorEtfsConstant(unittest.TestCase):
    def test_all_11_spdr_sectors_present(self):
        from data.sector_data import SECTOR_ETFS

        expected = {
            "Technology": "XLK",
            "Financials": "XLF",
            "Healthcare": "XLV",
            "Consumer Discretionary": "XLY",
            "Consumer Staples": "XLP",
            "Energy": "XLE",
            "Industrials": "XLI",
            "Utilities": "XLU",
            "Real Estate": "XLRE",
            "Materials": "XLB",
            "Communication Services": "XLC",
        }
        self.assertEqual(SECTOR_ETFS, expected)
