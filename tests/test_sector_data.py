"""Tests for data/sector_data.py — sector lookup and concentration checks."""
import unittest
from unittest.mock import patch


class TestGetSector(unittest.TestCase):

    def test_known_symbol_returns_correct_sector(self):
        from data.sector_data import get_sector
        self.assertEqual(get_sector("AAPL"), "Technology")
        self.assertEqual(get_sector("JPM"), "Financials")
        self.assertEqual(get_sector("XOM"), "Energy")
        self.assertEqual(get_sector("SPY"), "ETF")
        self.assertEqual(get_sector("LMT"), "Industrials")
        self.assertEqual(get_sector("PFE"), "Healthcare")

    def test_unknown_symbol_returns_unknown(self):
        from data.sector_data import get_sector
        self.assertEqual(get_sector("ZZZZ"), "Unknown")

    def test_all_mapped_symbols_have_non_empty_sector(self):
        from data.sector_data import get_sector, SECTOR_MAP
        for sym in SECTOR_MAP:
            self.assertNotEqual(get_sector(sym), "Unknown")
            self.assertNotEqual(get_sector(sym), "")


class TestCheckSectorConcentration(unittest.TestCase):

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
        self.assertEqual(len(result), 2)   # 1 excess per sector

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
            "Technology": 3.2, "Financials": 2.1, "Energy": 1.5,
            "Healthcare": -0.3, "Industrials": -1.0,
        }
        with patch("data.sector_data.get_sector_performance", return_value=mock_perf):
            result = get_leading_sectors(top_n=3)
        self.assertEqual(result, ["Technology", "Financials", "Energy"])

    def test_returns_empty_when_no_performance_data(self):
        from data.sector_data import get_leading_sectors
        with patch("data.sector_data.get_sector_performance", return_value={}):
            result = get_leading_sectors()
        self.assertEqual(result, [])
