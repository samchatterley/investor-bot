import unittest
from datetime import date

from risk.macro_calendar import (
    CPI_RELEASE_DATES,
    FOMC_ANNOUNCEMENT_DATES,
    NFP_RELEASE_DATES,
    get_macro_risk,
)


class TestGetMacroRisk(unittest.TestCase):

    def test_fomc_date_is_high_risk(self):
        fomc_day = next(iter(FOMC_ANNOUNCEMENT_DATES))
        result = get_macro_risk(fomc_day)
        self.assertTrue(result["is_high_risk"])
        self.assertIn("FOMC", result["event"])

    def test_cpi_date_is_high_risk(self):
        cpi_day = next(iter(CPI_RELEASE_DATES))
        result = get_macro_risk(cpi_day)
        self.assertTrue(result["is_high_risk"])
        self.assertIn("CPI", result["event"])

    def test_nfp_date_is_high_risk(self):
        nfp_day = next(iter(NFP_RELEASE_DATES))
        result = get_macro_risk(nfp_day)
        self.assertTrue(result["is_high_risk"])

    def test_random_tuesday_is_not_high_risk(self):
        quiet = date(2026, 2, 10)
        result = get_macro_risk(quiet)
        self.assertFalse(result["is_high_risk"])
        self.assertIsNone(result["event"])

    def test_returns_dict_with_required_keys(self):
        result = get_macro_risk(date(2026, 3, 1))
        self.assertIn("is_high_risk", result)
        self.assertIn("event", result)

    def test_no_overlap_between_sets(self):
        fomc = FOMC_ANNOUNCEMENT_DATES
        cpi = CPI_RELEASE_DATES
        nfp = NFP_RELEASE_DATES
        self.assertEqual(len(fomc & cpi), 0)
        self.assertEqual(len(fomc & nfp), 0)
        self.assertEqual(len(cpi & nfp), 0)
