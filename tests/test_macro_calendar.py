import unittest
from datetime import date

from risk.macro_calendar import (
    CPI_RELEASE_DATES,
    FOMC_ANNOUNCEMENT_DATES,
    NFP_RELEASE_DATES,
    NYSE_HOLIDAYS,
    _next_trading_day,
    _third_friday,
    get_macro_risk,
    get_seasonal_context,
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

    def test_nfp_date_is_not_high_risk(self):
        # NFP lands before market open; reaction absorbed by our 10:00 ET buy window.
        nfp_day = next(iter(NFP_RELEASE_DATES))
        result = get_macro_risk(nfp_day)
        self.assertFalse(result["is_high_risk"])

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

    def test_fomc_date_returns_true_and_event_label(self):
        fomc_day = date(2026, 1, 28)  # a known FOMC announcement date
        result = get_macro_risk(fomc_day)
        self.assertTrue(result["is_high_risk"])
        self.assertIsNotNone(result["event"])
        self.assertIn("FOMC", result["event"])

    def test_defaults_to_today_when_no_date_passed(self):
        # Line 44: check_date is None → today_et() is called
        from unittest.mock import patch

        fixed_date = date(2026, 2, 10)  # a quiet day
        with patch("risk.macro_calendar.today_et", return_value=fixed_date):
            result = get_macro_risk()
        self.assertFalse(result["is_high_risk"])
        self.assertIsNone(result["event"])


class TestThirdFriday(unittest.TestCase):
    def test_jan_2026_third_friday(self):
        # Jan 2026: 1st=Thu, first Fri=2nd, 3rd Fri=16th
        self.assertEqual(_third_friday(2026, 1), date(2026, 1, 16))

    def test_jun_2026_third_friday(self):
        # Jun 2026: 1st=Mon, first Fri=5th, 3rd Fri=19th
        self.assertEqual(_third_friday(2026, 6), date(2026, 6, 19))

    def test_is_always_a_friday(self):
        for month in range(1, 13):
            self.assertEqual(_third_friday(2026, month).weekday(), 4)

    def test_dec_2026_third_friday(self):
        # Dec 2026: 1st=Tue, first Fri=4th, 3rd Fri=18th
        self.assertEqual(_third_friday(2026, 12), date(2026, 12, 18))


class TestNextTradingDay(unittest.TestCase):
    def test_friday_before_holiday_skips_to_tuesday(self):
        # Dec 24, 2026 (Thu) → next is Dec 25 (holiday Fri) → Dec 28 (Mon)
        # Actually Dec 25 2026 is Friday (a holiday), so next after Thursday Dec 24
        # would be Mon Dec 28
        result = _next_trading_day(date(2026, 12, 24))
        self.assertEqual(result, date(2026, 12, 28))

    def test_monday_through_friday_normal_week(self):
        # A normal Monday → next is Tuesday
        result = _next_trading_day(date(2026, 6, 8))  # Monday
        self.assertEqual(result, date(2026, 6, 9))

    def test_friday_normal_week_goes_to_monday(self):
        result = _next_trading_day(date(2026, 6, 12))  # Friday
        self.assertEqual(result, date(2026, 6, 15))

    def test_saturday_goes_to_monday(self):
        result = _next_trading_day(date(2026, 6, 13))  # Saturday
        self.assertEqual(result, date(2026, 6, 15))


class TestNYSEHolidays(unittest.TestCase):
    def test_christmas_2026_in_set(self):
        self.assertIn(date(2026, 12, 25), NYSE_HOLIDAYS)

    def test_labor_day_2026_in_set(self):
        self.assertIn(date(2026, 9, 7), NYSE_HOLIDAYS)

    def test_regular_trading_day_not_in_set(self):
        self.assertNotIn(date(2026, 6, 10), NYSE_HOLIDAYS)

    def test_all_holidays_are_weekdays(self):
        for h in NYSE_HOLIDAYS:
            self.assertLess(h.weekday(), 6, f"{h} is a Sunday")


class TestGetSeasonalContext(unittest.TestCase):
    def test_returns_all_required_keys(self):
        ctx = get_seasonal_context(date(2026, 6, 15))
        for key in (
            "turn_of_month",
            "opex_week",
            "post_opex",
            "halloween_bullish",
            "quarter_end_dressing",
            "pre_holiday",
        ):
            self.assertIn(key, ctx)

    def test_turn_of_month_last_day(self):
        # Jun 30, 2026 = last day of month → turn_of_month True
        self.assertTrue(get_seasonal_context(date(2026, 6, 30))["turn_of_month"])

    def test_turn_of_month_first_day_of_next(self):
        # Jul 1 = first day of month → turn_of_month True
        self.assertTrue(get_seasonal_context(date(2026, 7, 1))["turn_of_month"])

    def test_turn_of_month_mid_month_false(self):
        self.assertFalse(get_seasonal_context(date(2026, 6, 15))["turn_of_month"])

    def test_opex_week_on_third_friday(self):
        # Jun 2026 OPEX Friday = Jun 19
        self.assertTrue(get_seasonal_context(date(2026, 6, 19))["opex_week"])

    def test_opex_week_monday_before_third_friday(self):
        # Monday of OPEX week = Jun 15
        self.assertTrue(get_seasonal_context(date(2026, 6, 15))["opex_week"])

    def test_opex_week_false_on_saturday_after(self):
        # Saturday after OPEX = Jun 20
        self.assertFalse(get_seasonal_context(date(2026, 6, 20))["opex_week"])

    def test_post_opex_monday_after_opex(self):
        # Monday after Jun OPEX (Jun 19) = Jun 22
        self.assertTrue(get_seasonal_context(date(2026, 6, 22))["post_opex"])

    def test_post_opex_tuesday_after_opex(self):
        self.assertTrue(get_seasonal_context(date(2026, 6, 23))["post_opex"])

    def test_post_opex_wednesday_is_false(self):
        self.assertFalse(get_seasonal_context(date(2026, 6, 24))["post_opex"])

    def test_halloween_bullish_november(self):
        self.assertTrue(get_seasonal_context(date(2026, 11, 1))["halloween_bullish"])

    def test_halloween_bullish_april(self):
        self.assertTrue(get_seasonal_context(date(2026, 4, 1))["halloween_bullish"])

    def test_halloween_bearish_june(self):
        self.assertFalse(get_seasonal_context(date(2026, 6, 1))["halloween_bullish"])

    def test_quarter_end_dressing_last_week_march(self):
        # Mar 27, 2026 = 4 days before Mar 31 → quarter_end True
        self.assertTrue(get_seasonal_context(date(2026, 3, 27))["quarter_end_dressing"])

    def test_quarter_end_dressing_mid_june_false(self):
        self.assertFalse(get_seasonal_context(date(2026, 6, 1))["quarter_end_dressing"])

    def test_quarter_end_dressing_false_in_non_quarter_month(self):
        # May is not a quarter-end month
        self.assertFalse(get_seasonal_context(date(2026, 5, 30))["quarter_end_dressing"])

    def test_pre_holiday_day_before_christmas(self):
        # Dec 24 2026 (Thu): next weekday = Dec 25 (Fri, NYSE holiday) → True
        ctx = get_seasonal_context(date(2026, 12, 24))
        self.assertTrue(ctx["pre_holiday"])

    def test_pre_holiday_friday_before_labor_day_monday(self):
        # Sep 4 2026 (Fri): next weekday = Sep 7 (Mon, Labor Day) → True
        ctx = get_seasonal_context(date(2026, 9, 4))
        self.assertTrue(ctx["pre_holiday"])

    def test_pre_holiday_regular_day_false(self):
        ctx = get_seasonal_context(date(2026, 6, 10))
        self.assertFalse(ctx["pre_holiday"])

    def test_defaults_to_today_when_no_date(self):
        from unittest.mock import patch

        fixed = date(2026, 11, 15)  # mid-November, should be halloween_bullish
        with patch("risk.macro_calendar.today_et", return_value=fixed):
            ctx = get_seasonal_context()
        self.assertTrue(ctx["halloween_bullish"])
