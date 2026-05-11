"""Tests for data/universe_history.py — get_universe_for_date, _FIRST_TRADEABLE."""

import unittest
from datetime import date
from unittest.mock import patch

from data.universe_history import (
    _FIRST_TRADEABLE,
    get_universe_for_date,
)


class TestFirstTradeableDates(unittest.TestCase):
    def test_uber_not_before_ipo(self):
        self.assertEqual(_FIRST_TRADEABLE["UBER"], date(2019, 5, 10))

    def test_snow_not_before_ipo(self):
        self.assertEqual(_FIRST_TRADEABLE["SNOW"], date(2020, 9, 16))

    def test_pltr_not_before_ipo(self):
        self.assertEqual(_FIRST_TRADEABLE["PLTR"], date(2020, 9, 30))

    def test_coin_not_before_ipo(self):
        self.assertEqual(_FIRST_TRADEABLE["COIN"], date(2021, 4, 14))

    def test_xyz_after_ticker_change(self):
        self.assertEqual(_FIRST_TRADEABLE["XYZ"], date(2023, 11, 1))


class TestGetUniverseForDate(unittest.TestCase):
    _BASE = ["AAPL", "MSFT", "SNOW", "PLTR", "UBER", "XYZ"]

    def test_2015_excludes_post_2015_ipos(self):
        result = get_universe_for_date(date(2015, 1, 2), self._BASE)
        self.assertIn("AAPL", result)
        self.assertIn("MSFT", result)
        self.assertNotIn("SNOW", result)
        self.assertNotIn("PLTR", result)
        self.assertNotIn("UBER", result)
        self.assertNotIn("XYZ", result)

    def test_2019_includes_uber_on_ipo_day(self):
        result = get_universe_for_date(date(2019, 5, 10), self._BASE)
        self.assertIn("UBER", result)

    def test_2019_excludes_uber_day_before_ipo(self):
        result = get_universe_for_date(date(2019, 5, 9), self._BASE)
        self.assertNotIn("UBER", result)

    def test_2021_includes_all_except_xyz_and_pcor(self):
        candidates = ["AAPL", "UBER", "SNOW", "PLTR", "COIN", "XYZ"]
        result = get_universe_for_date(date(2021, 6, 1), candidates)
        self.assertIn("AAPL", result)
        self.assertIn("UBER", result)
        self.assertIn("SNOW", result)
        self.assertIn("COIN", result)
        self.assertNotIn("XYZ", result)

    def test_preserves_input_order(self):
        candidates = ["MSFT", "AAPL", "GOOGL"]
        result = get_universe_for_date(date(2015, 1, 2), candidates)
        self.assertEqual(result, ["MSFT", "AAPL", "GOOGL"])

    def test_empty_candidates_returns_empty(self):
        result = get_universe_for_date(date(2020, 1, 2), [])
        self.assertEqual(result, [])

    def test_last_tradeable_excludes_after_date(self):
        # Temporarily patch _LAST_TRADEABLE to test the delisting gate
        from data import universe_history

        original = dict(universe_history._LAST_TRADEABLE)
        try:
            universe_history._LAST_TRADEABLE["FAKE"] = date(2020, 6, 1)
            result_before = get_universe_for_date(date(2020, 5, 31), ["FAKE"])
            result_after = get_universe_for_date(date(2020, 6, 2), ["FAKE"])
            self.assertIn("FAKE", result_before)
            self.assertNotIn("FAKE", result_after)
        finally:
            universe_history._LAST_TRADEABLE.clear()
            universe_history._LAST_TRADEABLE.update(original)

    def test_sp500_filter_disabled_by_default(self):
        # apply_sp500_filter=False (default) — should not call Wikipedia
        with patch("data.universe_history._fetch_sp500_changes") as mock_fetch:
            result = get_universe_for_date(date(2020, 1, 2), ["AAPL", "MSFT"])
            mock_fetch.assert_not_called()
        self.assertEqual(result, ["AAPL", "MSFT"])

    def test_sp500_filter_enabled_falls_back_on_empty(self):
        # When Wikipedia returns empty, no symbols should be filtered out
        with patch("data.universe_history._fetch_sp500_changes", return_value=[]):
            result = get_universe_for_date(
                date(2020, 1, 2), ["AAPL", "MSFT"], apply_sp500_filter=True
            )
        self.assertEqual(result, ["AAPL", "MSFT"])

    def test_2025_includes_all_known_symbols(self):
        result = get_universe_for_date(date(2025, 1, 2), self._BASE)
        self.assertEqual(sorted(result), sorted(self._BASE))


class TestFetchSp500ChangesGracefulFallback(unittest.TestCase):
    def test_network_error_returns_empty(self):
        from data.universe_history import _fetch_sp500_changes

        # Clear cache to force a fresh call
        _fetch_sp500_changes.cache_clear()
        with patch("data.universe_history.pd.read_html", side_effect=Exception("timeout")):
            result = _fetch_sp500_changes()
        self.assertEqual(result, [])
        _fetch_sp500_changes.cache_clear()
