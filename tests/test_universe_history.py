"""Tests for data/universe_history.py — get_universe_for_date, _FIRST_TRADEABLE."""

import unittest
from datetime import date
from unittest.mock import patch

import pandas as pd

from data.universe_history import (
    _FIRST_TRADEABLE,
    _build_sp500_membership,
    _fetch_sp500_changes,
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

    def test_sp500_filter_excludes_symbol_added_after_date(self):
        """Symbol with added=2021-01-01 should be excluded for dt=2020-01-01."""
        changes = [{"symbol": "NEWNEW", "added": date(2021, 1, 1), "removed": None}]
        with patch("data.universe_history._fetch_sp500_changes", return_value=changes):
            result = get_universe_for_date(
                date(2020, 1, 1), ["AAPL", "NEWNEW"], apply_sp500_filter=True
            )
        # AAPL has no membership record → passes; NEWNEW was added after dt → filtered out
        self.assertIn("AAPL", result)
        self.assertNotIn("NEWNEW", result)

    def test_sp500_filter_excludes_symbol_removed_before_date(self):
        """Symbol removed on 2018-06-01 should be excluded for dt=2019-01-01."""
        changes = [{"symbol": "OLDOLD", "added": date(2010, 1, 1), "removed": date(2018, 6, 1)}]
        with patch("data.universe_history._fetch_sp500_changes", return_value=changes):
            result = get_universe_for_date(
                date(2019, 1, 1), ["AAPL", "OLDOLD"], apply_sp500_filter=True
            )
        self.assertIn("AAPL", result)
        self.assertNotIn("OLDOLD", result)

    def test_sp500_filter_passes_symbol_within_membership_range(self):
        """Symbol added 2010-01-01, removed 2025-01-01 should be included for dt=2018-01-01."""
        changes = [{"symbol": "MID", "added": date(2010, 1, 1), "removed": date(2025, 1, 1)}]
        with patch("data.universe_history._fetch_sp500_changes", return_value=changes):
            result = get_universe_for_date(date(2018, 1, 1), ["MID"], apply_sp500_filter=True)
        self.assertIn("MID", result)

    def test_2025_includes_all_known_symbols(self):
        result = get_universe_for_date(date(2025, 1, 2), self._BASE)
        self.assertEqual(sorted(result), sorted(self._BASE))


class TestFetchSp500ChangesGracefulFallback(unittest.TestCase):
    def setUp(self):
        _fetch_sp500_changes.cache_clear()

    def tearDown(self):
        _fetch_sp500_changes.cache_clear()

    def test_network_error_returns_empty(self):
        with patch("data.universe_history.pd.read_html", side_effect=Exception("timeout")):
            result = _fetch_sp500_changes()
        self.assertEqual(result, [])

    def _make_changes_df(self):
        """Minimal 2-table read_html return with a well-formed changes table."""
        current_df = pd.DataFrame({"Symbol": ["AAPL", "MSFT"], "Security": ["Apple", "Microsoft"]})
        changes_df = pd.DataFrame(
            {
                "Date": ["2020-01-15", "2019-06-10"],
                "Added Ticker": ["AAPL", ""],
                "Removed Ticker": ["", "IBM"],
            }
        )
        return [current_df, changes_df]

    def test_returns_records_on_success(self):
        with patch("data.universe_history.pd.read_html", return_value=self._make_changes_df()):
            result = _fetch_sp500_changes()
        self.assertIsInstance(result, list)
        symbols = [r["symbol"] for r in result]
        self.assertIn("AAPL", symbols)
        self.assertIn("IBM", symbols)

    def test_returns_empty_when_fewer_than_2_tables(self):
        single_table = pd.DataFrame({"Symbol": ["AAPL"]})
        with patch("data.universe_history.pd.read_html", return_value=[single_table]):
            result = _fetch_sp500_changes()
        self.assertEqual(result, [])

    def test_returns_empty_when_table_missing_date_and_added_columns(self):
        current_df = pd.DataFrame({"Symbol": ["AAPL"]})
        bad_changes_df = pd.DataFrame({"foo": [1], "bar": [2]})
        with patch("data.universe_history.pd.read_html", return_value=[current_df, bad_changes_df]):
            result = _fetch_sp500_changes()
        self.assertEqual(result, [])

    def test_nan_date_row_is_skipped(self):
        """Line 99: row with NaN in the date column → continue."""
        current_df = pd.DataFrame({"Symbol": ["AAPL"]})
        changes_df = pd.DataFrame(
            {
                "Date": [float("nan"), "2020-01-15"],
                "Added Ticker": ["SKIP", "AAPL"],
                "Removed Ticker": ["", ""],
            }
        )
        with patch("data.universe_history.pd.read_html", return_value=[current_df, changes_df]):
            result = _fetch_sp500_changes()
        # "SKIP" row should be absent because its date was NaN
        symbols = [r["symbol"] for r in result]
        self.assertNotIn("SKIP", symbols)
        self.assertIn("AAPL", symbols)

    def test_unparseable_date_row_is_skipped(self):
        """Lines 102-103: date that pd.to_datetime raises on → continue."""
        current_df = pd.DataFrame({"Symbol": ["AAPL"]})
        # "not-a-date" is non-NaN so passes line 98, but pd.to_datetime raises
        changes_df = pd.DataFrame(
            {
                "Date": ["not-a-date", "2020-01-15"],
                "Added Ticker": ["BADROW", "AAPL"],
                "Removed Ticker": ["", ""],
            }
        )

        original_to_datetime = pd.to_datetime

        def _raise_for_bad(arg, **kwargs):
            if str(arg) == "not-a-date":
                raise ValueError("unconvertible date")
            return original_to_datetime(arg, **kwargs)

        with (
            patch("data.universe_history.pd.read_html", return_value=[current_df, changes_df]),
            patch("data.universe_history.pd.to_datetime", side_effect=_raise_for_bad),
        ):
            result = _fetch_sp500_changes()

        symbols = [r["symbol"] for r in result]
        self.assertNotIn("BADROW", symbols)
        self.assertIn("AAPL", symbols)


class TestBuildSp500Membership(unittest.TestCase):
    def test_single_add_event(self):
        changes = [{"symbol": "AAPL", "added": date(2010, 1, 1), "removed": None}]
        membership = _build_sp500_membership(changes)
        self.assertIn("AAPL", membership)
        self.assertEqual(membership["AAPL"][0], date(2010, 1, 1))
        self.assertIsNone(membership["AAPL"][1])

    def test_single_remove_event(self):
        changes = [{"symbol": "IBM", "added": None, "removed": date(2020, 6, 1)}]
        membership = _build_sp500_membership(changes)
        self.assertIn("IBM", membership)
        self.assertIsNone(membership["IBM"][0])
        self.assertEqual(membership["IBM"][1], date(2020, 6, 1))

    def test_multiple_events_same_symbol_uses_earliest_add_and_latest_remove(self):
        changes = [
            {"symbol": "XYZ", "added": date(2015, 1, 1), "removed": None},
            {"symbol": "XYZ", "added": date(2012, 3, 1), "removed": None},
            {"symbol": "XYZ", "added": None, "removed": date(2022, 5, 1)},
        ]
        membership = _build_sp500_membership(changes)
        self.assertEqual(membership["XYZ"][0], date(2012, 3, 1))
        self.assertEqual(membership["XYZ"][1], date(2022, 5, 1))

    def test_empty_changes_returns_empty(self):
        self.assertEqual(_build_sp500_membership([]), {})
