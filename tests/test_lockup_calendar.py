"""Tests for data/lockup_calendar.py."""

import json
import unittest
from datetime import date, timedelta
from unittest.mock import MagicMock, mock_open, patch

import pandas as pd

_TODAY = date(2026, 6, 12)


class TestLoadSaveCacheLockup(unittest.TestCase):
    def test_load_returns_empty_on_missing_file(self):
        with patch("builtins.open", side_effect=FileNotFoundError):
            from data.lockup_calendar import _load_ipo_cache

            result = _load_ipo_cache()
        self.assertEqual(result, {})

    def test_load_returns_empty_on_json_error(self):
        with (
            patch("builtins.open", mock_open(read_data="not json")),
            patch(
                "data.lockup_calendar.json.load",
                side_effect=json.JSONDecodeError("err", "", 0),
            ),
        ):
            from data.lockup_calendar import _load_ipo_cache

            result = _load_ipo_cache()
        self.assertEqual(result, {})

    def test_save_writes_json_on_success(self):
        m = mock_open()
        with (
            patch("data.lockup_calendar.os.makedirs"),
            patch("builtins.open", m),
            patch("data.lockup_calendar.json.dump") as mock_dump,
        ):
            from data.lockup_calendar import _save_ipo_cache

            _save_ipo_cache({"AAPL": "2026-01-01"})
        mock_dump.assert_called_once()

    def test_save_logs_warning_on_os_error(self):
        with (
            patch("data.lockup_calendar.os.makedirs"),
            patch("builtins.open", side_effect=OSError("disk full")),
        ):
            from data.lockup_calendar import _save_ipo_cache

            _save_ipo_cache({"AAPL": "2026-01-01"})  # should not raise


class TestDetectIpoDate(unittest.TestCase):
    def test_uses_ipo_expected_date_from_info(self):
        recent_ipo = (_TODAY - timedelta(days=30)).isoformat()
        mock_ticker = MagicMock()
        mock_ticker.info = {"ipoExpectedDate": recent_ipo}
        with (
            patch("data.lockup_calendar.yf.Ticker", return_value=mock_ticker),
            patch("data.lockup_calendar.today_et", return_value=_TODAY),
        ):
            from data.lockup_calendar import _detect_ipo_date

            result = _detect_ipo_date("NEWCO")
        self.assertEqual(result, recent_ipo)

    def test_uses_epoch_timestamp_from_info(self):
        import calendar

        recent = _TODAY - timedelta(days=60)
        epoch = int(calendar.timegm(recent.timetuple()))
        mock_ticker = MagicMock()
        mock_ticker.info = {"firstTradeDateEpochUtc": epoch}
        with (
            patch("data.lockup_calendar.yf.Ticker", return_value=mock_ticker),
            patch("data.lockup_calendar.today_et", return_value=_TODAY),
        ):
            from data.lockup_calendar import _detect_ipo_date

            result = _detect_ipo_date("NEWCO")
        self.assertEqual(result, recent.isoformat())

    def test_returns_none_for_old_ipo_from_info(self):
        old_ipo = (_TODAY - timedelta(days=600)).isoformat()
        mock_ticker = MagicMock()
        mock_ticker.info = {"ipoExpectedDate": old_ipo}
        with (
            patch("data.lockup_calendar.yf.Ticker", return_value=mock_ticker),
            patch("data.lockup_calendar.today_et", return_value=_TODAY),
        ):
            from data.lockup_calendar import _detect_ipo_date

            result = _detect_ipo_date("OLD")
        self.assertIsNone(result)

    def test_falls_back_to_price_history(self):
        recent = _TODAY - timedelta(days=60)
        mock_hist = pd.DataFrame(
            {"Close": [10.0]},
            index=pd.DatetimeIndex([pd.Timestamp(recent)]),
        )
        mock_ticker = MagicMock()
        mock_ticker.info = {}
        mock_ticker.history.return_value = mock_hist
        with (
            patch("data.lockup_calendar.yf.Ticker", return_value=mock_ticker),
            patch("data.lockup_calendar.today_et", return_value=_TODAY),
        ):
            from data.lockup_calendar import _detect_ipo_date

            result = _detect_ipo_date("NEWCO")
        self.assertEqual(result, recent.isoformat())

    def test_returns_none_when_history_empty(self):
        mock_ticker = MagicMock()
        mock_ticker.info = {}
        mock_ticker.history.return_value = pd.DataFrame()
        with (
            patch("data.lockup_calendar.yf.Ticker", return_value=mock_ticker),
            patch("data.lockup_calendar.today_et", return_value=_TODAY),
        ):
            from data.lockup_calendar import _detect_ipo_date

            result = _detect_ipo_date("NODATA")
        self.assertIsNone(result)

    def test_returns_none_when_history_too_old(self):
        old_date = _TODAY - timedelta(days=700)
        mock_hist = pd.DataFrame(
            {"Close": [10.0]},
            index=pd.DatetimeIndex([pd.Timestamp(old_date)]),
        )
        mock_ticker = MagicMock()
        mock_ticker.info = {}
        mock_ticker.history.return_value = mock_hist
        with (
            patch("data.lockup_calendar.yf.Ticker", return_value=mock_ticker),
            patch("data.lockup_calendar.today_et", return_value=_TODAY),
        ):
            from data.lockup_calendar import _detect_ipo_date

            result = _detect_ipo_date("OLD")
        self.assertIsNone(result)

    def test_returns_none_on_exception(self):
        with (
            patch("data.lockup_calendar.yf.Ticker", side_effect=RuntimeError("bad")),
            patch("data.lockup_calendar.today_et", return_value=_TODAY),
        ):
            from data.lockup_calendar import _detect_ipo_date

            result = _detect_ipo_date("ERR")
        self.assertIsNone(result)

    def test_invalid_ipo_date_string_falls_back_to_history(self):
        """Invalid ipoExpectedDate string falls through to price history fallback."""
        recent = _TODAY - timedelta(days=60)
        mock_hist = pd.DataFrame(
            {"Close": [10.0]},
            index=pd.DatetimeIndex([pd.Timestamp(recent)]),
        )
        mock_ticker = MagicMock()
        mock_ticker.info = {"ipoExpectedDate": "not-a-date"}
        mock_ticker.history.return_value = mock_hist
        with (
            patch("data.lockup_calendar.yf.Ticker", return_value=mock_ticker),
            patch("data.lockup_calendar.today_et", return_value=_TODAY),
        ):
            from data.lockup_calendar import _detect_ipo_date

            result = _detect_ipo_date("NEWCO")
        # Falls back to price history
        self.assertEqual(result, recent.isoformat())


class TestRefreshIpoDates(unittest.TestCase):
    def test_adds_new_symbols_and_saves(self):
        recent_ipo = (_TODAY - timedelta(days=30)).isoformat()
        with (
            patch("data.lockup_calendar.today_et", return_value=_TODAY),
            patch("data.lockup_calendar._load_ipo_cache", return_value={}),
            patch("data.lockup_calendar._detect_ipo_date", return_value=recent_ipo),
            patch("data.lockup_calendar._save_ipo_cache") as mock_save,
        ):
            from data.lockup_calendar import refresh_ipo_dates

            added = refresh_ipo_dates(["NEWCO"])
        self.assertEqual(added, 1)
        mock_save.assert_called_once()

    def test_skips_already_cached_symbols(self):
        recent_ipo = (_TODAY - timedelta(days=30)).isoformat()
        existing_cache = {"AAPL": recent_ipo}
        with (
            patch("data.lockup_calendar.today_et", return_value=_TODAY),
            patch("data.lockup_calendar._load_ipo_cache", return_value=existing_cache),
            patch("data.lockup_calendar._detect_ipo_date") as mock_detect,
            patch("data.lockup_calendar._save_ipo_cache"),
        ):
            from data.lockup_calendar import refresh_ipo_dates

            added = refresh_ipo_dates(["AAPL"])
        mock_detect.assert_not_called()
        self.assertEqual(added, 0)

    def test_prunes_stale_entries(self):
        stale_ipo = (_TODAY - timedelta(days=600)).isoformat()
        old_cache = {"STALE": stale_ipo}
        with (
            patch("data.lockup_calendar.today_et", return_value=_TODAY),
            patch("data.lockup_calendar._load_ipo_cache", return_value=old_cache),
            patch("data.lockup_calendar._detect_ipo_date", return_value=None),
            patch("data.lockup_calendar._save_ipo_cache") as mock_save,
        ):
            from data.lockup_calendar import refresh_ipo_dates

            refresh_ipo_dates(["NEWCO"])
        # Stale entry pruned, so save was called
        mock_save.assert_called_once()

    def test_no_save_when_nothing_changed(self):
        recent_ipo = (_TODAY - timedelta(days=30)).isoformat()
        existing_cache = {"AAPL": recent_ipo}
        with (
            patch("data.lockup_calendar.today_et", return_value=_TODAY),
            patch("data.lockup_calendar._load_ipo_cache", return_value=existing_cache),
            patch("data.lockup_calendar._save_ipo_cache") as mock_save,
        ):
            from data.lockup_calendar import refresh_ipo_dates

            added = refresh_ipo_dates(["AAPL"])
        mock_save.assert_not_called()
        self.assertEqual(added, 0)

    def test_uses_stock_universe_when_symbols_none(self):
        with (
            patch("data.lockup_calendar.STOCK_UNIVERSE", {"AAPL"}),
            patch("data.lockup_calendar.today_et", return_value=_TODAY),
            patch("data.lockup_calendar._load_ipo_cache", return_value={}),
            patch("data.lockup_calendar._detect_ipo_date", return_value=None),
            patch("data.lockup_calendar._save_ipo_cache"),
        ):
            from data.lockup_calendar import refresh_ipo_dates

            added = refresh_ipo_dates(None)
        self.assertEqual(added, 0)

    def test_detect_returns_none_means_not_added(self):
        with (
            patch("data.lockup_calendar.today_et", return_value=_TODAY),
            patch("data.lockup_calendar._load_ipo_cache", return_value={}),
            patch("data.lockup_calendar._detect_ipo_date", return_value=None),
            patch("data.lockup_calendar._save_ipo_cache"),
        ):
            from data.lockup_calendar import refresh_ipo_dates

            added = refresh_ipo_dates(["NOIPO"])
        self.assertEqual(added, 0)


class TestGetLockupExpiryFlags(unittest.TestCase):
    def _make_cache(self, ipo_days_ago: int) -> dict:
        ipo_date = (_TODAY - timedelta(days=ipo_days_ago)).isoformat()
        return {"NEWCO": ipo_date}

    def test_flags_stock_7_days_before_lockup(self):
        # Lockup = IPO + 180 days; 7 days before = IPO was 173 days ago
        cache = self._make_cache(ipo_days_ago=173)
        with (
            patch("data.lockup_calendar._load_ipo_cache", return_value=cache),
            patch("data.lockup_calendar.today_et", return_value=_TODAY),
        ):
            from data.lockup_calendar import get_lockup_expiry_flags

            result = get_lockup_expiry_flags(["NEWCO"])
        self.assertIn("NEWCO", result)
        self.assertTrue(result["NEWCO"]["lockup_expiry_soon"])
        self.assertEqual(result["NEWCO"]["days_to_lockup"], 7)

    def test_does_not_flag_outside_window(self):
        # Lockup is 20 days away — outside 5–10 day window
        cache = self._make_cache(ipo_days_ago=160)
        with (
            patch("data.lockup_calendar._load_ipo_cache", return_value=cache),
            patch("data.lockup_calendar.today_et", return_value=_TODAY),
        ):
            from data.lockup_calendar import get_lockup_expiry_flags

            result = get_lockup_expiry_flags(["NEWCO"])
        self.assertNotIn("NEWCO", result)

    def test_does_not_flag_passed_lockup(self):
        # Lockup already expired (IPO was 200 days ago)
        cache = self._make_cache(ipo_days_ago=200)
        with (
            patch("data.lockup_calendar._load_ipo_cache", return_value=cache),
            patch("data.lockup_calendar.today_et", return_value=_TODAY),
        ):
            from data.lockup_calendar import get_lockup_expiry_flags

            result = get_lockup_expiry_flags(["NEWCO"])
        self.assertNotIn("NEWCO", result)

    def test_skips_symbol_not_in_cache(self):
        with (
            patch("data.lockup_calendar._load_ipo_cache", return_value={}),
            patch("data.lockup_calendar.today_et", return_value=_TODAY),
        ):
            from data.lockup_calendar import get_lockup_expiry_flags

            result = get_lockup_expiry_flags(["UNKNOWN"])
        self.assertNotIn("UNKNOWN", result)

    def test_skips_invalid_date_string(self):
        with (
            patch("data.lockup_calendar._load_ipo_cache", return_value={"NEWCO": "bad-date"}),
            patch("data.lockup_calendar.today_et", return_value=_TODAY),
        ):
            from data.lockup_calendar import get_lockup_expiry_flags

            result = get_lockup_expiry_flags(["NEWCO"])
        self.assertNotIn("NEWCO", result)

    def test_flags_exactly_at_window_edge_5_days(self):
        # 5 days before lockup = IPO 175 days ago (boundary: _ALERT_WINDOW_LATE=5)
        cache = self._make_cache(ipo_days_ago=175)
        with (
            patch("data.lockup_calendar._load_ipo_cache", return_value=cache),
            patch("data.lockup_calendar.today_et", return_value=_TODAY),
        ):
            from data.lockup_calendar import get_lockup_expiry_flags

            result = get_lockup_expiry_flags(["NEWCO"])
        self.assertIn("NEWCO", result)

    def test_flags_exactly_at_window_edge_10_days(self):
        # 10 days before lockup = IPO 170 days ago (boundary: _ALERT_WINDOW_EARLY=10)
        cache = self._make_cache(ipo_days_ago=170)
        with (
            patch("data.lockup_calendar._load_ipo_cache", return_value=cache),
            patch("data.lockup_calendar.today_et", return_value=_TODAY),
        ):
            from data.lockup_calendar import get_lockup_expiry_flags

            result = get_lockup_expiry_flags(["NEWCO"])
        self.assertIn("NEWCO", result)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
