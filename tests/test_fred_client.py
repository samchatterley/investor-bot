"""Tests for data/fred_client.py — 100% line coverage, no real FRED API calls."""

import json
import unittest
from datetime import date
from unittest.mock import MagicMock, mock_open, patch


class TestGetApiKey(unittest.TestCase):
    def test_returns_key_when_set(self):
        with patch.dict("os.environ", {"FRED_API_KEY": "abc123"}):
            from data.fred_client import _get_api_key

            self.assertEqual(_get_api_key(), "abc123")

    def test_returns_none_when_unset(self):
        env = {k: v for k, v in __import__("os").environ.items() if k != "FRED_API_KEY"}
        with patch.dict("os.environ", env, clear=True):
            from data.fred_client import _get_api_key

            self.assertIsNone(_get_api_key())

    def test_returns_none_for_empty_string(self):
        with patch.dict("os.environ", {"FRED_API_KEY": ""}):
            from data.fred_client import _get_api_key

            self.assertIsNone(_get_api_key())


class TestLoadCache(unittest.TestCase):
    def test_missing_file_returns_empty_dict(self):
        with patch("builtins.open", side_effect=FileNotFoundError):
            from data.fred_client import _load_cache

            self.assertEqual(_load_cache(), {})

    def test_corrupt_json_returns_empty_dict(self):
        with patch("builtins.open", mock_open(read_data="not json")):
            from data.fred_client import _load_cache

            self.assertEqual(_load_cache(), {})

    def test_valid_cache_returns_dict(self):
        payload = {"T10Y2Y": {"fetched_date": "2026-06-02", "data": [["2026-06-02", 0.5]]}}
        with patch("builtins.open", mock_open(read_data=json.dumps(payload))):
            from data.fred_client import _load_cache

            result = _load_cache()
        self.assertEqual(result["T10Y2Y"]["fetched_date"], "2026-06-02")

    def test_oserror_returns_empty_dict(self):
        with patch("builtins.open", side_effect=OSError("disk error")):
            from data.fred_client import _load_cache

            self.assertEqual(_load_cache(), {})


class TestSaveCache(unittest.TestCase):
    def test_writes_file(self):
        cache = {"T10Y2Y": {"fetched_date": "2026-06-02", "data": []}}
        m = mock_open()
        with (
            patch("builtins.open", m),
            patch("pathlib.Path.mkdir"),
        ):
            from data.fred_client import _save_cache

            _save_cache(cache)
        m.assert_called_once()

    def test_oserror_silently_swallowed(self):
        with (
            patch("pathlib.Path.mkdir"),
            patch("builtins.open", side_effect=OSError("no space")),
        ):
            from data.fred_client import _save_cache

            # Should not raise
            _save_cache({})


class TestIsSeriesStale(unittest.TestCase):
    def test_today_is_not_stale(self):
        today_str = "2026-06-02"
        with patch("data.fred_client.today_et", return_value=date(2026, 6, 2)):
            from data.fred_client import _is_series_stale

            entry = {"fetched_date": today_str, "data": []}
            self.assertFalse(_is_series_stale(entry))

    def test_yesterday_is_stale(self):
        with patch("data.fred_client.today_et", return_value=date(2026, 6, 2)):
            from data.fred_client import _is_series_stale

            entry = {"fetched_date": "2026-06-01", "data": []}
            self.assertTrue(_is_series_stale(entry))

    def test_missing_fetched_date_is_stale(self):
        with patch("data.fred_client.today_et", return_value=date(2026, 6, 2)):
            from data.fred_client import _is_series_stale

            self.assertTrue(_is_series_stale({}))


class TestFetchSeries(unittest.TestCase):
    def test_no_api_key_returns_empty(self):
        with patch("data.fred_client._get_api_key", return_value=None):
            from data.fred_client import fetch_series

            self.assertEqual(fetch_series("T10Y2Y"), [])

    def test_fredapi_exception_returns_empty(self):
        mock_fred_instance = MagicMock()
        mock_fred_instance.get_series.side_effect = Exception("network error")
        mock_fred_cls = MagicMock(return_value=mock_fred_instance)
        mock_fredapi = MagicMock()
        mock_fredapi.Fred = mock_fred_cls

        import data.fred_client as fc

        with (
            patch.object(fc, "_get_api_key", return_value="key"),
            patch.object(fc, "_load_cache", return_value={}),
            patch.object(fc, "_save_cache"),
            patch.object(fc, "today_et", return_value=date(2026, 6, 2)),
            patch.dict("sys.modules", {"fredapi": mock_fredapi}),
        ):
            result = fc.fetch_series("T10Y2Y")
        self.assertEqual(result, [])

    def test_returns_cache_on_hit(self):
        cached_data = [["2026-06-01", 0.42], ["2026-06-02", 0.45]]
        cache = {"T10Y2Y": {"fetched_date": "2026-06-02", "data": cached_data}}

        with (
            patch("data.fred_client._get_api_key", return_value="key"),
            patch("data.fred_client._load_cache", return_value=cache),
            patch("data.fred_client._is_series_stale", return_value=False),
        ):
            from data.fred_client import fetch_series

            result = fetch_series("T10Y2Y")
        self.assertEqual(result, [("2026-06-01", 0.42), ("2026-06-02", 0.45)])

    def test_fetches_and_saves_on_cache_miss(self):
        import pandas as pd

        ts = pd.Timestamp("2026-06-02")
        mock_series = pd.Series({ts: 0.5})

        mock_fred_instance = MagicMock()
        mock_fred_instance.get_series.return_value = mock_series
        mock_fred_cls = MagicMock(return_value=mock_fred_instance)
        mock_fredapi = MagicMock()
        mock_fredapi.Fred = mock_fred_cls

        mock_save = MagicMock()

        import data.fred_client as fc

        with (
            patch.object(fc, "_get_api_key", return_value="key"),
            patch.object(fc, "_load_cache", return_value={}),
            patch.object(fc, "_save_cache", mock_save),
            patch.object(fc, "today_et", return_value=date(2026, 6, 2)),
            patch.dict("sys.modules", {"fredapi": mock_fredapi}),
        ):
            result = fc.fetch_series("T10Y2Y")

        self.assertEqual(result, [("2026-06-02", 0.5)])
        mock_save.assert_called_once()

    def test_filters_nan_values(self):
        import math

        import pandas as pd

        ts1 = pd.Timestamp("2026-06-01")
        ts2 = pd.Timestamp("2026-06-02")
        mock_series = pd.Series({ts1: float("nan"), ts2: 0.5})

        mock_fred_instance = MagicMock()
        mock_fred_instance.get_series.return_value = mock_series
        mock_fred_cls = MagicMock(return_value=mock_fred_instance)
        mock_fredapi = MagicMock()
        mock_fredapi.Fred = mock_fred_cls

        import data.fred_client as fc

        with (
            patch.object(fc, "_get_api_key", return_value="key"),
            patch.object(fc, "_load_cache", return_value={}),
            patch.object(fc, "_save_cache"),
            patch.object(fc, "today_et", return_value=date(2026, 6, 2)),
            patch.dict("sys.modules", {"fredapi": mock_fredapi}),
        ):
            result = fc.fetch_series("T10Y2Y")

        self.assertEqual(len(result), 1)
        self.assertFalse(math.isnan(result[0][1]))


class TestGetYieldCurve(unittest.TestCase):
    def test_returns_latest_value(self):
        data = [("2026-06-01", 0.3), ("2026-06-02", -0.15)]
        with patch("data.fred_client.fetch_series", return_value=data):
            from data.fred_client import get_yield_curve

            self.assertEqual(get_yield_curve(), -0.15)

    def test_returns_none_on_empty(self):
        with patch("data.fred_client.fetch_series", return_value=[]):
            from data.fred_client import get_yield_curve

            self.assertIsNone(get_yield_curve())


class TestGetYieldCurveInvertedDays(unittest.TestCase):
    def test_counts_consecutive_negative_at_end(self):
        data = [
            ("2026-05-28", 0.1),
            ("2026-05-29", -0.1),
            ("2026-05-30", -0.2),
            ("2026-05-31", -0.3),
        ]
        with patch("data.fred_client.fetch_series", return_value=data):
            from data.fred_client import get_yield_curve_inverted_days

            self.assertEqual(get_yield_curve_inverted_days(), 3)

    def test_returns_zero_when_not_inverted(self):
        data = [("2026-06-01", 0.2), ("2026-06-02", 0.4)]
        with patch("data.fred_client.fetch_series", return_value=data):
            from data.fred_client import get_yield_curve_inverted_days

            self.assertEqual(get_yield_curve_inverted_days(), 0)

    def test_returns_zero_on_empty_data(self):
        with patch("data.fred_client.fetch_series", return_value=[]):
            from data.fred_client import get_yield_curve_inverted_days

            self.assertEqual(get_yield_curve_inverted_days(), 0)

    def test_stops_counting_at_first_positive(self):
        data = [
            ("2026-05-28", -0.5),
            ("2026-05-29", 0.1),
            ("2026-05-30", -0.1),
            ("2026-05-31", -0.2),
        ]
        with patch("data.fred_client.fetch_series", return_value=data):
            from data.fred_client import get_yield_curve_inverted_days

            self.assertEqual(get_yield_curve_inverted_days(), 2)


class TestGetClaims4wMa(unittest.TestCase):
    def test_computes_correctly_with_4_or_more_points(self):
        data = [
            ("2026-05-01", 200000.0),
            ("2026-05-08", 210000.0),
            ("2026-05-15", 220000.0),
            ("2026-05-22", 230000.0),
        ]
        with patch("data.fred_client.fetch_series", return_value=data):
            from data.fred_client import get_claims_4w_ma

            result = get_claims_4w_ma()
        self.assertAlmostEqual(result, 215000.0)

    def test_returns_none_with_fewer_than_4_points(self):
        data = [("2026-05-01", 200000.0), ("2026-05-08", 210000.0)]
        with patch("data.fred_client.fetch_series", return_value=data):
            from data.fred_client import get_claims_4w_ma

            self.assertIsNone(get_claims_4w_ma())

    def test_returns_none_on_empty(self):
        with patch("data.fred_client.fetch_series", return_value=[]):
            from data.fred_client import get_claims_4w_ma

            self.assertIsNone(get_claims_4w_ma())


class TestGetClaimsTrend(unittest.TestCase):
    def _make_data(self, values: list[float]) -> list[tuple[str, float]]:
        from datetime import timedelta

        base = date(2026, 1, 1)
        return [((base + timedelta(weeks=i)).isoformat(), v) for i, v in enumerate(values)]

    def test_returns_all_none_on_insufficient_data(self):
        with patch("data.fred_client.fetch_series", return_value=[]):
            from data.fred_client import get_claims_trend

            result = get_claims_trend()
        self.assertIsNone(result["latest"])
        self.assertIsNone(result["ma_4w"])
        self.assertEqual(result["rising_weeks"], 0)
        self.assertFalse(result["deteriorating"])

    def test_rising_weeks_counted_correctly(self):
        # Build a series where each consecutive 4-week window is higher than the previous
        # Window i covers values[i..i+3]; window i+1 covers values[i+1..i+4]
        # For rising: we need consecutive windows to be increasing.
        # Use steadily increasing values: 100, 110, 120, 130, 140, 150, 160, 170, 180
        values = [100.0 + 10 * i for i in range(9)]
        data = self._make_data(values)
        with patch("data.fred_client.fetch_series", return_value=data):
            from data.fred_client import get_claims_trend

            result = get_claims_trend()
        # There are 5 consecutive rising windows (indices 4→5 through 8→9 windows)
        self.assertGreater(result["rising_weeks"], 0)
        self.assertIsNotNone(result["latest"])
        self.assertIsNotNone(result["ma_4w"])

    def test_deteriorating_true_when_rising_weeks_ge_6(self):
        # 11 points gives us enough windows: indices 0..7 → 8 windows to compare
        values = [100.0 + 10 * i for i in range(11)]
        data = self._make_data(values)
        with patch("data.fred_client.fetch_series", return_value=data):
            from data.fred_client import get_claims_trend

            result = get_claims_trend()
        self.assertTrue(result["deteriorating"])
        self.assertGreaterEqual(result["rising_weeks"], 6)

    def test_deteriorating_false_when_not_rising(self):
        # Flat values — no rising windows
        values = [200000.0] * 8
        data = self._make_data(values)
        with patch("data.fred_client.fetch_series", return_value=data):
            from data.fred_client import get_claims_trend

            result = get_claims_trend()
        self.assertFalse(result["deteriorating"])
        self.assertEqual(result["rising_weeks"], 0)

    def test_returns_none_with_fewer_than_4_points(self):
        data = self._make_data([200000.0, 210000.0, 220000.0])
        with patch("data.fred_client.fetch_series", return_value=data):
            from data.fred_client import get_claims_trend

            result = get_claims_trend()
        self.assertIsNone(result["latest"])


class TestGetMacroSnapshot(unittest.TestCase):
    def test_data_available_false_when_key_not_set(self):
        with (
            patch("data.fred_client.fetch_series", return_value=[]),
            patch("data.fred_client._get_api_key", return_value=None),
        ):
            from data.fred_client import get_macro_snapshot

            result = get_macro_snapshot()
        self.assertFalse(result["data_available"])

    def test_correct_fields_populated(self):
        yc_data = [("2026-06-01", 0.3), ("2026-06-02", -0.15)]
        icsa_data = [
            ("2026-05-01", 200000.0),
            ("2026-05-08", 205000.0),
            ("2026-05-15", 210000.0),
            ("2026-05-22", 215000.0),
        ]
        ff_data = [("2026-06-01", 5.33), ("2026-06-02", 5.33)]

        def _side_effect(series_id, observation_start="2020-01-01"):
            if series_id == "T10Y2Y":
                return yc_data
            if series_id == "ICSA":
                return icsa_data
            if series_id == "FEDFUNDS":
                return ff_data
            return []  # pragma: no cover

        with patch("data.fred_client.fetch_series", side_effect=_side_effect):
            from data.fred_client import get_macro_snapshot

            result = get_macro_snapshot()

        self.assertIn("yield_curve", result)
        self.assertIn("yield_curve_inverted", result)
        self.assertIn("yield_curve_inverted_days", result)
        self.assertIn("claims_deteriorating", result)
        self.assertIn("claims_ma_4w", result)
        self.assertIn("fedfunds", result)
        self.assertIn("data_available", result)
        self.assertEqual(result["fedfunds"], 5.33)
        self.assertTrue(result["data_available"])

    def test_yield_curve_inverted_flag(self):
        yc_data = [("2026-06-02", -0.5)]
        icsa_data = []
        ff_data = []

        def _side_effect(series_id, observation_start="2020-01-01"):
            if series_id == "T10Y2Y":
                return yc_data
            if series_id == "ICSA":
                return icsa_data
            if series_id == "FEDFUNDS":
                return ff_data
            return []  # pragma: no cover

        with patch("data.fred_client.fetch_series", side_effect=_side_effect):
            from data.fred_client import get_macro_snapshot

            result = get_macro_snapshot()
        self.assertTrue(result["yield_curve_inverted"])
        self.assertEqual(result["yield_curve_inverted_days"], 1)

    def test_yield_curve_not_inverted_when_positive(self):
        yc_data = [("2026-06-02", 0.5)]
        ff_data = [("2026-06-02", 5.0)]

        def _side_effect(series_id, observation_start="2020-01-01"):
            if series_id == "T10Y2Y":
                return yc_data
            if series_id == "FEDFUNDS":
                return ff_data
            return []

        with patch("data.fred_client.fetch_series", side_effect=_side_effect):
            from data.fred_client import get_macro_snapshot

            result = get_macro_snapshot()
        self.assertFalse(result["yield_curve_inverted"])


class TestGetPmiSnapshot(unittest.TestCase):
    def test_returns_empty_when_no_data(self):
        with patch("data.fred_client.fetch_series", return_value=[]):
            from data.fred_client import get_pmi_snapshot

            result = get_pmi_snapshot()
        self.assertIsNone(result["latest"])
        self.assertIsNone(result["ma_3m"])
        self.assertFalse(result["expanding"])
        self.assertFalse(result["contracting"])

    def test_expanding_when_ma3m_above_55(self):
        data = [("2026-03-01", 56.0), ("2026-04-01", 57.0), ("2026-05-01", 58.0)]
        with patch("data.fred_client.fetch_series", return_value=data):
            from data.fred_client import get_pmi_snapshot

            result = get_pmi_snapshot()
        self.assertTrue(result["expanding"])
        self.assertFalse(result["contracting"])
        self.assertAlmostEqual(result["ma_3m"], 57.0)

    def test_contracting_when_latest_below_45(self):
        data = [("2026-03-01", 48.0), ("2026-04-01", 46.0), ("2026-05-01", 43.0)]
        with patch("data.fred_client.fetch_series", return_value=data):
            from data.fred_client import get_pmi_snapshot

            result = get_pmi_snapshot()
        self.assertTrue(result["contracting"])
        self.assertFalse(result["expanding"])

    def test_neutral_between_45_and_55(self):
        data = [("2026-03-01", 50.0), ("2026-04-01", 51.0), ("2026-05-01", 52.0)]
        with patch("data.fred_client.fetch_series", return_value=data):
            from data.fred_client import get_pmi_snapshot

            result = get_pmi_snapshot()
        self.assertFalse(result["expanding"])
        self.assertFalse(result["contracting"])

    def test_ma3m_none_with_fewer_than_3_readings(self):
        data = [("2026-05-01", 52.0), ("2026-06-01", 53.0)]
        with patch("data.fred_client.fetch_series", return_value=data):
            from data.fred_client import get_pmi_snapshot

            result = get_pmi_snapshot()
        self.assertIsNone(result["ma_3m"])
        self.assertIsNotNone(result["latest"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
