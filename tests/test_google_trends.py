"""Tests for data/google_trends.py."""

import json
import unittest
from unittest.mock import MagicMock, mock_open, patch

import pandas as pd

_TODAY_STR = "2026-06-12"


def _today():
    from datetime import date

    return date.fromisoformat(_TODAY_STR)


class TestLoadSaveCache(unittest.TestCase):
    def test_load_returns_empty_on_missing_file(self):
        with patch("builtins.open", side_effect=FileNotFoundError):
            from data.google_trends import _load_cache

            result = _load_cache()
        self.assertEqual(result, {})

    def test_load_returns_empty_on_json_error(self):
        with (
            patch("builtins.open", mock_open(read_data="bad json")),
            patch(
                "data.google_trends.json.load",
                side_effect=json.JSONDecodeError("err", "", 0),
            ),
        ):
            from data.google_trends import _load_cache

            result = _load_cache()
        self.assertEqual(result, {})

    def test_save_writes_on_success(self):
        m = mock_open()
        with (
            patch("data.google_trends.os.makedirs"),
            patch("builtins.open", m),
            patch("data.google_trends.json.dump") as mock_dump,
        ):
            from data.google_trends import _save_cache

            _save_cache({_TODAY_STR: {"AAPL": False}})
        mock_dump.assert_called_once()

    def test_save_logs_warning_on_os_error(self):
        with (
            patch("data.google_trends.os.makedirs"),
            patch("builtins.open", side_effect=OSError("no space")),
        ):
            from data.google_trends import _save_cache

            _save_cache({_TODAY_STR: {}})  # should not raise


class TestLiveFetchTrends(unittest.TestCase):
    def test_returns_false_when_pytrends_not_installed(self):
        # Setting sys.modules entry to None makes Python raise ImportError on import
        import sys

        with patch.dict("sys.modules", {"pytrends": None, "pytrends.request": None}):
            if "data.google_trends" in sys.modules:
                del sys.modules["data.google_trends"]
            from data.google_trends import _live_fetch_trends

            result = _live_fetch_trends(["AAPL"])
        self.assertEqual(result, {"AAPL": False})

    def test_returns_false_when_trendreq_init_fails(self):
        mock_trendreq_cls = MagicMock(side_effect=RuntimeError("init failed"))
        with patch.dict(
            "sys.modules",
            {"pytrends": MagicMock(), "pytrends.request": MagicMock(TrendReq=mock_trendreq_cls)},
        ):
            import sys

            if "data.google_trends" in sys.modules:
                del sys.modules["data.google_trends"]
            from data.google_trends import _live_fetch_trends

            result = _live_fetch_trends(["AAPL"])
        self.assertEqual(result, {"AAPL": False})

    def test_spike_detected_when_current_exceeds_threshold(self):
        """Integration path: spike=True when current >= 1.5x baseline avg."""
        values = [20, 22, 21, 23, 19, 20, 22, 21, 20, 23, 22, 21, 40]  # last = spike
        sym = "AAPL"
        df = pd.DataFrame({sym: values, "isPartial": [False] * len(values)})
        mock_pt = MagicMock()
        mock_pt.interest_over_time.return_value = df

        mock_req_module = MagicMock()
        mock_req_module.TrendReq.return_value = mock_pt

        with patch.dict(
            "sys.modules", {"pytrends": MagicMock(), "pytrends.request": mock_req_module}
        ):
            import sys

            if "data.google_trends" in sys.modules:
                del sys.modules["data.google_trends"]
            import data.google_trends as gt

            with patch.object(gt, "time") as mock_time:
                mock_time.sleep = MagicMock()
                result = gt._live_fetch_trends([sym])
        self.assertTrue(result[sym])

    def test_no_spike_when_below_threshold(self):
        """No spike when current < 1.5x baseline."""
        values = [20, 22, 21, 23, 19, 20, 22, 21, 20, 23, 22, 21, 22]  # last = no spike
        sym = "AAPL"
        df = pd.DataFrame({sym: values, "isPartial": [False] * len(values)})
        mock_pt = MagicMock()
        mock_pt.interest_over_time.return_value = df

        mock_req_module = MagicMock()
        mock_req_module.TrendReq.return_value = mock_pt

        with patch.dict(
            "sys.modules", {"pytrends": MagicMock(), "pytrends.request": mock_req_module}
        ):
            import sys

            if "data.google_trends" in sys.modules:
                del sys.modules["data.google_trends"]
            import data.google_trends as gt

            with patch.object(gt, "time") as mock_time:
                mock_time.sleep = MagicMock()
                result = gt._live_fetch_trends([sym])
        self.assertFalse(result[sym])

    def test_low_baseline_returns_false(self):
        """Baseline < _MIN_BASELINE returns False to avoid noise."""
        values = [3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 10]
        sym = "AAPL"
        df = pd.DataFrame({sym: values, "isPartial": [False] * len(values)})
        mock_pt = MagicMock()
        mock_pt.interest_over_time.return_value = df

        mock_req_module = MagicMock()
        mock_req_module.TrendReq.return_value = mock_pt

        with patch.dict(
            "sys.modules", {"pytrends": MagicMock(), "pytrends.request": mock_req_module}
        ):
            import sys

            if "data.google_trends" in sys.modules:
                del sys.modules["data.google_trends"]
            import data.google_trends as gt

            with patch.object(gt, "time") as mock_time:
                mock_time.sleep = MagicMock()
                result = gt._live_fetch_trends([sym])
        self.assertFalse(result[sym])

    def test_too_few_values_returns_false(self):
        """Fewer than 4 data points returns False."""
        sym = "AAPL"
        df = pd.DataFrame({sym: [20, 22, 21], "isPartial": [False, False, False]})
        mock_pt = MagicMock()
        mock_pt.interest_over_time.return_value = df

        mock_req_module = MagicMock()
        mock_req_module.TrendReq.return_value = mock_pt

        with patch.dict(
            "sys.modules", {"pytrends": MagicMock(), "pytrends.request": mock_req_module}
        ):
            import sys

            if "data.google_trends" in sys.modules:
                del sys.modules["data.google_trends"]
            import data.google_trends as gt

            with patch.object(gt, "time") as mock_time:
                mock_time.sleep = MagicMock()
                result = gt._live_fetch_trends([sym])
        self.assertFalse(result[sym])

    def test_sym_not_in_df_columns_returns_false(self):
        """Symbol not in DataFrame columns returns False."""
        sym = "AAPL"
        df = pd.DataFrame({"OTHER": [20, 21, 22, 23]})
        mock_pt = MagicMock()
        mock_pt.interest_over_time.return_value = df

        mock_req_module = MagicMock()
        mock_req_module.TrendReq.return_value = mock_pt

        with patch.dict(
            "sys.modules", {"pytrends": MagicMock(), "pytrends.request": mock_req_module}
        ):
            import sys

            if "data.google_trends" in sys.modules:
                del sys.modules["data.google_trends"]
            import data.google_trends as gt

            with patch.object(gt, "time") as mock_time:
                mock_time.sleep = MagicMock()
                result = gt._live_fetch_trends([sym])
        self.assertFalse(result[sym])

    def test_none_df_returns_false(self):
        """interest_over_time returns None → False."""
        sym = "AAPL"
        mock_pt = MagicMock()
        mock_pt.interest_over_time.return_value = None

        mock_req_module = MagicMock()
        mock_req_module.TrendReq.return_value = mock_pt

        with patch.dict(
            "sys.modules", {"pytrends": MagicMock(), "pytrends.request": mock_req_module}
        ):
            import sys

            if "data.google_trends" in sys.modules:
                del sys.modules["data.google_trends"]
            import data.google_trends as gt

            with patch.object(gt, "time") as mock_time:
                mock_time.sleep = MagicMock()
                result = gt._live_fetch_trends([sym])
        self.assertFalse(result[sym])

    def test_exception_per_symbol_returns_false(self):
        """Per-symbol exception results in False, not a crash."""
        sym = "AAPL"
        mock_pt = MagicMock()
        mock_pt.interest_over_time.side_effect = RuntimeError("quota exceeded")

        mock_req_module = MagicMock()
        mock_req_module.TrendReq.return_value = mock_pt

        with patch.dict(
            "sys.modules", {"pytrends": MagicMock(), "pytrends.request": mock_req_module}
        ):
            import sys

            if "data.google_trends" in sys.modules:
                del sys.modules["data.google_trends"]
            import data.google_trends as gt

            with patch.object(gt, "time") as mock_time:
                mock_time.sleep = MagicMock()
                result = gt._live_fetch_trends([sym])
        self.assertFalse(result[sym])


class TestGetGoogleTrendsSignals(unittest.TestCase):
    def test_returns_from_cache_when_warm(self):
        cached = {_TODAY_STR: {"AAPL": True}}
        with (
            patch("data.google_trends.today_et", return_value=_today()),
            patch("data.google_trends._load_cache", return_value=cached),
            patch("data.google_trends._save_cache") as mock_save,
        ):
            from data.google_trends import get_google_trends_signals

            result = get_google_trends_signals(["AAPL"])
        self.assertTrue(result["AAPL"])
        mock_save.assert_not_called()

    def test_fetches_on_cache_miss_and_saves(self):
        cached = {_TODAY_STR: {}}
        with (
            patch("data.google_trends.today_et", return_value=_today()),
            patch("data.google_trends._load_cache", return_value=cached),
            patch("data.google_trends._live_fetch_trends", return_value={"AAPL": False}),
            patch("data.google_trends._save_cache") as mock_save,
        ):
            from data.google_trends import get_google_trends_signals

            result = get_google_trends_signals(["AAPL"])
        self.assertFalse(result["AAPL"])
        mock_save.assert_called_once()

    def test_returns_false_for_missing_symbol(self):
        cached = {_TODAY_STR: {"MSFT": False}}
        with (
            patch("data.google_trends.today_et", return_value=_today()),
            patch("data.google_trends._load_cache", return_value=cached),
            patch("data.google_trends._live_fetch_trends", return_value={"AAPL": False}),
            patch("data.google_trends._save_cache"),
        ):
            from data.google_trends import get_google_trends_signals

            result = get_google_trends_signals(["AAPL", "MSFT"])
        self.assertIn("AAPL", result)
        self.assertIn("MSFT", result)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
