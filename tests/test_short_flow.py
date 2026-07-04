"""Tests for data/short_flow.py — FINRA Reg SHO daily short-volume feed."""

import os
import tempfile
import unittest
from datetime import date
from unittest.mock import MagicMock, patch

import data.short_flow as sf

_FILE = (
    "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
    "20260701|AAPL|1000|5|4000|B,Q,N\n"
    "20260701|MSFT|300|0|1000|B,Q,N\n"
    "20260701|BAD|x|0|1000|B,Q,N\n"
    "20260701|ZERO|10|0|0|B,Q,N\n"
    "short|row\n"
)


class TestParseDay(unittest.TestCase):
    def test_parses_ratio_and_skips_bad_rows(self):
        out = sf._parse_day(_FILE)
        self.assertAlmostEqual(out["AAPL"], 0.25)
        self.assertAlmostEqual(out["MSFT"], 0.30)
        self.assertNotIn("BAD", out)  # non-numeric
        self.assertNotIn("ZERO", out)  # zero total volume

    def test_symbol_filter(self):
        out = sf._parse_day(_FILE, symbols={"MSFT"})
        self.assertEqual(list(out), ["MSFT"])

    def test_empty_text(self):
        self.assertEqual(sf._parse_day("header only"), {})


class TestFetchDayText(unittest.TestCase):
    @patch("data.short_flow.requests.get")
    def test_ok(self, mget):
        resp = MagicMock()
        resp.text = _FILE
        mget.return_value = resp
        self.assertEqual(sf._fetch_day_text(date(2026, 7, 1)), _FILE)
        resp.raise_for_status.assert_called_once()

    @patch("data.short_flow.requests.get", side_effect=RuntimeError("net down"))
    def test_failure_returns_none(self, _):
        self.assertIsNone(sf._fetch_day_text(date(2026, 7, 1)))


class TestGetDay(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._p = patch.object(sf, "_CACHE_DIR", self._tmp)
        self._p.start()

    def tearDown(self):
        self._p.stop()

    def test_fetches_parses_and_caches(self):
        with patch.object(sf, "_fetch_day_text", return_value=_FILE) as mf:
            out = sf.get_day(date(2026, 7, 1))
        self.assertAlmostEqual(out["AAPL"], 0.25)
        mf.assert_called_once()
        # second call served from disk — no network
        with patch.object(sf, "_fetch_day_text") as mf2:
            out2 = sf.get_day(date(2026, 7, 1), symbols={"AAPL"})
        mf2.assert_not_called()
        self.assertEqual(list(out2), ["AAPL"])

    def test_fetch_failure_returns_empty(self):
        with patch.object(sf, "_fetch_day_text", return_value=None):
            self.assertEqual(sf.get_day(date(2026, 7, 2)), {})

    def test_no_cache_bypasses_disk(self):
        with patch.object(sf, "_fetch_day_text", return_value=_FILE):
            out = sf.get_day(date(2026, 7, 3), use_cache=False)
        self.assertIn("AAPL", out)
        self.assertFalse(os.path.exists(os.path.join(self._tmp, "2026-07-03.txt")))

    def test_corrupt_cache_file_refetches(self):
        path = os.path.join(self._tmp, "2026-07-04.txt")
        os.makedirs(self._tmp, exist_ok=True)
        with open(path, "w") as f:
            f.write(_FILE)
        os.chmod(path, 0o000)  # unreadable → OSError → refetch
        try:
            with patch.object(sf, "_fetch_day_text", return_value=_FILE) as mf:
                out = sf.get_day(date(2026, 7, 4))
            mf.assert_called_once()
            self.assertIn("AAPL", out)
        finally:
            os.chmod(path, 0o644)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
