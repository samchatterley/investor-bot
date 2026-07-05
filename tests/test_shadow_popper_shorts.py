"""Tests for analysis/shadow_popper_shorts.py — crowded-popper short shadow logger."""

import json
import unittest
from unittest.mock import patch

import analysis.shadow_popper_shorts as sp

# 100+ SVR values so the q3 computation engages; universe names UNI000..UNI119 plus our targets.
_SVR = {f"UNI{i:03d}": 0.30 + (i % 10) * 0.01 for i in range(120)}
_SVR["POPC"] = 0.90  # crowded popper (>= q3)
_SVR["POPL"] = 0.05  # light-flow popper (< q3)


def _snap(sym, ret1):
    return {"symbol": sym, "ret_1d_pct": ret1}


class TestCapture(unittest.TestCase):
    def test_crowded_popper_logged_light_and_nonpop_skipped(self):
        snaps = [_snap("POPC", 12.0), _snap("POPL", 15.0), _snap("UNI000", 2.0)]
        with patch.object(sp, "_latest_svr", return_value=("2026-07-03", _SVR)):
            n = sp.capture(snaps, "open", "run1", today="2026-07-04")
        self.assertEqual(n, 1)
        with open(sp.SHADOW_LOG_PATH) as f:
            rows = [json.loads(line) for line in f]
        self.assertEqual(rows[0]["symbol"], "POPC")
        self.assertEqual(rows[0]["svr_day"], "2026-07-03")

    def test_same_day_dedupe(self):
        snaps = [_snap("POPC", 12.0)]
        with patch.object(sp, "_latest_svr", return_value=("2026-07-03", _SVR)):
            first = sp.capture(snaps, "open", "run1", today="2026-07-04")
            second = sp.capture(snaps, "midday", "run2", today="2026-07-04")
        self.assertEqual((first, second), (1, 0))

    def test_dedupe_skips_corrupt_and_other_date_lines(self):
        """_symbols_logged_on tolerates garbage lines and ignores rows from other dates."""
        import os

        os.makedirs(os.path.dirname(sp.SHADOW_LOG_PATH), exist_ok=True)
        with open(sp.SHADOW_LOG_PATH, "w") as f:
            f.write("{not json\n")
            f.write(json.dumps({"date": "2026-07-01", "symbol": "POPC"}) + "\n")
        with patch.object(sp, "_latest_svr", return_value=("2026-07-03", _SVR)):
            n = sp.capture([_snap("POPC", 12.0)], "open", "r", today="2026-07-04")
        self.assertEqual(n, 1)  # old-date row does NOT dedupe today's capture

    def test_no_svr_data_returns_zero(self):
        with patch.object(sp, "_latest_svr", return_value=("", {})):
            self.assertEqual(sp.capture([_snap("POPC", 12.0)], "open", "r", today="2026-07-04"), 0)

    def test_too_few_svr_names_returns_zero(self):
        with patch.object(sp, "_latest_svr", return_value=("2026-07-03", {"POPC": 0.9})):
            self.assertEqual(sp.capture([_snap("POPC", 12.0)], "open", "r", today="2026-07-04"), 0)

    def test_missing_ret_or_symbol_skipped(self):
        snaps = [{"symbol": "POPC"}, {"ret_1d_pct": 20.0}, _snap("NOSVR", 20.0)]
        with patch.object(sp, "_latest_svr", return_value=("2026-07-03", _SVR)):
            self.assertEqual(sp.capture(snaps, "open", "r", today="2026-07-04"), 0)

    def test_capture_never_raises(self):
        with patch.object(sp, "_latest_svr", side_effect=RuntimeError("feed down")):
            self.assertEqual(sp.capture([_snap("POPC", 12.0)], "open", "r"), 0)


class TestLatestSvr(unittest.TestCase):
    def test_walks_back_to_most_recent_file(self):
        calls = []

        def fake_get_day(d, symbols=None):
            calls.append(d)
            return {"AAA": 0.4} if len(calls) >= 3 else {}

        with patch("data.short_flow.get_day", side_effect=fake_get_day):
            day, svr = sp._latest_svr()
        self.assertEqual(len(calls), 3)
        self.assertEqual(svr, {"AAA": 0.4})
        self.assertEqual(day, calls[2].isoformat())

    def test_gives_up_after_five_days(self):
        with patch("data.short_flow.get_day", return_value={}):
            day, svr = sp._latest_svr()
        self.assertEqual((day, svr), ("", {}))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
