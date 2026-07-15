"""Tests for analysis/macro_gate_shadow.py — macro-gate efficacy shadow log + per-event scoring."""

import os
import tempfile
import unittest
from unittest.mock import patch

from analysis.macro_gate_shadow import capture, load, score_event


class TestCapture(unittest.TestCase):
    def _path(self):
        d = tempfile.mkdtemp()
        self.addCleanup(__import__("shutil").rmtree, d)
        return os.path.join(d, "sub", "macro_gate_shadow.jsonl")  # parent missing → makedirs

    def test_writes_record_with_filtered_candidates(self):
        p = self._path()
        capture(
            "CPI inflation release",
            [
                {"symbol": "BAC", "confidence": 8, "key_signal": "pead"},
                {"symbol": "", "confidence": 7},  # no symbol → dropped
                "not-a-dict",  # non-dict → dropped
                {"confidence": 6},  # no symbol key → dropped
            ],
            today="2026-07-15",
            mode="open",
            regime="NEUTRAL_CHOP",
            vix=16.3,
            log_path=p,
        )
        rows = load(p)
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual(r["macro_event"], "CPI inflation release")
        self.assertEqual(r["date"], "2026-07-15")
        self.assertEqual(r["mode"], "open")
        self.assertEqual([c["symbol"] for c in r["candidates"]], ["BAC"])

    def test_none_event_and_candidates_tolerated(self):
        p = self._path()
        capture(None, None, today="2026-07-15", log_path=p)
        r = load(p)[0]
        self.assertEqual(r["macro_event"], "(unspecified)")
        self.assertEqual(r["candidates"], [])

    def test_default_date_is_today(self):
        from datetime import date

        p = self._path()
        capture("FOMC", [{"symbol": "X", "confidence": 8}], log_path=p)
        self.assertEqual(load(p)[0]["date"], date.today().isoformat())

    def test_capture_is_failsafe(self):
        # an unwritable open must be swallowed, not raised
        with patch("builtins.open", side_effect=OSError("disk full")):
            capture(
                "CPI", [{"symbol": "X", "confidence": 8}], log_path="/whatever.jsonl"
            )  # no raise


class TestLoad(unittest.TestCase):
    def test_missing_file_returns_empty(self):
        self.assertEqual(load("/no/such/macro_gate.jsonl"), [])

    def test_skips_blank_lines(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.jsonl")
            with open(p, "w") as f:
                f.write('{"date":"a"}\n\n{"date":"b"}\n')
            self.assertEqual([r["date"] for r in load(p)], ["a", "b"])

    def test_malformed_returns_empty(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.jsonl")
            with open(p, "w") as f:
                f.write("{bad json}\n")
            self.assertEqual(load(p), [])


class TestScoreEvent(unittest.TestCase):
    def _cands(self, *pairs):
        return [{"symbol": s, "confidence": c, "key_signal": "x"} for s, c in pairs]

    def test_gate_saved_us_when_names_lag_market(self):
        # SPY flat (+0%), both blocked names down → negative excess ⇒ gate saved us
        prices = {"SPY": (100.0, 100.0), "BAC": (100.0, 99.0), "CI": (100.0, 98.0)}
        n, mean = score_event(self._cands(("BAC", 8), ("CI", 7)), prices)
        self.assertEqual(n, 2)
        self.assertAlmostEqual(mean, -1.5)  # (-1 + -2)/2

    def test_gate_cost_us_when_names_beat_market(self):
        prices = {"SPY": (100.0, 100.0), "BAC": (100.0, 103.0)}
        n, mean = score_event(self._cands(("BAC", 8)), prices)
        self.assertEqual((n, round(mean, 3)), (1, 3.0))

    def test_excess_is_net_of_spy(self):
        # name +2%, SPY +1% → excess +1%
        prices = {"SPY": (100.0, 101.0), "BAC": (100.0, 102.0)}
        _, mean = score_event(self._cands(("BAC", 8)), prices)
        self.assertAlmostEqual(mean, 1.0)

    def test_low_confidence_excluded(self):
        prices = {"SPY": (100.0, 100.0), "LOW": (100.0, 90.0), "HI": (100.0, 95.0)}
        n, mean = score_event(self._cands(("LOW", 5), ("HI", 7)), prices)
        self.assertEqual(n, 1)  # LOW (conf 5) excluded
        self.assertAlmostEqual(mean, -5.0)

    def test_missing_price_skipped(self):
        prices = {"SPY": (100.0, 100.0), "BAC": (100.0, 99.0)}  # CI absent
        n, _ = score_event(self._cands(("BAC", 8), ("CI", 8)), prices)
        self.assertEqual(n, 1)

    def test_no_spy_returns_none(self):
        self.assertEqual(score_event(self._cands(("BAC", 8)), {"BAC": (100.0, 99.0)}), (0, None))

    def test_bad_spy_prices_return_none(self):
        self.assertEqual(score_event(self._cands(("BAC", 8)), {"SPY": (0.0, 100.0)}), (0, None))
        self.assertEqual(score_event(self._cands(("BAC", 8)), {"SPY": (100.0, None)}), (0, None))

    def test_nothing_scorable_returns_none(self):
        prices = {"SPY": (100.0, 100.0), "BAC": (0.0, 99.0)}  # bad entry price
        self.assertEqual(score_event(self._cands(("BAC", 8)), prices), (0, None))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
