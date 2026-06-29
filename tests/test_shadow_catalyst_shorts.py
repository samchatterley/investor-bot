"""Tests for analysis/shadow_catalyst_shorts.py — regime-independent catalyst-short capture."""

import json
import unittest

import analysis.shadow_catalyst_shorts as scs
from signals.registry import CATALYST_SHORT_SIGNALS


def _snap(symbol="AAA", price=100.0, **flags):
    s = {"symbol": symbol, "current_price": price}
    s.update(flags)
    return s


def _read():
    with open(scs.SHADOW_LOG_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


class TestFlagSignalMapping(unittest.TestCase):
    def test_mapping_matches_registry(self):
        # Drift guard: the shadow flags must map exactly onto the live catalyst short signals.
        self.assertEqual(set(scs._FLAG_TO_SIGNAL.values()), set(CATALYST_SHORT_SIGNALS))


class TestCatalystSignals(unittest.TestCase):
    def test_returns_sorted_signals(self):
        s = _snap(eps_estimate_cut=True, accounting_concern=True)
        self.assertEqual(
            scs._catalyst_signals(s), ["accounting_concern_short", "eps_revision_down_short"]
        )

    def test_empty_when_no_flags(self):
        self.assertEqual(scs._catalyst_signals(_snap()), [])

    def test_ignores_falsey_flags(self):
        self.assertEqual(scs._catalyst_signals(_snap(eps_estimate_cut=False)), [])


class TestSymbolsLoggedOn(unittest.TestCase):
    def test_missing_file_returns_empty(self):
        self.assertEqual(scs._symbols_logged_on("2026-06-29"), set())

    def test_filters_by_date_and_skips_bad_json(self):
        with open(scs.SHADOW_LOG_PATH, "w") as f:
            f.write("not json\n")
            f.write(json.dumps({"date": "2026-06-29", "symbol": "AAA"}) + "\n")
            f.write(json.dumps({"date": "2026-01-01", "symbol": "OLD"}) + "\n")
        self.assertEqual(scs._symbols_logged_on("2026-06-29"), {"AAA"})


class TestCapture(unittest.TestCase):
    def test_writes_catalyst_names_and_returns_count(self):
        snaps = [
            _snap("AAA", eps_estimate_cut=True),
            _snap("BBB", insider_sell_cluster=True),
            _snap("CCC"),  # no catalyst
        ]
        n = scs.capture(snaps, "NEUTRAL_CHOP", "midday", "run1", today="2026-06-29")
        self.assertEqual(n, 2)
        rows = _read()
        self.assertEqual({r["symbol"] for r in rows}, {"AAA", "BBB"})
        aaa = next(r for r in rows if r["symbol"] == "AAA")
        self.assertEqual(aaa["regime"], "NEUTRAL_CHOP")
        self.assertEqual(aaa["entry_price"], 100.0)
        self.assertEqual(aaa["catalyst_signals"], ["eps_revision_down_short"])

    def test_no_catalysts_returns_zero_and_writes_nothing(self):
        n = scs.capture([_snap("AAA")], "BULL_TREND", "open", "r", today="2026-06-29")
        self.assertEqual(n, 0)
        self.assertEqual(scs._symbols_logged_on("2026-06-29"), set())

    def test_skips_names_without_price(self):
        snaps = [_snap("AAA", price=None, eps_estimate_cut=True)]
        self.assertEqual(scs.capture(snaps, "BULL", "open", "r", today="2026-06-29"), 0)

    def test_skips_names_without_symbol(self):
        snaps = [{"current_price": 10.0, "eps_estimate_cut": True}]
        self.assertEqual(scs.capture(snaps, "BULL", "open", "r", today="2026-06-29"), 0)

    def test_dedups_within_call(self):
        snaps = [_snap("AAA", eps_estimate_cut=True), _snap("AAA", accounting_concern=True)]
        self.assertEqual(scs.capture(snaps, "BULL", "open", "r", today="2026-06-29"), 1)

    def test_dedups_against_todays_log(self):
        scs.capture([_snap("AAA", eps_estimate_cut=True)], "BULL", "open", "r", today="2026-06-29")
        n = scs.capture(
            [_snap("AAA", accounting_concern=True)], "BULL", "midday", "r", today="2026-06-29"
        )
        self.assertEqual(n, 0)

    def test_recaptures_on_a_different_day(self):
        scs.capture([_snap("AAA", eps_estimate_cut=True)], "BULL", "open", "r", today="2026-06-29")
        n = scs.capture(
            [_snap("AAA", eps_estimate_cut=True)], "BULL", "open", "r", today="2026-06-30"
        )
        self.assertEqual(n, 1)

    def test_default_today_used_when_omitted(self):
        n = scs.capture([_snap("ZZZ", eps_estimate_cut=True)], "BULL", "open", "r")
        self.assertEqual(n, 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
