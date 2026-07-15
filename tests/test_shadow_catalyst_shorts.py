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


class TestNetShortReturn(unittest.TestCase):
    def test_short_profits_when_stock_falls_net_of_costs(self):
        # stock -5%, SPY flat, borrow 3%/yr over 5d (~0.06%), slippage 15bps (0.15%)
        net = scs.net_short_return(-5.0, 0.0, borrow_annual_pct=3.0, hold_days=5, slippage_bps=15.0)
        self.assertAlmostEqual(net, 5.0 - (3.0 * 5 / 252.0) - 0.15, places=4)

    def test_market_neutral_subtracts_spy(self):
        # stock -5%, SPY -4%: the short only earns the 1% idiosyncratic move (no costs here)
        net = scs.net_short_return(-5.0, -4.0, borrow_annual_pct=0.0, hold_days=5, slippage_bps=0.0)
        self.assertAlmostEqual(net, 1.0)

    def test_higher_borrow_lowers_net(self):
        lo = scs.net_short_return(-5.0, 0.0, borrow_annual_pct=3.0, hold_days=5, slippage_bps=0.0)
        hi = scs.net_short_return(-5.0, 0.0, borrow_annual_pct=50.0, hold_days=5, slippage_bps=0.0)
        self.assertGreater(lo, hi)


class TestScoreShortEdge(unittest.TestCase):
    def _obs(self, stock, spy, *sigs):
        return {"stock_ret": stock, "spy_ret": spy, "signals": list(sigs)}

    def test_per_signal_and_pooled(self):
        obs = [
            self._obs(-4.0, 0.0, "guidance_downgrade"),
            self._obs(-2.0, 0.0, "guidance_downgrade", "eps_revision_down_short"),
            self._obs(+3.0, 0.0, "eps_revision_down_short"),  # rose → short lost
        ]
        edges = scs.score_short_edge(obs, borrow_annual_pct=0.0, hold_days=5, slippage_bps=0.0)
        self.assertEqual(edges["__all__"][0], 3)  # pooled n
        gd_n, gd_net, gd_hit = edges["guidance_downgrade"]
        self.assertEqual(gd_n, 2)
        self.assertAlmostEqual(gd_net, 3.0)  # (+4 + +2)/2
        self.assertAlmostEqual(gd_hit, 100.0)
        eps_n, _, eps_hit = edges["eps_revision_down_short"]
        self.assertEqual(eps_n, 2)
        self.assertAlmostEqual(eps_hit, 50.0)  # one win (+2), one loss (-3)

    def test_none_returns_skipped(self):
        obs = [self._obs(None, 0.0, "x"), self._obs(-4.0, None, "x"), self._obs(-4.0, 0.0, "x")]
        edges = scs.score_short_edge(obs, borrow_annual_pct=0.0, hold_days=5, slippage_bps=0.0)
        self.assertEqual(edges["__all__"][0], 1)

    def test_empty_observations(self):
        self.assertEqual(scs.score_short_edge([]), {})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
