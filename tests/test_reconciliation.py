"""Tests for experiment/reconciliation.py — live-vs-sim replay-fidelity reconciliation."""

import os
import shutil
import tempfile
import unittest

from experiment.reconciliation import (
    Divergence,
    build_reconciliation_lines,
    load_reconciliation_summary,
    reconcile_snapshot,
    save_reconciliation_summary,
    summarise,
)


def _snap(rsi=50.0, bb=0.5, signals=("mean_reversion",), **extra):
    return {"rsi_14": rsi, "bb_pct": bb, "fired_signals": list(signals), **extra}


class TestReconcileSnapshot(unittest.TestCase):
    def test_identical_within_tolerance_is_clean(self):
        self.assertEqual(
            reconcile_snapshot(_snap(rsi=30.0), _snap(rsi=30.005)), []
        )  # within tol 0.01

    def test_numeric_divergence_beyond_tolerance(self):
        divs = reconcile_snapshot(_snap(rsi=30.0), _snap(rsi=35.0))
        self.assertEqual(len(divs), 1)
        self.assertEqual(divs[0].field, "rsi_14")
        self.assertAlmostEqual(divs[0].delta, -5.0)

    def test_field_present_on_one_side_only(self):
        live = {"rsi_14": 30.0, "fired_signals": []}
        sim = {"fired_signals": []}  # no rsi_14 in sim
        divs = reconcile_snapshot(live, sim)
        self.assertEqual(
            [(d.field, d.live, d.sim, d.delta) for d in divs], [("rsi_14", 30.0, None, None)]
        )

    def test_non_numeric_field_ignored(self):
        # a string/None value is not a numeric mismatch (both non-numeric -> skipped)
        self.assertEqual(reconcile_snapshot({"rsi_14": None}, {"rsi_14": "x"}), [])

    def test_signal_mismatch_reported_both_directions(self):
        live = _snap(signals=["pead", "mean_reversion"])
        sim = _snap(signals=["mean_reversion", "momentum"])
        divs = {d.field: (d.live, d.sim) for d in reconcile_snapshot(live, sim)}
        self.assertEqual(divs["signal:pead"], (1.0, 0.0))  # live-only
        self.assertEqual(divs["signal:momentum"], (0.0, 1.0))  # sim-only


class TestSummarise(unittest.TestCase):
    def test_all_match_is_full_fidelity(self):
        s = summarise([(_snap(), _snap()), (_snap(rsi=40), _snap(rsi=40))])
        self.assertEqual((s["n_pairs"], s["n_matched"], s["fidelity"]), (2, 2, 1.0))
        self.assertIsNone(s["worst_field"])

    def test_mismatch_lowers_fidelity_and_tracks_drift(self):
        pairs = [
            (_snap(rsi=30), _snap(rsi=30)),  # match
            (_snap(rsi=30), _snap(rsi=36)),  # rsi drift 6
            (_snap(rsi=30), _snap(rsi=34)),  # rsi drift 4
        ]
        s = summarise(pairs)
        self.assertEqual((s["n_pairs"], s["n_matched"]), (3, 1))
        self.assertAlmostEqual(s["fidelity"], round(1 / 3, 4))
        self.assertEqual(s["worst_field"], "rsi_14")
        self.assertEqual(s["field_drift"]["rsi_14"]["n_mismatch"], 2)
        self.assertAlmostEqual(s["field_drift"]["rsi_14"]["mean_abs_delta"], 5.0)  # (6+4)/2

    def test_signal_mismatch_tracked_with_zero_delta(self):
        # a fired-signal divergence has no numeric delta -> mean_abs_delta stays 0.0
        s = summarise([(_snap(signals=["pead"]), _snap(signals=["momentum"]))])
        self.assertEqual(s["n_matched"], 0)
        self.assertEqual(s["field_drift"]["signal:pead"]["n_mismatch"], 1)
        self.assertEqual(s["field_drift"]["signal:pead"]["mean_abs_delta"], 0.0)

    def test_empty_pairs(self):
        s = summarise([])
        self.assertEqual((s["n_pairs"], s["fidelity"], s["worst_field"]), (0, 0.0, None))


class TestBuildLines(unittest.TestCase):
    def test_empty_summary(self):
        lines = build_reconciliation_lines({})
        self.assertEqual(len(lines), 1)
        self.assertIn("no reconstructed snapshots yet", lines[0])

    def test_passing_fidelity_shows_ok_and_worst_field(self):
        summary = {
            "n_pairs": 10,
            "fidelity": 0.95,
            "field_drift": {"rsi_14": {"n_mismatch": 1, "mean_abs_delta": 2.0}},
            "worst_field": "rsi_14",
        }
        joined = "\n".join(build_reconciliation_lines(summary))
        self.assertIn("95.0%", joined)
        self.assertIn("OK", joined)
        self.assertIn("worst-drifting field: rsi_14", joined)

    def test_below_floor_flags(self):
        summary = {"n_pairs": 10, "fidelity": 0.5, "field_drift": {}, "worst_field": None}
        joined = "\n".join(build_reconciliation_lines(summary))
        self.assertIn("FLAG", joined)
        self.assertNotIn("worst-drifting", joined)  # no worst_field -> no second line


class TestPersistence(unittest.TestCase):
    def _p(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d)
        return os.path.join(d, "sub", "reconciliation_summary.json")

    def test_round_trip(self):
        p = self._p()
        save_reconciliation_summary({"n_pairs": 3, "fidelity": 0.66}, p)
        self.assertEqual(load_reconciliation_summary(p)["n_pairs"], 3)

    def test_missing_file_returns_empty(self):
        self.assertEqual(load_reconciliation_summary("/no/such/recon.json"), {})

    def test_malformed_returns_empty(self):
        p = self._p()
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write("{nope")
        self.assertEqual(load_reconciliation_summary(p), {})


class TestDivergenceDataclass(unittest.TestCase):
    def test_fields(self):
        d = Divergence("rsi_14", 30.0, 35.0, -5.0)
        self.assertEqual((d.field, d.live, d.sim, d.delta), ("rsi_14", 30.0, 35.0, -5.0))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
