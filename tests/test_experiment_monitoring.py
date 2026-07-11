import os
import tempfile
import unittest
from datetime import date
from unittest.mock import patch

from experiment.monitoring import (
    MONITORING_BANNER,
    ExperimentState,
    append_log_entry,
    build_edge_anatomy_lines,
    build_monitoring_lines,
    build_three_arm_summary,
    load_scored_observations,
)


def _obs(symbol, *, sel=False, conf=8, fr5=0.5, rsi=50.0, sig="pead", date="2026-06-17", cost=0.0):
    return {
        "symbol": symbol,
        "date": date,
        "extra": {
            "decision_type": "buy_candidate",
            "mode": "open",
            "arm3_ai_selected": sel,
            "arm3_ai_confidence": conf,
        },
        "features": {"rsi_14": rsi},
        "fired_signals": [sig],
        "outcomes": {"forward_r_5d": fr5, "cost_r_estimate": cost},
    }


class TestBuildMonitoringLines(unittest.TestCase):
    def test_default_lines_report_phase0_status(self):
        lines = build_monitoring_lines()
        self.assertEqual(lines[0], MONITORING_BANNER)
        joined = "\n".join(lines)
        self.assertIn("Phase 0", joined)
        self.assertIn("Gate A", joined)
        self.assertIn("Gate B", joined)
        self.assertIn("N_eff = 0", joined)
        self.assertIn("N_eff >= 200", joined)

    def test_populated_state_includes_arm_metrics_and_notes(self):
        state = ExperimentState(
            phase="live-shadow",
            noise_audit_status="passed (3-level stable)",
            n_eff_accumulated=120.0,
            next_milestone=200,
            arm_metrics={"Arm 3 minus Arm 2 incremental IC": "0.04 (n=120)"},
            notes=["context-present candidates this week: 18"],
        )
        joined = "\n".join(build_monitoring_lines(state))
        self.assertIn("live-shadow", joined)
        self.assertIn("Arm 3 minus Arm 2 incremental IC: 0.04 (n=120)", joined)
        self.assertIn("context-present candidates this week: 18", joined)
        self.assertIn("N_eff = 120", joined)

    def test_banner_has_no_em_dash(self):
        self.assertNotIn("—", MONITORING_BANNER)


class TestAppendLogEntry(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        import shutil

        self.addCleanup(shutil.rmtree, self.tmpdir)

    def test_writes_dated_block(self):
        path = os.path.join(self.tmpdir, "sub", "log.md")  # parent does not exist yet
        block = append_log_entry(["alpha", "beta"], entry_date="2026-06-14", log_path=path)
        self.assertIn("## 2026-06-14", block)
        self.assertIn("- alpha", block)
        self.assertIn("- beta", block)
        with open(path) as f:
            self.assertIn("- alpha", f.read())

    def test_default_date_is_today(self):
        path = os.path.join(self.tmpdir, "log.md")
        block = append_log_entry(["x"], log_path=path)
        self.assertIn(f"## {date.today().isoformat()}", block)

    def test_appends_newest_last(self):
        path = os.path.join(self.tmpdir, "log.md")
        append_log_entry(["one"], entry_date="2026-06-01", log_path=path)
        append_log_entry(["two"], entry_date="2026-06-08", log_path=path)
        with open(path) as f:
            text = f.read()
        self.assertLess(text.index("2026-06-01"), text.index("2026-06-08"))

    def test_filename_only_skips_makedirs(self):
        old = os.getcwd()
        os.chdir(self.tmpdir)
        try:
            block = append_log_entry(["a"], entry_date="2026-06-14", log_path="log.md")
        finally:
            os.chdir(old)
        self.assertIn("## 2026-06-14", block)
        self.assertTrue(os.path.exists(os.path.join(self.tmpdir, "log.md")))

    def test_oserror_is_swallowed(self):
        path = os.path.join(self.tmpdir, "log.md")
        with patch("builtins.open", side_effect=OSError("disk full")):
            result = append_log_entry(["x"], entry_date="2026-06-14", log_path=path)
        self.assertEqual(result, "")


class TestBuildThreeArmSummary(unittest.TestCase):
    def test_empty_is_scaffold(self):
        lines = build_three_arm_summary()
        self.assertEqual(len(lines), 1)
        self.assertIn("No matched decisions yet", lines[0])

    def test_populated_lists_arms_and_headline(self):
        lines = build_three_arm_summary(
            {
                "arm1": "avg R 0.05 (n=120)",
                "arm2": "avg R 0.06 (n=120)",
                "arm3": "avg R 0.11 (n=40)",
                "headline": "Arm3 minus Arm2 incremental IC 0.04 (not yet significant)",
            }
        )
        joined = "\n".join(lines)
        self.assertIn("Arm 1 (Champion, deterministic): avg R 0.05 (n=120)", joined)
        self.assertIn("Arm 3 (contextual LLM): avg R 0.11 (n=40)", joined)
        self.assertIn("incremental IC 0.04", joined)

    def test_partial_arms(self):
        self.assertEqual(
            build_three_arm_summary({"arm1": "x"}), ["Arm 1 (Champion, deterministic): x"]
        )

    def test_truthy_but_no_known_keys(self):
        self.assertEqual(build_three_arm_summary({"foo": "bar"}), ["No arm metrics available yet."])


class TestLoadScoredObservations(unittest.TestCase):
    def test_reads_jsonl_skipping_blanks(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "scored.jsonl")
            with open(p, "w") as f:
                f.write('{"symbol": "AAA"}\n\n{"symbol": "BBB"}\n')
            rows = load_scored_observations(p)
        self.assertEqual([r["symbol"] for r in rows], ["AAA", "BBB"])

    def test_missing_file_returns_empty(self):
        self.assertEqual(load_scored_observations("/no/such/scored.jsonl"), [])

    def test_malformed_json_returns_empty(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "scored.jsonl")
            with open(p, "w") as f:
                f.write("{not json}\n")
            self.assertEqual(load_scored_observations(p), [])


class TestBuildEdgeAnatomyLines(unittest.TestCase):
    def test_no_scored_candidates(self):
        self.assertEqual(
            build_edge_anatomy_lines([]), ["Edge anatomy: no scored open-mode buy candidates yet."]
        )

    def test_ignores_non_open_and_unclosed_horizon(self):
        rows = [
            {
                "symbol": "X",
                "date": "d",
                "extra": {"decision_type": "buy_candidate", "mode": "close"},
                "outcomes": {"forward_r_5d": 1.0},
            },
            {
                "symbol": "Y",
                "date": "d",
                "extra": {"decision_type": "buy_candidate", "mode": "open"},
                "outcomes": {"forward_r_5d": None},
            },
        ]
        self.assertEqual(
            build_edge_anatomy_lines(rows)[0],
            "Edge anatomy: no scored open-mode buy candidates yet.",
        )

    def test_dedups_duplicate_symbol_date(self):
        # same (symbol, date) across 3 run-bursts must count once
        rows = [_obs("DUP", sel=True, fr5=0.5) for _ in range(3)]
        line = build_edge_anatomy_lines(rows)[0]
        self.assertIn("field n=1", line)
        self.assertIn("AI picks n=1", line)

    def test_confidence_buckets_and_accumulating_trigger(self):
        rows = [
            _obs("A", sel=True, conf=7, fr5=0.4),
            _obs("B", sel=True, conf=8, fr5=0.9),
            _obs("C", sel=True, conf=9, fr5=0.8),
            _obs("D", sel=False, conf=None, fr5=0.3),
        ]
        out = "\n".join(build_edge_anatomy_lines(rows))
        self.assertIn("conf<=7: n=1", out)
        self.assertIn("conf=8: n=1", out)
        self.assertIn("conf>=9: n=1", out)
        self.assertIn("accumulating", out)  # far below n>=50

    def test_trigger_met(self):
        # a large non-pick field at 0.5 sets field_mean ~0.5; conf<=7 picks match it (~0 edge),
        # conf=8 picks sit well above -> the pre-registered pattern holds at n>=50
        rows = [_obs(f"F{i}", sel=False, conf=None, fr5=0.50) for i in range(400)]
        rows += [_obs(f"L{i}", sel=True, conf=7, fr5=0.50) for i in range(60)]
        rows += [_obs(f"H{i}", sel=True, conf=8, fr5=0.80) for i in range(60)]
        out = "\n".join(build_edge_anatomy_lines(rows))
        self.assertIn("PRE-REGISTERED TRIGGER MET", out)

    def test_trigger_present_but_pattern_fails(self):
        # both buckets clear n>=50 but conf=8 shows no real edge over the field -> do not raise
        rows = [_obs(f"F{i}", sel=False, conf=None, fr5=0.50) for i in range(400)]
        rows += [_obs(f"L{i}", sel=True, conf=7, fr5=0.50) for i in range(60)]
        rows += [_obs(f"H{i}", sel=True, conf=8, fr5=0.52) for i in range(60)]
        out = "\n".join(build_edge_anatomy_lines(rows))
        self.assertIn("did NOT hold", out)

    def test_extension_and_signal_family_lines(self):
        rows = [_obs(f"E{i}", sel=(i < 5), conf=8, fr5=0.3, rsi=65, sig="pead") for i in range(10)]
        rows += [
            _obs(f"N{i}", sel=(i < 5), conf=8, fr5=0.6, rsi=40, sig="momentum") for i in range(10)
        ]
        out = "\n".join(build_edge_anatomy_lines(rows))
        self.assertIn("extended(rsi>=60)", out)
        self.assertIn("not-extended", out)
        self.assertIn("signal pead", out)  # >=4 picks
        self.assertIn("signal momentum", out)

    def test_missing_rsi_and_no_fired_signals_are_tolerated(self):
        rows = [
            {
                "symbol": f"Z{i}",
                "date": "d",
                "extra": {
                    "decision_type": "buy_candidate",
                    "mode": "open",
                    "arm3_ai_selected": True,
                    "arm3_ai_confidence": 8,
                },
                "outcomes": {"forward_r_5d": 0.5},
            }
            for i in range(4)
        ]  # no features, no fired_signals
        out = "\n".join(build_edge_anatomy_lines(rows))
        self.assertIn("signal (none)", out)  # falls back to (none) family


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
