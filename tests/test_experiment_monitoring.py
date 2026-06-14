import os
import tempfile
import unittest
from datetime import date
from unittest.mock import patch

from experiment.monitoring import (
    MONITORING_BANNER,
    ExperimentState,
    append_log_entry,
    build_monitoring_lines,
    build_three_arm_summary,
)


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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
