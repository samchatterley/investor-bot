"""Tests for scripts/run_diagnostics.py — run_diagnostics() and _save_report()."""
import importlib.util
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch


def _load_diagnostics_module():
    """
    Import run_diagnostics without executing the __main__ block.
    Stubs out config so no real credentials are needed.
    """
    stubs = {
        "config": MagicMock(LOG_DIR="/tmp/test_diag_logs"),
    }
    with patch.dict(sys.modules, stubs):
        sys.modules.pop("scripts.run_diagnostics", None)

        spec = importlib.util.spec_from_file_location(
            "scripts.run_diagnostics",
            os.path.join(os.path.dirname(__file__), "..", "scripts", "run_diagnostics.py"),
        )
        mod = types.ModuleType(spec.name)
        mod.__spec__ = spec
        mod.__file__ = spec.origin
        mod.__package__ = "scripts"
        spec.loader.exec_module(mod)
        return mod


def _make_suite(failures=None, errors=None, tests_run=3):
    """
    Return a (loader_mock, suite_mock, result_class) triple.
    The result_class's instances will report the requested outcome.
    """
    failures = failures or []
    errors = errors or []

    suite_mock = MagicMock()

    def _run_suite(result):
        result.testsRun = tests_run
        result.failures = failures
        result.errors = errors

    suite_mock.run.side_effect = _run_suite

    loader_mock = MagicMock()
    loader_mock.discover.return_value = suite_mock
    return loader_mock, suite_mock


class TestRunDiagnosticsAllPass(unittest.TestCase):

    def test_returns_pass_status_when_no_failures(self):
        mod = _load_diagnostics_module()
        loader_mock, suite_mock = _make_suite(tests_run=5)

        with patch.object(mod.unittest, "TestLoader", return_value=loader_mock), \
             patch.object(mod, "_save_report"):
            report = mod.run_diagnostics()

        self.assertEqual(report["status"], "PASS")

    def test_correct_total_and_passed_counts(self):
        mod = _load_diagnostics_module()
        loader_mock, suite_mock = _make_suite(tests_run=7)

        with patch.object(mod.unittest, "TestLoader", return_value=loader_mock), \
             patch.object(mod, "_save_report"):
            report = mod.run_diagnostics()

        self.assertEqual(report["total"], 7)
        self.assertEqual(report["passed"], 7)
        self.assertEqual(report["failed"], 0)
        self.assertEqual(report["errors"], 0)
        self.assertEqual(report["failures"], [])

    def test_report_has_timestamp_and_duration(self):
        mod = _load_diagnostics_module()
        loader_mock, suite_mock = _make_suite(tests_run=2)

        with patch.object(mod.unittest, "TestLoader", return_value=loader_mock), \
             patch.object(mod, "_save_report"):
            report = mod.run_diagnostics()

        self.assertIn("timestamp", report)
        self.assertIn("duration_seconds", report)
        self.assertIsInstance(report["duration_seconds"], float)

    def test_save_report_called(self):
        mod = _load_diagnostics_module()
        loader_mock, suite_mock = _make_suite(tests_run=3)
        mock_save = MagicMock()

        with patch.object(mod.unittest, "TestLoader", return_value=loader_mock), \
             patch.object(mod, "_save_report", mock_save):
            mod.run_diagnostics()

        mock_save.assert_called_once()


class TestRunDiagnosticsWithFailures(unittest.TestCase):

    def test_returns_fail_status_when_failures_exist(self):
        mod = _load_diagnostics_module()
        test_obj = MagicMock()
        test_obj.__str__ = lambda self: "TestFoo.test_bar"
        failures = [(test_obj, "AssertionError: 1 != 2\n  assert 1 == 2\nAssertionError: 1 != 2")]
        loader_mock, suite_mock = _make_suite(failures=failures, tests_run=3)

        with patch.object(mod.unittest, "TestLoader", return_value=loader_mock), \
             patch.object(mod, "_save_report"):
            report = mod.run_diagnostics()

        self.assertEqual(report["status"], "FAIL")

    def test_failure_details_included_in_report(self):
        mod = _load_diagnostics_module()
        test_obj = MagicMock()
        test_obj.__str__ = lambda self: "TestFoo.test_bar"
        failures = [(test_obj, "Traceback...\nAssertionError: expected True")]
        loader_mock, suite_mock = _make_suite(failures=failures, tests_run=4)

        with patch.object(mod.unittest, "TestLoader", return_value=loader_mock), \
             patch.object(mod, "_save_report"):
            report = mod.run_diagnostics()

        self.assertEqual(len(report["failures"]), 1)
        self.assertIn("message", report["failures"][0])
        self.assertIn("test", report["failures"][0])

    def test_errors_count_in_report(self):
        mod = _load_diagnostics_module()
        test_obj = MagicMock()
        test_obj.__str__ = lambda self: "TestFoo.test_baz"
        errors = [(test_obj, "RuntimeError: something broke")]
        loader_mock, suite_mock = _make_suite(errors=errors, tests_run=5)

        with patch.object(mod.unittest, "TestLoader", return_value=loader_mock), \
             patch.object(mod, "_save_report"):
            report = mod.run_diagnostics()

        self.assertEqual(report["errors"], 1)
        self.assertEqual(report["status"], "FAIL")

    def test_passed_count_is_total_minus_failures_and_errors(self):
        mod = _load_diagnostics_module()
        test_obj = MagicMock()
        test_obj.__str__ = lambda self: "TestFoo.test_x"
        failures = [(test_obj, "AssertionError")]
        errors_list = [(test_obj, "RuntimeError")]
        loader_mock, suite_mock = _make_suite(
            failures=failures, errors=errors_list, tests_run=10
        )

        with patch.object(mod.unittest, "TestLoader", return_value=loader_mock), \
             patch.object(mod, "_save_report"):
            report = mod.run_diagnostics()

        self.assertEqual(report["passed"], 8)  # 10 - 1 failure - 1 error


class TestRunDiagnosticsTimings(unittest.TestCase):

    def test_duration_is_non_negative(self):
        mod = _load_diagnostics_module()
        loader_mock, suite_mock = _make_suite(tests_run=1)

        with patch.object(mod.unittest, "TestLoader", return_value=loader_mock), \
             patch.object(mod, "_save_report"):
            report = mod.run_diagnostics()

        self.assertGreaterEqual(report["duration_seconds"], 0.0)

    def test_duration_measured_via_monotonic(self):
        """Duration reflects actual elapsed time measured by time.monotonic."""
        mod = _load_diagnostics_module()
        loader_mock, suite_mock = _make_suite(tests_run=1)

        # Simulate 0.5 second run
        call_count = [0]

        def fake_monotonic():
            call_count[0] += 1
            return call_count[0] * 0.5

        with patch.object(mod.unittest, "TestLoader", return_value=loader_mock), \
             patch.object(mod, "_save_report"), \
             patch.object(mod.time, "monotonic", side_effect=fake_monotonic):
            report = mod.run_diagnostics()

        self.assertAlmostEqual(report["duration_seconds"], 0.5, places=1)


class TestSaveReport(unittest.TestCase):

    def test_saves_json_to_log_dir(self):
        report = {"status": "PASS", "total": 3, "passed": 3}

        mock_open = MagicMock()
        mock_file = MagicMock()
        mock_open.return_value.__enter__ = MagicMock(return_value=mock_file)
        mock_open.return_value.__exit__ = MagicMock(return_value=False)

        with patch("builtins.open", mock_open), \
             patch("os.makedirs"), \
             patch.dict(sys.modules, {"config": MagicMock(LOG_DIR="/tmp/fake_logs")}):
            # Reload to pick up fresh config stub
            mod2 = _load_diagnostics_module()
            with patch("builtins.open", mock_open), \
                 patch("os.makedirs"):
                mod2._save_report(report)

        mock_open.assert_called_once()
        call_path = mock_open.call_args[0][0]
        self.assertIn("test_report_", call_path)
        self.assertIn(".json", call_path)

    def test_handles_exception_gracefully(self):
        """If open() raises, _save_report must not propagate the exception."""
        mod = _load_diagnostics_module()
        report = {"status": "PASS"}

        with patch("builtins.open", side_effect=OSError("disk full")), \
             patch("os.makedirs"):
            try:
                mod._save_report(report)
            except Exception as exc:
                self.fail(f"_save_report raised unexpectedly: {exc}")

    def test_handles_makedirs_exception_gracefully(self):
        """If makedirs raises, _save_report logs and swallows the error."""
        mod = _load_diagnostics_module()
        report = {"status": "PASS"}

        with patch("os.makedirs", side_effect=PermissionError("no permission")):
            try:
                mod._save_report(report)
            except Exception as exc:
                self.fail(f"_save_report raised unexpectedly: {exc}")

    def test_json_dump_called_with_report(self):
        """json.dump is called with the full report dict."""
        mod = _load_diagnostics_module()
        report = {"status": "PASS", "total": 5}

        mock_open = MagicMock()
        mock_file = MagicMock()
        mock_open.return_value.__enter__ = MagicMock(return_value=mock_file)
        mock_open.return_value.__exit__ = MagicMock(return_value=False)
        mock_dump = MagicMock()

        with patch("builtins.open", mock_open), \
             patch("os.makedirs"), \
             patch.object(mod.json, "dump", mock_dump):
            mod._save_report(report)

        mock_dump.assert_called_once()
        dumped_report = mock_dump.call_args[0][0]
        self.assertEqual(dumped_report["status"], "PASS")
        self.assertEqual(dumped_report["total"], 5)


if __name__ == "__main__":
    unittest.main()
