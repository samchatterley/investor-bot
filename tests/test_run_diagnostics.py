"""Tests for scripts/run_diagnostics.py — pytest-subprocess diagnostics + JUnit XML parsing."""

import importlib.util
import os
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch
from xml.etree import ElementTree


def _load_diagnostics_module():
    """Import run_diagnostics without executing the __main__ block; stub config."""
    stubs = {"config": MagicMock(LOG_DIR="/tmp/test_diag_logs")}
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


def _write_xml(content: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".xml")
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


class TestParseJunit(unittest.TestCase):
    def test_bare_testsuite_all_pass(self):
        mod = _load_diagnostics_module()
        xml = (
            '<testsuite tests="3" failures="0" errors="0" skipped="0">'
            '<testcase classname="t.A" name="test_a"/></testsuite>'
        )
        p = _write_xml(xml)
        try:
            r = mod._parse_junit(p)
        finally:
            os.remove(p)
        self.assertEqual(r["total"], 3)
        self.assertEqual(r["failed"], 0)
        self.assertEqual(r["failures"], [])

    def test_testsuites_wrapper_with_failure_error_and_skip(self):
        mod = _load_diagnostics_module()
        xml = (
            '<testsuites><testsuite tests="4" failures="1" errors="1" skipped="1">'
            '<testcase classname="t.A" name="test_pass"/>'
            '<testcase classname="t.B" name="test_fail">'
            '<failure message="AssertionError: 1 != 2">trace\nAssertionError: 1 != 2</failure>'
            "</testcase>"
            '<testcase classname="t.C" name="test_err"><error message="RuntimeError: boom"/></testcase>'
            '<testcase classname="t.D" name="test_skip"><skipped/></testcase>'
            "</testsuite></testsuites>"
        )
        p = _write_xml(xml)
        try:
            r = mod._parse_junit(p)
        finally:
            os.remove(p)
        self.assertEqual(r["total"], 4)
        self.assertEqual(r["failed"], 1)
        self.assertEqual(r["errors"], 1)
        self.assertEqual(r["skipped"], 1)
        self.assertEqual(len(r["failures"]), 2)
        names = {f["test"] for f in r["failures"]}
        self.assertEqual(names, {"t.B.test_fail", "t.C.test_err"})

    def test_failure_message_falls_back_to_element_text(self):
        mod = _load_diagnostics_module()
        xml = (
            '<testsuite tests="1" failures="1" errors="0" skipped="0">'
            '<testcase classname="t.A" name="test_x"><failure>body line one\nlast line</failure>'
            "</testcase></testsuite>"
        )
        p = _write_xml(xml)
        try:
            r = mod._parse_junit(p)
        finally:
            os.remove(p)
        self.assertEqual(r["failures"][0]["message"], "last line")


class TestRunDiagnostics(unittest.TestCase):
    _PASS = {"total": 5, "failed": 0, "errors": 0, "skipped": 0, "failures": []}

    def _run(self, mod, parsed=None, timeout=False):
        run_mock = MagicMock()
        if timeout:
            run_mock.side_effect = subprocess.TimeoutExpired(cmd="pytest", timeout=1800)
        with (
            patch.object(mod.subprocess, "run", run_mock),
            patch.object(mod, "_parse_junit", MagicMock(return_value=parsed or self._PASS)),
            patch.object(mod, "_save_report"),
            patch.object(mod.os, "remove"),
        ):
            return mod.run_diagnostics()

    def test_pass_status_when_no_failures(self):
        mod = _load_diagnostics_module()
        report = self._run(mod)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["passed"], 5)
        self.assertEqual(report["total"], 5)

    def test_fail_status_and_passed_count(self):
        mod = _load_diagnostics_module()
        parsed = {
            "total": 5,
            "failed": 1,
            "errors": 1,
            "skipped": 1,
            "failures": [{"test": "t.A.test_x", "message": "boom"}],
        }
        report = self._run(mod, parsed)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["passed"], 2)  # 5 - 1 - 1 - 1
        self.assertEqual(report["failed"], 1)
        self.assertEqual(report["errors"], 1)
        self.assertEqual(report["skipped"], 1)

    def test_errors_only_marks_fail(self):
        mod = _load_diagnostics_module()
        parsed = {
            "total": 3,
            "failed": 0,
            "errors": 1,
            "skipped": 0,
            "failures": [{"test": "t.A.test_e", "message": "err"}],
        }
        report = self._run(mod, parsed)
        self.assertEqual(report["status"], "FAIL")

    def test_timeout_marks_fail(self):
        mod = _load_diagnostics_module()
        report = self._run(mod, timeout=True)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["errors"], 1)
        self.assertIn("timed out", report["failures"][0]["message"])

    def test_parse_error_marks_fail(self):
        mod = _load_diagnostics_module()
        with (
            patch.object(mod.subprocess, "run", MagicMock()),
            patch.object(mod, "_parse_junit", side_effect=ElementTree.ParseError("bad xml")),
            patch.object(mod, "_save_report"),
            patch.object(mod.os, "remove"),
        ):
            report = mod.run_diagnostics()
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["errors"], 1)
        self.assertIn("could not parse", report["failures"][0]["message"])

    def test_missing_xml_marks_fail(self):
        mod = _load_diagnostics_module()
        with (
            patch.object(mod.subprocess, "run", MagicMock()),
            patch.object(mod, "_parse_junit", side_effect=FileNotFoundError("no xml")),
            patch.object(mod, "_save_report"),
            patch.object(mod.os, "remove"),
        ):
            report = mod.run_diagnostics()
        self.assertEqual(report["status"], "FAIL")

    def test_xml_remove_oserror_is_suppressed(self):
        mod = _load_diagnostics_module()
        with (
            patch.object(mod.subprocess, "run", MagicMock()),
            patch.object(mod, "_parse_junit", MagicMock(return_value=self._PASS)),
            patch.object(mod, "_save_report"),
            patch.object(mod.os, "remove", side_effect=OSError("already gone")),
        ):
            report = mod.run_diagnostics()
        self.assertEqual(report["status"], "PASS")

    def test_save_report_called(self):
        mod = _load_diagnostics_module()
        save_mock = MagicMock()
        with (
            patch.object(mod.subprocess, "run", MagicMock()),
            patch.object(mod, "_parse_junit", MagicMock(return_value=self._PASS)),
            patch.object(mod, "_save_report", save_mock),
            patch.object(mod.os, "remove"),
        ):
            mod.run_diagnostics()
        save_mock.assert_called_once()

    def test_duration_is_non_negative_float(self):
        mod = _load_diagnostics_module()
        report = self._run(mod)
        self.assertIsInstance(report["duration_seconds"], float)
        self.assertGreaterEqual(report["duration_seconds"], 0.0)


class TestSaveReport(unittest.TestCase):
    def test_saves_json_to_log_dir(self):
        report = {"status": "PASS", "total": 3, "passed": 3}
        mock_open = MagicMock()
        mock_open.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_open.return_value.__exit__ = MagicMock(return_value=False)
        with (
            patch.dict(sys.modules, {"config": MagicMock(LOG_DIR="/tmp/fake_logs")}),
        ):
            mod2 = _load_diagnostics_module()
            with patch("builtins.open", mock_open), patch("os.makedirs"):
                mod2._save_report(report)
        mock_open.assert_called_once()
        call_path = mock_open.call_args[0][0]
        self.assertIn("test_report_", call_path)
        self.assertIn(".json", call_path)

    def test_handles_open_exception_gracefully(self):
        mod = _load_diagnostics_module()
        with patch("builtins.open", side_effect=OSError("disk full")), patch("os.makedirs"):
            mod._save_report({"status": "PASS"})  # must not raise

    def test_handles_makedirs_exception_gracefully(self):
        mod = _load_diagnostics_module()
        with patch("os.makedirs", side_effect=PermissionError("no permission")):
            mod._save_report({"status": "PASS"})  # must not raise

    def test_json_dump_called_with_report(self):
        mod = _load_diagnostics_module()
        report = {"status": "PASS", "total": 5}
        mock_open = MagicMock()
        mock_open.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_open.return_value.__exit__ = MagicMock(return_value=False)
        mock_dump = MagicMock()
        with (
            patch("builtins.open", mock_open),
            patch("os.makedirs"),
            patch.object(mod.json, "dump", mock_dump),
        ):
            mod._save_report(report)
        mock_dump.assert_called_once()
        self.assertEqual(mock_dump.call_args[0][0]["status"], "PASS")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
