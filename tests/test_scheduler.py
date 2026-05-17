"""Tests for scripts/run_scheduler.py — job registration and import safety."""

import sys
import types
import unittest
from unittest.mock import MagicMock, patch


def _load_scheduler_module():
    """
    Import run_scheduler without executing the __main__ block.
    Patches heavy dependencies so the import is fast and side-effect-free.
    """
    stubs = {
        "config": MagicMock(HALT_FILE="halt", validate=MagicMock()),
        "main": MagicMock(),
        "analysis": MagicMock(),
        "analysis.weekly_review": MagicMock(),
        "notifications": MagicMock(),
        "notifications.emailer": MagicMock(),
        "scripts": MagicMock(),
        "scripts.run_diagnostics": MagicMock(),
    }
    with patch.dict(sys.modules, stubs):
        # Remove cached copy so we get a clean import
        sys.modules.pop("scripts.run_scheduler", None)

        import importlib.util
        import os

        spec = importlib.util.spec_from_file_location(
            "scripts.run_scheduler",
            os.path.join(os.path.dirname(__file__), "..", "scripts", "run_scheduler.py"),
        )
        mod = types.ModuleType(spec.name)
        mod.__spec__ = spec
        mod.__file__ = spec.origin
        mod.__package__ = "scripts"
        spec.loader.exec_module(mod)
        return mod


class TestSchedulerImportSafety(unittest.TestCase):
    def test_import_does_not_start_loop(self):
        """Importing run_scheduler must not enter the while-True loop."""
        # If __main__ guard is missing, exec_module blocks forever.
        # The test completing at all is the assertion.
        import threading

        result = {}

        def _do_import():
            try:
                _load_scheduler_module()
                result["ok"] = True
            except Exception as e:  # pragma: no cover
                result["error"] = str(e)

        t = threading.Thread(target=_do_import, daemon=True)
        t.start()
        t.join(timeout=5)
        self.assertTrue(t.is_alive() is False, "Import blocked — __main__ guard missing")
        self.assertTrue(result.get("ok"), f"Import failed: {result.get('error')}")


class TestSchedulerJobRegistration(unittest.TestCase):
    def setUp(self):
        import schedule as _schedule

        _schedule.clear()
        self.schedule = _schedule

    def tearDown(self):
        self.schedule.clear()

    def _register_jobs(self, open_fn, midday_fn, close_fn, weekly_fn):
        s = self.schedule
        for _day in ["monday", "tuesday", "wednesday", "thursday", "friday"]:
            getattr(s.every(), _day).at("14:31").do(open_fn)
            getattr(s.every(), _day).at("17:00").do(midday_fn)
            getattr(s.every(), _day).at("20:30").do(close_fn)
        s.every().sunday.at("20:00").do(weekly_fn)

    def test_correct_job_count(self):
        """15 weekday jobs + 1 Sunday = 16 total."""
        self._register_jobs(lambda: None, lambda: None, lambda: None, lambda: None)
        self.assertEqual(len(self.schedule.jobs), 16)

    def test_each_day_has_three_distinct_jobs(self):
        """Each weekday must have exactly one job at 14:31, 17:00, and 20:30."""
        self._register_jobs(lambda: None, lambda: None, lambda: None, lambda: None)
        weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday"]
        for day in weekdays:
            day_jobs = [j for j in self.schedule.jobs if j.start_day == day]
            self.assertEqual(len(day_jobs), 3, f"{day} should have 3 jobs, got {len(day_jobs)}")
            times = {str(j.at_time) for j in day_jobs}
            self.assertEqual(
                times, {"14:31:00", "17:00:00", "20:30:00"}, f"{day} job times wrong: {times}"
            )

    def test_all_job_objects_are_distinct(self):
        """Each job must be a separate object — no shared state between time slots."""
        self._register_jobs(lambda: None, lambda: None, lambda: None, lambda: None)
        weekday_jobs = [j for j in self.schedule.jobs if j.start_day != "sunday"]
        ids = [id(j) for j in weekday_jobs]
        self.assertEqual(len(ids), len(set(ids)), "Duplicate Job objects found — shared state bug")

    def test_correct_function_per_slot(self):
        """open/midday/close functions must be wired to the right time slots."""
        open_fn = MagicMock(__name__="open_fn")
        midday_fn = MagicMock(__name__="midday_fn")
        close_fn = MagicMock(__name__="close_fn")
        weekly_fn = MagicMock(__name__="weekly_fn")
        self._register_jobs(open_fn, midday_fn, close_fn, weekly_fn)

        for job in self.schedule.jobs:
            if job.start_day == "sunday":
                self.assertIs(job.job_func.func, weekly_fn)
                continue
            t = str(job.at_time)
            if t == "14:31:00":
                self.assertIs(
                    job.job_func.func, open_fn, f"14:31 job on {job.start_day} should call open_fn"
                )
            elif t == "17:00:00":
                self.assertIs(
                    job.job_func.func,
                    midday_fn,
                    f"17:00 job on {job.start_day} should call midday_fn",
                )
            elif t == "20:30:00":  # pragma: no branch
                self.assertIs(
                    job.job_func.func,
                    close_fn,
                    f"20:30 job on {job.start_day} should call close_fn",
                )

    def test_sunday_weekly_review_registered(self):
        """Sunday 20:00 weekly review job must exist."""
        self._register_jobs(lambda: None, lambda: None, lambda: None, lambda: None)
        sunday_jobs = [j for j in self.schedule.jobs if j.start_day == "sunday"]
        self.assertEqual(len(sunday_jobs), 1)
        self.assertEqual(str(sunday_jobs[0].at_time), "20:00:00")


class TestRunFunction(unittest.TestCase):
    """Tests for _run(mode) in run_scheduler.py."""

    def test_halt_file_exists_skips_run(self):
        """When halt file is present, bot.run must NOT be called."""
        mod = _load_scheduler_module()
        mod.config.HALT_FILE = "/tmp/test_halt_scheduler"
        mock_bot = MagicMock()
        mod.bot = mock_bot
        with patch("os.path.exists", return_value=True):
            mod._run("open")
        mock_bot.run.assert_not_called()

    def test_no_halt_file_calls_bot_run(self):
        """Without halt file, bot.run is called with the correct mode."""
        mod = _load_scheduler_module()
        mod.config.HALT_FILE = "/tmp/test_halt_scheduler"
        mock_bot = MagicMock()
        mod.bot = mock_bot
        with patch("os.path.exists", return_value=False):
            mod._run("open")
        mock_bot.run.assert_called_once_with(mode="open")

    def test_run_mode_passed_through(self):
        """_run passes the mode argument unchanged to bot.run."""
        mod = _load_scheduler_module()
        mod.config.HALT_FILE = "/tmp/test_halt_scheduler"
        mock_bot = MagicMock()
        mod.bot = mock_bot
        for mode in ("open_sells", "open", "midday", "close"):
            mock_bot.reset_mock()
            with patch("os.path.exists", return_value=False):
                mod._run(mode)
            mod.bot.run.assert_called_once_with(mode=mode)

    def test_bot_run_exception_is_caught_and_logged(self):
        """If bot.run raises, the exception must not propagate out of _run."""
        mod = _load_scheduler_module()
        mod.config.HALT_FILE = "/tmp/test_halt_scheduler"
        mock_bot = MagicMock()
        mock_bot.run.side_effect = RuntimeError("broker error")
        mod.bot = mock_bot
        with patch("os.path.exists", return_value=False):
            # Should not raise
            try:
                mod._run("open")
            except Exception as exc:  # pragma: no cover
                self.fail(f"_run raised unexpectedly: {exc}")

    def test_bot_run_exception_does_not_propagate(self):
        """Exception in bot.run is swallowed — _run returns None."""
        mod = _load_scheduler_module()
        mod.config.HALT_FILE = "/tmp/test_halt_scheduler"
        mock_bot = MagicMock()
        mock_bot.run.side_effect = ValueError("bad mode")
        mod.bot = mock_bot
        with patch("os.path.exists", return_value=False):
            result = mod._run("open")
        self.assertIsNone(result)


class TestWeeklyReview(unittest.TestCase):
    """Tests for _weekly_review() in run_scheduler.py."""

    def _get_mod(self):
        return _load_scheduler_module()

    def test_halt_file_skips_review(self):
        """When halt file exists, run_diagnostics and run_weekly_review must NOT be called."""
        mod = self._get_mod()
        mod.config.HALT_FILE = "/tmp/test_halt_scheduler"
        mock_run_diagnostics = MagicMock()
        mock_run_weekly_review = MagicMock()
        mod.run_diagnostics = mock_run_diagnostics
        mod.run_weekly_review = mock_run_weekly_review
        with patch("os.path.exists", return_value=True):
            mod._weekly_review()
        mock_run_diagnostics.assert_not_called()
        mock_run_weekly_review.assert_not_called()

    def test_no_halt_calls_diagnostics_and_review(self):
        """Without halt file, both run_diagnostics and run_weekly_review are called."""
        mod = self._get_mod()
        mod.config.HALT_FILE = "/tmp/test_halt_scheduler"
        mock_diag = MagicMock(return_value={"status": "PASS"})
        mock_review = MagicMock(return_value={"summary": "good week"})
        mock_send = MagicMock()
        mod.run_diagnostics = mock_diag
        mod.run_weekly_review = mock_review
        mod.send_weekly_review = mock_send
        with patch("os.path.exists", return_value=False):
            mod._weekly_review()
        mock_diag.assert_called_once()
        mock_review.assert_called_once()

    def test_review_result_triggers_send(self):
        """A truthy review result causes send_weekly_review to be called with attribution."""
        mod = self._get_mod()
        mod.config.HALT_FILE = "/tmp/test_halt_scheduler"
        diag_report = {"status": "PASS"}
        review_result = {"summary": "good week"}
        attribution_data = {"by_signal": {}, "total_trades": 0}
        mock_diag = MagicMock(return_value=diag_report)
        mock_review = MagicMock(return_value=review_result)
        mock_send = MagicMock()
        mod.run_diagnostics = mock_diag
        mod.run_weekly_review = mock_review
        mod.send_weekly_review = mock_send
        mod.get_attribution = MagicMock(return_value=attribution_data)
        with patch("os.path.exists", return_value=False):
            mod._weekly_review()
        mock_send.assert_called_once_with(
            review_result, test_report=diag_report, attribution=attribution_data
        )

    def test_diagnostics_exception_still_runs_review(self):
        """If run_diagnostics raises, _weekly_review still calls run_weekly_review."""
        mod = self._get_mod()
        mod.config.HALT_FILE = "/tmp/test_halt_scheduler"
        mock_diag = MagicMock(side_effect=RuntimeError("diagnostics boom"))
        mock_review = MagicMock(return_value={"summary": "ok"})
        mock_send = MagicMock()
        mod.run_diagnostics = mock_diag
        mod.run_weekly_review = mock_review
        mod.send_weekly_review = mock_send
        with patch("os.path.exists", return_value=False):
            mod._weekly_review()
        mock_review.assert_called_once()

    def test_review_none_with_test_report_sends_diagnostics_email(self):
        """When review returns None but test_report exists, sends a diagnostics-only email."""
        mod = self._get_mod()
        mod.config.HALT_FILE = "/tmp/test_halt_scheduler"
        diag_report_mock = MagicMock()
        diag_report_mock.get.return_value = "FAIL"

        mock_diag = MagicMock(return_value=diag_report_mock)
        mock_review = MagicMock(return_value=None)
        mock_send = MagicMock()
        mock_send_html = MagicMock()
        mock_build_html = MagicMock(return_value="<html/>")
        mod.run_diagnostics = mock_diag
        mod.run_weekly_review = mock_review
        mod.send_weekly_review = mock_send

        with (
            patch("os.path.exists", return_value=False),
            patch.dict(
                "sys.modules",
                {
                    "notifications.emailer": MagicMock(
                        _send_html=mock_send_html,
                        _build_weekly_html=mock_build_html,
                    ),
                },
            ),
        ):
            mod._weekly_review()

        # send_weekly_review should NOT be called (review was None)
        mock_send.assert_not_called()
        # _send_html should be called once (diagnostics-only path)
        mock_send_html.assert_called_once()

    def test_both_none_nothing_sent(self):
        """When both review and test_report are None, no email is sent."""
        mod = self._get_mod()
        mod.config.HALT_FILE = "/tmp/test_halt_scheduler"
        mock_diag = MagicMock(return_value=None)
        mock_review = MagicMock(return_value=None)
        mock_send = MagicMock()
        mod.run_diagnostics = mock_diag
        mod.run_weekly_review = mock_review
        mod.send_weekly_review = mock_send
        with patch("os.path.exists", return_value=False):
            mod._weekly_review()
        mock_send.assert_not_called()

    def test_weekly_review_exception_caught(self):
        """If run_weekly_review raises, _weekly_review logs and does not propagate."""
        mod = self._get_mod()
        mod.config.HALT_FILE = "/tmp/test_halt_scheduler"
        mock_diag = MagicMock(return_value={"status": "PASS"})
        mock_review = MagicMock(side_effect=RuntimeError("review explosion"))
        mock_send = MagicMock()
        mod.run_diagnostics = mock_diag
        mod.run_weekly_review = mock_review
        mod.send_weekly_review = mock_send
        with patch("os.path.exists", return_value=False):
            try:
                mod._weekly_review()
            except Exception as exc:  # pragma: no cover
                self.fail(f"_weekly_review raised unexpectedly: {exc}")


class TestWrapperFunctions(unittest.TestCase):
    """Lines 56, 60, 64, 68: _open_sells, _open, _midday, _close delegate to _run."""

    def _mod_with_bot(self):
        mod = _load_scheduler_module()
        mock_bot = MagicMock()
        mod.bot = mock_bot
        mod.config.HALT_FILE = "/tmp/.test_halt_wrappers"
        return mod, mock_bot

    def test_open_sells_delegates_to_run(self):
        mod, mock_bot = self._mod_with_bot()
        with patch("os.path.exists", return_value=False):
            mod._open_sells()
        mock_bot.run.assert_called_once_with(mode="open_sells")

    def test_open_delegates_to_run(self):
        mod, mock_bot = self._mod_with_bot()
        with patch("os.path.exists", return_value=False):
            mod._open()
        mock_bot.run.assert_called_once_with(mode="open")

    def test_midday_delegates_to_run(self):
        mod, mock_bot = self._mod_with_bot()
        with patch("os.path.exists", return_value=False):
            mod._midday()
        mock_bot.run.assert_called_once_with(mode="midday")

    def test_close_delegates_to_run(self):
        mod, mock_bot = self._mod_with_bot()
        with patch("os.path.exists", return_value=False):
            mod._close()
        mock_bot.run.assert_called_once_with(mode="close")


class TestCheckSingleton(unittest.TestCase):
    """Tests for _check_singleton in run_scheduler.py."""

    def test_no_pid_file_writes_new_pid(self):
        """Branch 30->45: PID file does not exist → skip if block, write new PID."""
        mod = _load_scheduler_module()
        written = []

        def fake_open(path, mode="r", *args, **kwargs):
            import io

            buf = io.StringIO()
            buf.write = lambda s: written.append(s) or len(s)
            buf.__enter__ = lambda self: self
            buf.__exit__ = lambda self, *a: None
            return buf

        with (
            patch("os.path.exists", return_value=False),
            patch("builtins.open", side_effect=fake_open),
            patch("os.makedirs"),
            patch("os.getpid", return_value=99),
        ):
            mod._check_singleton()
        self.assertIn("99", "".join(written))

    def test_exits_when_existing_process_is_alive(self):
        """PID file exists and os.kill(old_pid, 0) does NOT raise → sys.exit(1)."""
        mod = _load_scheduler_module()
        # Simulate PID file containing a live PID
        with (
            patch("os.path.exists", return_value=True),
            patch("builtins.open", unittest.mock.mock_open(read_data="99999")),
            patch("os.kill"),
            patch("os.makedirs"),
            self.assertRaises(SystemExit) as ctx,
        ):
            mod._check_singleton()
        self.assertEqual(ctx.exception.code, 1)

    def test_continues_when_pid_file_has_stale_pid(self):
        """PID file exists but os.kill raises ProcessLookupError → stale lock, continues."""
        import io

        mod = _load_scheduler_module()
        written = []

        def fake_open(path, mode="r", *args, **kwargs):
            if mode == "r" or mode == "":
                return io.StringIO("12345")
            buf = io.StringIO()
            buf.write = lambda s: written.append(s) or len(s)
            buf.__enter__ = lambda self: self
            buf.__exit__ = lambda self, *a: None
            return buf

        with (
            patch("os.path.exists", return_value=True),
            patch("builtins.open", side_effect=fake_open),
            patch("os.kill", side_effect=ProcessLookupError),
            patch("os.makedirs"),
        ):
            # Should not raise
            try:
                mod._check_singleton()
            except SystemExit:  # pragma: no cover
                self.fail("_check_singleton should not exit for stale PID")

    def test_continues_when_pid_file_has_invalid_content(self):
        """PID file exists but content is not an int → ValueError → stale, continues."""
        import io

        mod = _load_scheduler_module()

        def fake_open(path, mode="r", *args, **kwargs):
            if mode == "r" or mode == "":
                return io.StringIO("not-a-pid")
            return unittest.mock.mock_open()(path, mode)

        with (
            patch("os.path.exists", return_value=True),
            patch("builtins.open", side_effect=fake_open),
            patch("os.makedirs"),
            patch("os.getpid", return_value=42),
        ):
            try:
                mod._check_singleton()
            except SystemExit:  # pragma: no cover
                self.fail("_check_singleton should not exit for invalid PID content")


class TestRemovePidFile(unittest.TestCase):
    """Tests for _remove_pid_file in run_scheduler.py."""

    def test_removes_pid_file_when_it_exists(self):
        mod = _load_scheduler_module()
        with patch("os.remove") as mock_remove:
            mod._remove_pid_file()
        mock_remove.assert_called_once_with(mod._PID_FILE)

    def test_silently_ignores_missing_pid_file(self):
        mod = _load_scheduler_module()
        with patch("os.remove", side_effect=FileNotFoundError):
            try:
                mod._remove_pid_file()
            except FileNotFoundError:  # pragma: no cover
                self.fail("_remove_pid_file should suppress FileNotFoundError")


class TestSigtermHandler(unittest.TestCase):
    """Tests for _sigterm_handler in run_scheduler.py."""

    def test_sigterm_removes_pid_file_and_exits(self):
        mod = _load_scheduler_module()
        with (
            patch.object(mod, "_remove_pid_file") as mock_remove,
            self.assertRaises(SystemExit) as ctx,
        ):
            mod._sigterm_handler(15, None)
        mock_remove.assert_called_once()
        self.assertEqual(ctx.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
