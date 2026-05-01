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
            except Exception as e:
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
            self.assertEqual(times, {"14:31:00", "17:00:00", "20:30:00"},
                             f"{day} job times wrong: {times}")

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
                self.assertIs(job.job_func.func, open_fn,
                              f"14:31 job on {job.start_day} should call open_fn")
            elif t == "17:00:00":
                self.assertIs(job.job_func.func, midday_fn,
                              f"17:00 job on {job.start_day} should call midday_fn")
            elif t == "20:30:00":
                self.assertIs(job.job_func.func, close_fn,
                              f"20:30 job on {job.start_day} should call close_fn")

    def test_sunday_weekly_review_registered(self):
        """Sunday 20:00 weekly review job must exist."""
        self._register_jobs(lambda: None, lambda: None, lambda: None, lambda: None)
        sunday_jobs = [j for j in self.schedule.jobs if j.start_day == "sunday"]
        self.assertEqual(len(sunday_jobs), 1)
        self.assertEqual(str(sunday_jobs[0].at_time), "20:00:00")


if __name__ == "__main__":
    unittest.main()
