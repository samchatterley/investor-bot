import unittest

from experiment.feed_health import (
    DEGRADED,
    EMPTY,
    ERROR,
    OK,
    STALE,
    FeedResult,
    check_feed,
    format_report,
    run_health_checks,
    summarise,
)


class TestCheckFeed(unittest.TestCase):
    def test_exception_is_error(self):
        def boom():
            raise RuntimeError("network down")

        r = check_feed("x", boom)
        self.assertEqual(r.status, ERROR)
        self.assertIn("RuntimeError", r.detail)
        self.assertFalse(r.healthy)

    def test_none_is_empty(self):
        r = check_feed("x", lambda: None)
        self.assertEqual(r.status, EMPTY)

    def test_empty_collection_is_empty(self):
        self.assertEqual(check_feed("a", lambda: []).status, EMPTY)
        self.assertEqual(check_feed("b", lambda: {}).status, EMPTY)
        self.assertEqual(check_feed("c", lambda: "").status, EMPTY)

    def test_nonempty_without_assess_is_ok(self):
        r = check_feed("x", lambda: {"AAPL": 1})
        self.assertEqual(r.status, OK)
        self.assertTrue(r.healthy)

    def test_assess_can_flag_degraded(self):
        r = check_feed("x", lambda: 1.0, assess=lambda v: (DEGRADED, "all NaN"))
        self.assertEqual(r.status, DEGRADED)
        self.assertEqual(r.detail, "all NaN")

    def test_assess_ok_passes_through(self):
        r = check_feed("x", lambda: 42, assess=lambda v: (OK, "vix=42"))
        self.assertEqual(r.status, OK)
        self.assertEqual(r.detail, "vix=42")

    def test_bool_value_reaches_assess(self):
        # is_available()-style probe: a bool is neither None nor an empty collection
        r = check_feed("finbert", lambda: False, assess=lambda v: (OK if v else DEGRADED, ""))
        self.assertEqual(r.status, DEGRADED)


class TestRunAndSummarise(unittest.TestCase):
    def _probes(self):
        return [
            ("good", lambda: {"x": 1}, None),
            ("bad", lambda: None, None),
            ("err", lambda: (_ for _ in ()).throw(ValueError("x")), None),
        ]

    def test_run_isolates_failures(self):
        results = run_health_checks(self._probes())
        self.assertEqual([r.status for r in results], [OK, EMPTY, ERROR])

    def test_summarise_counts(self):
        ok, bad, all_green = summarise(run_health_checks(self._probes()))
        self.assertEqual(ok, 1)
        self.assertEqual(bad, 2)
        self.assertFalse(all_green)

    def test_summarise_all_green(self):
        results = [FeedResult("a", OK), FeedResult("b", OK)]
        ok, bad, all_green = summarise(results)
        self.assertEqual((ok, bad), (2, 0))
        self.assertTrue(all_green)


class TestFormatReport(unittest.TestCase):
    def test_lists_unhealthy_first_and_verdict(self):
        results = [FeedResult("good", OK, "fresh"), FeedResult("bad", STALE, "30d old")]
        report = format_report(results)
        self.assertIn("STALE", report)
        self.assertIn("need attention", report)
        # unhealthy sorts before healthy
        self.assertLess(report.index("bad"), report.index("good"))

    def test_all_green_verdict(self):
        report = format_report([FeedResult("a", OK)])
        self.assertIn("ALL GREEN", report)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
