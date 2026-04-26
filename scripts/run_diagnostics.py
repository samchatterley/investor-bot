"""
Runs the full unit test suite and returns a structured diagnostic report.
Called from scripts/run_scheduler.py as part of the Sunday evening job.
"""
import json
import logging
import os
import sys
import time
import unittest
from datetime import datetime, timezone

# Ensure the project root is on the path when this script is run directly
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPTS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_TESTS_DIR = os.path.join(_ROOT, "tests")

logger = logging.getLogger(__name__)


class _SilentResult(unittest.TestResult):
    """Collects results without printing to stdout."""
    pass


def run_diagnostics() -> dict:
    """
    Discover and run all tests in tests/. Returns a report dict with counts,
    duration, status (PASS/FAIL), and details of any failures.
    """
    loader = unittest.TestLoader()
    suite = loader.discover(_TESTS_DIR, pattern="test_*.py")

    result = _SilentResult()
    start = time.monotonic()
    suite.run(result)
    duration = round(time.monotonic() - start, 2)

    failures = []
    for test, traceback in result.failures + result.errors:
        failures.append({
            "test": str(test),
            "message": traceback.strip().split("\n")[-1],
        })

    passed = result.testsRun - len(result.failures) - len(result.errors)
    status = "PASS" if not failures else "FAIL"

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total": result.testsRun,
        "passed": passed,
        "failed": len(result.failures),
        "errors": len(result.errors),
        "duration_seconds": duration,
        "status": status,
        "failures": failures,
    }

    logger.info(f"Diagnostics: {passed}/{result.testsRun} passed | {status} | {duration:.1f}s")
    if failures:
        for f in failures:
            logger.warning(f"  FAIL: {f['test']} — {f['message']}")

    _save_report(report)
    return report


def _save_report(report: dict):
    try:
        from config import LOG_DIR
        os.makedirs(LOG_DIR, exist_ok=True)
        today = datetime.now(timezone.utc).date().isoformat()
        path = os.path.join(LOG_DIR, f"test_report_{today}.json")
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Test report saved to {path}")
    except Exception as e:
        logger.warning(f"Could not save test report: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
    report = run_diagnostics()
    print(f"\n{'='*40}")
    print(f"  {report['status']}  —  {report['passed']}/{report['total']} tests passed  ({report['duration_seconds']}s)")
    if report["failures"]:
        print(f"\n  Failures:")
        for f in report["failures"]:
            print(f"    • {f['test']}")
            print(f"      {f['message']}")
    print(f"{'='*40}\n")
    sys.exit(0 if report["status"] == "PASS" else 1)
