"""
Runs the full pytest suite in a subprocess and returns a structured diagnostic report.
Called from scripts/run_scheduler.py as part of the Sunday evening job.

Uses pytest (the CI source of truth) in a SUBPROCESS — deliberately NOT unittest.discover
in-process. The old in-process approach had two bugs: (1) it ran tests inside the scheduler
process, so test logging leaked into scheduler.log, and (2) unittest.discover lacks pytest's
conftest/monkeypatch fixture isolation, so module-global patches leaked between tests and produced
false failures (the weekly email reported "failing tests" that pass under pytest/CI). A subprocess
keeps all test side effects isolated and makes the report match CI exactly.
"""

import contextlib
import json
import logging
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from xml.etree import ElementTree

# Ensure the project root is on the path when this script is run directly
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPTS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)  # pragma: no cover

_TESTS_DIR = os.path.join(_ROOT, "tests")

# Hard ceiling so a hung/runaway suite can't block the sequential scheduler indefinitely.
_PYTEST_TIMEOUT_SECONDS = 1800  # suite runs in ~11 min; 30 min is generous headroom

logger = logging.getLogger(__name__)


def _parse_junit(xml_path: str) -> dict:
    """Parse a pytest JUnit XML file into total/failed/errors/skipped counts + a failures list."""
    root = ElementTree.parse(xml_path).getroot()
    # The root is <testsuites> wrapping one or more <testsuite>, or a bare <testsuite>.
    suites = root.findall("testsuite") or [root]
    total = failed = errors = skipped = 0
    failures: list[dict] = []
    for suite in suites:
        total += int(suite.get("tests", 0))
        failed += int(suite.get("failures", 0))
        errors += int(suite.get("errors", 0))
        skipped += int(suite.get("skipped", 0))
        for case in suite.findall("testcase"):
            problem = case.find("failure")
            if problem is None:
                problem = case.find("error")
            if problem is None:
                continue
            name = f"{case.get('classname', '')}.{case.get('name', '')}".strip(".")
            message = (problem.get("message") or problem.text or "").strip()
            failures.append({"test": name, "message": message.split("\n")[-1][:300]})
    return {
        "total": total,
        "failed": failed,
        "errors": errors,
        "skipped": skipped,
        "failures": failures,
    }


def run_diagnostics() -> dict:
    """
    Run the full pytest suite in a subprocess. Returns a report dict with counts,
    duration, status (PASS/FAIL), and details of any failures.
    """
    xml_path = os.path.join(_ROOT, f".diagnostics_junit_{os.getpid()}.xml")
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        _TESTS_DIR,
        "-o",
        "addopts=",  # drop coverage/strict opts — this is a fast pass/fail health check
        "-p",
        "no:cacheprovider",
        "-q",
        "--no-header",
        f"--junit-xml={xml_path}",
    ]

    start = time.monotonic()
    timed_out = False
    try:
        subprocess.run(
            cmd, cwd=_ROOT, capture_output=True, text=True, timeout=_PYTEST_TIMEOUT_SECONDS
        )
    except subprocess.TimeoutExpired:
        timed_out = True
    duration = round(time.monotonic() - start, 2)

    if timed_out:
        parsed = {
            "total": 0,
            "failed": 0,
            "errors": 1,
            "skipped": 0,
            "failures": [
                {"test": "pytest", "message": f"suite timed out after {_PYTEST_TIMEOUT_SECONDS}s"}
            ],
        }
    else:
        try:
            parsed = _parse_junit(xml_path)
        except (FileNotFoundError, ElementTree.ParseError) as e:
            parsed = {
                "total": 0,
                "failed": 0,
                "errors": 1,
                "skipped": 0,
                "failures": [{"test": "pytest", "message": f"could not parse results: {e}"}],
            }
        finally:
            with contextlib.suppress(OSError):
                os.remove(xml_path)

    passed = parsed["total"] - parsed["failed"] - parsed["errors"] - parsed["skipped"]
    status = "FAIL" if (parsed["failed"] or parsed["errors"] or timed_out) else "PASS"

    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "total": parsed["total"],
        "passed": passed,
        "failed": parsed["failed"],
        "errors": parsed["errors"],
        "skipped": parsed["skipped"],
        "duration_seconds": duration,
        "status": status,
        "failures": parsed["failures"],
    }

    logger.info(f"Diagnostics: {passed}/{parsed['total']} passed | {status} | {duration:.1f}s")
    if parsed["failures"]:
        for f in parsed["failures"]:
            logger.warning(f"  FAIL: {f['test']} — {f['message']}")

    _save_report(report)
    return report


def _save_report(report: dict):
    try:
        from config import LOG_DIR

        os.makedirs(LOG_DIR, exist_ok=True)
        today = datetime.now(UTC).date().isoformat()
        path = os.path.join(LOG_DIR, f"test_report_{today}.json")
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Test report saved to {path}")
    except Exception as e:
        logger.warning(f"Could not save test report: {e}")


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
    report = run_diagnostics()
    print(f"\n{'=' * 40}")
    print(
        f"  {report['status']}  —  {report['passed']}/{report['total']} tests passed  ({report['duration_seconds']}s)"
    )
    if report["failures"]:
        print("\n  Failures:")
        for f in report["failures"]:
            print(f"    • {f['test']}")
            print(f"      {f['message']}")
    print(f"{'=' * 40}\n")
    sys.exit(0 if report["status"] == "PASS" else 1)
