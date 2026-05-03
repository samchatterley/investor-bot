"""
Runs the trading bot four times per trading day (all times America/New_York):
  09:31 ET — open sells       (earnings exits, AI sell decisions, no new buys)
  10:00 ET — open buys        (fresh AI buy analysis after open noise settles)
  12:00 ET — midday check     (partial exits, no new buys)
  15:30 ET — pre-close check  (final position review)

Splitting open into two windows avoids buying into the noisy first 30 minutes
of the session while still executing time-sensitive exits at the bell.
Times are scheduled in NYSE timezone directly — no BST/GMT conversion needed.
Leave this process running in a terminal or tmux session.
"""

import os
import sys

# Ensure the project root is on the path when this script is run directly
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPTS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)  # pragma: no cover

import logging  # noqa: E402
import time  # noqa: E402

import schedule  # noqa: E402

import config  # noqa: E402

config.validate()
import main as bot  # noqa: E402
from analysis.weekly_review import run_weekly_review  # noqa: E402
from notifications.emailer import send_weekly_review  # noqa: E402
from scripts.run_diagnostics import run_diagnostics  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _run(mode: str):
    if os.path.exists(config.HALT_FILE):
        logger.critical(f"Trading HALTED — skipping {mode} run. Run: python main.py --clear-halt")
        return
    logger.info(f"Scheduled trigger: mode={mode}")
    try:
        bot.run(mode=mode)
    except Exception as e:
        logger.error(f"Run failed ({mode}): {e}", exc_info=True)


def _open_sells():
    _run("open_sells")


def _open():
    _run("open")


def _midday():
    _run("midday")


def _close():
    _run("close")


def _weekly_review():
    if os.path.exists(config.HALT_FILE):
        logger.info("Trading HALTED — skipping weekly review.")
        return

    logger.info("Running system diagnostics...")
    test_report = None
    try:
        test_report = run_diagnostics()
    except Exception as e:
        logger.error(f"Diagnostics failed: {e}", exc_info=True)

    logger.info("Running weekly self-review...")
    try:
        review = run_weekly_review()
        if review:
            send_weekly_review(review, test_report=test_report)
        elif test_report:
            from datetime import date as _date

            from notifications.emailer import _build_weekly_html, _send_html

            review_stub = {
                "week_summary": "No trade history available for this week.",
                "what_worked": [],
                "what_didnt": [],
                "lessons": [],
                "proposed_changes": [],
            }
            _send_html(
                subject=f"Weekly Diagnostics {_date.today().isoformat()} · tests {test_report.get('status', '')}",
                html_fn=lambda name: _build_weekly_html(review_stub, name, test_report),
            )
    except Exception as e:
        logger.error(f"Weekly review failed: {e}", exc_info=True)


if __name__ == "__main__":
    _ET = "America/New_York"
    for _day in ["monday", "tuesday", "wednesday", "thursday", "friday"]:
        getattr(schedule.every(), _day).at("09:31", _ET).do(_open_sells)
        getattr(schedule.every(), _day).at("10:00", _ET).do(_open)
        getattr(schedule.every(), _day).at("12:00", _ET).do(_midday)
        getattr(schedule.every(), _day).at("15:30", _ET).do(_close)

    schedule.every().sunday.at("15:00", _ET).do(_weekly_review)

    logger.info(
        "Scheduler running — Mon–Fri at 09:31 (sells) / 10:00 (buys) / 12:00 / 15:30 ET (America/New_York)"
    )
    logger.info("Ctrl+C to stop.")

    while True:
        schedule.run_pending()
        time.sleep(30)
