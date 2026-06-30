"""
Runs the trading bot four times per trading day (all times America/New_York):
  09:31 ET — open sells       (earnings exits, AI sell decisions, no new buys)
  10:00 ET — open buys        (fresh AI buy analysis after open noise settles)
  12:00 ET — midday           (partial exits + new buys on confirmed signals)
  15:30 ET — pre-close check  (final position review, no new buys)

Splitting open into two windows avoids buying into the noisy first 30 minutes
of the session while still executing time-sensitive exits at the bell.
Midday buys capture intraday signal confirmation (VWAP reclaim, ORB follow-
through) that isn't visible at 10:00; the daily notional budget and position
slot cap apply across both open and midday buy phases.
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

# ── Single-instance guard ─────────────────────────────────────────────────────
# Prevent duplicate schedulers accumulating across tmux sessions / restarts.
# Uses a PID file so stale locks from crashes are auto-cleared on next start.
_PID_FILE = os.path.join(_ROOT, "logs", "scheduler.pid")


def _check_singleton() -> None:
    if os.path.exists(_PID_FILE):
        try:
            with open(_PID_FILE) as _f:
                _old_pid = int(_f.read().strip())
            # Check if that process is still alive
            os.kill(_old_pid, 0)
            print(
                f"ERROR: scheduler already running (PID {_old_pid}). "
                f"Kill it first or remove {_PID_FILE}.",
                file=sys.stderr,
            )
            sys.exit(1)
        except (ProcessLookupError, ValueError):
            pass  # stale PID file — previous run crashed; safe to continue

    os.makedirs(os.path.dirname(_PID_FILE), exist_ok=True)
    with open(_PID_FILE, "w") as _f:
        _f.write(str(os.getpid()))


def _remove_pid_file() -> None:
    import contextlib

    with contextlib.suppress(FileNotFoundError):
        os.remove(_PID_FILE)


import atexit  # noqa: E402
import logging  # noqa: E402
import signal  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

import schedule  # noqa: E402

import config  # noqa: E402

config.validate()
import main as bot  # noqa: E402
from analysis.performance import get_attribution  # noqa: E402
from analysis.weekly_review import run_weekly_review  # noqa: E402
from data.analyst_revisions import prefetch_analyst_revisions  # noqa: E402
from data.av_sentiment import prefetch_av_sentiment  # noqa: E402
from data.earnings_surprise import prefetch_earnings_data  # noqa: E402
from data.edgar_client import prefetch_edgar_data  # noqa: E402
from data.insider_feed import prefetch_insider_activity  # noqa: E402
from data.macro_data import get_macro_snapshot  # noqa: E402
from data.market_data import (  # noqa: E402
    migrate_bulk_caches_to_subdir,
    prefetch_market_data,
)
from data.sector_data import build_sector_map  # noqa: E402
from data.sentiment_client import get_fear_greed_composite  # noqa: E402
from data.short_interest import prefetch_short_interest  # noqa: E402
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


def _prefetch():
    if os.path.exists(config.HALT_FILE):
        return
    logger.info("Pre-market data prefetch starting...")
    try:
        prefetch_market_data(list(config.STOCK_UNIVERSE))
    except Exception as e:
        logger.error(f"Prefetch failed (non-fatal): {e}", exc_info=True)
    try:
        # Populate the symbol→sector cache (audit F7). Without this the cache stays empty and
        # get_sector falls back to a 53-symbol legacy map → "Unknown" for ~all of the universe,
        # which silently no-ops the sector-momentum long gate. Incremental: only fetches symbols
        # not already cached, so it's a one-time full build then cheap daily top-ups.
        build_sector_map()
    except Exception as e:
        logger.error(f"Sector map prefetch failed (non-fatal): {e}", exc_info=True)
    try:
        prefetch_insider_activity()
    except Exception as e:
        logger.error(f"Insider prefetch failed (non-fatal): {e}", exc_info=True)
    try:
        prefetch_earnings_data()
    except Exception as e:
        logger.error(f"Earnings prefetch failed (non-fatal): {e}", exc_info=True)
    try:
        prefetch_short_interest()
    except Exception as e:
        logger.error(f"Short interest prefetch failed (non-fatal): {e}", exc_info=True)
    try:
        prefetch_av_sentiment()
    except Exception as e:
        logger.error(f"AV sentiment prefetch failed (non-fatal): {e}", exc_info=True)
    try:
        prefetch_edgar_data()
    except Exception as e:
        logger.error(f"EDGAR prefetch failed (non-fatal): {e}", exc_info=True)
    try:
        # Warm analyst rating-shift + EPS-revision data (feeds analyst_downgrade_signal and
        # eps_revision_down_short on the short side). Cached daily; cheap reads during the session.
        prefetch_analyst_revisions()
    except Exception as e:
        logger.error(f"Analyst revisions prefetch failed (non-fatal): {e}", exc_info=True)
    try:
        get_macro_snapshot()
    except Exception as e:
        logger.error(f"Macro data prefetch failed (non-fatal): {e}", exc_info=True)
    try:
        get_fear_greed_composite()
    except Exception as e:
        logger.error(f"Fear & Greed prefetch failed (non-fatal): {e}", exc_info=True)


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
        attribution = get_attribution(90)
        review = run_weekly_review()
        if review:
            send_weekly_review(review, test_report=test_report, attribution=attribution)
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


def _backfill_outcomes():
    """Score forward outcomes for logged experiment observations (horizons fill in as they close).

    Runs after the close so the day's now-matured horizons get scored into
    logs/experiment_scored.jsonl. Without this the observations accumulate with no outcomes and the
    experiment can never progress. Fail-safe — instrumentation must never block the scheduler.
    """
    if os.path.exists(config.HALT_FILE):
        return
    logger.info("Backfilling experiment outcomes...")
    try:
        from scripts.backfill_outcomes import main as _run_backfill

        _run_backfill()
    except Exception as e:
        logger.error(f"Outcome backfill failed (non-fatal): {e}", exc_info=True)


def _startup_prefetch() -> None:
    """Warm caches immediately on startup in case the 07:00 ET prefetch was missed.

    Runs _prefetch() in a daemon thread so the scheduler loop is not blocked.
    No-op on weekends.  Each prefetch function returns instantly when the
    same-day cache is already warm, so this is a fast no-op on normal days.
    """
    if config.today_et().weekday() >= 5:  # Saturday=5, Sunday=6
        return
    logger.info("Startup: launching background cache warm (no-op if 07:00 prefetch already ran)...")
    threading.Thread(target=_prefetch, daemon=True, name="startup-prefetch").start()


def _sigterm_handler(_signum, _frame):
    _remove_pid_file()
    sys.exit(0)


if __name__ == "__main__":  # pragma: no cover
    _check_singleton()
    atexit.register(_remove_pid_file)
    signal.signal(signal.SIGTERM, _sigterm_handler)
    migrate_bulk_caches_to_subdir()  # adopt logs/market_data/ foldering for legacy installs
    _startup_prefetch()

    # Append to log file so history survives launchd restarts
    _log_path = os.path.join(_ROOT, "logs", "scheduler.log")
    _fh = logging.FileHandler(_log_path, mode="a", encoding="utf-8")
    _fh.setFormatter(
        logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
    )
    logging.getLogger().addHandler(_fh)
    _ET = "America/New_York"
    for _day in ["monday", "tuesday", "wednesday", "thursday", "friday"]:
        getattr(schedule.every(), _day).at("07:00", _ET).do(_prefetch)
        getattr(schedule.every(), _day).at("09:31", _ET).do(_open_sells)
        getattr(schedule.every(), _day).at("10:00", _ET).do(_open)
        getattr(schedule.every(), _day).at("12:00", _ET).do(_midday)
        getattr(schedule.every(), _day).at("15:30", _ET).do(_close)
        getattr(schedule.every(), _day).at("16:15", _ET).do(_backfill_outcomes)

    schedule.every().sunday.at("15:30", _ET).do(_weekly_review)

    logger.info(
        "Scheduler running — Mon–Fri at 07:00 (prefetch) / 09:31 (sells) / 10:00 (buys) / 12:00 / "
        "15:30 (close) / 16:15 (outcome backfill) ET (America/New_York)"
    )
    logger.info("Ctrl+C to stop.")

    while True:
        try:
            schedule.run_pending()
        except Exception as e:
            logger.error(f"Unexpected scheduler loop error: {e}", exc_info=True)
        time.sleep(30)
