"""
Runs the trading bot three times per trading day:
  09:31 ET — full open cycle  (new buys + position management)
  12:00 ET — midday check     (partial exits, no new buys)
  15:30 ET — pre-close check  (final position review)

UK times (BST, UTC+1):
  14:31 / 17:00 / 20:30

Leave this process running in a terminal or tmux session.
Alternatively, use cron (see README or inline comments below).
"""

import os
import schedule
import time
import logging
import config
import main as bot
from analysis.weekly_review import run_weekly_review
from notifications.emailer import send_weekly_review

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


def _open():   _run("open")
def _midday(): _run("midday")
def _close():  _run("close")


def _weekly_review():
    if os.path.exists(config.HALT_FILE):
        logger.info("Trading HALTED — skipping weekly review.")
        return
    logger.info("Running weekly self-review...")
    try:
        review = run_weekly_review()
        if review:
            send_weekly_review(review)
    except Exception as e:
        logger.error(f"Weekly review failed: {e}", exc_info=True)


for day in [schedule.every().monday, schedule.every().tuesday, schedule.every().wednesday,
            schedule.every().thursday, schedule.every().friday]:
    day.at("14:31").do(_open)    # 09:31 ET  (BST)
    day.at("17:00").do(_midday)  # 12:00 ET  (BST)
    day.at("20:30").do(_close)   # 15:30 ET  (BST)

schedule.every().sunday.at("20:00").do(_weekly_review)  # Sunday 20:00 BST

logger.info("Scheduler running — Mon–Fri at 14:31 / 17:00 / 20:30 UK time (BST)")
logger.info("Ctrl+C to stop.")

while True:
    schedule.run_pending()
    time.sleep(30)
