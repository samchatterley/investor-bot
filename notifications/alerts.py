import smtplib
import logging
from email.mime.text import MIMEText
from config import EMAIL_FROM, EMAIL_TO, EMAIL_APP_PASSWORD

logger = logging.getLogger(__name__)


def _send(subject: str, body: str):
    """Send an emergency alert to the owner only. Silently skips if email is not configured."""
    if not EMAIL_FROM or not EMAIL_APP_PASSWORD or not EMAIL_TO:
        return
    try:
        msg = MIMEText(body, "plain")
        msg["Subject"] = subject
        msg["From"] = EMAIL_FROM
        msg["To"] = EMAIL_TO
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_FROM, EMAIL_APP_PASSWORD)
            server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
    except Exception as e:
        logger.warning(f"Alert email failed: {e}")


# ── Emergency alerts only (owner-only, sent immediately) ─────────────────────

def alert_circuit_breaker(drawdown_pct: float):
    _send(
        "[BOT] Circuit breaker activated",
        f"Portfolio drawdown: {drawdown_pct:.1f}%. All new buys halted for today.",
    )


def alert_daily_loss(loss_pct: float):
    _send(
        "[BOT] Daily loss limit hit",
        f"Down {abs(loss_pct):.1f}% today. All positions being closed.",
    )


def alert_error(context: str, error: str):
    _send(f"[BOT] Error in {context}", f"Error: {error}")
