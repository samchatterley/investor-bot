import logging
from datetime import date, timedelta
from typing import Optional
import yfinance as yf
import pandas as pd

logger = logging.getLogger(__name__)


def get_next_earnings_date(symbol: str) -> Optional[date]:
    """Return the next earnings date for a symbol, or None if unavailable."""
    try:
        ticker = yf.Ticker(symbol)
        cal = ticker.calendar

        if cal is None:
            return None

        # yfinance returns different formats depending on version
        if isinstance(cal, dict):
            raw = cal.get("Earnings Date", [None])[0]
        elif isinstance(cal, pd.DataFrame):
            if "Earnings Date" in cal.columns:
                raw = cal["Earnings Date"].iloc[0]
            elif not cal.empty:
                raw = cal.iloc[0, 0]
            else:
                return None
        else:
            return None

        if raw is None:
            return None
        ts = pd.Timestamp(raw)
        return ts.date()

    except Exception as e:
        logger.debug(f"Earnings lookup failed for {symbol}: {e}")
        return None


def days_until_earnings(symbol: str) -> Optional[int]:
    """Return how many calendar days until next earnings, or None."""
    ed = get_next_earnings_date(symbol)
    if ed is None:
        return None
    today = date.today()
    delta = (ed - today).days
    return delta if delta >= 0 else None


def get_earnings_risk_positions(symbols: list[str], warning_days: int = 2) -> dict[str, date]:
    """
    Return {symbol: earnings_date} for any symbol with earnings within warning_days.
    These positions should be exited to avoid gap risk.
    """
    at_risk = {}
    today = date.today()
    for sym in symbols:
        ed = get_next_earnings_date(sym)
        if ed is None:
            continue
        days_away = (ed - today).days
        if 0 <= days_away <= warning_days:
            at_risk[sym] = ed
            logger.warning(f"{sym} earnings in {days_away} day(s) on {ed} — flagged for exit")
    return at_risk
