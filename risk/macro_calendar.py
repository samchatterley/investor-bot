import logging
from datetime import date, timedelta

from config import today_et

logger = logging.getLogger(__name__)

# High-risk scheduled macro events for 2026.
# On these days, volatility is structurally elevated and unpredictable —
# we skip new buys but still manage existing positions.
#
# Sources: Federal Reserve calendar, BLS CPI schedule, BLS NFP schedule.

FOMC_ANNOUNCEMENT_DATES = {
    date(2026, 1, 28),
    date(2026, 3, 18),
    date(2026, 5, 6),
    date(2026, 6, 17),
    date(2026, 7, 29),
    date(2026, 9, 16),
    date(2026, 10, 28),
    date(2026, 12, 9),
}

CPI_RELEASE_DATES = {
    date(2026, 1, 15),
    date(2026, 2, 12),
    date(2026, 3, 12),
    date(2026, 4, 10),
    date(2026, 5, 14),
    date(2026, 6, 11),
    date(2026, 7, 15),
    date(2026, 8, 13),
    date(2026, 9, 10),
    date(2026, 10, 9),
    date(2026, 11, 12),
    date(2026, 12, 10),
}

NFP_RELEASE_DATES = {
    date(2026, 1, 9),
    date(2026, 2, 6),
    date(2026, 3, 6),
    date(2026, 4, 3),
    date(2026, 5, 1),
    date(2026, 6, 5),
    date(2026, 7, 10),
    date(2026, 8, 7),
    date(2026, 9, 4),
    date(2026, 10, 2),
    date(2026, 11, 6),
    date(2026, 12, 4),
}

EVENT_LABELS = {
    **dict.fromkeys(FOMC_ANNOUNCEMENT_DATES, "FOMC rate decision"),
    **dict.fromkeys(CPI_RELEASE_DATES, "CPI inflation release"),
    **dict.fromkeys(NFP_RELEASE_DATES, "Non-Farm Payrolls release"),
}


NYSE_HOLIDAYS: frozenset[date] = frozenset(
    {
        # 2026
        date(2026, 1, 1),
        date(2026, 1, 19),
        date(2026, 2, 16),
        date(2026, 4, 3),
        date(2026, 5, 25),
        date(2026, 7, 3),  # observed Friday (Jul 4 = Saturday)
        date(2026, 9, 7),
        date(2026, 11, 26),
        date(2026, 12, 25),
        # 2027
        date(2027, 1, 1),
        date(2027, 1, 18),
        date(2027, 2, 15),
        date(2027, 3, 26),
        date(2027, 5, 31),
        date(2027, 7, 5),  # observed Monday (Jul 4 = Sunday)
        date(2027, 9, 6),
        date(2027, 11, 25),
        date(2027, 12, 24),  # observed Friday (Dec 25 = Saturday)
        # 2028
        date(2028, 1, 1),
    }
)


def _third_friday(year: int, month: int) -> date:
    """Return the date of the 3rd Friday in the given month/year."""
    first = date(year, month, 1)
    days_to_fri = (4 - first.weekday()) % 7  # Friday = weekday 4
    return first + timedelta(days=days_to_fri + 14)


def _next_weekday(d: date) -> date:
    """Return the next calendar day that is a weekday (Mon–Fri), skipping weekends only."""
    candidate = d + timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate


def _next_trading_day(d: date) -> date:
    """Return the next calendar day that is a weekday and not in NYSE_HOLIDAYS."""
    candidate = d + timedelta(days=1)
    while candidate.weekday() >= 5 or candidate in NYSE_HOLIDAYS:
        candidate += timedelta(days=1)
    return candidate


def get_seasonal_context(check_date: date | None = None) -> dict:
    """Return seasonal/calendar market context flags for the given date.

    Keys
    ----
    turn_of_month        : bool — within ≈2 trading days of month-end (either side)
    opex_week            : bool — Mon–Fri of the 3rd-Friday (options expiry) week
    post_opex            : bool — Mon/Tue immediately after OPEX Friday
    halloween_bullish    : bool — Nov–Apr (historically stronger market half)
    quarter_end_dressing : bool — last ≈5 trading days of Mar/Jun/Sep/Dec
    pre_holiday          : bool — next trading day is a NYSE holiday
    """
    if check_date is None:
        check_date = today_et()

    year, month = check_date.year, check_date.month

    if month == 12:
        next_month_first = date(year + 1, 1, 1)
    else:
        next_month_first = date(year, month + 1, 1)
    last_of_month = next_month_first - timedelta(days=1)
    days_from_month_end = (check_date - last_of_month).days  # negative = before EOM

    # Last day of the previous month (for detecting early-month window)
    prev_month_last = date(year, month, 1) - timedelta(days=1)
    days_after_prev_month_end = (check_date - prev_month_last).days  # 1 = first of month

    # Within ≈2 trading days of month-end OR ≈3 trading days into new month
    turn_of_month = -3 <= days_from_month_end <= 0 or 0 < days_after_prev_month_end <= 4

    opex_friday = _third_friday(year, month)
    opex_week = (opex_friday - timedelta(days=4)) <= check_date <= opex_friday
    post_opex = (opex_friday + timedelta(days=3)) <= check_date <= (opex_friday + timedelta(days=4))

    halloween_bullish = month in (11, 12, 1, 2, 3, 4)

    quarter_end_dressing = month in (3, 6, 9, 12) and days_from_month_end >= -7

    pre_holiday = _next_weekday(check_date) in NYSE_HOLIDAYS

    return {
        "turn_of_month": turn_of_month,
        "opex_week": opex_week,
        "post_opex": post_opex,
        "halloween_bullish": halloween_bullish,
        "quarter_end_dressing": quarter_end_dressing,
        "pre_holiday": pre_holiday,
    }


# High-risk days flagged for buy suspension. NFP is deliberately EXCLUDED: it releases at
# 08:30 ET, before the 10:00 ET buy window, so its gap reaction is already priced by the time
# we trade. FOMC (14:00 ET, intraday) and CPI (rate-expectation driver with all-day vol) are
# flagged. NFP dates are still tracked for the calendar-freshness boundary below.
_HIGH_RISK_DATES = FOMC_ANNOUNCEMENT_DATES | CPI_RELEASE_DATES
_LAST_MACRO_EVENT_DATE = max(FOMC_ANNOUNCEMENT_DATES | CPI_RELEASE_DATES | NFP_RELEASE_DATES)


def get_macro_risk(check_date: date | None = None) -> dict:
    """
    Returns {'is_high_risk': bool, 'event': str | None}.

    High-risk days: FOMC announcements and CPI releases. NFP is intentionally NOT flagged —
    it releases pre-market (08:30 ET), so its reaction is already in prices by the 10:00 ET
    buy window (see test_nfp_date_is_not_high_risk).

    The calendar is hardcoded through _LAST_MACRO_EVENT_DATE; past that, no days are flagged
    and a logged warning fires so the silent degradation is visible and the calendar refreshed (M2).
    """
    if check_date is None:
        check_date = today_et()

    if check_date > _LAST_MACRO_EVENT_DATE:
        logger.warning(
            "macro_calendar: %s is past the last hardcoded event date (%s) — "
            "macro high-risk days are no longer flagged. Refresh FOMC/CPI/NFP dates.",
            check_date,
            _LAST_MACRO_EVENT_DATE,
        )

    if check_date in _HIGH_RISK_DATES:
        return {"is_high_risk": True, "event": EVENT_LABELS.get(check_date, "Macro event")}

    return {"is_high_risk": False, "event": None}
